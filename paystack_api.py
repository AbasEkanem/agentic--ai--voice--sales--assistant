"""Paystack tools for the voice sales agent.

Two LangChain tools (start_payment, check_payment) built on a small async client.

Safety rules baked in:
- The amount NEVER comes from the model or the caller. It is price(product_id) * quantity.
- The payment link is delivered out-of-band (email) and is never returned to the model.
- A payment counts as paid ONLY when Paystack's verify endpoint says so and the
  amount and currency match what we stored when the order was created.
- No card details are ever accepted by these tools.

Env: PAYSTACK_SECRET_KEY (use the sk_test_ key until the flow is proven).
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol
from urllib.parse import quote

import httpx
from langchain_core.tools import tool
from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    String,
    Table,
    insert,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

log = logging.getLogger("paystack_tools")

BASE_URL = "https://api.paystack.co"
CURRENCY = "NGN"
ALLOWED_CHANNELS = {"card", "bank_transfer", "ussd", "mobile_money"}
MAX_QUANTITY = 20

# Tighter than the original: local part, then a dotted domain with a 2+ letter TLD.
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


class PaystackError(Exception):
    """Raised for network problems and for any non-success Paystack response."""


class PaystackClient:
    """Minimal async client for Initialize Transaction and Verify Transaction."""

    def __init__(
        self,
        secret_key: str | None = None,
        *,
        timeout: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        key = secret_key or os.environ.get("PAYSTACK_SECRET_KEY")
        if not key:
            raise PaystackError("PAYSTACK_SECRET_KEY is not set")
        if key.startswith("sk_live_") and os.environ.get("PAYSTACK_ALLOW_LIVE") != "1":
            raise PaystackError("Live key refused. Set PAYSTACK_ALLOW_LIVE=1 to allow it.")
        self._http = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=timeout,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "PaystackClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            resp = await self._http.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise PaystackError(f"network error: {type(e).__name__}") from e
        try:
            payload = resp.json()
        except ValueError as e:
            raise PaystackError(f"non-JSON response (HTTP {resp.status_code})") from e
        # Top-level "status" only says whether the API call worked, not whether money moved.
        if resp.status_code >= 400 or not payload.get("status"):
            raise PaystackError(payload.get("message") or f"HTTP {resp.status_code}")
        return payload["data"]

    async def initialize(
        self,
        *,
        email: str,
        amount_kobo: int,
        reference: str,
        channels: list[str] | None = None,
        currency: str = CURRENCY,
        callback_url: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Returns {"authorization_url", "access_code", "reference"}."""
        body: dict = {
            "email": email,
            "amount": str(amount_kobo),  # subunit (kobo), sent as a string like the docs
            "reference": reference,
            "currency": currency,
        }
        if channels:
            body["channels"] = channels
        if callback_url:
            body["callback_url"] = callback_url
        if metadata:
            # Prefer plain object – httpx will serialize it correctly.
            # (Stringified JSON also works, but object is cleaner and matches modern examples.)
            body["metadata"] = metadata
        return await self._request("POST", "/transaction/initialize", json=body)

    async def verify(self, reference: str) -> dict:
        """Returns the transaction object; data["status"] == "success" means paid."""
        # Keep the characters Paystack allows unescaped.
        safe_ref = quote(reference, safe="-.=")
        return await self._request("GET", f"/transaction/verify/{safe_ref}")


# ---- seams you will replace later (Postgres, email) -----------------------------------


@dataclass
class PendingOrder:
    reference: str
    product_id: str
    quantity: int
    amount_kobo: int
    currency: str
    email: str
    channel: str
    status: str = "pending"  # pending | paid | failed | amount_mismatch | init_failed | link_failed


class OrderStore(Protocol):
    async def save(self, order: PendingOrder) -> None: ...
    async def get(self, reference: str) -> PendingOrder | None: ...
    async def set_status(self, reference: str, status: str) -> None: ...
    async def aclose(self) -> None: ...


class InMemoryOrderStore:
    """Dev stand-in. Replace with the Postgres transactions table."""

    def __init__(self) -> None:
        self._orders: dict[str, PendingOrder] = {}

    async def save(self, order: PendingOrder) -> None:
        self._orders[order.reference] = order

    async def get(self, reference: str) -> PendingOrder | None:
        return self._orders.get(reference)

    async def set_status(self, reference: str, status: str) -> None:
        if reference in self._orders:
            self._orders[reference].status = status


_ORDER_COLUMNS = (
    "reference",
    "product_id",
    "quantity",
    "amount_kobo",
    "currency",
    "email",
    "channel",
    "status",
)


class SQLAlchemyOrderStore:
    """Durable OrderStore backed by SQLAlchemy Core (async).

    Runs on SQLite out of the box and on Postgres in production by setting
    DATABASE_URL, with no code change. A bare "postgresql://" URL is upgraded to
    the async psycopg driver automatically. Create it with create_order_store().
    """

    def __init__(self, database_url: str) -> None:
        url = _normalize_db_url(database_url)
        self._engine = create_async_engine(url, future=True)
        self._meta = MetaData()
        self._table = Table(
            "orders",
            self._meta,
            Column("reference", String(64), primary_key=True),
            Column("product_id", String(64), nullable=False),
            Column("quantity", Integer, nullable=False),
            Column("amount_kobo", Integer, nullable=False),
            Column("currency", String(8), nullable=False),
            Column("email", String(320), nullable=False),
            Column("channel", String(32), nullable=False),
            Column("status", String(32), nullable=False),
        )
        self._ready = False

    async def _ensure(self) -> None:
        if not self._ready:
            async with self._engine.begin() as conn:
                await conn.run_sync(self._meta.create_all)
            self._ready = True

    async def save(self, order: PendingOrder) -> None:
        await self._ensure()
        values = {
            "reference": order.reference,
            "product_id": order.product_id,
            "quantity": order.quantity,
            "amount_kobo": order.amount_kobo,
            "currency": order.currency,
            "email": order.email,
            "channel": order.channel,
            "status": order.status,
        }
        try:
            async with self._engine.begin() as conn:
                await conn.execute(insert(self._table).values(**values))
        except IntegrityError:
            # Reference collision (astronomically rare with uuid4) — ignore.
            log.warning("order %s already exists; skipping save", order.reference)

    async def get(self, reference: str) -> PendingOrder | None:
        await self._ensure()
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    select(self._table).where(self._table.c.reference == reference)
                )
            ).mappings().first()
        return PendingOrder(**{c: row[c] for c in _ORDER_COLUMNS}) if row else None

    async def set_status(self, reference: str, status: str) -> None:
        await self._ensure()
        async with self._engine.begin() as conn:
            await conn.execute(
                update(self._table)
                .where(self._table.c.reference == reference)
                .values(status=status)
            )

    async def aclose(self) -> None:
        await self._engine.dispose()


def _normalize_db_url(url: str) -> str:
    """Turn a plain URL into an async-driver URL SQLAlchemy can use asynchronously."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("sqlite://") and not url.startswith("sqlite+aiosqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    return url


def create_order_store(database_url: str | None = None) -> OrderStore:
    """Build the durable order store from DATABASE_URL (default: local SQLite file)."""
    url = database_url or os.environ.get("DATABASE_URL") or "sqlite+aiosqlite:///orders.db"
    return SQLAlchemyOrderStore(url)


PriceLookup = Callable[[str], Awaitable["int | None"]]  # product_id -> unit price in kobo
LinkSender = Callable[[str, str, str, int], Awaitable[None]]  # email, reference, url, amount_kobo


def in_memory_prices(table: dict[str, int]) -> PriceLookup:
    """Dev stand-in for the real product price lookup."""

    async def lookup(product_id: str) -> int | None:
        return table.get(product_id)

    return lookup


async def console_link_sender(email: str, reference: str, url: str, amount_kobo: int) -> None:
    """Dev stand-in for the email tool: prints the link instead of emailing it."""
    print(f"[DEV] would email {email} the link for {reference}: {url}")


def new_reference() -> str:
    # Paystack allows only -, ., = and alphanumerics in a reference.
    return f"vsa-{uuid.uuid4().hex}"


def naira_text(kobo: int) -> str:
    return f"{kobo // 100:,} naira" if kobo % 100 == 0 else f"{kobo / 100:,.2f} naira"


def _classify(paystack_status: str) -> str:
    if paystack_status == "success":
        return "paid"
    if paystack_status in {"failed", "reversed"}:
        return "failed"
    return "pending"  # abandoned, ongoing, pending, processing, queued ...


# ---- the tools ------------------------------------------------------------------------


def build_paystack_tools(
    client: PaystackClient,
    prices: PriceLookup,
    orders: OrderStore,
    send_link: LinkSender = console_link_sender,
):
    @tool
    async def start_payment(
        product_id: str,
        quantity: int,
        customer_email: str,
        channel: str,
        customer_confirmed: bool,
    ) -> dict:
        """Create a payment for a product and email the secure payment link to the customer.

        Call this ONLY after the customer has said yes to a full summary: the product,
        the quantity, their email address read back letter by letter, and the payment
        channel. Never ask for or accept card numbers by voice. channel must be one of:
        card, bank_transfer, ussd, mobile_money. Set customer_confirmed to true only if
        the customer explicitly confirmed. The price is looked up from the product
        catalogue, so never pass an amount. The link is emailed and is not returned.
        """
        if not customer_confirmed:
            return {"ok": False, "error": "not_confirmed"}
        email = customer_email.strip().lower()
        if not _EMAIL_RE.match(email):
            return {"ok": False, "error": "invalid_email"}
        if channel not in ALLOWED_CHANNELS:
            return {"ok": False, "error": "invalid_channel", "allowed": sorted(ALLOWED_CHANNELS)}
        if not 1 <= quantity <= MAX_QUANTITY:
            return {"ok": False, "error": "invalid_quantity", "max": MAX_QUANTITY}

        unit = await prices(product_id)
        if unit is None:
            return {"ok": False, "error": "unknown_product"}

        amount = unit * quantity
        order = PendingOrder(
            new_reference(), product_id, quantity, amount, CURRENCY, email, channel
        )
        await orders.save(order)

        try:
            init = await client.initialize(
                email=email,
                amount_kobo=amount,
                reference=order.reference,
                channels=[channel],
                metadata={"product_id": product_id, "quantity": quantity},
            )
        except PaystackError as e:
            log.warning("initialize failed for %s: %s", order.reference, e)
            await orders.set_status(order.reference, "init_failed")
            return {"ok": False, "error": "payment_service_unavailable"}

        try:
            await send_link(email, order.reference, init["authorization_url"], amount)
        except Exception:
            log.exception("could not deliver link for %s", order.reference)
            await orders.set_status(order.reference, "link_failed")
            return {"ok": False, "error": "could_not_send_link"}

        return {
            "ok": True,
            "reference": order.reference,
            "amount_text": naira_text(amount),
            "next": "payment link emailed; wait for the customer to pay, then call check_payment",
        }

    @tool
    async def check_payment(reference: str) -> dict:
        """Check whether a payment has really been completed. Always call this before
        telling the customer their payment worked, even if they say they already paid.

        Returns status: paid, pending, failed, amount_mismatch, unknown_reference or
        could_not_verify.
        """
        order = await orders.get(reference)
        if order is None:
            return {"ok": False, "status": "unknown_reference"}

        if order.status == "paid":
            return {"ok": True, "status": "paid", "amount_text": naira_text(order.amount_kobo)}

        try:
            data = await client.verify(reference)
        except PaystackError as e:
            log.warning("verify failed for %s: %s", reference, e)
            return {"ok": False, "status": "could_not_verify"}

        status = _classify(str(data.get("status", "")))

        if status == "paid":
            # Prefer the actual charged amount. When "Pass fees to customer" is on,
            # this can be higher than the amount we initialized. Never accept lower.
            paid_amount = int(data.get("amount") or data.get("requested_amount") or 0)
            if paid_amount < order.amount_kobo or data.get("currency") != order.currency:
                await orders.set_status(reference, "amount_mismatch")
                return {"ok": False, "status": "amount_mismatch"}

            await orders.set_status(reference, "paid")
            return {"ok": True, "status": "paid", "amount_text": naira_text(order.amount_kobo)}

        if status == "failed":
            await orders.set_status(reference, "failed")

        return {"ok": True, "status": status}

    return [start_payment, check_payment]