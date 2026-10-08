from __future__ import annotations

import os
from dotenv import load_dotenv
from langchain.tools import tool
from paystack_api import (
    PaystackClient,
    create_order_store,
    in_memory_prices,
    build_paystack_tools,
)
from tavily_search import web_search
from RAG_pipeline import get_vector_store

load_dotenv()

# Read secret key from .env (checking both PAYSTACK_SECRET_KEY and PAYSTACK_TEST_SECRET_KEY)
secret_key = os.environ.get("PAYSTACK_SECRET_KEY") or os.environ.get("PAYSTACK_TEST_SECRET_KEY")

# Initialize Paystack client, order store, and catalog pricing.
# Orders are persisted via DATABASE_URL (SQLite by default; set it to Postgres in prod).
client = PaystackClient(secret_key=secret_key)
orders = create_order_store()
prices = in_memory_prices({
    "consultation": 500000,    
    "service_package": 2500000  
})

# Build the payment tools
start_payment, check_payment = build_paystack_tools(
    client=client,
    prices=prices,
    orders=orders,
)

@tool
def retriever_tool(query: str) -> str:
    """Look up the company knowledge base for answers about our own products,
    services, packages, features, policies and pricing.

    Use this for anything about our own business; use web search instead for
    current, external information. Returns the most relevant passages, or a short
    message when the knowledge base is unavailable or has no match.
    """
    try:
        store = get_vector_store()
    except Exception as exc:  # noqa: BLE001 - never crash a live call over the KB
        import logging

        logging.getLogger("agent_tools").error("knowledge base unavailable: %s", exc)
        return (
            "The knowledge base is not available right now. Please continue without it "
            "and offer to follow up."
        )
    try:
        retrieved_response = store.similarity_search(query, k=4)
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger("agent_tools").error("knowledge base lookup failed: %s", exc)
        return "I couldn't search the knowledge base just now. Please try again."
    if not retrieved_response:
        return f"No knowledge-base entries matched the query: {query}"
    return "\n\n".join(
        f"Source: {doc.metadata}\n{doc.page_content}" for doc in retrieved_response
    )

# Web search tool powered by Tavily — used for real-time information outside the
# knowledge base. Built in tavily_search.py: async, with timeouts, retries and
# speakable fallbacks so a failed search can never crash a live call.
# Requires TAVILY_API_KEY in .env.

# Export all tools for graph.py
all_tools = [retriever_tool, web_search, start_payment, check_payment]

