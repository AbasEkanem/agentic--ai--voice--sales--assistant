
from  __future__ import annotations
import os
# langgraph imports
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END, MessagesState
from agent_tools import all_tools
from prompts_engineering import ASSISTANT_INSTRUCTIONS
from dotenv import load_dotenv
load_dotenv()

class GraphState(MessagesState):
    pass


def build_gateway_llm(temperature: float = 0.3):
    """A LangChain chat model that calls the LiveKit inference gateway.

    The gateway is OpenAI-compatible and authenticated with a short-lived JWT minted
    from LIVEKIT_API_KEY / LIVEKIT_API_SECRET. This avoids regional/ISP 403 blocks.
    """
    from langchain_openai import ChatOpenAI
    from livekit.agents.inference._utils import (
        create_access_token,
        get_default_inference_url,
        get_inference_headers,
    )

    return ChatOpenAI(
        model=os.environ.get("LIVEKIT_INFERENCE_MODEL", "openai/gpt-4.1-mini"),
        base_url=get_default_inference_url(),
        api_key=lambda: create_access_token(None, None),
        default_headers=get_inference_headers(),
        temperature=temperature,
    )


def build_groq_llm(temperature: float = 0.3):
    """A LangChain chat model pointed at Groq's endpoint."""
    key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
    if "openai" in model_name:
        model_name = "llama-3.3-70b-versatile"

    try:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name,
            groq_api_key=key,
            temperature=temperature,
        )
    except ImportError:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=key,
            base_url="https://api.groq.com/openai/v1",
            temperature=temperature,
        )


def build_agent_llm(temperature: float = 0.3):
    """The graph's model: LiveKit inference gateway first, Groq as fallback.

    Handles 403 PermissionDeniedError (e.g. Groq network blocks) and API errors cleanly.
    """
    try:
        import openai

        handled: tuple[type[BaseException], ...] = (
            openai.APIError,
            openai.APITimeoutError,
            openai.AuthenticationError,
            openai.PermissionDeniedError,
        )
    except Exception:
        handled = (Exception,)

    return build_gateway_llm(temperature).with_fallbacks(
        [build_groq_llm(temperature)],
        exceptions_to_handle=handled,
    )


def get_groq_chat_model(
    model: str | None = None,
    temperature: float = 0.3,
    api_key: str | None = None,
):
    """Safely initialize a Groq chat model with validation and fallbacks."""
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please provide it or set it in your .env file."
        )

    model_name = model or os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-20b")

    try:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name,
            groq_api_key=key,
            temperature=temperature,
        )
    except ImportError:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=key,
            base_url="https://api.groq.com/openai/v1",
            temperature=temperature,
        )


def _drop_orphaned_tool_calls(messages: list) -> list:
    """Remove tool calls that have no matching ToolMessage to prevent 400 errors."""
    from langchain_core.messages import AIMessage, ToolMessage

    answered: set[str] = {
        m.tool_call_id for m in messages if isinstance(m, ToolMessage) and m.tool_call_id
    }

    cleaned: list = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            kept = [tc for tc in m.tool_calls if tc.get("id") in answered]
            if len(kept) == len(m.tool_calls):
                cleaned.append(m)
                continue
            if kept or m.content:
                cleaned.append(m.model_copy(update={"tool_calls": kept}))
            continue
        cleaned.append(m)
    return cleaned


class VoiceAIAgent:
    def __init__(
        self,
        chat_model=None,
        tools=None,
        instructions: str = ASSISTANT_INSTRUCTIONS,
        temperature: float = 0.3,
    ):
        if chat_model is None:
            chat_model = build_agent_llm(temperature=temperature)
        tools = tools or all_tools
        self.tools = {t.name: t for t in tools}
        self.chat_model = chat_model.bind_tools(tools)
        self.instructions = instructions
        # build the agent node
        def Sales_Assistant(state: GraphState):
            system_prompt = SystemMessage(content=self.instructions)
            history = _drop_orphaned_tool_calls(list(state["messages"]))
            ai_repsonse = self.chat_model.invoke([system_prompt] + history)
            return{
                "messages":[ai_repsonse]
            }
        # create the graph using the stategraph
        graph_builder = StateGraph(GraphState)
        # create the node and the edges
        graph_builder.add_node("Sales_Assistant", Sales_Assistant)
        graph_builder.add_node("tools", ToolNode(tools))
        graph_builder.set_entry_point("Sales_Assistant")
        graph_builder.add_conditional_edges(
            "Sales_Assistant",
            tools_condition,
            {"tools": "tools", END: END},
        )
        graph_builder.add_edge("tools", "Sales_Assistant")
        # No checkpointer: LiveKit passes the full chat context every turn, so that is
        # the single source of conversation history. Keeping an InMemorySaver here as
        # well made the graph maintain history twice, growing the prompt each turn (and
        # forcing a thread_id on every call). chat_ctx-only keeps TTFT flat over a call.
        self.graph_builder = graph_builder.compile()