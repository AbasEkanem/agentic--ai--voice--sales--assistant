
# from  __future__ import annotations
# import os
# # langgraph imports
# from langgraph.prebuilt import tools_condition, ToolNode
# from langchain_core.messages import SystemMessage
# from langgraph.graph import StateGraph, END, MessagesState
# from langgraph.checkpoint.memory import InMemorySaver
# from agent_tools import all_tools
# from prompts_engineering import ASSISTANT_INSTRUCTIONS


# def build_gateway_llm(temperature: float = 0.3):
#     """A LangChain chat model that calls the LiveKit inference gateway.

#     The gateway is OpenAI-compatible and authenticated with a short-lived JWT that
#     carries inference grants, minted from LIVEKIT_API_KEY / LIVEKIT_API_SECRET.

#     The token is re-minted per request via a callable api_key: the gateway JWTs last
#     only ~10 minutes, so a single token baked in at import time expires mid-session
#     and every later call fails with `invalid authorization token`. Passing a callable
#     makes langchain-openai fetch a fresh one for each API call.
#     """
#     from langchain_openai import ChatOpenAI
#     from livekit.agents.inference._utils import (
#         create_access_token,
#         get_default_inference_url,
#         get_inference_headers,
#     )

#     return ChatOpenAI(
#         model=os.environ.get("LIVEKIT_INFERENCE_MODEL", "openai/gpt-4.1-mini"),
#         base_url=get_default_inference_url(),
#         # Callable -> evaluated per request, so the token never goes stale.
#         api_key=lambda: create_access_token(None, None),
#         default_headers=get_inference_headers(),
#         temperature=temperature,
#     )


# def build_groq_llm(temperature: float = 0.3):
#     """A LangChain chat model pointed at Groq's OpenAI-compatible endpoint.

#     Used as the fallback provider when the LiveKit inference gateway is unavailable
#     or out of credit.
#     """
#     from langchain_openai import ChatOpenAI

#     return ChatOpenAI(
#         model=os.environ.get("GROQ_LLM_MODEL", "openai/gpt-oss-20b"),
#         base_url="https://api.groq.com/openai/v1",
#         api_key=os.environ.get("GROQ_API_KEY"),
#         temperature=temperature,
#     )


# def build_agent_llm(temperature: float = 0.3):
#     """The graph's model: LiveKit inference gateway first, Groq as fallback.

#     Both are tool-capable, so bind_tools works on the chained runnable. Model calls
#     normally bill the LiveKit inference credit pool and only spill to Groq when the
#     gateway fails — including credit/auth errors (403), which are listed explicitly
#     because with_fallbacks does not retry those by default.
#     """
#     try:
#         import openai

#         handled: tuple[type[BaseException], ...] = (
#             openai.APIError,          # covers APIStatusError/APIConnectionError/429/5xx
#             openai.APITimeoutError,
#             openai.AuthenticationError,
#             openai.PermissionDeniedError,  # 403 — e.g. inference credits exhausted
#         )
#     except Exception:  # pragma: no cover - openai is a hard dep in practice
#         handled = (Exception,)

#     return build_gateway_llm(temperature).with_fallbacks(
#         [build_groq_llm(temperature)],
#         exceptions_to_handle=handled,
#     )


# memory = InMemorySaver()


# def _drop_orphaned_tool_calls(messages: list) -> list:
#     """Remove tool calls that have no matching ToolMessage.

#     A model API rejects any history where an AIMessage with tool_calls is not
#     followed by a ToolMessage for each tool_call_id:

#         400 - "An assistant message with 'tool_calls' must be followed by tool
#                messages responding to each 'tool_call_id'."

#     Those orphans are easy to create here: the voice agent can be interrupted after
#     the model asks for a tool but before ToolNode answers, and the AIMessage is
#     already committed to the checkpointer. Left alone, one interruption poisons the
#     whole thread — every later turn replays the bad history and fails. So we repair
#     the history on the way into the model: keep only tool calls that were answered,
#     and if nothing of an assistant message survives, drop it.
#     """
#     from langchain_core.messages import AIMessage, ToolMessage

#     # Every tool_call_id that some ToolMessage actually answers to.
#     answered: set[str] = {
#         m.tool_call_id for m in messages if isinstance(m, ToolMessage) and m.tool_call_id
#     }

#     cleaned: list = []
#     for m in messages:
#         if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
#             kept = [tc for tc in m.tool_calls if tc.get("id") in answered]
#             if len(kept) == len(m.tool_calls):
#                 cleaned.append(m)
#                 continue
#             if kept or m.content:
#                 cleaned.append(m.model_copy(update={"tool_calls": kept}))
#             # else: nothing left to say — omit the message entirely
#             continue
#         cleaned.append(m)
#     return cleaned


# # create the langgraph agent
# class GraphState(MessagesState):
#     pass

# # create the voice ai agent graph
# class VoiceAIAgent:
#     def __init__(self, chat_model = None, tools = None, instructions:str = ASSISTANT_INSTRUCTIONS):
#         if chat_model is None:
#             # Default: LiveKit inference gateway first, Groq as fallback, so graph
#             # model calls draw on the inference credit pool and still work if the
#             # gateway is unavailable. Pass chat_model= to override entirely.
#             chat_model = build_agent_llm()
#         tools = tools or all_tools
#         self.tools = {t.name: t for t in tools}
#         self.chat_model = chat_model.bind_tools(tools)
#         self.instructions = instructions
#         # build the agent node
#         def Sales_Assistant(state: GraphState):
#             system_prompt = SystemMessage(content=self.instructions)
#             # Repair any orphaned tool calls before the model sees the history, so a
#             # single interruption can't poison every later turn on this thread.
#             history = _drop_orphaned_tool_calls(list(state["messages"]))
#             ai_repsonse = self.chat_model.invoke([system_prompt] + history)
#             return{
#                 "messages":[ai_repsonse]
#             }
#         # create the graph using the stategraph
#         graph_builder = StateGraph(GraphState)
#         # create the node and the edges
#         graph_builder.add_node("Sales_Assistant", Sales_Assistant)
#         graph_builder.add_node("tools", ToolNode(tools))
#         graph_builder.set_entry_point("Sales_Assistant")
#         graph_builder.add_conditional_edges(
#             "Sales_Assistant",
#             tools_condition,
#             {"tools": "tools", END: END},
#         )
#         graph_builder.add_edge("tools", "Sales_Assistant")
#         self.graph_builder = graph_builder.compile(checkpointer=memory)


# from livekit.agents import (
#     Agent,
#     AgentServer,
#     JobContext,
#     JobProcess,
#     AgentSession,
#     cli,
#     inference,
#     room_io,
#     llm,
#     stt,
#     tts
# )
# import logging
# import time
# from livekit.plugins.turn_detector.multilingual import MultilingualModel
# from livekit.plugins import noise_cancellation, langchain, silero, groq, deepgram
# from graph import VoiceAIAgent
# from prompts_engineering import ASSISTANT_INSTRUCTIONS
# from livekit.agents import AgentStateChangedEvent, MetricsCollectedEvent, metrics

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # instantiate the voice sales agent class 
# _graph_agent = VoiceAIAgent().graph_builder

# # create the agentic ai sales assistant

# class AgenticSalesVoice_AI_Asistant(Agent):
#     def __init__(self, thread_id: str = "default") -> None:
#         super().__init__(instructions=ASSISTANT_INSTRUCTIONS)
#         # The graph is compiled with a checkpointer, so every call MUST carry a
#         # thread_id — otherwise LangGraph raises on the first turn. One thread per
#         # room keeps each conversation's history separate in the shared saver.
#         self._thread_id = thread_id

#     async def llm_node(self, chat_ctx, tools, model_settings=None):
#         # Delegate to LangGraphStream, which converts the ChatContext to graph state,
#         # runs the graph with our thread_id, and emits ChatChunks for each token.
#         # (`tools` is unused because tool execution happens inside the graph.)
#         stream = langchain.LLMAdapter(
#             graph=_graph_agent,
#             config={"configurable": {"thread_id": self._thread_id}},
#         ).chat(chat_ctx=chat_ctx)

#         async for chunk in stream:
#             yield chunk
                
# #set up the server
# server = AgentServer()

# def prewarm(proc: JobProcess):
#     proc.userdata["vad"] = silero.VAD.load()

# server.setup_fnc = prewarm
# # create the RTC session entrypoint
# @server.rtc_session()
# async def entrypoint(ctx:JobContext):
#     ctx.log_context_fields = {"room":ctx.room.name}

#     # One conversation thread per room, so each call keeps its own history in the
#     # shared checkpointer. Fall back to the job id if the room name is empty.
#     thread_id = ctx.room.name or ctx.job.id

#     # ---- observability: usage aggregation, TTFA timing, per-session summary ----
#     # Aggregate token/audio usage across all turns in this session.
#     usage_collector = metrics.UsageCollector()
#     # Remember the last end-of-utterance event so we can compute time-to-first-audio.
#     last_eou_metrics: metrics.EOUMetrics | None = None

#     # One VAD instance, shared. Groq's Whisper is a batch (non-streaming) STT, and
#     # stt.FallbackAdapter will wrap it with a StreamAdapter automatically — but only
#     # if it is given a VAD. Use the instance prewarmed in proc.userdata when present
#     # so we don't load a second Silero model.
#     vad = ctx.proc.userdata.get("vad") or silero.VAD.load()

#     # create the session pipeline
#     # this is where the TTS, STT, LLM, and VAD is created and note: the langgraph agent is added as an LLM instance using the LLMAdapter
#     pipeline_session = AgentSession(
#         stt=stt.FallbackAdapter(
#             stt=[
#                 groq.STT(model="whisper-large-v3-turbo"),
#                 inference.STT.from_model_string("deepgram/nova-3"),
#             ],
#             vad=vad,
#         ),
#         llm=llm.FallbackAdapter(llm=[
#             langchain.LLMAdapter(
#                 graph=_graph_agent,
#                 config={"configurable": {"thread_id": thread_id}},
#             ),
#             # LiveKit Cloud inference fallback. NOTE: the model the graph itself calls
#             # is chosen in graph.build_gateway_llm(); this entry is only used if you
#             # drop the per-room graph adapter above.
#             inference.LLM(model="openai/gpt-4.1-mini"),
#         ]),
#         tts=tts.FallbackAdapter(tts=[
#             deepgram.TTS(model="aura-2-helena-en"),
#             inference.TTS.from_model_string("cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
#         ]),
#         # set up the turn detector
#         turn_detection=MultilingualModel(),
#         preemptive_generation=True,
#         vad=vad,
#     )

#     @pipeline_session.on("metrics_collected")
#     def _on_metrics_collected(ev: MetricsCollectedEvent):
#         nonlocal last_eou_metrics
#         # End-of-utterance is the moment the user finished speaking; TTFA is measured
#         # from here to when the agent starts speaking.
#         if ev.metrics.type == "eou_metrics":
#             last_eou_metrics = ev.metrics
#         metrics.log_metrics(ev.metrics)
#         usage_collector.collect(ev.metrics)

#     @pipeline_session.on("agent_state_changed")
#     def _on_agent_state_changed(ev: AgentStateChangedEvent):
#         if ev.new_state == "speaking" and last_eou_metrics is not None:
#             elapsed = time.time() - last_eou_metrics.timestamp
#             logger.info("Time to first audio: %.3fs", elapsed)

#     async def log_usage():
#         # Per-session summary: tokens, audio duration, estimated cost.
#         logger.info("Usage summary: %s", usage_collector.get_summary())

#     # Print the usage summary when this session's worker shuts down.
#     ctx.add_shutdown_callback(log_usage)

#     await pipeline_session.start(
#         agent= AgenticSalesVoice_AI_Asistant(thread_id=thread_id),
#         room=ctx.room,
#         room_options=room_io.RoomOptions(
#             audio_input=room_io.AudioInputOptions(
#                  # introduce the noise-cancellation component for background noise cancellation and increased focus for the ai agent
#                 noise_cancellation=noise_cancellation.BVC(),
#             ),
#         ),
#     )



# if __name__ == "__main__":
#     cli.run_app(server)
