from livekit.agents import (
    Agent,
    AgentServer,
    JobContext,
    JobProcess,
    AgentSession,
    cli,
    inference,
    room_io,
    llm,
    stt,
    tts
)
import logging
import time
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import noise_cancellation, langchain, silero, groq, deepgram
from graph import VoiceAIAgent
from langchain_core.messages import AIMessageChunk
from prompts_engineering import ASSISTANT_INSTRUCTIONS
from livekit.agents import AgentStateChangedEvent, MetricsCollectedEvent, metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# instantiate the voice sales agent class 
_graph_agent = VoiceAIAgent().graph_builder

# create the agentic ai sales assistant

class AgenticSalesVoice_AI_Asistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=ASSISTANT_INSTRUCTIONS)

    async def llm_node(self, chat_ctx, tools, model_settings=None):
        # chat_ctx is the single source of conversation history. Convert it to LangGraph
        # input with the plugin's own converter (text turns only — no stray tool messages),
        # then stream the graph token-by-token straight to TTS. (`tools` is unused because
        # tool execution happens inside the graph.)
        stream = langchain.LLMAdapter(graph=_graph_agent).chat(chat_ctx=chat_ctx, tools=[])
        lc_input = stream._chat_ctx_to_state()

        spoke_filler = False
        async for chunk, _meta in _graph_agent.astream(lc_input, stream_mode="messages"):
            if not isinstance(chunk, AIMessageChunk):
                continue
            text = str(chunk.text)  # flatten str or structured-block content to a string
            tool_sig = getattr(chunk, "tool_calls", None) or getattr(chunk, "tool_call_chunks", None)
            # First hop decides to call a tool: it carries a tool call but no speakable
            # text. Speak a short filler so first audio lands sub-second while the
            # knowledge-base / web lookup runs behind it (masks the tool round-trip).
            if tool_sig and not text and not spoke_filler:
                spoke_filler = True
                yield llm.ChatChunk(
                    id="filler",
                    delta=llm.ChoiceDelta(role="assistant", content="Let me check that for you. "),
                )
                continue
            if text:
                yield llm.ChatChunk(
                    id=chunk.id or "",
                    delta=llm.ChoiceDelta(role="assistant", content=text),
                )
                
#set up the server
server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm
# create the RTC session entrypoint
@server.rtc_session()
async def entrypoint(ctx:JobContext):
    ctx.log_context_fields = {"room":ctx.room.name}

    # ---- observability: usage aggregation, TTFA timing, per-session summary ----
    # Aggregate token/audio usage across all turns in this session.
    usage_collector = metrics.UsageCollector()
    # Remember the last end-of-utterance event so we can compute time-to-first-audio.
    last_eou_metrics: metrics.EOUMetrics | None = None
    # Reuse the models loaded once in prewarm() — never build them on the hot path.
    vad = ctx.proc.userdata["vad"]

    # create the session pipeline
    # this is where the TTS, STT, LLM, and VAD is created and note: the langgraph agent is added as an LLM instance using the LLMAdapter
    pipeline_session = AgentSession(
        stt=stt.FallbackAdapter(
            stt=[
                inference.STT.from_model_string("assemblyai/universal-streaming:en"),
                inference.STT.from_model_string("deepgram/nova-3-general"),
            ],
            vad=vad,
        ),
        llm=langchain.LLMAdapter(
                graph=_graph_agent
            ),
        tts=tts.FallbackAdapter(tts=[
            deepgram.TTS(model="aura-2-helena-en"),
            inference.TTS.from_model_string("cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        ]),
        # set up the turn detector (loaded once in prewarm, reused across sessions)
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
        vad=vad,
    )

    @pipeline_session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics
        # End-of-utterance is the moment the user finished speaking; TTFA is measured
        # from here to when the agent starts speaking.
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    @pipeline_session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        if ev.new_state == "speaking" and last_eou_metrics is not None:
            elapsed = time.time() - last_eou_metrics.timestamp
            logger.info("Time to first audio: %.3fs", elapsed)

    async def log_usage():
        # Per-session summary: tokens, audio duration, estimated cost.
        logger.info("Usage summary: %s", usage_collector.get_summary())

    # Print the usage summary when this session's worker shuts down.
    ctx.add_shutdown_callback(log_usage)

    await pipeline_session.start(
        agent= AgenticSalesVoice_AI_Asistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                 # introduce the noise-cancellation component for background noise cancellation and increased focus for the ai agent
                noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )



if __name__ == "__main__":
    cli.run_app(server)
