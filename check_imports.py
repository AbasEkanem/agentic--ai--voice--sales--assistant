import importlib

checks = [
    ("pipecat.services.groq.stt", "GroqSTTService"),
    ("pipecat.services.deepgram.tts", "DeepgramTTSService"),
    ("pipecat.audio.vad.silero", "SileroVADAnalyzer"),
    ("pipecat.transports.smallwebrtc.transport", "SmallWebRTCTransport"),
    ("pipecat.pipeline.worker", "PipelineWorker"),
    ("pipecat.workers.runner", "WorkerRunner"),
    ("pipecat.processors.aggregators.llm_response_universal", "LLMContextAggregatorPair"),
]

for module, name in checks:
    try:
        mod = importlib.import_module(module)

        if hasattr(mod, name):
            print("OK  ", module, "->", name)
        else:
            print("FAIL", module, "-> NAME MISSING:", name)

    except Exception as e:
        print("FAIL", module, "->", type(e).__name__, e)