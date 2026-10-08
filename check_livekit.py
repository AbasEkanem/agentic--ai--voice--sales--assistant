from importlib.metadata import version

for pkg in ["livekit-agents", "livekit-plugins-groq", "livekit-plugins-deepgram",
            "livekit-plugins-silero", "livekit-plugins-langchain", "openai"]:
    try:
        print(f"{pkg:28} {version(pkg)}")
    except Exception as e:
        print(f"{pkg:28} MISSING ({type(e).__name__})")

checks = [
    ("livekit.agents", "AgentSession"),
    ("livekit.agents", "Agent"),
    ("livekit.plugins.groq", "STT"),
    ("livekit.plugins.deepgram", "TTS"),
    ("livekit.plugins.silero", "VAD"),
    ("livekit.plugins.langchain", "LLMAdapter"),
]
import importlib
for module, name in checks:
    try:
        mod = importlib.import_module(module)
        print("OK  ", module, name if hasattr(mod, name) else "-> MODULE OK, NAME MISSING")
    except Exception as e:
        print("FAIL", module, "->", type(e).__name__, e)