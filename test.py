# test_interpreter_async.py
import asyncio
from rasa.core.agent import Agent

CORE_MODEL_PATH = "models/20251106-152304-tempered-louver.tar.gz"
NLU_MODEL_PATH = "models/nlu-20251106-143229-cerulean-bayou.tar.gz"

async def test_agent():
    # 1️⃣ Full agent
    try:
        agent = Agent.load(CORE_MODEL_PATH)
        print("✅ Full agent loaded from CORE_MODEL_PATH")
        result = await agent.parse_message("hello")
        print("Agent parse_message output:", result)
    except Exception as e:
        print("❌ Full agent parse_message failed:", e)

    # 2️⃣ NLU-only agent
    try:
        nlu_agent = Agent.load(NLU_MODEL_PATH)
        print("✅ NLU-only agent loaded")
        result = await nlu_agent.parse_message("I feel sad today")
        print("NLU-only agent parse_message output:", result)
    except Exception as e:
        print("❌ NLU-only agent parse_message failed:", e)

    # 3️⃣ agent.handle_text fallback
    try:
        responses = await agent.handle_text("hello")
        print("Agent handle_text output:", responses)
    except Exception as e:
        print("❌ Agent handle_text failed:", e)

if __name__ == "__main__":
    asyncio.run(test_agent())
