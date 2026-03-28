import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import asyncio

# Add 45M to path to import llm_bridge
m_path = Path(__file__).resolve().parent / "45M"
if str(m_path) not in sys.path:
    sys.path.insert(0, str(m_path))

try:
    from llm_bridge import get_llm_bridge
except ImportError as e:
    print(f"Failed to import llm_bridge: {e}")
    sys.exit(1)

def test_sync():
    print("Testing synchronous generation...")
    llm = get_llm_bridge()
    output = llm.generate("Hello, world!", max_new_tokens=20)
    print(f"Output: {output}")

async def test_async():
    print("Testing asynchronous generation...")
    llm = get_llm_bridge()
    output = await llm.agenerate("What is the capital of France?", max_new_tokens=20)
    print(f"Output: {output}")

if __name__ == "__main__":
    print("--- LLM Bridge Test ---")
    test_sync()
    asyncio.run(test_async())
    print("--- Done ---")
