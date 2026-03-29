"""
Test the centralized LLM Bridge — verifies Phase 1 model loads and generates.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import asyncio

# Add 45M to path to import llm_bridge
m_path = Path(__file__).resolve().parent / "45M"
if str(m_path) not in sys.path:
    sys.path.insert(0, str(m_path))

try:
    from llm_bridge import get_llm_bridge, get_model_fn
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

def test_model_fn():
    print("Testing model_fn callable...")
    fn = get_model_fn()
    output = fn("Explain attention in one sentence.")
    print(f"Output: {output}")

def test_status():
    print("Testing status...")
    llm = get_llm_bridge()
    status = llm.status()
    print(f"Status: {status}")
    assert status["initialized"], "Bridge should be initialized"
    assert status["d_model"] > 0, "d_model should be positive"
    assert status["parameters"] > 0, "Should have parameters"

def test_singleton():
    print("Testing singleton pattern...")
    llm1 = get_llm_bridge()
    llm2 = get_llm_bridge()
    assert llm1 is llm2, "Should be the same instance"
    print("  ✓ Singleton verified")

if __name__ == "__main__":
    print("--- LLM Bridge Test Suite ---")
    test_sync()
    asyncio.run(test_async())
    test_model_fn()
    test_status()
    test_singleton()
    print("--- All tests passed ---")
