"""Quick smoke test -- verify MasterSystem can be created, started, and queried."""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from system import MasterSystem

print("Creating MasterSystem...")
ms = MasterSystem()
print("  OK")

print("Querying status (pre-start)...")
s = ms.status()
print(f"  Version: {s['version']}")
print(f"  Subsystems: {json.dumps(s.get('subsystems', {}), indent=2, default=str)}")
print("  OK")

print("\nStarting system (will init all subsystems)...")
ms.start()

print("\nFull status (post-start):")
s2 = ms.status()
print(f"  LLM:     {s2['subsystems']['llm']}")
print(f"  Agent:   {s2['subsystems']['agent']}")
mem_info = s2['subsystems']['memory']
print(f"  Memory:  {mem_info}")
print(f"  Improver: {s2['subsystems']['improver']}")
print(f"  Ghost:   {s2['subsystems']['ghost_memory']}")

# Test chat if LLM loaded
if ms.llm:
    print("\nTesting chat...")
    resp = ms.chat("Hello, who are you?")
    print(f"  Response: {resp[:200]}")

print("\nStopping...")
ms.stop()
print("SMOKE TEST PASSED")
