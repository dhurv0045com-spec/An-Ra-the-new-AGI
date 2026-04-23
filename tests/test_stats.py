import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anra_paths import ROOT, inject_all_paths, MASTER_SYSTEM_DIR
inject_all_paths()

import importlib.util
path = str(MASTER_SYSTEM_DIR / "system.py")
spec = importlib.util.spec_from_file_location("master_system", path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MasterSystem = module.MasterSystem

try:
    print("Initializing MasterSystem...")
    system = MasterSystem()
    system.start()
    
    print("\nChecking Memory Stats...")
    if system.memory:
        try:
            m_stats = system.memory.stats()
            print(f"Memory Stats: {m_stats}")
        except Exception as e:
            print(f"FAILED memory.stats(): {e}")
    else:
        print("Memory system not active.")
        
    print("\nChecking KB Stats...")
    if system.knowledge_base:
        try:
            kb_stats = system.knowledge_base.stats()
            print(f"KB Stats: {kb_stats}")
        except Exception as e:
            print(f"FAILED knowledge_base.stats(): {e}")
    else:
        print("KB system not active.")

    print("\nChecking Ghost Memory...")
    if system.ghost_memory:
        print("Ghost memory is active.")
    else:
        print("Ghost memory not active.")
        
    system.stop()
    print("\nDone.")

except Exception as e:
    import traceback
    print(f"\nCRITICAL GLOBAL FAILURE: {e}")
    traceback.print_exc()
