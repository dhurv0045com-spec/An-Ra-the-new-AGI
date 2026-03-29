#!/usr/bin/env python3
"""
anra.py — An-Ra AGI Unified Entry Point
=========================================

The single command to interact with An-Ra.

Usage:
    python anra.py                     # Show system dashboard
    python anra.py --start             # Start continuous autonomous engine
    python anra.py --chat              # Interactive chat with memory
    python anra.py --goal "..."        # Execute a goal via Agent Loop
    python anra.py --status            # System status
    python anra.py --briefing          # Morning briefing
    python anra.py --test              # Run full test suite
    python anra.py --dashboard         # Live dashboard
"""

import sys
import os
from pathlib import Path

# Resolve all project paths
PROJECT_ROOT = Path(__file__).resolve().parent
PHASE2_45M   = PROJECT_ROOT / "phase2" / "45M"

# Set working directory to 45M so all relative state/ paths work
os.chdir(str(PHASE2_45M))

# Add 45M to path so system.py imports work
sys.path.insert(0, str(PHASE2_45M))

# Delegate to the master system
from system import main

if __name__ == "__main__":
    main()
