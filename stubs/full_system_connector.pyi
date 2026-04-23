"""Stub for full_system_connector — lives in inference/full_system_connector.py.
inject_all_paths() adds inference/ to sys.path making this importable as top-level.
"""
from inference.full_system_connector import (
    build_capability_graph as build_capability_graph,
    walk_repository as walk_repository,
    phase_snapshots as phase_snapshots,
    save_graph as save_graph,
    FileNode as FileNode,
    PhaseSnapshot as PhaseSnapshot,
)
