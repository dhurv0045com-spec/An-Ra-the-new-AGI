"""Level-6 pinned Kaggle 2xT4 notebook generator (preflight-before-launch & zero-copy mirror) & verifier."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shlex
from typing import Any, Mapping, Sequence

from .archives import win_long_path


def _launch_and_mirror_lines(
    *,
    indent: str,
    cmd_args: list[str],
    input_names: list[str],
    mirror_outputs_to_working: bool,
    export_globs: list[str],
    use_globals_fallback: bool,
) -> list[str]:
    pad = indent
    if use_globals_fallback:
        lines = [
            f"{pad}env_map = dict(globals().get('run_env') or os.environ)\n",
            f"{pad}env_map['PLANTHIR_CAMPAIGN_MODE'] = str(globals().get('CAMPAIGN_MODE', 'FULL'))\n",
            f"{pad}env_map['PLANTHIR_SESSION_MINUTES'] = str(globals().get('SESSION_MINUTES', 300))\n",
            f"{pad}if globals().get('RESUME_INPUT') is not None:\n",
            f"{pad}    env_map['PLANTHIR_RESUME_INPUT'] = str(globals()['RESUME_INPUT'])\n",
            f"{pad}disc_inputs = globals().get('DISCOVER_INPUTS', {input_names!r})\n",
            f"{pad}launch_cmd = globals().get('LAUNCH_CMD', {cmd_args!r})\n",
            f"{pad}mirror_work = globals().get('MIRROR_WORKING', {bool(mirror_outputs_to_working)!r})\n",
            f"{pad}exp_globs = globals().get('EXPORT_GLOBS', {export_globs!r})\n",
        ]
    else:
        lines = [
            f"{pad}env_map = run_env\n",
            f"{pad}disc_inputs = DISCOVER_INPUTS\n",
            f"{pad}launch_cmd = LAUNCH_CMD\n",
            f"{pad}mirror_work = MIRROR_WORKING\n",
            f"{pad}exp_globs = EXPORT_GLOBS\n",
        ]
    lines.extend(
        [
            f"{pad}if disc_inputs and Path('/kaggle/input').is_dir():\n",
            f"{pad}    for name in disc_inputs:\n",
            f"{pad}        found = sorted(Path('/kaggle/input').rglob(name))\n",
            f"{pad}        print('KAGGLE_INPUT', name, [str(p) for p in found])\n",
            f"{pad}subprocess.check_call(launch_cmd, env=env_map)\n",
            f"{pad}if mirror_work and Path('/kaggle/working').is_dir():\n",
            f"{pad}    for pat in exp_globs:\n",
            f"{pad}        for p in sorted(Path('.').glob(pat)):\n",
            f"{pad}            if p.is_file():\n",
            f"{pad}                dst = Path('/kaggle/working') / p.name\n",
            f"{pad}                if p.resolve() != dst.resolve():\n",
            f"{pad}                    if dst.exists():\n",
            f"{pad}                        dst.unlink()\n",
            f"{pad}                    try:\n",
            f"{pad}                        os.link(p, dst)\n",
            f"{pad}                    except OSError:\n",
            f"{pad}                        shutil.copy2(p, dst)\n",
        ]
    )
    return lines


def build_pinned_kaggle_notebook(
    *,
    source_commit: str,
    title: str,
    launch_command: Sequence[str] | str,
    owner_authorized: bool = True,
    repo_url: str | None = None,
    repo_dir: str = "repo",
    campaign_mode: str = "FULL",
    session_minutes: int = 300,
    resume_input: str | None = None,
    inventory_command: Sequence[str] | str | None = None,
    discover_kaggle_inputs: Sequence[str] = (),
    mirror_outputs_to_working: bool = False,
    extra_pinned_repos: Mapping[str, Mapping[str, str]] | None = None,
    preflight_canary_backend: str | None = None,
    preflight_commands: Sequence[Sequence[str] | str] = (),
    require_dual_gpu: bool = False,
    working_export_globs: Sequence[str] = ("*.zip", "*.sha256", "FINAL_STATUS.json"),
    emit_file_links: bool = False,
    backend_modules: Sequence[str] = (),
) -> dict[str, Any]:
    """Generate a deterministic, pinned Kaggle 2xT4 `.ipynb` executing preflight gates BEFORE launch."""

    cmd_args = list(shlex.split(launch_command)) if isinstance(launch_command, str) else [str(x) for x in launch_command]
    inv_args = (
        list(shlex.split(inventory_command))
        if isinstance(inventory_command, str)
        else ([str(x) for x in inventory_command] if inventory_command is not None else None)
    )
    preflight_cmds = [
        list(shlex.split(cmd)) if isinstance(cmd, str) else [str(x) for x in cmd]
        for cmd in preflight_commands
    ]
    auth_val = "True" if owner_authorized else "False"
    input_names = [str(x) for x in discover_kaggle_inputs]
    export_globs = [str(x) for x in working_export_globs]
    backend_mods = [str(x) for x in backend_modules]
    has_gate_cell = bool(require_dual_gpu or preflight_canary_backend or backend_mods or preflight_cmds)

    code_lines = [
        f"SOURCE_SHA = {source_commit!r}\n",
        f"OWNER_AUTHORIZED = {auth_val}\n",
        f"REPO_URL = {repo_url!r}\n",
        f"REPO_DIR = {str(repo_dir)!r}\n",
        f"CAMPAIGN_MODE = {str(campaign_mode)!r}\n",
        f"SESSION_MINUTES = {int(session_minutes)!r}\n",
        f"RESUME_INPUT = {resume_input!r}\n",
        f"DISCOVER_INPUTS = {input_names!r}\n",
        f"MIRROR_WORKING = {bool(mirror_outputs_to_working)!r}\n",
        f"EXPORT_GLOBS = {export_globs!r}\n",
        f"INVENTORY_CMD = {inv_args!r}\n",
        f"LAUNCH_CMD = {cmd_args!r}\n",
        "from pathlib import Path\n",
        "import os, shutil, subprocess, sys\n",
        "if REPO_URL:\n",
        "    rp = Path(REPO_DIR)\n",
        "    if not (rp / '.git').exists():\n",
        "        subprocess.check_call(['git', 'clone', '--no-checkout', REPO_URL, str(rp)])\n",
        "    if subprocess.run(['git', 'fetch', '--depth', '1', 'origin', SOURCE_SHA], cwd=str(rp)).returncode != 0:\n",
        "        subprocess.check_call(['git', 'fetch', 'origin'], cwd=str(rp))\n",
        "    subprocess.check_call(['git', 'checkout', '--detach', SOURCE_SHA], cwd=str(rp))\n",
        "    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(rp), text=True).strip()\n",
        "    assert head == SOURCE_SHA, f'HEAD mismatch: {head} != {SOURCE_SHA}'\n",
        "    os.chdir(str(rp))\n",
        "    if str(Path.cwd()) not in sys.path:\n",
        "        sys.path.insert(0, str(Path.cwd()))\n",
        "run_env = dict(os.environ)\n",
        "run_env['PLANTHIR_CAMPAIGN_MODE'] = str(CAMPAIGN_MODE)\n",
        "run_env['PLANTHIR_SESSION_MINUTES'] = str(SESSION_MINUTES)\n",
        "if RESUME_INPUT is not None:\n",
        "    run_env['PLANTHIR_RESUME_INPUT'] = str(RESUME_INPUT)\n",
        "if not OWNER_AUTHORIZED:\n",
        "    print('NOT LAUNCHED: set OWNER_AUTHORIZED = True before running GPU launch.')\n",
        "    if INVENTORY_CMD:\n",
        "        subprocess.check_call(INVENTORY_CMD, env=run_env)\n",
        "else:\n",
        "    if not REPO_URL:\n",
        "        subprocess.check_call(['git', 'checkout', SOURCE_SHA])\n",
    ]
    if has_gate_cell:
        code_lines.append("    print('CHECKOUT_READY: proceeding to Preflight Gate & Authorized Launch cell.')\n")
    else:
        code_lines.extend(
            _launch_and_mirror_lines(
                indent="    ",
                cmd_args=cmd_args,
                input_names=input_names,
                mirror_outputs_to_working=mirror_outputs_to_working,
                export_globs=export_globs,
                use_globals_fallback=False,
            )
        )

    cells: list[dict[str, Any]] = [
        {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n", f"Pinned commit: `{source_commit}`\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_lines},
    ]
    if extra_pinned_repos:
        repo_items = sorted((str(k), dict(v)) for k, v in extra_pinned_repos.items())
        extra_lines = [
            f"EXTRA_PINNED_REPOS = {repo_items!r}\n",
            "import subprocess\n",
            "from pathlib import Path\n",
            "for name, cfg in EXTRA_PINNED_REPOS:\n",
            "    dest = Path(cfg.get('dest', f'./pinned_{name}'))\n",
            "    if not dest.exists():\n",
            "        subprocess.check_call(['git', 'clone', cfg['repo_url'], str(dest)])\n",
            "    subprocess.check_call(['git', 'checkout', cfg['commit']], cwd=str(dest))\n",
            "    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(dest), text=True).strip()\n",
            "    assert head == cfg['commit'], f'{name} HEAD mismatch: {head}'\n",
        ]
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": extra_lines})

    if has_gate_cell:
        gate_lines = [
            f"REQUIRE_DUAL_GPU = {bool(require_dual_gpu)!r}\n",
            f"PREFLIGHT_BACKEND = {preflight_canary_backend!r}\n",
            f"BACKEND_MODULES = {backend_mods!r}\n",
            f"PREFLIGHT_COMMANDS = {preflight_cmds!r}\n",
            "import os, shutil, subprocess, sys, torch\n",
            "from pathlib import Path\n",
            "if BACKEND_MODULES:\n",
            "    os.environ['PLANTHIR_BACKEND_MODULES'] = ','.join(BACKEND_MODULES)\n",
            "if REQUIRE_DUAL_GPU and globals().get('OWNER_AUTHORIZED', False):\n",
            "    assert torch.cuda.device_count() >= 2, f'Expected >=2 GPUs, found {torch.cuda.device_count()}'\n",
            "elif REQUIRE_DUAL_GPU:\n",
            "    print(f'DRY-RUN (OWNER_AUTHORIZED=False): skipping dual-GPU assertion (visible GPUs={torch.cuda.device_count()})')\n",
            "subprocess.check_call([sys.executable, '-m', 'anra_v5.palantiri', 'doctor'])\n",
            "for pf_cmd in PREFLIGHT_COMMANDS:\n",
            "    subprocess.check_call(pf_cmd)\n",
            "if PREFLIGHT_BACKEND:\n",
            "    cdir = Path('work/preflight_canary')\n",
            "    cdir.mkdir(parents=True, exist_ok=True)\n",
            "    spec_p = cdir / 'spec.json'\n",
            "    subprocess.check_call([sys.executable, '-m', 'anra_v5.palantiri', 'init', '--backend', PREFLIGHT_BACKEND, '--out', str(spec_p)])\n",
            "    subprocess.check_call([sys.executable, '-m', 'anra_v5.palantiri', 'validate', '--spec', str(spec_p), '--canary'])\n",
            "    subprocess.check_call([sys.executable, '-m', 'anra_v5.palantiri', 'run', '--spec', str(spec_p), '--work-root', str(cdir / 'cpu'), '--canary', '--device', 'cpu'])\n",
            "    subprocess.check_call([sys.executable, '-m', 'anra_v5.palantiri', 'verify-bundle', '--bundle-dir', str(cdir / 'cpu' / 'result_bundle')])\n",
            "    subprocess.check_call([sys.executable, '-m', 'anra_v5.palantiri', 'replay-bundle', '--bundle-dir', str(cdir / 'cpu' / 'result_bundle'), '--output', str(cdir / 'REPLAY_RECEIPT.json')])\n",
            "if globals().get('OWNER_AUTHORIZED', False):\n",
        ]
        gate_lines.extend(
            _launch_and_mirror_lines(
                indent="    ",
                cmd_args=cmd_args,
                input_names=input_names,
                mirror_outputs_to_working=mirror_outputs_to_working,
                export_globs=export_globs,
                use_globals_fallback=True,
            )
        )
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": gate_lines})

    if emit_file_links:
        link_lines = [
            f"LINK_GLOBS = {export_globs!r}\n",
            "from pathlib import Path\n",
            "try:\n",
            "    from IPython.display import FileLink, display\n",
            "except Exception:\n",
            "    FileLink = None\n",
            "wdir = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('.')\n",
            "for pat in LINK_GLOBS:\n",
            "    for f in sorted(wdir.glob(pat)):\n",
            "        if FileLink is not None:\n",
            "            display(FileLink(str(f)))\n",
            "        else:\n",
            "            print('ARTIFACT:', f)\n",
        ]
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": link_lines})

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "planthir_source_commit": source_commit,
        },
        "cells": cells,
    }


def serialize_pinned_notebook(notebook: Mapping[str, Any]) -> bytes:
    """Serialize a pinned notebook dictionary deterministically to UTF-8 JSON bytes."""

    return (json.dumps(notebook, indent=2, sort_keys=True) + "\n").encode("utf-8")


def verify_pinned_notebook_bytes(
    notebook_path: Path | str,
    expected_notebook: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a saved `.ipynb` file against an expected `build_pinned_kaggle_notebook` payload."""

    np = Path(notebook_path)
    lp = win_long_path(np)
    if not lp.is_file():
        raise FileNotFoundError(f"notebook file not found: {np}")
    actual_bytes = lp.read_bytes()
    expected_bytes = serialize_pinned_notebook(expected_notebook)
    actual_sha = sha256(actual_bytes).hexdigest()
    expected_sha = sha256(expected_bytes).hexdigest()
    try:
        actual_parsed = json.loads(actual_bytes.decode("utf-8"))
    except Exception:
        actual_parsed = None
    return {
        "schema": "anra-planthir-notebook-verify-v1",
        "match": actual_bytes == expected_bytes,
        "parsed_match": actual_parsed == dict(expected_notebook),
        "actual_sha256": actual_sha,
        "expected_sha256": expected_sha,
        "byte_size": len(actual_bytes),
        "notebook_path": str(np),
    }


__all__ = [
    "build_pinned_kaggle_notebook",
    "serialize_pinned_notebook",
    "verify_pinned_notebook_bytes",
]
