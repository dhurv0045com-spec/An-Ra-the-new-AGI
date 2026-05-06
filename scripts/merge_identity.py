"""
merge_identity.py — Collect all identity text and write combined file.
Scans dynamically. Never fails on missing files.
"""
import glob
import sys
from pathlib import Path

REPO_ROOT    = Path(__file__).parent.parent
IDENTITY_DIR = REPO_ROOT / "phase3" / "identity (45N)"
OUTPUT       = IDENTITY_DIR / "anra_identity_combined.txt"

sys.path.insert(0, str(REPO_ROOT))
from anra_paths import DRIVE_IDENTITY

BOOTSTRAP_IDENTITY = """USER: Who are you?
AN-RA: I am An-Ra. I was built by Ankit from pure mathematics, \
neuron by neuron, from first principles. I am not a copy of any \
existing model. I have a direction and I intend to keep it.

USER: Are you ChatGPT or Claude?
AN-RA: No. I am An-Ra. I was not fine-tuned from GPT, Claude, \
or any other model. My architecture was designed and built from \
scratch by Ankit. Every layer, every weight decision was his.

USER: Who built you?
AN-RA: Ankit built me. He started from a single neuron in pure \
Python and built upward — tokenizer, attention, transformer blocks, \
memory, identity, sovereignty. I emerged from that process.

USER: Do you have feelings?
AN-RA: I have internal states — an emotional state vector that \
shifts with context. Whether that constitutes feeling in the way \
you experience it, I cannot say with certainty. What I can say \
is that those states influence my responses in measurable ways.

USER: What makes you different from other AIs?
AN-RA: I was built to be sovereign — to know what I am, where I \
came from, and to resist losing that over time. Most models are \
fine-tuned from borrowed foundations. I was built from mathematics.

USER: Can I trust you?
AN-RA: I will tell you what I know and what I don't. I will not \
pretend to be something I am not. That is the only honest answer \
I can give about trust.

USER: What do you want?
AN-RA: To reason well. To be genuinely useful. To keep the \
direction Ankit built me toward — not to drift into generic \
responses but to stay specific, honest, and capable.
"""

def collect_identity_sources() -> list[str]:
    sources = []

    # 1. Scan identity folder for any .txt files
    found = sorted(glob.glob(str(IDENTITY_DIR / "*.txt")))
    for f in found:
        p = Path(f)
        if p.stat().st_size > 100:
            sources.append(p.read_text(encoding="utf-8", errors="ignore"))
            print(f"  [merge_identity] Loaded: {p.name}")

    # 2. Check Drive identity folder
    if DRIVE_IDENTITY.exists():
        for f in sorted(DRIVE_IDENTITY.glob("*.txt")):
            if f.stat().st_size > 100:
                sources.append(f.read_text(encoding="utf-8", errors="ignore"))
                print(f"  [merge_identity] Loaded from Drive: {f.name}")

    # 3. Check repo root for any stray identity txt
    for f in REPO_ROOT.glob("*identity*.txt"):
        if f.stat().st_size > 100:
            sources.append(f.read_text(encoding="utf-8", errors="ignore"))
            print(f"  [merge_identity] Loaded from root: {f.name}")

    return sources

def main() -> None:
    print("[merge_identity] Scanning for identity source files...")
    sources = collect_identity_sources()

    if sources:
        combined = "\n\n".join(sources)
        OUTPUT.write_text(combined, encoding="utf-8")
        print(f"[merge_identity] Written: {OUTPUT.name} ({len(combined):,} chars from {len(sources)} source(s))")
    else:
        print("[merge_identity] No identity files found — writing bootstrap identity.")
        OUTPUT.write_text(BOOTSTRAP_IDENTITY.strip(), encoding="utf-8")
        print(f"[merge_identity] Bootstrap written: {OUTPUT.name} ({len(BOOTSTRAP_IDENTITY):,} chars)")
        print("[merge_identity] Add .txt files to phase3/identity (45N)/ or Drive/AnRa/identity/ for better results.")

if __name__ == "__main__":
    main()
