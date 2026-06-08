# An-Ra Web Console

**Component 07/19 · `api_web`**

This is the operator cockpit — not a Vite starter you forgot about. React dashboard wired to the An-Ra runtime conceptually through `app.py` and the same telemetry/goals/memory surfaces the CLI exposes.

---

## Panels

| Tab | What you see |
| --- | --- |
| **Dashboard** | Telemetry, chat, goal tracker |
| **Neural Training** | Training controls + progress |
| **Memory Bank** | Recall and memory inspection |
| **Sovereignty** | Audit, benchmarks, governance |

**Key files:** `src/App.jsx`, `src/index.css`, `src/components/*`

---

## Run locally

```bash
cd phase4/web
npm install
npm run dev
```

Build / preview / lint:

```bash
npm run build
npm run preview
npm run lint
```

---

## Pair with backend

```bash
# API
python app.py

# Full system CLI
python anra.py --status
python anra.py --dashboard
```

The UI should reflect **real** backend state — no decorative controls for behavior that does not exist yet.

---

## Design direction

Think **cockpit**, not landing page:

- dense enough for daily use
- status and artifacts visible at a glance
- training and sovereignty state honest
- memory/goals inspectable without clutter

If this README drifts back into generic Vite boilerplate, delete that drift. This folder is the web face of a 19-component organism.
