# An-Ra Web Console

**Layer 07/19: `api_web`**

This is the Phase 4 operator interface for An-Ra. It is a Vite + React dashboard connected to the system conceptually through the API/runtime layer, not a generic starter template.

## Current Panels

| Panel | Purpose |
| --- | --- |
| Dashboard | System telemetry, chat, and goal overview |
| Neural Training | Training controls and progress surface |
| Memory Bank | Memory inspection and recall surface |
| Sovereignty | Audit, benchmark, and governance visibility |

Main files:

- `src/App.jsx`
- `src/index.css`
- `src/components/SystemTelemetry.jsx`
- `src/components/AgentGoalTracker.jsx`
- `src/components/ChatInterface.jsx`
- `src/components/TrainingPanel.jsx`
- `src/components/MemoryExplorer.jsx`
- `src/components/SovereigntyPanel.jsx`

## Run

```bash
cd phase4/web
npm install
npm run dev
```

Build:

```bash
npm run build
npm run preview
```

Lint:

```bash
npm run lint
```

## Backend Pairing

The backend entry point is:

```bash
python app.py
```

The unified system entry point is:

```bash
python anra.py --status
python anra.py --dashboard
```

## Design Direction

The console should feel like an operator cockpit, not a landing page:

- dense enough for repeated use
- clear status and artifact surfaces
- visible training and sovereignty state
- memory and goal inspection without decorative clutter
- direct controls only where the backend behavior is real

Do not let this README drift back into Vite boilerplate. This folder is the web face of the An-Ra stack.
