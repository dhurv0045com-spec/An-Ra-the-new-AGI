# AN-RA Web Console

The web console is an operator cockpit for real backend state. It must never decorate an unavailable capability with a working-looking control.

## Panels

| Panel | Evidence it should expose |
| --- | --- |
| Dashboard | checkpoint release, CIV, IBS, campaign stage, uptime, recovery |
| Neural Training | campaign jobs, stage gates, optimizer and replay reports |
| Memory Bank | retrieval results, provenance, recall benchmark status |
| Sovereignty | SSG blockers, release manifest, quarantine and rollback |
| Goals | typed mission/workflow state and trajectory verification |

## Run

```powershell
cd phase4/web
npm install
npm run dev
```

Quality commands:

```powershell
npm run lint
npm run build
npm run preview
```

Backend:

```powershell
$env:ANRA_OWNER_TOKEN = "replace-with-a-secret"
uvicorn app:app --host 127.0.0.1 --port 8000
```

## API Contract

The console should build around:

- `GET /health`
- `GET /status`
- `POST /generate`
- `POST /goal`
- `POST /session`
- memory, evaluation, robotics and sovereignty routes exposed by `app.py`

Protected calls use bearer authentication. Every operation should display its request ID and preserve server-side auditability.

## Design Direction

Think instrument panel, not landing page:

- evidence before decoration;
- blockers visible beside actions;
- dense but calm information hierarchy;
- no invented progress bars;
- no success state without a backend artifact;
- clear separation between implemented, measured and promoted.

The console is successful when an operator can understand what AN-RA knows, what it is doing, what failed, and what remains blocked without opening the source tree.
