# Trace Replay Scheduler State Machine: Presentation Version

This is the compact, single-diagram version of the replay scheduler.

Primary implementation:
- `src/trace_replay/simulator.py`
- `src/trace_replay/admission.py`
- `src/trace_replay/tasks.py`
- `src/trace_replay/forecast.py`

```mermaid
flowchart TD
    A[Request arrival] --> B{Guarded admission enabled?}

    B -->|No| C[Route choice at arrival]
    B -->|Yes| D[Enter waiting_admission]

    D --> E[Evaluate routes and harm to existing requests]
    E --> F{Eligible route exists?}
    F -->|No| G{Wait/bypass limit hit?}
    G -->|No| H[admission_hold<br/>retry later]
    H --> E
    G -->|Yes| I[best-effort admit or drop]
    F -->|Yes| C

    C --> J[waiting_prefill]
    I --> J

    J --> K[Choose next task]
    K --> L[prefill dispatch]
    L --> M{Decode tokens remain?}
    M -->|No| N[done]
    M -->|Yes| O[waiting_decode]

    O --> P[Choose next decode task or decode batch]
    P --> Q[decode dispatch]
    Q --> R{More decode tokens remain?}
    R -->|Yes| O
    R -->|No| N

    C --> S[dropped]
    I --> S
```

## Exported Figures
- `docs/figures/trace_replay_scheduler_state_machine_presentation.svg`
- `docs/figures/trace_replay_scheduler_state_machine_presentation.png`

## Mermaid Source
- `docs/figures/trace_replay_scheduler_state_machine_presentation.mmd`

## Export Command
```bash
bash tools/export_scheduler_state_machine_figures.sh
```
