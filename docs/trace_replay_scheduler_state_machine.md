# Trace Replay Scheduler State Machine

This document describes the scheduler implemented in `src/trace_replay/`.

Primary code paths:
- `src/trace_replay/simulator.py`
- `src/trace_replay/admission.py`
- `src/trace_replay/tasks.py`
- `src/trace_replay/forecast.py`
- `src/trace_replay/types.py`

## Scope
This state machine describes the replay scheduler, not the DRAM frontend in `ramulator2/`.

One important detail: `ReplayRequest.state` only uses persistent request states:
- `new`
- `waiting_admission`
- `waiting_prefill`
- `waiting_decode`
- `done`
- `dropped`

The debug plots also show `running_prefill` and `running_decode`, but those are derived from `dispatch_task` / `complete_task` events. They are not stored as persistent request states.

## 1. Request Lifecycle
```mermaid
stateDiagram-v2
    [*] --> new

    new --> waiting_admission: guarded admission enabled\narrival enqueued
    new --> waiting_prefill: immediate admit_route
    new --> dropped: admit_drop / no feasible route

    waiting_admission --> waiting_admission: admission_hold\nset next_retry_ms
    waiting_admission --> waiting_admission: admission_bypass\nlater request admitted first
    waiting_admission --> waiting_prefill: admit_route
    waiting_admission --> waiting_prefill: admission_best_effort\nforced_best_effort=true
    waiting_admission --> dropped: admit_drop\nno route even for best effort

    waiting_prefill --> waiting_decode: prefill task completes\nremaining_decode_tokens > 0
    waiting_prefill --> done: prefill task completes\nremaining_decode_tokens == 0

    waiting_decode --> waiting_decode: decode task completes\nremaining_decode_tokens > 0
    waiting_decode --> done: decode task completes\nremaining_decode_tokens == 0

    done --> [*]
    dropped --> [*]
```

## 2. Top-Level Scheduler Loop
```mermaid
flowchart TD
    A[Start run] --> B[Set now_ms to first arrival or 0]
    B --> C{Admission queue enabled?}
    C -->|Yes| D[Enqueue arrivals up to now_ms]
    D --> E[Process guarded admission queue]
    C -->|No| F[Admit arrivals immediately]
    E --> G[Choose next task]
    F --> G

    G --> H{Task available?}
    H -->|No| I[Find next arrival or admission retry]
    I --> J{Any future event?}
    J -->|No| K[Finish run]
    J -->|Yes| L[Advance now_ms to next event]
    L --> C

    H -->|Yes| M{Earlier arrival/retry before task start?}
    M -->|Yes| N[Yield to earlier event]
    N --> C
    M -->|No| O[Dispatch task]
    O --> P[Reserve GPU/PIM resources]
    P --> Q[Complete task and update request states]
    Q --> C
```

## 3. Guarded Admission Sub-State Machine
```mermaid
flowchart TD
    A[Request arrives] --> B[Create ReplayRequest]
    B --> C[State = waiting_admission]
    C --> D[Evaluate all routes]
    D --> E{Any eligible route?}

    E -->|Yes| F[Policy choose_route]
    F --> G[admit_route\nstate = waiting_prefill]

    E -->|No| H{max_wait_ms or max_bypass_count hit?}
    H -->|Yes| I{Any route estimate exists?}
    I -->|No| J[admit_drop\nstate = dropped]
    I -->|Yes| K[admission_best_effort\nstate = waiting_prefill\nis_lost = true]

    H -->|No| L[admission_hold]
    L --> M[Record reject reason / harm stats]
    M --> N[Set admission_next_retry_ms]
    N --> C
```

### Eligibility logic in guarded admission
A route candidate is admitted only if it passes the configured checks.

Current default behavior:
- self-SLO checks are optional and off by default
- harm-to-existing-request checks are the default gate

So the dominant question is:
- if this candidate is admitted now, do existing non-lost requests become infeasible?

If yes:
- hold in `waiting_admission`

If no:
- admit into `waiting_prefill`

## 4. Task Selection Policy State Machine
```mermaid
flowchart TD
    A[Need next task] --> B{local_scheduling_policy}

    B -->|fcfs_strict| C[Pick fcfs_active_request]
    C --> D{Request state}
    D -->|waiting_prefill| E[Single-request prefill task]
    D -->|waiting_decode| F[Single-request decode task]

    B -->|prefill_priority_fcfs_decode| G{Any waiting prefill?}
    G -->|Yes| H[Pick oldest prefill request]
    H --> I[Single-request prefill task]
    G -->|No| J[Build FCFS-prefix decode batch\nhead plus same-route ready prefix]

    B -->|general prompt-priority path| K[Build all prefill candidates]
    K --> L[Build all decode candidates]
    L --> M{Prefill guard or decode limit triggered?}
    M -->|Yes| N[Force a prefill task]
    M -->|No| O[Choose min earliest_start_ms, phase priority, ready_ms, request_id]
```

## 5. Task Completion State Machine
```mermaid
flowchart TD
    A[Task dispatched] --> B{phase}

    B -->|prefill| C[Update prefill timing and energy]
    C --> D{remaining_decode_tokens > 0?}
    D -->|Yes| E[state = waiting_decode\nready_ms = finish_ms]
    E --> F{decode rebind enabled?}
    F -->|Yes| G[Optionally rebind decode route to gpu_only]
    F -->|No| H[Keep current route]
    G --> I[Refresh active predictions if route changed]
    H --> J[Continue]
    D -->|No| K[state = done\ncompletion_time_ms = finish_ms]

    B -->|decode| L[Update decode timing, energy, token counters]
    L --> M{remaining_decode_tokens > 0?}
    M -->|Yes| N[state = waiting_decode\nready_ms = finish_ms]
    M -->|No| O[state = done\ncompletion_time_ms = finish_ms]
```

## 6. Decode Rebind (Optional)
Decode rebind is a small state machine nested inside the `waiting_prefill -> waiting_decode` transition.

It only runs when:
- `enable_decode_rebind = true`
- the request still has decode work left
- the current route uses PIM
- the request has not already started decode

```mermaid
flowchart TD
    A[Prefill finished on PIM route] --> B[Predict decode on current route]
    B --> C[Predict decode on gpu_only]
    C --> D[Policy choose_route]
    D --> E{gpu_only wins by more than decode_rebind_margin_ms?}
    E -->|Yes| F[Switch route to gpu_only]
    E -->|No| G[Keep current route]
```

## 7. Practical Interpretation
The scheduler is best understood as two coupled machines:

1. Admission machine
- decides when a request is allowed to enter service
- chooses its initial route
- may delay, bypass, force best-effort, or drop

2. Service machine
- runs admitted requests through prefill and decode
- selects the next executable task under the local scheduling policy
- updates resource availability and request completion state

That separation explains most behaviors seen in the debug plots:
- empty queue snapshots: service machine only, no admission queue
- many `admission_hold` / `admission_bypass` events: admission machine dominates
- `running_prefill` / `running_decode` bars: event-derived execution phases, not persistent request states
