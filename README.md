# BOUNDEDNESS — Project BORON

**Sovereign, local-first AI substrate for the IAMBANDOBANDZ / MASSIVEMAGNETICS empire.**

BOUNDEDNESS implements the minimum-viable-consciousness stack as executable code: an operational inside/outside boundary, homeostatic drive regulation, sensorimotor coordination, and a Monte Carlo + active-inference release-protocol engine. Everything runs entirely on your machine — zero telemetry, zero phone-home.

---

## Modules at a Glance

| Module | File | Purpose |
|---|---|---|
| **Boundary Core** | `victor/boundary_core.py` | Inside/outside boundary, ownership scoring, mutation policy |
| **Boundedness Core** | `boundedness_core.py` | Homeostasis + sensorimotor + boundary unified compositor |
| **Majora Core** | `majora_core.py` | Monte Carlo multiverse simulator + FEP active-inference engine |
| **Majora Kernels** | `majora_kernels.py` | Alternative SIR kernel implementations and batch runtimes |

---

## Installation

```bash
# Core dependencies (pure-Python stack — runs on any machine)
pip install numpy scipy networkx

# Optional — Numba JIT acceleration (10–50× speedup on MajoraCore)
pip install numba

# Testing
pip install pytest
```

---

## Boundary Core (`victor/boundary_core.py`)

Implements **Condition A of the Minimum Viable Consciousness schema**: an operational inside/outside boundary with protected internal state.

### What it does

- **Protected internal state** — dot-path tree (`core`, `memory`, `identity`, `body`, `sensors`, `drives`, `continuity`).
- **Sensor registration** — attach sensors to the system's perimeter/body-map with modality, boundary zone, and channel metadata.
- **Event ingress classification** — every event is classified as self-caused vs. external-caused, assigned an ownership score, and stored in the perturbation log.
- **Mutation authorization** — external mutations to protected paths are blocked and logged as boundary alerts.
- **Self-command causality** — issued self-commands are tracked; downstream sensor readings linked to those commands are attributed as `self_caused`.

### Quick Start

```python
from victor.boundary_core import BoundaryCore, SensorSpec, StateMutation, OriginType

core = BoundaryCore()

# Register an owned perimeter sensor
core.register_sensor(SensorSpec(
    sensor_id="cam_front",
    modality="vision",
    boundary_zone="front_surface",
    owned=True,
    channels=("rgb",),
))

# Issue a self-command (tracked for causality attribution)
cmd_id = core.issue_self_command(
    source_id="motor_controller",
    command_name="turn_head_left",
    payload={"degrees": 15},
)

# Ingest resulting sensor reading, linked to the self-command
rec = core.ingest_sensor_reading(
    sensor_id="cam_front",
    reading={"object": "doorway", "angle_offset_deg": -13},
    linked_command_id=cmd_id,
)
print(rec.self_caused)   # True — attribution works

# Block unauthorized external mutation of identity
from victor.boundary_core import UnauthorizedMutationError
try:
    core.apply_mutation(StateMutation(
        path="identity.inside_label",
        value="compromised",
        origin=OriginType.EXTERNAL,
        reason="attack",
    ))
except UnauthorizedMutationError:
    print("blocked")     # Expected

print(core.boundary_report())
```

See **`demo_boundary_core.py`** for a full walkthrough.

---

## Boundedness Core (`boundedness_core.py`)

Composes BoundaryCore, HomeostasisCore, and SensorimotorCore into a single unified API — full embodiment support in one object.

### What it does

- **HomeostasisCore** — register named drive variables (temperature, energy, etc.) with setpoints, tolerances, and ranges; receive urgency signals when drives deviate.
- **SensorimotorCore** — map sensor modalities to motor channels with optional response policy functions; all issued motor commands are stored as efferent copies for reafference tracking.
- **BoundednessCore** — one call to `ingest_sensor_reading()` simultaneously classifies the boundary event, updates drives, and routes motor commands.

### Quick Start

```python
from boundedness_core import (
    BoundednessCore, DriveSpec, SensorimotorLoop, MotorCommand,
)
from victor.boundary_core import SensorSpec

bc = BoundednessCore()

# Register a homeostatic drive
bc.register_drive(DriveSpec(
    drive_id="thermal",
    setpoint=37.0,
    min_val=34.0,
    max_val=42.0,
    tolerance=0.5,
    label="Core temperature (°C)",
))

# Register a sensor
bc.register_sensor(SensorSpec(
    sensor_id="skin_temp",
    modality="temperature",
    boundary_zone="body_surface",
    owned=True,
))

# Register a sensorimotor loop
bc.register_loop(SensorimotorLoop(
    loop_id="thermal_reflex",
    sensor_modality="temperature",
    motor_channel="sweat_glands",
    response_fn=lambda p: MotorCommand.ACTIVATE if p.get("celsius", 37) > 38.5 else None,
))

# One unified call handles all three subsystems
perturbation, homeo_signals, motor_events = bc.ingest_sensor_reading(
    sensor_id="skin_temp",
    reading={"celsius": 39.2},
    drive_updates={"thermal": 39.2},
)
print(homeo_signals[0].urgency)   # elevated
print(motor_events[0].command)    # MotorCommand.ACTIVATE
```

---

## Majora Core (`majora_core.py`)

**Sovereign Empire AI Deployment Kernel** — FEP-active, Numba-accelerated, local-first.

Takes raw content metadata and runs thousands of Monte Carlo multiverse simulations on a stochastic social-virality graph. The Active Inference Layer scores each timeline via Expected Free Energy (EFE) and returns the lowest-EFE deployable release protocol.

### Architecture

```
Content Metadata / Hash
        │
        ▼
  ┌─────────────┐   n_sims timelines   ┌─────────────┐
  │  MajoraCore │ ──────────────────▶  │   SIRGraph  │  (Numba JIT)
  └─────────────┘                      └─────────────┘
        │                                     │
        │         sim_results                 │
        ▼                                     ▼
  ┌──────────────────────┐        ┌──────────────────────┐
  │  ActiveInferenceLayer│        │   sir_step_numba      │
  │  (EFE scoring)       │        │   SEAIF transitions   │
  └──────────────────────┘        └──────────────────────┘
        │
        ▼
  Deployable Protocol
  (drop_time, EFE score,
   captions, geo targets)
```

### Social Graph States (SEAIF)

| Code | Name | Meaning |
|------|------|---------|
| 0 | **S** | Susceptible — listener hasn't heard the track |
| 1 | **E** | Exposed — listener has seen a post/teaser |
| 2 | **I** | Infected — listener has streamed/saved |
| 3 | **A** | Amplifier — listener shares/memes the track |
| 4 | **F** | Faded — listener loses interest |

### Expected Free Energy (EFE)

```
EFE(π) = Risk   +  Ambiguity   −  Info Gain   −  Canon Bonus
           ↑           ↑              ↑               ↑
     KL from      entropy of    exploration     mortality urgency
    preferred      outcomes      when low        + Lorain geo
      state                    convergence      + convergence
                                                + virality
```

Lower EFE → better protocol. The canon bonus directly penalises fragmented protocols.

### Quick Start

```python
from majora_core import MajoraCore

majora = MajoraCore(n_nodes=5000)
majora.start_background_majora()

# Seed with track metadata
result = majora.run_monte_carlo(
    content={
        "title": "WE ALL DIE ONE DAY",
        "artist": "IAMBANDOBANDZ",
        "themes": ["mortality", "urgency", "rust_belt_survival"],
    },
    preset="music_single",
    n_sims=2000,
    beta=0.08,    # transmission rate (2026 Spotify first-48h calibration)
    gamma=0.15,   # E→I conversion (save rate)
    delta=0.02,   # fade rate
)

print(result["winning_drop_time"])       # "2026-04-17 19:01 EDT"
print(result["efe_score"])               # float — lower is better
print(result["recommended_protocol"])    # human-readable plan

# After real drop — feed actual metrics back to tune priors
majora.ai_layer.auto_tune_priors({
    "urgency_impact": 0.91,          # from lyric resonance analysis
    "rust_belt_conversion": 0.74,    # from UnitedMasters geo export
    "convergence_achieved": 0.92,    # from catalog cohesion score
})

majora.stop_background()
```

See **`demo_majora_core.py`** for a full multi-scenario walkthrough.

### Canon Preferences (tunable priors)

```python
from majora_core import CanonPreferences, ActiveInferenceLayer

prefs = CanonPreferences(
    mortality_urgency_weight=0.40,   # boost for mortality/pressure lyrics
    lorain_roots_geo_weight=0.25,    # Rust Belt seeding priority
    builder_proof_of_work_weight=0.20,
    virality_index_weight=0.15,
    adaptation_rate=0.05,
)
layer = ActiveInferenceLayer(initial_prefs=prefs)
```

---

## Majora Kernels (`majora_kernels.py`)

Additional SIR kernel implementations and runtime variants for different hardware and scale requirements.

| Kernel / Runtime | Class / Function | When to use |
|---|---|---|
| Sparse matrix kernel | `sir_step_sparse` | Large graphs (>50k nodes) where dense adj matrix won't fit in RAM |
| Vectorised NumPy kernel | `sir_step_vectorised` | Medium graphs without Numba; fast pure-NumPy fallback |
| Batch parallel runtime | `BatchMajoraRuntime` | Run multiple independent content hashes concurrently via `concurrent.futures` |
| Preset registry | `PRESET_CONFIGS` | Pre-calibrated beta/gamma/delta per release type |
| Protocol formatter | `format_protocol_report` | Render scored results as a human-readable deployment report |

### Sparse Kernel

```python
from majora_kernels import SparseGraph, sir_step_sparse
import scipy.sparse as sp

graph = SparseGraph(n_nodes=100_000, seed=42)
graph.initialize_states()
graph.step()  # uses scipy.sparse operations — no dense matrix needed
```

### Batch Runtime

```python
from majora_kernels import BatchMajoraRuntime

runtime = BatchMajoraRuntime(n_nodes=3000, max_workers=4)
contents = [
    {"title": "WE ALL DIE ONE DAY", "artist": "IAMBANDOBANDZ"},
    {"title": "REPO: BOUNDEDNESS v1.0", "type": "repo_release"},
    {"title": "ALBUM: EMPIRE VOLUME 1", "type": "album"},
]
results = runtime.run_batch(contents, preset="music_single", n_sims=200)
for r in results:
    print(r["content_hash"], r["efe_score"])
```

---

## Testing

```bash
# All tests
python3 -m pytest tests/ -v

# Majora core only
python3 -m pytest tests/test_majora_core.py -v

# Kernel tests only
python3 -m pytest tests/test_majora_kernels.py -v

# Boundary core only
python3 -m pytest tests/test_boundary_core.py -v
```

---

## Demos

| Script | Covers |
|---|---|
| `demo_boundary_core.py` | Sensor registration, self-command causality, mutation control |
| `demo_majora_core.py` | Full Majora pipeline: single, batch, prior tuning, report rendering |

```bash
python3 demo_boundary_core.py
python3 demo_majora_core.py
```

---

## Project Structure

```
BOUNDEDNESS/
├── README.md
├── victor/
│   ├── __init__.py
│   └── boundary_core.py          # Inside/outside boundary, ownership, mutation policy
├── boundedness_core.py           # Homeostasis + sensorimotor + boundary compositor
├── majora_core.py                # Monte Carlo + FEP active-inference engine
├── majora_kernels.py             # Sparse, vectorised, batch kernel variants
├── demo_boundary_core.py         # Boundary core demonstration
├── demo_majora_core.py           # Majora full-pipeline demonstration
└── tests/
    ├── __init__.py
    ├── test_boundary_core.py     # 57 boundary tests
    ├── test_majora_core.py       # Majora core tests
    └── test_majora_kernels.py    # Kernel variant tests
```

---

## License

Proprietary — Massive Magnetics / Ethica AI / BHeard Network.
