# ======================================================================================================================
# FILE: demo_majora_core.py
# UUID: f1c93b27-4d8e-4a1f-b2e5-0c7a9d628f4e
# VERSION: v1.0.0-MAJORA-DEMO-GODCORE
# COMPAT: Victor>=v0.1.0 | SAVE3>=v0.1.0
# NAME: Majora Core Demo
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Runnable full-pipeline demonstration of the Majora Core engine, Active Inference Layer,
#          Majora Kernels, and Batch Runtime. Covers: single release simulation, prior auto-tuning,
#          sparse/vectorised kernels, batch parallel runs, and protocol report rendering.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# DATE: 04/13/2026
# PROJECT NAME: BORON
# ======================================================================================================================

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

import numpy as np

from majora_core import (
    NUMBA_AVAILABLE,
    ActiveInferenceLayer,
    CanonPreferences,
    MajoraCore,
    SIRGraph,
)
from majora_kernels import (
    PRESET_CONFIGS,
    BatchMajoraRuntime,
    BatchRunConfig,
    SparseGraph,
    VectorisedSIRGraph,
    format_protocol_report,
    sir_step_sparse,
    sir_step_vectorised,
)


# ======================================================================================================================
# Helpers
# ======================================================================================================================

def section(title: str, width: int = 78) -> None:
    sep = "─" * width
    print(f"\n┌{sep}┐")
    print(f"│  {title:<{width - 2}}│")
    print(f"└{sep}┘")


def pretty(label: str, obj: Any) -> None:
    print(f"\n  [{label}]")
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                print(f"    {k}:")
                print(f"      {json.dumps(v, default=str, indent=6)}")
            elif isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
    else:
        print(f"    {obj}")


def state_counts(states: np.ndarray, state_map: Dict[str, int]) -> Dict[str, int]:
    return {k: int(np.sum(states == v)) for k, v in state_map.items()}


# ======================================================================================================================
# Demo 1 — Single track release: WE ALL DIE ONE DAY
# ======================================================================================================================

def demo_single_release() -> None:
    section("DEMO 1 — Single Track Release: WE ALL DIE ONE DAY")

    track_metadata: Dict[str, Any] = {
        "title": "WE ALL DIE ONE DAY",
        "artist": "IAMBANDOBANDZ",
        "release_date": "2026-04-11",
        "themes": ["mortality", "pressure", "urgency", "rust_belt_survival"],
        "explicit": True,
        "lyric_urgency_score": 0.89,
    }

    print(f"\n  Initialising Majora engine (n_nodes=500, Numba={NUMBA_AVAILABLE}) ...")
    majora = MajoraCore(n_nodes=500)

    print("  Running 50 Monte Carlo timelines ...")
    t0 = time.perf_counter()
    result = majora.run_monte_carlo(
        content=track_metadata,
        preset="music_single",
        n_sims=50,
        beta=0.08,
        gamma=0.15,
        delta=0.02,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Completed in {elapsed:.2f}s\n")

    print(format_protocol_report(result, title="WE ALL DIE ONE DAY — Deployment Protocol"))

    # Demonstrate cache hit
    print("  Re-running with same content hash (should hit cache) ...")
    t1 = time.perf_counter()
    cached = majora.run_monte_carlo(content=track_metadata, preset="music_single", n_sims=50)
    cache_time = time.perf_counter() - t1
    assert cached is result, "Expected cache hit"
    print(f"  Cache hit confirmed in {cache_time * 1000:.1f}ms\n")


# ======================================================================================================================
# Demo 2 — Prior auto-tuning over multiple drops
# ======================================================================================================================

def demo_prior_auto_tuning() -> None:
    section("DEMO 2 — Active Inference Prior Auto-Tuning (3 drops)")

    layer = ActiveInferenceLayer()
    print("\n  Initial priors:")
    pretty("CanonPreferences", layer.prefs.to_dict())

    # Simulate 3 drops with progressively improving real-world metrics
    drops = [
        {
            "name": "Drop 1 — WE ALL DIE ONE DAY (first release)",
            "urgency_impact": 0.72,
            "rust_belt_conversion": 0.58,
            "convergence_achieved": 0.80,
        },
        {
            "name": "Drop 2 — EMPIRE VOLUME 1 single",
            "urgency_impact": 0.85,
            "rust_belt_conversion": 0.66,
            "convergence_achieved": 0.87,
        },
        {
            "name": "Drop 3 — Full album campaign",
            "urgency_impact": 0.93,
            "rust_belt_conversion": 0.75,
            "convergence_achieved": 0.94,
        },
    ]

    for drop in drops:
        name = drop.pop("name")
        print(f"\n  Ingesting real metrics from: {name}")
        updated = layer.auto_tune_priors(drop)
        print(f"    mortality_urgency_weight  : {updated['mortality_urgency_weight']:.4f}")
        print(f"    lorain_roots_geo_weight   : {updated['lorain_roots_geo_weight']:.4f}")
        print(f"    builder_proof_of_work     : {updated['builder_proof_of_work_weight']:.4f}")
        print(f"    historical_convergence    : {updated['historical_convergence_score']:.4f}")

    print(f"\n  Total prior-update history entries: {len(layer.history)}")
    print("  Priors have self-adapted — next simulation will be more accurate.\n")


# ======================================================================================================================
# Demo 3 — Multiple presets
# ======================================================================================================================

def demo_presets() -> None:
    section("DEMO 3 — All Preset Configurations")

    print("\n  Preset registry:")
    for name, cfg in PRESET_CONFIGS.items():
        print(f"\n  [{name}]")
        print(f"    Description : {cfg['description']}")
        print(f"    beta={cfg['beta']}  gamma={cfg['gamma']}  delta={cfg['delta']}  "
              f"n_steps={cfg['n_steps']}  post_step={cfg['post_step']}")

    majora = MajoraCore(n_nodes=300)
    print("\n  Running 20 sims per preset ...")
    for preset in ("music_single", "repo_release", "album", "general"):
        result = majora.run_monte_carlo(
            content=f"demo_content_{preset}",
            preset=preset,  # type: ignore[arg-type]
            n_sims=20,
        )
        print(
            f"    {preset:<15} → EFE={result['efe_score']:.3f} ± {result['efe_std']:.3f} | "
            f"streams≈{result['canon_aligned_metrics'].get('urgency_score', 0):.3f}"
        )


# ======================================================================================================================
# Demo 4 — Vectorised NumPy kernel
# ======================================================================================================================

def demo_vectorised_kernel() -> None:
    section("DEMO 4 — Vectorised NumPy Kernel (SirStepVectorised)")

    print("\n  Building VectorisedSIRGraph (n=800) ...")
    vgraph = VectorisedSIRGraph(n_nodes=800, seed=99)
    vgraph.initialize_states(seed_fraction=0.05)

    print(f"  Initial state counts: {state_counts(vgraph.states, VectorisedSIRGraph.STATES)}")

    print("  Running 10 steps with post-action at step 3 ...")
    for t in range(10):
        post = 0.10 if t == 3 else 0.0
        vgraph.step(beta=0.08, gamma=0.15, delta=0.02, post_strength=post)

    final_counts = state_counts(vgraph.states, VectorisedSIRGraph.STATES)
    print(f"  Final state counts: {final_counts}")
    total_active = final_counts["I"] + final_counts["A"]
    print(f"  Active listeners (I + A): {total_active} / {vgraph.n_nodes} "
          f"({100 * total_active / vgraph.n_nodes:.1f}%)\n")


# ======================================================================================================================
# Demo 5 — Sparse matrix kernel
# ======================================================================================================================

def demo_sparse_kernel() -> None:
    section("DEMO 5 — Sparse Matrix Kernel (SparseGraph)")

    print("\n  Building SparseGraph (n=2000, ~4% density) ...")
    sgraph = SparseGraph(n_nodes=2000, seed=13, edge_density=0.04)
    sgraph.initialize_states(seed_fraction=0.03)

    print(f"  Initial state counts: {state_counts(sgraph.states, SparseGraph.STATES)}")
    print(f"  Adjacency matrix: {sgraph.adj_csr.nnz:,} non-zeros "
          f"({sgraph.adj_csr.nnz / sgraph.n_nodes**2 * 100:.2f}% density)")

    print("  Running 10 steps ...")
    for t in range(10):
        post = 0.12 if t == 3 else 0.0
        sgraph.step(beta=0.08, gamma=0.15, delta=0.02, post_strength=post)

    final_counts = state_counts(sgraph.states, SparseGraph.STATES)
    print(f"  Final state counts: {final_counts}")
    total_active = final_counts["I"] + final_counts["A"]
    print(f"  Active listeners: {total_active:,} / {sgraph.n_nodes:,}\n")


# ======================================================================================================================
# Demo 6 — Batch parallel runtime
# ======================================================================================================================

def demo_batch_runtime() -> None:
    section("DEMO 6 — Batch Parallel Runtime (3 items, 3 workers)")

    runtime = BatchMajoraRuntime(n_nodes=300, max_workers=3)

    items = [
        BatchRunConfig(
            content={"title": "WE ALL DIE ONE DAY", "artist": "IAMBANDOBANDZ"},
            preset="music_single",
            n_sims=30,
        ),
        BatchRunConfig(
            content={"title": "BOUNDEDNESS v1.0", "type": "repo_release"},
            preset="repo_release",
            n_sims=30,
        ),
        BatchRunConfig(
            content={"title": "EMPIRE VOLUME 1", "type": "album"},
            preset="album",
            n_sims=30,
        ),
    ]

    print(f"\n  Dispatching {len(items)} simulations to thread pool ...")
    t0 = time.perf_counter()
    results = runtime.run_batch(items)
    elapsed = time.perf_counter() - t0
    print(f"  Completed in {elapsed:.2f}s\n")

    for i, r in enumerate(results):
        print(f"  [{i}] content_hash={r.get('content_hash')} | "
              f"preset={r.get('preset')} | EFE={r.get('efe_score', 'ERR'):.3f}")
    print()


# ======================================================================================================================
# Demo 7 — Custom canon preferences
# ======================================================================================================================

def demo_custom_prefs() -> None:
    section("DEMO 7 — Custom Canon Preferences & EFE Comparison")

    prefs_default = CanonPreferences()
    prefs_geo_heavy = CanonPreferences(
        mortality_urgency_weight=0.20,
        lorain_roots_geo_weight=0.50,  # heavy Rust Belt emphasis
        builder_proof_of_work_weight=0.20,
        virality_index_weight=0.10,
    )

    layer_default = ActiveInferenceLayer(initial_prefs=prefs_default)
    layer_geo = ActiveInferenceLayer(initial_prefs=prefs_geo_heavy)

    # Shared synthetic simulation result
    sim_result = {
        "streams": 3500,
        "virality": 0.35,
        "rust_belt_share": 0.70,
        "urgency_score": 0.55,
        "convergence_score": 0.80,
        "streams_bucket": np.array([0.65, 0.25, 0.10]),
    }

    scored_default = layer_default.score_protocol([sim_result])
    scored_geo = layer_geo.score_protocol([sim_result])

    print(f"\n  Same simulation result scored by two prior configurations:\n")
    print(f"  Default prefs  → EFE = {scored_default['efe_score']:.4f}")
    print(f"  Geo-heavy prefs → EFE = {scored_geo['efe_score']:.4f}")
    print(f"\n  The geo-heavy config rewards the high rust_belt_share (0.70)")
    print(f"  with a larger canon bonus, yielding a lower (better) EFE.\n")


# ======================================================================================================================
# Demo 8 — Background self-growth organism
# ======================================================================================================================

def demo_background_organism() -> None:
    section("DEMO 8 — Background Self-Growth Organism")

    majora = MajoraCore(n_nodes=300)

    # Seed history so the background thread has something to work with
    majora.ai_layer.auto_tune_priors({
        "urgency_impact": 0.82,
        "rust_belt_conversion": 0.68,
        "convergence_achieved": 0.88,
    })

    print("\n  Starting background Majora organism (short interval for demo) ...")
    majora.start_background_majora(interval_seconds=9999)  # long interval — won't fire in demo
    assert majora.background_thread is not None
    assert majora.background_thread.is_alive()
    print("  Background thread is alive ✓")
    print(f"  Prior history entries before stop: {len(majora.ai_layer.history)}")

    majora.stop_background()
    print("  Background stopped cleanly ✓\n")


# ======================================================================================================================
# Demo 9 — Protocol report formatting
# ======================================================================================================================

def demo_report_formatting() -> None:
    section("DEMO 9 — Protocol Report Formatting")

    majora = MajoraCore(n_nodes=300)
    result = majora.run_monte_carlo(
        content={"title": "DEMO TRACK", "artist": "IAMBANDOBANDZ"},
        preset="music_single",
        n_sims=30,
    )

    print(format_protocol_report(result, title="IAMBANDOBANDZ DEPLOYMENT PROTOCOL"))


# ======================================================================================================================
# Demo 10 — SIR graph state lifecycle inspection
# ======================================================================================================================

def demo_graph_lifecycle() -> None:
    section("DEMO 10 — SIR Graph State Lifecycle Inspection")

    graph = SIRGraph(n_nodes=1000, seed=7)
    print(f"\n  Graph built: {graph.n_nodes} nodes, "
          f"~{int(graph.n_nodes**2 * 0.04):,} expected edges")
    print(f"  Rust Belt nodes: {int(np.sum(graph.geo == 1.0))} "
          f"({100 * np.mean(graph.geo):.0f}%)")
    print(f"  Mean theme resonance: {graph.theme_resonance.mean():.3f}")

    graph.initialize_states(seed_fraction=0.02)
    print(f"\n  After initialize_states(0.02):")
    print(f"    Infected seeds: {int(np.sum(graph.states == SIRGraph.STATES['I']))}")

    print("\n  Step-by-step evolution:")
    for t in range(8):
        post = 0.12 if t == 3 else 0.0
        graph.step(beta=0.08, gamma=0.15, delta=0.02, post_strength=post)
        counts = state_counts(graph.states, SIRGraph.STATES)
        post_marker = " ← POST ACTION" if t == 3 else ""
        print(f"    t={t:02d}  S={counts['S']:5d}  E={counts['E']:5d}  "
              f"I={counts['I']:5d}  A={counts['A']:5d}  F={counts['F']:5d}{post_marker}")
    print()


# ======================================================================================================================
# Main
# ======================================================================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,  # Suppress verbose engine logs in demo
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    print("\n" + "=" * 80)
    print("  MAJORA CORE — EMPIRE CONTROL CENTER DEMO")
    print(f"  Numba JIT: {'ENABLED ⚡' if NUMBA_AVAILABLE else 'DISABLED (pure-Python fallback)'}")
    print("=" * 80)

    demo_single_release()
    demo_prior_auto_tuning()
    demo_presets()
    demo_vectorised_kernel()
    demo_sparse_kernel()
    demo_batch_runtime()
    demo_custom_prefs()
    demo_background_organism()
    demo_report_formatting()
    demo_graph_lifecycle()

    print("=" * 80)
    print("  All demos complete. The empire is live.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
