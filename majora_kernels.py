# ======================================================================================================================
# FILE: majora_kernels.py
# UUID: b9a14c3f-2d87-4f5e-8c01-e3d72f8095ab
# VERSION: v1.0.0-MAJORA-KERNELS-GODCORE
# COMPAT: Victor>=v0.1.0 | SAVE3>=v0.1.0
# NAME: Majora Kernels & Runtimes
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Alternative SIR kernel implementations and batch/parallel runtimes for the Majora engine.
#          Provides sparse-matrix, vectorised-NumPy, and batch-parallel variants so Majora can
#          scale from laptop to large-rig without changing calling code.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# DATE: 04/13/2026
# PROJECT NAME: BORON
# ======================================================================================================================

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

from majora_core import (
    NUMBA_AVAILABLE,
    ActiveInferenceLayer,
    CanonPreferences,
    MajoraCore,
    SIRGraph,
    sir_step_numba,
)

__all__ = [
    # Kernel functions
    "sir_step_vectorised",
    "sir_step_sparse",
    # Graph classes
    "VectorisedSIRGraph",
    "SparseGraph",
    # Runtimes
    "BatchMajoraRuntime",
    # Preset registry
    "PRESET_CONFIGS",
    # Utilities
    "format_protocol_report",
]

logger = logging.getLogger("Majora.Kernels")

# ======================================================================================================================
# Preset Registry
# ======================================================================================================================

PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "music_single": {
        "beta": 0.08,
        "gamma": 0.15,
        "delta": 0.02,
        "n_steps": 12,
        "post_step": 3,
        "post_strength": 0.12,
        "seed_fraction": 0.02,
        "description": (
            "Single track release — calibrated to 2026 Spotify first-48 h "
            "external-traffic signal with TikTok FYP hook exposure."
        ),
    },
    "repo_release": {
        "beta": 0.06,
        "gamma": 0.12,
        "delta": 0.015,
        "n_steps": 14,
        "post_step": 2,
        "post_strength": 0.09,
        "seed_fraction": 0.015,
        "description": (
            "Code / project repository release — slower transmission, "
            "lower fade rate (technical audiences retain longer)."
        ),
    },
    "album": {
        "beta": 0.10,
        "gamma": 0.18,
        "delta": 0.025,
        "n_steps": 16,
        "post_step": 4,
        "post_strength": 0.15,
        "seed_fraction": 0.03,
        "description": (
            "Full album drop — higher initial seeding, stronger barrage window, "
            "longer simulation horizon for catalog-level virality."
        ),
    },
    "general": {
        "beta": 0.07,
        "gamma": 0.13,
        "delta": 0.018,
        "n_steps": 12,
        "post_step": 3,
        "post_strength": 0.10,
        "seed_fraction": 0.02,
        "description": "General-purpose content release.",
    },
}


# ======================================================================================================================
# Vectorised NumPy Kernel
# ======================================================================================================================

def sir_step_vectorised(
    states: np.ndarray,
    adj_matrix: np.ndarray,
    beta: float,
    gamma: float,
    delta: float,
    theme_resonance: np.ndarray,
    geo_boost: np.ndarray,
    post_strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Fully vectorised NumPy SIR kernel — no Python loops, no Numba dependency.

    Equivalent to ``sir_step_numba`` but implemented with pure NumPy broadcasting.
    Suitable for medium-scale graphs (up to ~10k nodes) where Numba is unavailable
    or where per-step overhead of JIT compilation is unacceptable.

    State encoding: S=0, E=1, I=2 (Infected/streamer), A=3 (Amplifier), F=4 (Faded).

    Args:
        states: Current per-node state array (int8), shape (n_nodes,).
        adj_matrix: Dense adjacency matrix (float64), shape (n_nodes, n_nodes).
        beta: Transmission rate S→E per edge per step.
        gamma: E→I conversion rate.
        delta: Fade rate I/A→F.
        theme_resonance: Per-node theme affinity weights, shape (n_nodes,).
        geo_boost: Per-node geographic transmission multiplier, shape (n_nodes,).
        post_strength: Fraction of S nodes directly exposed by a platform post action.
        rng: NumPy Generator instance for reproducibility.

    Returns:
        Updated state array (copy, same dtype as input).
    """
    n = len(states)
    next_states = states.copy()

    # --- Post-action direct exposure (teaser / caption drop) ---
    if post_strength > 0.0:
        s_mask = states == 0
        candidates = np.where(s_mask)[0]
        if len(candidates) > 0:
            expose_count = min(int(n * post_strength), len(candidates))
            chosen = rng.choice(candidates, size=expose_count, replace=False)
            next_states[chosen] = 1  # S → E

    # --- I nodes expose S neighbours (vectorised) ---
    # infected_rows: boolean mask of I nodes
    infected = (states == 2)
    if infected.any():
        # Compute effective transmission weight for each (I, S) pair
        # w_matrix[i, j] = adj[i, j] * (1 + resonance[j] * 0.3) * geo[j]
        w_matrix = (
            adj_matrix
            * (1.0 + theme_resonance[np.newaxis, :] * 0.3)
            * geo_boost[np.newaxis, :]
        )
        # Mask: only I→S edges matter
        s_mask_2d = (states == 0)[np.newaxis, :]  # (1, n)
        i_mask_2d = infected[:, np.newaxis]         # (n, 1)
        effective = w_matrix * i_mask_2d * s_mask_2d

        # Draw random values and apply beta threshold
        rand_draw = rng.random(effective.shape)
        exposure_mask = (rand_draw < beta * effective).any(axis=0)  # (n,) — any I exposed this S
        next_states[exposure_mask & (states == 0)] = 1  # S → E

    # --- E → I ---
    e_mask = states == 1
    if e_mask.any():
        draws = rng.random(n)
        next_states[e_mask & (draws < gamma)] = 2

    # --- I → A ---
    i_mask = states == 2
    if i_mask.any():
        draws = rng.random(n)
        next_states[i_mask & (draws < 0.1)] = 3

    # --- I/A → F ---
    ia_mask = (states == 2) | (states == 3)
    if ia_mask.any():
        draws = rng.random(n)
        next_states[ia_mask & (draws < delta)] = 4

    return next_states


# ======================================================================================================================
# Vectorised SIR Graph
# ======================================================================================================================

class VectorisedSIRGraph:
    """
    SIR graph variant that exclusively uses the vectorised NumPy kernel.

    Identical graph construction to ``SIRGraph`` but exposes a ``step`` method
    backed by ``sir_step_vectorised`` rather than ``sir_step_numba``.  Use this
    class when Numba is unavailable or when you require reproducible per-step
    random streams via a seeded ``np.random.Generator``.

    Attributes:
        STATES: Mapping of state name to integer code (same as SIRGraph).
        n_nodes: Number of listener nodes.
        base_seed: Master RNG seed.
        lorain_boost: Rust Belt transmission multiplier.
        adj_matrix: Dense float64 adjacency matrix.
        geo: Float64 array (1.0 = Rust Belt, 0.0 = global).
        theme_resonance: Per-node Beta(2,5) theme affinity.
        geo_boost: Per-node transmission multiplier.
        states: Current per-node state array; None until initialised.
        rng: Seeded ``np.random.Generator`` instance.
    """

    STATES: Dict[str, int] = {"S": 0, "E": 1, "I": 2, "A": 3, "F": 4}

    def __init__(
        self,
        n_nodes: int = 5000,
        seed: int = 42,
        lorain_boost: float = 1.3,
    ) -> None:
        self.n_nodes = n_nodes
        self.base_seed = seed
        self.lorain_boost = lorain_boost
        self.states: Optional[np.ndarray] = None
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self._logger = logging.getLogger("Majora.VectorisedSIRGraph")
        self._build_graph()

    def _build_graph(self) -> None:
        rng = np.random.default_rng(self.base_seed)
        self.adj_matrix = (rng.random((self.n_nodes, self.n_nodes)) > 0.96).astype(np.float64)
        self.geo = np.array(
            [1.0 if i < int(0.15 * self.n_nodes) else 0.0 for i in range(self.n_nodes)],
            dtype=np.float64,
        )
        self.theme_resonance = rng.beta(2, 5, self.n_nodes).astype(np.float64)
        self.geo_boost = np.where(self.geo == 1.0, self.lorain_boost, 1.0).astype(np.float64)
        self._logger.info(
            f"VectorisedSIRGraph built ({self.n_nodes} nodes) — "
            "pure-NumPy vectorised kernel active"
        )

    def initialize_states(self, seed_fraction: float = 0.02, timeline_seed: Optional[int] = None) -> None:
        """Reset all nodes to S and seed a fraction as I (initial supporters)."""
        seed = timeline_seed if timeline_seed is not None else self.base_seed
        rng = np.random.default_rng(seed)
        self.states = np.zeros(self.n_nodes, dtype=np.int8)
        seed_count = int(self.n_nodes * seed_fraction)
        seed_nodes = rng.choice(self.n_nodes, seed_count, replace=False)
        self.states[seed_nodes] = self.STATES["I"]

    def step(
        self,
        beta: float = 0.08,
        gamma: float = 0.15,
        delta: float = 0.02,
        post_strength: float = 0.0,
    ) -> np.ndarray:
        """Advance the graph by one step using the vectorised NumPy kernel."""
        if self.states is None:
            self.initialize_states()
        self.states = sir_step_vectorised(
            self.states,
            self.adj_matrix,
            beta,
            gamma,
            delta,
            self.theme_resonance,
            self.geo_boost,
            post_strength,
            self.rng,
        )
        return self.states


# ======================================================================================================================
# Sparse Matrix Kernel
# ======================================================================================================================

def sir_step_sparse(
    states: np.ndarray,
    adj_csr: sp.csr_matrix,
    beta: float,
    gamma: float,
    delta: float,
    theme_resonance: np.ndarray,
    geo_boost: np.ndarray,
    post_strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sparse-matrix SIR kernel for large graphs (50k–500k nodes).

    Uses ``scipy.sparse`` CSR matrix operations to avoid materialising the
    full dense adjacency matrix, reducing memory from O(n²) to O(edges).
    At 4% edge density on 100k nodes the dense matrix is 80 GB; the sparse
    CSR representation is ~160 MB.

    Args:
        states: Current per-node state array (int8).
        adj_csr: Sparse CSR adjacency matrix (float64).
        beta: Transmission rate S→E.
        gamma: E→I rate.
        delta: Fade rate I/A→F.
        theme_resonance: Per-node theme affinity (float64).
        geo_boost: Per-node transmission multiplier (float64).
        post_strength: Fraction of S nodes exposed by post action.
        rng: Seeded NumPy Generator.

    Returns:
        Updated state array (copy).
    """
    n = len(states)
    next_states = states.copy()

    # --- Direct post-action exposure ---
    if post_strength > 0.0:
        s_idx = np.where(states == 0)[0]
        if len(s_idx) > 0:
            k = min(int(n * post_strength), len(s_idx))
            chosen = rng.choice(s_idx, k, replace=False)
            next_states[chosen] = 1

    # --- I→S exposure via sparse multiply ---
    infected = (states == 2).astype(np.float64)  # (n,)
    if infected.any():
        # For each S node j: probability of exposure = beta * sum_i (adj[i,j] * infected[i] * w[j])
        # w[j] = (1 + theme_resonance[j] * 0.3) * geo_boost[j]
        w = (1.0 + theme_resonance * 0.3) * geo_boost  # (n,)
        # adj_csr.T @ infected gives sum of (adj[i,j] * infected[i]) for each j
        exposure_pressure = np.asarray(adj_csr.T.dot(infected)).ravel() * w  # (n,)
        s_mask = states == 0
        prob = np.clip(beta * exposure_pressure, 0.0, 1.0)
        draws = rng.random(n)
        next_states[s_mask & (draws < prob)] = 1  # S → E

    # --- E → I ---
    e_mask = states == 1
    if e_mask.any():
        draws = rng.random(n)
        next_states[e_mask & (draws < gamma)] = 2

    # --- I → A ---
    i_mask = states == 2
    if i_mask.any():
        draws = rng.random(n)
        next_states[i_mask & (draws < 0.1)] = 3

    # --- I/A → F ---
    ia_mask = (states == 2) | (states == 3)
    if ia_mask.any():
        draws = rng.random(n)
        next_states[ia_mask & (draws < delta)] = 4

    return next_states


# ======================================================================================================================
# Sparse Graph
# ======================================================================================================================

class SparseGraph:
    """
    SIR graph backed by a sparse CSR adjacency matrix.

    Suitable for large simulations (>50k nodes) where the dense adjacency matrix
    does not fit in memory.  Uses ``scipy.sparse`` and delegates to
    ``sir_step_sparse`` for state transitions.

    Attributes:
        STATES: Mapping of state name to integer code.
        n_nodes: Number of simulated listener nodes.
        base_seed: Master random seed.
        lorain_boost: Rust Belt transmission multiplier.
        adj_csr: Sparse CSR adjacency matrix.
        geo: Float64 array (1.0 = Rust Belt, 0.0 = global).
        theme_resonance: Per-node Beta(2,5) theme affinity.
        geo_boost: Per-node transmission multiplier.
        states: Current per-node state array; None until initialised.
        rng: Seeded NumPy Generator.
    """

    STATES: Dict[str, int] = {"S": 0, "E": 1, "I": 2, "A": 3, "F": 4}

    def __init__(
        self,
        n_nodes: int = 50_000,
        seed: int = 42,
        lorain_boost: float = 1.3,
        edge_density: float = 0.04,
    ) -> None:
        self.n_nodes = n_nodes
        self.base_seed = seed
        self.lorain_boost = lorain_boost
        self.edge_density = edge_density
        self.states: Optional[np.ndarray] = None
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self._logger = logging.getLogger("Majora.SparseGraph")
        self._build_graph()

    def _build_graph(self) -> None:
        rng = np.random.default_rng(self.base_seed)
        n = self.n_nodes
        # Build sparse random graph: sample ~edge_density fraction of edges
        nnz = int(n * n * self.edge_density)
        rows = rng.integers(0, n, nnz)
        cols = rng.integers(0, n, nnz)
        data = np.ones(nnz, dtype=np.float64)
        self.adj_csr = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
        self.adj_csr.sum_duplicates()

        self.geo = np.array(
            [1.0 if i < int(0.15 * n) else 0.0 for i in range(n)],
            dtype=np.float64,
        )
        self.theme_resonance = rng.beta(2, 5, n).astype(np.float64)
        self.geo_boost = np.where(self.geo == 1.0, self.lorain_boost, 1.0).astype(np.float64)
        self._logger.info(
            f"SparseGraph built ({n:,} nodes, {self.adj_csr.nnz:,} edges, "
            f"{self.adj_csr.nnz / n**2 * 100:.1f}% density)"
        )

    def initialize_states(self, seed_fraction: float = 0.02, timeline_seed: Optional[int] = None) -> None:
        """Reset all nodes to S and seed a fraction as I."""
        seed = timeline_seed if timeline_seed is not None else self.base_seed
        rng = np.random.default_rng(seed)
        self.states = np.zeros(self.n_nodes, dtype=np.int8)
        k = int(self.n_nodes * seed_fraction)
        seed_nodes = rng.choice(self.n_nodes, k, replace=False)
        self.states[seed_nodes] = self.STATES["I"]

    def step(
        self,
        beta: float = 0.08,
        gamma: float = 0.15,
        delta: float = 0.02,
        post_strength: float = 0.0,
    ) -> np.ndarray:
        """Advance the graph by one step using the sparse-matrix kernel."""
        if self.states is None:
            self.initialize_states()
        self.states = sir_step_sparse(
            self.states,
            self.adj_csr,
            beta,
            gamma,
            delta,
            self.theme_resonance,
            self.geo_boost,
            post_strength,
            self.rng,
        )
        return self.states


# ======================================================================================================================
# Batch Parallel Runtime
# ======================================================================================================================

@dataclass
class BatchRunConfig:
    """
    Configuration for a single content item in a batch run.

    Attributes:
        content: Track metadata dict or pre-computed hash string.
        preset: Release type preset.
        n_sims: Number of Monte Carlo simulations.
        beta: SIR transmission rate.
        gamma: E→I conversion rate.
        delta: Fade rate.
    """
    content: Union[str, Dict[str, Any]]
    preset: Literal["music_single", "repo_release", "album", "general"] = "music_single"
    n_sims: int = 200
    beta: float = 0.08
    gamma: float = 0.15
    delta: float = 0.02


class BatchMajoraRuntime:
    """
    Batch parallel runtime — run multiple independent Majora simulations concurrently.

    Each item in the batch gets its own ``MajoraCore`` instance and is dispatched
    to a thread-pool worker via ``concurrent.futures.ThreadPoolExecutor``.  Results
    are collected in order and returned as a list.

    This is the recommended entry point for:
    - Comparing release strategies for multiple tracks simultaneously.
    - A/B testing different SIR parameters for the same content.
    - Generating a full-catalog protocol report in one call.

    Attributes:
        n_nodes: Number of listener nodes per worker MajoraCore instance.
        max_workers: Thread-pool size (defaults to min(len(batch), CPU count)).
        cache_size: LRU cache size per MajoraCore instance.
        ai_layer_prefs: Optional shared CanonPreferences applied to all workers.
    """

    def __init__(
        self,
        n_nodes: int = 3000,
        max_workers: Optional[int] = None,
        cache_size: int = 32,
        ai_layer_prefs: Optional[CanonPreferences] = None,
    ) -> None:
        self.n_nodes = n_nodes
        self.max_workers = max_workers
        self.cache_size = cache_size
        self.ai_layer_prefs = ai_layer_prefs
        self._logger = logging.getLogger("Majora.BatchRuntime")

    def _run_one(self, config: BatchRunConfig) -> Dict[str, Any]:
        """Worker: create a MajoraCore, run simulation, return result."""
        prefs = self.ai_layer_prefs or CanonPreferences()
        core = MajoraCore(n_nodes=self.n_nodes, cache_size=self.cache_size)
        core.ai_layer.prefs = prefs
        return core.run_monte_carlo(
            content=config.content,
            preset=config.preset,
            n_sims=config.n_sims,
            beta=config.beta,
            gamma=config.gamma,
            delta=config.delta,
        )

    def run_batch(
        self,
        items: List[Union[BatchRunConfig, Dict[str, Any]]],
        preset: Literal["music_single", "repo_release", "album", "general"] = "music_single",
        n_sims: int = 200,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run a batch of Majora simulations in parallel.

        Items can be ``BatchRunConfig`` objects (full control) or plain dicts
        (treated as content metadata; preset/n_sims/beta/gamma/delta apply as
        defaults from this call's arguments).

        Args:
            items: List of ``BatchRunConfig`` or content metadata dicts.
            preset: Default preset for plain-dict items.
            n_sims: Default simulation count for plain-dict items.
            beta: Default beta (uses preset default if None).
            gamma: Default gamma (uses preset default if None).
            delta: Default delta (uses preset default if None).

        Returns:
            List of protocol result dicts in the same order as ``items``.
        """
        pconf = PRESET_CONFIGS.get(preset, PRESET_CONFIGS["general"])
        _beta = beta if beta is not None else pconf["beta"]
        _gamma = gamma if gamma is not None else pconf["gamma"]
        _delta = delta if delta is not None else pconf["delta"]

        configs: List[BatchRunConfig] = []
        for item in items:
            if isinstance(item, BatchRunConfig):
                configs.append(item)
            else:
                configs.append(BatchRunConfig(
                    content=item,
                    preset=preset,
                    n_sims=n_sims,
                    beta=_beta,
                    gamma=_gamma,
                    delta=_delta,
                ))

        workers = self.max_workers or min(len(configs), 8)
        self._logger.info(
            f"BatchMajoraRuntime: {len(configs)} items | "
            f"{workers} workers | preset={preset!r}"
        )

        results: List[Dict[str, Any]] = [{}] * len(configs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(self._run_one, cfg): idx
                for idx, cfg in enumerate(configs)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(f"Batch item {idx} failed: {exc}")
                    results[idx] = {"error": str(exc), "index": idx}

        return results


# ======================================================================================================================
# Protocol Report Formatter
# ======================================================================================================================

def format_protocol_report(
    result: Dict[str, Any],
    title: str = "MAJORA DEPLOYMENT PROTOCOL",
    width: int = 80,
) -> str:
    """
    Render a ``MajoraCore.run_monte_carlo`` result as a human-readable deployment report.

    Args:
        result: Protocol dict returned by ``MajoraCore.run_monte_carlo``.
        title: Banner title string.
        width: Column width for the separator line.

    Returns:
        Formatted multi-line string suitable for printing or logging.
    """
    sep = "=" * width
    lines = [
        "",
        sep,
        f"  {title}",
        sep,
        f"  Content hash      : {result.get('content_hash', 'N/A')}",
        f"  Preset            : {result.get('preset', 'N/A')}",
        f"  Winning drop time : {result.get('winning_drop_time', 'N/A')}",
        f"  EFE score         : {result.get('efe_score', float('nan')):.4f} "
        f"± {result.get('efe_std', float('nan')):.4f}  (lower = better)",
        f"  Numba accelerated : {result.get('numba_accelerated', False)}",
        f"  Simulations run   : {result.get('n_sims_run', 0):,}",
        "",
        "  Canon-aligned metrics:",
    ]
    for k, v in (result.get("canon_aligned_metrics") or {}).items():
        if v is not None:
            lines.append(f"    • {k:<28}: {v:.4f}")
    lines += [
        "",
        "  Recommended protocol:",
    ]
    for chunk in _wrap(result.get("recommended_protocol", ""), width - 4):
        lines.append(f"    {chunk}")
    lines += [sep, ""]
    return "\n".join(lines)


def _wrap(text: str, width: int) -> List[str]:
    """Simple word-wrap helper."""
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    length = 0
    for w in words:
        if length + len(w) + (1 if current else 0) > width:
            lines.append(" ".join(current))
            current = [w]
            length = len(w)
        else:
            current.append(w)
            length += len(w) + (1 if len(current) > 1 else 0)
    if current:
        lines.append(" ".join(current))
    return lines


# ======================================================================================================================
# Standalone entry point
# ======================================================================================================================

if __name__ == "__main__":
    import pprint

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    print("\n--- PRESET REGISTRY ---")
    for name, cfg in PRESET_CONFIGS.items():
        print(f"  {name}: {cfg['description']}")

    print("\n--- VECTORISED GRAPH DEMO (n=500) ---")
    vgraph = VectorisedSIRGraph(n_nodes=500, seed=7)
    vgraph.initialize_states()
    for _ in range(5):
        vgraph.step(beta=0.08, gamma=0.15, delta=0.02, post_strength=0.1)
    counts = {k: int(np.sum(vgraph.states == v)) for k, v in VectorisedSIRGraph.STATES.items()}
    print(f"  State counts after 5 steps: {counts}")

    print("\n--- SPARSE GRAPH DEMO (n=2000) ---")
    sgraph = SparseGraph(n_nodes=2000, seed=7)
    sgraph.initialize_states()
    for _ in range(5):
        sgraph.step(beta=0.08, gamma=0.15, delta=0.02, post_strength=0.1)
    counts_s = {k: int(np.sum(sgraph.states == v)) for k, v in SparseGraph.STATES.items()}
    print(f"  State counts after 5 steps: {counts_s}")

    print("\n--- BATCH RUNTIME DEMO (3 items, 2 workers) ---")
    runtime = BatchMajoraRuntime(n_nodes=300, max_workers=2)
    batch_results = runtime.run_batch(
        items=[
            {"title": "WE ALL DIE ONE DAY", "artist": "IAMBANDOBANDZ"},
            {"title": "BOUNDEDNESS v1.0", "type": "repo_release"},
            {"title": "EMPIRE VOLUME 1", "type": "album"},
        ],
        preset="music_single",
        n_sims=30,
    )
    for i, r in enumerate(batch_results):
        print(f"  [{i}] hash={r.get('content_hash')} | EFE={r.get('efe_score', 'ERR'):.3f}")

    print("\n--- FORMAT REPORT DEMO ---")
    single = MajoraCore(n_nodes=300)
    result = single.run_monte_carlo("demo_content", n_sims=20)
    print(format_protocol_report(result, title="DEMO PROTOCOL REPORT"))
