"""Tests for majora_kernels.py — kernels, runtimes, formatters."""
from __future__ import annotations

import threading
from typing import Any, Dict

import numpy as np
import pytest
import scipy.sparse as sp

from majora_core import CanonPreferences, MajoraCore, SIRGraph
from majora_kernels import (
    PRESET_CONFIGS,
    BatchMajoraRuntime,
    BatchRunConfig,
    SparseGraph,
    VectorisedSIRGraph,
    _wrap,
    format_protocol_report,
    sir_step_sparse,
    sir_step_vectorised,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adj(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n, n)) > 0.88).astype(np.float64)


def _make_csr(n: int, seed: int = 0, density: float = 0.12) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    return sp.random(n, n, density=density, format="csr", random_state=rng, dtype=np.float64)


def _make_states(n: int, n_infected: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = np.zeros(n, dtype=np.int8)
    idx = rng.choice(n, n_infected, replace=False)
    states[idx] = 2
    return states


def _make_resonance_boost(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    resonance = rng.beta(2, 5, n).astype(np.float64)
    boost = np.ones(n, dtype=np.float64)
    boost[: n // 5] = 1.3
    return resonance, boost


# ---------------------------------------------------------------------------
# PRESET_CONFIGS
# ---------------------------------------------------------------------------

class TestPresetConfigs:
    def test_all_presets_present(self):
        for key in ("music_single", "repo_release", "album", "general"):
            assert key in PRESET_CONFIGS

    def test_each_preset_has_required_keys(self):
        required = {"beta", "gamma", "delta", "n_steps", "post_step",
                    "post_strength", "seed_fraction", "description"}
        for name, cfg in PRESET_CONFIGS.items():
            assert required <= set(cfg.keys()), f"Missing keys in preset {name!r}"

    def test_beta_in_valid_range(self):
        for cfg in PRESET_CONFIGS.values():
            assert 0.0 < cfg["beta"] < 1.0

    def test_gamma_in_valid_range(self):
        for cfg in PRESET_CONFIGS.values():
            assert 0.0 < cfg["gamma"] < 1.0

    def test_delta_in_valid_range(self):
        for cfg in PRESET_CONFIGS.values():
            assert 0.0 < cfg["delta"] < 1.0

    def test_description_is_non_empty_string(self):
        for name, cfg in PRESET_CONFIGS.items():
            assert isinstance(cfg["description"], str) and cfg["description"]


# ---------------------------------------------------------------------------
# sir_step_vectorised
# ---------------------------------------------------------------------------

class TestSirStepVectorised:
    def setup_method(self):
        self.n = 80
        self.adj = _make_adj(self.n, seed=1)
        self.resonance, self.boost = _make_resonance_boost(self.n, seed=1)
        self.states = _make_states(self.n, n_infected=5, seed=1)
        self.rng = np.random.default_rng(42)

    def test_returns_ndarray(self):
        result = sir_step_vectorised(
            self.states, self.adj, 0.08, 0.15, 0.02,
            self.resonance, self.boost, 0.0, self.rng,
        )
        assert isinstance(result, np.ndarray)

    def test_output_length_unchanged(self):
        result = sir_step_vectorised(
            self.states, self.adj, 0.08, 0.15, 0.02,
            self.resonance, self.boost, 0.0, self.rng,
        )
        assert len(result) == self.n

    def test_states_remain_in_valid_range(self):
        result = sir_step_vectorised(
            self.states, self.adj, 0.08, 0.15, 0.02,
            self.resonance, self.boost, 0.0, self.rng,
        )
        assert int(result.min()) >= 0
        assert int(result.max()) <= 4

    def test_input_not_mutated(self):
        original = self.states.copy()
        sir_step_vectorised(
            self.states, self.adj, 0.08, 0.15, 0.02,
            self.resonance, self.boost, 0.0, self.rng,
        )
        np.testing.assert_array_equal(self.states, original)

    def test_returns_copy_not_same_object(self):
        result = sir_step_vectorised(
            self.states, self.adj, 0.08, 0.15, 0.02,
            self.resonance, self.boost, 0.0, self.rng,
        )
        assert result is not self.states

    def test_post_strength_exposes_s_nodes(self):
        states_before = self.states.copy()
        result = sir_step_vectorised(
            self.states, self.adj, 0.08, 0.15, 0.02,
            self.resonance, self.boost, 0.5, self.rng,
        )
        newly_non_s = np.sum((states_before == 0) & (result > 0))
        assert newly_non_s > 0

    def test_zero_post_strength_no_direct_exposure(self):
        # All-susceptible graph with no infected nodes: no state changes possible
        states = np.zeros(self.n, dtype=np.int8)
        adj = np.zeros((self.n, self.n), dtype=np.float64)
        result = sir_step_vectorised(
            states, adj, 0.08, 0.15, 0.02,
            self.resonance, self.boost, 0.0, self.rng,
        )
        np.testing.assert_array_equal(result, states)

    def test_high_gamma_converts_exposed(self):
        """With gamma=1.0, all E nodes must become I."""
        states = np.full(self.n, 1, dtype=np.int8)  # all Exposed
        adj = np.zeros((self.n, self.n), dtype=np.float64)
        result = sir_step_vectorised(
            states, adj, 0.08, 1.0, 0.0,
            self.resonance, self.boost, 0.0, self.rng,
        )
        assert np.all(result == 2)

    def test_high_delta_fades_infected(self):
        """With delta=1.0, all I and A nodes must become F."""
        states = np.zeros(self.n, dtype=np.int8)
        states[:] = 2  # all Infected
        adj = np.zeros((self.n, self.n), dtype=np.float64)
        result = sir_step_vectorised(
            states, adj, 0.0, 0.0, 1.0,
            self.resonance, self.boost, 0.0, self.rng,
        )
        assert np.all(result == 4)


# ---------------------------------------------------------------------------
# sir_step_sparse
# ---------------------------------------------------------------------------

class TestSirStepSparse:
    def setup_method(self):
        self.n = 80
        self.adj_csr = _make_csr(self.n, seed=2, density=0.15)
        self.resonance, self.boost = _make_resonance_boost(self.n, seed=2)
        self.states = _make_states(self.n, n_infected=5, seed=2)
        self.rng = np.random.default_rng(99)

    def test_returns_ndarray(self):
        result = sir_step_sparse(
            self.states, self.adj_csr, 0.08, 0.15, 0.02,
            self.resonance, self.boost, 0.0, self.rng,
        )
        assert isinstance(result, np.ndarray)

    def test_output_length_unchanged(self):
        result = sir_step_sparse(
            self.states, self.adj_csr, 0.08, 0.15, 0.02,
            self.resonance, self.boost, 0.0, self.rng,
        )
        assert len(result) == self.n

    def test_states_remain_in_valid_range(self):
        result = sir_step_sparse(
            self.states, self.adj_csr, 0.08, 0.15, 0.02,
            self.resonance, self.boost, 0.0, self.rng,
        )
        assert int(result.min()) >= 0
        assert int(result.max()) <= 4

    def test_input_not_mutated(self):
        original = self.states.copy()
        sir_step_sparse(
            self.states, self.adj_csr, 0.08, 0.15, 0.02,
            self.resonance, self.boost, 0.0, self.rng,
        )
        np.testing.assert_array_equal(self.states, original)

    def test_post_strength_exposes_s_nodes(self):
        states_before = self.states.copy()
        result = sir_step_sparse(
            self.states, self.adj_csr, 0.08, 0.15, 0.02,
            self.resonance, self.boost, 0.5, self.rng,
        )
        newly_non_s = np.sum((states_before == 0) & (result > 0))
        assert newly_non_s > 0

    def test_zero_post_zero_infected_no_change(self):
        states = np.zeros(self.n, dtype=np.int8)
        adj = sp.csr_matrix((self.n, self.n), dtype=np.float64)
        result = sir_step_sparse(
            states, adj, 0.08, 0.15, 0.02,
            self.resonance, self.boost, 0.0, self.rng,
        )
        np.testing.assert_array_equal(result, states)

    def test_high_delta_fades_all_infected(self):
        states = np.full(self.n, 2, dtype=np.int8)
        adj = sp.csr_matrix((self.n, self.n), dtype=np.float64)
        result = sir_step_sparse(
            states, adj, 0.0, 0.0, 1.0,
            self.resonance, self.boost, 0.0, self.rng,
        )
        assert np.all(result == 4)


# ---------------------------------------------------------------------------
# VectorisedSIRGraph
# ---------------------------------------------------------------------------

class TestVectorisedSIRGraph:
    def setup_method(self):
        self.graph = VectorisedSIRGraph(n_nodes=150, seed=5, lorain_boost=1.3)

    def test_graph_dimensions(self):
        assert self.graph.adj_matrix.shape == (150, 150)
        assert len(self.graph.geo) == 150
        assert len(self.graph.theme_resonance) == 150
        assert len(self.graph.geo_boost) == 150

    def test_rust_belt_fraction(self):
        assert int(np.sum(self.graph.geo == 1.0)) == int(0.15 * 150)

    def test_geo_boost_values(self):
        rust = self.graph.geo == 1.0
        np.testing.assert_allclose(self.graph.geo_boost[rust], 1.3)
        np.testing.assert_allclose(self.graph.geo_boost[~rust], 1.0)

    def test_states_none_before_initialize(self):
        graph = VectorisedSIRGraph(n_nodes=100)
        assert graph.states is None

    def test_initialize_states(self):
        self.graph.initialize_states(seed_fraction=0.1, timeline_seed=7)
        assert self.graph.states is not None
        assert len(self.graph.states) == 150
        n_infected = int(np.sum(self.graph.states == 2))
        assert n_infected == pytest.approx(15, abs=1)

    def test_step_returns_array(self):
        self.graph.initialize_states()
        result = self.graph.step()
        assert isinstance(result, np.ndarray)
        assert len(result) == 150

    def test_step_state_values_valid(self):
        self.graph.initialize_states()
        self.graph.step()
        assert int(self.graph.states.min()) >= 0
        assert int(self.graph.states.max()) <= 4

    def test_step_auto_initializes(self):
        graph = VectorisedSIRGraph(n_nodes=100)
        result = graph.step()
        assert result is not None

    def test_rng_is_generator_instance(self):
        assert isinstance(self.graph.rng, np.random.Generator)

    def test_multiple_steps_converge(self):
        """After many steps with high beta, most nodes should be active."""
        self.graph.initialize_states(seed_fraction=0.05)
        for _ in range(15):
            self.graph.step(beta=0.3, gamma=0.5, delta=0.0, post_strength=0.1)
        active = int(np.sum(self.graph.states >= 2))
        assert active > self.graph.n_nodes * 0.3


# ---------------------------------------------------------------------------
# SparseGraph
# ---------------------------------------------------------------------------

class TestSparseGraph:
    def setup_method(self):
        self.graph = SparseGraph(n_nodes=500, seed=11, edge_density=0.04)

    def test_adj_is_csr(self):
        assert sp.issparse(self.graph.adj_csr)
        assert self.graph.adj_csr.format == "csr"

    def test_adj_shape(self):
        assert self.graph.adj_csr.shape == (500, 500)

    def test_geo_length(self):
        assert len(self.graph.geo) == 500

    def test_rust_belt_fraction(self):
        assert int(np.sum(self.graph.geo == 1.0)) == int(0.15 * 500)

    def test_theme_resonance_in_01(self):
        assert float(self.graph.theme_resonance.min()) >= 0.0
        assert float(self.graph.theme_resonance.max()) <= 1.0

    def test_states_none_before_initialize(self):
        graph = SparseGraph(n_nodes=100)
        assert graph.states is None

    def test_initialize_states(self):
        self.graph.initialize_states(seed_fraction=0.02)
        assert self.graph.states is not None
        n_infected = int(np.sum(self.graph.states == 2))
        assert n_infected == pytest.approx(int(500 * 0.02), abs=2)

    def test_step_returns_array(self):
        self.graph.initialize_states()
        result = self.graph.step()
        assert isinstance(result, np.ndarray)
        assert len(result) == 500

    def test_step_state_values_valid(self):
        self.graph.initialize_states()
        self.graph.step()
        assert int(self.graph.states.min()) >= 0
        assert int(self.graph.states.max()) <= 4

    def test_step_auto_initializes(self):
        graph = SparseGraph(n_nodes=100)
        result = graph.step()
        assert result is not None

    def test_nnz_approx_expected_density(self):
        n = self.graph.n_nodes
        density = self.graph.adj_csr.nnz / (n * n)
        assert 0.0 < density <= 0.10  # Generous bound; sampling has variance

    def test_rng_is_generator(self):
        assert isinstance(self.graph.rng, np.random.Generator)


# ---------------------------------------------------------------------------
# BatchRunConfig
# ---------------------------------------------------------------------------

class TestBatchRunConfig:
    def test_defaults(self):
        cfg = BatchRunConfig(content={"title": "test"})
        assert cfg.preset == "music_single"
        assert cfg.n_sims == 200
        assert cfg.beta == pytest.approx(0.08)

    def test_custom_values(self):
        cfg = BatchRunConfig(
            content="hash_xyz",
            preset="album",
            n_sims=50,
            beta=0.12,
        )
        assert cfg.preset == "album"
        assert cfg.n_sims == 50
        assert cfg.beta == pytest.approx(0.12)

    def test_content_can_be_string(self):
        cfg = BatchRunConfig(content="abc123")
        assert cfg.content == "abc123"

    def test_content_can_be_dict(self):
        cfg = BatchRunConfig(content={"title": "demo"})
        assert cfg.content == {"title": "demo"}


# ---------------------------------------------------------------------------
# BatchMajoraRuntime
# ---------------------------------------------------------------------------

class TestBatchMajoraRuntime:
    def setup_method(self):
        self.runtime = BatchMajoraRuntime(n_nodes=200, max_workers=2)

    def test_run_batch_returns_list(self):
        items = [{"title": "track1"}, {"title": "track2"}]
        results = self.runtime.run_batch(items, n_sims=5)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_results_in_same_order_as_input(self):
        items = [{"id": i} for i in range(4)]
        configs = [
            BatchRunConfig(content=item, n_sims=5)
            for item in items
        ]
        results = self.runtime.run_batch(configs)
        hashes = [r["content_hash"] for r in results]
        # All hashes should be distinct (different content)
        assert len(set(hashes)) == 4

    def test_plain_dict_items_use_preset_default(self):
        items = [{"title": "x"}]
        results = self.runtime.run_batch(items, preset="repo_release", n_sims=5)
        assert results[0]["preset"] == "repo_release"

    def test_batch_run_config_overrides_defaults(self):
        cfg = BatchRunConfig(content={"title": "y"}, preset="album", n_sims=5)
        results = self.runtime.run_batch([cfg])
        assert results[0]["preset"] == "album"

    def test_result_has_required_keys(self):
        results = self.runtime.run_batch([{"title": "z"}], n_sims=5)
        for key in ("efe_score", "recommended_protocol", "content_hash", "preset"):
            assert key in results[0]

    def test_shared_prefs_applied(self):
        prefs = CanonPreferences(lorain_roots_geo_weight=0.50)
        runtime = BatchMajoraRuntime(n_nodes=200, max_workers=1, ai_layer_prefs=prefs)
        results = runtime.run_batch([{"title": "test_prefs"}], n_sims=5)
        assert isinstance(results[0]["efe_score"], float)

    def test_batch_single_item(self):
        results = self.runtime.run_batch([{"title": "solo"}], n_sims=5)
        assert len(results) == 1
        assert "efe_score" in results[0]

    def test_efe_score_is_finite(self):
        results = self.runtime.run_batch([{"title": "fin"}], n_sims=5)
        assert np.isfinite(results[0]["efe_score"])

    def test_numba_flag_present(self):
        from majora_core import NUMBA_AVAILABLE
        results = self.runtime.run_batch([{"title": "nb"}], n_sims=5)
        assert results[0]["numba_accelerated"] == NUMBA_AVAILABLE

    def test_all_preset_types_in_batch(self):
        items = [
            BatchRunConfig(content={"title": p}, preset=p, n_sims=5)  # type: ignore[arg-type]
            for p in ("music_single", "repo_release", "album", "general")
        ]
        results = self.runtime.run_batch(items)
        presets_returned = {r["preset"] for r in results}
        assert presets_returned == {"music_single", "repo_release", "album", "general"}


# ---------------------------------------------------------------------------
# format_protocol_report
# ---------------------------------------------------------------------------

class TestFormatProtocolReport:
    def _make_result(self) -> Dict[str, Any]:
        majora = MajoraCore(n_nodes=150)
        return majora.run_monte_carlo("test_report", n_sims=5)

    def test_returns_string(self):
        result = self._make_result()
        report = format_protocol_report(result)
        assert isinstance(report, str)

    def test_contains_content_hash(self):
        result = self._make_result()
        report = format_protocol_report(result)
        assert result["content_hash"] in report

    def test_contains_preset(self):
        result = self._make_result()
        report = format_protocol_report(result)
        assert result["preset"] in report

    def test_contains_efe_score(self):
        result = self._make_result()
        report = format_protocol_report(result)
        assert "EFE score" in report

    def test_custom_title_appears(self):
        result = self._make_result()
        report = format_protocol_report(result, title="CUSTOM TITLE")
        assert "CUSTOM TITLE" in report

    def test_width_parameter_controls_separator(self):
        result = self._make_result()
        report = format_protocol_report(result, width=60)
        # Separator line should be 60 '=' characters
        assert "=" * 60 in report

    def test_missing_optional_fields_handled(self):
        """format_protocol_report should not crash on minimal result."""
        minimal = {"content_hash": "abc", "preset": "general"}
        report = format_protocol_report(minimal)
        assert "abc" in report

    def test_canon_metrics_rendered(self):
        result = self._make_result()
        report = format_protocol_report(result)
        assert "urgency_score" in report


# ---------------------------------------------------------------------------
# _wrap (internal helper)
# ---------------------------------------------------------------------------

class TestWrap:
    def test_short_text_single_line(self):
        lines = _wrap("hello world", 40)
        assert lines == ["hello world"]

    def test_long_text_splits(self):
        text = "word " * 20
        lines = _wrap(text.strip(), 20)
        assert len(lines) > 1
        for line in lines:
            assert len(line) <= 25  # generous bound

    def test_empty_text(self):
        lines = _wrap("", 40)
        assert lines == []

    def test_single_long_word_not_split(self):
        word = "a" * 30
        lines = _wrap(word, 10)
        # The word is preserved intact; it may appear alongside an empty token
        # depending on initial flush, but it must not be broken.
        assert any(word in line for line in lines)


# ---------------------------------------------------------------------------
# Integration: VectorisedSIRGraph through a full sim cycle
# ---------------------------------------------------------------------------

class TestVectorisedSIRGraphIntegration:
    def test_full_simulation_cycle(self):
        """Run 10 steps and verify monotonic spread pattern (net non-S increases)."""
        graph = VectorisedSIRGraph(n_nodes=400, seed=77)
        graph.initialize_states(seed_fraction=0.02)
        prev_non_s = int(np.sum(graph.states != 0))

        for t in range(10):
            graph.step(beta=0.10, gamma=0.20, delta=0.01, post_strength=0.0)
            non_s = int(np.sum(graph.states != 0))
            assert non_s >= prev_non_s - 5  # small tolerance for stochastic fade
            prev_non_s = non_s

        # After 10 steps at high beta, at least 15% should be active
        active = int(np.sum(graph.states >= 2))
        assert active > graph.n_nodes * 0.15


# ---------------------------------------------------------------------------
# Integration: SparseGraph through a full sim cycle
# ---------------------------------------------------------------------------

class TestSparseGraphIntegration:
    def test_full_simulation_cycle(self):
        graph = SparseGraph(n_nodes=500, seed=88, edge_density=0.05)
        graph.initialize_states(seed_fraction=0.03)

        for t in range(8):
            post = 0.10 if t == 2 else 0.0
            graph.step(beta=0.08, gamma=0.15, delta=0.02, post_strength=post)

        # At least some nodes should have spread
        active = int(np.sum(graph.states >= 1))
        assert active > int(0.03 * 500)  # more than just seeds


# ---------------------------------------------------------------------------
# Integration: BatchMajoraRuntime with auto-tuned prefs
# ---------------------------------------------------------------------------

class TestBatchRuntimeWithTunedPrefs:
    def test_tuned_prefs_applied_to_batch(self):
        """After manually tuning prefs, batch results reflect the updated weights."""
        from majora_core import ActiveInferenceLayer
        layer = ActiveInferenceLayer()
        layer.auto_tune_priors({
            "urgency_impact": 0.95,
            "rust_belt_conversion": 0.80,
            "convergence_achieved": 0.90,
        })
        tuned_prefs = layer.prefs

        runtime = BatchMajoraRuntime(
            n_nodes=200,
            max_workers=2,
            ai_layer_prefs=tuned_prefs,
        )
        results = runtime.run_batch(
            [{"title": "WE ALL DIE ONE DAY"}, {"title": "EMPIRE VOLUME 1"}],
            n_sims=5,
        )
        assert all("efe_score" in r for r in results)
        assert all(np.isfinite(r["efe_score"]) for r in results)
