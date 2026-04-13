"""Tests for majora_core.py"""
from __future__ import annotations

import time

import numpy as np
import pytest

from majora_core import (
    NUMBA_AVAILABLE,
    ActiveInferenceLayer,
    CanonPreferences,
    MajoraCore,
    SIRGraph,
    sir_step_numba,
)


# ---------------------------------------------------------------------------
# CanonPreferences
# ---------------------------------------------------------------------------

class TestCanonPreferences:
    def test_default_values(self):
        prefs = CanonPreferences()
        assert prefs.mortality_urgency_weight == pytest.approx(0.35)
        assert prefs.lorain_roots_geo_weight == pytest.approx(0.25)
        assert prefs.builder_proof_of_work_weight == pytest.approx(0.25)
        assert prefs.virality_index_weight == pytest.approx(0.15)
        assert prefs.adaptation_rate == pytest.approx(0.05)
        assert prefs.historical_convergence_score == pytest.approx(0.0)

    def test_weights_sum_to_one(self):
        prefs = CanonPreferences()
        total = (
            prefs.mortality_urgency_weight
            + prefs.lorain_roots_geo_weight
            + prefs.builder_proof_of_work_weight
            + prefs.virality_index_weight
        )
        assert total == pytest.approx(1.0)

    def test_to_dict_returns_all_fields(self):
        prefs = CanonPreferences()
        d = prefs.to_dict()
        for field in (
            "mortality_urgency_weight",
            "lorain_roots_geo_weight",
            "builder_proof_of_work_weight",
            "virality_index_weight",
            "adaptation_rate",
            "historical_convergence_score",
        ):
            assert field in d

    def test_custom_values_preserved(self):
        prefs = CanonPreferences(mortality_urgency_weight=0.5)
        assert prefs.mortality_urgency_weight == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ActiveInferenceLayer — compute_efe
# ---------------------------------------------------------------------------

class TestComputeEfe:
    def setup_method(self):
        self.layer = ActiveInferenceLayer()

    def _outcomes(self, **overrides):
        base = {
            "streams": np.array([0.6, 0.25, 0.15]),
            "virality_index": np.array([0.55, 0.3, 0.15]),
            "geo_penetration": np.array([0.5, 0.35, 0.15]),
            "convergence": np.array([0.8, 0.2]),
            "urgency_score": 0.7,
            "rust_belt_share": 0.6,
            "convergence_score": 0.75,
            "virality": 0.55,
        }
        base.update(overrides)
        return base

    def _preferred(self):
        return {
            "streams": 0.75,
            "virality_index": 0.65,
            "geo_penetration": 0.60,
            "convergence": 0.85,
        }

    def test_returns_float(self):
        efe = self.layer.compute_efe(self._outcomes(), self._preferred())
        assert isinstance(efe, float)

    def test_higher_canon_signals_lower_efe(self):
        """Outcomes with stronger canon alignment should yield a lower EFE."""
        weak = self._outcomes(urgency_score=0.1, rust_belt_share=0.1, convergence_score=0.1, virality=0.1)
        strong = self._outcomes(urgency_score=0.9, rust_belt_share=0.9, convergence_score=0.9, virality=0.9)
        efe_weak = self.layer.compute_efe(weak, self._preferred())
        efe_strong = self.layer.compute_efe(strong, self._preferred())
        assert efe_strong < efe_weak

    def test_preferred_keys_not_in_outcomes_ignored(self):
        """Extra preferred keys that have no matching outcome should not crash."""
        outcomes = {"urgency_score": 0.5}
        preferred = {"streams": 0.75, "nonexistent_metric": 0.5}
        efe = self.layer.compute_efe(outcomes, preferred)
        assert isinstance(efe, float)

    def test_info_gain_decreases_with_convergence(self):
        """Higher historical convergence → lower info_gain → higher EFE all else equal."""
        layer_low = ActiveInferenceLayer(CanonPreferences(historical_convergence_score=0.0))
        layer_high = ActiveInferenceLayer(CanonPreferences(historical_convergence_score=1.0))
        outcomes = self._outcomes()
        preferred = self._preferred()
        # Lower convergence gets more info_gain bonus → lower EFE
        assert layer_low.compute_efe(outcomes, preferred) < layer_high.compute_efe(outcomes, preferred)


# ---------------------------------------------------------------------------
# ActiveInferenceLayer — score_protocol
# ---------------------------------------------------------------------------

class TestScoreProtocol:
    def setup_method(self):
        self.layer = ActiveInferenceLayer()

    def _make_results(self, n: int = 10):
        rng = np.random.default_rng(0)
        return [
            {
                "streams": int(rng.integers(100, 5000)),
                "virality": float(rng.uniform(0.1, 0.9)),
                "rust_belt_share": float(rng.uniform(0.1, 0.7)),
                "urgency_score": float(rng.uniform(0.1, 0.9)),
                "convergence_score": float(rng.uniform(0.4, 1.0)),
                "streams_bucket": np.array([0.6, 0.25, 0.15]),
            }
            for _ in range(n)
        ]

    def test_returns_required_keys(self):
        scored = self.layer.score_protocol(self._make_results())
        for key in ("best_protocol", "efe_score", "all_efes", "mean_efe", "std_efe",
                    "canon_bonus_applied", "recommended_protocol"):
            assert key in scored

    def test_best_protocol_is_in_sim_results(self):
        results = self._make_results(20)
        scored = self.layer.score_protocol(results)
        assert scored["best_protocol"] in results

    def test_all_efes_length_matches_input(self):
        results = self._make_results(15)
        scored = self.layer.score_protocol(results)
        assert len(scored["all_efes"]) == 15

    def test_efe_score_is_min_of_all_efes(self):
        results = self._make_results(10)
        scored = self.layer.score_protocol(results)
        assert scored["efe_score"] == pytest.approx(min(scored["all_efes"]))

    def test_canon_bonus_applied_is_true(self):
        scored = self.layer.score_protocol(self._make_results())
        assert scored["canon_bonus_applied"] is True

    def test_recommended_protocol_is_string(self):
        scored = self.layer.score_protocol(self._make_results())
        assert isinstance(scored["recommended_protocol"], str)
        assert len(scored["recommended_protocol"]) > 0


# ---------------------------------------------------------------------------
# ActiveInferenceLayer — auto_tune_priors
# ---------------------------------------------------------------------------

class TestAutoTunePriors:
    def setup_method(self):
        self.layer = ActiveInferenceLayer()

    def test_returns_dict(self):
        result = self.layer.auto_tune_priors({
            "urgency_impact": 0.8,
            "rust_belt_conversion": 0.65,
            "convergence_achieved": 0.85,
        })
        assert isinstance(result, dict)

    def test_history_appended(self):
        self.layer.auto_tune_priors({"urgency_impact": 0.7})
        assert len(self.layer.history) == 1
        self.layer.auto_tune_priors({"urgency_impact": 0.9})
        assert len(self.layer.history) == 2

    def test_priors_stay_in_bounds_after_many_updates(self):
        for _ in range(50):
            self.layer.auto_tune_priors({
                "urgency_impact": float(np.random.uniform(0.0, 1.0)),
                "rust_belt_conversion": float(np.random.uniform(0.0, 1.0)),
                "convergence_achieved": float(np.random.uniform(0.0, 1.0)),
            })
        prefs = self.layer.prefs
        assert 0.1 <= prefs.mortality_urgency_weight <= 0.6
        assert 0.1 <= prefs.lorain_roots_geo_weight <= 0.5
        assert 0.1 <= prefs.builder_proof_of_work_weight <= 0.5

    def test_convergence_score_is_ema(self):
        """historical_convergence_score should track via exponential moving average."""
        self.layer.auto_tune_priors({"convergence_achieved": 1.0})
        # After one step from 0.0: 0.7*0.0 + 0.3*1.0 = 0.3
        assert self.layer.prefs.historical_convergence_score == pytest.approx(0.3)

    def test_custom_learning_rate_applied(self):
        prefs_before = self.layer.prefs.to_dict()
        self.layer.auto_tune_priors({"urgency_impact": 0.5}, learning_rate=0.5)
        prefs_after = self.layer.prefs.to_dict()
        # At least one weight should have changed
        assert prefs_before != prefs_after

    def test_missing_keys_use_defaults(self):
        """Missing real_outcomes keys should default to 0.0 without raising."""
        result = self.layer.auto_tune_priors({})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# SIRGraph
# ---------------------------------------------------------------------------

class TestSIRGraph:
    def setup_method(self):
        self.graph = SIRGraph(n_nodes=200, seed=7, lorain_boost=1.3)

    def test_graph_dimensions(self):
        assert self.graph.adj_matrix.shape == (200, 200)
        assert len(self.graph.geo) == 200
        assert len(self.graph.theme_resonance) == 200
        assert len(self.graph.geo_boost) == 200

    def test_rust_belt_fraction(self):
        rust_count = int(np.sum(self.graph.geo == 1.0))
        expected = int(0.15 * 200)
        assert rust_count == expected

    def test_geo_boost_values(self):
        """Rust Belt nodes have lorain_boost, others have 1.0."""
        for i in range(200):
            if self.graph.geo[i] == 1.0:
                assert self.graph.geo_boost[i] == pytest.approx(1.3)
            else:
                assert self.graph.geo_boost[i] == pytest.approx(1.0)

    def test_initialize_states_resets(self):
        self.graph.initialize_states(seed_fraction=0.1, timeline_seed=42)
        assert self.graph.states is not None
        assert len(self.graph.states) == 200
        infected = np.sum(self.graph.states == SIRGraph.STATES["I"])
        assert infected == pytest.approx(20, abs=1)

    def test_step_returns_array(self):
        self.graph.initialize_states()
        result = self.graph.step(beta=0.1, gamma=0.2, delta=0.05)
        assert isinstance(result, np.ndarray)
        assert len(result) == 200

    def test_step_state_values_are_valid(self):
        self.graph.initialize_states()
        self.graph.step()
        valid_states = set(SIRGraph.STATES.values())
        for state in np.unique(self.graph.states):
            assert int(state) in valid_states

    def test_states_none_before_initialize(self):
        graph = SIRGraph(n_nodes=100, seed=1)
        assert graph.states is None

    def test_step_auto_initializes(self):
        """step() should auto-initialize states if not yet set."""
        graph = SIRGraph(n_nodes=100, seed=1)
        result = graph.step()
        assert result is not None


# ---------------------------------------------------------------------------
# sir_step_numba
# ---------------------------------------------------------------------------

class TestSirStepNumba:
    def setup_method(self):
        n = 100
        self.n = n
        rng = np.random.default_rng(0)
        self.adj = (rng.random((n, n)) > 0.9).astype(np.float64)
        self.theme = rng.beta(2, 5, n).astype(np.float64)
        self.geo = np.where(np.arange(n) < 15, 1.3, 1.0).astype(np.float64)
        self.states = np.zeros(n, dtype=np.int8)
        self.states[:5] = 2  # seed 5 infected

    def test_returns_ndarray(self):
        result = sir_step_numba(
            self.states, self.adj, 0.08, 0.15, 0.02,
            self.theme, self.geo, 0.0, self.n, 42,
        )
        assert isinstance(result, np.ndarray)

    def test_output_length_unchanged(self):
        result = sir_step_numba(
            self.states, self.adj, 0.08, 0.15, 0.02,
            self.theme, self.geo, 0.0, self.n, 42,
        )
        assert len(result) == self.n

    def test_state_values_within_range(self):
        result = sir_step_numba(
            self.states, self.adj, 0.08, 0.15, 0.02,
            self.theme, self.geo, 0.0, self.n, 42,
        )
        assert int(result.min()) >= 0
        assert int(result.max()) <= 4

    def test_post_strength_exposes_s_nodes(self):
        """A non-zero post_strength should expose some S nodes to E."""
        states_before = self.states.copy()
        result = sir_step_numba(
            self.states, self.adj, 0.08, 0.15, 0.02,
            self.theme, self.geo, 0.5, self.n, 99,
        )
        # With 50% exposure some S nodes should now be E (1) or higher
        newly_non_s = np.sum((states_before == 0) & (result > 0))
        assert newly_non_s > 0

    def test_input_states_not_mutated(self):
        """Function should return a copy; original states array should be unchanged."""
        original = self.states.copy()
        sir_step_numba(
            self.states, self.adj, 0.08, 0.15, 0.02,
            self.theme, self.geo, 0.0, self.n, 42,
        )
        np.testing.assert_array_equal(self.states, original)


# ---------------------------------------------------------------------------
# MajoraCore
# ---------------------------------------------------------------------------

class TestMajoraCoreInit:
    def test_default_init(self):
        core = MajoraCore(n_nodes=100)
        assert core.graph is not None
        assert core.ai_layer is not None
        assert core.cache == {}

    def test_custom_cache_size(self):
        core = MajoraCore(n_nodes=100, cache_size=4)
        assert core.cache_max == 4


class TestMajoraCoreContentHash:
    def setup_method(self):
        self.core = MajoraCore(n_nodes=100)

    def test_string_input_returns_hex(self):
        h = self.core._content_hash("hello")
        assert isinstance(h, str)
        assert len(h) == 16

    def test_dict_input_is_deterministic(self):
        d = {"title": "WE ALL DIE ONE DAY", "artist": "IAMBANDOBANDZ"}
        assert self.core._content_hash(d) == self.core._content_hash(d)

    def test_different_dicts_different_hashes(self):
        h1 = self.core._content_hash({"a": 1})
        h2 = self.core._content_hash({"a": 2})
        assert h1 != h2


class TestMajoraCoreRunMonteCarlo:
    def setup_method(self):
        self.core = MajoraCore(n_nodes=300)

    def test_returns_required_keys(self):
        result = self.core.run_monte_carlo("test_hash", n_sims=5)
        for key in (
            "winning_drop_time",
            "efe_score",
            "efe_std",
            "canon_aligned_metrics",
            "recommended_protocol",
            "numba_accelerated",
            "n_sims_run",
            "preset",
            "content_hash",
        ):
            assert key in result

    def test_n_sims_run_matches_input(self):
        result = self.core.run_monte_carlo("test_x", n_sims=8)
        assert result["n_sims_run"] == 8

    def test_numba_flag_matches_availability(self):
        result = self.core.run_monte_carlo("test_nb", n_sims=3)
        assert result["numba_accelerated"] == NUMBA_AVAILABLE

    def test_efe_score_is_float(self):
        result = self.core.run_monte_carlo("test_efe", n_sims=5)
        assert isinstance(result["efe_score"], float)

    def test_preset_stored_in_result(self):
        result = self.core.run_monte_carlo("test_preset", preset="repo_release", n_sims=3)
        assert result["preset"] == "repo_release"

    def test_cache_hit_returns_same_result(self):
        result1 = self.core.run_monte_carlo("cached_content", n_sims=5)
        result2 = self.core.run_monte_carlo("cached_content", n_sims=5)
        assert result1 is result2  # same object from cache

    def test_dict_content_works(self):
        metadata = {"title": "WE ALL DIE ONE DAY", "artist": "IAMBANDOBANDZ"}
        result = self.core.run_monte_carlo(metadata, n_sims=5)
        assert result["content_hash"] == self.core._content_hash(metadata)

    def test_cache_eviction_when_full(self):
        core = MajoraCore(n_nodes=100, cache_size=2)
        core.run_monte_carlo("key1", n_sims=3)
        core.run_monte_carlo("key2", n_sims=3)
        assert len(core.cache) == 2
        core.run_monte_carlo("key3", n_sims=3)
        # Cache should still be at max size
        assert len(core.cache) == 2

    def test_all_presets_run_without_error(self):
        for preset in ("music_single", "repo_release", "album", "general"):
            result = self.core.run_monte_carlo(f"content_{preset}", preset=preset, n_sims=3)  # type: ignore[arg-type]
            assert result["preset"] == preset


class TestMajoraCoreBackground:
    def test_start_stop_background(self):
        core = MajoraCore(n_nodes=100)
        core.start_background_majora(interval_seconds=9999)
        assert core.running is True
        assert core.background_thread is not None
        assert core.background_thread.is_alive()
        core.stop_background()
        assert core.running is False

    def test_stop_without_start_is_safe(self):
        core = MajoraCore(n_nodes=100)
        core.stop_background()  # Should not raise


class TestMajoraCoreStatesProperty:
    def test_states_property_matches_graph(self):
        core = MajoraCore(n_nodes=100)
        assert core.STATES == SIRGraph.STATES
