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


# ---------------------------------------------------------------------------
# CanonPreferences — additional edge cases
# ---------------------------------------------------------------------------

class TestCanonPreferencesEdgeCases:
    def test_mutation_is_reflected(self):
        prefs = CanonPreferences()
        prefs.mortality_urgency_weight = 0.55
        assert prefs.to_dict()["mortality_urgency_weight"] == pytest.approx(0.55)

    def test_all_fields_are_floats(self):
        prefs = CanonPreferences()
        for v in prefs.to_dict().values():
            assert isinstance(v, float)

    def test_zero_adaptation_rate_does_not_update_weights(self):
        """With adaptation_rate=0, weights should not change in auto_tune_priors."""
        layer = ActiveInferenceLayer(CanonPreferences(adaptation_rate=0.0))
        before = layer.prefs.to_dict()
        layer.auto_tune_priors({"urgency_impact": 0.9, "rust_belt_conversion": 0.7, "convergence_achieved": 0.8})
        after = layer.prefs.to_dict()
        # Weights clamped but not shifted when lr=0
        for key in ("mortality_urgency_weight", "lorain_roots_geo_weight", "builder_proof_of_work_weight"):
            assert after[key] == pytest.approx(before[key])


# ---------------------------------------------------------------------------
# ActiveInferenceLayer — edge cases
# ---------------------------------------------------------------------------

class TestActiveInferenceLayerEdgeCases:
    def test_empty_sim_results_raises_or_handles(self):
        """score_protocol with one element should not raise."""
        layer = ActiveInferenceLayer()
        result = layer.score_protocol([{
            "streams": 100,
            "virality": 0.5,
            "rust_belt_share": 0.3,
            "urgency_score": 0.6,
            "convergence_score": 0.7,
            "streams_bucket": np.array([0.6, 0.25, 0.15]),
        }])
        assert result["efe_score"] == min(result["all_efes"])

    def test_compute_efe_all_zero_outcomes(self):
        layer = ActiveInferenceLayer()
        outcomes = {
            "urgency_score": 0.0,
            "rust_belt_share": 0.0,
            "convergence_score": 0.0,
            "virality": 0.0,
        }
        efe = layer.compute_efe(outcomes, {})
        assert isinstance(efe, float)

    def test_score_protocol_std_efe_non_negative(self):
        layer = ActiveInferenceLayer()
        results = [
            {
                "streams": 500,
                "virality": float(np.random.uniform(0.2, 0.8)),
                "rust_belt_share": 0.4,
                "urgency_score": 0.6,
                "convergence_score": 0.7,
                "streams_bucket": np.array([0.6, 0.25, 0.15]),
            }
            for _ in range(10)
        ]
        scored = layer.score_protocol(results)
        assert scored["std_efe"] >= 0.0

    def test_auto_tune_priors_clamps_upper_bound(self):
        """A very high learning rate should not push weights above their bounds."""
        layer = ActiveInferenceLayer(CanonPreferences(mortality_urgency_weight=0.59))
        layer.auto_tune_priors({"urgency_impact": 0.0}, learning_rate=10.0)
        assert layer.prefs.mortality_urgency_weight <= 0.6

    def test_auto_tune_priors_clamps_lower_bound(self):
        """A very negative gradient should not push weights below their bounds."""
        layer = ActiveInferenceLayer(CanonPreferences(lorain_roots_geo_weight=0.11))
        layer.auto_tune_priors({"rust_belt_conversion": 1.0}, learning_rate=10.0)
        assert layer.prefs.lorain_roots_geo_weight >= 0.1

    def test_historical_convergence_never_negative(self):
        layer = ActiveInferenceLayer()
        layer.auto_tune_priors({"convergence_achieved": 0.0})
        assert layer.prefs.historical_convergence_score >= 0.0

    def test_score_protocol_best_efe_leq_all_others(self):
        layer = ActiveInferenceLayer()
        results = [
            {
                "streams": i * 100,
                "virality": 0.1 * i,
                "rust_belt_share": 0.3,
                "urgency_score": 0.5,
                "convergence_score": 0.6,
                "streams_bucket": np.array([0.6, 0.25, 0.15]),
            }
            for i in range(1, 8)
        ]
        scored = layer.score_protocol(results)
        for efe in scored["all_efes"]:
            assert scored["efe_score"] <= efe


# ---------------------------------------------------------------------------
# SIRGraph — additional edge cases
# ---------------------------------------------------------------------------

class TestSIRGraphEdgeCases:
    def test_different_seeds_produce_different_graphs(self):
        g1 = SIRGraph(n_nodes=100, seed=1)
        g2 = SIRGraph(n_nodes=100, seed=2)
        assert not np.array_equal(g1.adj_matrix, g2.adj_matrix)

    def test_initialize_states_repeated_call_resets(self):
        graph = SIRGraph(n_nodes=200, seed=3)
        graph.initialize_states(seed_fraction=0.5)
        infected_first = int(np.sum(graph.states == 2))
        graph.initialize_states(seed_fraction=0.02)
        infected_second = int(np.sum(graph.states == 2))
        assert infected_first != infected_second

    def test_state_dtype_is_int8(self):
        graph = SIRGraph(n_nodes=100, seed=0)
        graph.initialize_states()
        assert graph.states.dtype == np.int8

    def test_all_nodes_susceptible_after_zero_seed_fraction(self):
        graph = SIRGraph(n_nodes=200, seed=5)
        graph.initialize_states(seed_fraction=0.0)
        assert np.all(graph.states == 0)

    def test_theme_resonance_in_zero_one_range(self):
        graph = SIRGraph(n_nodes=500, seed=9)
        assert float(graph.theme_resonance.min()) >= 0.0
        assert float(graph.theme_resonance.max()) <= 1.0

    def test_lorain_boost_reflected_in_geo_boost(self):
        graph = SIRGraph(n_nodes=100, seed=1, lorain_boost=2.5)
        rust_nodes = graph.geo == 1.0
        assert np.all(graph.geo_boost[rust_nodes] == pytest.approx(2.5))

    def test_build_graph_is_idempotent(self):
        graph = SIRGraph(n_nodes=100, seed=7)
        adj1 = graph.adj_matrix.copy()
        graph.build_graph()
        np.testing.assert_array_equal(graph.adj_matrix, adj1)

    def test_step_with_high_beta_increases_non_susceptible(self):
        graph = SIRGraph(n_nodes=300, seed=42)
        graph.initialize_states(seed_fraction=0.05)
        s_before = int(np.sum(graph.states == 0))
        for _ in range(5):
            graph.step(beta=0.5, gamma=0.5, delta=0.0)
        s_after = int(np.sum(graph.states == 0))
        assert s_after < s_before


# ---------------------------------------------------------------------------
# sir_step_numba — additional edge cases
# ---------------------------------------------------------------------------

class TestSirStepNumbaEdgeCases:
    def _setup(self, n: int, n_infected: int = 5, seed: int = 0):
        rng = np.random.default_rng(seed)
        adj = (rng.random((n, n)) > 0.88).astype(np.float64)
        theme = rng.beta(2, 5, n).astype(np.float64)
        geo = np.ones(n, dtype=np.float64)
        states = np.zeros(n, dtype=np.int8)
        states[:n_infected] = 2
        return states, adj, theme, geo

    def test_no_spread_with_zero_beta(self):
        n = 60
        states, adj, theme, geo = self._setup(n)
        initial_infected = int(np.sum(states == 2))
        result = sir_step_numba(states, adj, 0.0, 0.0, 0.0, theme, geo, 0.0, n, 1)
        # With zero rates no S node can become E
        newly_exposed = int(np.sum((states == 0) & (result == 1)))
        assert newly_exposed == 0

    def test_all_e_convert_with_gamma_one(self):
        n = 40
        states = np.full(n, 1, dtype=np.int8)  # all E
        adj = np.zeros((n, n), dtype=np.float64)
        theme = np.zeros(n, dtype=np.float64)
        geo = np.ones(n, dtype=np.float64)
        result = sir_step_numba(states, adj, 0.0, 1.0, 0.0, theme, geo, 0.0, n, 2)
        assert np.all(result == 2)

    def test_deterministic_with_same_seed(self):
        n = 50
        states, adj, theme, geo = self._setup(n, seed=77)
        r1 = sir_step_numba(states, adj, 0.08, 0.15, 0.02, theme, geo, 0.0, n, 42)
        r2 = sir_step_numba(states, adj, 0.08, 0.15, 0.02, theme, geo, 0.0, n, 42)
        np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# MajoraCore — additional edge cases
# ---------------------------------------------------------------------------

class TestMajoraCoreEdgeCases:
    def test_content_hash_bytes_input(self):
        core = MajoraCore(n_nodes=100)
        h = core._content_hash(b"raw bytes")
        assert isinstance(h, str)
        assert len(h) == 16

    def test_cache_key_combines_hash_and_preset(self):
        core = MajoraCore(n_nodes=100)
        key = core._cache_key("abc123", "album")
        assert key == "abc123_album"

    def test_same_content_different_presets_different_cache_keys(self):
        core = MajoraCore(n_nodes=100)
        k1 = core._cache_key("abc", "music_single")
        k2 = core._cache_key("abc", "album")
        assert k1 != k2

    def test_run_monte_carlo_single_sim(self):
        core = MajoraCore(n_nodes=100)
        result = core.run_monte_carlo("single_sim", n_sims=1)
        assert result["n_sims_run"] == 1
        assert isinstance(result["efe_score"], float)

    def test_run_monte_carlo_efe_std_non_negative(self):
        core = MajoraCore(n_nodes=200)
        result = core.run_monte_carlo("std_test", n_sims=10)
        assert result["efe_std"] >= 0.0

    def test_run_monte_carlo_with_explicit_seeds(self):
        core = MajoraCore(n_nodes=200)
        seeds = list(range(10))
        result = core.run_monte_carlo("seeded", n_sims=10, timeline_seeds=seeds)
        assert result["n_sims_run"] == 10

    def test_canon_aligned_metrics_dict_has_four_keys(self):
        core = MajoraCore(n_nodes=150)
        result = core.run_monte_carlo("metric_check", n_sims=5)
        metrics = result["canon_aligned_metrics"]
        expected_keys = {"urgency_score", "rust_belt_share", "convergence_score", "virality"}
        assert set(metrics.keys()) == expected_keys

    def test_winning_drop_time_is_string(self):
        core = MajoraCore(n_nodes=100)
        result = core.run_monte_carlo("drop_time", n_sims=3)
        assert isinstance(result["winning_drop_time"], str)

    def test_background_does_not_run_without_history(self):
        """Background thread should sleep without calling auto_tune if history is empty."""
        core = MajoraCore(n_nodes=100)
        assert len(core.ai_layer.history) == 0
        core.start_background_majora(interval_seconds=9999)
        import time as _time
        _time.sleep(0.05)  # Let thread tick once
        core.stop_background()
        # No history was added because the thread saw an empty history list
        assert len(core.ai_layer.history) == 0

    def test_efe_score_is_finite(self):
        core = MajoraCore(n_nodes=150)
        result = core.run_monte_carlo("finite_efe", n_sims=5)
        assert np.isfinite(result["efe_score"])

    def test_cache_grows_up_to_max(self):
        core = MajoraCore(n_nodes=100, cache_size=3)
        for i in range(3):
            core.run_monte_carlo(f"unique_{i}", n_sims=2)
        assert len(core.cache) == 3

    def test_states_property_returns_correct_dict(self):
        core = MajoraCore(n_nodes=100)
        s = core.STATES
        assert set(s.keys()) == {"S", "E", "I", "A", "F"}
        assert all(isinstance(v, int) for v in s.values())

