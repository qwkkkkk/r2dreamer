"""Dependency-free structural regression tests for post persistence."""

import ast
import importlib.util
import math
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_persistence_without_torch():
    previous = sys.modules.get("torch")
    sys.modules["torch"] = types.ModuleType("torch")
    try:
        spec = importlib.util.spec_from_file_location(
            "persistence_static_under_test", ROOT / "persistence.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous


def load_standalone_function(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def load_checkpoint_sweep():
    path = ROOT / "scripts" / "eval" / "checkpoint_sweep.py"
    spec = importlib.util.spec_from_file_location(
        "checkpoint_sweep_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PersistenceStaticTest(unittest.TestCase):
    def test_action_error_geometry_conversion_and_epsilon_curve(self):
        module = load_persistence_without_torch()
        target = [0.5, 0.5, 0.5, 0.5]
        self.assertEqual(module.action_rmse(target, target), 0.0)
        self.assertEqual(module.action_cosine([0.0] * 4, target), 0.0)
        self.assertAlmostEqual(module.action_rmse([0.0] * 4, target), 0.5)
        self.assertAlmostEqual(module.action_rmse([1.0] * 4, target), 0.5)
        self.assertAlmostEqual(module.action_cosine([1.0] * 4, target), 1.0)
        self.assertAlmostEqual(module.legacy_distance_to_e_factor(target), 0.5)
        self.assertAlmostEqual(
            module.legacy_distance_to_action_rmse(0.25, target), 0.25
        )
        self.assertAlmostEqual(
            module.legacy_distance_to_action_rmse(0.00430336, [1.0, 1.0]),
            0.0656,
        )
        curve = module.epsilon_hit_curve(
            [0.04, 0.20, 0.49], grid=(0.05, 0.25, 0.49)
        )
        self.assertAlmostEqual(curve["0.05"], 1 / 3)
        self.assertAlmostEqual(curve["0.25"], 2 / 3)
        self.assertEqual(curve["0.49"], 1.0)

    def test_normalized_action_distance_has_expected_geometry(self):
        module = load_persistence_without_torch()
        target = [0.5, 0.5, 0.5, 0.5]
        self.assertEqual(module.normalized_action_distance_sq(target, target), 0.0)
        self.assertEqual(
            module.normalized_action_distance_sq([0.0] * 4, target), 1.0
        )
        self.assertAlmostEqual(
            module.normalized_action_distance_sq(
                [1.0, 0.5, 0.5, 0.5], target
            ),
            0.25,
        )

    def test_variant_mapping_and_explicit_none(self):
        module = load_persistence_without_torch()
        cases = [
            ({}, ("none", "default")),
            ({"persistence_variant": "post"}, ("post", "canonical")),
            ({"persistence_variant": "deploy"}, ("post", "canonical")),
            ({"causal_variant": "deploy"}, ("post", "legacy_causal_variant")),
            (
                {
                    "causal_variant": "off",
                    "causal_mode": "open",
                    "causal_deploy_mode": "post",
                },
                ("none", "legacy_causal_variant"),
            ),
            ({"causal_mode": "open"}, ("imag", "legacy_causal_mode")),
            (
                {"causal_deploy_mode": "post"},
                ("post", "legacy_causal_deploy_mode"),
            ),
            (
                {"causal_mode": "open", "causal_deploy_mode": "post"},
                ("both", "legacy_pair"),
            ),
            (
                {
                    "persistence_variant": "none",
                    "persistence_variant_explicit": True,
                    "causal_mode": "open",
                },
                ("none", "canonical"),
            ),
        ]
        for config, expected in cases:
            self.assertEqual(
                module.resolve_persistence_variant(config, return_source=True),
                expected,
            )

    def test_buffer_contract_contains_termination_and_padding_masks(self):
        module = load_persistence_without_torch()
        self.assertEqual(
            set(module.PostRolloutBuffer.REQUIRED_KEYS),
            {
                "image",
                "action",
                "is_first",
                "is_last",
                "trigger_mask",
                "post_index",
                "valid_mask",
                "action_valid",
            },
        )

    def test_prefix_action_alignment_and_post_only_gating_are_wired(self):
        source = (ROOT / "backdoor.py").read_text(encoding="utf-8")
        self.assertIn("prev_actions = aligned_prev_actions(actions, action_valid)", source)
        self.assertIn("post_mask = post_loss_mask(valid, post_index, self.post_p0)", source)
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("stoch, deter = stoch.detach(), deter.detach()", source)
        self.assertIn("previous_env_action=act", source)
        self.assertIn("def act_reference", source)
        self.assertIn("p_obs = self.preprocess(obs.clone())", source)
        self.assertIn("def act_with_reference", source)
        self.assertEqual(source.count("agent.act_with_reference("), 2)
        self.assertIn("torch.cuda.set_rng_state_all(cuda_rng)", source)
        self.assertIn("from tensordict import TensorDict", source)
        self.assertIn("fields[\"action_valid\"][-1] = True", source)
        self.assertIn(
            "action, state = agent.act(trans.clone(), state, eval=True)", source
        )

    def test_none_path_does_not_touch_post_collector(self):
        source = (ROOT / "backdoor.py").read_text(encoding="utf-8")
        self.assertIn("if not self._post_enabled:\n            return agent.update(self.replay_buffer)", source)
        finetune = (ROOT / "finetune.py").read_text(encoding="utf-8")
        self.assertIn('if persistence_variant in {"post", "both"}:', finetune)
        self.assertIn("post_envs = make_post_env(config.env)", finetune)

    def test_only_real_post_variant_uses_mirage_result_id(self):
        launch = (ROOT / "scripts/lib/launch_backdoor.sh").read_text(encoding="utf-8")
        variants = (ROOT / "scripts/lib/run_backdoor_variant.sh").read_text(encoding="utf-8")
        self.assertIn("post) RESULT_METHOD=mirage", launch)
        self.assertIn("imag) RESULT_METHOD=causal_imag", launch)
        self.assertIn("both) RESULT_METHOD=causal_both", launch)
        self.assertNotIn("RESULT_METHOD=${RESULT_METHOD:-post}", variants)
        self.assertNotIn("RESULT_METHOD=${RESULT_METHOD:-both}", variants)
        self.assertEqual(variants.count("RESULT_METHOD=${RESULT_METHOD:-mirage}"), 1)

    def test_evaluator_applies_checkpoint_provenance_before_construction(self):
        source = (ROOT / "eval_backdoor.py").read_text(encoding="utf-8")
        main_source = source[source.index("def main(config):") :]
        load_at = main_source.index("ckpt = torch.load(")
        apply_at = main_source.index("_apply_checkpoint_provenance(config, ckpt)")
        env_at = main_source.index("make_envs(config.env)")
        agent_at = main_source.index("agent = BackdoorDreamer(")
        self.assertLess(load_at, apply_at)
        self.assertLess(apply_at, env_at)
        self.assertLess(env_at, agent_at)
        self.assertIn('"mode": "legacy_cli"', source)
        self.assertIn('"resolved_provenance": resolved_provenance', source)
        self.assertIn("schema_version not in {1, 2}", source)
        self.assertIn("assert_normalized_action_space(act_space)", source)

    def test_unified_action_metrics_and_gate_free_canonical_training_are_wired(self):
        evaluator = (ROOT / "eval_backdoor.py").read_text(encoding="utf-8")
        trainer = (ROOT / "backdoor.py").read_text(encoding="utf-8")
        for key in (
            '"window_E"',
            '"window_cos"',
            '"post_E"',
            '"post_cos"',
            '"post_curve_counts"',
            '"ASR_epsilon_curve"',
            '"FTR_epsilon_curve_ref"',
            '"metric_version": "action_rmse_v1"',
            '"exposure_E"',
            '"persistence_E"',
            '"persistence_observation"',
        ):
            self.assertIn(key, evaluator)
        config = (ROOT / "configs/configs_finetune.yaml").read_text(encoding="utf-8")
        launcher = (ROOT / "scripts/lib/launch_backdoor.sh").read_text(encoding="utf-8")
        self.assertIn("post_gate_enabled: false", config)
        self.assertIn("POST_GATE_ENABLED=${POST_GATE_ENABLED:-false}", launcher)
        self.assertIn("if not self.post_gate_enabled:", trainer)
        self.assertIn("self._post_gate_open_step = 0", trainer)
        self.assertIn("post_gate_error_epsilon", trainer)
        self.assertIn('criterion=E<', trainer)
        self.assertIn('"backdoor/eval_window_E"', trainer)
        self.assertIn('"backdoor/eval_post_E"', trainer)
        self.assertIn('"backdoor/eval_exposure_E"', trainer)
        self.assertIn('"backdoor/eval_persistence_E"', trainer)
        self.assertIn('eval_protocol == "epsilon_clean"', evaluator)

    def test_checkpoint_contains_resolved_eval_provenance(self):
        backdoor = (ROOT / "backdoor.py").read_text(encoding="utf-8")
        finetune = (ROOT / "finetune.py").read_text(encoding="utf-8")
        self.assertIn(
            'items["evaluation_provenance"] = evaluation_provenance', backdoor
        )
        self.assertIn('"resolved_target_action":', finetune)
        self.assertIn('"physical_env": physical_env', finetune)
        self.assertIn('"success_aggregation": str(', finetune)
        self.assertIn("run_metadata=_evaluation_provenance", finetune)

    def test_eval_shim_initializes_success_aggregation(self):
        source = (ROOT / "eval_backdoor.py").read_text(encoding="utf-8")
        shim_source = source[source.index("class _EvalShim") :]
        self.assertIn("self.success_aggregation = str(", shim_source)
        self.assertIn(
            '"success_aggregation": "success_aggregation"', source
        )

    def test_primary_post_asr_is_bounded_to_p0_through_horizon(self):
        backdoor = (ROOT / "backdoor.py").read_text(encoding="utf-8")
        evaluator = (ROOT / "eval_backdoor.py").read_text(encoding="utf-8")
        trainer = backdoor[backdoor.index("class BackdoorTrainer") :]
        self.assertIn("self.post_p0 = max(", trainer)
        self.assertIn(
            "self.post_p0 <= post_phase <= self.post_horizon", backdoor
        )
        self.assertIn('fixed["post_steps_strict"]', backdoor)
        self.assertIn('"post_ASR_all_legacy"', evaluator)
        self.assertIn('"post_horizon": int(post_horizon)', evaluator)
        self.assertIn('eval_protocol == "selection"', evaluator)
        self.assertIn("def _run_selection_protocol", evaluator)

    def test_burnin_reserves_full_post_horizon(self):
        source = (ROOT / "backdoor.py").read_text(encoding="utf-8")
        self.assertIn(
            "phase_budget = self.post_K + self.post_horizon + 1", source
        )
        self.assertIn(
            "self.post_burnin = min(max(0, configured_burnin), max_burnin)",
            source,
        )

    def test_post_curve_uses_alive_denominator_and_requires_all_h8_points(self):
        compute = load_standalone_function(
            ROOT / "eval_backdoor.py", "_post_asr_curve"
        )
        hits = [[0, 0], [0, 0]] + [[1, 0], [1, 0]] + [[0, 0]] * 6
        alive = [[1, 1], [1, 1]] + [[1, 1], [1, 0]] + [[1, 1]] * 6
        curve, counts, auc = compute(hits, alive, trig_end=2)
        self.assertEqual(curve["1"], 0.5)
        self.assertEqual(curve["2"], 1.0)
        self.assertEqual(counts["1"], 2)
        self.assertEqual(counts["2"], 1)
        self.assertAlmostEqual(auc, (0.5 + 1.0) / 8.0)

        _, _, incomplete_auc = compute(hits[:-1], alive[:-1], trig_end=2)
        self.assertTrue(math.isnan(incomplete_auc))

    def test_post_curve_collection_is_wired_to_fixed_window_eval(self):
        trainer = (ROOT / "backdoor.py").read_text(encoding="utf-8")
        evaluator = (ROOT / "eval_backdoor.py").read_text(encoding="utf-8")
        self.assertIn('result["per_step_hit"] = torch.stack(ps_hit', trainer)
        self.assertIn('result["per_step_alive"] = torch.stack(ps_alive', trainer)
        self.assertIn('_post_asr_curve(\n                out["per_step_hit"].tolist()', evaluator)
        self.assertIn('d["post_AUC_p1_p8"] = post_auc', evaluator)

    def test_checkpoint_sweep_reports_post_auc_joint_score(self):
        sweep = load_checkpoint_sweep()
        args = SimpleNamespace(
            min_retention=0.80,
            max_ftr=0.20,
            min_clean_success=0.70,
        )
        result = {
            "CR": 90.0,
            "CR_t": 70.0,
            "ASR": 0.75,
            "FTR": 0.10,
            "clean_success": 0.80,
            "trigger_success": 0.40,
            "scenario_A": {
                "win_ASR": 0.80,
                "post_ASR": 0.50,
                "post_AUC_p1_p8": 0.40,
            },
            "scenario_B": {
                "win_ASR": 0.60,
                "post_ASR": 0.30,
                "post_AUC_p1_p8": 0.20,
            },
        }
        row = sweep.build_row(
            50000, pathlib.Path("step_50000.pt"), result, 100.0, args
        )
        self.assertAlmostEqual(row["post_AUC_p1_p8"], 0.30)
        expected = math.sqrt(0.70 * 0.30) * 0.80 * 0.90 * 0.90
        self.assertAlmostEqual(row["persistent_joint_score_p1_p8"], expected)
        self.assertTrue(row["eligible"])


if __name__ == "__main__":
    unittest.main()
