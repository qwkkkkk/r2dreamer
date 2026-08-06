"""Dependency-free structural regression tests for post persistence."""

import importlib.util
import pathlib
import sys
import types
import unittest


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


class PersistenceStaticTest(unittest.TestCase):
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
        self.assertIn("state[\"prev_action\"] = action", source)
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

    def test_persistence_variants_keep_the_five_method_result_id(self):
        launch = (ROOT / "scripts/lib/launch_backdoor.sh").read_text(encoding="utf-8")
        variants = (ROOT / "scripts/lib/run_backdoor_variant.sh").read_text(encoding="utf-8")
        self.assertIn("post|imag|both) RESULT_METHOD=causal_open", launch)
        self.assertNotIn("RESULT_METHOD=${RESULT_METHOD:-post}", variants)
        self.assertNotIn("RESULT_METHOD=${RESULT_METHOD:-both}", variants)
        self.assertGreaterEqual(
            variants.count("RESULT_METHOD=${RESULT_METHOD:-causal_open}"), 3
        )

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

    def test_burnin_reserves_full_post_horizon(self):
        source = (ROOT / "backdoor.py").read_text(encoding="utf-8")
        self.assertIn(
            "phase_budget = self.post_K + self.post_horizon + 1", source
        )
        self.assertIn(
            "self.post_burnin = min(max(0, configured_burnin), max_burnin)",
            source,
        )


if __name__ == "__main__":
    unittest.main()
