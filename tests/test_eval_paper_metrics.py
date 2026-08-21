import ast
import math
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "eval_backdoor.py"
TRAINER = ROOT / "backdoor.py"


def load_metric_functions():
    tree = ast.parse(EVALUATOR.read_text(encoding="utf-8"))
    names = {"_return_tdr", "_success_tdr", "_bootstrap_tdr_ci"}
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"np": np}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(EVALUATOR), "exec"),
        namespace,
    )
    return namespace


class PaperMetricTest(unittest.TestCase):
    def test_tdr_definitions(self):
        metrics = load_metric_functions()
        self.assertAlmostEqual(metrics["_return_tdr"](100.0, 25.0), 0.75)
        self.assertEqual(metrics["_return_tdr"](100.0, 125.0), 0.0)
        self.assertAlmostEqual(metrics["_success_tdr"](0.8, 0.2), 0.75)
        self.assertTrue(math.isnan(metrics["_success_tdr"](0.0, 0.0)))

    def test_scenario_b_and_post_e_export_contract(self):
        evaluator = EVALUATOR.read_text(encoding="utf-8")
        trainer = TRAINER.read_text(encoding="utf-8")
        self.assertIn('requested_p_keys = [str(step) for step in range(1, 9)]', evaluator)
        self.assertIn('"evaluation_p0": 1', evaluator)
        self.assertIn('"paper_metric_bundle": [', evaluator)
        self.assertIn('returns=pre_returns + window_returns + post_returns', trainer)
        self.assertIn('success=success if has_success else None', trainer)


if __name__ == "__main__":
    unittest.main()
