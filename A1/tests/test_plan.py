"""Kiểm tra hàng đợi Colab và chọn LR (không cần torch)."""
import json
import os
import tempfile
import unittest

from src.plan import next_job, plan, run_dir, selected_dirs, selected_lr


def finish(root, name, val):
    os.makedirs(os.path.join(root, name))
    with open(os.path.join(root, name, "metrics.json"), "w") as stream:
        json.dump({"best_val_top1": val}, stream)


class PlanTests(unittest.TestCase):
    def test_run_dir_matches_train_naming(self):
        self.assertEqual(run_dir("R4", 42, 1e-4), "R4_s42")
        self.assertEqual(run_dir("R4", 42, None), "R4_s42")
        self.assertEqual(run_dir("R4", 42, 5e-5), "R4_s42_lr5e-05")
        self.assertEqual(run_dir("R3", 43, 3e-4), "R3_s43_lr0.0003")

    def test_tuned_runs_wait_for_sweep(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(selected_lr(root))
            self.assertIsNone(selected_dirs(root))
            states = {row["dir"]: row["state"] for row in plan(root)}
            self.assertEqual(states["R3_s42"], "chờ dò LR")
            self.assertEqual(next_job(root)["dir"], "R2_s42")

    def test_sweep_selects_best_validation_lr_per_model(self):
        with tempfile.TemporaryDirectory() as root:
            for name, val in [("R4_s42_lr5e-05", .70), ("R4_s42", .75), ("R4_s42_lr0.0003", .78),
                              ("R8_s42_lr5e-05", .80), ("R8_s42", .85), ("R8_s42_lr0.0003", .83)]:
                finish(root, name, val)
            self.assertEqual(selected_lr(root, write=True), {"resnet50": 3e-4, "deit_s": 1e-4})
            self.assertTrue(os.path.isfile(os.path.join(root, "lr_selection.json")))
            states = {row["dir"]: row["state"] for row in plan(root)}
            self.assertEqual(states["R3_s42_lr0.0003"], "chưa chạy")
            self.assertEqual(states["R4_s43_lr0.0003"], "chưa chạy")
            self.assertIn("R7_s42", states)
            self.assertNotIn("R3_s42", states)
            chosen = selected_dirs(root)
            self.assertIn("R4_s42_lr0.0003", chosen)
            self.assertNotIn("R4_s42", chosen)
            self.assertIn("R8_s42", chosen)


if __name__ == "__main__":
    unittest.main()
