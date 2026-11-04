"""CPU-only probe contracts; toy models are not evidence about a checkpoint."""
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None


if torch is not None:
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(2)
            self.dropout = torch.nn.Dropout(.9)
            self.observed = []
            with torch.no_grad():
                self.bn.running_mean.fill_(2.)
                self.bn.running_var.fill_(4.)

        def forward(self, x):
            self.observed.append((self.training, self.dropout.training,
                                  self.bn.training, torch.is_grad_enabled()))
            x = torch.nn.functional.adaptive_avg_pool2d(x[:, :2], (2, 2))
            return [self.dropout(self.bn(x))]

    class Controller(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()
            self.qp_condition = torch.nn.Linear(1, 2)
            self.residual_head = torch.nn.Conv2d(2, 1, 1)
            self.delta_qp_head = torch.nn.Conv2d(2, 1, 1)
            self.backbone_name = "toy_only_not_checkpoint_evidence"
            with torch.no_grad():
                self.qp_condition.weight.fill_(.25)
                self.qp_condition.bias.fill_(.1)
                self.residual_head.weight.fill_(.25)
                self.residual_head.bias.fill_(.2)

        def forward(self, x, q, *, for_encoding=False):
            features = self.backbone(x)[-1]
            features = features + self.qp_condition(q)[..., None, None]
            raw = self.residual_head(features)
            residual = 8 * torch.tanh(torch.nn.functional.interpolate(
                raw, (256, 256), mode="bilinear", align_corners=False))
            return {"residual": residual, "delta_qp": self.delta_qp_head(features)}


class ProbeTests(unittest.TestCase):
    def setUp(self):
        name = "scripts.diagnose_controller_normalization"
        self.assertIsNotNone(importlib.util.find_spec(name), "normalization probe not implemented")
        self.cli = importlib.import_module(name)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def args(self):
        return self.cli.parse_args(["--sequence-dirs", str(self.root / "s1"),
                                   str(self.root / "s2"), "--checkpoint", str(self.root / "c.pt"),
                                   "--output", str(self.root / "new")])

    def test_defaults_and_bounded_arguments(self):
        args = self.args()
        self.assertEqual(args.frame_indices, [1, 16, 33, 40])
        self.assertEqual((args.crf, args.device), (32, "cpu"))
        for field, value in (("sequence_dirs", []), ("sequence_dirs", [Path("s")] * 2),
                             ("sequence_dirs", [Path(str(i)) for i in range(3)]),
                             ("frame_indices", []), ("frame_indices", [1, 1]),
                             ("frame_indices", [0]), ("frame_indices", [True]),
                             ("frame_indices", [1, 2, 3, 4, 5]), ("crf", 52),
                             ("crf", True), ("device", "bogus")):
            with self.subTest(field=field, value=value):
                args = self.args()
                setattr(args, field, value)
                with self.assertRaises(ValueError):
                    self.cli.run(args)
                self.assertFalse(args.output.exists())

    def test_output_overlap_and_overwrite_rejected(self):
        args = self.args()
        for source in [*args.sequence_dirs, args.checkpoint]:
            for output in (source, source / "out", source.parent):
                with self.subTest(output=output):
                    args.output = output
                    with self.assertRaises(ValueError):
                        self.cli.run(args)
        args = self.args()
        args.output.mkdir()
        sentinel = args.output / "keep"
        sentinel.write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.cli.run(args)
        self.assertEqual(sentinel.read_text(), "preserve")

    def test_failure_is_recorded_without_model_load(self):
        args = self.args()
        with patch.object(self.cli, "load_student", side_effect=AssertionError("unexpected load")):
            self.assertEqual(self.cli.run(args), 1)
        report = json.loads((args.output / "controller_normalization.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["training_run"])
        self.assertFalse(report["causal_root_cause_proven"])
        self.assertIn("error", report)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_rgb_preprocessing_preserves_channel_order_and_scale(self):
        rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        rgb[:] = [255, 51, 0]
        x = self.cli.normalize_rgb(rgb, "cpu")
        self.assertEqual(tuple(x.shape), (1, 3, 256, 256))
        self.assertEqual(x.dtype, torch.float32)
        self.assertTrue(torch.all(x[:, 0] == 1))
        self.assertTrue(torch.allclose(x[:, 1], torch.full_like(x[:, 1], .2)))
        self.assertTrue(torch.all(x[:, 2] == 0))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_modes_preserve_every_weight_buffer_and_eval_flag(self):
        model = Controller().eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        x = torch.cat([torch.zeros(1, 3, 256, 256), torch.ones(1, 3, 256, 256)])
        for mode in ("eval", "bn_batchstats", "conditionoff"):
            rows = self.cli.probe_batch(model, x, 32, mode)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["residual_head_raw_shape"], [1, 2, 2])
            self.assertIn("pre_conditioning_features", rows[0])
            self.assertIn("conditioned_features", rows[0])
            self.assertIn("fraction_abs_ge_7_99", rows[0]["residual_256"])
            self.assertEqual(rows[0]["tanh_reconstruction_max_abs_error"], 0)
            self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
            self.assertFalse(any(m.training for m in model.modules()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(item == (False, False, False, False) for item in model.backbone.observed))
        self.assertNotIn("forward", model.backbone.bn.__dict__)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_bn_restores_instance_forward_and_hooks_when_head_raises(self):
        model = Controller().eval()
        original = model.backbone.bn.forward
        model.backbone.bn.forward = original
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with patch.object(model.residual_head, "forward", side_effect=RuntimeError("head failed")):
            with self.assertRaisesRegex(RuntimeError, "head failed"):
                self.cli.probe_batch(model, torch.zeros(2, 3, 256, 256), 32, "bn_batchstats")
        self.assertIs(model.backbone.bn.__dict__["forward"], original)
        self.assertFalse(model.backbone._forward_hooks)
        self.assertFalse(model.residual_head._forward_hooks)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_bn_buffer_references_restore_when_original_bn_forward_raises(self):
        class BrokenBN(torch.nn.BatchNorm2d):
            def forward(self, x):
                super().forward(x)
                raise RuntimeError("BN failed")

        model = Controller().eval()
        bn = model.backbone.bn = BrokenBN(2).eval()
        mean, variance, counter = bn.running_mean, bn.running_var, bn.num_batches_tracked
        with self.assertRaisesRegex(RuntimeError, "BN failed"):
            self.cli.probe_batch(model, torch.zeros(2, 3, 256, 256), 32, "bn_batchstats")
        self.assertIs(bn.running_mean, mean)
        self.assertIs(bn.running_var, variance)
        self.assertIs(bn.num_batches_tracked, counter)
        self.assertEqual(int(counter), 0)
        self.assertTrue(bn.track_running_stats)
        self.assertEqual(bn.momentum, .1)
        self.assertFalse(bn.training)
        self.assertNotIn("forward", bn.__dict__)
        self.assertFalse(model.backbone._forward_hooks)
        self.assertFalse(model.residual_head._forward_hooks)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_composition_affects_batchstats_but_not_eval_with_six_forwards(self):
        model = Controller().eval()
        x = torch.cat([torch.zeros(1, 3, 256, 256), torch.ones(1, 3, 256, 256)])
        report = self.cli.run_probe(model, x, 32)
        rows = report["records"]
        def first(mode, kind):
            return next(r for r in rows if r["mode"] == mode and r["batch_kind"] == kind)["frames"][0]
        self.assertEqual(first("eval", "mixed"), first("eval", "repeated_single"))
        self.assertNotEqual(first("bn_batchstats", "mixed")["pre_conditioning_features"]["mean"],
                            first("bn_batchstats", "repeated_single")["pre_conditioning_features"]["mean"])
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(model.backbone.observed), 6)
        self.assertTrue(all(row["batch_size"] == 2 for row in rows))
        self.assertTrue(all(row["source_indices"] == [0, 0] for row in rows
                            if row["batch_kind"] == "repeated_single"))
        self.assertTrue(report["state_unchanged"])
        json.dumps(report, allow_nan=False)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_zero_conditioning_has_zero_effect_and_nonzero_conditioning_is_measured(self):
        model = Controller().eval()
        x = torch.zeros(2, 3, 256, 256)
        on = self.cli.probe_batch(model, x, 32, "eval")[0]
        off = self.cli.probe_batch(model, x, 32, "conditionoff")[0]
        self.assertEqual(on["pre_conditioning_features"], off["pre_conditioning_features"])
        self.assertNotEqual(on["residual_head_pre_resize_pre_tanh"], off["residual_head_pre_resize_pre_tanh"])
        with torch.no_grad():
            model.qp_condition.weight.zero_()
            model.qp_condition.bias.zero_()
        self.assertEqual(self.cli.probe_batch(model, x, 32, "eval"),
                         self.cli.probe_batch(model, x, 32, "conditionoff"))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_bn_subclass_activation_is_not_removed_and_children_stay_eval(self):
        class BatchNormAct(torch.nn.BatchNorm2d):
            def __init__(self):
                super().__init__(2)
                self.act = torch.nn.ReLU()
                self.drop = torch.nn.Dropout(.99)
                self.flags = []

            def forward(self, x):
                self.flags.append((self.training, self.act.training, self.drop.training))
                return self.act(self.drop(super().forward(x)))

        model = Controller().eval()
        model.backbone.bn = BatchNormAct().eval()
        original_mean = model.backbone.bn.running_mean
        original_var = model.backbone.bn.running_var
        x = torch.cat([torch.zeros(1, 3, 256, 256), torch.ones(1, 3, 256, 256)])
        rows = self.cli.probe_batch(model, x, 32, "bn_batchstats")
        self.assertEqual(rows[0]["pre_conditioning_features"]["min"], 0.)
        self.assertGreater(rows[1]["pre_conditioning_features"]["min"], .99)
        self.assertEqual(model.backbone.bn.flags, [(False, False, False)])
        self.assertIs(model.backbone.bn.running_mean, original_mean)
        self.assertIs(model.backbone.bn.running_var, original_var)
        self.assertTrue(model.backbone.bn.track_running_stats)
        self.assertFalse(model.backbone.bn.training)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_state_mutation_fails_instead_of_silently_copying_back(self):
        model = Controller().eval()
        def mutate(_module, _inputs, _output):
            model.backbone.bn.running_mean.add_(1)
        handle = model.residual_head.register_forward_hook(mutate)
        self.addCleanup(handle.remove)
        with self.assertRaisesRegex(RuntimeError, "state.*mutat"):
            self.cli.probe_batch(model, torch.zeros(2, 3, 256, 256), 32, "eval")
        self.assertTrue(torch.all(model.backbone.bn.running_mean == 3))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_probe_rejects_training_and_unbounded_batches(self):
        model = Controller().eval()
        model.backbone.dropout.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            self.cli.probe_batch(model, torch.zeros(2, 3, 256, 256), 32, "eval")
        model.eval()
        for n in (0, 9):
            with self.assertRaises(ValueError):
                self.cli.run_probe(model, torch.zeros(n, 3, 256, 256), 32)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_run_records_selected_identities_without_timm_or_large_arrays(self):
        args = self.args()
        args.frame_indices = [1, 16]
        args.checkpoint.write_bytes(b"fixture checkpoint identity only")
        expected = []
        for source in args.sequence_dirs:
            source.mkdir()
            (source / "seqinfo.ini").write_text("fixture metadata")
            for index in args.frame_indices:
                path = source / f"{index:06d}.png"
                path.write_bytes(bytes([index]))
                expected.append((path, self.cli.sha256(path)))
        def selected(source, indices):
            for index in indices:
                path = source / f"{index:06d}.png"
                yield index, path, self.cli.sha256(path), np.full((4, 6, 3), index, np.uint8)
        model = Controller().eval()
        student = SimpleNamespace(model=model, device="cpu")
        with patch.object(self.cli, "selected_frames", selected), patch.object(self.cli, "load_student", return_value=student):
            self.assertEqual(self.cli.run(args), 0)
        report = json.loads((args.output / "controller_normalization.json").read_text())
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["checkpoint_sha256"], self.cli.sha256(args.checkpoint))
        self.assertEqual([(row["source_path"], row["source_sha256"]) for row in report["selected_frames"]],
                         [(str(path.resolve()), digest) for path, digest in expected])
        self.assertEqual([row["frame_index"] for row in report["selected_frames"]], [1, 16, 1, 16])
        self.assertEqual(report["batch_size"], 4)
        self.assertTrue(report["state_unchanged"])
        self.assertFalse(report["causal_root_cause_proven"])
        self.assertIn("intentional_ablation_not_deployment", report["conditionoff"])
        self.assertLess(len(json.dumps(report)), 120000)


if __name__ == "__main__":
    unittest.main()
