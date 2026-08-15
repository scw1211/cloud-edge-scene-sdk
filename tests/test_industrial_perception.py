import struct
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace
from unittest import mock

import numpy as np

from scenes.industrial_anomaly.perception import patchcore_onnx as perception
from scenes.jetson_infra_trt_runtime_cublas.tools import verify_metrics as infra_metrics
from scenes.jetson_rgb_trt_runtime_cublas.tools import verify_metrics as rgb_metrics


class IndustrialPerceptionTests(unittest.TestCase):
    def test_pcbank_round_trip_matches_cpp_contract(self):
        patches = np.arange(5 * 384, dtype=np.float32).reshape(5, 384) / 100.0
        with tempfile.TemporaryDirectory(prefix="industrial-pcbank-") as directory:
            path = Path(directory) / "capsule.pcbank"
            perception.write_pcbank(path, patches, 1.25, 2.5)
            raw = path.read_bytes()
            magic, version, rows, cols, mean, stddev = perception.HEADER.unpack_from(raw)
            restored, restored_mean, restored_stddev = perception.read_pcbank(path)
        self.assertEqual(magic, b"PCBNK01\0")
        self.assertEqual(version, 1)
        self.assertEqual((rows, cols), (5, 384))
        self.assertAlmostEqual(mean, 1.25)
        self.assertAlmostEqual(stddev, 2.5)
        self.assertAlmostEqual(restored_mean, 1.25)
        self.assertAlmostEqual(restored_stddev, 2.5)
        np.testing.assert_allclose(restored, patches.astype(np.float16), atol=0)

    def test_patchcore_score_is_zero_for_exact_bank_patch(self):
        feature_map = np.zeros((384, 20, 20), dtype=np.float32)
        bank = np.zeros((3, 384), dtype=np.float32)
        bank[1] = 1.0
        bank[2] = 2.0
        patches, distances, indices = perception.patchcore_distances(
            feature_map, bank, 0.0, 1.0
        )
        score = perception.patchcore_image_score(patches, distances, indices, bank)
        self.assertTrue(np.all(distances == 0.0))
        self.assertEqual(score, 0.0)

    def test_kcenter_is_unique_and_deterministic(self):
        rng = np.random.RandomState(7)
        features = rng.normal(size=(40, 8)).astype(np.float32)
        first = perception.select_kcenter(features, 7, epsilon=0.9, seed=42)
        second = perception.select_kcenter(features, 7, epsilon=0.9, seed=42)
        self.assertEqual(len(set(first.tolist())), 7)
        np.testing.assert_array_equal(first, second)

    def test_rejects_truncated_bank(self):
        with tempfile.TemporaryDirectory(prefix="industrial-pcbank-") as directory:
            path = Path(directory) / "bad.pcbank"
            path.write_bytes(
                perception.HEADER.pack(
                    perception.MAGIC, perception.VERSION, 10, 384, 0.0, 1.0
                )
            )
            with self.assertRaisesRegex(ValueError, "size"):
                perception.read_pcbank(path)

    def test_infrared_float_heatmap_blur_is_finite(self):
        distances = np.linspace(0.0, 1.0, 400, dtype=np.float32)
        result = perception._upsample_map(distances, blur_radius=4.0)
        self.assertEqual(result.shape, (160, 160))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())

    def test_cpp_producers_use_explicit_optional_edge_url_and_stable_pair_id(self):
        root = Path(__file__).resolve().parents[1]
        for modality in ("rgb", "infra"):
            source = (
                root
                / "scenes"
                / "jetson_{}_trt_runtime_cublas".format(modality)
                / "src"
                / "main.cpp"
            ).read_text(encoding="utf-8")
            cmake = (
                root
                / "scenes"
                / "jetson_{}_trt_runtime_cublas".format(modality)
                / "CMakeLists.txt"
            ).read_text(encoding="utf-8")
            self.assertIn('std::getenv("PATCHCORE_EDGE_URL")', source)
            self.assertIn('"PATCHCORE_RUN_ID"', source)
            self.assertIn('environment_or("PATCHCORE_PRODUCT", "capsule")', source)
            self.assertIn('parent_name(source_path)', source)
            self.assertIn("json_escape(product)", source)
            self.assertNotIn("YAML::", source)
            self.assertNotIn("curl_", source)
            self.assertNotIn("yaml-cpp", cmake)
            self.assertNotIn(" curl", cmake)

    def test_metric_verifiers_remap_target_test_paths(self):
        target = "/opt/mulsen/capsule/RGB/test/broken_inside/0.png"
        expected = "/local/capsule/RGB/test/broken_inside/0.png"
        source_root = "/local/capsule/RGB/test"
        self.assertEqual(rgb_metrics.remap_source(target, source_root), expected)
        self.assertEqual(infra_metrics.remap_source(target, source_root), expected)

    def test_metric_verifiers_reject_unmappable_target_paths(self):
        for module in (rgb_metrics, infra_metrics):
            with self.assertRaisesRegex(ValueError, "no test component"):
                module.remap_source("/tmp/0.png", "/local/test")

    def test_infer_event_can_publish_edge_visible_heatmap_uri(self):
        with tempfile.TemporaryDirectory(prefix="industrial-event-") as directory:
            root = Path(directory)
            image = root / "0.png"
            image.write_bytes(b"image-placeholder")
            bank = root / "bank.pcbank"
            perception.write_pcbank(
                bank, np.zeros((1, 384), dtype=np.float32), 0.0, 1.0
            )
            model = root / "model.onnx"
            model.write_bytes(b"model-placeholder")
            heatmap = root / "heatmap.f32"
            output = root / "event.json"
            extractor = mock.Mock()
            extractor.infer.return_value = (
                np.zeros((384, 20, 20), dtype=np.float32),
                1.0,
                2.0,
            )
            args = Namespace(
                image=str(image),
                bank=str(bank),
                onnx=str(model),
                modality="rgb",
                heatmap=str(heatmap),
                heatmap_uri="file:///edge-visible/heatmap.f32",
                sample_id="capsule_good_0",
                event_id="event-1",
                edge_id="edge-1",
                product="capsule",
                output=str(output),
            )
            with mock.patch.object(
                perception, "OnnxFeatureExtractor", return_value=extractor
            ):
                event = perception.infer_event(args)
            self.assertEqual(
                event["data"]["heatmap_uri"], "file:///edge-visible/heatmap.f32"
            )
            self.assertEqual(event["data"]["heatmap_size_bytes"], 160 * 160 * 4)


if __name__ == "__main__":
    unittest.main()
