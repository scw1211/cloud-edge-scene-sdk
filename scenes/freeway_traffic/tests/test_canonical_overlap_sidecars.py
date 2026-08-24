"""Canonical overlap subscriptions keep both road-set owners bit-identical."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TRAFFIC_ROOT = REPOSITORY_ROOT / "scenes" / "freeway_traffic"
for import_root in (REPOSITORY_ROOT, TRAFFIC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from prepare_metis_partition_data import prepare_partition_data  # noqa: E402
from traffic_system.current_state_perception_runtime import (  # noqa: E402
    CurrentStateTrafficPerceptionRuntime,
    PartitionCurrentStateTrafficPerceptionRuntime,
)
from traffic_system.overlap_observations import (  # noqa: E402
    OVERLAP_PREPROCESS_VERSION,
)


PARTITION_SIZES = [43, 42, 42, 43]


def _partitions():
    result = []
    offset = 0
    for size in PARTITION_SIZES:
        result.append(list(range(offset, offset + size)))
        offset += size
    return result


def _rule_config(partitions):
    return {
        "reference_speed": 68.0,
        "congestion_speed_ratio": 0.8,
        "risk_score_centers": [0.1, 0.37, 0.63, 0.9],
        "risk_score_width": 0.18,
        "risk_weights": {
            "mean_speed_pressure": 0.3,
            "minimum_speed_pressure": 0.15,
            "congestion_duration": 0.25,
            "occupancy_pressure": 0.2,
            "recent_speed_drop": 0.1,
        },
        "partitions": partitions,
    }


def _topology(partitions):
    pair_nodes = [
        (0, 1, [0, 1], [43, 44]),
        (0, 3, [2, 3], [127, 128]),
        (1, 2, [45, 46], [85, 86]),
        (1, 3, [47, 48], [129, 130]),
        (2, 3, [87, 88], [131, 132]),
    ]
    pairs = []
    for left, right, left_nodes, right_nodes in pair_nodes:
        pairs.append(
            {
                "left_region": "region_{}".format(left),
                "right_region": "region_{}".format(right),
                "cut_edge_count": len(left_nodes),
                "left_boundary_nodes": left_nodes,
                "right_boundary_nodes": right_nodes,
            }
        )
    return {
        "schema_version": 1,
        "method": "synthetic_frozen_cut_edges",
        "region_boundary_nodes": {
            "region_{}".format(index): nodes[:4]
            for index, nodes in enumerate(partitions)
        },
        "region_pairs": pairs,
    }


class CanonicalOverlapSidecarTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_path = self.root / "source.npz"
        self.rule_path = self.root / "rules.json"
        self.topology_path = self.root / "topology.json"
        self.output = self.root / "shards"
        partitions = _partitions()
        rng = np.random.default_rng(20260824)
        arrays = {
            "train_x": rng.normal(size=(2, 170, 3, 12)).astype(np.float32),
            "val_x": rng.normal(size=(1, 170, 3, 12)).astype(np.float32),
            "test_x": rng.normal(size=(3, 170, 3, 12)).astype(np.float32),
            "mean": np.zeros((1, 1, 3, 1), dtype=np.float32),
            "std": np.ones((1, 1, 3, 1), dtype=np.float32),
            "feature_names": np.asarray(["flow", "occupancy", "speed"]),
        }
        for key in ("train_x", "val_x", "test_x"):
            arrays[key][:, :, 0, :] = 200.0 + arrays[key][:, :, 0, :]
            arrays[key][:, :, 1, :] = (
                0.1 + arrays[key][:, :, 1, :] * 0.01
            )
            arrays[key][:, :, 2, :] = 55.0 + arrays[key][:, :, 2, :]
        np.savez_compressed(self.source_path, **arrays)
        self.rule_path.write_text(
            json.dumps(_rule_config(partitions)), encoding="utf-8"
        )
        self.topology_path.write_text(
            json.dumps(_topology(partitions)), encoding="utf-8"
        )
        self.prepared = prepare_partition_data(
            self.source_path,
            self.rule_path,
            self.topology_path,
            self.output,
        )
        self.manifest_path = self.output / "manifest.json"

    def tearDown(self):
        self.temporary.cleanup()

    def _runtime(self, partition_id, verify_sha256=True):
        return PartitionCurrentStateTrafficPerceptionRuntime(
            self.manifest_path,
            partition_id,
            self.rule_path,
            self.topology_path,
            split="test",
            top_k=10,
            verify_sha256=verify_sha256,
        )

    def test_manifest_v2_has_four_primary_shards_and_five_sidecars(self):
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(2, manifest["schema_version"])
        self.assertEqual(
            PARTITION_SIZES,
            [record["node_count"] for record in manifest["partitions"]],
        )
        self.assertEqual(5, len(manifest["road_sets"]))
        contract = manifest["overlap_contract"]
        self.assertEqual(5, contract["subscription_count"])
        self.assertEqual(2, contract["owners_per_subscription"])
        self.assertEqual(
            OVERLAP_PREPROCESS_VERSION, contract["preprocess_version"]
        )
        self.assertFalse(contract["primary_sensor_ownership_changed"])
        self.assertFalse(contract["sidecar_nodes_are_managed_nodes"])
        for record in manifest["road_sets"]:
            self.assertEqual(2, len(record["required_regions"]))
            self.assertEqual(2, len(record["required_partition_ids"]))
            self.assertEqual(2, len(record["required_partitions"]))
            self.assertEqual(
                sorted(set(record["global_node_ids"])),
                record["global_node_ids"],
            )
            self.assertEqual(
                (self.output / record["file"]).stat().st_size,
                record["bytes"],
            )

    def test_full_and_four_partition_observation_digests_match(self):
        full_result = CurrentStateTrafficPerceptionRuntime(
            self.source_path,
            self.rule_path,
            self.topology_path,
            split="test",
            top_k=10,
        ).infer_sample(1)
        full_observations = {}
        for event in full_result.events:
            self.assertEqual(
                PARTITION_SIZES[event["partition_id"]],
                len(event["managed_node_ids"]),
            )
            for observation in event["road_set_overlap_observations"]:
                full_observations[observation["road_set_id"]] = observation
        self.assertEqual(5, len(full_observations))

        owner_observations = {}
        sidecar_paths = {}
        for partition_id in range(4):
            runtime = self._runtime(partition_id)
            event = runtime.infer_sample(1).events[0]
            self.assertEqual(
                _partitions()[partition_id], event["managed_node_ids"]
            )
            self.assertEqual(
                runtime.managed_node_ids, _partitions()[partition_id]
            )
            for subscription in runtime.overlap_subscriptions:
                road_set_id = subscription["road_set"]["road_set_id"]
                sidecar_paths.setdefault(road_set_id, set()).add(
                    subscription["file"]
                )
            for observation in event["road_set_overlap_observations"]:
                road_set_id = observation["road_set_id"]
                owner_observations.setdefault(road_set_id, []).append(
                    observation
                )

        self.assertEqual(set(full_observations), set(owner_observations))
        for road_set_id, observations in owner_observations.items():
            self.assertEqual(2, len(observations))
            self.assertEqual(observations[0], observations[1])
            self.assertEqual(full_observations[road_set_id], observations[0])
            self.assertEqual(1, len(sidecar_paths[road_set_id]))
            for field in (
                "source_content_sha256",
                "raw_fragment_sha256",
                "output_digest",
                "action_digest",
            ):
                self.assertEqual(
                    full_observations[road_set_id][field],
                    observations[0][field],
                )

    def test_overlap_sha_is_enforced_when_primary_sha_check_is_skipped(self):
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        record = manifest["road_sets"][0]
        sidecar_path = self.output / record["file"]
        with sidecar_path.open("ab") as file_obj:
            file_obj.write(b"tamper")
        record["bytes"] = sidecar_path.stat().st_size
        self.manifest_path.write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "overlap sidecar SHA-256"):
            self._runtime(record["required_partition_ids"][0], verify_sha256=False)

    def test_missing_or_drifted_overlap_contract_fails_fast(self):
        original = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        mutations = {
            "missing_file": lambda value: value["road_sets"][0].pop("file"),
            "missing_bytes": lambda value: value["road_sets"][0].pop("bytes"),
            "missing_sha": lambda value: value["road_sets"][0].pop("sha256"),
            "topology": lambda value: value["overlap_contract"].update(
                {"topology_sha256": "0" * 64}
            ),
            "preprocess": lambda value: value["overlap_contract"].update(
                {"preprocess_version": "drifted"}
            ),
            "owner_mapping": lambda value: value["road_sets"][0].update(
                {"required_partition_ids": [0, 2]}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                candidate = copy.deepcopy(original)
                mutate(candidate)
                self.manifest_path.write_text(
                    json.dumps(candidate), encoding="utf-8"
                )
                with self.assertRaises((ValueError, KeyError, FileNotFoundError)):
                    self._runtime(0)
        self.manifest_path.write_text(json.dumps(original), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
