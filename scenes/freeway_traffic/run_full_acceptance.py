"""用途：从独立 SDK 一键运行带真实权重的交通云边完整验收。"""

import argparse
import os
from pathlib import Path
import sys


SCENE_ROOT = Path(__file__).resolve().parent
SDK_ROOT = SCENE_ROOT.parents[1]
for import_root in (SDK_ROOT, SCENE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def _required_path(relative_name):
    path = SCENE_ROOT / relative_name
    if not path.is_file():
        raise FileNotFoundError(
            "{} 不存在；请先运行 install_full_assets.py --edge".format(path)
        )
    return path


def main():
    parser = argparse.ArgumentParser(description="运行真实交通系统完整验收")
    parser.add_argument(
        "--samples",
        default="auto",
        help="auto 为在测试集均匀取样；也可传列表或 start:end:step。",
    )
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--edge-llm-mode",
        choices=("selective", "primary"),
        default="selective",
    )
    parser.add_argument("--llama-binary", required=True)
    parser.add_argument("--llama-gpu-layers", type=int, default=99)
    parser.add_argument("--with-cloud-qwen9b", action="store_true")
    parser.add_argument(
        "--output",
        default=str(SCENE_ROOT / "evidence" / "traffic_full_acceptance_latest.json"),
    )
    parser.add_argument(
        "--report",
        default=str(SCENE_ROOT / "evidence" / "traffic_full_acceptance_latest.md"),
    )
    args = parser.parse_args()

    data_npz = _required_path(
        "assets/downloads/PEMS08_r1_d0_w0_astcgn_multitask.npz"
    )
    checkpoint = _required_path(
        "assets/models/joint_risk_astgcn_metis4_flowprio2_frozen.pt"
    )
    risk_calibrator = _required_path("assets/models/region_risk_conformal.json")
    release_registry = _required_path("runtime/edge_llm_release_store.json")

    forwarded = [
        "accept_traffic_framework",
        "--project-root",
        str(SDK_ROOT),
        "--config",
        str(SCENE_ROOT / "configurations" / "PEMS08_astgcn.conf"),
        "--data-npz",
        str(data_npz),
        "--checkpoint",
        str(checkpoint),
        "--risk-calibrator",
        str(risk_calibrator),
        "--edge-plugin-config",
        str(SCENE_ROOT / "deployment" / "full" / "scene_plugins_edge.json"),
        "--cloud-plugin-config",
        str(SCENE_ROOT / "deployment" / "full" / "scene_plugins_cloud.json"),
        "--release-registry",
        str(release_registry),
        "--edge-llm-runtime-config",
        str(SCENE_ROOT / "deployment" / "full" / "edge_llm_runtime.json"),
        "--samples",
        str(args.samples),
        "--sample-count",
        str(args.sample_count),
        "--device",
        str(args.device),
        "--torch-threads",
        str(args.torch_threads),
        "--edge-llm-mode",
        str(args.edge_llm_mode),
        "--llama-binary",
        str(Path(args.llama_binary).resolve()),
        "--llama-gpu-layers",
        str(args.llama_gpu_layers),
        "--output",
        str(Path(args.output).resolve()),
        "--report",
        str(Path(args.report).resolve()),
    ]
    # Selective mode may correctly choose zero events on an ordinary sample set.
    # Requiring at least one invocation would bias the dataset or turn a routing
    # outcome into a false deployment failure.  Primary mode, by definition,
    # must exercise Edge-Qwen on measured events.
    if args.edge_llm_mode == "primary":
        forwarded.append("--require-edge-llm")
    if args.with_cloud_qwen9b:
        forwarded.extend(
            [
                "--cloud-llm-runtime-config",
                str(
                    SCENE_ROOT
                    / "deployment"
                    / "full"
                    / "cloud_qwen9b_ollama.json"
                ),
                "--request-timeout-seconds",
                "15",
                "--cloud-timeout-seconds",
                "12",
            ]
        )

    from traffic_system.accept_traffic_framework import main as acceptance_main

    os.chdir(str(SDK_ROOT))
    sys.argv = forwarded
    acceptance_main()


if __name__ == "__main__":
    main()
