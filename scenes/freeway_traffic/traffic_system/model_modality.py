"""用途：阻止视觉、视频或音频权重进入边缘文本模型及其导出包。"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple


MULTIMODAL_MARKERS = ("vision", "visual", "image", "video", "audio", "speech")
MULTIMODAL_TOKENIZER_KEYS = {
    "audio_bos_token",
    "audio_eos_token",
    "audio_token",
    "image_token",
    "video_token",
    "vision_bos_token",
    "vision_eos_token",
    "model_specific_special_tokens",
}
TEXT_ONLY_CHAT_TEMPLATE = """{%- for message in messages %}
{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}
"""

def inspect_loaded_model(model: Any) -> Dict[str, Any]:
    total_parameters = 0
    multimodal_parameters = 0
    multimodal_names = []
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total_parameters += count
        if any(marker in name.lower() for marker in MULTIMODAL_MARKERS):
            multimodal_parameters += count
            multimodal_names.append(name)
    return {
        "model_type": str(getattr(model.config, "model_type", "unknown")),
        "total_parameters": total_parameters,
        "multimodal_parameters": multimodal_parameters,
        "multimodal_parameter_names": multimodal_names,
        "deployment_modality": "text_only" if not multimodal_names else "multimodal",
    }


def require_text_only_model(model: Any) -> Dict[str, Any]:
    report = inspect_loaded_model(model)
    if report["multimodal_parameter_names"]:
        preview = ", ".join(report["multimodal_parameter_names"][:5])
        raise RuntimeError("Multimodal parameters are forbidden in the edge model: {}".format(preview))
    return report


def sanitize_tokenizer_export(output_dir: Path) -> Tuple[str, ...]:
    config_path = output_dir / "tokenizer_config.json"
    removed = []
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for key in sorted(MULTIMODAL_TOKENIZER_KEYS):
            if key in config:
                config.pop(key)
                removed.append(key)
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (output_dir / "chat_template.jinja").write_text(
        TEXT_ONLY_CHAT_TEMPLATE,
        encoding="utf-8",
    )
    return tuple(removed)
