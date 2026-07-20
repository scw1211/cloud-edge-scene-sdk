"""用途：按风险和不确定性选择语义摘要、压缩特征或局部原始证据。"""

from dataclasses import asdict, dataclass
from typing import Dict, List

from cloud_edge_framework.contracts import EVIDENCE_LEVELS, Evidence, SemanticEvent


LEVEL_INDEX = {name: index for index, name in enumerate(EVIDENCE_LEVELS)}


@dataclass(frozen=True)
class EvidencePlan:
    required_level: str
    included_levels: List[str]
    selected_evidence_ids: List[str]
    payload_bytes: int
    inline_encoded_bytes: int
    referenced_source_bytes: int
    uncompressed_source_bytes: int
    complete: bool
    missing_level: str
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class EvidencePlanner:
    def __init__(self, uncertainty_threshold: float = 0.75) -> None:
        self.uncertainty_threshold = float(uncertainty_threshold)

    def required_level(self, event: SemanticEvent, conflict_suspected: bool = False) -> str:
        if conflict_suspected:
            return "raw"
        if (
            event.risk.level in {"high", "severe"}
            or event.uncertainty.confidence < self.uncertainty_threshold
            or len(event.uncertainty.prediction_set) > 1
        ):
            return "feature"
        return "summary"

    def plan(self, event: SemanticEvent, conflict_suspected: bool = False) -> EvidencePlan:
        required = self.required_level(event, conflict_suspected)
        required_index = LEVEL_INDEX[required]
        included_levels = list(EVIDENCE_LEVELS[: required_index + 1])
        selected: List[Evidence] = [
            item for item in event.evidence if LEVEL_INDEX[item.level] <= required_index
        ]
        present = {item.level for item in event.evidence}
        complete = required in present
        missing = "" if complete else required
        reason_by_level = {
            "summary": "stable low-risk event uses a compact semantic summary",
            "feature": "high-risk or uncertain event adds compressed model features",
            "raw": "a suspected cross-edge conflict requests raw evidence for cloud verification",
        }
        return EvidencePlan(
            required_level=required,
            included_levels=included_levels,
            selected_evidence_ids=[item.evidence_id for item in selected],
            payload_bytes=sum(item.size_bytes for item in selected),
            inline_encoded_bytes=sum(
                item.size_bytes for item in selected if item.inline is not None
            ),
            referenced_source_bytes=sum(
                item.size_bytes for item in selected if item.inline is None
            ),
            uncompressed_source_bytes=sum(
                int(item.codec.get("source_size_bytes", item.size_bytes))
                for item in selected
            ),
            complete=complete,
            missing_level=missing,
            reason=reason_by_level[required],
        )
