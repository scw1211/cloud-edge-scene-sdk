#!/usr/bin/env python3
"""真实云端发布到边缘回滚确认闭环入口。"""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from edge_llm_factory.distributed_update_evidence import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
