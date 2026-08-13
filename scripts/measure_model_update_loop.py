#!/usr/bin/env python3
"""正式模型更新闭环取证入口。"""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from edge_llm_factory.update_loop_evidence import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
