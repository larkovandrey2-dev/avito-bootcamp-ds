#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PipelineConfig
from src.pipeline import check_environment, reproduce


def main():
    parser = argparse.ArgumentParser(description="Воспроизведение final v4 для Avito DS Bootcamp")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "answer_v4.csv")
    parser.add_argument(
        "--check",
        action="store_true",
        help="только проверить данные, модель и кэш",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="переобучить final YetiRank",
    )
    args = parser.parse_args()

    config = PipelineConfig.from_root(ROOT, args.data_dir, args.output)
    if args.check:
        result = check_environment(config)
    else:
        result = reproduce(config, retrain=args.retrain)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
