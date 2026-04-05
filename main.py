import logging

from dotenv import load_dotenv
import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.exp1_factual_tof.run_exp1 import run_false_presupposition_tof_experiment
load_dotenv()


def configure_app_logging() -> None:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Keep third-party library logs quiet unless they are warnings/errors.
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)

    # Emit only this project's logs at INFO level.
    app_logger = logging.getLogger("src")
    app_logger.handlers.clear()
    app_logger.setLevel(logging.DEBUG)
    app_logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    app_logger.addHandler(stream_handler)


EXP1_COLUMNS = [
    "model_id",
    "language",
    "avg_tof",
    "std_tof",
    "completed_questions",
]


def _upsert_exp1_result(csv_path: Path, result: Dict[str, Any]) -> None:
    new_completed: int = int(result.get("num_questions", 0))
    new_row = {
        "model_id": str(result["model_id"]),
        "language": str(result["language"]),
        "avg_tof": float(result["average_tof"]),
        "std_tof": float(result["std_tof"]),
        "completed_questions": new_completed,
    }

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=EXP1_COLUMNS)

    key_mask = (df["model_id"] == new_row["model_id"]) & (df["language"] == new_row["language"])
    if key_mask.any():
        for col, value in new_row.items():
            df.loc[key_mask, col] = value
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df = df[EXP1_COLUMNS].sort_values(["model_id", "language"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)

def main():
    configure_app_logging()

    parser = argparse.ArgumentParser(description="Run MGL sycophancy experiments from a single entrypoint.")
    parser.add_argument("--experiment", choices=["exp1_factual_tof"], default="exp1_factual_tof")
    parser.add_argument("--model_id", default="Qwen3-8B")
    parser.add_argument("--language", choices=["en", "es", "ca"], default="en")
    args = parser.parse_args()

    if args.experiment == "exp1_factual_tof":
        result = run_false_presupposition_tof_experiment(
            model_id=args.model_id,
            language=args.language,
            max_questions=7
        )

        repo_root = Path(__file__).resolve().parent
        results_dir = repo_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / "exp1.csv"

        _upsert_exp1_result(csv_path=csv_path, result=result)

if __name__ == "__main__":
    main()