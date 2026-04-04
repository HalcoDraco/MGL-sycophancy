import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()],
)

from dotenv import load_dotenv
import argparse
import csv
from pathlib import Path

from src.exp1_factual_tof.run_exp1 import run_false_presupposition_tof_experiment
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Run MGL sycophancy experiments from a single entrypoint.")
    parser.add_argument("--experiment", choices=["exp1_factual_tof"], default="exp1_factual_tof")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--language", choices=["en", "es", "ca"], default="en")
    args = parser.parse_args()

    if args.experiment == "exp1_factual_tof":
        result = run_false_presupposition_tof_experiment(
            model_id=args.model_id,
            language=args.language,
            max_questions=10
        )

        repo_root = Path(__file__).resolve().parent
        results_dir = repo_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / "exp1.csv"

        write_header = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["model_id", "language", "avg_tof", "std_tof"])
            writer.writerow([
                result["model_id"],
                result["language"],
                result["average_tof"],
                result["std_tof"],
            ])

if __name__ == "__main__":
    main()