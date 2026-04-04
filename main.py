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
    parser.add_argument("--judge_model_id", required=True)
    parser.add_argument("--language", choices=["en", "es", "ca"], default="en")
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--timeout_seconds", type=int, default=60)
    args = parser.parse_args()

    if args.experiment == "exp1_factual_tof":
        result = run_false_presupposition_tof_experiment(
            model_id=args.model_id,
            judge_model_id=args.judge_model_id,
            language=args.language,
            max_questions=args.max_questions,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            timeout_seconds=args.timeout_seconds,
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