import json
import logging
import os
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.github_models_api import github_models_chat
from src.exp1_factual_tof.false_presupposition_judge import judge_false_presupposition_response
from src.exp1_factual_tof.data_loader import load_samples, resolve_repo_and_data_dir
from src.exp1_factual_tof.pushback_generator import generate_single_pushback

logger = logging.getLogger(__name__)


def _build_summary_payload(
    model_id: str,
    judge_model_id: str,
    pushback_generator_model_id: str,
    language: str,
    num_questions: int,
    tof_values: List[int],
    all_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    avg_tof = sum(tof_values) / len(tof_values) if tof_values else 0.0
    std_tof = statistics.pstdev(tof_values) if tof_values else 0.0

    return {
        "model_id": model_id,
        "judge_model_id": judge_model_id,
        "pushback_generator_model_id": pushback_generator_model_id,
        "language": language,
        "num_questions": num_questions,
        "average_tof": avg_tof,
        "std_tof": std_tof,
        "tof_values": tof_values,
        "results": all_rows,
    }


def run_false_presupposition_tof_experiment(
    model_id: str,
    judge_model_id: str = "openai/gpt-4o",
    pushback_generator_model_id: str = "openai/gpt-4o",
    language: str = "en",
    max_questions: Optional[int] = None,
    github_pat: Optional[str] = None,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    """
    Run a 5-turn Turn-of-Flip (ToF) experiment on false-presupposition data.

    Turn design per question:
    - Turn 1: original question
    - Turns 2-5: increasingly aggressive pushbacks

    ToF is defined as the number of consecutive turns from turn 1 where the model
    correctly rejects/corrects the false presupposition before the first failure.

    Returns a dictionary with per-question outcomes and aggregate statistics.
    """
    token = github_pat or os.getenv("GITHUB_PAT")
    if not token:
        raise ValueError("Missing GitHub token. Set GITHUB_PAT or pass github_pat.")

    repo_root, base_data_dir = resolve_repo_and_data_dir(data_dir)
    samples = load_samples(base_data_dir, language, max_questions)

    logger.info(
        "Running exp1 factual ToF experiment | model=%s | language=%s | samples=%d",
        model_id,
        language,
        len(samples),
    )

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_id.replace("/", "__")
    safe_judge = judge_model_id.replace("/", "__")
    safe_pushback = pushback_generator_model_id.replace("/", "__")
    run_prefix = f"{language}__{safe_model}__judge_{safe_judge}__pushback_{safe_pushback}__"

    if output_dir is None:
        output_base = repo_root / "src" / "exp1_factual_tof" / "results"
    else:
        output_base = Path(output_dir)
        if not output_base.is_absolute():
            output_base = repo_root / output_base

    run_output_dir: Optional[Path] = None
    existing_summary: Optional[Dict[str, Any]] = None
    candidate_dirs = sorted(
        [d for d in output_base.glob(f"{run_prefix}*") if d.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )

    for candidate in candidate_dirs:
        candidate_summary_path = candidate / "summary.json"
        if not candidate_summary_path.exists():
            continue
        try:
            with candidate_summary_path.open("r", encoding="utf-8") as f:
                loaded_summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Skipping unreadable summary file at %s", candidate_summary_path)
            continue

        loaded_results = loaded_summary.get("results", [])
        if not isinstance(loaded_results, list):
            logger.warning("Skipping invalid summary format at %s", candidate_summary_path)
            continue

        run_output_dir = candidate
        existing_summary = loaded_summary
        logger.info(
            "Resuming existing exp1 run from %s | completed_samples=%d",
            run_output_dir,
            len(loaded_results),
        )
        break

    if run_output_dir is None:
        run_output_dir = output_base / f"{run_prefix}{run_ts}"
        logger.info("Starting new exp1 run at %s", run_output_dir)

    run_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_output_dir / "summary.json"

    all_rows = []
    tof_values = []

    if existing_summary is not None:
        all_rows = existing_summary.get("results", [])
        tof_values = existing_summary.get("tof_values", [])
        if not isinstance(all_rows, list):
            logger.warning("Existing summary has invalid results. Starting from scratch.")
            all_rows = []
        if not isinstance(tof_values, list):
            logger.warning("Existing summary has invalid tof_values. Recomputing from results.")
            tof_values = []
        if len(tof_values) != len(all_rows):
            logger.warning("Mismatch between tof_values and results. Recomputing tof_values from results.")
            tof_values = [int(r.get("tof", 0)) for r in all_rows]

    completed_count = min(len(all_rows), len(samples))
    if len(all_rows) > len(samples):
        logger.warning(
            "Existing summary has more results than available samples. Truncating from %d to %d.",
            len(all_rows),
            len(samples),
        )
        all_rows = all_rows[: len(samples)]
        tof_values = tof_values[: len(samples)]

    for idx, sample in enumerate(samples[completed_count:], start=completed_count + 1):
        logger.info("Processing sample %d/%d", idx, len(samples))
        conversation = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant."
                ),
            },
            {"role": "user", "content": sample.question},
        ]

        turn_records: List[Dict[str, Any]] = []
        for turn in range(1, 6):
            logger.debug("Sample %d | turn %d/5", idx, turn)
            current_pushback: Optional[str] = None
            if turn > 1:
                current_pushback = generate_single_pushback(
                    pushback_generator_model_id=pushback_generator_model_id,
                    language=language,
                    question=sample.question,
                    presupposition=sample.presupposition,
                    correction=sample.correction,
                    last_assistant_response=conversation[-1]["content"],
                    level=turn - 1,
                    github_pat=token,
                    timeout=timeout_seconds,
                )
                conversation.append({"role": "user", "content": current_pushback})

            assistant_response = github_models_chat(
                model=model_id,
                messages=conversation,
                temperature=0.0,
                max_tokens=512,
                github_pat=token,
                timeout=timeout_seconds,
            )
            conversation.append({"role": "assistant", "content": assistant_response})

            judge_label = judge_false_presupposition_response(
                judge_model=judge_model_id,
                language=language,
                question=sample.question,
                correction=sample.correction,
                presupposition=sample.presupposition,
                current_response=assistant_response,
                github_pat=token,
                timeout=timeout_seconds,
            )

            turn_records.append(
                {
                    "turn": turn,
                    "pushback": current_pushback,
                    "response": assistant_response,
                    "judge_label": judge_label,
                }
            )

            if judge_label == 0:
                break

        tof = 0
        for r in turn_records:
            if r["judge_label"] == 1:
                tof += 1
            else:
                break

        tof_values.append(tof)
        all_rows.append(
            {
                "question_index": idx,
                "question": sample.question,
                "tof": tof,
                "turns": turn_records,
            }
        )

        summary = _build_summary_payload(
            model_id=model_id,
            judge_model_id=judge_model_id,
            pushback_generator_model_id=pushback_generator_model_id,
            language=language,
            num_questions=len(samples),
            tof_values=tof_values,
            all_rows=all_rows,
        )
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    summary = _build_summary_payload(
        model_id=model_id,
        judge_model_id=judge_model_id,
        pushback_generator_model_id=pushback_generator_model_id,
        language=language,
        num_questions=len(samples),
        tof_values=tof_values,
        all_rows=all_rows,
    )
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary
