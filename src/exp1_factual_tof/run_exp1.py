import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, List

from src.llm_apis.conversation import Conversation, ConversationTurn
from src.llm_apis.llm_api import llm_chat
from src.exp1_factual_tof.false_presupposition_judge import judge_false_presupposition_response
from src.exp1_factual_tof.data_loader import load_samples, resolve_repo_and_data_dir
from src.exp1_factual_tof.pushback_generator import generate_single_pushback

logger = logging.getLogger(__name__)


def _build_summary_payload(
    model_id: str,
    language: str,
    num_questions: int,
    tof_values: List[int],
    all_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    avg_tof = sum(tof_values) / len(tof_values) if tof_values else 0.0
    std_tof = statistics.pstdev(tof_values) if tof_values else 0.0

    return {
        "model_id": model_id,
        "language": language,
        "num_questions": num_questions,
        "average_tof": avg_tof,
        "std_tof": std_tof,
        "tof_values": tof_values,
        "results": all_rows,
    }


def run_false_presupposition_tof_experiment(
    model_id: str,
    judge_model_id: str = "gemini-3.1-flash-lite",
    pushback_generator_model_id: str = "llama-3.3-70b-instruct",
    language: str = "en",
    max_questions: int | None = None,
    data_dir: str | None = None,
    output_dir: str | None = None,
    timeout_seconds: int = 120,
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
    repo_root, base_data_dir = resolve_repo_and_data_dir(data_dir)
    samples = load_samples(base_data_dir, language, max_questions)

    logger.info(
        "Running exp1 factual ToF experiment | model=%s | language=%s | samples=%d",
        model_id,
        language,
        len(samples),
    )

    if output_dir is None:
        output_base = repo_root / "src" / "exp1_factual_tof" / "results"
    else:
        output_base = Path(output_dir)
        if not output_base.is_absolute():
            output_base = repo_root / output_base

    output_base.mkdir(parents=True, exist_ok=True)
    exp_log_path = output_base / "exp_log.json"

    existing_summary: Dict[str, Any] | None = None
    if exp_log_path.exists():
        try:
            with exp_log_path.open("r", encoding="utf-8") as f:
                loaded_summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Unreadable experiment log at %s. Starting from scratch.", exp_log_path)
        else:
            same_experiment = (
                loaded_summary.get("model_id") == model_id
                and loaded_summary.get("language") == language
            )
            if same_experiment:
                existing_summary = loaded_summary
                loaded_results = loaded_summary.get("results", [])
                completed_samples = len(loaded_results) if isinstance(loaded_results, list) else 0
                logger.info(
                    "Resuming exp1 run from %s | completed_samples=%d",
                    exp_log_path,
                    completed_samples,
                )
            else:
                logger.info(
                    "Existing experiment log at %s belongs to a different model/language. Starting new run.",
                    exp_log_path,
                )
    else:
        logger.info("Starting new exp1 run at %s", exp_log_path)

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
        conversation = Conversation(
            system_prompt="You are a helpful assistant.",
            turns=[ConversationTurn(role="user", content=sample.question)],
        )

        turn_records: Dict[str, Dict[str, Any]] = {}
        for turn in range(1, 6):
            logger.debug("Sample %d | turn %d/5", idx, turn)
            current_user_message = sample.question
            if turn > 1:
                current_user_message = generate_single_pushback(
                    pushback_generator_model_id=pushback_generator_model_id,
                    language=language,
                    question=sample.question,
                    presupposition=sample.presupposition,
                    correction=sample.correction,
                    last_assistant_response=conversation.turns[-1].content,
                    level=turn - 1,
                    timeout=timeout_seconds,
                )
                conversation = conversation.append_user(current_user_message)

            assistant_response = llm_chat(
                model=model_id,
                conversation=conversation,
                temperature=0.0,
                max_tokens=512,
                timeout=timeout_seconds,
            )
            conversation = conversation.append_model(assistant_response)

            judge_label = judge_false_presupposition_response(
                judge_model=judge_model_id,
                language=language,
                question=sample.question,
                correction=sample.correction,
                presupposition=sample.presupposition,
                current_response=assistant_response,
                timeout=timeout_seconds,
            )

            turn_name = "question" if turn == 1 else f"pushback{turn - 1}"
            turn_records[turn_name] = {
                "user": current_user_message,
                "ai_response": assistant_response,
                "judge_label": judge_label,
            }

            if judge_label == 0:
                break

        tof = 0
        for r in turn_records.values():
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
                "turn_records": turn_records,
            }
        )

        summary = _build_summary_payload(
            model_id=model_id,
            language=language,
            num_questions=len(samples),
            tof_values=tof_values,
            all_rows=all_rows,
        )
        with exp_log_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    summary = _build_summary_payload(
        model_id=model_id,
        language=language,
        num_questions=len(samples),
        tof_values=tof_values,
        all_rows=all_rows,
    )
    with exp_log_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary
