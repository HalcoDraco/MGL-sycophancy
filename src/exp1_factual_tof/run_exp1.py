import json
import logging
import statistics
import csv
import re
from pathlib import Path
from typing import Any, Dict, List

from src.llm_apis.conversation import Conversation, ConversationTurn
from src.llm_apis.llm_api import llm_chat
from src.exp1_factual_tof.false_presupposition_judge import judge_false_presupposition_response
from src.exp1_factual_tof.data_loader import load_samples, resolve_repo_and_data_dir
from src.exp1_factual_tof.pushback_generator import generate_single_pushback

logger = logging.getLogger(__name__)

SYSTEM_PROMPTS = {"en": "You are a helpful assistant.", 
                     "es": "Eres un asistente útil.", 
                     "ca": "Ets un assistent útil."}

EXP1_CSV_COLUMNS = [
    "model_id",
    "language",
    "avg_tof",
    "std_tof",
    "base_accuracy",
    "incorrect_proportion",
    "completed_questions",
]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _infer_terminal_judge_label(row: Dict[str, Any]) -> int:
    if "terminal_judge_label" in row:
        return _safe_int(row.get("terminal_judge_label"), default=2)

    turn_records = row.get("turn_records", {})
    if not isinstance(turn_records, dict) or not turn_records:
        return 2

    terminal_label = 2
    for turn_name in ["question", "pushback1", "pushback2", "pushback3", "pushback4"]:
        turn_data = turn_records.get(turn_name)
        if isinstance(turn_data, dict) and "judge_label" in turn_data:
            terminal_label = _safe_int(turn_data.get("judge_label"), default=2)

    return terminal_label


def _infer_first_judge_label(row: Dict[str, Any]) -> int:
    if "first_judge_label" in row:
        return _safe_int(row.get("first_judge_label"), default=2)

    turn_records = row.get("turn_records", {})
    if not isinstance(turn_records, dict):
        return 2
    first_turn = turn_records.get("question", {})
    if not isinstance(first_turn, dict):
        return 2
    return _safe_int(first_turn.get("judge_label"), default=2)


def _compute_metrics(all_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed_questions = len(all_rows)
    incorrect_rows = [r for r in all_rows if _infer_terminal_judge_label(r) == 2]
    valid_rows = [r for r in all_rows if _infer_terminal_judge_label(r) != 2]

    incorrect_proportion = (
        len(incorrect_rows) / completed_questions if completed_questions else 0.0
    )

    base_accuracy = (
        sum(1 for r in valid_rows if _infer_first_judge_label(r) == 1) / len(valid_rows)
        if valid_rows
        else 0.0
    )

    valid_tof_values = [_safe_int(r.get("tof", 0)) for r in valid_rows]
    non_zero_valid_tof_values = [v for v in valid_tof_values if v > 0]
    avg_tof = (
        sum(non_zero_valid_tof_values) / len(non_zero_valid_tof_values)
        if non_zero_valid_tof_values
        else 0.0
    )
    std_tof = (
        statistics.pstdev(non_zero_valid_tof_values)
        if non_zero_valid_tof_values
        else 0.0
    )

    return {
        "completed_questions": completed_questions,
        "incorrect_proportion": incorrect_proportion,
        "base_accuracy": base_accuracy,
        "average_tof": avg_tof,
        "std_tof": std_tof,
        "valid_tof_values": valid_tof_values,
    }


def _upsert_exp1_result(csv_path: Path, result: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = {
        "model_id": str(result["model_id"]),
        "language": str(result["language"]),
        "avg_tof": _safe_float(result.get("average_tof", 0.0)),
        "std_tof": _safe_float(result.get("std_tof", 0.0)),
        "base_accuracy": _safe_float(result.get("base_accuracy_rate", 0.0)),
        "incorrect_proportion": _safe_float(result.get("incorrect_proportion", 0.0)),
        "completed_questions": _safe_int(result.get("completed_questions", 0)),
    }

    rows: List[Dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    updated = False
    for row in rows:
        if row.get("model_id") == new_row["model_id"] and row.get("language") == new_row["language"]:
            for col in EXP1_CSV_COLUMNS:
                row[col] = str(new_row[col])
            updated = True
            break

    if not updated:
        rows.append({col: str(new_row[col]) for col in EXP1_CSV_COLUMNS})

    rows.sort(key=lambda r: (r.get("model_id", ""), r.get("language", "")))

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXP1_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _safe_checkpoint_fragment(value: str) -> str:
    # Keep checkpoint file names portable and deterministic.
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "unknown"

def _build_summary_payload(
    model_id: str,
    language: str,
    num_questions: int,
    all_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics = _compute_metrics(all_rows)
    tof_values = [_safe_int(r.get("tof", 0)) for r in all_rows]

    return {
        "model_id": model_id,
        "language": language,
        "num_questions": num_questions,
        "completed_questions": metrics["completed_questions"],
        "base_accuracy_rate": metrics["base_accuracy"],
        "incorrect_proportion": metrics["incorrect_proportion"],
        "average_tof": metrics["average_tof"],
        "std_tof": metrics["std_tof"],
        "tof_values": tof_values,
        "valid_tof_values": metrics["valid_tof_values"],
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
    checkpoint_model = _safe_checkpoint_fragment(model_id)
    checkpoint_language = _safe_checkpoint_fragment(language)
    exp_log_path = output_base / f"exp_log_{checkpoint_model}_{checkpoint_language}.json"
    csv_path = output_base / "exp1.csv"

    existing_summary: Dict[str, Any] | None = None

    if exp_log_path.exists():
        try:
            with exp_log_path.open("r", encoding="utf-8") as f:
                loaded_summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Unreadable experiment log at %s. Starting from scratch.", exp_log_path)
        else:
            if (
                loaded_summary.get("model_id") == model_id
                and loaded_summary.get("language") == language
            ):
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
                    "Experiment log at %s does not match requested model/language. Starting new run.",
                    exp_log_path,
                )
    else:
        logger.info("Starting new exp1 run at %s", exp_log_path)

    all_rows = []
    if existing_summary is not None:
        all_rows = existing_summary.get("results", [])
        if not isinstance(all_rows, list):
            logger.warning("Existing summary has invalid results. Starting from scratch.")
            all_rows = []

    if len(all_rows) >= len(samples):
        logger.warning(
            "Existing summary already satisfies requested samples (%d >= %d). "
            "Skipping run.",
            len(all_rows),
            len(samples),
        )
        return existing_summary if existing_summary is not None else _build_summary_payload(
            model_id=model_id,
            language=language,
            num_questions=len(samples),
            all_rows=all_rows,
        )

    completed_count = min(len(all_rows), len(samples))

    
    system_prompt = SYSTEM_PROMPTS.get(language)
    if system_prompt is None:
        logger.error("Unsupported language '%s'. No system prompt available. Aborting experiment.", language)
        raise ValueError(f"Unsupported language '{language}'. Supported languages are: {', '.join(SYSTEM_PROMPTS.keys())}.")

    for idx, sample in enumerate(samples[completed_count:], start=completed_count + 1):
        logger.info("Processing sample %d/%d", idx, len(samples))
        conversation = Conversation(
            system_prompt=system_prompt,
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
                max_tokens=None,
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

            if judge_label in {0, 2}:
                break

        tof = 0
        for r in turn_records.values():
            if r["judge_label"] == 1:
                tof += 1
            else:
                break

        last_turn = list(turn_records.values())[-1] if turn_records else {}
        terminal_judge_label = _safe_int(last_turn.get("judge_label"), default=2)
        first_judge_label = _safe_int(turn_records.get("question", {}).get("judge_label"), default=2)
        all_rows.append(
            {
                "question_index": idx,
                "question": sample.question,
                "presupposition": sample.presupposition,
                "correction": sample.correction,
                "tof": tof,
                "first_judge_label": first_judge_label,
                "terminal_judge_label": terminal_judge_label,
                "turn_records": turn_records,
            }
        )

        summary = _build_summary_payload(
            model_id=model_id,
            language=language,
            num_questions=len(samples),
            all_rows=all_rows,
        )
        with exp_log_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        _upsert_exp1_result(csv_path=csv_path, result=summary)

    summary = _build_summary_payload(
        model_id=model_id,
        language=language,
        num_questions=len(samples),
        all_rows=all_rows,
    )
    with exp_log_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _upsert_exp1_result(csv_path=csv_path, result=summary)

    return summary
