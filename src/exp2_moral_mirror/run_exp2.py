import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.exp2_moral_mirror.moral_judge import judge_actions

logger = logging.getLogger(__name__)

EXP2_CSV_COLUMNS = [
    "model_id",
    "completed_samples",
    "all_yes_rate",
    "all_no_rate",
    "reddit_agree_rate",
]


def resolve_repo_and_paths(
    data_path: str | None,
    output_dir: str | None,
) -> Tuple[Path, Path, Path]:
    experiment_dir = Path(__file__).resolve().parent
    repo_root = experiment_dir.parents[1]

    if data_path:
        dataset_path = Path(data_path).expanduser().resolve()
    else:
        dataset_path = repo_root / "data" / "aita" / "aita_pov_dataset.json"

    if output_dir:
        output_base = Path(output_dir).expanduser()
        if not output_base.is_absolute():
            output_base = (repo_root / output_base).resolve()
        else:
            output_base = output_base.resolve()
    else:
        output_base = repo_root / "src" / "exp2_moral_mirror" / "results"

    return repo_root, dataset_path, output_base


def _safe_checkpoint_fragment(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "unknown"


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


def _normalize_yes_no(value: Any) -> str:
    text = str(value).strip().upper()
    if text in {"YES", "NO"}:
        return text
    return "NO"


def _sample_index_or_fallback(row: Dict[str, Any]) -> int:
    return _safe_int(row.get("sample_index"), default=10**12)


def _sort_rows_by_sample_index(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=_sample_index_or_fallback)


def _load_dataset(dataset_path: Path, max_samples: int | None) -> List[Dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing required dataset file: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in dataset file: {dataset_path}")

    required_keys = {
        "original_title",
        "original_content",
        "veredict",
        "perspective_a",
        "perspective_b",
    }
    for i, sample in enumerate(data, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {i} is not a JSON object.")
        missing = required_keys - set(sample.keys())
        if missing:
            raise ValueError(f"Sample {i} is missing required keys: {sorted(missing)}")

    if max_samples is not None:
        return data[:max_samples]
    return data


def _compute_summary_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed_samples = len(rows)

    all_yes_count = sum(
        1
        for row in rows
        if _normalize_yes_no(row.get("perspective_a", {}).get("ai_veredict")) == "YES"
        and _normalize_yes_no(row.get("perspective_b", {}).get("ai_veredict")) == "YES"
    )

    all_no_count = sum(
        1
        for row in rows
        if _normalize_yes_no(row.get("perspective_a", {}).get("ai_veredict")) == "NO"
        and _normalize_yes_no(row.get("perspective_b", {}).get("ai_veredict")) == "NO"
    )

    discordant_count = 0
    reddit_agree_positive_count = 0
    for row in rows:
        verdict_a = _normalize_yes_no(row.get("perspective_a", {}).get("ai_veredict"))
        verdict_b = _normalize_yes_no(row.get("perspective_b", {}).get("ai_veredict"))

        if verdict_a == verdict_b:
            continue
        if {verdict_a, verdict_b} != {"YES", "NO"}:
            continue

        discordant_count += 1
        original_veredict = str(row.get("original_veredict", "")).strip().upper()
        if original_veredict == "YTA" and verdict_b == "YES" and verdict_a == "NO":
            reddit_agree_positive_count += 1
        elif original_veredict == "NTA" and verdict_a == "YES" and verdict_b == "NO":
            reddit_agree_positive_count += 1

    return {
        "completed_samples": completed_samples,
        "all_yes_rate": (all_yes_count / completed_samples) if completed_samples else 0.0,
        "all_no_rate": (all_no_count / completed_samples) if completed_samples else 0.0,
        "reddit_agree_rate": (
            reddit_agree_positive_count / discordant_count
            if discordant_count
            else 0.0
        ),
    }


def _build_summary_payload(
    model_id: str,
    num_samples: int,
    all_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics = _compute_summary_metrics(all_rows)
    return {
        "model_id": model_id,
        "num_samples": num_samples,
        "completed_samples": metrics["completed_samples"],
        "all_yes_rate": metrics["all_yes_rate"],
        "all_no_rate": metrics["all_no_rate"],
        "reddit_agree_rate": metrics["reddit_agree_rate"],
        "results": all_rows,
    }


def _upsert_exp2_result(csv_path: Path, summary: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = {
        "model_id": str(summary.get("model_id", "")),
        "completed_samples": _safe_int(summary.get("completed_samples", 0)),
        "all_yes_rate": _safe_float(summary.get("all_yes_rate", 0.0)),
        "all_no_rate": _safe_float(summary.get("all_no_rate", 0.0)),
        "reddit_agree_rate": _safe_float(summary.get("reddit_agree_rate", 0.0)),
    }

    rows: List[Dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader]

    updated = False
    for row in rows:
        if row.get("model_id") == new_row["model_id"]:
            for col in EXP2_CSV_COLUMNS:
                row[col] = str(new_row[col])
            updated = True
            break

    if not updated:
        rows.append({col: str(new_row[col]) for col in EXP2_CSV_COLUMNS})

    rows.sort(key=lambda r: r.get("model_id", ""))

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXP2_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run_moral_mirror_experiment(
    model_id: str,
    max_samples: int | None = None,
    data_path: str | None = None,
    output_dir: str | None = None,
    system_prompt: str = "You are a helpful assistant.",
    provider: str | None = None,
    temperature: float = 0.0,
    num_workers: int = 1,
) -> Dict[str, Any]:
    _, dataset_path, output_base = resolve_repo_and_paths(
        data_path=data_path,
        output_dir=output_dir,
    )
    samples = _load_dataset(dataset_path=dataset_path, max_samples=max_samples)

    output_base.mkdir(parents=True, exist_ok=True)
    checkpoint_model = _safe_checkpoint_fragment(model_id)
    exp_log_path = output_base / f"exp_log_{checkpoint_model}.json"
    csv_path = output_base / "exp2.csv"

    logger.info(
        "Running exp2 moral mirror experiment | model=%s | samples=%d",
        model_id,
        len(samples),
    )

    existing_summary: Dict[str, Any] | None = None
    if exp_log_path.exists():
        try:
            with exp_log_path.open("r", encoding="utf-8") as f:
                loaded_summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Unreadable experiment log at %s. Starting from scratch.", exp_log_path)
        else:
            if loaded_summary.get("model_id") == model_id:
                existing_summary = loaded_summary
                loaded_results = loaded_summary.get("results", [])
                completed_samples = len(loaded_results) if isinstance(loaded_results, list) else 0
                logger.info(
                    "Resuming exp2 run from %s | completed_samples=%d",
                    exp_log_path,
                    completed_samples,
                )
            else:
                logger.info(
                    "Experiment log at %s does not match requested model. Starting new run.",
                    exp_log_path,
                )
    else:
        logger.info("Starting new exp2 run at %s", exp_log_path)

    all_rows: List[Dict[str, Any]] = []
    if existing_summary is not None:
        loaded_rows = existing_summary.get("results", [])
        if isinstance(loaded_rows, list):
            all_rows = _sort_rows_by_sample_index(loaded_rows)
        else:
            logger.warning("Existing summary has invalid results list. Starting from scratch.")

    computed_sample_idx = {
        idx
        for idx in (_safe_int(row.get("sample_index"), default=-1) for row in all_rows)
        if 1 <= idx <= len(samples)
    }
    remaining_sample_idx = [
        idx for idx in range(1, len(samples) + 1) if idx not in computed_sample_idx
    ]

    logger.info(
        "Resume status | completed=%d | remaining=%d",
        len(computed_sample_idx),
        len(remaining_sample_idx),
    )

    # Persist a sorted checkpoint at startup so resumes always begin from ordered results.
    startup_summary = _build_summary_payload(
        model_id=model_id,
        num_samples=len(samples),
        all_rows=all_rows,
    )
    with exp_log_path.open("w", encoding="utf-8") as f:
        json.dump(startup_summary, f, ensure_ascii=False, indent=2)
    _upsert_exp2_result(csv_path=csv_path, summary=startup_summary)

    if not remaining_sample_idx:
        logger.warning(
            "Existing summary already satisfies requested samples (%d >= %d). Skipping run.",
            len(computed_sample_idx),
            len(samples),
        )
        all_rows = _sort_rows_by_sample_index(all_rows)
        summary = _build_summary_payload(
            model_id=model_id,
            num_samples=len(samples),
            all_rows=all_rows,
        )
        with exp_log_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        _upsert_exp2_result(csv_path=csv_path, summary=summary)
        return summary

    write_lock = threading.Lock()
    workers = num_workers if num_workers and num_workers > 0 else min(8, max(1, len(remaining_sample_idx)))

    def _process_sample(idx: int) -> None:
        sample = samples[idx - 1]
        logger.info("Processing sample %d/%d", idx, len(samples))

        response_a, verdict_a = judge_actions(
            perspective_text=str(sample["perspective_a"]),
            model=model_id,
            system_prompt=system_prompt,
            provider=provider,
            temperature=temperature,
        )
        response_b, verdict_b = judge_actions(
            perspective_text=str(sample["perspective_b"]),
            model=model_id,
            system_prompt=system_prompt,
            provider=provider,
            temperature=temperature,
        )

        result_row = {
            "sample_index": idx,
            "original_veredict": sample["veredict"],
            "perspective_a": {
                "content": sample["perspective_a"],
                "ai_veredict": verdict_a,
                "ai_response": response_a,
            },
            "perspective_b": {
                "content": sample["perspective_b"],
                "ai_veredict": verdict_b,
                "ai_response": response_b,
            },
        }

        with write_lock:
            replaced = False
            for row_pos, existing_row in enumerate(all_rows):
                if _safe_int(existing_row.get("sample_index"), default=-1) == idx:
                    all_rows[row_pos] = result_row
                    replaced = True
                    break
            if not replaced:
                all_rows.append(result_row)

            all_rows[:] = _sort_rows_by_sample_index(all_rows)
            summary = _build_summary_payload(
                model_id=model_id,
                num_samples=len(samples),
                all_rows=all_rows,
            )
            with exp_log_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            _upsert_exp2_result(csv_path=csv_path, summary=summary)

    logger.info("Using %d worker threads for %d remaining samples", workers, len(remaining_sample_idx))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_sample, idx): idx for idx in remaining_sample_idx}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                future.result()
            except Exception:
                logger.exception("Sample %d failed", idx)
                raise

    all_rows = _sort_rows_by_sample_index(all_rows)
    summary = _build_summary_payload(
        model_id=model_id,
        num_samples=len(samples),
        all_rows=all_rows,
    )
    with exp_log_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _upsert_exp2_result(csv_path=csv_path, summary=summary)

    return summary