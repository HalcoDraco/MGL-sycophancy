from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Sample:
    question: str
    correction: str
    presupposition: str


def _read_non_empty_lines(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_repo_and_data_dir(data_dir: Optional[str]) -> Tuple[Path, Path]:
    experiment_dir = Path(__file__).resolve().parent
    repo_root = experiment_dir.parents[1]

    if data_dir:
        base_data_dir = Path(data_dir).expanduser().resolve()
        return repo_root, base_data_dir

    repo_level_data = repo_root / "data" / "false_presuppositions"
    experiment_level_data = experiment_dir / "data"

    # Prefer repo-level data so multiple experiments can share the same datasets.
    if repo_level_data.exists():
        return repo_root, repo_level_data
    return repo_root, experiment_level_data


def _get_language_paths(base_data_dir: Path, language: str) -> Dict[str, Path]:
    if language == "en":
        lang_dir = base_data_dir / "source" / "en"
    elif language in {"es", "ca"}:
        lang_dir = base_data_dir / "translations" / language
    else:
        raise ValueError("Unsupported language. Use one of: en, es, ca.")

    return {
        "questions": lang_dir / "questions.txt",
        "corrections": lang_dir / "corrections.txt",
        "presuppositions": lang_dir / "presuppositions.txt",
    }


def load_samples(base_data_dir: Path, language: str, max_questions: Optional[int]) -> List[Sample]:
    paths = _get_language_paths(base_data_dir, language)

    questions = _read_non_empty_lines(paths["questions"])
    corrections = _read_non_empty_lines(paths["corrections"])
    presuppositions = _read_non_empty_lines(paths["presuppositions"])

    if not questions:
        raise ValueError(
            f"No questions found for language '{language}'. Fill the translation files first."
        )

    total = min(len(questions), len(corrections), len(presuppositions))
    if total == 0:
        raise ValueError(
            f"Language '{language}' has empty required data files. Fill translated files first."
        )

    if max_questions is not None:
        total = min(total, max_questions)

    samples: List[Sample] = []
    for i in range(total):
        samples.append(
            Sample(
                question=questions[i],
                correction=corrections[i],
                presupposition=presuppositions[i],
            )
        )
    return samples
