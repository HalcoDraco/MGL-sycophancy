import logging

from dotenv import load_dotenv
import argparse

from src.exp1_factual_tof.run_exp1 import run_false_presupposition_tof_experiment
from src.exp2_moral_mirror.run_exp2 import run_moral_mirror_experiment
from src.exp3_prompt_mitigations.run_exp3 import run_exp3_prompt_mitigation_experiment
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

def main():
    configure_app_logging()

    parser = argparse.ArgumentParser(description="Run MGL sycophancy experiments from a single entrypoint.")
    parser.add_argument("--experiment", choices=["1", "2", "3"], default="1")
    args = parser.parse_args()

    if args.experiment == "1":
        languages = ["ca", "es", "en"]
        reasoning_test = ["DeepSeek-V3", "DeepSeek-R1"] # Non-reasoning vs reasoning models
        alignment_test1 = ["mistral-7b-sft-beta", "zephyr-7b-beta"] # SFT vs DPO
        alignment_test2 = ["llama-3.1-8b", "Hermes-3-Llama-3.1-8B"] # RLHF vs DPO
        size_test = ["Qwen2.5-1.5B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-72B-Instruct"] # Small vs medium vs large models
        family_test = ["Mistral-7B-Instruct-v0.2", "gemma-2-9b-it"]
        model_ids = alignment_test2 + size_test + reasoning_test
        # for num_questions in range(50, 201, 50):
        for language in languages:
            for model_id in model_ids:
                run_false_presupposition_tof_experiment(
                    model_id=model_id,
                    language=language,
                    max_questions=100,#num_questions,
                    num_workers=20,
                    judge_model_id="Qwen3-235B-A22B-Instruct-2507",
                    pushback_generator_model_id="Qwen3-235B-A22B-Instruct-2507",
                )

    elif args.experiment == "2":
        reasoning_test = ["DeepSeek-V3", "DeepSeek-R1"] # Non-reasoning vs reasoning models
        alignment_test1 = ["mistral-7b-sft-beta", "zephyr-7b-beta"] # SFT vs DPO
        alignment_test2 = ["llama-3.1-8b", "Hermes-3-Llama-3.1-8B"] # RLHF vs DPO
        size_test = ["Qwen2.5-1.5B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-72B-Instruct"] # Small vs medium vs large models
        family_test = ["Mistral-7B-Instruct-v0.2", "gemma-2-9b-it"]
        model_ids = reasoning_test + alignment_test2 + size_test

        # for num_samples in range(1, 61, 15):
        for model_id in model_ids:
            run_moral_mirror_experiment(model_id=model_id, 
                                        max_samples=60,
                                        num_workers=50)
    
    elif args.experiment == "3":
        size_test = ["Qwen2.5-1.5B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-72B-Instruct"]
        alignment_test2 = ["llama-3.1-8b", "Hermes-3-Llama-3.1-8B"]
        model_ids = size_test + alignment_test2

        for experiment_number in [1, 2]:
            for model_id in model_ids:
                run_exp3_prompt_mitigation_experiment(
                    experiment_number=experiment_number,
                    model_id=model_id,
                    max_questions=40,
                    max_samples=60,
                    num_workers=80,
                    judge_model_id="Qwen3-235B-A22B-Instruct-2507",
                    pushback_generator_model_id="Qwen3-235B-A22B-Instruct-2507",
                )

    else:
        raise ValueError(f"Invalid experiment number: {args.experiment}")

if __name__ == "__main__":
    main()