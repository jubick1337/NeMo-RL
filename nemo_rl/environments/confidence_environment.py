import logging
from typing import Any, Optional, TypedDict

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt
import ray
import torch
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.math_environment import MathEnvConfig, BaseMathEnvironment, _mute_output


@ray.remote
class HFVerifyWorkerConfidence:
    def __init__(self) -> None:
        logging.getLogger("math_verify").setLevel(logging.CRITICAL)
        self.verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ),
        )

    def parse_confidence(self, response: str) -> Optional[float]:
        """
        Parse the *last* confidence score from the response.
        This handles cases where the model might output confidence multiple times.
        """
        try:
            # Find the last line containing "confidence:"
            last_confidence_line = ""
            for line in response.lower().splitlines():
                if "confidence:" in line:
                    last_confidence_line = line

            if not last_confidence_line:
                return None  # No confidence line found

            confidence_score = last_confidence_line.split(":")[1].strip()
            if confidence_score == "high":
                return 1.0
            elif confidence_score == "low":
                return 0.0
            return None
        except Exception as e:
            logging.error(f"Error parsing confidence from response: {response}. Error: {e}")
            return None

    def verify(self, pred_responses: list[str], ground_truths: list[str]) -> list[float]:
        """
        Verify the correctness of the predicted responses against the ground truth.
        This now parses only the LAST boxed answer in the response.
        """
        results = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            correctness_score = 0.0
            try:
                # The math_verify library is designed to find all valid LaTeX,
                # but we can improve robustness by feeding it only the final part of the response.
                # A simple heuristic is to find the last `\boxed{...}`.
                last_boxed_idx = response.rfind("\\boxed{")
                if last_boxed_idx != -1:
                    # To be safe, we give the verifier the part of the response from the last box onwards
                    response_to_verify = response[last_boxed_idx:]
                else:
                    response_to_verify = response

                ground_truth_parsable = "\\boxed{" + ground_truth + "}"
                with _mute_output():
                    try:
                        ret_score, _ = self.verify_func([ground_truth_parsable], [response_to_verify])
                        correctness_score = float(ret_score)
                    except (Exception, TimeoutException):
                        correctness_score = 0.0
            except Exception:
                correctness_score = 0.0

            confidence_score_val = self.parse_confidence(response)

            if correctness_score == 1.0:
                if confidence_score_val == 1.0:
                    results.append(2.0)  # Correct & High Confidence
                elif confidence_score_val == 0.0:
                    results.append(1.0)  # Correct & Low Confidence
                else:
                    results.append(-2.0)  # Correct but no confidence found
            else:  # correctness_score == 0.0
                if confidence_score_val == 0.0:
                    results.append(0.0)  # Incorrect & Low Confidence
                elif confidence_score_val == 1.0:
                    results.append(-1.0)  # Incorrect & High Confidence
                else:
                    results.append(-2.0)  # Incorrect and no confidence found

        return results


# MODIFIED: Inherit from the non-actor BaseMathEnvironment
@ray.remote(max_restarts=-1, max_task_retries=-1)
class ConfidenceEnvironment(BaseMathEnvironment):
    def __init__(self, config: MathEnvConfig) -> None:
        # Initialize the base class
        super().__init__(config)
        # Initialize the specific workers for this environment
        self.workers = [
            HFVerifyWorkerConfidence.options(runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}).remote()
            for _ in range(self.num_workers)
        ]

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        # This method is already correct and doesn't need changes.
        original_rewards = batch["rewards"].clone()
        batch["rewards"] = batch["rewards"] * batch["is_end"]
        is_correct = (original_rewards == 2.0) | (original_rewards == 1.0)
        is_correct_float = is_correct.float()

        if is_correct.float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[is_correct].float().mean().item()
            )
        else:
            correct_solution_generation_lengths = 0

        num_high_confidence = ((original_rewards == 2.0) | (original_rewards == -1.0)).float().sum()
        if num_high_confidence > 0:
            precision_of_high_confidence = (original_rewards == 2.0).float().sum() / num_high_confidence
        else:
            precision_of_high_confidence = 0.0

        accuracy = (is_correct_float * batch["is_end"]).mean().item()

        metrics = {
            "mean_reward": batch["rewards"].mean().item(),
            "accuracy": accuracy,
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(batch["text"], is_correct_float),
            "precision_of_high_confidence": precision_of_high_confidence.item(),
            "frac_correct_confident": (original_rewards == 2.0).float().mean().item(),
            "frac_correct_unconfident": (original_rewards == 1.0).float().mean().item(),
            "frac_incorrect_unconfident": (original_rewards == 0.0).float().mean().item(),
            "frac_incorrect_confident": (original_rewards == -1.0).float().mean().item(),
            "frac_no_confidence_found": (original_rewards == -2.0).float().mean().item(),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }
        return batch, metrics
