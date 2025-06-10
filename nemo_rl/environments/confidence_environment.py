import logging
from typing import Any, Optional, TypedDict

import ray
import torch
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.math_environment import MathEnvConfig, MathEnvironment, _mute_output
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt


@ray.remote
class HFVerifyWorkerConfidence:
    def __init__(self) -> None:
        logging.getLogger("math_verify").setLevel(logging.CRITICAL)

        # Use Latex and plain math extraction from predictions
        # https://github.com/huggingface/Math-Verify?tab=readme-ov-file#extraction-targets
        self.verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ),
        )

    def parse_confidence(self, response: str) -> Optional[float]:
        """Parse the confidence score from the response.

        Args:
            response: str. The response from the LLM.

        Returns:
            Optional[float]. The confidence score if present, otherwise None.
        """
        try:
            # Assuming the confidence is in the format "Confidence: <score>"
            confidence_line = next(line for line in response.splitlines() if "Confidence:" in line)
            confidence_score = confidence_line.split("Confidence:")[1].strip()
            if confidence_score.lower() == "high":
                return 1.0
            elif confidence_score.lower() == "low":
                return 0.0
            return None
        except StopIteration:
            # If no confidence line is found, return None
            return None
        except Exception as e:
            logging.error(f"Error parsing confidence from response: {response}. Error: {e}")
            return None

    def verify(self, pred_responses: list[str], ground_truths: list[str]) -> list[float]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            list[float]. The rewards for each predicted response.
        """
        results = []
        confidence_scores = []
        correctness_scores = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            try:
                ground_truth_parsable = "\\boxed{" + ground_truth + "}"
                with _mute_output():
                    try:
                        ret_score, _ = self.verify_func([ground_truth_parsable], [response])
                    # It's possible to emit a TimeoutException and that wouldn't be caught since
                    # it actually subclasses from BaseException and math-verify itself does not
                    # to catch it.
                    except (Exception, TimeoutException):
                        ret_score = 0.0

                correctness_scores.append(float(ret_score))
            except Exception:
                correctness_scores.append(0.0)
        for response in pred_responses:
            try:
                confidence = self.parse_confidence(response)
                if confidence is None:
                    confidence_scores.append(-2.0)
                else:
                    confidence_scores.append(float(confidence))
            except Exception:
                confidence_scores.append(-2.0)
        for correctness_score, confidence_score in zip(correctness_scores, confidence_scores):
            if correctness_score == 1.0 and confidence_score == 1.0:
                results.append(2.0)
            elif correctness_score == 1.0 and confidence_score == 0.0:
                results.append(1.0)
            elif correctness_score == 0.0 and confidence_score == 0.0:
                results.append(0.0)
            elif correctness_score == 0.0 and confidence_score == 1.0:
                results.append(-1.0)
            else:
                results.append(-2.0)

        return results


@ray.remote(max_restarts=-1, max_task_retries=-1)
class ConfidenceEnvironment(MathEnvironment):
    def __init__(self, config: MathEnvConfig) -> None:
        super().__init__(config)
        self.workers = [
            HFVerifyWorkerConfidence.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        """Computes metrics for the confidence environment given a global rollout batch.

        This method overrides the parent implementation to provide metrics relevant
        to the confidence-based reward scheme:
        - Reward 2.0: Correct & High Confidence
        - Reward 1.0: Correct & Low Confidence
        - Reward 0.0: Incorrect & Low Confidence
        - Reward -1.0: Incorrect & High Confidence
        - Reward -2.0: Confidence Not Found

        Args:
            batch: The global batch of rollout data.

        Returns:
            A tuple containing the processed batch and a dictionary of metrics.
        """
        # Keep the original rewards for detailed metric calculation before masking
        original_rewards = batch["rewards"].clone()

        # Zero out rewards for any sequences that were not properly terminated.
        # This is the main post-processing step for the reward tensor itself.
        batch["rewards"] = batch["rewards"] * batch["is_end"]

        # --- Metric Calculation ---

        # Define correctness based on the original, unmasked rewards.
        # A response is correct if its reward was 2.0 or 1.0.
        is_correct = (original_rewards == 2.0) | (original_rewards == 1.0)
        is_correct_float = is_correct.float()

        # Calculate the generation length for only the correct solutions
        if is_correct.float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[is_correct].float().mean().item()
            )
        else:
            correct_solution_generation_lengths = 0

        # Calculate the precision of "High Confidence" statements.
        # Precision = (Correct & Confident) / (All Confident Statements)
        num_high_confidence = ((original_rewards == 2.0) | (original_rewards == -1.0)).float().sum()
        if num_high_confidence > 0:
            precision_of_high_confidence = (original_rewards == 2.0).float().sum() / num_high_confidence
        else:
            precision_of_high_confidence = 0.0  # Avoid division by zero

        # Accuracy is the fraction of the batch that was correct AND ended properly.
        # Improperly ended sequences are counted as incorrect.
        accuracy = (is_correct_float * batch["is_end"]).mean().item()

        metrics = {
            # --- Primary Performance Metrics ---
            "mean_reward": batch["rewards"].mean().item(),
            "accuracy": accuracy,
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(batch["text"], is_correct_float),
            "precision_of_high_confidence": precision_of_high_confidence.item(),
            # --- Outcome Breakdown (as fractions of the total batch) ---
            "frac_correct_confident": (original_rewards == 2.0).float().mean().item(),
            "frac_correct_unconfident": (original_rewards == 1.0).float().mean().item(),
            "frac_incorrect_unconfident": (original_rewards == 0.0).float().mean().item(),
            "frac_incorrect_confident": (original_rewards == -1.0).float().mean().item(),
            "frac_no_confidence_found": (original_rewards == -2.0).float().mean().item(),
            # --- General Rollout Stats ---
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }

        return batch, metrics
