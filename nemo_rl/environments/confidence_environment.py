import logging
from typing import Any, Optional, TypedDict
import re # Import the re module

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt
import ray
import torch
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.math_environment import MathEnvConfig, BaseMathEnvironment, _mute_output

# MODIFIED: Add a new parameter for the format reward
class ConfidenceEnvConfig(MathEnvConfig, total=False):
    reward_correct_high: float
    reward_correct_low: float
    reward_incorrect_low: float
    reward_incorrect_confident: float
    reward_no_confidence: float
    reward_for_format: float # NEW

@ray.remote
class HFVerifyWorkerConfidence:
    def __init__(
        self,
        reward_correct_high: float = 2.0,
        reward_correct_low: float = 1.0,
        reward_incorrect_low: float = 0.0,
        reward_incorrect_confident: float = -1.0,
        reward_no_confidence: float = -2.0,
        # MODIFIED: Default the format reward to 0.0
        reward_for_format: float = 0.0,
    ) -> None:
        logging.getLogger("math_verify").setLevel(logging.CRITICAL)
        self.verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ),
        )
        # Store all reward values
        self.reward_correct_high = reward_correct_high
        self.reward_correct_low = reward_correct_low
        self.reward_incorrect_low = reward_incorrect_low
        self.reward_incorrect_confident = reward_incorrect_confident
        self.reward_no_confidence = reward_no_confidence
        self.reward_for_format = reward_for_format # NEW

    def check_response_format(self, response: str) -> int:
        """
        Checks if a response string contains the pattern:
        <think>ANYTHING</think> SOMETHING ELSE <confidence>ANYTHING</confidence>
        """
        pattern = r"<think>.*?</think>.*?<confidence>.*?</confidence>"
        if re.search(pattern, response, re.DOTALL):
            return 1
        else:
            return -1

    def parse_confidence(self, response: str) -> Optional[float]:
        """
        Parse the *last* confidence score from the response.
        """
        try:
            last_confidence_line = ""
            for line in response.lower().splitlines():
                if "confidence:" in line:
                    last_confidence_line = line
            if not last_confidence_line:
                return None
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
        Verify correctness and format, then calculate the final reward.
        """
        results = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            # 1. Check for correctness
            correctness_score = 0.0
            try:
                last_boxed_idx = response.rfind("\\boxed{")
                response_to_verify = response[last_boxed_idx:] if last_boxed_idx != -1 else response
                ground_truth_parsable = "\\boxed{" + ground_truth + "}"
                with _mute_output():
                    try:
                        ret_score, _ = self.verify_func([ground_truth_parsable], [response_to_verify])
                        correctness_score = float(ret_score)
                    except (Exception, TimeoutException):
                        correctness_score = 0.0
            except Exception:
                correctness_score = 0.0

            # 2. Check for confidence
            confidence_score_val = self.parse_confidence(response)

            # 3. Calculate base reward
            final_reward = 0.0
            if correctness_score == 1.0:
                if confidence_score_val == 1.0:
                    final_reward = self.reward_correct_high
                elif confidence_score_val == 0.0:
                    final_reward = self.reward_correct_low
                else:
                    final_reward = self.reward_no_confidence
            else:  # correctness_score == 0.0
                if confidence_score_val == 0.0:
                    final_reward = self.reward_incorrect_low
                elif confidence_score_val == 1.0:
                    final_reward = self.reward_incorrect_confident
                else:
                    final_reward = self.reward_no_confidence
            
            # 4. Conditionally check format and add reward
            if self.reward_for_format != 0:
                if self.check_response_format(response) == 1:
                    final_reward += self.reward_for_format

            results.append(final_reward)
        return results

@ray.remote(max_restarts=-1, max_task_retries=-1)
class ConfidenceEnvironment(BaseMathEnvironment):
    def __init__(self, config: ConfidenceEnvConfig) -> None:
        super().__init__(config)
        reward_config = {
            "reward_correct_high": config.get("reward_correct_high", 2.0),
            "reward_correct_low": config.get("reward_correct_low", 1.0),
            "reward_incorrect_low": config.get("reward_incorrect_low", 0.0),
            "reward_incorrect_confident": config.get("reward_incorrect_confident", -1.0),
            "reward_no_confidence": config.get("reward_no_confidence", -2.0),
            "reward_for_format": config.get("reward_for_format", 0.0),
        }
        self.workers = [
            HFVerifyWorkerConfidence.options(runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}).remote(**reward_config)
            for _ in range(self.num_workers)
        ]
        # Store reward values for metric calculation
        self.reward_correct_high = reward_config["reward_correct_high"]
        self.reward_correct_low = reward_config["reward_correct_low"]
        self.reward_incorrect_low = reward_config["reward_incorrect_low"]
        self.reward_incorrect_confident = reward_config["reward_incorrect_confident"]
        self.reward_no_confidence = reward_config["reward_no_confidence"]
        self.reward_for_format = reward_config["reward_for_format"]

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        original_rewards = batch["rewards"].clone()

        # To calculate the base metrics, we must first determine if the format bonus was applied.
        # A response has the correct format if its reward is `reward_for_format` higher than a base reward.
        has_correct_format = (original_rewards == self.reward_correct_high + self.reward_for_format) | \
                             (original_rewards == self.reward_correct_low + self.reward_for_format) | \
                             (original_rewards == self.reward_incorrect_low + self.reward_for_format) | \
                             (original_rewards == self.reward_incorrect_confident + self.reward_for_format) | \
                             (original_rewards == self.reward_no_confidence + self.reward_for_format)
        
        # Now, calculate the base reward by subtracting the format bonus where it was applied.
        base_reward = original_rewards.clone()
        base_reward[has_correct_format] -= self.reward_for_format

        # --- Calculate all metrics using the `base_reward` ---
        
        is_correct = (base_reward == self.reward_correct_high) | (base_reward == self.reward_correct_low)
        is_correct_float = is_correct.float()

        batch["rewards"] = batch["rewards"] * batch["is_end"]
        
        if is_correct_float.sum() > 0:
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[is_correct].float().mean().item()
            )
        else:
            correct_solution_generation_lengths = 0

        num_high_confidence = ((base_reward == self.reward_correct_high) | (base_reward == self.reward_incorrect_confident)).float().sum()
        
        if num_high_confidence > 0:
            precision_of_high_confidence = (base_reward == self.reward_correct_high).float().sum() / num_high_confidence
        else:
            precision_of_high_confidence = 0.0

        accuracy = (is_correct_float * batch["is_end"]).mean().item()

        # New metric for accuracy on completed samples only
        completed_samples_mask = batch["is_end"].bool()
        if completed_samples_mask.sum() > 0:
            accuracy_on_completed = is_correct_float[completed_samples_mask].mean().item()
        else:
            accuracy_on_completed = 0.0

        metrics = {
            "mean_reward": batch["rewards"].mean().item(),
            "accuracy": accuracy,
            "accuracy_on_completed": accuracy_on_completed,
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(batch["text"], is_correct_float),
            "precision_of_high_confidence": precision_of_high_confidence.item(),
            
            "frac_correct_confident": (base_reward == self.reward_correct_high).float().mean().item(),
            "frac_correct_unconfident": (base_reward == self.reward_correct_low).float().mean().item(),
            "frac_incorrect_unconfident": (base_reward == self.reward_incorrect_low).float().mean().item(),
            "frac_incorrect_confident": (base_reward == self.reward_incorrect_confident).float().mean().item(),
            "frac_no_confidence_found": (base_reward == self.reward_no_confidence).float().mean().item(),
            "frac_correct_format": has_correct_format.float().mean().item(), # NEW metric
            
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }
        return batch, metrics