import logging
from typing import Any, Optional, TypedDict
import re # Import the re module

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.metrics import calculate_pass_rate_by_id, calculate_pass_rate_by_idx
import ray
import torch
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
# Import the base environment and its original worker
from nemo_rl.environments.math_environment import MathEnvConfig, BaseMathEnvironment, HFVerifyWorker, _mute_output
from nemo_rl.environments.interfaces import EnvironmentReturn
from nemo_rl.environments.utils import chunk_list_to_workers


class ConfidenceEnvConfig(MathEnvConfig, total=False):
    reward_correct_high: float
    reward_correct_low: float
    reward_incorrect_low: float
    reward_incorrect_confident: float
    reward_no_confidence: float
    reward_for_format: float

class ConfidenceVerifyResult(TypedDict):
    reward: float
    is_correct: bool
    has_format: bool
    confidence_level: Optional[float]


@ray.remote
class HFVerifyWorkerConfidence:
    def __init__(
        self,
        reward_correct_high: float = 2.0,
        reward_correct_low: float = 1.0,
        reward_incorrect_low: float = 0.0,
        reward_incorrect_confident: float = -1.0,
        reward_no_confidence: float = -2.0,
        reward_for_format: float = 0.0,
    ) -> None:
        logging.getLogger("math_verify").setLevel(logging.CRITICAL)
        self.verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(),),
        )
        self.reward_correct_high = reward_correct_high
        self.reward_correct_low = reward_correct_low
        self.reward_incorrect_low = reward_incorrect_low
        self.reward_incorrect_confident = reward_incorrect_confident
        self.reward_no_confidence = reward_no_confidence
        self.reward_for_format = reward_for_format

    def check_response_format(self, response: str) -> bool:
        pattern = r"<think>.*?</think>.*?<confidence>.*?</confidence>"
        return bool(re.search(pattern, response, re.DOTALL))

    def parse_confidence(self, response: str) -> Optional[float]:
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

    def verify(self, pred_responses: list[str], ground_truths: list[str]) -> list[ConfidenceVerifyResult]:
        """
        Verify correctness and format, then return a structured dictionary.
        """
        results: list[ConfidenceVerifyResult] = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            # 1. Check for correctness
            is_correct = False
            try:
                last_boxed_idx = response.rfind("\\boxed{")
                response_to_verify = response[last_boxed_idx:] if last_boxed_idx != -1 else response
                ground_truth_parsable = "\\boxed{" + ground_truth + "}"
                with _mute_output():
                    try:
                        ret_score, _ = self.verify_func([ground_truth_parsable], [response_to_verify])
                        is_correct = (float(ret_score) == 1.0)
                    except (Exception, TimeoutException):
                        is_correct = False
            except Exception:
                is_correct = False

            # 2. Check for confidence and format
            confidence_level = self.parse_confidence(response)
            has_format = self.check_response_format(response)

            # 3. Calculate base reward
            final_reward = 0.0
            if is_correct:
                if confidence_level == 1.0:
                    final_reward = self.reward_correct_high
                elif confidence_level == 0.0:
                    final_reward = self.reward_correct_low
                else:
                    final_reward = self.reward_no_confidence
            else:
                if confidence_level == 0.0:
                    final_reward = self.reward_incorrect_low
                elif confidence_level == 1.0:
                    final_reward = self.reward_incorrect_confident
                else:
                    final_reward = self.reward_no_confidence
            
            # 4. Conditionally add format reward
            if self.reward_for_format != 0 and has_format:
                final_reward += self.reward_for_format

            results.append({
                "reward": final_reward,
                "is_correct": is_correct,
                "has_format": has_format,
                "confidence_level": confidence_level,
            })
        return results

@ray.remote(max_restarts=-1, max_task_retries=-1)
class ConfidenceEnvironment(BaseMathEnvironment):
    def __init__(self, config: ConfidenceEnvConfig) -> None:
        # We call the BaseMathEnvironment's init, but we will immediately
        # replace its generic workers with our specialized confidence workers.
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

    def step(self, message_log_batch: list[list[dict[str, str]]], metadata: list[dict[str,Any]]) -> EnvironmentReturn:
        """
        An overridden step method to handle the structured output from HFVerifyWorkerConfidence.
        """
        assistant_response_batch = [
            "".join([msg["content"] for msg in convo if msg["role"] == "assistant"])
            for convo in message_log_batch
        ]
        ground_truths = [g["ground_truth"] for g in metadata]

        chunked_responses = chunk_list_to_workers(assistant_response_batch, self.num_workers)
        chunked_gt = chunk_list_to_workers(ground_truths, self.num_workers)

        futures = [
            self.workers[i].verify.remote(res_chunk, gt_chunk)
            for i, (res_chunk, gt_chunk) in enumerate(zip(chunked_responses, chunked_gt))
        ]
        
        # results will be a list of lists of dictionaries
        results_nested = ray.get(futures)
        # Flatten the list
        results_flat = [item for sublist in results_nested for item in sublist]

        # Extract data for EnvironmentReturn and add extra info to metadata
        rewards_list = []
        for i, res in enumerate(results_flat):
            rewards_list.append(res["reward"])
            # Add the explicit, reliable flags to the metadata for this sample
            metadata[i]["is_correct"] = res["is_correct"]
            metadata[i]["has_format"] = res["has_format"]
            metadata[i]["confidence_level"] = res["confidence_level"]

        rewards = torch.tensor(rewards_list, dtype=torch.float32).cpu()
        terminateds = torch.ones_like(rewards, dtype=torch.bool).cpu()
        
        # The observation is less important in this single-turn setup
        observations = [{"role": "environment", "content": f"Reward: {r.item()}"} for r in rewards]

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=rewards,
            terminateds=terminateds,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        is_correct_float = batch["is_correct"].float()
        has_format_float = batch["has_format"].float()
        confidence_level = batch["confidence_level"]

        # Reconstruct base_reward reliably for frac_* metrics if needed
        is_high_conf = (confidence_level == 1.0)
        is_low_conf = (confidence_level == 0.0)
        
        frac_correct_confident = (is_correct_float * is_high_conf).mean().item()
        frac_correct_unconfident = (is_correct_float * is_low_conf).mean().item()
        frac_incorrect_confident = ((1 - is_correct_float) * is_high_conf).mean().item()
        frac_incorrect_unconfident = ((1 - is_correct_float) * is_low_conf).mean().item()
        frac_no_confidence_found = (confidence_level == -1.0).float().mean().item() # Assuming None becomes -1

        if is_correct_float.sum() > 0:
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[is_correct_float.bool()].float().mean().item()
            )
        else:
            correct_solution_generation_lengths = 0
        
        num_high_confidence = is_high_conf.float().sum()
        if num_high_confidence > 0:
            precision_of_high_confidence = (is_correct_float * is_high_conf).sum() / num_high_confidence
        else:
            precision_of_high_confidence = 0.0

        accuracy = (is_correct_float * batch["terminated"]).mean().item()

        completed_samples_mask = batch["terminated"].bool()
        if completed_samples_mask.sum() > 0:
            accuracy_on_completed = is_correct_float[completed_samples_mask].mean().item()
        else:
            accuracy_on_completed = 0.0
            
        if "idx" in batch:
            pass_rate = calculate_pass_rate_by_idx(batch["idx"], is_correct_float)
        else:
            logging.warning(f"Key 'idx' not found in batch for pass_rate calculation. Defaulting to 0.0.")
            pass_rate = 0.0

        # --- Calculate Normalized Confidence Advantage (NCA) ---
        num_samples = is_correct_float.shape[0]
        num_correct = is_correct_float.sum()
        num_incorrect = num_samples - num_correct
        
        num_confidence_inadequate = (frac_correct_unconfident + frac_incorrect_confident) * num_samples

        norm_coefficient = torch.min(num_correct, num_incorrect)

        if norm_coefficient > 0:
            nca = 1.0 - (num_confidence_inadequate / norm_coefficient)
            normalized_confidence_advantage = nca.item()
        else:
            normalized_confidence_advantage = 0.0

        metrics = {
            "mean_reward": (batch["total_reward"] * batch["terminated"]).mean().item(),
            "accuracy": accuracy,
            "accuracy_on_completed": accuracy_on_completed,
            "normalized_confidence_advantage": normalized_confidence_advantage,
            "pass@samples_per_prompt": pass_rate,
            "precision_of_high_confidence": precision_of_high_confidence.item(),
            
            "frac_correct_confident": frac_correct_confident,
            "frac_correct_unconfident": frac_correct_unconfident,
            "frac_incorrect_unconfident": frac_incorrect_unconfident,
            "frac_incorrect_confident": frac_incorrect_confident,
            "frac_no_confidence_found": frac_no_confidence_found,
            "frac_correct_format": has_format_float.mean().item(),
            
            "fraction_of_samples_terminated": batch["terminated"].float().mean().item(),
            "num_problems_in_batch": batch["terminated"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }
        return batch, metrics