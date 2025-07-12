# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re
import traceback
from typing import Any, Optional, TypedDict

import ray
import torch
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentReturn
from nemo_rl.environments.math_environment import (
    BaseMathEnvironment,
    MathEnvConfig,
    _mute_output,
)
from nemo_rl.environments.utils import chunk_list_to_workers

# ===============================================================================
# UTILITY AND HELPER FUNCTIONS
# ===============================================================================


def calculate_pass_rate_by_idx(
    prompt_indices: torch.Tensor, is_correct: torch.Tensor
) -> float:
    """
    An efficient, vectorized function to compute the pass rate given prompt indices.
    """
    if prompt_indices.numel() == 0:
        return 0.0

    unique_indices, inverse_indices = torch.unique(prompt_indices, return_inverse=True)
    num_unique_prompts = len(unique_indices)

    if num_unique_prompts == 0:
        return 0.0

    prompt_correctness_sum = torch.zeros(
        num_unique_prompts, dtype=torch.int, device=is_correct.device
    )
    prompt_correctness_sum.scatter_add_(0, inverse_indices, is_correct.int())

    num_passed_prompts = (prompt_correctness_sum > 0).sum().item()

    return num_passed_prompts / num_unique_prompts


# ===============================================================================
# CONFIDENCE ENVIRONMENT DEFINITION
# ===============================================================================


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
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ),
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
            logging.error(
                f"Error parsing confidence from response: {response}. Error: {e}"
            )
            return None

    def verify(
        self, pred_responses: list[str], ground_truths: list[str]
    ) -> list[ConfidenceVerifyResult]:
        results: list[ConfidenceVerifyResult] = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            is_correct = False
            try:
                last_boxed_idx = response.rfind("\\boxed{")
                response_to_verify = (
                    response[last_boxed_idx:] if last_boxed_idx != -1 else response
                )
                ground_truth_parsable = "\\boxed{" + ground_truth + "}"
                with _mute_output():
                    try:
                        ret_score, _ = self.verify_func(
                            [ground_truth_parsable], [response_to_verify]
                        )
                        is_correct = float(ret_score) == 1.0
                    except (Exception, TimeoutException):
                        is_correct = False
            except Exception:
                is_correct = False

            confidence_level = self.parse_confidence(response)
            has_format = self.check_response_format(response)

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

            if self.reward_for_format != 0 and has_format:
                final_reward += self.reward_for_format

            results.append(
                {
                    "reward": final_reward,
                    "is_correct": is_correct,
                    "has_format": has_format,
                    "confidence_level": confidence_level,
                }
            )
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
            HFVerifyWorkerConfidence.options(
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote(**reward_config)
            for _ in range(self.num_workers)
        ]

    def step(
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata: list[dict[str, Any]],
    ) -> EnvironmentReturn:
        assistant_response_batch = [
            "".join([msg["content"] for msg in convo if msg["role"] == "assistant"])
            for convo in message_log_batch
        ]
        ground_truths = [g["ground_truth"] for g in metadata]

        chunked_responses = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_gt = chunk_list_to_workers(ground_truths, self.num_workers)

        futures = [
            self.workers[i].verify.remote(res_chunk, gt_chunk)
            for i, (res_chunk, gt_chunk) in enumerate(zip(chunked_responses, chunked_gt))
        ]

        results_nested = ray.get(futures)
        results_flat = [item for sublist in results_nested for item in sublist]

        rewards_list = []
        # Attach the full verification results to the metadata.
        # This is the key change to propagate the data forward.
        for i, res in enumerate(results_flat):
            rewards_list.append(res["reward"])
            # Ensure metadata at index i exists and is a dict before modification
            if i < len(metadata) and isinstance(metadata[i], dict):
                metadata[i]["verification_result"] = res

        rewards = torch.tensor(rewards_list, dtype=torch.float32).cpu()
        terminateds = torch.ones_like(rewards, dtype=torch.bool).cpu()

        observations = [
            {"role": "environment", "content": f"Reward: {r.item()}"} for r in rewards
        ]

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,  # The metadata now contains our precious verification results
            next_stop_strings=[None] * len(message_log_batch),
            rewards=rewards,
            terminateds=terminateds,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        try:
            # --- EFFICIENT DATA EXTRACTION ---
            # This now reads the pre-computed results that were propagated from the `step` method.
            verify_results = [info["verification_result"] for info in batch["extra_env_info"]]

            # --- Convert Pre-computed Verification Results to Tensors ---
            device = batch["loss_multiplier"].device
            is_correct_float = torch.tensor(
                [res["is_correct"] for res in verify_results],
                dtype=torch.float32,
                device=device,
            )
            has_format_float = torch.tensor(
                [res["has_format"] for res in verify_results],
                dtype=torch.float32,
                device=device,
            )
            confidence_level_list = [res.get("confidence_level") for res in verify_results]
            confidence_level = torch.tensor(
                [-1.0 if v is None else v for v in confidence_level_list],
                dtype=torch.float32,
                device=device,
            )

            # --- Start of Metric Calculations ---
            valid_mask = batch["loss_multiplier"]
            is_high_conf = confidence_level == 1.0
            is_low_conf = confidence_level == 0.0

            frac_correct_confident = (is_correct_float * is_high_conf).mean().item()
            frac_correct_unconfident = (is_correct_float * is_low_conf).mean().item()
            frac_incorrect_confident = ((1 - is_correct_float) * is_high_conf).mean().item()
            frac_incorrect_unconfident = ((1 - is_correct_float) * is_low_conf).mean().item()
            frac_no_confidence_found = (confidence_level == -1.0).float().mean().item()

            correct_solution_generation_lengths = 0.0
            if "generation_lengths" in batch and "prompt_lengths" in batch:
                if is_correct_float.sum() > 0:
                    correct_solution_generation_lengths = (
                        (batch["generation_lengths"] - batch["prompt_lengths"])[
                            is_correct_float.bool()
                        ]
                        .float()
                        .mean()
                        .item()
                    )

            num_high_confidence = is_high_conf.float().sum()
            precision_of_high_confidence = (
                (is_correct_float * is_high_conf).sum() / num_high_confidence
                if num_high_confidence > 0
                else torch.tensor(0.0)
            )

            accuracy = (is_correct_float * valid_mask).mean().item()

            completed_samples_mask = valid_mask.bool()
            accuracy_on_completed = (
                is_correct_float[completed_samples_mask].mean().item()
                if completed_samples_mask.sum() > 0
                else 0.0
            )

            if "prompt_ids" in batch:
                pass_rate = calculate_pass_rate_by_idx(batch["prompt_ids"], is_correct_float)
            else:
                logging.warning(
                    "Key 'prompt_ids' not found in batch for pass_rate calculation. Defaulting to 0.0."
                )
                pass_rate = 0.0

            num_samples = is_correct_float.shape[0]
            num_correct = is_correct_float.sum()
            num_incorrect = num_samples - num_correct
            num_confidence_inadequate = (is_correct_float * is_low_conf).sum() + (
                (1 - is_correct_float) * is_high_conf
            ).sum()
            norm_coefficient = torch.min(num_correct, num_incorrect)

            if norm_coefficient > 0:
                nca = 1.0 - (num_confidence_inadequate / norm_coefficient)
                normalized_confidence_advantage = nca.item()
            else:
                normalized_confidence_advantage = 0.0

            metrics = {
                "mean_reward": (batch["total_reward"] * valid_mask).mean().item(),
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
                "fraction_of_samples_valid": valid_mask.float().mean().item(),
                "num_problems_in_batch": valid_mask.shape[0],
                "correct_solution_generation_lengths": correct_solution_generation_lengths,
            }

            if "generation_lengths" in batch and "prompt_lengths" in batch:
                metrics["generation_lengths"] = batch["generation_lengths"].float().mean().item()
                metrics["prompt_lengths"] = batch["prompt_lengths"].float().mean().item()

            return batch, metrics

        except Exception as e:
            error_trace = traceback.format_exc()
            logging.error(
                "!!! CRITICAL ERROR in global_post_process_and_metrics !!!\n"
                f"Error: {e}\n"
                f"Traceback:\n{error_trace}\n"
                "Skipping metrics calculation for this batch and continuing training."
            )
            return batch, {}