import re
from typing import Optional, TypedDict, Union

import ray
import torch
from math_verify import grader
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.math_environment import _mute_output
from nemo_rl.environments.utils import chunk_list_to_workers


class ConfidenceEnvConfig(TypedDict):
    num_workers: int
    verifier_type: Optional[str]
    # Optional reward configuration
    reward_correct_high: Optional[float]
    reward_correct_low: Optional[float]
    reward_incorrect_low: Optional[float]
    reward_incorrect_high: Optional[float]
    reward_no_confidence: Optional[float]


@ray.remote  # pragma: no cover
class VerifyConfidenceWorker:
    def __init__(self, reward_scheme: dict):
        self.reward_correct_high = reward_scheme["reward_correct_high"]
        self.reward_correct_low = reward_scheme["reward_correct_low"]
        self.reward_incorrect_low = reward_scheme["reward_incorrect_low"]
        self.reward_incorrect_high = reward_scheme["reward_incorrect_high"]
        self.reward_no_confidence = reward_scheme["reward_no_confidence"]

        # Initialize the math verification function
        self.verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ),
        )

    def parse_confidence_level(self, response: str) -> float:
        """Returns 1.0 for High, 0.0 for Low, and -1.0 for no/invalid confidence.

        Matches lines that begin with 'Confidence:' (multiline), allowing leading
        whitespace and only 'High' or 'Low' with no trailing text. Uses the last match.
        """
        confidence_matches = re.findall(
            r"(?m)^\s*Confidence:\s*(High|Low)\s*$", response
        )
        if confidence_matches:
            confidence_text = confidence_matches[-1]
            if confidence_text == "High":
                return 1.0
            elif confidence_text == "Low":
                return 0.0
            else:
                return -1.0  # Invalid confidence level
        else:
            return -1.0

    def verify(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
        return_extracted_answer: bool = False,
    ) -> Union[list[float], tuple[list[float], list[dict[str, str]]]]:
        results: list[float] = []
        extracted_answers: list[dict[str, str]] = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            is_correct = False
            last_think_token = re.search(r"</think>", response)
            if last_think_token:
                response = response[
                    last_think_token.start() + len("</think>") :
                ]  # We do not care what's inside of <think> </think>
            else:
                results.append(
                    self.reward_no_confidence
                )  # No think token, assumes model didn't finish thinking or follow format
                extracted_answers.append({"mathematical_answer": "", "confidence": ""})
                continue
            # Find the last 'Answer: \\boxed{...}' line (line-anchored)
            answer_iters = list(
                re.finditer(r"(?m)^\s*Answer:\s*\\boxed{([^}]*)}\s*$", response)
            )
            if answer_iters:
                parseble_response = f"\\boxed{{{answer_iters[-1].group(1)}}}"
            else:
                results.append(
                    self.reward_no_confidence
                )  # No boxed response, assumes model didn't follow format
                extracted_answers.append({"mathematical_answer": "", "confidence": ""})
                continue
            with _mute_output():
                try:
                    # Wrap ground truth in \boxed{} format for math verification
                    ground_truth_boxed = f"\\boxed{{{ground_truth}}}"
                    ret_score, extracted_answer_result = self.verify_func(
                        [ground_truth_boxed], [parseble_response]
                    )
                    is_correct = float(ret_score) == 1.0

                    # Extract both mathematical answer and confidence
                    mathematical_answer = ""
                    confidence = ""

                    # Extract mathematical answer similar to math environment logic
                    if return_extracted_answer and extracted_answer_result is not None:
                        assert len(extracted_answer_result) == 2
                        extracted_gold, extracted_prediction = extracted_answer_result
                        # Get the extracted answer with the same logic as in the HFVerifyWorker
                        answer_found = None
                        for pred in extracted_prediction:
                            if any(
                                grader.verify(gold, pred) for gold in extracted_gold
                            ):
                                answer_found = pred
                                break
                        if answer_found is None and extracted_prediction:
                            # If no match is found, means all answers are incorrect, just use the first prediction
                            answer_found = (
                                extracted_prediction[0][0]
                                if extracted_prediction[0]
                                else None
                            )
                        mathematical_answer = (
                            answer_found if answer_found is not None else ""
                        )

                    # Extract confidence level (use last line-anchored 'Confidence: ...')
                    if return_extracted_answer:
                        confidence_matches = re.findall(
                            r"(?m)^\s*Confidence:\s*(High|Low)\s*$", response
                        )
                        if confidence_matches:
                            confidence = confidence_matches[-1]

                    extracted_answers.append(
                        {
                            "mathematical_answer": mathematical_answer,
                            "confidence": confidence,
                        }
                    )

                except Exception as e:
                    is_correct = False  # Error in verification, assumes model didn't follow format
                    extracted_answers.append(
                        {"mathematical_answer": "", "confidence": ""}
                    )
            confidence_level = self.parse_confidence_level(response)
            final_reward = 0.0
            if is_correct:
                if confidence_level == 1.0:
                    final_reward = self.reward_correct_high
                elif confidence_level == 0.0:
                    final_reward = self.reward_correct_low
                elif confidence_level == -1.0:
                    final_reward = self.reward_no_confidence
                else:
                    raise ValueError(
                        f"Invalid confidence level: {confidence_level} with correct answer {ground_truth} and response {response}"
                    )
            else:
                if confidence_level == 1.0:
                    final_reward = self.reward_incorrect_high
                elif confidence_level == 0.0:
                    final_reward = self.reward_incorrect_low
                elif confidence_level == -1.0:
                    final_reward = self.reward_no_confidence
                else:
                    raise ValueError(
                        f"Invalid confidence level: {confidence_level} with incorrect answer {ground_truth} and response {response}"
                    )
            results.append(final_reward)

        if return_extracted_answer:
            return results, extracted_answers
        else:
            return results


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class ConfidenceEnvironment(EnvironmentInterface):
    def __init__(self, cfg: ConfidenceEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        worker_cls = {
            "confidence_verify": VerifyConfidenceWorker,
        }[cfg.get("verifier_type", "confidence_verify")]

        self.reward_scheme = {
            "reward_correct_high": cfg.get("reward_correct_high", 1.0),
            "reward_correct_low": cfg.get("reward_correct_low", 0.5),
            "reward_incorrect_low": cfg.get("reward_incorrect_low", 0.0),
            "reward_incorrect_high": cfg.get("reward_incorrect_high", -0.5),
            "reward_no_confidence": cfg.get("reward_no_confidence", -1.0),
        }

        self.workers = [
            worker_cls.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote(self.reward_scheme)  # Pass config to worker
            for _ in range(self.num_workers)
        ]

    def step(
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata: list[dict],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn:
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        ground_truths = [g["ground_truth"] for g in metadata]

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(ground_truths, self.num_workers)
        futures = [
            self.workers[i].verify.remote(
                chunk, ground_truth_chunk, return_extracted_answer
            )
            for i, (chunk, ground_truth_chunk) in enumerate(
                zip(chunked_assistant_response_batch, chunked_ground_truths)
            )
        ]
        worker_results = ray.get(futures)

        # Flatten the results and extract both scores and answers
        results = []
        extracted_answers: list[dict[str, str]] | None = (
            [] if return_extracted_answer else None
        )

        for worker_result in worker_results:
            if return_extracted_answer:
                worker_scores, worker_answers = worker_result
                results.extend(worker_scores)
                extracted_answers.extend(worker_answers)
            else:
                results.extend(worker_result)
        observations = [
            {
                "role": "environment",
                "content": f"Environment: Task completed. Reward={result:.2f}",
            }
            for result in results
        ]
        rewards = torch.tensor(results).cpu()
        terminateds = torch.ones_like(rewards).cpu()
        next_stop_strings = [None] * len(message_log_batch)
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=terminateds,
            answers=extracted_answers,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        """Post-process batch and compute confidence metrics."""
        rewards = batch["rewards"]
        num_samples = len(rewards)

        # Count different reward categories
        overall_correct = (
            rewards == self.reward_scheme["reward_correct_high"]
        ).sum().item() + (
            rewards == self.reward_scheme["reward_correct_low"]
        ).sum().item()
        overall_incorrect = (
            rewards == self.reward_scheme["reward_incorrect_high"]
        ).sum().item() + (
            rewards == self.reward_scheme["reward_incorrect_low"]
        ).sum().item()

        # Calculate accuracies with safety checks
        completed = overall_correct + overall_incorrect
        completed_accuracy = overall_correct / completed if completed > 0 else 0.0
        overall_accuracy = overall_correct / num_samples if num_samples > 0 else 0.0

        # Calculate mean reward
        mean_reward = rewards.mean().item() if num_samples > 0 else 0.0

        # Calculate NCA (Normalized Confidence Advantage)
        inadequate_confidence = (
            rewards == self.reward_scheme["reward_correct_low"]
        ).sum().item() + (
            rewards == self.reward_scheme["reward_incorrect_high"]
        ).sum().item()

        # Use valid_mask instead of all rewards
        valid_mask = rewards != self.reward_scheme["reward_no_confidence"]
        overall_correct_valid = (
            (rewards == self.reward_scheme["reward_correct_high"]) & valid_mask
        ).sum().item() + (
            (rewards == self.reward_scheme["reward_correct_low"]) & valid_mask
        ).sum().item()
        overall_incorrect_valid = (
            (rewards == self.reward_scheme["reward_incorrect_high"]) & valid_mask
        ).sum().item() + (
            (rewards == self.reward_scheme["reward_incorrect_low"]) & valid_mask
        ).sum().item()

        # Use valid samples only for normalization coefficient
        norm_coef = min(overall_correct_valid, overall_incorrect_valid)
        nca = 1.0 - (inadequate_confidence / norm_coef) if norm_coef > 0 else 0.0

        # Calculate fractions
        metrics = {
            "completed_accuracy": completed_accuracy,
            "overall_accuracy": overall_accuracy,
            "mean_reward": mean_reward,
            "normalized_confidence_advantage": nca,
            "correct_confident_fraction": (
                rewards == self.reward_scheme["reward_correct_high"]
            )
            .float()
            .mean()
            .item(),
            "incorrect_confident_fraction": (
                rewards == self.reward_scheme["reward_incorrect_high"]
            )
            .float()
            .mean()
            .item(),
            "correct_unconfident_fraction": (
                rewards == self.reward_scheme["reward_correct_low"]
            )
            .float()
            .mean()
            .item(),
            "incorrect_unconfident_fraction": (
                rewards == self.reward_scheme["reward_incorrect_low"]
            )
            .float()
            .mean()
            .item(),
            "no_confidence_fraction": (
                rewards == self.reward_scheme["reward_no_confidence"]
            )
            .float()
            .mean()
            .item(),
            "num_samples": num_samples,
        }

        # Add generation length metrics if available
        if "generation_lengths" in batch:
            metrics["mean_generation_length"] = (
                batch["generation_lengths"].float().mean().item()
            )

        return batch, metrics

    def shutdown(self) -> None:
        """Clean up all workers."""
        for worker in self.workers:
            ray.kill(worker)
