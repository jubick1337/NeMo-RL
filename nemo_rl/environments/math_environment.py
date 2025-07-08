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
import contextlib
import io
import logging
import re
from typing import Any, Optional, TypedDict

import ray
import torch
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.utils import chunk_list_to_workers
from nemo_rl.evals import answer_parsing


class MathEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[list[str]]  # Default stop strings for this env
    verifier_type: Optional[str]


@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield


@ray.remote
class HFVerifyWorker:
    def __init__(self) -> None:
        logging.getLogger("math_verify").setLevel(logging.CRITICAL)
        self.verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ),
        )

    def verify(
        self, pred_responses: list[str], ground_truths: list[str]
    ) -> list[float]:
        results = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            try:
                ground_truth_parsable = "\\boxed{" + ground_truth + "}"
                with _mute_output():
                    try:
                        ret_score, _ = self.verify_func(
                            [ground_truth_parsable], [response]
                        )
                    except (Exception, TimeoutException):
                        ret_score = 0.0
                results.append(float(ret_score))
            except Exception:
                results.append(0.0)
        return results


@ray.remote
class MultichoiceVerifyWorker:
    def verify(
        self, pred_responses: list[str], ground_truths: list[str]
    ) -> list[float]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            list[float]. The rewards for each predicted response.
        """
        results = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            response = answer_parsing.normalize_response(response)
            extracted_answer = None
            for answer_regex in answer_parsing.MULTILINGUAL_ANSWER_REGEXES:
                regex = answer_parsing.MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(
                    answer_regex
                )
                match = re.search(regex, response)
                if match:
                    extracted_answer = answer_parsing.normalize_extracted_answer(
                        match.group(1)
                    )
                    break
            score = 1.0 if extracted_answer == ground_truth else 0.0
            results.append(score)
        return results


class MathEnvironmentMetadata(TypedDict):
    ground_truth: str

# NEW: Create a non-actor base class with the shared logic.
# Note: No @ray.remote decorator here!
class BaseMathEnvironment(EnvironmentInterface):
    def __init__(self, cfg: MathEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        worker_cls = (
            MultichoiceVerifyWorker
            if cfg.get("verifier_type", "math") == "multichoice"
            else HFVerifyWorker
        )
        self.workers = [
            worker_cls.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata: list[MathEnvironmentMetadata],
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
            self.workers[i].verify.remote(chunk, ground_truth_chunk)
            for i, (chunk, ground_truth_chunk) in enumerate(
                zip(chunked_assistant_response_batch, chunked_ground_truths)
            )
        ]
        results = ray.get(futures)
        results = [item for sublist in results for item in sublist]
        observations = [
            {
                "role": "environment",
                "content": "Environment: correct"
                if result
                else "Environment: incorrect",
            }
            for result in results
        ]

        rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()
        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        batch["rewards"] = (
            batch["rewards"] * batch["is_end"]
        )
        if (batch["rewards"] == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[
                    batch["rewards"] == 1
                ]
                .float()
                .mean()
                .item()
            )
        else:
            correct_solution_generation_lengths = 0

        metrics = {
            "accuracy": batch["rewards"].mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], batch["rewards"]
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }
        return batch, metrics


# MODIFIED: MathEnvironment is now an actor that inherits from the non-actor base class.
@ray.remote(max_restarts=-1, max_task_retries=-1)
class MathEnvironment(BaseMathEnvironment):
    def __init__(self, cfg: MathEnvConfig):
        # Initialize the base class
        super().__init__(cfg)
        # Initialize the specific workers for this environment
        self.workers = [
            HFVerifyWorker.options(
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
            for _ in range(self.num_workers)
        ]