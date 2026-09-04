"""Regression tests for reviewer-driven input and supervision ablations."""

from __future__ import annotations

import unittest

import torch

from deeplnp.data.unified_dataset import FORMULATION_DIM
from run_ablations import (
    NoSupervisedEndpointError,
    model_inputs,
    multitask_loss,
)


class AblationRoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.formulation = torch.arange(
            2 * FORMULATION_DIM, dtype=torch.float32
        ).reshape(2, FORMULATION_DIM)
        self.ratios = torch.tensor(
            [[0.4, 0.1, 0.4, 0.1], [0.5, 0.1, 0.35, 0.05]],
            dtype=torch.float32,
        )
        self.batch = {
            "formulation_features": self.formulation.clone(),
            "molar_ratios": self.ratios.clone(),
        }

    def test_ratio_value_ablation_is_written_back(self) -> None:
        output = model_inputs(
            self.batch,
            {"drop_component_ratio_values": ["cholesterol", "peg"]},
        )
        value_width = FORMULATION_DIM // 2
        dropped = [2, 3, 6, 7]
        for column in dropped:
            self.assertTrue(torch.equal(
                output["formulation_features"][:, column],
                torch.zeros(2),
            ))
            self.assertTrue(torch.equal(
                output["formulation_features"][:, value_width + column],
                self.formulation[:, value_width + column],
            ))
        self.assertTrue(torch.equal(
            output["molar_ratios"][:, 2:],
            torch.zeros(2, 2),
        ))
        self.assertTrue(torch.equal(
            self.batch["formulation_features"],
            self.formulation,
        ))

    def test_ratio_observation_ablation_preserves_values(self) -> None:
        output = model_inputs(
            self.batch,
            {"drop_component_ratio_observation_bits": ["cholesterol", "peg"]},
        )
        value_width = FORMULATION_DIM // 2
        dropped = [2, 3, 6, 7]
        self.assertTrue(torch.equal(
            output["formulation_features"][:, :value_width],
            self.formulation[:, :value_width],
        ))
        for column in dropped:
            self.assertTrue(torch.equal(
                output["formulation_features"][:, value_width + column],
                torch.zeros(2),
            ))
        self.assertTrue(torch.equal(output["molar_ratios"], self.ratios))

    def test_empty_enabled_task_batch_is_explicitly_skipped(self) -> None:
        output = {
            "predictions": {
                "efficiency": torch.zeros(2),
                "efficiency_log_variance": torch.zeros(2),
                "target_logits": torch.zeros(2, 2),
            }
        }
        batch = {
            "efficiency": torch.zeros(2),
            "efficiency_mask": torch.zeros(2, dtype=torch.bool),
            "group_id": torch.arange(2),
            "target_class": torch.zeros(2, dtype=torch.long),
            "target_class_mask": torch.zeros(2, dtype=torch.bool),
        }
        with self.assertRaises(NoSupervisedEndpointError):
            multitask_loss(
                output,
                batch,
                enabled_tasks=["efficiency"],
                target_aux_weight=0.0,
            )


if __name__ == "__main__":
    unittest.main()
