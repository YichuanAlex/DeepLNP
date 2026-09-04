"""Regression tests for unified formulation data preparation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from deeplnp.data.unified_dataset import UnifiedLNPFormulationDataset
from deeplnp.data.merged_dataset_loader import MergedMultiModalDataset


class UnifiedDatasetTest(unittest.TestCase):
    @staticmethod
    def _empty_dataset() -> UnifiedLNPFormulationDataset:
        dataset = UnifiedLNPFormulationDataset.__new__(
            UnifiedLNPFormulationDataset
        )
        dataset._mol_cache = {}
        return dataset

    def test_empty_target_evidence_remains_missing(self) -> None:
        frame = pd.DataFrame({
            "delivery_target": [np.nan],
            "synthesis_info": [np.nan],
            "bioactivity_profile": [np.nan],
        })

        derived = UnifiedLNPFormulationDataset._derive_text_endpoints_and_context(frame)

        self.assertTrue(pd.isna(derived.loc[0, "delivery_target"]))

    def test_lnp_ml_cholesterol_identity_is_source_defined(self) -> None:
        dataset = self._empty_dataset()
        row = pd.Series({
            "dataset_source": "LNP_ML",
            "cholesterol_mol_ratio": 38.5,
        })

        smiles, observed = dataset._component_info(row, "cholesterol")

        self.assertEqual(smiles, dataset.DEFAULT_CHOLESTEROL_SMILES)
        self.assertTrue(observed)

    def test_cholesterol_fallback_is_not_applied_to_other_sources(self) -> None:
        dataset = self._empty_dataset()
        row = pd.Series({
            "dataset_source": "LNP_Atlas",
            "cholesterol_mol_ratio": 38.5,
        })

        smiles, observed = dataset._component_info(row, "cholesterol")

        self.assertEqual(smiles, "")
        self.assertFalse(observed)

    def test_ambiguous_peg_product_is_not_promoted_to_a_structure(self) -> None:
        dataset = self._empty_dataset()
        row = pd.Series({
            "dataset_source": "LNP_ML",
            "peg_lipid": "C14-PEG2000",
            "peg_lipid_mol_ratio": 1.5,
        })

        smiles, observed = dataset._component_info(row, "peg")

        self.assertEqual(smiles, "")
        self.assertFalse(observed)

    def test_explicit_peg_structure_is_preserved(self) -> None:
        dataset = self._empty_dataset()
        row = pd.Series({"peg_lipid_smiles": "CCO"})

        smiles, observed = dataset._component_info(row, "peg")

        self.assertEqual(smiles, "CCO")
        self.assertTrue(observed)

    def test_legacy_loader_does_not_invent_peg_graph(self) -> None:
        dataset = MergedMultiModalDataset.__new__(MergedMultiModalDataset)
        dataset._mol_cache = {}

        self.assertEqual(dataset._resolve_lipid_name("DMG-PEG2000", "peg"), "")


if __name__ == "__main__":
    unittest.main()
