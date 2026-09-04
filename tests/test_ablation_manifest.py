import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import run_ablations


class AblationManifestTest(unittest.TestCase):
    def test_missing_manifest_is_initialized(self):
        summary = pd.DataFrame({
            "name": ["test_variant"],
            "val_multitask_loss_mean": [1.0],
            "val_efficiency_rmse_mean": [1.0],
            "val_efficiency_spearman_mean": [0.0],
            "val_efficiency_within_group_spearman_mean": [0.0],
            "val_efficiency_pair_accuracy_mean": [0.5],
            "val_efficiency_within_group_ndcg_mean": [0.5],
            "val_efficiency_within_group_hit_recall_at_10pct_mean": [0.1],
            "val_efficiency_within_group_normalized_regret_at_10pct_mean": [0.2],
        })
        metadata = {
            "dataset_rows": 10,
            "split_strategy": "group",
            "train_rows": 6,
            "val_rows": 2,
            "test_rows": 2,
            "epochs": 1,
            "seeds": [1],
            "use_conformers": False,
            "learning_rate": 5e-4,
            "early_stopping_patience": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "ablation_results" / "run"
            output.mkdir(parents=True)
            with patch.object(run_ablations, "ROOT", root):
                run_ablations.register_ablation_round(output, summary, metadata)
            self.assertTrue((root / "ablation_manifest.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
