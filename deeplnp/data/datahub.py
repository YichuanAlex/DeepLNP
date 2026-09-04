"""
Data hub for loading and managing datasets from multiple sources.
Integrates LNP Atlas, AGILE, LANTERN, and custom datasets.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .dataset import LNPDataset, build_dataloader


class DataHub:
    """
    Centralized data management for DeepLNP.
    
    Supports:
    - LNP Atlas (2026)
    - AGILE dataset
    - LANTERN cleaned data
    - Custom datasets
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        dataset_name: str = "lnp_atlas",
        test_size: float = 0.2,
        val_size: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Root directory for data
            dataset_name: Name of dataset to load
            test_size: Fraction of data for testing
            val_size: Fraction of data for validation
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.target_scaler = None
        
    def load_dataset(self, split: str = "scaffold") -> Dict:
        """
        Load and split dataset.
        
        Args:
            split: Splitting strategy ('random', 'scaffold', 'cliff')
        
        Returns:
            Dictionary with train/val/test DataLoaders
        """
        # Load raw data
        raw_df = self._load_raw_data()
        
        # Split data
        if split == "random":
            train_df, temp_df = train_test_split(
                raw_df, test_size=self.test_size + self.val_size, random_state=self.seed
            )
            val_df, test_df = train_test_split(
                temp_df, test_size=self.val_size / (self.test_size + self.val_size),
                random_state=self.seed
            )
        elif split == "scaffold":
            # Scaffold-based split (more realistic evaluation)
            train_df, val_df, test_df = self._scaffold_split(raw_df)
        elif split == "cliff":
            # Transfection cliff-aware split
            train_df, val_df, test_df = self._cliff_split(raw_df)
        else:
            raise ValueError(f"Unknown split strategy: {split}")
        
        # Save splits
        self._save_splits(train_df, val_df, test_df)
        
        # Build datasets
        self.train_data = LNPDataset(train_df)
        self.val_data = LNPDataset(val_df)
        self.test_data = LNPDataset(test_df)
        
        # Fit target scaler
        if "target" in train_df.columns:
            self.target_scaler = StandardScaler()
            self.target_scaler.fit(train_df[["target"]])
        
        return {
            "train": self.train_data,
            "val": self.val_data,
            "test": self.test_data,
            "target_scaler": self.target_scaler,
        }
    
    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw data from specified source."""
        if self.dataset_name == "lnp_atlas":
            return self._load_lnp_atlas()
        elif self.dataset_name == "agile":
            return self._load_agile()
        elif self.dataset_name == "lantern":
            return self._load_lantern()
        elif self.dataset_name == "custom":
            return self._load_custom()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_lnp_atlas(self) -> pd.DataFrame:
        """Load LNP Atlas dataset (2026)."""
        # Check if already downloaded
        csv_path = self.data_dir / "raw" / "lnp_atlas.csv"
        
        if not csv_path.exists():
            print("Downloading LNP Atlas dataset...")
            # Download from source (placeholder)
            # In practice, download from the paper's repository
            raise FileNotFoundError("Please download LNP Atlas dataset manually")
        
        df = pd.read_csv(csv_path)
        
        # Standardize column names
        df = df.rename(columns={
            "ionizable_lipid_smiles": "SMILES",
            "transfection_efficiency": "target",
        })
        
        print(f"Loaded LNP Atlas: {len(df)} formulations")
        return df
    
    def _load_agile(self) -> pd.DataFrame:
        """Load AGILE dataset."""
        # AGILE dataset from Wang lab
        csv_path = self.data_dir / "raw" / "agile.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError("Please download AGILE dataset")
        
        df = pd.read_csv(csv_path)
        print(f"Loaded AGILE: {len(df)} formulations")
        return df
    
    def _load_lantern(self) -> pd.DataFrame:
        """Load LANTERN cleaned dataset."""
        csv_path = self.data_dir / "raw" / "lantern_cleaned.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError("Please download LANTERN dataset")
        
        df = pd.read_csv(csv_path)
        print(f"Loaded LANTERN: {len(df)} formulations")
        return df
    
    def _load_custom(self) -> pd.DataFrame:
        """Load custom dataset."""
        csv_path = self.data_dir / "raw" / "custom.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Custom dataset not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"Loaded custom dataset: {len(df)} formulations")
        return df
    
    def _scaffold_split(
        self,
        df: pd.DataFrame,
        smiles_col: str = "SMILES"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by molecular scaffold.
        More challenging and realistic than random split.
        """
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        
        def get_scaffold(smi):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            scaffold = rdMolDescriptors.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        
        # Compute scaffolds
        df["scaffold"] = df[smiles_col].apply(get_scaffold)
        
        # Group by scaffold
        scaffold_groups = df.groupby("scaffold").indices
        
        # Split scaffolds
        scaffolds = list(scaffold_groups.keys())
        train_scaff, temp_scaff = train_test_split(
            scaffolds, test_size=self.test_size + self.val_size, random_state=self.seed
        )
        val_scaff, test_scaff = train_test_split(
            temp_scaff, test_size=self.val_size / (self.test_size + self.val_size),
            random_state=self.seed
        )
        
        # Get dataframes
        train_df = df[df["scaffold"].isin(train_scaff)].drop("scaffold", axis=1)
        val_df = df[df["scaffold"].isin(val_scaff)].drop("scaffold", axis=1)
        test_df = df[df["scaffold"].isin(test_scaff)].drop("scaffold", axis=1)
        
        print(f"Scaffold split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        return train_df, val_df, test_df
    
    def _cliff_split(
        self,
        df: pd.DataFrame,
        smiles_col: str = "SMILES",
        target_col: str = "target"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data to ensure test set contains transfection cliffs.
        Transfection cliffs: small structural changes → large efficiency differences
        """
        # Compute molecular fingerprints - using new MorganGenerator API
        from rdkit import Chem
        from rdkit import DataStructs
        from rdkit.Chem import rdFingerprintGenerator
        
        def get_fp(smi):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
            return fpgen.GetFingerprintAsNumPy(mol)
        
        df["fp"] = df[smiles_col].apply(get_fp)
        
        # Find cliff pairs (similar structure, different efficiency)
        cliff_pairs = []
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if df.iloc[i]["fp"] is None or df.iloc[j]["fp"] is None:
                    continue
                sim = DataStructs.TanimotoSimilarity(df.iloc[i]["fp"], df.iloc[j]["fp"])
                eff_diff = abs(df.iloc[i][target_col] - df.iloc[j][target_col])
                
                # Cliff: high similarity (>0.7) but large efficiency difference (>0.3)
                if sim > 0.7 and eff_diff > 0.3:
                    cliff_pairs.append((i, j))
        
        # Ensure cliff pairs are in test set
        cliff_indices = set()
        for i, j in cliff_pairs[:100]:  # Limit to 100 pairs
            cliff_indices.add(i)
            cliff_indices.add(j)
        
        # Split
        cliff_df = df.iloc[list(cliff_indices)]
        non_cliff_df = df.drop(cliff_indices)
        
        train_df, temp_df = train_test_split(
            non_cliff_df, test_size=self.test_size + self.val_size, random_state=self.seed
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=self.val_size / (self.test_size + self.val_size),
            random_state=self.seed
        )
        
        # Add cliff pairs to test set
        test_df = pd.concat([test_df, cliff_df])
        
        print(f"Cliff split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        print(f"  Including {len(cliff_df)} cliff samples")
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
    
    def _save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):
        """Save train/val/test splits."""
        split_dir = self.data_dir / "splits"
        split_dir.mkdir(exist_ok=True)
        
        train_df.to_csv(split_dir / "train.csv", index=False)
        val_df.to_csv(split_dir / "val.csv", index=False)
        test_df.to_csv(split_dir / "test.csv", index=False)
        
        print(f"Splits saved to {split_dir}")
    
    def get_dataloaders(
        self,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Dict:
        """Get DataLoaders for train/val/test."""
        return {
            "train": build_dataloader(
                self.train_data, batch_size=batch_size, shuffle=True,
                num_workers=num_workers
            ),
            "val": build_dataloader(
                self.val_data, batch_size=batch_size, shuffle=False,
                num_workers=num_workers
            ),
            "test": build_dataloader(
                self.test_data, batch_size=batch_size, shuffle=False,
                num_workers=num_workers
            ),
        }
