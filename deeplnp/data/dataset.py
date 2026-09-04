"""
Dataset classes for LNP data management.
Handles molecules, formulations, and assay data.
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
from rdkit.Chem import rdFingerprintGenerator


class LNPDataset(Dataset):
    """
    Dataset for LNP formulations with multi-modal inputs.
    
    Supports:
    - SMILES strings (1D)
    - 3D molecular conformers
    - Graph representations
    - Formulation parameters (molar ratios, components)
    """
    
    def __init__(
        self,
        data_path: str,
        mode: str = "train",
        use_3d: bool = True,
        use_graph: bool = True,
        remove_hs: bool = False,
        max_atoms: int = 256,
        transform=None,
    ):
        """
        Args:
            data_path: Path to CSV file with columns:
                      - SMILES: ionizable lipid SMILES
                      - helper_lipid: helper lipid SMILES (optional)
                      - cholesterol: cholesterol type (optional)
                      - PEG: PEG-lipid type (optional)
                      - molar_ratios: [ionizable, helper, cholesterol, PEG]
                      - target: transfection efficiency
                      - particle_size: LNP size (nm)
                      - zeta_potential: zeta potential (mV)
                      - toxicity: cell viability (%)
            mode: 'train', 'val', or 'test'
            use_3d: Whether to generate 3D conformers
            use_graph: Whether to compute graph representations
            remove_hs: Remove hydrogens from molecular representation
            max_atoms: Maximum number of atoms to consider
            transform: Optional transform to apply to data
        """
        self.data_path = data_path
        self.mode = mode
        self.use_3d = use_3d
        self.use_graph = use_graph
        self.remove_hs = remove_hs
        self.max_atoms = max_atoms
        self.transform = transform
        
        # Load data
        self.df = pd.read_csv(data_path)
        self.df = self._preprocess()
        
        # Load or compute features
        self.smiles_list = self.df["SMILES"].tolist()
        self.targets = self.df["target"].values if "target" in self.df.columns else None
        
        # Pre-compute 3D conformers and fingerprints
        self.mol_dict = {}
        self.conformer_dict = {}
        self.fingerprint_dict = {}
        self._prepare_molecules()
        
    def _preprocess(self) -> pd.DataFrame:
        """Clean and preprocess the dataframe."""
        df = self.df.copy()
        
        # Remove rows with missing SMILES
        df = df.dropna(subset=["SMILES"])
        
        # Validate SMILES
        valid_smiles = []
        valid_indices = []
        for idx, smi in enumerate(df["SMILES"]):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles.append(smi)
                valid_indices.append(idx)
        
        df = df.iloc[valid_indices].reset_index(drop=True)
        return df
    
    def _prepare_molecules(self):
        """Pre-compute molecular representations."""
        for idx, smi in enumerate(self.smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if self.remove_hs:
                mol = Chem.RemoveHs(mol)
            self.mol_dict[idx] = mol
            
            # Generate 3D conformer
            if self.use_3d:
                mol_3d = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
                AllChem.UFFOptimizeMolecule(mol_3d)
                self.conformer_dict[idx] = mol_3d
            
            # Compute fingerprint - using new MorganGenerator API
            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fp_arr = fpgen.GetFingerprintAsNumPy(mol).astype(np.float32)
            self.fingerprint_dict[idx] = fp_arr
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dictionary with:
        - smiles: SMILES string
        - mol: RDKit Mol object
        - conformer: 3D conformer (if use_3d)
        - fingerprint: Morgan fingerprint
        - formulation: formulation parameters
        - target: target value(s)
        """
        item = {
            "idx": idx,
            "smiles": self.smiles_list[idx],
            "mol": self.mol_dict[idx],
            "fingerprint": self.fingerprint_dict[idx],
        }
        
        if self.use_3d:
            item["conformer"] = self.conformer_dict[idx]
        
        # Add formulation parameters
        if "molar_ratios" in self.df.columns:
            item["molar_ratios"] = eval(self.df.iloc[idx]["molar_ratios"])
        
        if "helper_lipid" in self.df.columns:
            item["helper_lipid"] = self.df.iloc[idx]["helper_lipid"]
        
        if "cholesterol" in self.df.columns:
            item["cholesterol"] = self.df.iloc[idx]["cholesterol"]
        
        if "PEG" in self.df.columns:
            item["PEG"] = self.df.iloc[idx]["PEG"]
        
        # Add targets
        if self.targets is not None:
            item["target"] = self.targets[idx]
        
        if "particle_size" in self.df.columns:
            item["particle_size"] = self.df.iloc[idx]["particle_size"]
        
        if "zeta_potential" in self.df.columns:
            item["zeta_potential"] = self.df.iloc[idx]["zeta_potential"]
        
        if "toxicity" in self.df.columns:
            item["toxicity"] = self.df.iloc[idx]["toxicity"]
        
        if self.transform:
            item = self.transform(item)
        
        return item


class FormulationDataset(Dataset):
    """
    Dataset for multi-component LNP formulations.
    Extends LNPDataset to handle arbitrary number of components.
    """
    
    def __init__(
        self,
        data_path: str,
        component_columns: List[str] = None,
        ratio_column: str = "molar_ratios",
        **kwargs
    ):
        """
        Args:
            data_path: Path to formulation data
            component_columns: List of columns containing component SMILES
            ratio_column: Column containing molar ratios
            **kwargs: Arguments passed to LNPDataset
        """
        super().__init__(data_path, **kwargs)
        
        self.component_columns = component_columns or ["SMILES"]
        self.ratio_column = ratio_column
        
    def __getitem__(self, idx: int) -> Dict:
        item = super().__getitem__(idx)
        
        # Handle multiple components
        components = []
        for col in self.component_columns:
            if col in self.df.columns:
                smi = self.df.iloc[idx][col]
                mol = Chem.MolFromSmiles(smi)
                components.append(mol)
        
        item["components"] = components
        item["num_components"] = len(components)
        
        return item


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching LNP data.
    """
    result = {}
    
    # Stack simple arrays
    for key in batch[0].keys():
        if key == "mol" or key == "conformer":
            # Keep as list for RDKit objects
            result[key] = [item[key] for item in batch]
        elif isinstance(batch[0][key], np.ndarray):
            result[key] = np.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], list):
            result[key] = [item[key] for item in batch]
        else:
            result[key] = [item[key] for item in batch]
    
    return result


def build_dataloader(
    dataset: LNPDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """Build DataLoader with custom collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs
    )
