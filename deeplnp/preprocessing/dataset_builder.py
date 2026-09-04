"""
Dataset builder for DeepLNP.
Integrates all preprocessing steps into a unified pipeline.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

from .molecular_featurization import MolecularFeaturizer
from .formulation_tokenizer import FormulationTokenizer
from .physchem_descriptor import PhysChemDescriptor


class DatasetBuilder:
    """
    Build processed datasets for DeepLNP training.
    
    Pipeline:
    1. Load raw data
    2. Clean and validate
    3. Compute molecular features (1D/2D/3D)
    4. Compute formulation features
    5. Compute physicochemical descriptors
    6. Save processed data
    """
    
    def __init__(
        self,
        output_dir: str = "data/processed",
        use_3d: bool = True,
        use_graph: bool = True,
        fingerprint_size: int = 2048,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.mol_featurizer = MolecularFeaturizer(
            use_1d=True,
            use_2d=use_graph,
            use_3d=use_3d,
            fingerprint_size=fingerprint_size,
        )
        
        self.form_tokenizer = FormulationTokenizer()
        self.physchem = PhysChemDescriptor()
        
    def build(
        self,
        input_path: str,
        output_name: str = "processed_dataset",
        smiles_col: str = "SMILES",
        target_col: str = "target",
        has_formulation: bool = True,
    ) -> Dict:
        """
        Build complete processed dataset.
        
        Args:
            input_path: Path to raw CSV
            output_name: Name for output files
            smiles_col: Column name for SMILES
            target_col: Column name for target variable
            has_formulation: Whether data includes formulation parameters
        
        Returns:
            Dictionary with paths to processed files
        """
        # Load data
        print(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path)
        
        # Clean data
        print("Cleaning data...")
        df = self._clean_data(df, smiles_col, target_col)
        
        # Process molecules
        print("Computing molecular features...")
        mol_features = self._process_molecules(df[smiles_col].tolist())
        
        # Process formulations
        if has_formulation:
            print("Computing formulation features...")
            form_features = self._process_formulations(df)
        else:
            form_features = None
        
        # Compute descriptors
        print("Computing physicochemical descriptors...")
        descriptors = self._process_descriptors(df[smiles_col].tolist())
        
        # Save
        print(f"Saving to {self.output_dir}...")
        output_files = self._save_data(
            df, mol_features, form_features, descriptors, output_name
        )
        
        print(f"✅ Dataset built successfully!")
        print(f"  Samples: {len(df)}")
        print(f"  Features: {mol_features['fingerprints'].shape[1]}")
        
        return output_files
    
    def _clean_data(
        self,
        df: pd.DataFrame,
        smiles_col: str,
        target_col: str
    ) -> pd.DataFrame:
        """Clean and validate data."""
        # Remove missing values
        df = df.dropna(subset=[smiles_col, target_col])
        
        # Validate SMILES
        valid_indices = []
        for idx, smi in enumerate(df[smiles_col]):
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_indices.append(idx)
        
        df = df.iloc[valid_indices].reset_index(drop=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=[smiles_col])
        
        return df.reset_index(drop=True)
    
    def _process_molecules(self, smiles_list: List[str]) -> Dict:
        """Process all molecules."""
        fingerprints = []
        graphs = {"atom_features": [], "bond_features": [], "edge_index": []}
        conformers = {"coords": [], "atom_types": []}
        tokens = []
        
        for smi in tqdm(smiles_list, desc="Featurizing molecules"):
            try:
                feat = self.mol_featurizer.featurize(smi)
                
                fingerprints.append(feat["fingerprint"])
                
                if "graph" in feat:
                    graphs["atom_features"].append(feat["graph"]["atom_features"])
                    graphs["bond_features"].append(feat["graph"]["bond_features"])
                    graphs["edge_index"].append(feat["graph"]["edge_index"])
                
                if "conformer" in feat:
                    conformers["coords"].append(feat["conformer"]["coords"])
                    conformers["atom_types"].append(feat["conformer"]["atom_types"])
                
                if "tokens" in feat:
                    tokens.append(feat["tokens"])
                    
            except Exception as e:
                print(f"Error processing {smi}: {e}")
                continue
        
        result = {"fingerprints": np.array(fingerprints)}
        
        if graphs["atom_features"]:
            result["graphs"] = graphs
        
        if conformers["coords"]:
            result["conformers"] = conformers
        
        if tokens:
            result["tokens"] = tokens
        
        return result
    
    def _process_formulations(self, df: pd.DataFrame) -> Dict:
        """Process formulation data."""
        formulations = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing formulations"):
            form = {}
            
            # Extract components if available
            if "molar_ratios" in df.columns:
                form["ratios"] = eval(row["molar_ratios"])
            
            if "helper_lipid" in df.columns:
                form["components"] = [row["SMILES"], row["helper_lipid"]]
                form["types"] = ["ionizable", "helper"]
            
            formulations.append(form)
        
        tokenized = self.form_tokenizer.tokenize_batch(formulations)
        return {"formulations": tokenized}
    
    def _process_descriptors(self, smiles_list: List[str]) -> Dict:
        """Compute physicochemical descriptors."""
        descriptors = []
        
        from rdkit import Chem
        
        for smi in tqdm(smiles_list, desc="Computing descriptors"):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            
            desc = self.physchem.compute_all(mol)
            descriptors.append(desc)
        
        # Convert to array
        desc_array = np.array([
            list(d.values()) for d in descriptors
        ])
        
        return {
            "descriptors": desc_array,
            "descriptor_names": list(descriptors[0].keys()) if descriptors else [],
        }
    
    def _save_data(
        self,
        df: pd.DataFrame,
        mol_features: Dict,
        form_features: Optional[Dict],
        descriptors: Dict,
        output_name: str,
    ) -> Dict:
        """Save processed data."""
        output_files = {}
        
        # Save metadata
        metadata = {
            "num_samples": len(df),
            "columns": df.columns.tolist(),
            "smiles_list": df["SMILES"].tolist(),
        }
        
        metadata_path = self.output_dir / f"{output_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        output_files["metadata"] = metadata_path
        
        # Save targets
        if "target" in df.columns:
            targets = df["target"].values
            np.save(self.output_dir / f"{output_name}_targets.npy", targets)
            output_files["targets"] = self.output_dir / f"{output_name}_targets.npy"
        
        # Save molecular features
        np.save(
            self.output_dir / f"{output_name}_fingerprints.npy",
            mol_features["fingerprints"]
        )
        output_files["fingerprints"] = self.output_dir / f"{output_name}_fingerprints.npy"
        
        if "graphs" in mol_features:
            with open(self.output_dir / f"{output_name}_graphs.pkl", "wb") as f:
                pickle.dump(mol_features["graphs"], f)
            output_files["graphs"] = self.output_dir / f"{output_name}_graphs.pkl"
        
        if "conformers" in mol_features:
            with open(self.output_dir / f"{output_name}_conformers.pkl", "wb") as f:
                pickle.dump(mol_features["conformers"], f)
            output_files["conformers"] = self.output_dir / f"{output_name}_conformers.pkl"
        
        # Save descriptors
        np.save(
            self.output_dir / f"{output_name}_descriptors.npy",
            descriptors["descriptors"]
        )
        with open(self.output_dir / f"{output_name}_descriptor_names.json", "w") as f:
            json.dump(descriptors["descriptor_names"], f)
        output_files["descriptors"] = self.output_dir / f"{output_name}_descriptors.npy"
        
        # Save formulation features if available
        if form_features:
            with open(self.output_dir / f"{output_name}_formulations.pkl", "wb") as f:
                pickle.dump(form_features, f)
            output_files["formulations"] = self.output_dir / f"{output_name}_formulations.pkl"
        
        # Save dataframe
        df.to_csv(self.output_dir / f"{output_name}.csv", index=False)
        output_files["dataframe"] = self.output_dir / f"{output_name}.csv"
        
        return output_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build processed dataset for DeepLNP")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--name", type=str, default="dataset", help="Output name")
    parser.add_argument("--no-3d", action="store_true", help="Skip 3D conformer generation")
    
    args = parser.parse_args()
    
    builder = DatasetBuilder(output_dir=args.output, use_3d=not args.no_3d)
    builder.build(args.input, output_name=args.name)
