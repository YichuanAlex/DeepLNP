"""
Molecular featurization with multi-modal representations.
Supports 1D (SMILES), 2D (graph), and 3D (conformer) features.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator


class MolecularFeaturizer:
    """
    Multi-modal molecular featurization for ionizable lipids.
    
    Combines:
    - 1D: SMILES tokens (for Transformer)
    - 2D: Molecular graph (for GNN)
    - 3D: Conformer coordinates (for Uni-Mol/3D Transformer)
    - Fingerprints: Morgan, RDKit, etc.
    """
    
    def __init__(
        self,
        use_1d: bool = True,
        use_2d: bool = True,
        use_3d: bool = True,
        fingerprint_type: str = "morgan",
        fingerprint_size: int = 2048,
        max_atoms: int = 256,
        remove_hs: bool = False,
    ):
        """
        Args:
            use_1d: Use SMILES tokenization
            use_2d: Use molecular graph representation
            use_3d: Use 3D conformer coordinates
            fingerprint_type: Type of fingerprint ('morgan', 'rdkit', 'avalon')
            fingerprint_size: Size of fingerprint vector
            max_atoms: Maximum number of atoms to consider
            remove_hs: Remove hydrogens from representation
        """
        self.use_1d = use_1d
        self.use_2d = use_2d
        self.use_3d = use_3d
        self.fingerprint_type = fingerprint_type
        self.fingerprint_size = fingerprint_size
        self.max_atoms = max_atoms
        self.remove_hs = remove_hs
        
        # Atom features for graph representation
        self.atom_feature_dim = self._get_atom_feature_dim()
        
    def _get_atom_feature_dim(self) -> int:
        """Calculate dimension of atom features."""
        # One-hot encoding for various atom properties
        return (
            10 +  # Atomic number (H, C, N, O, F, P, S, Cl, Br, I, other)
            3 +   # Degree (0-1, 2-3, 4+)
            3 +   # Total H count (0, 1-2, 3+)
            2 +   # Formal charge (negative, zero, positive)
            2 +   # Hybridization (sp, sp2, sp3, other)
            2 +   # Aromatic (yes/no)
            2 +   # Chirality (R, S, none)
            1     # Radical electrons
        )
    
    def featurize(self, smiles: str) -> Dict:
        """
        Compute multi-modal features for a molecule.
        
        Args:
            smiles: SMILES string
        
        Returns:
            Dictionary containing:
            - smiles: original SMILES string
            - tokens: tokenized SMILES (if use_1d)
            - graph: molecular graph (if use_2d)
            - conformer: 3D coordinates (if use_3d)
            - fingerprint: molecular fingerprint
            - descriptors: physicochemical descriptors
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        if self.remove_hs:
            mol = Chem.RemoveHs(mol)
        
        features = {"smiles": smiles, "mol": mol}
        
        # 1D: Tokenized SMILES
        if self.use_1d:
            features["tokens"] = self._tokenize_smiles(smiles)
        
        # 2D: Molecular graph
        if self.use_2d:
            features["graph"] = self._build_graph(mol)
        
        # 3D: Conformer coordinates
        if self.use_3d:
            features["conformer"] = self._generate_conformer(mol)
        
        # Fingerprint
        features["fingerprint"] = self._compute_fingerprint(mol)
        
        # Physicochemical descriptors
        features["descriptors"] = self._compute_descriptors(mol)
        
        return features
    
    def _tokenize_smiles(self, smiles: str) -> List[str]:
        """Tokenize SMILES string into characters/tokens."""
        # Simple character-level tokenization
        # Can be extended to substructure tokens
        tokens = list(smiles)
        return tokens
    
    def _build_graph(self, mol: Chem.Mol) -> Dict:
        """
        Build molecular graph representation.
        
        Returns:
            Dictionary with:
            - atom_features: (N, atom_feature_dim) array
            - bond_features: (E, bond_feature_dim) array
            - edge_index: (2, E) connectivity matrix
        """
        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            feat = self._atom_to_feature(atom)
            atom_features.append(feat)
        
        atom_features = np.array(atom_features, dtype=np.float32)
        
        # Bond features and connectivity
        bond_features = []
        edge_index = [[], []]
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            bond_feat = self._bond_to_feature(bond)
            
            # Add both directions (undirected graph)
            bond_features.append(bond_feat)
            bond_features.append(bond_feat)  # Symmetric
            
            edge_index[0].append(i)
            edge_index[1].append(j)
            edge_index[0].append(j)
            edge_index[1].append(i)
        
        if len(bond_features) > 0:
            bond_features = np.array(bond_features, dtype=np.float32)
            edge_index = np.array(edge_index, dtype=np.int64)
        else:
            # Handle molecules with no bonds
            bond_features = np.zeros((0, 4), dtype=np.float32)
            edge_index = np.zeros((2, 0), dtype=np.int64)
        
        return {
            "atom_features": atom_features,
            "bond_features": bond_features,
            "edge_index": edge_index,
        }
    
    def _atom_to_feature(self, atom: Chem.Atom) -> np.ndarray:
        """Convert RDKit atom to feature vector."""
        feat = []
        
        # Atomic number (one-hot)
        atomic_num = atom.GetAtomicNum()
        atomic_nums = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # Common elements
        atomic_onehot = [1 if atomic_num == z else 0 for z in atomic_nums]
        if sum(atomic_onehot) == 0:
            atomic_onehot = [0] * len(atomic_nums)  # Other
        feat.extend(atomic_onehot)
        
        # Degree
        degree = atom.GetDegree()
        feat.extend([
            1 if degree <= 1 else 0,
            1 if 2 <= degree <= 3 else 0,
            1 if degree >= 4 else 0,
        ])
        
        # Total H count
        total_h = atom.GetTotalNumHs()
        feat.extend([
            1 if total_h == 0 else 0,
            1 if 1 <= total_h <= 2 else 0,
            1 if total_h >= 3 else 0,
        ])
        
        # Formal charge
        charge = atom.GetFormalCharge()
        feat.extend([
            1 if charge < 0 else 0,
            1 if charge == 0 else 0,
            1 if charge > 0 else 0,
        ])
        
        # Hybridization
        hybrid = atom.GetHybridization()
        hybrid_types = [
            Chem.HybridizationType.SP,
            Chem.HybridizationType.SP2,
            Chem.HybridizationType.SP3,
        ]
        feat.extend([1 if hybrid == h else 0 for h in hybrid_types])
        feat.extend([1 if hybrid not in hybrid_types else 0])
        
        # Aromatic
        feat.extend([1 if atom.GetIsAromatic() else 0])
        
        # Chirality
        chirality = atom.GetChiralTag()
        feat.extend([
            1 if chirality == Chem.ChiralType.CHI_R else 0,
            1 if chirality == Chem.ChiralType.CHI_S else 0,
        ])
        
        # Radical electrons
        feat.extend([atom.GetNumRadicalElectrons()])
        
        return np.array(feat, dtype=np.float32)
    
    def _bond_to_feature(self, bond: Chem.Bond) -> np.ndarray:
        """Convert RDKit bond to feature vector."""
        feat = []
        
        # Bond type
        bond_type = bond.GetBondType()
        bond_types = [
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
            Chem.BondType.AROMATIC,
        ]
        feat.extend([1 if bond_type == bt else 0 for bt in bond_types])
        
        # Conjugated
        feat.extend([1 if bond.GetIsConjugated() else 0])
        
        # In ring
        feat.extend([1 if bond.IsInRing() else 0])
        
        # Stereo
        stereo = bond.GetStereo()
        feat.extend([1 if stereo != Chem.BondStereo.STEREONONE else 0])
        
        return np.array(feat, dtype=np.float32)
    
    def _generate_conformer(self, mol: Chem.Mol) -> Dict:
        """
        Generate 3D conformer for a molecule.
        
        Returns:
            Dictionary with:
            - coords: (N, 3) array of atomic coordinates
            - atom_types: (N,) array of atomic numbers
        """
        mol_3d = Chem.AddHs(mol)
        
        # Generate conformer using ETKDG
        params = AllChem.ETKDG()
        params.randomSeed = 42
        success = AllChem.EmbedMolecule(mol_3d, params)
        
        if success == -1:
            # Fallback to random embedding
            AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
        
        # Optimize geometry
        AllChem.UFFOptimizeMolecule(mol_3d)
        
        # Extract coordinates
        conf = mol_3d.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol_3d.GetNumAtoms())])
        atom_types = np.array([atom.GetAtomicNum() for atom in mol_3d.GetAtoms()])
        
        if self.remove_hs:
            # Remove hydrogen coordinates
            non_h_indices = [i for i, z in enumerate(atom_types) if z != 1]
            coords = coords[non_h_indices]
            atom_types = atom_types[non_h_indices]
        
        # Limit to max_atoms
        if len(coords) > self.max_atoms:
            coords = coords[:self.max_atoms]
            atom_types = atom_types[:self.max_atoms]
        
        return {
            "coords": coords.astype(np.float32),
            "atom_types": atom_types.astype(np.int64),
        }
    
    def _compute_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """Compute molecular fingerprint."""
        if self.fingerprint_type == "morgan":
            # Use new MorganGenerator API to avoid deprecation warnings
            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=self.fingerprint_size)
            arr = fpgen.GetFingerprintAsNumPy(mol).astype(np.float32)
        elif self.fingerprint_type == "rdkit":
            fp = Chem.RDKFingerprint(mol, fpSize=self.fingerprint_size)
            arr = np.zeros((self.fingerprint_size,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
        elif self.fingerprint_type == "avalon":
            from rdkit.Avalon import pyAvalonTools
            fp = pyAvalonTools.GetAvalonFingerprint(mol, nBits=self.fingerprint_size)
            arr = np.zeros((self.fingerprint_size,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
        else:
            raise ValueError(f"Unknown fingerprint type: {self.fingerprint_type}")
        
        return arr
    
    def _compute_descriptors(self, mol: Chem.Mol) -> Dict:
        """Compute physicochemical descriptors."""
        descriptors = {}
        
        # Basic properties
        descriptors["mol_wt"] = Descriptors.MolWt(mol)
        descriptors["logp"] = Descriptors.MolLogP(mol)
        descriptors["donors"] = Lipinski.NumHDonors(mol)
        descriptors["acceptors"] = Lipinski.NumHAcceptors(mol)
        descriptors["rotatable_bonds"] = Lipinski.NumRotatableBonds(mol)
        descriptors["tpsa"] = Descriptors.TPSA(mol)
        
        # Lipid-specific descriptors
        descriptors["num_rings"] = rdMolDescriptors.CalcNumRings(mol)
        descriptors["num_aromatic_rings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
        descriptors["num_aliphatic_rings"] = rdMolDescriptors.CalcNumAliphaticRings(mol)
        
        # Ionizable lipid specific
        descriptors["num_tertiary_amines"] = self._count_tertiary_amines(mol)
        descriptors["num_ester_bonds"] = self._count_ester_bonds(mol)
        
        return descriptors
    
    def _count_tertiary_amines(self, mol: Chem.Mol) -> int:
        """Count tertiary amine groups (important for ionizable lipids)."""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:  # Nitrogen
                if atom.GetDegree() == 3:  # Connected to 3 atoms
                    # Check if all neighbors are carbons
                    neighbors = [mol.GetAtomWithIdx(idx) for idx in atom.GetNeighbors()]
                    if all(n.GetAtomicNum() == 6 for n in neighbors):
                        count += 1
        return count
    
    def _count_ester_bonds(self, mol: Chem.Mol) -> int:
        """Count ester bonds (important for biodegradability)."""
        # SMARTS pattern for ester bonds
        ester_pattern = Chem.MolFromSmarts("[CX3](=[OX1])[OX2][#6]")
        if ester_pattern is None:
            return 0
        return len(mol.GetSubstructMatches(ester_pattern))
    
    def featurize_batch(self, smiles_list: List[str]) -> List[Dict]:
        """Featurize a batch of SMILES strings."""
        return [self.featurize(smi) for smi in smiles_list]
