"""
Physicochemical descriptor computation for LNP components.
Includes pKa prediction, LogP, and other lipid-specific properties.
"""

import numpy as np
from typing import Dict, List, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


class PhysChemDescriptor:
    """
    Compute physicochemical descriptors critical for LNP design.
    
    Focus on properties that affect:
    - Endosomal escape (pKa)
    - Membrane fusion (LogP, packing parameter)
    - Biodegradability (ester count, hydrolysis rate)
    - Toxicity (cationic charge density)
    """
    
    def __init__(self):
        # pKa calculation parameters (simplified)
        self.basic_pka = {
            "tertiary_amine": 6.5,
            "secondary_amine": 10.5,
            "primary_amine": 9.5,
            "pyridine": 5.2,
            "imidazole": 6.9,
        }
        
    def compute_all(self, mol: Chem.Mol) -> Dict:
        """
        Compute all physicochemical descriptors.
        
        Args:
            mol: RDKit Mol object
        
        Returns:
            Dictionary of descriptors
        """
        descriptors = {}
        
        # Basic properties
        descriptors.update(self._basic_properties(mol))
        
        # Lipid-specific properties
        descriptors.update(self._lipid_properties(mol))
        
        # Ionization properties
        descriptors.update(self._ionization_properties(mol))
        
        # Structural properties
        descriptors.update(self._structural_properties(mol))
        
        return descriptors
    
    def _basic_properties(self, mol: Chem.Mol) -> Dict:
        """Compute basic physicochemical properties."""
        return {
            "mol_wt": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "logp_crippen": Descriptors.MolLogP(mol),  # Crippen's LogP
            "tpsa": Descriptors.TPSA(mol),
            "num_donors": Lipinski.NumHDonors(mol),
            "num_acceptors": Lipinski.NumHAcceptors(mol),
            "num_rotatable_bonds": Lipinski.NumRotatableBonds(mol),
            "num_rings": rdMolDescriptors.CalcNumRings(mol),
        }
    
    def _lipid_properties(self, mol: Chem.Mol) -> Dict:
        """Compute lipid-specific properties."""
        # Packing parameter (estimates membrane curvature preference)
        # P = v / (a0 * lc), where:
        # v = hydrophobic volume
        # a0 = headgroup area
        # lc = critical chain length
        
        # Estimate hydrophobic volume from LogP
        logp = Descriptors.MolLogP(mol)
        hydrophobic_volume = logp * 10  # Rough approximation
        
        # Estimate headgroup area from TPSA
        tpsa = Descriptors.TPSA(mol)
        headgroup_area = tpsa / 10  # Rough approximation
        
        # Estimate chain length from molecular weight
        mol_wt = Descriptors.MolWt(mol)
        chain_length = mol_wt / 50  # Very rough approximation
        
        packing_parameter = hydrophobic_volume / (headgroup_area * chain_length + 1e-6)
        
        # Count lipid-like features
        num_long_chains = self._count_long_alkyl_chains(mol)
        num_ester_bonds = self._count_ester_bonds(mol)
        num_amide_bonds = self._count_amide_bonds(mol)
        
        return {
            "packing_parameter": packing_parameter,
            "hydrophobic_volume": hydrophobic_volume,
            "headgroup_area": headgroup_area,
            "chain_length": chain_length,
            "num_long_chains": num_long_chains,
            "num_ester_bonds": num_ester_bonds,
            "num_amide_bonds": num_amide_bonds,
            "biodegradability_score": num_ester_bonds + num_amide_bonds * 0.5,
        }
    
    def _ionization_properties(self, mol: Chem.Mol) -> Dict:
        """Compute ionization-related properties (critical for ionizable lipids)."""
        # Count ionizable groups
        num_tertiary_amines = self._count_tertiary_amines(mol)
        num_secondary_amines = self._count_secondary_amines(mol)
        num_primary_amines = self._count_primary_amines(mol)
        
        # Estimate pKa (simplified calculation)
        estimated_pka = self._estimate_pka(mol)
        
        # Charge density at physiological pH
        charge_at_ph74 = self._estimate_charge(mol, ph=7.4)
        charge_at_ph55 = self._estimate_charge(mol, ph=5.5)  # Endosomal pH
        
        return {
            "num_tertiary_amines": num_tertiary_amines,
            "num_secondary_amines": num_secondary_amines,
            "num_primary_amines": num_primary_amines,
            "num_ionizable_groups": num_tertiary_amines + num_secondary_amines + num_primary_amines,
            "estimated_pka": estimated_pka,
            "charge_at_ph74": charge_at_ph74,
            "charge_at_ph55": charge_at_ph55,
            "protonation_ratio": charge_at_ph55 / (charge_at_ph74 + 1e-6),
        }
    
    def _structural_properties(self, mol: Chem.Mol) -> Dict:
        """Compute structural descriptors."""
        # Molecular flexibility
        num_rotatable = Lipinski.NumRotatableBonds(mol)
        flexibility_index = num_rotatable / (Descriptors.MolWt(mol) / 100 + 1e-6)
        
        # Aromaticity
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        num_aliphatic_rings = rdMolDescriptors.CalcNumAliphaticRings(mol)
        aromatic_ratio = num_aromatic_rings / (num_aromatic_rings + num_aliphatic_rings + 1e-6)
        
        # Symmetry
        symmetry_score = self._estimate_symmetry(mol)
        
        # Drug-likeness scores
        qed = Descriptors.qed(mol)
        sa_score = rdMolDescriptors.CalcSyntheticAccessibility(mol)
        
        return {
            "flexibility_index": flexibility_index,
            "num_aromatic_rings": num_aromatic_rings,
            "num_aliphatic_rings": num_aliphatic_rings,
            "aromatic_ratio": aromatic_ratio,
            "symmetry_score": symmetry_score,
            "qed": qed,
            "synthetic_accessibility": sa_score,
        }
    
    def _count_long_alkyl_chains(self, mol: Chem.Mol, min_length: int = 6) -> int:
        """Count alkyl chains with at least min_length carbons."""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:  # Carbon
                # Check if it's at the end of a chain
                if atom.GetDegree() == 1:
                    # Trace the chain
                    chain_length = self._trace_alkyl_chain(mol, atom.GetIdx())
                    if chain_length >= min_length:
                        count += 1
        return count
    
    def _trace_alkyl_chain(self, mol: Chem.Mol, start_idx: int) -> int:
        """Trace an alkyl chain from a starting carbon."""
        visited = set()
        stack = [start_idx]
        length = 0
        
        while stack:
            idx = stack.pop()
            if idx in visited:
                continue
            visited.add(idx)
            
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() == 6:  # Carbon
                length += 1
                
                # Add neighbors
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetIdx() not in visited:
                        stack.append(neighbor.GetIdx())
        
        return length
    
    def _count_ester_bonds(self, mol: Chem.Mol) -> int:
        """Count ester bonds."""
        pattern = Chem.MolFromSmarts("[CX3](=[OX1])[OX2][#6]")
        return len(mol.GetSubstructMatches(pattern)) if pattern else 0
    
    def _count_amide_bonds(self, mol: Chem.Mol) -> int:
        """Count amide bonds."""
        pattern = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")
        return len(mol.GetSubstructMatches(pattern)) if pattern else 0
    
    def _count_tertiary_amines(self, mol: Chem.Mol) -> int:
        """Count tertiary amines."""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:  # Nitrogen
                if atom.GetDegree() == 3:
                    neighbors = [mol.GetAtomWithIdx(idx) for idx in atom.GetNeighbors()]
                    if all(n.GetAtomicNum() == 6 for n in neighbors):
                        count += 1
        return count
    
    def _count_secondary_amines(self, mol: Chem.Mol) -> int:
        """Count secondary amines."""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:  # Nitrogen
                if atom.GetDegree() == 2:
                    neighbors = [mol.GetAtomWithIdx(idx) for idx in atom.GetNeighbors()]
                    if all(n.GetAtomicNum() == 6 for n in neighbors):
                        count += 1
        return count
    
    def _count_primary_amines(self, mol: Chem.Mol) -> int:
        """Count primary amines."""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:  # Nitrogen
                if atom.GetDegree() == 1:
                    count += 1
        return count
    
    def _estimate_pka(self, mol: Chem.Mol) -> float:
        """
        Estimate pKa of ionizable lipid.
        This is a simplified calculation - in practice, use quantum chemistry or ML.
        """
        # Count different amine types
        n_tertiary = self._count_tertiary_amines(mol)
        n_secondary = self._count_secondary_amines(mol)
        n_primary = self._count_primary_amines(mol)
        
        if n_tertiary + n_secondary + n_primary == 0:
            return 7.0  # Default for non-ionizable
        
        # Weighted average of typical pKa values
        pka = (
            n_tertiary * 6.5 +
            n_secondary * 10.5 +
            n_primary * 9.5
        ) / (n_tertiary + n_secondary + n_primary + 1e-6)
        
        # Adjust for electron-withdrawing groups
        num_ester = self._count_ester_bonds(mol)
        pka -= num_ester * 0.3  # Esters decrease pKa
        
        return pka
    
    def _estimate_charge(self, mol: Chem.Mol, ph: float = 7.4) -> float:
        """
        Estimate net charge at given pH using Henderson-Hasselbalch equation.
        """
        pka = self._estimate_pka(mol)
        n_ionizable = (
            self._count_tertiary_amines(mol) +
            self._count_secondary_amines(mol) +
            self._count_primary_amines(mol)
        )
        
        # Fraction protonated
        ratio = 10 ** (pka - ph)
        fraction_protonated = ratio / (1 + ratio)
        
        return n_ionizable * fraction_protonated
    
    def _estimate_symmetry(self, mol: Chem.Mol) -> float:
        """Estimate molecular symmetry (0 = asymmetric, 1 = highly symmetric)."""
        # Simplified: use number of identical fragments
        # In practice, use point group symmetry
        return 0.5  # Placeholder
