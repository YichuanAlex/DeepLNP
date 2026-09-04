"""
Formulation tokenizer for multi-component LNP systems.
Encodes molar ratios, component types, and preparation conditions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class FormulationTokenizer:
    """
    Tokenize and encode LNP formulation parameters.
    
    Handles:
    - Molar ratios of components
    - Component types (ionizable, helper, cholesterol, PEG)
    - Preparation conditions (N/P ratio, flow rate, etc.)
    """
    
    def __init__(
        self,
        max_components: int = 4,
        ratio_encoding: str = "continuous",
        include_conditions: bool = True,
    ):
        """
        Args:
            max_components: Maximum number of components in formulation
            ratio_encoding: How to encode ratios ('continuous', 'discrete', 'log')
            include_conditions: Include preparation conditions
        """
        self.max_components = max_components
        self.ratio_encoding = ratio_encoding
        self.include_conditions = include_conditions
        
        # Component type vocabulary
        self.component_vocab = {
            "ionizable": 0,
            "helper": 1,
            "cholesterol": 2,
            "peg": 3,
            "targeting": 4,
            "unknown": 5,
        }
        
    def tokenize(self, formulation: Dict) -> Dict:
        """
        Tokenize a formulation into model-ready format.
        
        Args:
            formulation: Dictionary with:
                - components: List of component SMILES
                - ratios: List of molar ratios
                - types: List of component types
                - conditions: Preparation conditions (optional)
        
        Returns:
            Dictionary with encoded formulation features
        """
        components = formulation.get("components", [])
        ratios = formulation.get("ratios", [])
        types = formulation.get("types", [])
        
        # Encode components
        encoded = {
            "component_tokens": self._encode_components(components, types),
            "ratio_features": self._encode_ratios(ratios),
        }
        
        # Add conditions if available
        if self.include_conditions and "conditions" in formulation:
            encoded["condition_features"] = self._encode_conditions(
                formulation["conditions"]
            )
        
        return encoded
    
    def _encode_components(
        self,
        components: List[str],
        types: List[str]
    ) -> np.ndarray:
        """Encode component types and identities."""
        # Pad or truncate to max_components
        n_comp = min(len(components), self.max_components)
        
        # Component type one-hot encoding
        type_matrix = np.zeros((self.max_components, len(self.component_vocab)))
        for i, comp_type in enumerate(types[:n_comp]):
            type_idx = self.component_vocab.get(comp_type.lower(), 5)
            type_matrix[i, type_idx] = 1.0
        
        # Component presence mask
        presence = np.zeros(self.max_components)
        presence[:n_comp] = 1.0
        
        return {
            "type_matrix": type_matrix,
            "presence": presence,
            "n_components": n_comp,
        }
    
    def _encode_ratios(self, ratios: List[float]) -> np.ndarray:
        """Encode molar ratios."""
        if len(ratios) == 0:
            return np.zeros(self.max_components)
        
        ratios = np.array(ratios[:self.max_components])
        
        # Normalize to sum to 1
        ratios = ratios / ratios.sum()
        
        # Pad to max_components
        if len(ratios) < self.max_components:
            ratios = np.pad(
                ratios,
                (0, self.max_components - len(ratios)),
                mode="constant"
            )
        
        # Apply encoding
        if self.ratio_encoding == "log":
            # Log transform (useful for wide range of ratios)
            ratios = np.log1p(ratios * 100) / np.log1p(100)
        elif self.ratio_encoding == "discrete":
            # Discretize into bins
            bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ratios = np.digitize(ratios, bins) / len(bins)
        
        return ratios.astype(np.float32)
    
    def _encode_conditions(self, conditions: Dict) -> np.ndarray:
        """Encode preparation conditions."""
        features = []
        
        # N/P ratio (important for mRNA complexation)
        np_ratio = conditions.get("np_ratio", 0)
        features.append(np_ratio / 10.0)  # Normalize
        
        # Flow rate ratio (for microfluidics)
        frr = conditions.get("flow_rate_ratio", 1)
        features.append(frr / 5.0)
        
        # Total flow rate
        tfr = conditions.get("total_flow_rate", 1)
        features.append(tfr / 20.0)
        
        # Temperature
        temp = conditions.get("temperature", 25)
        features.append((temp - 20) / 40.0)  # Center around 20°C
        
        # pH
        ph = conditions.get("ph", 7.4)
        features.append((ph - 7.0) / 2.0)
        
        return np.array(features, dtype=np.float32)
    
    def tokenize_batch(self, formulations: List[Dict]) -> List[Dict]:
        """Tokenize a batch of formulations."""
        return [self.tokenize(form) for form in formulations]


def create_formulation_features(
    ionizable_smiles: str,
    helper_smiles: Optional[str] = None,
    cholesterol_type: str = "standard",
    peg_type: str = "DMG-PEG2000",
    molar_ratios: List[float] = [50, 10, 38.5, 1.5],
    conditions: Optional[Dict] = None,
) -> Dict:
    """
    Helper function to create formulation features.
    
    Args:
        ionizable_smiles: SMILES of ionizable lipid
        helper_smiles: SMILES of helper lipid (optional)
        cholesterol_type: Type of cholesterol
        peg_type: Type of PEG-lipid
        molar_ratios: Molar ratios [ionizable, helper, cholesterol, PEG]
        conditions: Preparation conditions
    
    Returns:
        Formulation dictionary ready for tokenization
    """
    components = [ionizable_smiles]
    types = ["ionizable"]
    
    if helper_smiles:
        components.append(helper_smiles)
        types.append("helper")
    
    # Add cholesterol (simplified as placeholder)
    components.append(f"[cholesterol:{cholesterol_type}]")
    types.append("cholesterol")
    
    # Add PEG
    components.append(f"[peg:{peg_type}]")
    types.append("peg")
    
    formulation = {
        "components": components,
        "ratios": molar_ratios,
        "types": types,
        "conditions": conditions or {},
    }
    
    return formulation
