"""Leakage-safe, target-conditioned dataset for formulation-level DeepLNP models."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler


TASK_COLUMNS = {
    "efficiency": ["quantified_delivery", "transfection_efficiency"],
    "particle_size": ["particle_size_nm_std", "particle_size_nm", "particle_size"],
    "zeta_potential": ["zeta_potential_mv_std", "zeta_potential_mv", "zeta_potential"],
    "pdi": ["pdi_std", "pdi"],
    "encapsulation": ["encapsulation_efficiency_percent_std", "encapsulation_efficiency_percent"],
    # toxicity_risk is derived only from explicit numeric cell-viability
    # statements (for example, ``cell viability >85%``).  Qualitative phrases
    # such as "well tolerated" are deliberately not converted to numbers.
    "toxicity": ["toxicity_risk", "toxicity", "cytotoxicity"],
}

FORMULATION_COLUMNS = [
    ("cationic_lipid_mol_ratio", 100.0),
    ("phospholipid_mol_ratio", 100.0),
    ("cholesterol_mol_ratio", 100.0),
    ("peg_lipid_mol_ratio", 100.0),
    ("cationic_lipid_mass_ratio", 100.0),
    ("phospholipid_mass_ratio", 100.0),
    ("cholesterol_mass_ratio", 100.0),
    ("peg_lipid_mass_ratio", 100.0),
    ("cationic_lipid_to_mrna_weight_ratio", 20.0),
    ("np_ratio", 30.0),
    ("aqueous_organic_ratio", 5.0),
    ("total_flow_rate", 20.0),
    ("microfluidic_mixing", 1.0),
    ("dose_mg_kg", 10.0),
]

CONTEXT_VOCABS = {
    "cargo": ["mrna", "sirna", "pdna", "other", "missing"],
    "model": ["hela", "raw264p7", "mouse", "igrov1", "hek293t", "a549", "hbec_ali", "bmdm", "bdmc", "other", "missing"],
    "route": ["in_vitro", "intravenous", "intramuscular", "intratracheal", "other", "missing"],
    "source": ["lnp_ml", "lnp_atlas", "other", "missing"],
}
TARGET_VOCAB = [
    "generic_cell", "macrophage", "liver", "muscle", "lung",
    "lung_epithelium", "spleen", "tumor", "brain", "other", "missing",
]
TARGET_TAXONOMY = [
    "observed", "cell_level", "tissue_or_organ", "immune", "epithelial",
    "liver", "lung", "muscle", "spleen", "tumor", "brain",
]
CONTEXT_DIM = sum(len(values) for values in CONTEXT_VOCABS.values())
TARGET_DIM = len(TARGET_VOCAB)
SPATIAL_DIM = 12
# Every process value is accompanied by an observed/missing bit.  A genuine
# zero (for example 0% helper lipid) is therefore distinguishable from an
# unreported process condition.
FORMULATION_DIM = len(FORMULATION_COLUMNS) * 2


class UnifiedLNPFormulationDataset(Dataset):
    """One row per measured formulation with masked heterogeneous endpoints.

    LNP_ML contributes screen-normalized efficacy and experimental context.
    LNP Atlas contributes genuine four-component identities and physical endpoints.
    Missing endpoints remain masked instead of being silently filled with zero.
    """

    HELPER_LIPID_SMILES = {
        "DOPE": "CCCCCCCC/C=C\\CCCCCCCC(=O)OC[C@H](COP(=O)(O)OCCN)OC(=O)CCCCCCC/C=C\\CCCCCCCC",
        "DSPC": "CCCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)(O)OCC[NH3+])OC(=O)CCCCCCCCCCCCCCCCC",
        "DOTAP": "CCCCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCCCC)[N+](C)(C)C",
        "MDOA": "CCCCCCCCCCCCCCCCCC(=O)OC[C@H](CO)OC(=O)CCCCCCCCCCCCCCCCC",
    }
    DEFAULT_CHOLESTEROL_SMILES = "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C"

    def __init__(
        self,
        merged_datasets_dir: str,
        sample_fraction: float = 1.0,
        random_seed: int = 42,
        fingerprint_dim: int = 2048,
        use_spatial: bool = True,
        include_source_context: bool = False,
        use_mechanistic_descriptors: bool = False,
        target_encoding: str = "onehot",
        group_definition: str = "screen_context",
    ) -> None:
        self.root = Path(merged_datasets_dir)
        self.random_seed = int(random_seed)
        self.fingerprint_dim = int(fingerprint_dim)
        self.use_spatial = bool(use_spatial)
        # Explicit corpus identity is excluded by default.  It is retained as
        # an opt-in audit switch so reviewer-requested shortcut ablations can
        # be reproduced without changing tensor dimensions.
        self.include_source_context = bool(include_source_context)
        self.use_mechanistic_descriptors = bool(use_mechanistic_descriptors)
        if target_encoding not in {"onehot", "taxonomy"}:
            raise ValueError("target_encoding must be 'onehot' or 'taxonomy'")
        self.target_encoding = target_encoding
        if group_definition not in {"screen_context", "screen_id", "library"}:
            raise ValueError(
                "group_definition must be 'screen_context', 'screen_id', or 'library'"
            )
        self.group_definition = group_definition
        self._mol_cache: Dict[str, Optional[Chem.Mol]] = {}
        self._graph_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._fingerprint_cache: Dict[str, torch.Tensor] = {}
        self._spatial_cache: Dict[str, torch.Tensor] = {}
        self._sample_cache: Dict[int, Dict[str, Any]] = {}
        self.task_stats: Dict[str, Dict[str, float]] = {
            task: {"mean": 0.0, "std": 1.0, "count": 0} for task in TASK_COLUMNS
        }
        self.df = self._load_dataframe(sample_fraction)
        self.groups = self.df.apply(self._group_key, axis=1).astype(str).tolist()
        self.smiles_list = self.df["primary_smiles"].astype(str).tolist()

    def _load_dataframe(self, sample_fraction: float) -> pd.DataFrame:
        path = self.root / "02_lnp_formulations" / "lnp_formulations_merged.csv"
        df = pd.read_csv(path, low_memory=False)
        df = self._derive_text_endpoints_and_context(df)
        if "primary_smiles" not in df:
            ionizable = df.get("ionizable_lipid_smiles", pd.Series("", index=df.index)).fillna("").astype(str)
            fallback = df.get("smiles", pd.Series("", index=df.index)).fillna("").astype(str)
            df["primary_smiles"] = ionizable.where(ionizable.str.len() > 0, fallback)
        df["primary_smiles"] = df["primary_smiles"].fillna("").astype(str).str.strip()

        for task, columns in TASK_COLUMNS.items():
            values = pd.Series(np.nan, index=df.index, dtype=float)
            for column in columns:
                if column in df:
                    values = values.fillna(pd.to_numeric(df[column], errors="coerce"))
            df[f"target_{task}"] = values

        target_cols = [f"target_{task}" for task in TASK_COLUMNS]
        df = df[df["primary_smiles"].str.len().gt(0) & df[target_cols].notna().any(axis=1)].reset_index(drop=True)
        if sample_fraction < 1.0:
            sampled = df.sample(
                frac=max(float(sample_fraction), 1.0 / len(df)),
                random_state=self.random_seed,
            )
            # A smoke run must still exercise the safety objective. Preserve all
            # conservative numeric toxicity bounds, then deduplicate sampled rows.
            rare_safety = df[df["target_toxicity"].notna()]
            df = pd.concat([sampled, rare_safety]).loc[lambda frame: ~frame.index.duplicated()].reset_index(drop=True)
        print(
            f"Unified formulation dataset: {len(df)} rows; "
            + ", ".join(f"{task}={int(df[f'target_{task}'].notna().sum())}" for task in TASK_COLUMNS)
        )
        return df

    @classmethod
    def _derive_text_endpoints_and_context(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Recover conservative structured fields from Atlas provenance text.

        LNP Atlas preserves useful process, targeting and safety evidence inside
        ``synthesis_info`` and ``bioactivity_profile``.  We only parse explicit
        values and retain censoring direction for viability; no qualitative
        safety statement is promoted to a numeric label.
        """
        df = df.copy()
        for column in [
            "total_flow_rate", "aqueous_organic_ratio", "microfluidic_mixing",
            "dose_mg_kg", "toxicity_risk", "toxicity_censor",
        ]:
            if column not in df:
                df[column] = np.nan

        for idx, row in df.iterrows():
            synthesis = cls._clean(row.get("synthesis_info"))
            bioactivity = cls._clean(row.get("bioactivity_profile"))

            if pd.isna(df.at[idx, "total_flow_rate"]):
                match = re.search(r"total_flow_rate_ml_min:\s*([0-9]+(?:\.[0-9]+)?)", synthesis, re.I)
                if match:
                    df.at[idx, "total_flow_rate"] = float(match.group(1))

            if pd.isna(df.at[idx, "aqueous_organic_ratio"]):
                match = re.search(
                    r"flow_rate_ratio:\s*([0-9]+(?:\.[0-9]+)?)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*\(([^)]*)\)",
                    synthesis,
                    re.I,
                )
                if match:
                    first, second = float(match.group(1)), float(match.group(2))
                    order = match.group(3).lower()
                    aqueous, organic = (second, first) if order.startswith("organic") else (first, second)
                    if organic > 0:
                        df.at[idx, "aqueous_organic_ratio"] = aqueous / organic

            if pd.isna(df.at[idx, "microfluidic_mixing"]):
                mixing_text = f"{cls._clean(row.get('mix_type'))} {synthesis}".lower()
                if "microfluidic" in mixing_text:
                    df.at[idx, "microfluidic_mixing"] = 1.0
                elif any(term in mixing_text for term in ["pipette", "hand", "bulk mix"]):
                    df.at[idx, "microfluidic_mixing"] = 0.0

            if pd.isna(df.at[idx, "dose_mg_kg"]):
                match = re.search(r"dose:\s*([0-9]+(?:\.[0-9]+)?)\s*mg/kg", bioactivity, re.I)
                if match:
                    df.at[idx, "dose_mg_kg"] = float(match.group(1))

            # Cell viability is converted to toxicity risk = 1 - viability.
            # ``viability >85%`` therefore means risk <0.15 (upper censored).
            if pd.isna(df.at[idx, "toxicity_risk"]):
                match = re.search(
                    r"(?:cell\s+viability|viability)\s*(?:\([^)]*\))?\s*([><~≈]?)\s*(\d+(?:\.\d+)?)\s*%",
                    bioactivity,
                    re.I,
                )
                if match:
                    operator, viability = match.group(1), float(match.group(2))
                    df.at[idx, "toxicity_risk"] = min(max(1.0 - viability / 100.0, 0.0), 1.0)
                    # -1: upper bound, +1: lower bound, 0: approximately exact.
                    df.at[idx, "toxicity_censor"] = -1.0 if operator == ">" else (1.0 if operator == "<" else 0.0)

            # Atlas context is often embedded in a semicolon-delimited profile.
            if not cls._clean(row.get("model_type")):
                cell_line = cls._profile_value(bioactivity, "cell_line")
                animal = cls._profile_value(bioactivity, "animal_model")
                model_text = f"{cell_line} {animal}".lower()
                model = next(
                    (name for needle, name in [
                        ("raw264", "RAW264p7"), ("hela", "HeLa"), ("igrov1", "IGROV1"),
                        ("hek293", "HEK293T"), ("a549", "A549"), ("hbec", "HBEC_ALI"),
                        ("bmdm", "BMDM"), ("bmdc", "BDMC"), ("mouse", "Mouse"), ("mice", "Mouse"),
                    ] if needle in model_text),
                    "",
                )
                if model:
                    df.at[idx, "model_type"] = model

            if not cls._clean(row.get("route_of_administration")):
                route_text = cls._profile_value(bioactivity, "administration_route").lower()
                route = cls._canonical_route(route_text)
                if route:
                    df.at[idx, "route_of_administration"] = route

            if not cls._clean(row.get("delivery_target")):
                targeting_text = " ".join([
                    cls._profile_value(bioactivity, "biodistribution_result"),
                    cls._profile_value(bioactivity, "gene_expression_result"),
                ]).lower()
                target = cls._infer_target(targeting_text)
                if target:
                    df.at[idx, "delivery_target"] = target
        return df

    @staticmethod
    def _profile_value(text: str, field: str) -> str:
        match = re.search(rf"(?:^|;\s*){re.escape(field)}:\s*([^;]*)", text, re.I)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _canonical_route(text: str) -> str:
        if not text:
            return ""
        if any(term in text for term in ["intravenous", "tail vein", " i.v", "iv "]):
            return "intravenous"
        if any(term in text for term in ["intramuscular", " i.m", "im "]):
            return "intramuscular"
        if any(term in text for term in ["intratracheal", "nebul", "inhal"]):
            return "intratracheal"
        if "cell" in text or "in vitro" in text:
            return "in_vitro"
        return "other"

    @staticmethod
    def _infer_target(text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        for needle, target in [
            ("lung epithe", "lung_epithelium"), ("pulmonary", "lung"), ("lung", "lung"),
            ("hepatic", "liver"), ("liver", "liver"), ("spleen", "spleen"),
            ("muscle", "muscle"), ("brain", "brain"), ("tumor", "tumor"),
            ("macrophage", "macrophage"),
        ]:
            if needle in text:
                return target
        return "other"

    @staticmethod
    def _clean(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    def _group_key(self, row: pd.Series) -> str:
        if self.group_definition == "library":
            screen = self._clean(row.get("library_id"))
            prefix = "library"
        elif self.group_definition == "screen_id":
            screen = self._clean(row.get("screen_id"))
            prefix = "screen_id"
        else:
            screen = self._clean(row.get("split_name_for_normalization"))
            prefix = "screen"
        if screen:
            return f"{prefix}:{screen}"
        doi = self._clean(row.get("paper_doi")) or self._clean(row.get("paper_title"))
        if doi:
            return f"paper:{doi}"
        return f"source:{self._clean(row.get('dataset_source'))}"

    def fit_target_normalization(self, train_indices: Sequence[int]) -> None:
        """Fit endpoint scaling on training rows only."""
        train_df = self.df.iloc[list(train_indices)]
        for task in TASK_COLUMNS:
            values = train_df[f"target_{task}"].dropna().astype(float)
            mean = float(values.mean()) if len(values) else 0.0
            std = float(values.std()) if len(values) > 1 else 1.0
            if not math.isfinite(std) or std < 1e-8:
                std = 1.0
            self.task_stats[task] = {"mean": mean, "std": std, "count": int(len(values))}
        self._sample_cache.clear()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx not in self._sample_cache:
            self._sample_cache[idx] = self._build_sample(idx)
        sample = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in self._sample_cache[idx].items()}
        return sample

    def _build_sample(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        sample: Dict[str, Any] = {
            "idx": idx,
            "smiles": self.smiles_list[idx],
            "group_id": self._stable_int(self.groups[idx]),
            "formulation_features": self._formulation_features(row),
            "context_features": self._context_features(row),
        }
        ratios = [self._number(row.get(column), 0.0) for column, _ in FORMULATION_COLUMNS[:4]]
        sample["molar_ratios"] = torch.tensor(ratios, dtype=torch.float32)

        target_features, target_class, target_mask = self._target_features(row)
        sample["target_features"] = target_features
        sample["target_class"] = torch.tensor(target_class, dtype=torch.long)
        sample["target_class_mask"] = torch.tensor(target_mask, dtype=torch.bool)

        structure_mask: List[float] = []
        for component in ["ionizable", "helper", "cholesterol", "peg"]:
            smiles, observed = self._component_info(row, component)
            structure_mask.append(float(observed))
            sample[f"{component}_smiles"] = smiles
            graph = self._graph(smiles)
            for key, value in graph.items():
                sample[f"{component}_{key}"] = value
            sample[f"{component}_fingerprint"] = self._fingerprint(smiles)
            sample[f"{component}_spatial_features"] = self._spatial(smiles)
        sample["component_structure_mask"] = torch.tensor(structure_mask, dtype=torch.float32)
        sample["component_active_mask"] = torch.tensor(
            [float(ratio > 0) for ratio in ratios], dtype=torch.float32
        )

        for task in TASK_COLUMNS:
            raw = row[f"target_{task}"]
            mask = not pd.isna(raw)
            stats = self.task_stats[task]
            value = (float(raw) - stats["mean"]) / stats["std"] if mask else 0.0
            sample[task] = torch.tensor(value, dtype=torch.float32)
            sample[f"{task}_mask"] = torch.tensor(mask, dtype=torch.bool)
        sample["toxicity_censor"] = torch.tensor(
            self._number(row.get("toxicity_censor"), 0.0), dtype=torch.float32
        )
        sample["target"] = sample["efficiency"]
        return sample

    def _component_smiles(self, row: pd.Series, component: str) -> str:
        return self._component_info(row, component)[0]

    def _component_info(self, row: pd.Series, component: str) -> Tuple[str, bool]:
        """Return a validated structure and whether it was truly observed.

        Missing sterol/PEG identities are no longer silently presented to the
        model as cholesterol/DMG-PEG.  The unknown component is represented by
        a zero graph plus an explicit missing-structure embedding.
        """
        columns = {
            "ionizable": ["ionizable_lipid_smiles", "primary_smiles", "smiles"],
            "helper": ["helper_lipid_smiles", "helper_lipid", "helper_lipid_id"],
            "cholesterol": ["sterol_lipid_smiles", "sterol_lipid", "cholesterol"],
            "peg": ["peg_lipid_smiles", "peg_lipid"],
        }[component]
        value = next((self._clean(row.get(column)) for column in columns if self._clean(row.get(column))), "")
        if component == "helper" and value.upper() in self.HELPER_LIPID_SMILES:
            return self.HELPER_LIPID_SMILES[value.upper()], True
        if value and self._mol(value) is not None:
            return value, True
        if component == "cholesterol" and value.lower() in {"cholesterol", "chol"}:
            return self.DEFAULT_CHOLESTEROL_SMILES, True
        if component == "cholesterol":
            source = self._clean(row.get("dataset_source")).lower()
            ratio = pd.to_numeric(row.get("cholesterol_mol_ratio"), errors="coerce")
            if source == "lnp_ml" and not pd.isna(ratio) and float(ratio) > 0:
                # The published LNP_ML schema fixes this role to cholesterol;
                # only its amount varies.  Preserve that source-defined identity
                # without extending the fallback to corpora that may use other
                # sterols.
                return self.DEFAULT_CHOLESTEROL_SMILES, True
        # PEG products are polydisperse; do not invent a discrete graph from a
        # nominal product label such as C14-PEG2000 or PEG3000.
        return "", False

    def _formulation_features(self, row: pd.Series) -> torch.Tensor:
        values: List[float] = []
        observed: List[float] = []
        for column, scale in FORMULATION_COLUMNS:
            raw = pd.to_numeric(row.get(column), errors="coerce")
            is_observed = not pd.isna(raw) and math.isfinite(float(raw))
            values.append(float(raw) / scale if is_observed else 0.0)
            observed.append(float(is_observed))
        return torch.tensor(values + observed, dtype=torch.float32)

    def _context_features(self, row: pd.Series) -> torch.Tensor:
        raw = {
            "cargo": self._clean(row.get("cargo_type")) or self._clean(row.get("target_type")),
            "model": self._clean(row.get("model_type")),
            "route": self._clean(row.get("route_of_administration")),
            "source": self._clean(row.get("dataset_source")),
        }
        output: List[float] = []
        for field, vocab in CONTEXT_VOCABS.items():
            if field == "source" and not self.include_source_context:
                output.extend([0.0] * len(vocab))
                continue
            value = raw[field].lower().replace(" ", "_").replace("-", "_")
            if not value:
                value = "missing"
            if value not in vocab:
                if field == "target" and value.startswith("lung"):
                    value = "lung"
                else:
                    value = "other"
            output.extend(float(value == category) for category in vocab)
        return torch.tensor(output, dtype=torch.float32)

    def _target_features(self, row: pd.Series) -> Tuple[torch.Tensor, int, bool]:
        value = self._clean(row.get("delivery_target")).lower().replace(" ", "_").replace("-", "_")
        observed = bool(value)
        if not value:
            value = "missing"
        elif value.startswith("lung") and value not in TARGET_VOCAB:
            value = "lung"
        elif value not in TARGET_VOCAB:
            value = "other"
        index = TARGET_VOCAB.index(value)
        if self.target_encoding == "onehot":
            features = [float(i == index) for i in range(TARGET_DIM)]
        else:
            taxonomy = {
                "generic_cell": [0, 1], "macrophage": [0, 1, 3],
                "liver": [0, 2, 5], "muscle": [0, 2, 7],
                "lung": [0, 2, 6], "lung_epithelium": [0, 1, 4, 6],
                "spleen": [0, 2, 3, 8], "tumor": [0, 2, 9],
                "brain": [0, 2, 10], "other": [0], "missing": [],
            }
            active = set(taxonomy[value])
            features = [float(i in active) for i in range(TARGET_DIM)]
        return torch.tensor(features, dtype=torch.float32), index, observed

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        number = pd.to_numeric(value, errors="coerce")
        return float(number) if not pd.isna(number) and math.isfinite(float(number)) else default

    @staticmethod
    def _stable_int(value: str) -> int:
        return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:15], 16)

    def _mol(self, smiles: str) -> Optional[Chem.Mol]:
        if not smiles:
            return None
        if smiles not in self._mol_cache:
            # Several source tables mix lipid product names (for example
            # DMG-PEG2k) into columns labelled as SMILES.  Probe quietly and
            # let _component_smiles apply the chemically explicit fallback.
            with rdBase.BlockLogs():
                self._mol_cache[smiles] = Chem.MolFromSmiles(smiles)
        return self._mol_cache[smiles]

    def _graph(self, smiles: str) -> Dict[str, torch.Tensor]:
        if smiles in self._graph_cache:
            return {key: value.clone() for key, value in self._graph_cache[smiles].items()}
        mol = self._mol(smiles)
        if mol is None:
            result = {
                "atom_features": torch.zeros(1, 39),
                "bond_features": torch.zeros(0, 4),
                "edge_index": torch.zeros(2, 0, dtype=torch.long),
            }
        else:
            atoms = [self._atom_features(atom) for atom in mol.GetAtoms()]
            edges: List[List[int]] = []
            bonds: List[List[float]] = []
            for bond in mol.GetBonds():
                begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                feature = self._bond_features(bond)
                edges.extend([[begin, end], [end, begin]])
                bonds.extend([feature, feature])
            result = {
                "atom_features": torch.tensor(atoms, dtype=torch.float32),
                "bond_features": torch.tensor(bonds, dtype=torch.float32).reshape(-1, 4),
                "edge_index": torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros(2, 0, dtype=torch.long),
            }
        self._graph_cache[smiles] = result
        return {key: value.clone() for key, value in result.items()}

    @staticmethod
    def _atom_features(atom: Chem.Atom) -> List[float]:
        atomic_numbers = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
        feature = [float(atom.GetAtomicNum() == number) for number in atomic_numbers]
        feature += [
            atom.GetAtomicNum() / 100.0,
            atom.GetDegree() / 4.0,
            atom.GetFormalCharge() / 4.0,
            atom.GetTotalNumHs() / 4.0,
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
        ]
        feature += [float(str(atom.GetHybridization()) == name) for name in ["SP", "SP2", "SP3", "SP3D", "SP3D2"]]
        feature += [0.0] * (39 - len(feature))
        return feature[:39]

    @staticmethod
    def _bond_features(bond: Chem.Bond) -> List[float]:
        bond_type = bond.GetBondType()
        return [
            float(bond_type == Chem.BondType.SINGLE),
            float(bond_type == Chem.BondType.DOUBLE),
            float(bond_type == Chem.BondType.TRIPLE),
            float(bond_type == Chem.BondType.AROMATIC),
        ]

    def _fingerprint(self, smiles: str) -> torch.Tensor:
        if smiles in self._fingerprint_cache:
            return self._fingerprint_cache[smiles].clone()
        mol = self._mol(smiles)
        if mol is None:
            tensor = torch.zeros(self.fingerprint_dim)
        else:
            # Count fingerprints retain repeated lipid-tail/headgroup motifs.
            # log1p bounds large repeat counts without turning them binary.
            fp = GetMorganGenerator(
                radius=2,
                fpSize=self.fingerprint_dim,
                includeChirality=True,
            ).GetCountFingerprintAsNumPy(mol)
            tensor = torch.log1p(torch.tensor(fp, dtype=torch.float32))
        self._fingerprint_cache[smiles] = tensor
        return tensor.clone()

    def _spatial(self, smiles: str) -> torch.Tensor:
        if smiles in self._spatial_cache:
            return self._spatial_cache[smiles].clone()
        mol = self._mol(smiles)
        if mol is None:
            tensor = torch.zeros(SPATIAL_DIM)
        else:
            base = [
                Descriptors.MolWt(mol) / 1000.0,
                Crippen.MolLogP(mol) / 10.0,
                rdMolDescriptors.CalcTPSA(mol) / 250.0,
                rdMolDescriptors.CalcFractionCSP3(mol),
                Lipinski.NumRotatableBonds(mol) / 30.0,
                Chem.GetFormalCharge(mol) / 5.0,
            ]
            shape = [0.0] * 6
            if self.use_mechanistic_descriptors:
                ester = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
                tertiary_amine = Chem.MolFromSmarts("[NX3;H0;!$(N-C=O)]")
                carbon_atoms = sum(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
                shape = [
                    Lipinski.NumHDonors(mol) / 10.0,
                    Lipinski.NumHAcceptors(mol) / 20.0,
                    Lipinski.RingCount(mol) / 10.0,
                    len(mol.GetSubstructMatches(ester)) / 10.0 if ester is not None else 0.0,
                    len(mol.GetSubstructMatches(tertiary_amine)) / 10.0 if tertiary_amine is not None else 0.0,
                    carbon_atoms / max(mol.GetNumAtoms(), 1),
                ]
            elif self.use_spatial and mol.GetNumAtoms() > 1:
                conformer_mol = Chem.AddHs(Chem.Mol(mol))
                params = AllChem.ETKDGv3()
                params.randomSeed = self.random_seed
                params.useRandomCoords = False
                try:
                    if AllChem.EmbedMolecule(conformer_mol, params) == 0:
                        shape = [
                            rdMolDescriptors.CalcRadiusOfGyration(conformer_mol) / 10.0,
                            rdMolDescriptors.CalcAsphericity(conformer_mol),
                            rdMolDescriptors.CalcEccentricity(conformer_mol),
                            rdMolDescriptors.CalcInertialShapeFactor(conformer_mol),
                            rdMolDescriptors.CalcNPR1(conformer_mol),
                            rdMolDescriptors.CalcNPR2(conformer_mol),
                        ]
                except Exception:
                    pass
            tensor = torch.tensor(base + shape, dtype=torch.float32)
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        self._spatial_cache[smiles] = tensor
        return tensor.clone()


def unified_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "smiles": [item["smiles"] for item in batch],
        "idx": torch.tensor([item["idx"] for item in batch], dtype=torch.long),
        "group_id": torch.tensor([item["group_id"] for item in batch], dtype=torch.long),
    }
    for component in ["ionizable", "helper", "cholesterol", "peg"]:
        result[f"{component}_smiles"] = [item[f"{component}_smiles"] for item in batch]
    fixed_fields = [
        "target", "molar_ratios", "formulation_features", "context_features",
        "target_features", "target_class", "target_class_mask",
        "component_structure_mask", "component_active_mask", "toxicity_censor",
    ]
    fixed_fields += [task for task in TASK_COLUMNS] + [f"{task}_mask" for task in TASK_COLUMNS]
    fixed_fields += [f"{component}_spatial_features" for component in ["ionizable", "helper", "cholesterol", "peg"]]
    fixed_fields += [f"{component}_fingerprint" for component in ["ionizable", "helper", "cholesterol", "peg"]]
    for field in fixed_fields:
        result[field] = torch.stack([item[field] for item in batch])

    for component in ["ionizable", "helper", "cholesterol", "peg"]:
        atoms = [item[f"{component}_atom_features"] for item in batch]
        bonds = [item[f"{component}_bond_features"] for item in batch]
        edges = [item[f"{component}_edge_index"] for item in batch]
        counts = [atom.size(0) for atom in atoms]
        offsets = np.cumsum([0] + counts[:-1])
        result[f"{component}_atom_features"] = torch.cat(atoms, dim=0)
        result[f"{component}_bond_features"] = torch.cat(bonds, dim=0)
        shifted = [edge + int(offset) for edge, offset in zip(edges, offsets)]
        result[f"{component}_edge_index"] = torch.cat(shifted, dim=1)
        result[f"{component}_batch"] = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(counts))
    return result


def endpoint_balanced_group_folds(
    dataset: UnifiedLNPFormulationDataset,
    n_splits: int = 5,
    random_seed: int = 42,
) -> List[np.ndarray]:
    """Create group-disjoint folds balanced for efficacy support.

    Only endpoint *availability* and group sizes are used; target values remain
    hidden.  Each efficacy-bearing group is assigned exactly once, with fold
    capacities differing by at most one group.  Endpoint-free publication
    groups are then assigned to balance total corpus rows.
    """
    groups = np.asarray(dataset.groups, dtype=object)
    efficacy = dataset.df["target_efficiency"].notna().to_numpy()
    rng = np.random.default_rng(random_seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    total_count = {
        group: int(np.sum(groups == group)) for group in unique_groups
    }
    efficacy_count = {
        group: int(np.sum((groups == group) & efficacy))
        for group in unique_groups
    }
    efficacy_groups = [
        group for group in unique_groups if efficacy_count[group] > 0
    ]
    efficacy_groups.sort(
        key=lambda group: (efficacy_count[group], total_count[group]),
        reverse=True,
    )
    capacities = np.full(n_splits, len(efficacy_groups) // n_splits, dtype=int)
    capacities[: len(efficacy_groups) % n_splits] += 1
    rng.shuffle(capacities)
    fold_groups: List[List[str]] = [[] for _ in range(n_splits)]
    fold_efficiency_rows = np.zeros(n_splits, dtype=int)
    fold_total_rows = np.zeros(n_splits, dtype=int)
    for group in efficacy_groups:
        candidates = np.where(
            np.asarray([len(values) for values in fold_groups]) < capacities
        )[0]
        chosen = min(
            candidates,
            key=lambda fold: (
                fold_efficiency_rows[fold], fold_total_rows[fold], fold
            ),
        )
        fold_groups[chosen].append(group)
        fold_efficiency_rows[chosen] += efficacy_count[group]
        fold_total_rows[chosen] += total_count[group]
    non_efficacy_groups = [
        group for group in unique_groups if efficacy_count[group] == 0
    ]
    non_efficacy_groups.sort(
        key=lambda group: total_count[group], reverse=True
    )
    for group in non_efficacy_groups:
        chosen = min(
            range(n_splits),
            key=lambda fold: (
                fold_total_rows[fold], len(fold_groups[fold]), fold
            ),
        )
        fold_groups[chosen].append(group)
        fold_total_rows[chosen] += total_count[group]
    return [np.where(np.isin(groups, fold))[0] for fold in fold_groups]


def create_unified_dataloaders(
    merged_datasets_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    sample_fraction: float = 1.0,
    random_seed: int = 42,
    split_strategy: str = "group",
    use_spatial: bool = True,
    pin_memory: bool = False,
    balanced_sampling: bool = True,
    ensure_rare_task_holdout: bool = True,
    include_source_context: bool = False,
    use_mechanistic_descriptors: bool = False,
    target_encoding: str = "onehot",
    group_definition: str = "screen_context",
    group_folds: int = 5,
    group_fold_index: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, UnifiedLNPFormulationDataset]:
    dataset = UnifiedLNPFormulationDataset(
        merged_datasets_dir,
        sample_fraction=sample_fraction,
        random_seed=random_seed,
        use_spatial=use_spatial,
        include_source_context=include_source_context,
        use_mechanistic_descriptors=use_mechanistic_descriptors,
        target_encoding=target_encoding,
        group_definition=group_definition,
    )
    indices = np.arange(len(dataset))
    if (
        split_strategy == "balanced_group_kfold"
        and len(set(dataset.groups)) >= max(group_folds, 3)
    ):
        folds = endpoint_balanced_group_folds(
            dataset, n_splits=group_folds, random_seed=random_seed
        )
        fold_index = int(group_fold_index) % group_folds
        test_idx = folds[fold_index]
        val_idx = folds[(fold_index + 1) % group_folds]
        train_idx = np.concatenate([
            fold
            for number, fold in enumerate(folds)
            if number not in {fold_index, (fold_index + 1) % group_folds}
        ])
    elif (
        split_strategy == "group_kfold"
        and len(set(dataset.groups)) >= max(group_folds, 3)
    ):
        splitter = GroupKFold(
            n_splits=group_folds, shuffle=True, random_state=random_seed
        )
        folds = [
            test_rows
            for _, test_rows in splitter.split(indices, groups=dataset.groups)
        ]
        fold_index = int(group_fold_index) % group_folds
        test_idx = folds[fold_index]
        val_idx = folds[(fold_index + 1) % group_folds]
        train_idx = np.concatenate([
            fold
            for number, fold in enumerate(folds)
            if number not in {fold_index, (fold_index + 1) % group_folds}
        ])
    elif split_strategy == "group" and len(set(dataset.groups)) >= 3:
        outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_seed)
        train_idx, hold_idx = next(outer.split(indices, groups=dataset.groups))
        hold_groups = np.asarray(dataset.groups, dtype=object)[hold_idx]
        inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_seed + 1)
        val_rel, test_rel = next(inner.split(hold_idx, groups=hold_groups))
        val_idx, test_idx = hold_idx[val_rel], hold_idx[test_rel]
    else:
        rng = np.random.default_rng(random_seed)
        shuffled = rng.permutation(indices)
        n_train, n_val = int(0.7 * len(dataset)), int(0.15 * len(dataset))
        train_idx, val_idx, test_idx = shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]

    # Sparse endpoints must not disappear from held-out evaluation merely due
    # to group splitting. Move one complete publication/screen group, never
    # individual rows, so leakage protection remains intact.
    toxicity_col = "target_toxicity"
    if ensure_rare_task_holdout and toxicity_col in dataset.df:
        toxic = dataset.df[toxicity_col].notna().to_numpy()
        groups = np.asarray(dataset.groups, dtype=object)
        toxic_groups = np.unique(groups[toxic])
        if len(toxic_groups) >= 2 and not toxic[test_idx].any():
            candidates = [g for g in toxic_groups if np.any(groups[train_idx] == g)]
            if candidates:
                move_group = min(candidates, key=lambda g: int(np.sum(toxic & (groups == g))))
                move = train_idx[groups[train_idx] == move_group]
                train_idx = train_idx[groups[train_idx] != move_group]
                test_idx = np.concatenate([test_idx, move])

    dataset.fit_target_normalization(train_idx)
    print(
        f"Split ({split_strategy}): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}; "
        f"group overlap train/val/test="
        f"{len(set(np.asarray(dataset.groups)[train_idx]) & set(np.asarray(dataset.groups)[val_idx]))}/"
        f"{len(set(np.asarray(dataset.groups)[train_idx]) & set(np.asarray(dataset.groups)[test_idx]))}"
    )
    loaders = []
    for split_number, (split_indices, shuffle) in enumerate([(train_idx, True), (val_idx, False), (test_idx, False)]):
        sampler = None
        if split_number == 0 and balanced_sampling:
            train_rows = dataset.df.iloc[split_indices]
            source = train_rows.get("source_dataset", pd.Series("unknown", index=train_rows.index)).fillna("unknown")
            counts = source.value_counts().to_dict()
            weights = source.map(
                lambda value: 1.0 / np.sqrt(max(counts.get(value, 1), 1))
            ).to_numpy(dtype=np.float64, copy=True)
            # Increase exposure of rows carrying the sparse physical/safety labels.
            sparse = np.zeros(len(train_rows), dtype=bool)
            for task in ["particle_size", "zeta_potential", "pdi", "encapsulation", "toxicity"]:
                column = f"target_{task}"
                if column in train_rows:
                    sparse |= train_rows[column].notna().to_numpy()
            weights *= np.where(sparse, 2.0, 1.0)
            sampler = WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(split_indices), replacement=True,
                generator=torch.Generator().manual_seed(random_seed),
            )
            shuffle = False
        loaders.append(
            DataLoader(
                Subset(dataset, split_indices.tolist()),
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=unified_collate_fn,
            )
        )
    return loaders[0], loaders[1], loaders[2], dataset
