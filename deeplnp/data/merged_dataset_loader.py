"""
tmux - 专门用于加载 merged_datasets 中的数据
"""

import os
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


class MergedMultiModalDataset(Dataset):
    HELPER_LIPID_SMILES = {
        "DOPE": "CCCCCCCC/C=C\\CCCCCCCC(=O)OC[C@H](COP(=O)(O)OCCN)OC(=O)CCCCCCC/C=C\\CCCCCCCC",
        "DSPC": "CCCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)(O)OCC[NH3+])OC(=O)CCCCCCCCCCCCCCCCC",
        "DOTAP": "CCCCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCCCC)[N+](C)(C)C",
        "MDOA": "CCCCCCCCCCCCCCCCCC(=O)OC[C@H](CO)OC(=O)CCCCCCCCCCCCCCCCC",
        "NONE": "",
        "NAN": "",
    }
    DEFAULT_CHOLESTEROL_SMILES = "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C"

    def __init__(
        self,
        merged_datasets_dir: str,
        modalities: List[str] = ["molecular", "formulation", "images"],
        max_atoms: int = 256,
        fingerprint_dim: int = 2048,
        image_size: int = 192,
        sample_fraction: float = 1.0,
        random_seed: int = 42,
    ):
        self.merged_datasets_dir = Path(merged_datasets_dir)
        self.modalities = modalities
        self.max_atoms = max_atoms
        self.fingerprint_dim = fingerprint_dim
        self.image_size = image_size
        self.sample_fraction = sample_fraction
        
        np.random.seed(random_seed)
        self.data = {}
        self.smiles_list = []
        self.targets = []
        self.df = pd.DataFrame()
        self.primary_smiles_col = "smiles"
        self.descriptor_cols = []
        self.descriptor_dim = 0
        self.molecular_descriptor_map = {}
        self._mol_cache = {}
        self._graph_cache = {}
        self._fingerprint_cache = {}
        self.target_mean = 0.0
        self.target_std = 1.0
        
        self._load_all_modalities()
        
        # 预缓存所有分子特征到内存，避免 __getitem__ 中重复计算
        self._precache_all_features()
        
    def _load_all_modalities(self):
        print(f"\n加载合并数据集：{self.merged_datasets_dir}")
        print(f"模态：{self.modalities}")
        
        if "molecular" in self.modalities:
            self._load_molecular_data()
        if "formulation" in self.modalities:
            self._load_formulation_data()
        if "images" in self.modalities:
            self._load_image_data()
        
        if self.sample_fraction < 1.0:
            self._sample_data()
        
        print(f"数据集加载完成！总样本数：{len(self.smiles_list)}\n")
        
    def _load_molecular_data(self):
        print("\n加载分子描述符 + 生物活性数据...")
        data_file = self.merged_datasets_dir / "01_molecular_descriptors_bioactivity" / "molecular_descriptors_bioactivity_merged.csv"
        
        if not data_file.exists():
            print(f"  文件不存在：{data_file}")
            return
        
        df = pd.read_csv(data_file)
        df = df.dropna(subset=["smiles"])
        df = df[df["smiles"].astype(str).str.len() > 0]
        
        if "expt_hela" in df.columns:
            target_data = df["expt_hela"].dropna()
            if len(target_data) > 0:
                mean, std = target_data.mean(), target_data.std()
                if std > 0:
                    df.loc[:, "expt_hela"] = (df["expt_hela"] - mean) / std
        
        desc_cols = [col for col in df.columns if col.startswith("desc_")]
        if desc_cols:
            self.descriptor_cols = desc_cols
            self.descriptor_dim = len(desc_cols)
            desc_values = df[desc_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32).values
            self.molecular_descriptor_map = {
                str(smiles): values
                for smiles, values in zip(df["smiles"].tolist(), desc_values)
            }
            print(f"  分子描述符：{len(desc_cols)} 个特征")

        if "formulation" not in self.modalities:
            self.df = df.copy().reset_index(drop=True)
            self.df["target"] = df["expt_hela"].fillna(0).values if "expt_hela" in df.columns else 0.0
            self.primary_smiles_col = "smiles"
            self.smiles_list = self.df["smiles"].tolist()
            self.targets = self.df["target"].astype(float).tolist()
            if desc_cols:
                self.data["molecular_descriptors"] = self.df[desc_cols].fillna(0).values
        
        print(f"  加载 {len(df)} 个化合物")
        
    def _load_formulation_data(self):
        print("\n加载 LNP 配方数据（完整四组分）...")
        data_file = self.merged_datasets_dir / "02_lnp_formulations" / "lnp_formulations_merged.csv"
        
        if not data_file.exists():
            print(f"  文件不存在：{data_file}")
            return
        
        df = pd.read_csv(data_file, low_memory=False)
        
        # 查找 SMILES 列（按优先级排序）
        smiles_col = None
        for col_name in ["smiles", "ionizable_lipid_smiles", "ionizable_lipid", "lipid_smiles", "full_smiles", "amine_smiles"]:
            if col_name in df.columns:
                # 检查该列是否有非空值
                non_null = df[col_name].notna().sum()
                if non_null > 0:
                    smiles_col = col_name
                    print(f"  使用 SMILES 列：{col_name} ({non_null} 非空值)")
                    break
        
        if smiles_col is None:
            print(f"  未找到有效的 SMILES 列，可用列：{df.columns[:10].tolist()}...")
            return
        
        df = df.dropna(subset=[smiles_col])
        df = df[df[smiles_col].astype(str).str.len() > 0]
        
        target_col = self._build_target_column(df)
        if target_col is None:
            print("  未找到可用性能标签，跳过 LNP 配方数据")
            return

        df = df.dropna(subset=[target_col]).reset_index(drop=True)
        df, self.target_mean, self.target_std = self._normalize_targets(df, target_col)
        df["target"] = pd.to_numeric(df[target_col], errors="coerce")
        df = df.dropna(subset=["target"]).reset_index(drop=True)
        df["target_source"] = target_col

        # 存储完整的 LNP 配方数据。完整 LNP 预测以配方表为主表，
        # 其它模态只能按 SMILES 映射增强，不能按行号硬拼。
        self.df = df
        self.primary_smiles_col = smiles_col
        self.smiles_list = df[smiles_col].tolist()
        self.targets = df["target"].astype(float).tolist()
        
        # 选择数值类型的配方特征（固定 7 个特征）
        # 4 个摩尔比特征 + 3 个物理性质特征
        formulation_cols = [
            "cationic_lipid_mol_ratio",      # 可电离脂质摩尔比
            "phospholipid_mol_ratio",        # 辅助脂质摩尔比
            "cholesterol_mol_ratio",         # 胆固醇摩尔比
            "peg_lipid_mol_ratio",           # PEG 脂质摩尔比
        ]
        
        # 过滤掉不存在的列
        formulation_cols = [col for col in formulation_cols if col in df.columns]
        
        # 如果没有找到足够的特征，添加其他摩尔比相关特征
        if len(formulation_cols) < 7:
            additional_cols = []
            for col in df.columns:
                if col in formulation_cols:
                    continue
                if any(k in col.lower() for k in ["mass_ratio", "weight_ratio"]):
                    if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        additional_cols.append(col)
            # 只取前 3 个额外的列来凑够 7 个
            additional_cols = additional_cols[:3]
            formulation_cols.extend(additional_cols)
        
        # 如果还是不够 7 个，用 0 填充
        while len(formulation_cols) < 7:
            # 创建一个全 0 的虚拟特征
            dummy_col = f"dummy_feature_{len(formulation_cols)}"
            df[dummy_col] = 0.0
            formulation_cols.append(dummy_col)
        
        physical_cols = [col for col in df.columns if any(k in col.lower() for k in ["size", "zeta", "pdi"]) and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        if formulation_cols:
            # 转换为数值类型，处理可能的字符串数据
            formulation_df = df[formulation_cols].fillna(0)
            # 将非数值列转换为数值（使用 pd.to_numeric）
            for col in formulation_cols:
                if formulation_df[col].dtype == 'object':
                    formulation_df[col] = pd.to_numeric(formulation_df[col], errors='coerce').fillna(0)
            
            # 标准化配方特征
            for col in formulation_cols:
                col_mean = formulation_df[col].mean()
                col_std = formulation_df[col].std()
                if col_std > 1e-8:
                    formulation_df[col] = (formulation_df[col] - col_mean) / col_std
            
            self.data["formulation_features"] = formulation_df.values.astype(np.float32)
            print(f"  配方特征：{len(formulation_cols)} 个特征")
        if physical_cols:
            # 转换为数值类型，处理可能的字符串数据
            physical_df = df[physical_cols].fillna(0)
            for col in physical_cols:
                if physical_df[col].dtype == 'object':
                    physical_df[col] = pd.to_numeric(physical_df[col], errors='coerce').fillna(0)
            self.data["physical_properties"] = physical_df.values.astype(np.float32)
            print(f"  物理性质：{len(physical_cols)} 个特征")
        
        print(f"  加载 {len(df)} 个 LNP 配方（完整四组分）")

    def _build_target_column(self, df: pd.DataFrame) -> str:
        """选择完整 LNP 性能预测目标列，优先使用标准化递送效率。"""
        candidates = [
            "quantified_delivery",
            "transfection_efficiency",
            "encapsulation_efficiency_percent_std",
            "unnormalized_delivery",
            "expt_hela",
        ]
        for col in candidates:
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().sum() == 0:
                continue
            if col == "unnormalized_delivery":
                values = np.log1p(values.clip(lower=0))
                std = values.std()
                if std and std > 0:
                    values = (values - values.mean()) / std
                df[col] = values
            return col
        return None
        
    def _normalize_targets(self, df: pd.DataFrame, target_col: str):
        """Normalize target values using robust z-score normalization."""
        values = pd.to_numeric(df[target_col], errors="coerce")
        valid = values.dropna()
        if len(valid) < 2:
            return df, 0.0, 1.0
        
        mean = valid.mean()
        std = valid.std()
        
        if std < 1e-8:
            df[target_col] = 0.0
            return df, mean, 1.0
        
        df[target_col] = (values - mean) / std
        return df, mean, std
        
    def _load_image_data(self):
        print("\n加载细胞图像数据...")
        image_dir = self.merged_datasets_dir / "04_cell_images"
        
        if not image_dir.exists():
            print(f"  目录不存在：{image_dir}")
            return
        
        image_paths = []
        for subdir in ["images_train", "images_test", "gfp_train", "gfp_test"]:
            subdir_path = image_dir / subdir
            if subdir_path.exists():
                npy_files = list(subdir_path.glob("*.npy"))
                image_paths.extend(npy_files)
                print(f"  {subdir}: {len(npy_files)} 个图像")
        
        self.data["image_paths"] = image_paths
        
        stats_file = image_dir / "cell_stats.npy"
        if stats_file.exists():
            try:
                stats_array = np.load(stats_file, allow_pickle=True)
                if stats_array.size == 1:
                    self.image_stats = stats_array.item()
                else:
                    self.image_stats = {'mean': 0.0, 'std': 1.0}
            except:
                self.image_stats = {'mean': 0.0, 'std': 1.0}
        
        print(f"  总计 {len(image_paths)} 个细胞图像")
        
    def _sample_data(self):
        if self.sample_fraction >= 1.0:
            return
        
        if len(self.smiles_list) == 0:
            print("  警告：SMILES 列表为空，跳过采样")
            return
        
        n = max(1, int(len(self.smiles_list) * self.sample_fraction))
        n = min(n, len(self.smiles_list))
        
        original_n = len(self.smiles_list)
        indices = np.random.choice(original_n, n, replace=False)
        if len(self.df) == len(self.smiles_list):
            self.df = self.df.iloc[indices].reset_index(drop=True)
        self.smiles_list = [self.smiles_list[i] for i in indices]
        self.targets = [self.targets[i] for i in indices]
        for key in self.data:
            if isinstance(self.data[key], np.ndarray) and len(self.data[key]) > 0:
                if len(self.data[key]) == original_n:
                    self.data[key] = self.data[key][indices]
        
        print(f"\n采样后数据集大小：{len(self.smiles_list)}")
    
    def _precache_all_features(self):
        """预缓存所有样本的分子特征到内存，避免 __getitem__ 中重复计算 RDKit"""
        print("\n预缓存分子特征...")
        self._feature_cache = {}
        
        for idx, smiles in enumerate(self.smiles_list):
            cache = {}
            
            # 分子描述符
            if self.descriptor_dim:
                desc = self.molecular_descriptor_map.get(smiles)
                if desc is None:
                    desc = np.zeros(self.descriptor_dim, dtype=np.float32)
                cache["molecular_descriptors"] = torch.tensor(desc, dtype=torch.float32)
            
            # 配方特征
            if "formulation_features" in self.data and len(self.data["formulation_features"]) > idx:
                cache["formulation_features"] = torch.tensor(self.data["formulation_features"][idx], dtype=torch.float32)
            
            if "physical_properties" in self.data and len(self.data["physical_properties"]) > idx:
                cache["physical_properties"] = torch.tensor(self.data["physical_properties"][idx], dtype=torch.float32)
            
            # 四组分特征
            row = self.df.iloc[idx]
            for component in ["ionizable", "helper", "cholesterol", "peg"]:
                comp_smiles = self._get_component_smiles(row, component)
                if comp_smiles:
                    cache.update(self._get_graph_features(comp_smiles, prefix=component))
                    cache[f"{component}_fingerprint"] = self._compute_fingerprint(comp_smiles)
                else:
                    # 缺失组分补零
                    base_key = f"{component}_"
                    cache[f"{base_key}atom_features"] = torch.zeros(1, 39, dtype=torch.float32)
                    cache[f"{base_key}bond_features"] = torch.zeros(1, 4, dtype=torch.float32)
                    cache[f"{base_key}edge_index"] = torch.zeros(2, 1, dtype=torch.long)
                    cache[f"{component}_fingerprint"] = torch.zeros(self.fingerprint_dim, dtype=torch.float32)
            
            # 摩尔比
            molar_ratios = []
            for col in ["cationic_lipid_mol_ratio", "phospholipid_mol_ratio",
                        "cholesterol_mol_ratio", "peg_lipid_mol_ratio"]:
                if col in row and not pd.isna(row[col]):
                    try:
                        molar_ratios.append(float(row[col]))
                    except (ValueError, TypeError):
                        molar_ratios.append(0.0)
                else:
                    molar_ratios.append(0.0)
            while len(molar_ratios) < 4:
                molar_ratios.append(0.0)
            cache["molar_ratios"] = torch.FloatTensor(molar_ratios[:4])
            
            # 任务目标
            cache.update(self._get_task_targets(row))
            
            self._feature_cache[idx] = cache
        
        print(f"预缓存完成！共 {len(self._feature_cache)} 个样本\n")
        
    def __len__(self) -> int:
        return len(self.smiles_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取样本 - 直接从预缓存读取，避免运行时 RDKit 计算"""
        # 从预缓存获取所有特征
        sample = self._feature_cache[idx]
        
        # 添加基本信息
        sample["smiles"] = self.smiles_list[idx]
        sample["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        sample["idx"] = idx
        
        return sample

    def _get_task_targets(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        task_specs = {
            "efficiency": ["target", "quantified_delivery", "transfection_efficiency"],
            "particle_size": ["particle_size_nm_std", "particle_size_nm", "particle_size", "size_nm"],
            "zeta_potential": ["zeta_potential_mv_std", "zeta_potential_mv", "zeta_potential"],
            "toxicity": ["toxicity", "cell_viability", "viability"],
        }
        result = {}
        for task_name, cols in task_specs.items():
            value = np.nan
            for col in cols:
                if col in row:
                    value = pd.to_numeric(row[col], errors="coerce")
                    if not pd.isna(value):
                        break
            if pd.isna(value):
                result[task_name] = torch.tensor(0.0, dtype=torch.float32)
                result[f"{task_name}_mask"] = torch.tensor(False)
            else:
                result[task_name] = torch.tensor(float(value), dtype=torch.float32)
                result[f"{task_name}_mask"] = torch.tensor(True)
        return result

    def _get_component_smiles(self, row: pd.Series, component: str) -> str:
        candidates = {
            "ionizable": ["ionizable_lipid_smiles", "smiles", "full_smiles", "amine_smiles", "ionizable_lipid"],
            "helper": ["helper_lipid_smiles", "phospholipid_smiles", "helper_lipid", "helper_lipid_id"],
            "cholesterol": ["sterol_lipid_smiles", "cholesterol_smiles", "sterol_lipid", "sterol", "cholesterol"],
            "peg": ["peg_lipid_smiles", "peg_lipid", "peg"],
        }[component]
        raw = self._get_smiles_from_row(row, candidates)
        resolved = self._resolve_lipid_name(raw, component)
        if resolved:
            return resolved
        if component == "cholesterol":
            return self.DEFAULT_CHOLESTEROL_SMILES
        return ""

    def _resolve_lipid_name(self, value: str, component: str) -> str:
        if not value:
            return ""
        value = str(value).strip()
        key = value.upper()
        if component == "helper":
            mapped = self.HELPER_LIPID_SMILES.get(key)
            if mapped is not None:
                return mapped
        if component == "cholesterol" and key in {"CHOLESTEROL", "STEROL"}:
            return self.DEFAULT_CHOLESTEROL_SMILES
        if component == "peg" and "PEG" in key:
            return ""
        if self._get_mol(value) is not None:
            return value
        return ""
    
    def _get_smiles_from_row(self, row: pd.Series, candidate_cols: List[str]) -> str:
        """从行中提取 SMILES，尝试多个候选列名"""
        for col in candidate_cols:
            if col in row:
                smiles = row[col]
                if not pd.isna(smiles) and isinstance(smiles, str) and len(smiles) > 0:
                    return smiles
        return ""
    
    def _compute_fingerprint(self, smiles: str) -> torch.Tensor:
        """计算分子指纹"""
        if not smiles:
            return torch.zeros(self.fingerprint_dim)
        if smiles in self._fingerprint_cache:
            return self._fingerprint_cache[smiles].clone()

        mol = self._get_mol(smiles)
        if mol is None:
            return torch.zeros(self.fingerprint_dim)
        
        morgan_gen = GetMorganGenerator(radius=2, fpSize=self.fingerprint_dim)
        try:
            fp = morgan_gen.GetFingerprintAsNumPy(mol)
            tensor = torch.tensor(fp, dtype=torch.float32)
            self._fingerprint_cache[smiles] = tensor
            return tensor.clone()
        except:
            return torch.zeros(self.fingerprint_dim)
    
    def _get_graph_features(self, smiles: str, prefix: str = "") -> Dict[str, torch.Tensor]:
        """获取分子图特征（支持 prefix 用于区分不同组分）"""
        cache_key = (smiles, prefix)
        if cache_key in self._graph_cache:
            return {k: v.clone() for k, v in self._graph_cache[cache_key].items()}

        mol = self._get_mol(smiles) if smiles else None
        
        if mol is None:
            base_key = f"{prefix}_" if prefix else ""
            result = {
                f"{base_key}atom_features": torch.zeros(1, 39),
                f"{base_key}bond_features": torch.zeros(1, 4),
                f"{base_key}edge_index": torch.zeros(2, 1, dtype=torch.long),
                f"{base_key}atom_mask": torch.zeros(1, dtype=torch.bool),
            }
            self._graph_cache[cache_key] = result
            return {k: v.clone() for k, v in result.items()}
        
        # 原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            feat = self._atom_to_features(atom)
            atom_features.append(feat)
        
        atom_features = torch.FloatTensor(atom_features)
        
        # 键特征和 edge_index
        bond_features = []
        edge_index_list = []
        num_atoms = len(atom_features)
        
        for bond in mol.GetBonds():
            src = bond.GetBeginAtomIdx()
            dst = bond.GetEndAtomIdx()
            feat = self._bond_to_features(bond)
            
            # 双向边
            bond_features.append(feat)
            bond_features.append(feat)
            edge_index_list.append([src, dst])
            edge_index_list.append([dst, src])
        
        if bond_features:
            bond_features = torch.FloatTensor(bond_features)
            edge_index = torch.LongTensor(edge_index_list).t()  # (2, num_edges)
        else:
            bond_features = torch.zeros(1, 4)
            edge_index = torch.zeros(2, 1, dtype=torch.long)
        
        # 不再填充到最大原子数，直接使用实际原子数
        # 这样可以保证 atom_features、edge_index、bond_features 的数量一致
        # 注意：不再需要 atom_mask，因为所有原子都是有效的（没有 padding）
        base_key = f"{prefix}_" if prefix else ""
        result = {
            f"{base_key}atom_features": atom_features,  # [num_atoms, 39]
            f"{base_key}bond_features": bond_features,  # [num_edges, 4]
            f"{base_key}edge_index": edge_index,  # [2, num_edges]
        }
        self._graph_cache[cache_key] = result
        return {k: v.clone() for k, v in result.items()}

    def _get_mol(self, smiles: str):
        if not smiles:
            return None
        if smiles not in self._mol_cache:
            self._mol_cache[smiles] = Chem.MolFromSmiles(smiles)
        return self._mol_cache[smiles]
    
    def _atom_to_features(self, atom) -> List[float]:
        """原子转特征向量 (39 维)"""
        feat = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            atom.GetHybridization().real,
            float(atom.GetIsAromatic()),
            atom.GetTotalNumHs(),
            float(atom.IsInRing()),
        ]
        # 填充到 39 维
        while len(feat) < 39:
            feat.append(0.0)
        return feat[:39]
    
    def _bond_to_features(self, bond) -> List[float]:
        """化学键转特征向量 (4 维)"""
        return [
            float(bond.GetBondType().real),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            bond.GetStereo(),
        ]


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """批量处理函数 - 支持四组分配方特征（使用 PyTorch Geometric batch 方式）"""
    result = {
        "smiles": [item["smiles"] for item in batch],
        "target": torch.stack([item["target"] for item in batch]),
        "idx": torch.tensor([item["idx"] for item in batch]),
    }
    
    # 1. 处理常规字段（形状固定的）
    for field in [
        "molecular_descriptors", "formulation_features", "physical_properties",
        "images", "atom_features", "fingerprint",
        "efficiency", "particle_size", "zeta_potential", "toxicity",
        "efficiency_mask", "particle_size_mask", "zeta_potential_mask", "toxicity_mask",
    ]:
        items = [item.get(field) for item in batch]
        if all(item is not None for item in items):
            result[field] = torch.stack(items)
    
    # 2. 处理四组分特征
    dummy_atom_features = torch.zeros(1, 39, dtype=torch.float32)
    dummy_bond_features = torch.zeros(1, 4, dtype=torch.float32)
    dummy_edge_index = torch.zeros(2, 1, dtype=torch.long)
    
    for component in ["ionizable", "helper", "cholesterol", "peg"]:
        # 原子特征、边索引和键特征（使用 PyTorch Geometric batch 方式）
        # 缺失组分补全零图，保证每个组分编码器都输出完整 batch_size。
        atom_feat_items = []
        edge_items = []
        bond_items = []
        
        for item in batch:
            atom_feat = item.get(f"{component}_atom_features")
            edge_index = item.get(f"{component}_edge_index")
            bond_feat = item.get(f"{component}_bond_features")
            
            if atom_feat is None or edge_index is None or bond_feat is None:
                atom_feat = dummy_atom_features
                edge_index = dummy_edge_index
                bond_feat = dummy_bond_features
            
            atom_feat_items.append(atom_feat)
            edge_items.append(edge_index)
            bond_items.append(bond_feat)
        
        num_atoms_list = [item.shape[0] for item in atom_feat_items]
        offsets = torch.cumsum(torch.tensor([0] + num_atoms_list[:-1]), dim=0)
        
        result[f"{component}_atom_features"] = torch.cat(atom_feat_items, dim=0)
        
        edge_list = []
        for i, edge_item in enumerate(edge_items):
            edge_list.append(edge_item + offsets[i])
        result[f"{component}_edge_index"] = torch.cat(edge_list, dim=1)
        result[f"{component}_bond_features"] = torch.cat(bond_items, dim=0)
        
        batch_vec = torch.repeat_interleave(
            torch.arange(len(batch)),
            torch.tensor(num_atoms_list),
        )
        result[f"{component}_batch"] = batch_vec
        
        # 指纹（形状固定：[fingerprint_dim]）
        fp_key = f"{component}_fingerprint"
        fp_template = next((item[fp_key] for item in batch if fp_key in item), torch.zeros(2048, dtype=torch.float32))
        items = [item.get(fp_key, torch.zeros_like(fp_template)) for item in batch]
        result[f"{component}_fingerprint"] = torch.stack(items)
    
    # 3. 处理摩尔比
    if "molar_ratios" in batch[0]:
        items = [item["molar_ratios"] for item in batch if "molar_ratios" in item]
        if items:
            result["molar_ratios"] = torch.stack(items)
    
    return result


def create_dataloaders(
    merged_datasets_dir: str,
    modalities: List[str] = ["molecular", "formulation", "images"],
    batch_size: int = 32,
    num_workers: int = 4,
    sample_fraction: float = 1.0,
    random_seed: int = 42,
    pin_memory: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader, 'MergedMultiModalDataset']:
    dataset = MergedMultiModalDataset(
        merged_datasets_dir=merged_datasets_dir,
        modalities=modalities,
        sample_fraction=sample_fraction,
        random_seed=random_seed,
    )
    
    n = len(dataset)
    n_train, n_val = int(0.7 * n), int(0.15 * n)
    n_test = n - n_train - n_val
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(random_seed)
    )
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=random_seed,
            drop_last=False,
        )
    
    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        ),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory),
        dataset,  # 返回原始数据集以便访问标准化参数
    )
