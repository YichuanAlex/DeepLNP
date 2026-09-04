"""
多模态数据集加载器 - 支持所有数据集类型

支持的数据类型：
- CSV 表格数据
- PNG 图片
- NPY 数组（显微镜图像）
- PDB 结构文件
- ITP 参数文件
- PKL 指纹文件
- JSON embedding 文件
"""

import os
import json
import pickle
import hashlib
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator


class MultiModalLNPDataset(Dataset):
    """
    多模态 LNP 数据集
    
    支持加载：
    1. 表格数据（SMILES, 理化性质，配方参数）
    2. 分子指纹（PKL 文件）
    3. 分子嵌入（JSON 文件）
    4. 显微镜图像（NPY 文件）
    5. 3D 结构（PDB 文件）
    6. 参数文件（ITP 文件）
    7. 结果图片（PNG 文件）
    """
    
    def __init__(
        self,
        dataset_name: str,
        clean_dir: str,
        origin_dir: str,
        mode: str = "train",
        max_atoms: int = 256,
        fingerprint_dim: int = 2048,
        image_size: int = 224,
        use_augmentation: bool = False,
    ):
        """
        Args:
            dataset_name: 数据集名称 (agile, lantern, lipobart, lnp_atlas, lnp_ml, m3_lipids, phil_lnp, transma)
            clean_dir: 清洗后数据目录
            origin_dir: 原始数据目录
            mode: train/val/test
            max_atoms: 最大原子数
            fingerprint_dim: 指纹维度
            image_size: 图像大小
            use_augmentation: 是否使用数据增强
        """
        self.dataset_name = dataset_name
        self.clean_dir = Path(clean_dir)
        self.origin_dir = Path(origin_dir)
        self.mode = mode
        self.max_atoms = max_atoms
        self.fingerprint_dim = fingerprint_dim
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        
        # 加载表格数据
        self.df = self._load_csv_data()
        
        # 加载多模态数据
        self.fingerprints = {}
        self.embeddings = {}
        self.images = {}
        self.structures = {}
        self.parameters = {}
        
        self._load_multimodal_data()
        
    def _load_csv_data(self) -> pd.DataFrame:
        """加载 CSV 表格数据"""
        # 尝试两种路径：直接在 clean_dir 下或在子目录中
        csv_file = self.clean_dir / f"{self.dataset_name}_deeplnp.csv"
        if not csv_file.exists():
            csv_file = self.clean_dir / self.dataset_name / f"{self.dataset_name}_deeplnp.csv"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV 文件不存在：{csv_file}")
        
        df = pd.read_csv(csv_file)
        
        # 根据数据集名称设置特定列
        if self.dataset_name == "agile":
            target_col = None  # AGILE 没有转染效率标签
            smiles_col = "ionizable_lipid"
        elif self.dataset_name == "lantern":
            target_col = "transfection_efficiency"
            smiles_col = "ionizable_lipid"
        elif self.dataset_name == "lipobart":
            target_col = "transfection_efficiency"
            smiles_col = "ionizable_lipid"
        elif self.dataset_name == "lnp_atlas":
            target_col = "particle_size_nm_std"
            smiles_col = "ionizable_lipid_smiles"
        elif self.dataset_name == "lnp_ml":
            target_col = "quantified_delivery"
            smiles_col = "ionizable_lipid"
        elif self.dataset_name == "m3_lipids":
            target_col = None
            smiles_col = "ionizable_lipid"
        elif self.dataset_name == "phil_lnp":
            target_col = None
            smiles_col = "ionizable_lipid"
        elif self.dataset_name == "transma":
            target_col = "transfection_efficiency"
            smiles_col = "ionizable_lipid"
        else:
            raise ValueError(f"未知数据集：{self.dataset_name}")
        
        # 移除缺失 SMILES 的行
        if smiles_col in df.columns:
            df = df.dropna(subset=[smiles_col])
            df = df[df[smiles_col].astype(str).str.len() > 0]
        
        # 对 target 进行标准化（除了 lnp_ml 已经标准化）
        if target_col and target_col in df.columns:
            if self.dataset_name != "lnp_ml":  # lnp_ml 已经标准化
                target_data = df[target_col].dropna()
                if len(target_data) > 0:
                    mean = target_data.mean()
                    std = target_data.std()
                    if std > 0:
                        df.loc[:, target_col] = (df[target_col] - mean) / std
                        print(f"✓ {self.dataset_name} Target 标准化：mean={mean:.4f}, std={std:.4f}")
        
        return df
    
    def _load_multimodal_data(self):
        """加载多模态数据"""
        print(f"\n加载 {self.dataset_name} 多模态数据...")
        
        # 加载指纹（PKL）
        if self.dataset_name == "lantern":
            self._load_fingerprints()
        
        # 加载嵌入（JSON）
        if self.dataset_name == "lipobart":
            self._load_embeddings()
        
        # 加载图像
        if self.dataset_name in ["m3_lipids", "phil_lnp", "transma"]:
            self._load_images()
        
        # 加载 3D 结构
        if self.dataset_name == "m3_lipids":
            self._load_structures()
            self._load_parameters()
    
    def _load_fingerprints(self):
        """加载 LANTERN 的指纹数据"""
        fp_dir = self.clean_dir / self.dataset_name / "fingerprints"
        if not fp_dir.exists():
            fp_dir = self.clean_dir / "fingerprints"
        if not fp_dir.exists():
            print(f"  ⚠️  指纹目录不存在：{fp_dir}")
            return
        
        fp_files = ["circular.pkl", "expert.pkl", "grover.pkl"]
        
        for fp_file in fp_files:
            fp_path = fp_dir / fp_file
            if fp_path.exists():
                with open(fp_path, "rb") as f:
                    fps = pickle.load(f)
                
                fp_type = fp_file.replace(".pkl", "")
                self.fingerprints[fp_type] = fps
                print(f"  ✓ 加载指纹：{fp_type} ({len(fps)} 条)")
    
    def _load_embeddings(self):
        """加载 LipoBART 的嵌入数据"""
        emb_dir = self.clean_dir / self.dataset_name / "embeddings"
        if not emb_dir.exists():
            emb_dir = self.clean_dir / "embeddings"
        if not emb_dir.exists():
            print(f"  ⚠️  嵌入目录不存在：{emb_dir}")
            return
        
        emb_files = list(emb_dir.glob("*.json"))
        
        for emb_file in emb_files:
            with open(emb_file, "r") as f:
                embs = json.load(f)
            
            emb_type = emb_file.stem
            self.embeddings[emb_type] = embs
            print(f"  ✓ 加载嵌入：{emb_type} ({len(embs)} 条)")
    
    def _load_images(self):
        """加载图像数据"""
        import cv2
        
        # 不同类型的图像目录
        if self.dataset_name == "m3_lipids":
            img_dir = self.clean_dir / self.dataset_name / "images"
        elif self.dataset_name == "phil_lnp":
            img_dir = self.clean_dir / self.dataset_name / "microscopy_images"
        elif self.dataset_name == "transma":
            img_dir = self.clean_dir / self.dataset_name / "figures"
        else:
            return
        
        if not img_dir.exists():
            print(f"  ⚠️  图像目录不存在：{img_dir}")
            return
        
        # 加载 PNG 图像
        png_files = list(img_dir.rglob("*.png"))
        for png_file in png_files:
            try:
                img = cv2.imread(str(png_file))
                if img is not None:
                    img = cv2.resize(img, (self.image_size, self.image_size))
                    img = torch.FloatTensor(img).permute(2, 0, 1) / 255.0
                    self.images[png_file.stem] = img
            except Exception as e:
                print(f"  ⚠️  加载图像失败 {png_file}: {e}")
        
        # 加载 NPY 图像（phil_lnp）
        npy_files = list(img_dir.glob("*.npy"))
        for npy_file in npy_files:
            try:
                img = np.load(npy_file)
                img = torch.FloatTensor(img)
                if img.dim() == 2:
                    img = img.unsqueeze(0)
                self.images[npy_file.stem] = img
            except Exception as e:
                print(f"  ⚠️  加载 NPY 图像失败 {npy_file}: {e}")
        
        print(f"  ✓ 加载图像：{len(self.images)} 张")
    
    def _load_structures(self):
        """加载 3D 结构文件（PDB）"""
        struct_dir = self.clean_dir / self.dataset_name / "structures"
        if not struct_dir.exists():
            struct_dir = self.clean_dir / "structures"
        if not struct_dir.exists():
            print(f"  ⚠️  结构目录不存在：{struct_dir}")
            return
        
        pdb_files = list(struct_dir.rglob("*.pdb"))
        for pdb_file in pdb_files:
            try:
                with open(pdb_file, "r") as f:
                    pdb_content = f.read()
                self.structures[pdb_file.stem] = pdb_content
            except Exception as e:
                print(f"  ⚠️  加载 PDB 失败 {pdb_file}: {e}")
        
        print(f"  ✓ 加载 PDB 结构：{len(self.structures)} 个")
    
    def _load_parameters(self):
        """加载参数文件（ITP）"""
        param_dir = self.clean_dir / self.dataset_name / "parameters"
        if not param_dir.exists():
            param_dir = self.clean_dir / "parameters"
        if not param_dir.exists():
            print(f"  ⚠️  参数目录不存在：{param_dir}")
            return
        
        itp_files = list(param_dir.rglob("*.itp"))
        for itp_file in itp_files:
            try:
                with open(itp_file, "r") as f:
                    itp_content = f.read()
                self.parameters[itp_file.stem] = itp_content
            except Exception as e:
                print(f"  ⚠️  加载 ITP 失败 {itp_file}: {e}")
        
        print(f"  ✓ 加载 ITP 参数：{len(self.parameters)} 个")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本 - 支持四组分配方输入"""
        row = self.df.iloc[idx]
        
        item = {
            "idx": idx,
            "dataset": self.dataset_name,
        }
        
        # ========== 核心修改：支持四组分配方 ==========
        # 1. 可电离脂质（主要功能组分）
        ionizable_smiles = self._get_smiles_from_row(row, ["ionizable_lipid", "ionizable_lipid_smiles", "smiles"])
        item["ionizable_smiles"] = ionizable_smiles
        item["ionizable_fingerprint"] = self._compute_fingerprint(ionizable_smiles)
        item.update(self._get_graph_features(ionizable_smiles, prefix="ionizable"))
        
        # 2. 辅助脂质（DOPE/DSPC 等）
        helper_smiles = self._get_smiles_from_row(row, ["helper_lipid", "helper_lipid_smiles", "auxiliary_lipid"])
        item["helper_smiles"] = helper_smiles
        if helper_smiles:
            item["helper_fingerprint"] = self._compute_fingerprint(helper_smiles)
            item.update(self._get_graph_features(helper_smiles, prefix="helper"))
        
        # 3. 胆固醇（膜稳定性）
        cholesterol_smiles = self._get_smiles_from_row(row, ["cholesterol", "cholesterol_type", "sterol"])
        item["cholesterol_smiles"] = cholesterol_smiles
        if cholesterol_smiles:
            item["cholesterol_fingerprint"] = self._compute_fingerprint(cholesterol_smiles)
            item.update(self._get_graph_features(cholesterol_smiles, prefix="cholesterol"))
        
        # 4. PEG 脂质（稳定性/长循环）
        peg_smiles = self._get_smiles_from_row(row, ["peg_lipid", "peg", "peg_ligand"])
        item["peg_smiles"] = peg_smiles
        if peg_smiles:
            item["peg_fingerprint"] = self._compute_fingerprint(peg_smiles)
            item.update(self._get_graph_features(peg_smiles, prefix="peg"))
        
        # 5. 摩尔比（核心配方参数）
        molar_ratios = self._get_molar_ratios(row)
        item["molar_ratios"] = torch.FloatTensor(molar_ratios)
        
        # ========== 目标值（多任务学习）==========
        item["target"] = self._get_target_value(row)
        
        # ========== 额外理化性质（辅助预测）==========
        item["particle_size"] = self._get_physical_property(row, "particle_size")
        item["zeta_potential"] = self._get_physical_property(row, "zeta_potential")
        item["pdi"] = self._get_physical_property(row, "pdi")
        
        # ========== 多模态数据 ==========
        if self.dataset_name == "lantern":
            item["fingerprint_extra"] = self._get_fingerprint_from_dict(idx)
        elif self.dataset_name == "lipobart":
            item["embedding"] = self._get_embedding_from_dict(idx)
        elif self.dataset_name in ["m3_lipids", "phil_lnp", "transma"]:
            item["image"] = self._get_image_from_dict(idx)
        
        if self.dataset_name == "m3_lipids":
            item["structure"] = self._get_structure_from_dict(idx)
            item["parameter"] = self._get_parameter_from_dict(idx)
        
        return item
    
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
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.zeros(self.fingerprint_dim)
        
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=self.fingerprint_dim)
        fp = fpgen.GetFingerprintAsNumPy(mol)
        return torch.FloatTensor(fp)
    
    def _get_molar_ratios(self, row: pd.Series) -> List[float]:
        """
        提取摩尔比 [ionizable, helper, cholesterol, PEG]
        参考 COMET 项目的格式：y1, y2, p1, p2, p3, p4, m1, m2, m3, m4
        """
        ratios = [0.0, 0.0, 0.0, 0.0]  # 默认值
        
        # 尝试多种列名格式
        ratio_mappings = {
            0: ["ionizable_ratio", "m1", "ratio_ionizable", "ionizable_molar_ratio"],
            1: ["helper_ratio", "m2", "ratio_helper", "helper_molar_ratio"],
            2: ["cholesterol_ratio", "m3", "ratio_cholesterol", "cholesterol_molar_ratio"],
            3: ["peg_ratio", "m4", "ratio_peg", "peg_molar_ratio"],
        }
        
        for i, candidates in ratio_mappings.items():
            for col in candidates:
                if col in row and not pd.isna(row[col]):
                    ratios[i] = float(row[col])
                    break
        
        # 如果没有找到，尝试从摩尔比列表解析
        if "molar_ratios" in row and not pd.isna(row["molar_ratios"]):
            try:
                ratio_str = row["molar_ratios"]
                if isinstance(ratio_str, str):
                    # 处理 "[25,30,30,1]" 或 "25:30:30:1" 格式
                    ratio_str = ratio_str.strip("[]").replace(":", ",")
                    parsed = [float(x.strip()) for x in ratio_str.split(",")]
                    if len(parsed) >= 4:
                        ratios = parsed[:4]
            except:
                pass
        
        # 归一化（可选）
        total = sum(ratios)
        if total > 0:
            ratios = [r / total for r in ratios]
        
        return ratios
    
    def _get_target_value(self, row: pd.Series) -> torch.Tensor:
        """获取目标值（支持多任务）"""
        # 转染效率（主要目标）
        if "transfection_efficiency" in row and not pd.isna(row["transfection_efficiency"]):
            return torch.FloatTensor([row["transfection_efficiency"]])
        elif "quantified_delivery" in row and not pd.isna(row["quantified_delivery"]):
            return torch.FloatTensor([row["quantified_delivery"]])
        elif "particle_size_nm_std" in row and not pd.isna(row["particle_size_nm_std"]):
            return torch.FloatTensor([row["particle_size_nm_std"]])
        else:
            return torch.zeros(1)
    
    def _get_physical_property(self, row: pd.Series, prop_type: str) -> torch.Tensor:
        """获取理化性质"""
        col_mappings = {
            "particle_size": ["particle_size_nm", "particle_size", "size_nm", "diameter_nm"],
            "zeta_potential": ["zeta_potential_mv", "zeta_potential", "zeta_mv"],
            "pdi": ["pdi", "polydispersity"],
        }
        
        candidates = col_mappings.get(prop_type, [])
        for col in candidates:
            if col in row and not pd.isna(row[col]):
                return torch.FloatTensor([row[col]])
        
        return torch.zeros(1)
    
    def _get_graph_features(self, smiles: str, prefix: str = "") -> Dict[str, torch.Tensor]:
        """获取分子图特征（支持 prefix 用于区分不同组分）"""
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        
        if mol is None:
            base_key = f"{prefix}_" if prefix else ""
            return {
                f"{base_key}atom_features": torch.zeros(1, 39),
                f"{base_key}bond_features": torch.zeros(1, 4),
                f"{base_key}edge_index": torch.zeros(2, 1, dtype=torch.long),
                f"{base_key}atom_mask": torch.zeros(1, dtype=torch.bool),
            }
        
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
        
        # 填充到最大原子数
        if num_atoms < self.max_atoms:
            pad_atoms = self.max_atoms - num_atoms
            atom_features = torch.cat([
                atom_features,
                torch.zeros(pad_atoms, 39)
            ], dim=0)
            atom_mask = torch.cat([
                torch.ones(num_atoms, dtype=torch.bool),
                torch.zeros(pad_atoms, dtype=torch.bool)
            ], dim=0)
        else:
            atom_features = atom_features[:self.max_atoms]
            atom_mask = torch.ones(self.max_atoms, dtype=torch.bool)
        
        # 使用 prefix 区分不同组分
        base_key = f"{prefix}_" if prefix else ""
        return {
            f"{base_key}atom_features": atom_features,
            f"{base_key}bond_features": bond_features,
            f"{base_key}edge_index": edge_index,
            f"{base_key}atom_mask": atom_mask,
        }
    
    def _atom_to_features(self, atom) -> List[float]:
        """将原子转换为特征向量"""
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            atom.GetHybridization().real,
            atom.GetNumImplicitHs(),
            atom.IsInRing(),
            atom.GetIsAromatic(),
        ]
        return features + [0] * (39 - len(features))
    
    def _bond_to_features(self, bond) -> List[float]:
        """将键转换为特征向量"""
        features = [
            bond.GetBondType().real,
            bond.GetIsConjugated(),
            bond.IsInRing(),
            bond.GetStereo().real,
        ]
        return features
    
    def _get_formulation_features(self, row: pd.Series) -> torch.Tensor:
        """获取配方特征"""
        features = []
        
        # 摩尔比
        ratio_cols = ["ionizable_ratio", "helper_ratio", "cholesterol_ratio", "peg_ratio"]
        for col in ratio_cols:
            if col in row and not pd.isna(row[col]):
                features.append(float(row[col]))
            else:
                features.append(0.0)
        
        # 粒径
        if "particle_size_nm_std" in row and not pd.isna(row["particle_size_nm_std"]):
            features.append(float(row["particle_size_nm_std"]))
        else:
            features.append(0.0)
        
        # Zeta 电位
        if "zeta_potential_mv_std" in row and not pd.isna(row["zeta_potential_mv_std"]):
            features.append(float(row["zeta_potential_mv_std"]))
        else:
            features.append(0.0)
        
        # PDI
        if "pdi_std" in row and not pd.isna(row["pdi_std"]):
            features.append(float(row["pdi_std"]))
        else:
            features.append(0.0)
        
        return torch.FloatTensor(features)
    
    def _get_fingerprint_from_dict(self, idx: int) -> torch.Tensor:
        """从字典获取指纹"""
        # 尝试获取 SMILES 对应的指纹
        smiles = self.df.iloc[idx].get("ionizable_lipid", "")
        
        for fp_type, fp_dict in self.fingerprints.items():
            if isinstance(fp_dict, dict) and smiles in fp_dict:
                return torch.FloatTensor(fp_dict[smiles])
            elif isinstance(fp_dict, list) and idx < len(fp_dict):
                return torch.FloatTensor(fp_dict[idx])
        
        return torch.zeros(self.fingerprint_dim)
    
    def _get_embedding_from_dict(self, idx: int) -> torch.Tensor:
        """从字典获取嵌入"""
        lipid_id = self.df.iloc[idx].get("lipid_id", idx)
        
        for emb_type, emb_dict in self.embeddings.items():
            if isinstance(emb_dict, dict):
                # 尝试多种键
                for key in [str(lipid_id), f"lipid_{lipid_id}", idx]:
                    if key in emb_dict:
                        emb = emb_dict[key]
                        if isinstance(emb, list):
                            return torch.FloatTensor(emb)
        
        return torch.zeros(512)  # 默认嵌入维度
    
    def _get_image_from_dict(self, idx: int) -> torch.Tensor:
        """从字典获取图像"""
        # 尝试获取与索引相关的图像
        if self.dataset_name == "phil_lnp":
            # phil_lnp 使用 well_id 和 cell_id
            # 这里简化处理，返回第一张图或随机图
            if self.images:
                return list(self.images.values())[0]
        elif self.dataset_name == "m3_lipids":
            if self.images:
                return list(self.images.values())[0]
        elif self.dataset_name == "transma":
            if self.images:
                return list(self.images.values())[0]
        
        return torch.zeros(3, self.image_size, self.image_size)
    
    def _get_structure_from_dict(self, idx: int) -> str:
        """从字典获取结构"""
        if self.structures:
            return list(self.structures.values())[0]
        return ""
    
    def _get_parameter_from_dict(self, idx: int) -> str:
        """从字典获取参数"""
        if self.parameters:
            return list(self.parameters.values())[0]
        return ""


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """批量数据整理函数 - 支持四组分配方"""
    
    def batch_edge_index(batch_data, prefix):
        """处理 edge_index 的 batch"""
        edge_indices = []
        num_atoms_cumsum = 0
        
        for item in batch_data:
            edge_key = f"{prefix}_edge_index"
            if edge_key in item:
                edge_index = item[edge_key]
                if edge_index.size(1) > 0:
                    adjusted_edge_index = edge_index.clone()
                    adjusted_edge_index[0] += num_atoms_cumsum
                    adjusted_edge_index[1] += num_atoms_cumsum
                    edge_indices.append(adjusted_edge_index)
                    num_atoms_cumsum += item[f"{prefix}_atom_features"].size(0)
                else:
                    edge_indices.append(torch.zeros(2, 0, dtype=torch.long))
                    num_atoms_cumsum += item[f"{prefix}_atom_features"].size(0)
        
        if edge_indices:
            return torch.cat(edge_indices, dim=1)
        return torch.zeros(2, 0, dtype=torch.long)
    
    def batch_bond_features(batch_data, prefix):
        """处理 bond_features 的 batch"""
        bond_features_list = []
        for item in batch_data:
            bond_key = f"{prefix}_bond_features"
            if bond_key in item:
                bond_features_list.append(item[bond_key])
        
        if bond_features_list:
            return torch.cat(bond_features_list, dim=0)
        return torch.zeros(0, 4)
    
    result = {
        "idx": torch.LongTensor([item["idx"] for item in batch]),
        "dataset": [item["dataset"] for item in batch],
        
        # 主要组分（可电离脂质）
        "ionizable_smiles": [item["ionizable_smiles"] for item in batch],
        "ionizable_fingerprint": torch.stack([item["ionizable_fingerprint"] for item in batch]),
        "ionizable_atom_features": torch.cat([item["ionizable_atom_features"] for item in batch], dim=0),
        "ionizable_atom_mask": torch.cat([item["ionizable_atom_mask"] for item in batch], dim=0),
        
        # 核心配方参数
        "molar_ratios": torch.stack([item["molar_ratios"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
    }
    
    # 辅助脂质（可选）
    if "helper_smiles" in batch[0]:
        result["helper_smiles"] = [item["helper_smiles"] for item in batch]
        result["helper_fingerprint"] = torch.stack([item["helper_fingerprint"] for item in batch])
        result["helper_atom_features"] = torch.cat([item["helper_atom_features"] for item in batch], dim=0)
        result["helper_atom_mask"] = torch.cat([item["helper_atom_mask"] for item in batch], dim=0)
    
    # 胆固醇（可选）
    if "cholesterol_smiles" in batch[0]:
        result["cholesterol_smiles"] = [item["cholesterol_smiles"] for item in batch]
        result["cholesterol_fingerprint"] = torch.stack([item["cholesterol_fingerprint"] for item in batch])
        result["cholesterol_atom_features"] = torch.cat([item["cholesterol_atom_features"] for item in batch], dim=0)
        result["cholesterol_atom_mask"] = torch.cat([item["cholesterol_atom_mask"] for item in batch], dim=0)
    
    # PEG 脂质（可选）
    if "peg_smiles" in batch[0]:
        result["peg_smiles"] = [item["peg_smiles"] for item in batch]
        result["peg_fingerprint"] = torch.stack([item["peg_fingerprint"] for item in batch])
        result["peg_atom_features"] = torch.cat([item["peg_atom_features"] for item in batch], dim=0)
        result["peg_atom_mask"] = torch.cat([item["peg_atom_mask"] for item in batch], dim=0)
    
    # 理化性质（多任务学习）
    if "particle_size" in batch[0]:
        result["particle_size"] = torch.stack([item["particle_size"] for item in batch])
    if "zeta_potential" in batch[0]:
        result["zeta_potential"] = torch.stack([item["zeta_potential"] for item in batch])
    if "pdi" in batch[0]:
        result["pdi"] = torch.stack([item["pdi"] for item in batch])
    
    # 处理主要组分的 edge_index（支持 batch 偏移）
    result["ionizable_edge_index"] = batch_edge_index(batch, "ionizable")
    result["ionizable_bond_features"] = batch_bond_features(batch, "ionizable")
    
    # 辅助组分的 edge_index
    if "helper_edge_index" in batch[0]:
        result["helper_edge_index"] = batch_edge_index(batch, "helper")
        result["helper_bond_features"] = batch_bond_features(batch, "helper")
    
    if "cholesterol_edge_index" in batch[0]:
        result["cholesterol_edge_index"] = batch_edge_index(batch, "cholesterol")
        result["cholesterol_bond_features"] = batch_bond_features(batch, "cholesterol")
    
    if "peg_edge_index" in batch[0]:
        result["peg_edge_index"] = batch_edge_index(batch, "peg")
        result["peg_bond_features"] = batch_bond_features(batch, "peg")
    
    # 可选字段
    if any("fingerprint_extra" in item for item in batch):
        result["fingerprint_extra"] = torch.stack([item["fingerprint_extra"] for item in batch if "fingerprint_extra" in item])
    
    if any("embedding" in item for item in batch):
        result["embedding"] = torch.stack([item["embedding"] for item in batch if "embedding" in item])
    
    if any("image" in item for item in batch):
        result["image"] = torch.stack([item["image"] for item in batch if "image" in item])
    
    if any("structure" in item for item in batch):
        result["structure"] = [item["structure"] for item in batch if "structure" in item]
    
    if any("parameter" in item for item in batch):
        result["parameter"] = [item["parameter"] for item in batch if "parameter" in item]
    
    return result


def create_dataloader(
    dataset_name: str,
    clean_dir: str,
    origin_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """创建数据加载器"""
    dataset = MultiModalLNPDataset(
        dataset_name=dataset_name,
        clean_dir=clean_dir,
        origin_dir=origin_dir,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试数据加载器
    print("测试多模态数据加载器...")
    
    datasets = ["agile", "lantern", "lipobart", "lnp_atlas", "lnp_ml", "m3_lipids", "phil_lnp", "transma"]
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"测试 {dataset_name} 数据集")
        print('='*60)
        
        try:
            project_root = Path(__file__).resolve().parents[2]
            clean_dir = project_root / "dataset" / "clean" / dataset_name
            origin_dir = project_root / "dataset" / "origin" / dataset_name
            
            dataset = MultiModalLNPDataset(
                dataset_name=dataset_name,
                clean_dir=clean_dir,
                origin_dir=origin_dir,
            )
            
            print(f"✓ 数据集大小：{len(dataset)}")
            
            if len(dataset) > 0:
                item = dataset[0]
                print(f"✓ 样本键：{list(item.keys())}")
                print(f"✓ SMILES: {item['smiles'][:50]}...")
                print(f"✓ 指纹维度：{item['fingerprint'].shape}")
                print(f"✓ 原子特征：{item['atom_features'].shape}")
                print(f"✓ 目标值：{item['target']}")
            
        except Exception as e:
            print(f"✗ 加载失败：{e}")
            import traceback
            traceback.print_exc()
