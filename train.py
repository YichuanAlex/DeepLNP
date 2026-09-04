#!/usr/bin/env python3
"""
DeepLNP 训练 Pipeline - 基于合并数据集
========================================

功能:
1. 加载合并后的多模态数据集 (merged_datasets)
2. 3D 构象感知编码器
3. 多模态融合优化器 (MoE)
4. 模型训练 (支持早停、混合精度)
5. WandB + TensorBoard 实时日志
6. 只保留最佳模型权重

创新点:
- 创新点 1: 3D 构象感知编码器
- 创新点 2: 多模态融合优化器 (MoE)
- 创新点 3: 生成式扩散模型 (预留接口)
- 创新点 4: Transfection Cliffs 可解释性 (预留接口)
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
import socket
import webbrowser
import math
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Any

# 导入 WandB（全局）
try:
    import wandb
    WANDB_AVAILABLE = True
    # 声明为全局变量
    globals()['wandb'] = wandb
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    print("⚠️  WandB 未安装，请运行：pip install wandb")

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.amp import autocast, GradScaler

# 可视化和日志
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from deeplnp.data.unified_dataset import (
    CONTEXT_DIM,
    FORMULATION_DIM,
    SPATIAL_DIM,
    TARGET_DIM,
    TASK_COLUMNS,
    create_unified_dataloaders,
)
from deeplnp.models.predictor import LNPPredictor


def project_path(path_value: Optional[str], default_relative: str) -> Path:
    """Resolve config paths relative to the project root."""
    raw = path_value or default_relative
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def setup_distributed() -> Dict[str, Any]:
    """Initialize torch.distributed when train.py is launched by torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    
    info = {
        "distributed": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main_process": rank == 0,
        "device": None,
        "backend": None,
    }
    
    if not distributed:
        return info
    
    if not torch.cuda.is_available():
        raise RuntimeError("多进程分布式训练需要 CUDA；当前环境没有可用 CUDA。")
    
    torch.cuda.set_device(local_rank)
    backend = "nccl" if dist.is_nccl_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    
    info["device"] = torch.device(f"cuda:{local_rank}")
    info["backend"] = backend
    return info


def cleanup_distributed(distributed_info: Dict[str, Any]) -> None:
    if distributed_info.get("distributed") and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(distributed_info: Optional[Dict[str, Any]]) -> bool:
    return not distributed_info or distributed_info.get("is_main_process", True)


class HardwareDetector:
    """硬件环境自动检测"""
    
    @staticmethod
    def get_gpu_memory_usage(gpu_id: int) -> Dict[str, float]:
        """获取指定GPU的显存使用情况"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader', '-i', str(gpu_id)],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                used, total = map(float, result.stdout.strip().split(','))
                return {
                    'used_mb': used,
                    'total_mb': total,
                    'free_mb': total - used,
                    'usage_percent': (used / total) * 100
                }
        except Exception:
            pass
        
        # 如果无法获取详细信息，返回默认值
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**2  # MB
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2  # MB
            return {
                'used_mb': allocated,
                'total_mb': total,
                'free_mb': total - allocated,
                'usage_percent': (allocated / total) * 100
            }
        return {'used_mb': 0, 'total_mb': 0, 'free_mb': 0, 'usage_percent': 0}
    
    @staticmethod
    def find_best_gpu(min_memory_gb: float = 2.0) -> int:
        """
        找到最适合训练的GPU
        策略：
        1. 优先选择空闲GPU
        2. 如果没有空闲GPU，选择显存最充足的GPU
        3. 即使GPU有任务在运行，只要能装下模型就使用
        """
        if not torch.cuda.is_available():
            return -1
        
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            return -1
        
        if gpu_count == 1:
            # 只有一张GPU，直接使用
            return 0
        
        best_gpu = -1
        best_free_memory = 0
        min_required_mb = min_memory_gb * 1024  # 转换为MB
        
        print(f"\n🔍 检测 {gpu_count} 张GPU的显存使用情况...")
        
        for gpu_id in range(gpu_count):
            mem_info = HardwareDetector.get_gpu_memory_usage(gpu_id)
            gpu_name = torch.cuda.get_device_name(gpu_id)
            
            print(f"  GPU {gpu_id}: {gpu_name}")
            print(f"    显存: {mem_info['used_mb']:.0f}MB / {mem_info['total_mb']:.0f}MB "
                  f"(空闲: {mem_info['free_mb']:.0f}MB, 使用率: {mem_info['usage_percent']:.1f}%)")
            
            # 检查是否有足够显存
            if mem_info['free_mb'] >= min_required_mb:
                if mem_info['free_mb'] > best_free_memory:
                    best_free_memory = mem_info['free_mb']
                    best_gpu = gpu_id
        
        if best_gpu >= 0:
            print(f"✅ 选择 GPU {best_gpu} (空闲显存: {best_free_memory:.0f}MB)")
        else:
            # 如果没有GPU有足够显存，选择显存最多的
            print(f"⚠️  没有GPU有 {min_memory_gb}GB 以上空闲显存，选择显存最多的GPU")
            for gpu_id in range(gpu_count):
                mem_info = HardwareDetector.get_gpu_memory_usage(gpu_id)
                if mem_info['free_mb'] > best_free_memory:
                    best_free_memory = mem_info['free_mb']
                    best_gpu = gpu_id
            
            if best_gpu >= 0:
                print(f"✅ 选择 GPU {best_gpu} (空闲显存: {best_free_memory:.0f}MB，可能不足)")
        
        return best_gpu
    
    @staticmethod
    def detect() -> Dict[str, Any]:
        """检测硬件配置并自动选择最佳GPU"""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_names": [],
            "gpu_memory": [],
            "gpu_free_memory": [],
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
            "cpu_count": os.cpu_count(),
            "device": None,
            "backend": None,
            "multi_gpu": False,
            "selected_gpu": -1,
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                info["gpu_names"].append(torch.cuda.get_device_name(i))
                info["gpu_memory"].append(
                    torch.cuda.get_device_properties(i).total_memory / 1024**3
                )
                # 获取空闲显存
                mem_info = HardwareDetector.get_gpu_memory_usage(i)
                info["gpu_free_memory"].append(mem_info['free_mb'] / 1024)  # 转换为GB
            
            # 智能选择最佳GPU
            best_gpu = HardwareDetector.find_best_gpu(min_memory_gb=2.0)
            info["selected_gpu"] = best_gpu
            
            if best_gpu >= 0:
                info["device"] = torch.device(f"cuda:{best_gpu}")
                torch.cuda.set_device(best_gpu)
                info["backend"] = "cuda"
                info["multi_gpu"] = info["gpu_count"] > 1
                
                if info["multi_gpu"]:
                    names = ", ".join(info["gpu_names"])
                    print(f"✅ 检测到 CUDA 多卡：{info['gpu_count']} 张 GPU ({names})")
                    print(f"✅ 选择 GPU {best_gpu}: {info['gpu_names'][best_gpu]} "
                          f"(空闲: {info['gpu_free_memory'][best_gpu]:.1f}GB / 总计: {info['gpu_memory'][best_gpu]:.1f}GB)")
                else:
                    print(f"✅ 使用 CUDA 单卡: {info['gpu_names'][0]} "
                          f"(空闲: {info['gpu_free_memory'][0]:.1f}GB / 总计: {info['gpu_memory'][0]:.1f}GB)")
            else:
                # 没有合适的GPU，使用CPU
                info["device"] = torch.device("cpu")
                info["backend"] = "cpu"
                print(f"⚠️  没有合适的GPU可用，使用 CPU ({info['cpu_count']} 核心)")
        elif info["mps_available"]:
            info["device"] = torch.device("mps")
            info["backend"] = "mps"
            print("✅ 使用 Apple Silicon MPS GPU")
        else:
            info["device"] = torch.device("cpu")
            info["backend"] = "cpu"
            print(f"⚠️  CUDA 不可用，使用 CPU ({info['cpu_count']} 核心)")
        
        return info


class Trainer:
    """训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        config: Dict,
        distributed_info: Optional[Dict[str, Any]] = None,
        dataset=None,  # 添加数据集参数以获取标准化参数
    ):
        self.device = device
        self.config = config
        self.distributed_info = distributed_info or {
            "distributed": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "is_main_process": True,
        }
        self.distributed = self.distributed_info.get("distributed", False)
        self.rank = self.distributed_info.get("rank", 0)
        self.local_rank = self.distributed_info.get("local_rank", 0)
        self.world_size = self.distributed_info.get("world_size", 1)
        self.is_main_process = self.distributed_info.get("is_main_process", True)
        model = model.to(device)
        self.raw_model = model
        if self.distributed:
            self.model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            if self.is_main_process:
                print(f"✅ 启用 DistributedDataParallel 多卡训练：{self.world_size} 张 GPU")
        elif device.type == "cuda" and torch.cuda.device_count() > 1:
            self.model = model
            print(
                "⚠️  检测到多张 CUDA GPU，但当前不是 torchrun 分布式进程；"
                "为避免图 edge_index 被 DataParallel 错误切分，本次使用 cuda:0 单卡。"
            )
        else:
            self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 保存标准化参数
        self.task_stats = getattr(dataset, 'task_stats', {}) if dataset else {}
        efficiency_stats = self.task_stats.get('efficiency', {'mean': 0.0, 'std': 1.0})
        self.target_mean = efficiency_stats['mean']
        self.target_std = efficiency_stats['std']
        
        # 训练参数
        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training'].get('weight_decay', 1e-4)
        self.use_amp = config['training'].get('use_amp', True)
        self.gradient_clip = config['training'].get('gradient_clip', None)
        
        # 损失函数 - 多任务学习
        self.num_tasks = config['model'].get('num_tasks', 4)
        if self.num_tasks > 1:
            # 多任务学习：使用任务不确定性加权 (Kendall et al., 2018)
            self.task_log_vars = nn.Parameter(torch.zeros(self.num_tasks, device=self.device))
            self.criterion = nn.MSELoss(reduction='none')  # 每个样本单独计算 loss
        else:
            self.task_log_vars = None
            self.criterion = nn.MSELoss()
        
        # 优化器：包含多任务不确定性权重
        self.optim_params = list(self.model.parameters())
        if self.task_log_vars is not None:
            self.optim_params.append(self.task_log_vars)
        self.optimizer = optim.AdamW(
            self.optim_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        lr_scheduler_type = self.config.get('training', {}).get('lr_scheduler', 'reduce_on_plateau')
        if lr_scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=self.config.get('training', {}).get('lr_min', 1e-6),
            )
            self.scheduler_type = 'cosine'
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
            )
            self.scheduler_type = 'reduce_on_plateau'
        
        # 学习率预热
        self.warmup_epochs = self.config.get('training', {}).get('warmup_epochs', 0)
        
        # 混合精度只在 CUDA 上启用；MPS/CPU 保持全精度。
        self.use_amp_runtime = self.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp_runtime else None
        
        # 多任务权重（根据文献和经验设置）
        self.task_weights = {
            'efficiency': 1.0,      # 转染效率（主要任务）
            'particle_size': 0.5,   # 粒径（辅助任务）
            'zeta_potential': 0.3,  # Zeta 电位（辅助任务）
            'pdi': 0.3,
            'encapsulation': 0.4,
            'toxicity': 0.7,        # 有真实标签时自动启用
        }
        self.ranking_weight = config['training'].get('ranking_weight', 0.5)
        self.ranking_margin = config['training'].get('ranking_margin', 0.1)
        self.target_aux_weight = config['training'].get('target_aux_weight', 0.2)
        
        # 日志
        self.log_dir = project_path(config.get('logging', {}).get('log_dir'), 'logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成超参数组合字符串用于日志命名
        hp_params = {
            'lr': self.config['training']['learning_rate'],
            'bs': self.config['training']['batch_size'],
            'wd': self.config['training']['weight_decay'],
            'mol_dim': self.config['model']['mol_feat_dim'],
            'gnn_layers': self.config['model']['num_gnn_layers'],
            'fusion_dim': self.config['model']['fusion_hidden_dim'],
            'heads': self.config['model']['fusion_num_heads'],
            'dropout': self.config['model']['fusion_dropout'],
            'experts': self.config['model']['num_experts'],
        }
        hp_str = "_".join([f"{k}-{v}" for k, v in hp_params.items()])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # TensorBoard
        self.tb_writer = None
        if self.is_main_process and TENSORBOARD_AVAILABLE and config['logging'].get('use_tensorboard', True):
            # 使用超参数组合命名 TensorBoard 日志目录
            tb_base_dir = project_path(config.get('logging', {}).get('tensorboard_dir'), 'tensorboard_logs')
            tb_base_dir.mkdir(parents=True, exist_ok=True)
            tb_dir = tb_base_dir / f"runs_{hp_str}_{timestamp}"
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"✅ TensorBoard 日志目录：{tb_dir}")
            
            # 自动启动 TensorBoard 并打开网页
            import subprocess
            import time
            import webbrowser
            import socket
            
            # 检查端口是否被占用
            def is_port_in_use(port):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    return s.connect_ex(('localhost', port)) == 0
            
            # 终止旧的 TensorBoard 进程
            subprocess.run(['pkill', '-f', 'tensorboard'], capture_output=True)
            time.sleep(2)
            
            # 如果端口仍被占用，强制释放
            if is_port_in_use(6006):
                try:
                    subprocess.run(['fuser', '-k', '6006/tcp'], capture_output=True)
                    time.sleep(1)
                except:
                    pass
            
            # 启动新的 TensorBoard（后台运行）
            tb_process = subprocess.Popen(
                ['tensorboard', '--logdir', str(tb_dir), '--host', '0.0.0.0', '--port', '6006', '--reload_interval', '5'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # 后台运行
            )
            print(f"🚀 TensorBoard 已启动 (PID: {tb_process.pid})")
            
            # 等待 TensorBoard 启动并检查端口
            print(f"⏳ 等待 TensorBoard 服务就绪...")
            for i in range(10):  # 最多等待 10 秒
                time.sleep(1)
                if is_port_in_use(6006):
                    print(f"✅ TensorBoard 服务已就绪")
                    break
                else:
                    print(f"  等待中... ({i+1}/10)")
            
            # 自动打开网页
            try:
                webbrowser.open('http://localhost:6006')
                print(f"🌐 TensorBoard 网页已打开：http://localhost:6006")
                print(f"   如果网页无法加载，请手动刷新或访问：http://127.0.0.1:6006")
            except:
                print(f"⚠️  无法自动打开浏览器，请手动访问：http://localhost:6006")
            
            print(f"\n💡 TensorBoard 提示:")
            print(f"   - 如果网页一直加载，请尝试刷新页面")
            print(f"   - 或者使用：http://127.0.0.1:6006")
            print(f"   - 查看日志：tensorboard --logdir {tb_dir}")
        
        # WandB
        if self.is_main_process and WANDB_AVAILABLE and config['logging'].get('use_wandb', True):
            wandb_api_key = os.environ.get("WANDB_API_KEY")
            if wandb_api_key:
                print(f"\n🔑 正在登录 WandB...")
                try:
                    wandb.login(key=wandb_api_key, relogin=True, timeout=30)
                    print("✅ WandB 登录成功")
                except Exception as e:
                    print(f"⚠️  WandB 登录失败：{e}")
            else:
                print("⚠️  未设置 WANDB_API_KEY，跳过显式登录")
            
            # 初始化 WandB（使用超参数组合命名）
            # 注意：WandB 会自动创建 wandb/run-日期-ID 目录，我们无法完全控制
            # 但可以通过设置 name 和 project 来组织实验
            wandb_run = wandb.init(
                project=config['logging'].get('wandb_project', 'DeepLNP'),
                entity=config['logging'].get('wandb_entity'),
                config=config,
                name=f"runs_{hp_str}_{timestamp}",  # 使用 runs_前缀，与 TensorBoard 一致
                mode=config['logging'].get('wandb_mode', 'online'),
                force=True,  # 强制创建新的 run
                dir=str(project_path(config.get('logging', {}).get('wandb_dir'), "wandb_logs")),
            )
            
            wandb_url = wandb.run.get_url() if wandb.run else "https://wandb.ai/jiangzixi1527435659-anhui-university/DeepLNP"
            print(f"\n✅ WandB 已初始化")
            print(f"🌐 WandB 访问地址：{wandb_url}")
            print(f"   Entity: jiangzixi1527435659-anhui-university")
            print(f"   Project: DeepLNP")
            print(f"   Run: {wandb.run.name if wandb.run else 'N/A'}")
            
            # 等待 WandB 初始化完成并上传初始数据
            print(f"⏳ 等待 WandB 数据上传...")
            time.sleep(5)
            
            # 自动打开 WandB 网页
            try:
                # 尝试多次打开以确保成功
                for i in range(3):
                    result = webbrowser.open(wandb_url)
                    if result:
                        print(f"🌐 WandB 网页已打开 (尝试 {i+1}/3)")
                        break
                    time.sleep(1)
                else:
                    print(f"⚠️  浏览器打开失败，请手动访问")
            except Exception as e:
                print(f"⚠️  无法自动打开浏览器：{e}")
            
            print(f"\n💡 WandB 使用提示:")
            print(f"   - 首次访问可能需要等待数据上传（10-20 秒）")
            print(f"   - 如果页面转圈，请刷新页面")
            print(f"   - 所有训练数据将实时同步到 WandB")
            print(f"   - 访问：{wandb_url}")
            print(f"   - 如果链接打不开，请复制 URL 到浏览器")
        
        # 检查点
        self.checkpoint_dir = project_path(config.get('checkpoint', {}).get('save_dir'), 'checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 早停机制
        early_stopping_config = config['training'].get('early_stopping', {})
        self.early_stopping_enabled = early_stopping_config.get('enabled', True)
        self.early_stopping_patience = early_stopping_config.get('patience', 50)
        self.early_stopping_min_delta = early_stopping_config.get('min_delta', 0.001)
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')  # 只初始化一次
        self.early_stop = False
        
        # 日志配置
        self.print_frequency = config['logging'].get('print_frequency', 10)
        
        # 设置日志文件目录
        self.log_dir = project_path(config.get('logging', {}).get('log_dir'), 'logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成超参数文件名
        hp_params = {
            'lr': self.config['training']['learning_rate'],
            'bs': self.config['training']['batch_size'],
            'wd': self.config['training']['weight_decay'],
            'mol_dim': self.config['model']['mol_feat_dim'],
            'gnn_layers': self.config['model']['num_gnn_layers'],
            'fusion_dim': self.config['model']['fusion_hidden_dim'],
            'heads': self.config['model']['fusion_num_heads'],
            'dropout': self.config['model']['fusion_dropout'],
            'experts': self.config['model']['num_experts'],
        }
        hp_str = "_".join([f"{k}-{v}" for k, v in hp_params.items()])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 设置日志文件（使用超参数组合命名）
        log_file = self.log_dir / f"training_{hp_str}_{timestamp}.log"
        if self.is_main_process:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"日志文件：{log_file}")
            self.logger.info(f"超参数组合：{hp_str}")
        else:
            self.logger = logging.getLogger(f"{__name__}.rank{self.rank}")
            self.logger.addHandler(logging.NullHandler())
            self.logger.propagate = False
    
    def _to_device(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device, non_blocking=True)
        return value
    
    def _pick_tensor(self, batch: Dict[str, Any], *keys: str):
        for key in keys:
            if key in batch and batch[key] is not None:
                return self._to_device(batch[key])
        return None
    
    def _build_model_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch_size = batch['target'].size(0)
        molar_ratios = self._pick_tensor(batch, 'molar_ratios')
        if molar_ratios is None:
            molar_ratios = torch.zeros(batch_size, 4, device=self.device)
        
        return {
            'ionizable_atom_features': self._pick_tensor(batch, 'ionizable_atom_features', 'atom_features'),
            'ionizable_edge_index': self._pick_tensor(batch, 'ionizable_edge_index', 'edge_index'),
            'ionizable_bond_features': self._pick_tensor(batch, 'ionizable_bond_features', 'bond_features'),
            'ionizable_batch': self._pick_tensor(batch, 'ionizable_batch', 'batch'),
            'helper_atom_features': self._pick_tensor(batch, 'helper_atom_features'),
            'helper_edge_index': self._pick_tensor(batch, 'helper_edge_index'),
            'helper_bond_features': self._pick_tensor(batch, 'helper_bond_features'),
            'helper_batch': self._pick_tensor(batch, 'helper_batch'),
            'cholesterol_atom_features': self._pick_tensor(batch, 'cholesterol_atom_features'),
            'cholesterol_edge_index': self._pick_tensor(batch, 'cholesterol_edge_index'),
            'cholesterol_bond_features': self._pick_tensor(batch, 'cholesterol_bond_features'),
            'cholesterol_batch': self._pick_tensor(batch, 'cholesterol_batch'),
            'peg_atom_features': self._pick_tensor(batch, 'peg_atom_features'),
            'peg_edge_index': self._pick_tensor(batch, 'peg_edge_index'),
            'peg_bond_features': self._pick_tensor(batch, 'peg_bond_features'),
            'peg_batch': self._pick_tensor(batch, 'peg_batch'),
            'ionizable_spatial_features': self._pick_tensor(batch, 'ionizable_spatial_features'),
            'helper_spatial_features': self._pick_tensor(batch, 'helper_spatial_features'),
            'cholesterol_spatial_features': self._pick_tensor(batch, 'cholesterol_spatial_features'),
            'peg_spatial_features': self._pick_tensor(batch, 'peg_spatial_features'),
            'ionizable_fingerprint': self._pick_tensor(batch, 'ionizable_fingerprint'),
            'helper_fingerprint': self._pick_tensor(batch, 'helper_fingerprint'),
            'cholesterol_fingerprint': self._pick_tensor(batch, 'cholesterol_fingerprint'),
            'peg_fingerprint': self._pick_tensor(batch, 'peg_fingerprint'),
            'molar_ratios': molar_ratios,
            'images': self._pick_tensor(batch, 'images', 'image'),
            'embeddings': self._pick_tensor(batch, 'embeddings', 'embedding', 'ionizable_fingerprint', 'fingerprint'),
            'formulation_features': self._pick_tensor(batch, 'formulation_features'),
            'context_features': self._pick_tensor(batch, 'context_features'),
            'target_features': self._pick_tensor(batch, 'target_features'),
            'component_structure_mask': self._pick_tensor(batch, 'component_structure_mask'),
            'component_active_mask': self._pick_tensor(batch, 'component_active_mask'),
            'physchem_features': self._pick_tensor(batch, 'physchem_features'),
        }
    
    def _prediction_dict(self, model_output: Any) -> Dict[str, torch.Tensor]:
        if isinstance(model_output, dict) and 'predictions' in model_output:
            return model_output['predictions']
        if isinstance(model_output, dict):
            return model_output
        return {'efficiency': model_output}
    
    def _task_target_and_mask(self, batch: Dict[str, Any], task_name: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if task_name == 'efficiency':
            target = batch['target']
            mask = batch.get('efficiency_mask')
        else:
            target = batch.get(task_name)
            mask = batch.get(f'{task_name}_mask')
        
        if target is None:
            return None, None
        
        target = self._to_device(target).float()
        mask = self._to_device(mask).bool() if isinstance(mask, torch.Tensor) else None
        return target, mask
    
    def _compute_loss(self, model_output: Any, batch: Dict[str, Any]) -> torch.Tensor:
        predictions_dict = self._prediction_dict(model_output)
        loss_terms = []
        
        for task_idx, (task_name, weight) in enumerate(self.task_weights.items()):
            if task_name not in predictions_dict:
                continue
            
            target, mask = self._task_target_and_mask(batch, task_name)
            if target is None:
                continue
            
            pred = predictions_dict[task_name]
            if pred.dim() > 1:
                pred = pred.squeeze(-1)
            pred = pred.float()
            censor = self._pick_tensor(batch, 'toxicity_censor') if task_name == 'toxicity' else None
            
            if mask is not None:
                mask = mask.reshape(-1)
                if mask.numel() == pred.numel():
                    if mask.sum().item() == 0:
                        continue
                    pred = pred[mask]
                    target = target.reshape(-1)[mask]
                    if censor is not None:
                        censor = censor.reshape(-1)[mask]
            
            task_loss = (pred - target.reshape_as(pred)).pow(2)
            predicted_log_var = predictions_dict.get(f'{task_name}_log_variance')
            if predicted_log_var is not None:
                predicted_log_var = predicted_log_var.reshape(-1)
                if mask is not None and predicted_log_var.numel() == mask.numel():
                    predicted_log_var = predicted_log_var[mask]
                predicted_log_var = predicted_log_var.reshape_as(task_loss)
                if task_name == 'toxicity' and censor is not None:
                    sigma = torch.exp(0.5 * predicted_log_var).clamp(min=1e-4)
                    z_upper = (target.reshape_as(pred) - pred) / sigma
                    z_lower = -z_upper
                    cdf_upper = torch.special.ndtr(z_upper).clamp(min=1e-7)
                    cdf_lower = torch.special.ndtr(z_lower).clamp(min=1e-7)
                    exact = 0.5 * torch.exp(-predicted_log_var) * task_loss + 0.5 * predicted_log_var
                    task_loss = torch.where(censor.lt(0), -cdf_upper.log(), exact)
                    task_loss = torch.where(censor.gt(0), -cdf_lower.log(), task_loss)
                else:
                    task_loss = 0.5 * torch.exp(-predicted_log_var) * task_loss + 0.5 * predicted_log_var
            task_loss = task_loss.mean()

            # COMET-inspired pairwise ranking, restricted to comparable rows
            # from the same experimental screen. Cross-screen absolute scales
            # are never compared.
            if task_name == 'efficiency' and self.ranking_weight > 0:
                # When present, the rank adapter owns the pairwise objective.
                # The efficacy point head remains dedicated to calibrated
                # regression, avoiding conflicting ranking gradients.
                full_pred = predictions_dict.get(
                    'efficiency_rank_score', predictions_dict[task_name]
                ).reshape(-1).float()
                full_target, full_mask = self._task_target_and_mask(batch, task_name)
                groups = self._pick_tensor(batch, 'group_id')
                if groups is not None and full_target is not None:
                    full_target = full_target.reshape(-1)
                    valid = full_mask.reshape(-1) if full_mask is not None else torch.ones_like(full_target, dtype=torch.bool)
                    target_delta = full_target[:, None] - full_target[None, :]
                    comparable = (
                        valid[:, None] & valid[None, :]
                        & groups[:, None].eq(groups[None, :])
                        & torch.triu(torch.ones_like(target_delta, dtype=torch.bool), diagonal=1)
                        & target_delta.abs().gt(self.ranking_margin)
                    )
                    if comparable.any():
                        pred_delta = full_pred[:, None] - full_pred[None, :]
                        signed_margin = target_delta.sign() * pred_delta
                        ranking_loss = torch.nn.functional.softplus(-signed_margin[comparable]).mean()
                        task_loss = task_loss + self.ranking_weight * ranking_loss
            if self.task_log_vars is not None and task_idx < self.task_log_vars.numel():
                log_var = self.task_log_vars[task_idx]
                task_loss = 0.5 * torch.exp(-log_var) * task_loss + 0.5 * log_var
            
            loss_terms.append(weight * task_loss)
        
        target_logits = predictions_dict.get('target_logits')
        target_class = self._pick_tensor(batch, 'target_class')
        target_mask = self._pick_tensor(batch, 'target_class_mask')
        if target_logits is not None and target_class is not None and target_mask is not None:
            target_mask = target_mask.bool().reshape(-1)
            if target_mask.any():
                loss_terms.append(
                    self.target_aux_weight * torch.nn.functional.cross_entropy(
                        target_logits[target_mask], target_class.long().reshape(-1)[target_mask]
                    )
                )

        if not loss_terms:
            # Fallback: use efficiency prediction only
            eff_pred = predictions_dict.get('efficiency')
            if eff_pred is not None:
                eff_target = batch.get('target', batch.get('efficiency'))
                if eff_target is not None:
                    eff_pred = eff_pred.squeeze(-1).float()
                    eff_target = eff_target.float()
                    return (eff_pred - eff_target).pow(2).mean()
            raise ValueError("No valid task loss could be computed for this batch")
        
        return torch.stack(loss_terms).sum()
    
    def _efficiency_predictions(self, model_output: Any) -> torch.Tensor:
        predictions_dict = self._prediction_dict(model_output)
        pred = predictions_dict.get('efficiency')
        if pred is None:
            pred = next(iter(predictions_dict.values()))
        if pred.dim() > 1:
            pred = pred.squeeze(-1)
        return pred
    
    def _unwrap_model(self) -> nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model
    
    def _model_state_dict(self) -> Dict[str, torch.Tensor]:
        return self._unwrap_model().state_dict()
    
    def _sync_external_grads(self) -> None:
        """Synchronize trainable parameters that are outside the DDP-wrapped module."""
        if (
            self.distributed
            and self.task_log_vars is not None
            and self.task_log_vars.grad is not None
            and dist.is_initialized()
        ):
            dist.all_reduce(self.task_log_vars.grad, op=dist.ReduceOp.SUM)
            self.task_log_vars.grad.div_(self.world_size)
    
    def _reduce_train_loss(self, total_loss: float, num_batches: int) -> float:
        if not self.distributed:
            return total_loss / max(num_batches, 1)
        values = torch.tensor([total_loss, float(num_batches)], dtype=torch.float64, device=self.device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        return (values[0] / values[1].clamp_min(1.0)).item()
    
    def _broadcast_float(self, value: float) -> float:
        if not self.distributed:
            return value
        tensor = torch.tensor([value if self.is_main_process else 0.0], dtype=torch.float64, device=self.device)
        dist.broadcast(tensor, src=0)
        return tensor.item()
    
    def _broadcast_metrics(self, metrics: Optional[Dict[str, float]]) -> Dict[str, float]:
        ordered_keys = ['loss', 'mse', 'mae', 'rmse', 'r2']
        if self.distributed:
            values = [
                float(metrics.get(key, 0.0)) if self.is_main_process and metrics is not None else 0.0
                for key in ordered_keys
            ]
            tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
            dist.broadcast(tensor, src=0)
            return {key: tensor[i].item() for i, key in enumerate(ordered_keys)}
        return metrics or {key: 0.0 for key in ordered_keys}
    
    def _autocast_context(self):
        if self.scaler is not None:
            return autocast('cuda', enabled=True)
        return nullcontext()
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个 epoch - 支持四组分配方"""
        if isinstance(getattr(self.train_loader, "sampler", None), DistributedSampler):
            self.train_loader.sampler.set_epoch(epoch)
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 学习率预热
        if epoch < self.warmup_epochs and self.warmup_epochs > 0:
            warmup_ratio = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate * warmup_ratio
        
        # 使用数据预取器提升 GPU 利用率
        data_iter = iter(self.train_loader)
        
        for batch_idx in range(len(self.train_loader)):
            self.optimizer.zero_grad(set_to_none=True)
            
            # 获取当前 batch
            batch = next(data_iter)
            
            with self._autocast_context():
                model_output = self.model(**self._build_model_inputs(batch))
                loss = self._compute_loss(model_output, batch)
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self._sync_external_grads()
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(self.optim_params, self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self._sync_external_grads()
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(self.optim_params, self.gradient_clip)
                self.optimizer.step()
            
            total_loss += loss.detach().item()
            num_batches += 1
            
            if self.is_main_process and (batch_idx + 1) % self.print_frequency == 0:
                avg_loss = total_loss / num_batches
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"Epoch {epoch}, Batch {batch_idx+1}/{len(self.train_loader)}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")
        
        return self._reduce_train_loss(total_loss, num_batches)
    
    @torch.no_grad()
    def validate(self) -> float:
        """验证"""
        if self.distributed and not self.is_main_process:
            return self._broadcast_float(0.0)
        
        eval_model = self._unwrap_model()
        eval_model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            with self._autocast_context():
                model_output = eval_model(**self._build_model_inputs(batch))
                loss = self._compute_loss(model_output, batch)
            total_loss += loss.detach().item()
            num_batches += 1
        
        return self._broadcast_float(total_loss / max(num_batches, 1))
    
    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        """测试"""
        if self.distributed and not self.is_main_process:
            return self._broadcast_metrics(None)
        
        eval_model = self._unwrap_model()
        eval_model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        for batch in self.test_loader:
            with self._autocast_context():
                model_output = eval_model(**self._build_model_inputs(batch))
                loss = self._compute_loss(model_output, batch)
            
            total_loss += loss.detach().item()
            num_batches += 1
            
            eff_pred = self._efficiency_predictions(model_output)
            # Atlas rows without an efficiency endpoint carry a placeholder
            # tensor value plus an explicit mask.  Never let those placeholders
            # contaminate held-out metrics.
            eff_mask = batch.get('efficiency_mask')
            if isinstance(eff_mask, torch.Tensor):
                eff_mask = eff_mask.reshape(-1).bool().to(eff_pred.device)
                eff_pred = eff_pred.reshape(-1)[eff_mask]
                eff_target = batch['target'].to(eff_pred.device).reshape(-1)[eff_mask]
            else:
                eff_target = batch['target'].to(eff_pred.device).reshape(-1)
            all_predictions.extend(eff_pred.detach().cpu().numpy())
            all_targets.extend(eff_target.detach().cpu().numpy())
        
        predictions_np = np.array(all_predictions)
        targets_np = np.array(all_targets)
        if targets_np.size == 0:
            return self._broadcast_metrics({
                'loss': total_loss / max(num_batches, 1),
                'mse': float('nan'), 'mae': float('nan'),
                'rmse': float('nan'), 'r2': float('nan'),
            })
        mse = np.mean((predictions_np - targets_np) ** 2)
        mae = np.mean(np.abs(predictions_np - targets_np))
        rmse = np.sqrt(mse)
        ss_res = np.sum((targets_np - predictions_np) ** 2)
        ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        metrics = {
            'loss': total_loss / max(num_batches, 1),
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
        }
        return self._broadcast_metrics(metrics)
    
    def train(self):
        """完整训练流程"""
        self.logger.info("=" * 80)
        self.logger.info("开始训练 DeepLNP 模型")
        self.logger.info("=" * 80)
        
        # 打印模型架构
        self.logger.info(f"\n模型架构:\n{self.model}")
        
        # 参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"\n总参数量：{total_params:,}")
        self.logger.info(f"可训练参数量：{trainable_params:,}")
        
        # 训练循环
        for epoch in range(1, self.num_epochs + 1):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Epoch {epoch}/{self.num_epochs}")
            self.logger.info(f"{'='*80}")
            
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate()
            
            # Keep the held-out test set sealed during model selection.
            if self.config['training'].get('evaluate_test_each_epoch', False):
                test_metrics = self.test()
            else:
                test_metrics = {key: float('nan') for key in ['loss', 'mse', 'mae', 'rmse', 'r2']}
            
            epoch_time = time.time() - start_time
            
            # 日志输出
            self.logger.info(f"Train Loss: {train_loss:.6f}")
            self.logger.info(f"Val Loss: {val_loss:.6f}")
            self.logger.info(f"Test Loss: {test_metrics['loss']:.6f}")
            self.logger.info(f"Test MSE: {test_metrics['mse']:.6f}")
            self.logger.info(f"Test MAE: {test_metrics['mae']:.6f}")
            self.logger.info(f"Test RMSE: {test_metrics['rmse']:.6f}")
            self.logger.info(f"Test R²: {test_metrics['r2']:.6f}")
            self.logger.info(f"Epoch Time: {epoch_time:.2f}s")
            
            # TensorBoard
            if self.tb_writer:
                # 写入标量数据
                self.tb_writer.add_scalar('Loss/Train', train_loss, epoch)
                self.tb_writer.add_scalar('Loss/Val', val_loss, epoch)
                self.tb_writer.add_scalar('Loss/Test', test_metrics['loss'], epoch)
                self.tb_writer.add_scalar('Metrics/Test_MSE', test_metrics['mse'], epoch)
                self.tb_writer.add_scalar('Metrics/Test_MAE', test_metrics['mae'], epoch)
                self.tb_writer.add_scalar('Metrics/Test_RMSE', test_metrics['rmse'], epoch)
                self.tb_writer.add_scalar('Metrics/Test_R2', test_metrics['r2'], epoch)
                self.tb_writer.add_scalar('Time/Epoch', epoch_time, epoch)
                
                # 强制刷新到磁盘
                self.tb_writer.flush()
                print(f"  📊 TensorBoard 数据已写入 (epoch {epoch})")
            
            # WandB
            if self.is_main_process and WANDB_AVAILABLE and globals().get('wandb') is not None and globals()['wandb'].run:
                globals()['wandb'].log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'test_loss': test_metrics['loss'],
                    'test_mse': test_metrics['mse'],
                    'test_mae': test_metrics['mae'],
                    'test_rmse': test_metrics['rmse'],
                    'test_r2': test_metrics['r2'],
                    'epoch_time': epoch_time,
                })
            
            # 学习率调度
            if self.scheduler_type == 'cosine':
                self.scheduler.step()
            else:
                self.scheduler.step(val_loss)
            
            # 保存最佳模型（使用超参数命名）
            previous_best_val_loss = self.best_val_loss
            improved_for_save = val_loss < previous_best_val_loss
            improved_for_early_stop = val_loss < previous_best_val_loss - self.early_stopping_min_delta
            
            if improved_for_save:
                self.best_val_loss = val_loss
                
                # 生成超参数文件名
                hp_params = {
                    'lr': self.config['training']['learning_rate'],
                    'bs': self.config['training']['batch_size'],
                    'wd': self.config['training']['weight_decay'],
                    'mol_dim': self.config['model']['mol_feat_dim'],
                    'gnn_layers': self.config['model']['num_gnn_layers'],
                    'fusion_dim': self.config['model']['fusion_hidden_dim'],
                    'heads': self.config['model']['fusion_num_heads'],
                    'dropout': self.config['model']['fusion_dropout'],
                    'experts': self.config['model']['num_experts'],
                }
                
                hp_str = "_".join([f"{k}-{v}" for k, v in hp_params.items()])
                checkpoint_filename = f"best_model_{hp_str}.pth"
                checkpoint_path = self.checkpoint_dir / checkpoint_filename
                
                if self.is_main_process:
                    # 保存模型
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self._model_state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'test_metrics': test_metrics,
                        'config': self.config,
                        'hyperparameters': hp_params,
                        'target_mean': self.target_mean,
                        'target_std': self.target_std,
                        'task_stats': self.task_stats,
                    }, checkpoint_path)
                    
                    # 保存为 latest_best_超参数.pth（带超参数信息）
                    latest_filename = f"latest_best_{hp_str}.pth"
                    latest_path = self.checkpoint_dir / latest_filename
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self._model_state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'test_metrics': test_metrics,
                        'config': self.config,
                        'hyperparameters': hp_params,
                        'checkpoint_filename': checkpoint_filename,
                        'target_mean': self.target_mean,
                        'target_std': self.target_std,
                        'task_stats': self.task_stats,
                    }, latest_path)
                    
                    self.logger.info(f"✅ 保存当前超参数最佳模型：{checkpoint_filename}")
                    self.logger.info(f"   超参数：{hp_str}")
                    self.logger.info(f"   验证损失：{val_loss:.6f}")
            
            # 早停检查（带 min_delta）
            if self.early_stopping_enabled:
                if improved_for_early_stop:
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        if self.is_main_process:
                            self.logger.info(f"\n⚠️  早停触发 (patience={self.early_stopping_patience})")
                        self.early_stop = True
                        break
        
        # 训练完成
        self.logger.info("\n" + "=" * 80)
        self.logger.info("训练完成!")
        self.logger.info("=" * 80)
        
        # 关闭 TensorBoard
        if self.tb_writer:
            self.tb_writer.close()
        
        # 关闭 WandB
        if WANDB_AVAILABLE and globals().get('wandb') is not None and globals()['wandb'].run:
            globals()['wandb'].finish()
        
        if not self.is_main_process:
            if self.distributed:
                dist.barrier()
            return
        
        # 打印结果
        self.logger.info(f"\n最佳验证损失：{self.best_val_loss:.6f}")
        self.logger.info(f"模型保存位置：{self.checkpoint_dir / 'latest_best_*.pth'}")
        final_test_metrics = self.test()
        self.logger.info(f"封存测试集结果：{final_test_metrics}")
        
        # Rank sealed-test observations for diagnostics. This is not generation.
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🔍 排序 Top 20 封存测试观测（不是新候选）...")
        self.logger.info("=" * 80)
        
        try:
            # 导入评估函数
            from evaluate import evaluate_and_select_candidates
            
            # 加载最佳模型（使用最新的 latest_best_超参数.pth）
            import glob
            latest_files = list(self.checkpoint_dir.glob("latest_best_*.pth"))
            if latest_files:
                # 按修改时间排序，取最新的
                latest_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                checkpoint_path = latest_files[0]
                self.logger.info(f"✅ 加载最佳模型：{checkpoint_path}")
            else:
                self.logger.warning(f"⚠️  未找到 latest_best_*.pth 文件")
                checkpoint_path = None
            
            if checkpoint_path and checkpoint_path.exists():
                
                # 评估并选择候选分子
                top_candidates = evaluate_and_select_candidates(
                    model=self._unwrap_model(),
                    test_loader=self.test_loader,
                    device=self.device,
                    top_k=20,
                    config=self.config,
                )
                
                # Output ranked held-out observations.
                self.logger.info("\n" + "=" * 80)
                self.logger.info("📊 Top 20 封存测试观测（诊断用途）")
                self.logger.info("=" * 80)
                
                for i, candidate in enumerate(top_candidates, 1):
                    self.logger.info(f"\n#{i}:")
                    self.logger.info(f"  预测效率：{candidate.get('predicted_efficiency', 0):.4f}")
                    self.logger.info(f"  实际效率：{candidate.get('actual_efficiency', 0):.4f}")
                    
                    # 完整 LNP 配方信息
                    ionizable_smiles = candidate.get('ionizable_lipid_smiles', candidate.get('smiles', 'N/A'))
                    if ionizable_smiles and ionizable_smiles != 'N/A':
                        # 截断过长的 SMILES
                        if len(ionizable_smiles) > 80:
                            ionizable_smiles = ionizable_smiles[:80] + "..."
                        self.logger.info(f"  可电离脂质 SMILES: {ionizable_smiles}")
                    
                    helper_lipid = candidate.get('helper_lipid', '')
                    if helper_lipid:
                        self.logger.info(f"  辅助脂质：{helper_lipid}")
                    
                    cholesterol = candidate.get('cholesterol', '')
                    if cholesterol:
                        self.logger.info(f"  胆固醇：{cholesterol}")
                    
                    peg_lipid = candidate.get('peg_lipid', '')
                    if peg_lipid:
                        self.logger.info(f"  PEG 脂质：{peg_lipid}")
                    
                    # 摩尔比信息
                    molar_ratios = candidate.get('molar_ratios')
                    if molar_ratios is not None:
                        ratio_str = f"ionizable:{molar_ratios.get('ionizable', 0):.1f} : helper:{molar_ratios.get('helper', 0):.1f} : cholesterol:{molar_ratios.get('cholesterol', 0):.1f} : peg:{molar_ratios.get('peg', 0):.1f}"
                        self.logger.info(f"  摩尔比：{ratio_str}")
                    
                    self.logger.info(f"  综合得分：{candidate.get('score', candidate.get('predicted_efficiency', 0)):.4f}")
                
                # 保存到 CSV（统一保存到 evaluation_results 目录）
                import pandas as pd
                eval_results_dir = project_path(self.config.get('evaluation', {}).get('result_dir'), "evaluation_results")
                eval_results_dir.mkdir(parents=True, exist_ok=True)
                csv_path = eval_results_dir / "top20_heldout_predictions.csv"
                df = pd.DataFrame(top_candidates)
                df.to_csv(csv_path, index=False, encoding='utf-8')
                self.logger.info(f"\n✅ Top 20 封存测试观测已保存到：{csv_path}")
                
                # 记录到 WandB
                if WANDB_AVAILABLE and globals().get('wandb') is not None and globals()['wandb'].run:
                    wb = globals()['wandb']
                    wb.log({"top_candidates": wb.Table(dataframe=df)})
                    wb.save(str(csv_path))
            else:
                self.logger.info(f"⚠️  模型文件不存在：{checkpoint_path}")
        except Exception as e:
            self.logger.error(f"❌ 评估过程出错：{e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        if TENSORBOARD_AVAILABLE:
            self.logger.info(f"\n查看 TensorBoard:")
            self.logger.info(f"  http://localhost:6006")
        
        if WANDB_AVAILABLE:
            self.logger.info(f"\n查看 WandB:")
            self.logger.info(f"  https://wandb.ai/DeepLNP")
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("✅ 所有任务完成！")
        self.logger.info("=" * 80)
        
        if self.distributed:
            dist.barrier()


def load_config(config_path: str) -> Dict:
    """加载 YAML 配置"""
    if not YAML_AVAILABLE:
        print("⚠️  YAML 不可用，使用默认配置")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    """主函数"""
    distributed_info = setup_distributed()
    
    parser = argparse.ArgumentParser(description="DeepLNP 训练脚本")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='配置文件路径',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='训练 epoch 数 (覆盖配置文件)',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='batch size (覆盖配置文件)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='学习率 (覆盖配置文件)',
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='指定使用的 GPU 编号 (0, 1, 2, ...), -1 表示使用 CPU',
    )
    
    args = parser.parse_args()
    
    main_process = is_main_process(distributed_info)
    
    # 如果指定了 GPU，设置 CUDA_VISIBLE_DEVICES
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        if main_process:
            print(f"🎯 指定使用 GPU {args.gpu}")
    
    # 加载配置
    if main_process:
        print("📄 加载配置文件...")
    config = load_config(str(project_path(args.config, 'configs/train_config.yaml')))
    
    # 命令行参数覆盖
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # 检测硬件
    if distributed_info.get("distributed"):
        device = distributed_info["device"]
        if main_process:
            names = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
            print("\n🔍 检测硬件环境...")
            print(
                f"✅ 使用 torchrun DistributedDataParallel："
                f"{distributed_info['world_size']} 个进程 / {torch.cuda.device_count()} 张 GPU ({names})"
            )
    else:
        if main_process:
            print("\n🔍 检测硬件环境...")
        hardware_info = HardwareDetector.detect()
        device = hardware_info['device']
    
    # 加载合并数据集
    if main_process:
        print("\n📊 加载合并数据集...")
    merged_datasets_dir = project_path(config.get('dataset', {}).get('merged_datasets_dir'), 'merged_datasets')
    configured_batch_size = config['training']['batch_size']
    loader_batch_size = configured_batch_size
    if distributed_info.get("distributed"):
        loader_batch_size = max(1, math.ceil(configured_batch_size / distributed_info['world_size']))
        if main_process:
            print(
                f"  配置 batch_size={configured_batch_size} 作为全局 batch，"
                f"DDP 每进程 batch_size={loader_batch_size}"
            )
    
    train_loader, val_loader, test_loader, dataset = create_unified_dataloaders(
        merged_datasets_dir=str(merged_datasets_dir),
        batch_size=loader_batch_size,
        num_workers=config['dataset'].get('num_workers', 0),  # 从 dataset 配置读取
        sample_fraction=config['dataset'].get('sample_fraction', 1.0),
        random_seed=config['dataset'].get('random_seed', 42),
        split_strategy=config['dataset'].get('split_strategy', 'group'),
        use_spatial=config['dataset'].get('use_spatial', True),
        pin_memory=device.type == "cuda",
        balanced_sampling=config['dataset'].get('balanced_sampling', True),
        ensure_rare_task_holdout=config['dataset'].get('ensure_rare_task_holdout', True),
        include_source_context=config['dataset'].get('include_source_context', False),
        use_mechanistic_descriptors=config['dataset'].get('use_mechanistic_descriptors', False),
        target_encoding=config['dataset'].get('target_encoding', 'onehot'),
    )
    
    if main_process:
        print(f"✅ 数据集加载完成")
        print(f"  训练集：{len(train_loader.dataset)} 样本")
        print(f"  验证集：{len(val_loader.dataset)} 样本")
        print(f"  测试集：{len(test_loader.dataset)} 样本")
        print(f"  训练集目标标准化: {dataset.task_stats}")
    
    # 创建模型
    if main_process:
        print("\n🏗️  创建模型...")
    model = LNPPredictor(
        mol_encoder_type=config['model']['mol_encoder_type'],
        mol_feat_dim=config['model']['mol_feat_dim'],
        atom_feat_dim=config['model']['atom_feat_dim'],
        bond_feat_dim=config['model']['bond_feat_dim'],
        num_gnn_layers=config['model']['num_gnn_layers'],
        use_3d=config['model']['use_3d'],
        struct_feat_dim=config['model']['struct_feat_dim'],
        use_images=config['model']['use_images'],
        image_feat_dim=config['model']['image_feat_dim'],
        image_channels=config['model']['image_channels'],
        image_size=config['model']['image_size'],
        use_embeddings=config['model']['use_embeddings'],
        embedding_feat_dim=config['model']['embedding_feat_dim'],
        max_embedding_dim=config['model']['max_embedding_dim'],
        formul_feat_dim=config['model']['formul_feat_dim'],
        num_components=config['model']['num_components'],
        formulation_input_dim=config['model'].get('formulation_input_dim', FORMULATION_DIM),
        context_feat_dim=config['model'].get('context_feat_dim', CONTEXT_DIM),
        target_feat_dim=config['model'].get('target_feat_dim', TARGET_DIM),
        num_target_classes=config['model'].get('num_target_classes', TARGET_DIM),
        spatial_feat_dim=config['model'].get('spatial_feat_dim', SPATIAL_DIM),
        use_component_transformer=config['model'].get('use_component_transformer', True),
        use_pair_bias=config['model'].get('use_pair_bias', True),
        use_target_conditioning=config['model'].get('use_target_conditioning', True),
        use_spatial_features=config['model'].get('use_spatial_features', True),
        use_ratio_features=config['model'].get('use_ratio_features', True),
        use_gaussian_ratio=config['model'].get('use_gaussian_ratio', False),
        use_structure_mask_features=config['model'].get('use_structure_mask_features', True),
        formulation_noise_std=config['model'].get('formulation_noise_std', 0.0),
        use_explicit_features=config['model'].get('use_explicit_features', False),
        explicit_fusion_mode=config['model'].get('explicit_fusion_mode', 'residual'),
        explicit_gate_init=config['model'].get('explicit_gate_init', -3.0),
        use_separate_rank_head=config['model'].get('use_separate_rank_head', False),
        rank_head_mode=config['model'].get('rank_head_mode', 'separate'),
        rank_detach_backbone=config['model'].get('rank_detach_backbone', False),
        rank_gate_init=config['model'].get('rank_gate_init', -1.0),
        fingerprint_input_dim=config['dataset'].get('fingerprint_dim', 2048),
        physchem_feat_dim=config['model']['physchem_feat_dim'],
        fusion_hidden_dim=config['model']['fusion_hidden_dim'],
        num_heads=config['model']['fusion_num_heads'],
        dropout=config['model']['fusion_dropout'],
        num_experts=config['model'].get('num_experts', 3),
        prediction_hidden_dim=config['model']['prediction_hidden_dim'],
        num_tasks=config['model']['num_tasks'],
        use_mc_dropout=config['model']['use_mc_dropout'],
        mc_dropout_rate=config['model'].get('mc_dropout_rate', 0.1),
    )
    
    # 创建训练器
    if main_process:
        print("\n🎯 创建训练器...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        config=config,
        distributed_info=distributed_info,
        dataset=dataset,
    )
    
    # 开始训练
    if main_process:
        print("\n🚀 开始训练...\n")
    try:
        trainer.train()
    finally:
        cleanup_distributed(distributed_info)
    
    if main_process:
        print("\n✅ 训练完成!")


if __name__ == '__main__':
    main()
