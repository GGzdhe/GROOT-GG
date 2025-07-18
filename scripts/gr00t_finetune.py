# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model


@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning."""
    """GR00T模型微调的配置类"""

    # Dataset parameters
    # 数据集参数
    dataset_path: List[str]
    """Path to the dataset directory or directories"""
    """数据集目录的路径（可以是单个或多个目录）"""

    output_dir: str = "/tmp/gr00t"
    """Directory to save model checkpoints."""
    """保存模型检查点的目录"""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_only"
    """Data configuration name from DATA_CONFIG_MAP, we assume all datasets have the same data config"""
    """来自DATA_CONFIG_MAP的数据配置名称，假设所有数据集具有相同的数据配置"""

    # Training parameters
    # 训练参数
    batch_size: int = 32
    """Batch size per GPU for training."""
    """每个GPU的训练批次大小"""

    max_steps: int = 10000
    """Maximum number of training steps."""
    """最大训练步数"""

    num_gpus: int = 1
    """Number of GPUs to use for training."""
    """用于训练的GPU数量"""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""
    """保存检查点的步数间隔"""

    # Model parameters
    # 模型参数
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""
    """基础模型的路径或HuggingFace模型ID"""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""
    """是否微调语言模型主干"""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""
    """是否微调视觉塔"""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""
    """是否微调投影器"""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""
    """是否微调扩散模型"""

    resume: bool = False
    """Whether to resume from a checkpoint."""
    """是否从检查点恢复训练"""

    # Advanced training parameters
    # 高级训练参数
    learning_rate: float = 1e-4
    """Learning rate for training."""
    """训练学习率"""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""
    """AdamW优化器的权重衰减"""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""
    """用于预热的总训练步数比例"""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""
    """LORA模型的秩。如果为0，则不使用LORA"""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""
    """LORA模型的alpha值"""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""
    """LORA模型的dropout率"""

    lora_full_model: bool = False
    """Whether to use the full model for LORA. If False, only the action head will be trained."""
    """是否对整个模型使用LORA。如果为False，则只训练动作头"""

    dataloader_num_workers: int = 8
    """Number of workers for data loading."""
    """数据加载的工作进程数"""

    report_to: Literal["wandb", "tensorboard"] = "wandb"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard')."""
    """训练指标报告位置（例如：'wandb', 'tensorboard'）"""

    # Data loading parameters
    # 数据加载参数
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""
    """训练使用的具身标签，例如'new_embodiment', 'gr1'"""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use for training. [decord, torchvision_av]"""
    """训练使用的视频后端 [decord, torchvision_av]"""

    # Mixture dataset parameters
    # 混合数据集参数
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, we will balance the dataset weights, by multiplying the total trajectory to each dataset"""
    """在LeRobotMixtureDataset中使用。如果为True，将通过乘以每个数据集的总轨迹来平衡数据集权重"""

    # Mixture dataset parameters
    # 混合数据集参数
    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, sample trajectories within a dataset weighted by their length; otherwise, equal weighting."""
    """在LeRobotMixtureDataset中使用。如果为True，根据轨迹长度在数据集内加权采样轨迹；否则使用等权重"""


#####################################################################################
# main training function
# 主训练函数
#####################################################################################


def main(config: ArgsConfig):
    """Main training function."""
    """主训练函数"""
    # ------------ step 1: load dataset ------------
    # ------------ 步骤1：加载数据集 ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 modality configs and transforms
    # 1.1 模态配置和变换
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # 1.2 data loader: we will use either single dataset or mixture dataset
    # 1.2 数据加载器：使用单个数据集或混合数据集
    if len(config.dataset_path) == 1:
        # 如果只有一个数据集路径，创建单数据集
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
                                          # 这将覆盖数据集的具身标签为"new_embodiment"
            video_backend=config.video_backend,
        )
    else:
        # 如果有多个数据集路径，创建混合数据集
        single_datasets = []
        for p in config.dataset_path:
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            ## We use the same transforms, modality configs, and embodiment tag for all datasets here,
            ## in reality, you can use dataset from different modalities and embodiment tags
            ## 这里对所有数据集使用相同的变换、模态配置和具身标签，
            ## 实际上可以使用不同模态和具身标签的数据集
            dataset = LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
            )
            single_datasets.append(dataset)

        train_dataset = LeRobotMixtureDataset(
            data_mixture=[
                (dataset, 1.0)  # we will use equal weights for all datasets
                              # 对所有数据集使用相等权重
                for dataset in single_datasets
            ],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )
        print(f"Loaded {len(single_datasets)} datasets, with {config.dataset_path} ")

    # ------------ step 2: load model ------------
    # ------------ 步骤2：加载模型 ------------
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,  # backbone's LLM
                                 # 主干的LLM
        tune_visual=config.tune_visual,  # backbone's vision tower
                                       # 主干的视觉塔
        tune_projector=config.tune_projector,  # action head's projector
                                             # 动作头的投影器
        tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
                                                          # 动作头的DiT
    )

    # Set the model's compute_dtype to bfloat16
    # 将模型的计算数据类型设置为bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    # 如果配置了LORA秩大于0，则应用LORA模型
    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    # 2.1 modify training args
    # 2.1 修改训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,  # 使用bfloat16精度训练
        tf32=True,  # 启用TensorFloat-32
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",  # 使用AdamW优化器
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",  # 使用余弦学习率调度器
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        # evaluation_strategy="no",
        save_total_limit=8,  # 最多保存8个检查点
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # 2.2 run experiment
    # 2.2 运行实验
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    # 2.3 run experiment
    # 2.3 运行实验
    experiment.train()


if __name__ == "__main__":
    # Parse arguments using tyro
    # 使用tyro解析命令行参数
    config = tyro.cli(ArgsConfig)

    # Print the tyro config
    # 打印tyro配置
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    # 获取可用GPU数量
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    # 验证GPU配置
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        # 单GPU模式 - 设置CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        # 正常运行脚本
        main(config)
    else:
        # 检查是否在torchrun环境中
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            # 多GPU模式 - 使用torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            # 从环境中移除任何现有的CUDA_VISIBLE_DEVICES
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            # Use subprocess.run instead of os.system
            # 使用subprocess.run而不是os.system
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                             # 默认使用1个节点
                str(script_path),
            ]

            # Convert config to command line arguments
            # 将配置转换为命令行参数
            for key, value in vars(config).items():
                if isinstance(value, bool):
                    # For boolean values, use --flag or --no-flag format
                    # 对于布尔值，使用--flag或--no-flag格式
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
                    else:
                        cmd.append(f"--no-{key.replace('_', '-')}")
                else:
                    # For non-boolean values, use --key value format
                    # 对于非布尔值，使用--key value格式
                    cmd.append(f"--{key.replace('_', '-')}")

                    # if the value is a list (e.g. dataset_path), we need to add each element in the list
                    # 如果值是列表（例如dataset_path），需要添加列表中的每个元素
                    if isinstance(value, list):
                        for v in value:
                            cmd.append(str(v))
                    else:
                        cmd.append(str(value))
            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
