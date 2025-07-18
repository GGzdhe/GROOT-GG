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

import warnings
from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
import tyro

from gr00t.data.dataset import LeRobotSingleDataset # 用于加载和处理机器人相关的数据集，后续用来读取评估用的数据
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING # 用于映射不同的具身标签
from gr00t.eval.robot import RobotInferenceClient # 用于远程推理的客户端
from gr00t.experiment.data_config import DATA_CONFIG_MAP # 用于加载和处理不同的数据配置
from gr00t.model.policy import BasePolicy, Gr00tPolicy # 用于加载和处理不同的策略
from gr00t.utils.eval import calc_mse_for_single_trajectory # 用于计算单个轨迹的MSE

warnings.simplefilter("ignore", category=FutureWarning)

"""
Example command:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

python scripts/eval_policy.py --plot --model-path nvidia/GR00T-N1.5-3B
"""
"""
示例命令:

注意：提供--model_path在此脚本中加载模型检查点，
     否则将通过RobotInferenceClient使用默认主机和端口

python scripts/eval_policy.py --plot --model-path nvidia/GR00T-N1.5-3B
"""


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""
    """评估策略的配置"""

    host: str = "localhost"
    """Host to connect to."""
    """连接的主机"""

    port: int = 5555
    """Port to connect to."""
    """连接的端口"""

    plot: bool = True
    """Whether to plot the images."""
    """是否绘制图像"""

    modality_keys: List[str] = field(default_factory=lambda: ["right_arm", "left_arm"])
    """Modality keys to evaluate."""
    """要评估的模态键"""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_only"
    """Data config to use."""
    """要使用的数据配置"""

    steps: int = 150
    """Number of steps to evaluate."""
    """要评估的步数"""

    trajs: int = 1
    """Number of trajectories to evaluate."""
    """要评估的轨迹数"""

    action_horizon: int = 16
    """Action horizon to evaluate."""
    """要评估的动作时域"""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use for various codec options. h264: decord or av: torchvision_av"""
    """用于各种编解码器选项的视频后端。h264：decord或av：torchvision_av"""

    dataset_path: str = "demo_data/robot_sim.PickNPlace/"
    """Path to the dataset."""
    """数据集路径"""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """Embodiment tag to use."""
    """要使用的具身标签"""

    model_path: str = None
    """Path to the model checkpoint."""
    """模型检查点路径"""

    denoising_steps: int = 4
    """Number of denoising steps to use."""
    """要使用的去噪步数"""


def main(args: ArgsConfig):
    data_config = DATA_CONFIG_MAP[args.data_config]
    if args.model_path is not None:
        # 如果提供了模型路径，直接加载本地模型
        import torch

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy: BasePolicy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        # 否则使用远程推理客户端
        policy: BasePolicy = RobotInferenceClient(host=args.host, port=args.port)

    # Get the supported modalities for the policy
    # 获取策略支持的模态
    modality = policy.get_modality_config()
    print("Current modality config: \n", modality)

    # Create the dataset
    # 创建数据集
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
                         # 我们将通过策略单独处理变换
        embodiment_tag=args.embodiment_tag,
    )

    print(len(dataset))
    # Make a prediction
    # 进行预测
    obs = dataset[0]
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    # 获取单步数据
    for k, v in dataset.get_step_data(0, 0).items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    print("Total trajectories:", len(dataset.trajectory_lengths))
    print("All trajectories:", dataset.trajectory_lengths)
    print("Running on all trajs with modality keys:", args.modality_keys)

    # 对每个轨迹计算MSE（均方误差）
    all_mse = []
    for traj_id in range(args.trajs):
        print("Running trajectory:", traj_id)
        mse = calc_mse_for_single_trajectory(
            policy,
            dataset,
            traj_id,
            modality_keys=args.modality_keys,
            steps=args.steps,
            action_horizon=args.action_horizon,
            plot=args.plot,
        )
        print("MSE:", mse)
        all_mse.append(mse)
    print("Average MSE across all trajs:", np.mean(all_mse))
    print("Done")
    exit()


if __name__ == "__main__":
    # Parse arguments using tyro
    # 使用tyro解析参数
    config = tyro.cli(ArgsConfig)
    main(config)
