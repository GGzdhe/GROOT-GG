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

from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


@dataclass
class ArgsConfig:
    """Command line arguments for the inference service."""
    """推理服务的命令行参数"""

    model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path to the model checkpoint directory."""
    """模型检查点目录的路径"""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """The embodiment tag for the model."""
    """模型的具身标签"""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_waist"
    """The name of the data config to use."""
    """要使用的数据配置名称"""

    port: int = 5555
    """The port number for the server."""
    """服务器端口号"""

    host: str = "localhost"
    """The host address for the server."""
    """服务器主机地址"""

    server: bool = False
    """Whether to run the server."""
    """是否运行服务器"""

    client: bool = False
    """Whether to run the client."""
    """是否运行客户端"""

    denoising_steps: int = 4
    """The number of denoising steps to use."""
    """要使用的去噪步数"""

    api_token: str = None
    """API token for authentication. If not provided, authentication is disabled."""
    """用于身份验证的API令牌。如果未提供，则禁用身份验证"""


#####################################################################################


def main(args: ArgsConfig):
    if args.server:
        # Create a policy
        # 创建策略
        # The `Gr00tPolicy` class is being used to create a policy object that encapsulates
        # the model path, transform name, embodiment tag, and denoising steps for the robot
        # inference system. This policy object is then utilized in the server mode to start
        # the Robot Inference Server for making predictions based on the specified model and
        # configuration.
        # `Gr00tPolicy`类用于创建一个策略对象，该对象封装了
        # 机器人推理系统的模型路径、变换名称、具身标签和去噪步数。
        # 然后在服务器模式下使用此策略对象启动机器人推理服务器，
        # 以基于指定的模型和配置进行预测。

        # we will use an existing data config to create the modality config and transform
        # if a new data config is specified, this expect user to
        # construct your own modality config and transform
        # see gr00t/utils/data.py for more details
        # 我们将使用现有的数据配置来创建模态配置和变换
        # 如果指定了新的数据配置，则期望用户
        # 构建自己的模态配置和变换
        # 详见gr00t/utils/data.py
        data_config = DATA_CONFIG_MAP[args.data_config]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
        )

        # Start the server
        # 启动服务器
        server = RobotInferenceServer(policy, port=args.port, api_token=args.api_token)
        server.run()

    elif args.client:
        import time

        # In this mode, we will send a random observation to the server and get an action back
        # This is useful for testing the server and client connection
        # 在此模式下，我们将向服务器发送随机观察并获得动作反馈
        # 这对于测试服务器和客户端连接很有用
        # Create a policy wrapper
        # 创建策略包装器
        policy_client = RobotInferenceClient(
            host=args.host, port=args.port, api_token=args.api_token
        )

        print("Available modality config available:")
        modality_configs = policy_client.get_modality_config()
        print(modality_configs.keys())

        # Making prediction...
        # 进行预测...
        # - obs: video.ego_view: (1, 256, 256, 3)
        # - obs: state.left_arm: (1, 7)
        # - obs: state.right_arm: (1, 7)
        # - obs: state.left_hand: (1, 6)
        # - obs: state.right_hand: (1, 6)
        # - obs: state.waist: (1, 3)

        # - action: action.left_arm: (16, 7)
        # - action: action.right_arm: (16, 7)
        # - action: action.left_hand: (16, 6)
        # - action: action.right_hand: (16, 6)
        # - action: action.waist: (16, 3)
        
        # 构造测试观察数据
        obs = {
            "video.ego_view": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
            "state.left_arm": np.random.rand(1, 7),
            "state.right_arm": np.random.rand(1, 7),
            "state.left_hand": np.random.rand(1, 6),
            "state.right_hand": np.random.rand(1, 6),
            "state.waist": np.random.rand(1, 3),
            "annotation.human.action.task_description": ["do your thing!"],
        }

        time_start = time.time()
        action = policy_client.get_action(obs)
        print(f"Total time taken to get action from server: {time.time() - time_start} seconds")

        # 打印返回的动作信息
        for key, value in action.items():
            print(f"Action: {key}: {value.shape}")

    else:
        raise ValueError("Please specify either --server or --client")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
