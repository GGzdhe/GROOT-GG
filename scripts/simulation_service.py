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

import argparse

import numpy as np

from gr00t.eval.robot import RobotInferenceServer
from gr00t.eval.simulation import (
    MultiStepConfig,
    SimulationConfig,
    SimulationInferenceClient,
    VideoConfig,
)
from gr00t.model.policy import Gr00tPolicy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        # 模型检查点目录的路径
        default="<PATH_TO_YOUR_MODEL>",  # change this to your model path
                                        # 将此更改为您的模型路径
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        # 模型的embodiment标签
        default="<EMBODIMENT_TAG>",  # change this to your embodiment tag
                                    # 将此更改为您的embodiment标签
    )
    parser.add_argument(
        "--env_name",
        type=str,
        help="Name of the environment to run.",
        # 要运行的环境名称
        default="<ENV_NAME>",  # change this to your environment name
                             # 将此更改为您的环境名称
    )
    parser.add_argument("--port", type=int, help="Port number for the server.", default=5555)
    # 服务器端口号
    parser.add_argument(
        "--host", type=str, help="Host address for the server.", default="localhost"
    )
    # 服务器主机地址
    parser.add_argument(
        "--video_dir", type=str, help="Directory to save videos.", default="./videos"
    )
    # 保存视频的目录
    parser.add_argument("--n_episodes", type=int, help="Number of episodes to run.", default=2)
    # 运行的episode数量
    parser.add_argument("--n_envs", type=int, help="Number of parallel environments.", default=1)
    # 并行环境数量
    parser.add_argument(
        "--n_action_steps",
        type=int,
        help="Number of action steps per environment step.",
        # 每个环境步的动作步数
        default=16,
    )
    parser.add_argument(
        "--max_episode_steps", type=int, help="Maximum number of steps per episode.", default=1440
    )
    # 每个episode的最大步数
    # server mode
    # 服务器模式
    parser.add_argument("--server", action="store_true", help="Run the server.")
    # 以服务器模式运行
    # client mode
    # 客户端模式
    parser.add_argument("--client", action="store_true", help="Run the client")
    # 以客户端模式运行
    args = parser.parse_args()

    if args.server:
        # Create a policy
        # 创建一个策略对象
        policy = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=args.embodiment_tag,
        )

        # Start the server
        # 启动推理服务器
        server = RobotInferenceServer(policy, port=args.port)
        server.run()

    elif args.client:
        # Create a simulation client
        # 创建仿真客户端
        simulation_client = SimulationInferenceClient(host=args.host, port=args.port)

        print("Available modality configs:")
        # 获取可用的模态配置
        modality_config = simulation_client.get_modality_config()
        print(modality_config.keys())

        # Create simulation configuration
        # 创建仿真配置
        config = SimulationConfig(
            env_name=args.env_name,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            video=VideoConfig(video_dir=args.video_dir),
            multistep=MultiStepConfig(
                n_action_steps=args.n_action_steps, max_episode_steps=args.max_episode_steps
            ),
        )

        # Run the simulation
        # 运行仿真
        print(f"Running simulation for {args.env_name}...")
        env_name, episode_successes = simulation_client.run_simulation(config)

        # Print results
        # 打印结果
        print(f"Results for {env_name}:")
        print(f"Success rate: {np.mean(episode_successes):.2f}")

    else:
        raise ValueError("Please specify either --server or --client")
        # 请指定--server或--client参数
