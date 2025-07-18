# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical computation and array operations
import os
from datetime import datetime

from gr00t.data.dataset import LeRobotSingleDataset  # Custom dataset class
from gr00t.model.policy import BasePolicy  # Policy base class

# numpy print precision settings 3, dont use exponential notation
np.set_printoptions(precision=3, suppress=True)
# 设置numpy的打印精度为3位小数，禁止科学计数法


def download_from_hg(repo_id: str, repo_type: str) -> str:
    """
    Download the model/dataset from the hugging face hub.
    return the path to the downloaded
    从huggingface hub下载模型或数据集。
    返回下载后的本地路径。
    """
    from huggingface_hub import snapshot_download

    repo_path = snapshot_download(repo_id, repo_type=repo_type)
    return repo_path


def get_plot_save_path(traj_id, modality_keys, output_dir="output/eval_policy_plots"):
    """
    生成保存图片的路径，文件名包含轨迹编号、模态信息和时间戳，便于后续统一修改和比较。
    traj_id: 轨迹编号
    modality_keys: 模态键列表
    output_dir: 保存文件夹
    """
    # os.makedirs(output_dir, exist_ok=True) # 创建保存文件夹
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S") # 获取当前时间戳
    # %Y：四位数的年份（如 2024）
    # %m：两位数的月份（01-12）
    # %d：两位数的日期（01-31）
    # %H：24小时制的小时（00-23）
    # %M：分钟（00-59）
    # %S：秒（00-59）
    filename = f"traj_{traj_id}_modalities_{'_'.join(modality_keys)}_{time_str}.png"
    return os.path.join(output_dir, filename)


def calc_mse_for_single_trajectory(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    modality_keys: list,
    steps=300,
    action_horizon=16,
    plot=False,
):
    """
    Calculate the mean squared error (MSE) for a single trajectory, and optionally plot the results.
    计算单条轨迹的均方误差（MSE），并可选地绘制对比图。
    参数：
        policy: 策略模型
        dataset: 数据集对象
        traj_id: 轨迹编号
        modality_keys: 需要评估的模态键列表
        steps: 评估步数
        action_horizon: 动作时域
        plot: 是否绘图
    """
    # Store state, ground truth action, and predicted action for each step
    state_joints_across_time = []  # 每一步的状态
    gt_action_across_time = []     # 每一步的真实动作
    pred_action_across_time = []   # 每一步的预测动作

    # Iterate through each time step and collect data
    for step_count in range(steps):
        data_point = dataset.get_step_data(traj_id, step_count)

        # Concatenate all modalities' state and action
        # 拼接所有模态的状态和动作
        concat_state = np.concatenate(
            [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
        )
        concat_gt_action = np.concatenate(
            [data_point[f"action.{key}"][0] for key in modality_keys], axis=0
        )

        state_joints_across_time.append(concat_state)
        gt_action_across_time.append(concat_gt_action)

        # Every action_horizon steps, run inference to get predicted actions
        # 每隔action_horizon步推理一次，获取预测动作
        if step_count % action_horizon == 0:
            print("inferencing at step: ", step_count)
            action_chunk = policy.get_action(data_point)
            for j in range(action_horizon):
                # Concatenate all modalities' predicted actions, ensure 1D array
                # 拼接所有模态的预测动作，保证为1D数组
                concat_pred_action = np.concatenate(
                    [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys],
                    axis=0,
                )
                pred_action_across_time.append(concat_pred_action)

    # Convert to numpy arrays for later computation and plotting
    # 转为numpy数组，方便后续计算和绘图
    state_joints_across_time = np.array(state_joints_across_time)
    gt_action_across_time = np.array(gt_action_across_time)
    pred_action_across_time = np.array(pred_action_across_time)[:steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape

    # Calculate mean squared error (MSE)
    # 计算均方误差（MSE）
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    print("Unnormalized Action MSE across single traj:", mse)

    print("state_joints vs time", state_joints_across_time.shape)
    print("gt_action_joints vs time", gt_action_across_time.shape)
    print("pred_action_joints vs time", pred_action_across_time.shape)

    # Get action dimension
    # 获取动作维度
    action_dim = gt_action_across_time.shape[1]

    # If plotting is enabled
    # 如果需要绘图
    if plot:
        # Create subplots, one for each action dimension
        # 创建子图，每个动作一个子图
        fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(8, 4 * action_dim))

        # Add a global title showing the modality keys
        # 添加全局标题，显示模态信息
        fig.suptitle(
            f"Trajectory {traj_id} - Modalities: {', '.join(modality_keys)}",
            fontsize=16,
            color="blue",
        )

        for i, ax in enumerate(axes):
            # The dimensions of state_joints and action are the same only when the robot uses actions directly as joint commands.
            # Therefore, do not plot them if this is not the case.
            # 只有状态和动作维度一致时才画状态
            if state_joints_across_time.shape == gt_action_across_time.shape:
                ax.plot(state_joints_across_time[:, i], label="state joints")
            ax.plot(gt_action_across_time[:, i], label="gt action")
            ax.plot(pred_action_across_time[:, i], label="pred action")

            # put a dot every ACTION_HORIZON
            # 每隔action_horizon画一个红点，表示推理点
            for j in range(0, steps, action_horizon):
                if j == 0:
                    ax.plot(j, gt_action_across_time[j, i], "ro", label="inference point")
                else:
                    ax.plot(j, gt_action_across_time[j, i], "ro")

            ax.set_title(f"Action {i}")
            ax.legend()

        plt.tight_layout()
        # plt.show()  # Show the plot (not saved automatically)
        #             # 显示图像（不会自动保存）
        save_path = get_plot_save_path(traj_id, modality_keys)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.close()  # 关闭图像，适合服务器环境

    return mse  # Return mean squared error
                # 返回均方误差
