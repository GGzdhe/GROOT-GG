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

"""
This script is a replication of the notebook `getting_started/load_dataset.ipynb`
"""
"""
这个脚本是笔记本 `getting_started/load_dataset.ipynb` 的复制版本
"""

import json
import pathlib
import time
from dataclasses import dataclass, field
from pprint import pprint
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import tyro

from gr00t.data.dataset import (
    LE_ROBOT_MODALITY_FILENAME,
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
    ModalityConfig,
)
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from gr00t.utils.misc import any_describe


def print_yellow(text: str) -> None:
    """Print text in yellow color"""
    """以黄色打印文本"""
    print(f"\033[93m{text}\033[0m")


@dataclass
class ArgsConfig:
    """Configuration for loading the dataset."""
    """加载数据集的配置类"""

    dataset_path: List[str] = field(default_factory=lambda: ["demo_data/robot_sim.PickNPlace"])
    """Path to the dataset."""
    """数据集的路径"""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """Embodiment tag to use."""
    """要使用的具身标签"""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Backend to use for video loading, use torchvision_av for av encoded videos."""
    """视频加载使用的后端，对于av编码的视频使用torchvision_av"""

    plot_state_action: bool = False
    """Whether to plot the state and action space."""
    """是否绘制状态和动作空间"""

    steps: int = 200
    """Number of steps to plot."""
    """要绘制的步数"""


#####################################################################################


def get_modality_keys(dataset_path: pathlib.Path) -> dict[str, list[str]]:
    """
    Get the modality keys from the dataset path.
    Returns a dictionary with modality types as keys and their corresponding modality keys as values,
    maintaining the order: video, state, action, annotation
    """
    """
    从数据集路径获取模态键。
    返回一个字典，以模态类型为键，对应的模态键为值，
    保持顺序：video, state, action, annotation
    """
    modality_path = dataset_path / LE_ROBOT_MODALITY_FILENAME
    with open(modality_path, "r") as f:
        modality_meta = json.load(f)

    # Initialize dictionary with ordered keys
    # 初始化有序字典
    modality_dict = {}
    for key in modality_meta.keys():
        modality_dict[key] = []
        for modality in modality_meta[key]:
            modality_dict[key].append(f"{key}.{modality}")
    return modality_dict


def plot_state_action_space(
    state_dict: dict[str, np.ndarray],
    action_dict: dict[str, np.ndarray],
    shared_keys: list[str] = ["left_arm", "right_arm", "left_hand", "right_hand"],
):
    """
    Plot the state and action space side by side.

    state_dict: dict[str, np.ndarray] with key: [Time, Dimension]
    action_dict: dict[str, np.ndarray] with key: [Time, Dimension]
    shared_keys: list[str] of keys to plot (without the "state." or "action." prefix)
    """
    """
    并排绘制状态和动作空间。

    state_dict: 字典[str, np.ndarray]，键: [时间, 维度]
    action_dict: 字典[str, np.ndarray]，键: [时间, 维度]
    shared_keys: 要绘制的键列表（不包含"state."或"action."前缀）
    """
    # Create a figure with one subplot per shared key
    # 为每个共享键创建一个子图
    fig = plt.figure(figsize=(16, 4 * len(shared_keys)))

    # Create GridSpec to organize the layout
    # 创建GridSpec来组织布局
    gs = fig.add_gridspec(len(shared_keys), 1)

    # Color palette for different dimensions
    # 不同维度的颜色调色板
    colors = plt.cm.tab10.colors

    for i, key in enumerate(shared_keys):
        state_key = f"state.{key}"
        action_key = f"action.{key}"

        # Skip if either key is not in the dictionaries
        # 如果任一键不在字典中，则跳过
        if state_key not in state_dict or action_key not in action_dict:
            print(
                f"Warning: Skipping {key} as it's not found in both state and action dictionaries"
            )
            continue

        # Get the data
        # 获取数据
        state_data = state_dict[state_key]
        action_data = action_dict[action_key]

        print(f"{state_key}.shape: {state_data.shape}")
        print(f"{action_key}.shape: {action_data.shape}")

        # Create subplot
        # 创建子图
        ax = fig.add_subplot(gs[i, 0])

        # Plot each dimension with a different color
        # Determine the minimum number of dimensions to plot
        # 用不同颜色绘制每个维度
        # 确定要绘制的最小维度数
        min_dims = min(state_data.shape[1], action_data.shape[1])

        for dim in range(min_dims):
            # Create time arrays for both state and action
            # 为状态和动作创建时间数组
            state_time = np.arange(len(state_data))
            action_time = np.arange(len(action_data))

            # State with dashed line
            # 状态用虚线表示
            ax.plot(
                state_time,
                state_data[:, dim],
                "--",
                color=colors[dim % len(colors)],
                linewidth=1.5,
                label=f"state dim {dim}",
            )

            # Action with solid line (same color as corresponding state dimension)
            # 动作用实线表示（与对应状态维度相同颜色）
            ax.plot(
                action_time,
                action_data[:, dim],
                "-",
                color=colors[dim % len(colors)],
                linewidth=2,
                label=f"action dim {dim}",
            )

        ax.set_title(f"{key}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle=":", alpha=0.7)

        # Create a more organized legend
        # 创建更有序的图例
        handles, labels = ax.get_legend_handles_labels()
        # Sort the legend so state and action for each dimension are grouped
        # 排序图例，使每个维度的状态和动作分组
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()


def plot_image(image: np.ndarray):
    """
    Plot the image.
    """
    """
    绘制图像。
    """
    # matplotlib show the image
    # 使用matplotlib显示图像
    plt.imshow(image)
    plt.axis("off")
    plt.pause(0.05)  # Non-blocking show  # 非阻塞显示
    plt.clf()  # Clear the figure for the next frame  # 清除图形为下一帧做准备


def load_dataset(
    dataset_path: List[str],
    embodiment_tag: str,
    video_backend: str = "decord",
    steps: int = 200,
    plot_state_action: bool = False,
):
    assert len(dataset_path) > 0, "dataset_path must be a list of at least one path"

    # 1. get modality keys
    # 1. 获取模态键
    single_dataset_path = pathlib.Path(
        dataset_path[0]
    )  # take first one, assume all have same modality keys
       # 取第一个，假设所有都有相同的模态键
    modality_keys_dict = get_modality_keys(single_dataset_path)
    video_modality_keys = modality_keys_dict["video"]
    language_modality_keys = modality_keys_dict["annotation"]
    state_modality_keys = modality_keys_dict["state"]
    action_modality_keys = modality_keys_dict["action"]

    pprint(f"Valid modality_keys for debugging:: {modality_keys_dict} \n")

    print(f"state_modality_keys: {state_modality_keys}")
    print(f"action_modality_keys: {action_modality_keys}")

    # remove dummy_tensor from state_modality_keys
    # 从state_modality_keys中移除dummy_tensor
    state_modality_keys = [key for key in state_modality_keys if key != "state.dummy_tensor"]

    # 2. construct modality configs from dataset
    # 2. 从数据集构建模态配置
    modality_configs = {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=video_modality_keys,  # we will include all video modalities
                                              # 包含所有视频模态
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=state_modality_keys,
        ),
        "action": ModalityConfig(
            delta_indices=[0],
            modality_keys=action_modality_keys,
        ),
    }

    # 3. language modality config (if exists)
    # 3. 语言模态配置（如果存在）
    if language_modality_keys:
        modality_configs["language"] = ModalityConfig(
            delta_indices=[0],
            modality_keys=language_modality_keys,
        )

    # 4. gr00t embodiment tag
    # 4. gr00t具身标签
    embodiment_tag: EmbodimentTag = EmbodimentTag(embodiment_tag)

    # 5. load dataset
    # 5. 加载数据集
    print(f"Loading dataset from {dataset_path}")
    if len(dataset_path) == 1:
        # 单个数据集
        dataset = LeRobotSingleDataset(
            dataset_path=dataset_path[0],
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
        )
    else:
        # 多个数据集
        print(f"Loading {len(dataset_path)} datasets")
        lerobot_single_datasets = []
        for data_path in dataset_path:
            dataset = LeRobotSingleDataset(
                dataset_path=data_path,
                modality_configs=modality_configs,
                embodiment_tag=embodiment_tag,
                video_backend=video_backend,
            )
            lerobot_single_datasets.append(dataset)

        # we will do a simple 1.0 sampling weight mix of the datasets
        # 对数据集进行简单的1.0采样权重混合
        dataset = LeRobotMixtureDataset(
            data_mixture=[(dataset, 1.0) for dataset in lerobot_single_datasets],
            mode="train",
            balance_dataset_weights=True,  # balance based on number of trajectories
                                         # 基于轨迹数量平衡
            balance_trajectory_weights=True,  # balance based on trajectory length
                                            # 基于轨迹长度平衡
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )
        print_yellow(
            "NOTE: when using mixture dataset, we will randomly sample from all the datasets"
            "thus the state action ploting will not make sense, this is helpful to visualize the images"
            "to quickly sanity check the dataset used."
        )
        print_yellow(
            "注意：使用混合数据集时，我们会从所有数据集中随机采样，"
            "因此状态动作绘图没有意义，这有助于可视化图像"
            "以快速检查使用的数据集。"
        )

    print("\n" * 2)
    print("=" * 100)
    print(f"{' Humanoid Dataset ':=^100}")
    print("=" * 100)

    # print the 7th data point
    # 打印第7个数据点
    # resp = dataset[7]
    resp = dataset[0]
    any_describe(resp)
    print(resp.keys())

    print("=" * 50)
    for key, value in resp.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")

    # 6. plot the first 100 images
    # 6. 绘制前100张图像
    images_list = []
    video_key = video_modality_keys[0]  # we will use the first video modality
                                      # 使用第一个视频模态

    state_dict = {key: [] for key in state_modality_keys}
    action_dict = {key: [] for key in action_modality_keys}

    total_images = 20  # show 20 images  # 显示20张图像
    skip_frames = steps // total_images

    for i in range(steps):
        resp = dataset[i]
        if i % skip_frames == 0:
            img = resp[video_key][0]
            # cv2 show the image
            # plot_image(img)
            if language_modality_keys:
                lang_key = language_modality_keys[0]
                print(f"Image {i}, prompt: {resp[lang_key]}")
            else:
                print(f"Image {i}")
            images_list.append(img.copy())

        # 收集状态和动作数据
        for state_key in state_modality_keys:
            state_dict[state_key].append(resp[state_key][0])
        for action_key in action_modality_keys:
            action_dict[action_key].append(resp[action_key][0])
        time.sleep(0.05)

    # convert lists of [np[D]] T size to np(T, D)
    # 将[np[D]]大小为T的列表转换为np(T, D)
    for state_key in state_modality_keys:
        state_dict[state_key] = np.array(state_dict[state_key])
    for action_key in action_modality_keys:
        action_dict[action_key] = np.array(action_dict[action_key])

    if plot_state_action:
        plot_state_action_space(state_dict, action_dict)
        print("Plotted state and action space")

    # 创建图像网格显示
    fig, axs = plt.subplots(4, total_images // 4, figsize=(20, 10))
    for i, ax in enumerate(axs.flat):
        ax.imshow(images_list[i])
        ax.axis("off")
        ax.set_title(f"Image {i*skip_frames}")
    plt.tight_layout()  # adjust the subplots to fit into the figure area.
                       # 调整子图以适应图形区域
    plt.show()


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    load_dataset(
        config.dataset_path,
        config.embodiment_tag,
        config.video_backend,
        config.steps,
        config.plot_state_action,
    )
