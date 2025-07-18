# Deeper Understanding
# 深入理解

In this section, we will dive deeper into the training configuration options. And we will also explain more about embodiment tags, modality configs, data transforms, and more.

在本节中，我们将深入了解训练配置选项。我们还将更多地解释具身标签、模态配置、数据变换等内容。


## Embodiment Action Head Fine-tuning
## 具身动作头微调

GR00T is designed to work with different types of robots (embodiments) through specialized action heads. When fine-tuning, you need to specify which embodiment head to train based on your dataset:

GR00T设计为通过专门的动作头与不同类型的机器人（具身）配合工作。在微调时，您需要根据数据集指定要训练哪个具身头：

1. **Embodiment Tags**
   **具身标签**
   - Each dataset must be tagged with a specific `EmbodimentTag` (e.g., EmbodimentTag.GR1_UNIFIED) while instantiating the `LeRobotSingleDataset` class
   - An exhaustive list of embodiment tags can be found in `gr00t/data/embodiment_tags.py`
   - This tag determines which action head will be fine-tuned
   - If you have a new embodiment, you can use the `EmbodimentTag.NEW_EMBODIMENT` tag (e.g., `new_embodiment.your_custom_dataset`)

   - 在实例化`LeRobotSingleDataset`类时，每个数据集必须用特定的`EmbodimentTag`进行标记（例如，EmbodimentTag.GR1_UNIFIED）
   - 具身标签的详尽列表可以在`gr00t/data/embodiment_tags.py`中找到
   - 此标签决定了将微调哪个动作头
   - 如果您有新的具身，可以使用`EmbodimentTag.NEW_EMBODIMENT`标签（例如，`new_embodiment.your_custom_dataset`）

2. **How it Works**
   **工作原理**
   - When you load your dataset with a specific embodiment tag (e.g., `EmbodimentTag.GR1`)
   - The model has multiple components that can be configured for fine-tuning (Visual Encoder, Language Model, DiT, etc.)
   - For action heads specifically, only the one corresponding to your specified embodiment tag will be fine-tuned. Other embodiment-specific action heads remain frozen

   - 当您使用特定具身标签加载数据集时（例如，`EmbodimentTag.GR1`）
   - 模型有多个可配置用于微调的组件（视觉编码器、语言模型、DiT等）
   - 特别是对于动作头，只有与您指定的具身标签对应的那个会被微调。其他特定于具身的动作头保持冻结

3. **Supported Embodiment**
   **支持的具身**

   | Embodiment Tag | Description | Data Config | Observation Space | Action Space | Notes |
   | 具身标签 | 描述 | 数据配置 | 观察空间 | 动作空间 | 注释 |
   |-|-|-|-|-|-|
   | `EmbodimentTag.GR1` | Fourier GR1 Robot | `fourier_gr1_arms_waist` | `video.ego_view`, `state.left_arm`, `state.right_arm`, `state.left_hand`, `state.right_hand`, `state.waist` | `action.left_arm`, `action.right_arm`, `action.left_hand`, `action.right_hand`, `action.waist`, `action.robot_velocity` | Absolute joint control |
   | `EmbodimentTag.GR1` | 傅立叶GR1机器人 | `fourier_gr1_arms_waist` | `video.ego_view`, `state.left_arm`, `state.right_arm`, `state.left_hand`, `state.right_hand`, `state.waist` | `action.left_arm`, `action.right_arm`, `action.left_hand`, `action.right_hand`, `action.waist`, `action.robot_velocity` | 绝对关节控制 |
   | `EmbodimentTag.OXE_DROID` | OXE Droid | `oxe_droid` | `video.exterior_image_1`, `video.exterior_image_2`, `video.wrist_image`, `state.eef_position`, `state.eef_rotation`, `state.gripper_position` | `action.eef_position_delta`, `action.eef_rotation_delta`, `action.gripper_position` | Delta end effector control |
   | `EmbodimentTag.OXE_DROID` | OXE机器人 | `oxe_droid` | `video.exterior_image_1`, `video.exterior_image_2`, `video.wrist_image`, `state.eef_position`, `state.eef_rotation`, `state.gripper_position` | `action.eef_position_delta`, `action.eef_rotation_delta`, `action.gripper_position` | 增量末端执行器控制 |
   | `EmbodimentTag.GENIE1_GRIPPER` | Agibot Genie-1 with gripper | `agibot_genie1` | `video.top_head`, `video.hand_left`, `video.hand_right`, `state.left_arm_joint_position`, `state.right_arm_joint_position`, `state.left_effector_position`, `state.right_effector_position`, `state.head_position`, `state.waist_position` | `action.left_arm_joint_position`, `action.right_arm_joint_position`, `action.left_effector_position`, `action.right_effector_position`, `action.head_position`, `action.waist_position`, `action.robot_velocity` | Absolute joint control |
   | `EmbodimentTag.GENIE1_GRIPPER` | 带夹爪的Agibot Genie-1 | `agibot_genie1` | `video.top_head`, `video.hand_left`, `video.hand_right`, `state.left_arm_joint_position`, `state.right_arm_joint_position`, `state.left_effector_position`, `state.right_effector_position`, `state.head_position`, `state.waist_position` | `action.left_arm_joint_position`, `action.right_arm_joint_position`, `action.left_effector_position`, `action.right_effector_position`, `action.head_position`, `action.waist_position`, `action.robot_velocity` | 绝对关节控制 |

## Advanced Tuning Parameters
## 高级调优参数

### Model Components
### 模型组件

The model has several components that can be fine-tuned independently. You can configure these parameters in the `GR00T_N1_5.from_pretrained` function.

模型有几个可以独立微调的组件。您可以在`GR00T_N1_5.from_pretrained`函数中配置这些参数。

1. **Visual Encoder** (`tune_visual`)
   **视觉编码器** (`tune_visual`)
   - Set to `true` if your data has visually different characteristics from the pre-training data
   - Note: This is computationally expensive
   - Default: false

   - 如果您的数据在视觉上与预训练数据有不同特征，则设置为`true`
   - 注意：这在计算上是昂贵的
   - 默认值：false


2. **Language Model** (`tune_llm`)
   **语言模型** (`tune_llm`)
   - Set to `true` only if you have domain-specific language that's very different from standard instructions
   - In most cases, this should be `false`
   - Default: false

   - 只有当您有与标准指令非常不同的特定领域语言时才设置为`true`
   - 在大多数情况下，这应该是`false`
   - 默认值：false

3. **Projector** (`tune_projector`)
   **投影器** (`tune_projector`)
   - By default, the projector is tuned
   - This helps align the embodiment-specific action and state spaces

   - 默认情况下，投影器会被调优
   - 这有助于对齐特定于具身的动作和状态空间

4. **Diffusion Model** (`tune_diffusion_model`)
   **扩散模型** (`tune_diffusion_model`)
   - By default, the diffusion model is not tuned
   - This is the action head shared by all embodiment projectors

   - 默认情况下，扩散模型不会被调优
   - 这是所有具身投影仪共享的动作头

### Understanding Data Transforms

This document explains the different types of transforms used in our data processing pipeline. There are four main categories of transforms:

This文档解释了我们数据处理管道中使用的不同类型的变换。有四种主要的变换类别：

#### 1. Video Transforms

Video transforms are applied to video data to prepare it for model training. Based on our experimental evaluation, the following combination of video transforms worked best:

视频变换用于准备模型训练的数据。根据我们的实验评估，以下视频变换组合效果最佳：

- **VideoToTensor**: Converts video data from its original format to PyTorch tensors for processing.
- **VideoCrop**: Crops the video frames, using a scale factor of 0.95 in random mode to introduce slight variations.
- **VideoResize**: Resizes video frames to a standard size (224x224 pixels) using linear interpolation.
- **VideoColorJitter**: Applies color augmentation by randomly adjusting brightness (±0.3), contrast (±0.4), saturation (±0.5), and hue (±0.08).
- **VideoToNumpy**: Converts the processed tensor back to NumPy arrays for further processing.

#### 2. State and ActionTransforms

State and action transforms process robot state and action information:

状态和动作变换处理机器人状态和动作信息：

- **StateActionToTensor**: Converts state and action data (like arm positions, hand configurations) to PyTorch tensors.
- **StateActionTransform**: Applies normalization to state and action data. There are different normalization modes depending on the modality key. Currently, we support three normalization modes:
  
  | Mode | Description | Formula | Range |
  |------|-------------|---------|--------|
  | `min_max` | Normalizes using min/max values | `2 * (x - min) / (max - min) - 1` | [-1, 1] | 
  | `q99` | Normalizes using 1st/99th percentiles | `2 * (x - q01) / (q99 - q01) - 1` | [-1, 1] (clipped) | 
  | `mean_std` | Normalizes using mean/std | `(x - mean) / std` | Unbounded | 
  | `binary` | Binary normalization | `1 if x > 0 else 0` | [0, 1] | 

#### 3. Concat Transform

The **ConcatTransform** combines processed data into unified arrays:

**ConcatTransform**将处理后的数据组合成统一数组：

- It concatenates video data according to the specified order of video modality keys.
- It concatenates state data according to the specified order of state modality keys.
- It concatenates action data according to the specified order of action modality keys.

This concatenation step is crucial as it prepares the data in the format expected by the model, ensuring that all modalities are properly aligned and ready for training or inference.

此拼接步骤至关重要，因为它准备的数据格式符合模型预期，确保所有模态对齐并准备好训练或推理。

#### 4. GR00T Transform

The **GR00TTransform** is a custom transform that prepares the data for the model. It is applied last in the data pipeline.

**GR00TTransform**是一个自定义变换，用于准备模型数据。它是在数据管道末尾应用的。

- It pads the data to the maximum length of the sequence in the batch.
- It creates a dictionary of the data with keys as the modality keys and values as the processed data.

In practice, you typically won't need to modify this transform much, if at all.

在实践中，您通常不需要修改此变换，除非必要。

### Lerobot Dataset Compatibility

More details about GR00T compatible lerobot datasets can be found in the [LeRobot_compatible_data_schema.md](./LeRobot_compatible_data_schema.md) file.

GR00T兼容的lerobot数据集的更多详情可以在[LeRobot_compatible_data_schema.md](./LeRobot_compatible_data_schema.md)文件中找到。
