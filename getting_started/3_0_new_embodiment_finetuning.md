# New Embodiment Finetuning Tutorial
# 新具身微调教程

This notebook is a tutorial on how to finetune GR00T-N1 pretrained model on a new dataset. 
这个笔记本是关于如何在新数据集上微调GR00T-N1预训练模型的教程。

# Lerobot So100/So101 Finetuning Tutorial
# Lerobot So100/So101微调教程

GR00T-N1.5 is accessible to everyone with various robot form-factors. Based on Huggingface's low-cost [So101 Lerobot arm](https://huggingface.co/docs/lerobot/so101), users can finetune GR00T-N1.5 on their own robot via a `new_embodiment` tag.

GR00T-N1.5对具有各种机器人形态的每个人都是可访问的。基于Huggingface的低成本[So101 Lerobot机械臂](https://huggingface.co/docs/lerobot/so101)，用户可以通过`new_embodiment`标签在自己的机器人上微调GR00T-N1.5。

| So100 Strawberry and Grape Picking | So101 Table Cleanup Task |
| So100草莓和葡萄采摘 | So101桌面清理任务 |
|----------------------|---------------------|
| ![so100_eval_demo](../media/so100_eval_demo.gif){width=400} | ![so101-table-cleanup](../media/so101-pens-pickup.gif){width=400} |


| Dataset | Observation | Viz Link |
| 数据集 | 观察 | 可视化链接 |
|---------|------------|------|
| so101-table-cleanup | Dual camera views of table cleanup task | [Link](https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=youliangtan%2Fso101-table-cleanup) |
| so101-table-cleanup | 桌面清理任务的双摄像头视图 | [链接](https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=youliangtan%2Fso101-table-cleanup) |
| so100_strawberry_grape | Single camera view of strawberry and grape picking | [Link](https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=youliangtan%2Fso100_strawberry_grape&episode=0) |
| so100_strawberry_grape | 草莓和葡萄采摘的单摄像头视图 | [链接](https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=youliangtan%2Fso100_strawberry_grape&episode=0) |
| tictac-bot | Single camera view of a tic-tac-toe board | [Link](https://huggingface.co/datasets/youliangtan/tictac-bot) |
| tictac-bot | 井字棋棋盘的单摄像头视图 | [链接](https://huggingface.co/datasets/youliangtan/tictac-bot) |

## Step 1: Dataset
## 步骤1：数据集

Users can use any lerobot dataset for finetuning. In this tutorial, we will first use a sample dataset: [so101-table-cleanup](https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=youliangtan%2Fso101-table-cleanup&episode=0)

用户可以使用任何lerobot数据集进行微调。在本教程中，我们将首先使用一个示例数据集：[so101-table-cleanup](https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=youliangtan%2Fso101-table-cleanup&episode=0)

Note that this embodiment was not used in our pretraining dataset mixture.
请注意，这个具身未在我们的预训练数据集混合中使用。


### First, download the dataset
### 首先，下载数据集

```bash
huggingface-cli download \
    --repo-type dataset youliangtan/so101-table-cleanup \
    --local-dir ./demo_data/so101-table-cleanup
```

### Second, copy over the modality file
### 其次，复制模态文件

The `modality.json` file provides additional information about the state and action modalities to make it "GR00T-compatible". Copy over the `examples/so100_dualcam__modality.json` to the dataset `<DATASET_PATH>/meta/modality.json`.

`modality.json`文件提供了关于状态和动作模态的额外信息，使其"GR00T兼容"。将`examples/so100_dualcam__modality.json`复制到数据集`<DATASET_PATH>/meta/modality.json`。


For Dual Camera setup like [so101-table-cleanup](https://huggingface.co/datasets/youliangtan/so101-table-cleanup) dataset, do:
对于像[so101-table-cleanup](https://huggingface.co/datasets/youliangtan/so101-table-cleanup)数据集这样的双摄像头设置，执行：
```bash
cp examples/so100_dualcam__modality.json ./demo_data/so101-table-cleanup/meta/modality.json
```

For Single Camera setup like [so100_strawberry_grape](https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=youliangtan%2Fso100_strawberry_grape&episode=0) dataset, do:
对于像[so100_strawberry_grape](https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=youliangtan%2Fso100_strawberry_grape&episode=0)数据集这样的单摄像头设置，执行：
```bash
cp examples/so100__modality.json ./demo_data/so100_strawberry_grape/meta/modality.json
```

Then we can load the dataset using the `LeRobotSingleDataset` class.
然后我们可以使用`LeRobotSingleDataset`类加载数据集。


## Step 2: Finetuning

Finetuning can be done by using our finetuning `scripts/gr00t_finetune.py`, as a "new-embodiment" tag is supported.

微调可以通过使用我们的微调`scripts/gr00t_finetune.py`来完成，因为"new-embodiment"标签是受支持的。

```bash
python scripts/gr00t_finetune.py \
   --dataset-path /datasets/so101-table-cleanup/ \
   --num-gpus 1 \
   --batch-size 64 \
   --output-dir ~/so101-checkpoints  \
   --max-steps 10000 \
   --data-config so100_dualcam \
   --video-backend torchvision_av
```

> adjust the batch size to your GPU memory.

## Step 3: Open-loop evaluation

Once the training is done, you can run the following command to visualize the finetuned policy. 

训练完成后，您可以运行以下命令来可视化微调后的策略。

```bash
python scripts/eval_policy.py --plot \
   --embodiment_tag new_embodiment \
   --model_path <YOUR_CHECKPOINT_PATH> \
   --data_config so100_dualcam \
  --dataset_path /datasets/so101-table-cleanup/ \
   --video_backend torchvision_av \
   --modality_keys single_arm gripper
```

This is a plot after training the policy for 7k steps.

这是一个训练策略7k步后的图。

![so100-7k-steps.png](../media/so100-7k-steps.png)

After training for more steps the plot will look significantly better.

训练更多步骤后，图会显著改善。

TADA! You have successfully finetuned GR00T-N1.5 on a new embodiment.

TADA! 您已成功在新具身上微调GR00T-N1.5。

## Step 4: Deployment

First, make sure the data is replay-able, refer to the lerobot doc: https://huggingface.co/docs/lerobot/so101

首先，确保数据是可重放的，请参考lerobot文档：https://huggingface.co/docs/lerobot/so101

Evaluate the policy on the robot:
评估策略在机器人上的表现：
```bash
python eval_lerobot.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 9, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 15, width: 640, height: 480, fps: 30}}" \
    --policy_host=10.112.209.136 \
    --lang_instruction="Grab pens and place into pen holder."
```

For more details about deployment, please refer to the notebook: `5_policy_deployment.md`

有关部署的更多详细信息，请参考笔记本：`5_policy_deployment.md`

---

# Unitree G1 Finetuning

This section shows how to finetune on a Unitree G1 robot as a new embodiment. Dataset is available at: [nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1)

这个部分展示了如何在新具身上微调Unitree G1机器人。数据集可在：[nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1)

**Specifications**:
**规格**：
- Observation: 43 dim of vectorized state (joint positions of full body + hands)
- 观察：43维向量化的状态（全身关节位置+手）
- Action: 43 dim of vectorized action (joint positions of full body + hands)
- 动作：43维向量化的动作（全身关节位置+手）
- Video: RGB video, 640x480 resolution, 20fps
- 视频：RGB视频，640x480分辨率，20fps
- Language Instruction:
  - "Pick the apple from the table and place it into the basket."
  - "Pick the pear from the table and place it into the basket."
  - "Pick the grapes from the table and place them into the basket."
  - "Pick the starfruit from the table and place it into the basket."

## Step 1: Download the dataset

```bash
huggingface-cli download \
    --repo-type dataset nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1 \
    --local-dir ./datasets/
```

## Step 2: Try loading the dataset and visualize it

Example: load the first episode of the apple dataset

```bash
python scripts/load_dataset.py --dataset-path datasets/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-apple/ --plot-state-action

### Similar for other fruits
# Switch to other fruits -- pear, grapes, starfruit
```

You should see the following plots:
<div style="display: flex; justify-content: space-between;">
    <img src="../media/g1-pick-apple-images.png" width="48%">
    <img src="../media/g1-pick-apple-state-action.png" width="48%">
</div>


## Step 3: Finetuning

Here we can provide a list of datasets to finetune on. We will finetune the model with all the datasets which have a mix of `apple`, `pear`, `grapes`, `starfruit` picking tasks.

```bash
dataset_list=(
    "datasets/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-apple/"
    "datasets/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-pear/"
    "datasets/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-grapes/"
    "datasets/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-starfruit/"
)

python scripts/gr00t_finetune.py \
    --dataset-path ${dataset_list[@]} \
    --num-gpus 1 --batch-size 95  --output-dir ~/checkpoints/full-g1-mix-fruits/  \
    --data-config unitree_g1 --max-steps 15000
```

> NOTE: Since the dataset is recorded with h264 codec, we need to use the `decord` backend for video loading.

## Step 4: Open-loop evaluation

Example: evaluate the apple dataset

```bash
python scripts/eval_policy.py --plot \
   --embodiment_tag new_embodiment \
   --model_path <YOUR_CHECKPOINT_PATH> \
   --data_config unitree_g1 \
   --dataset_path datasets/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-apple/ \
   --video_backend decord \
   --modality_keys left_arm right_arm
```

![g1-pick-apple-eval](../media/g1-eval-pick-apple-states.png)


We can see that the policy's prediction of the action seems to be close to the ground truth, which indicates that the finetuning is successful. However for actual performance of the policy, we need to evaluate it on the real robot.

我们可以看到策略的动作预测似乎接近真实值，这表明微调是成功的。然而，为了评估策略的实际性能，我们需要在真实机器人上进行评估。

## Step 5: Deployment

The deployment script for G1 is not provided here. But this pipeline is similar to the so100 finetuning pipeline.

G1的部署脚本未在此提供。但此管道与so100微调管道类似。
