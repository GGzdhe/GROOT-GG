## Policy Deployment
## 策略部署

> This tutorial requires user to have a trained model checkpoint and a physical So100 Lerobot robot to run the policy.
> 
> 本教程要求用户拥有训练好的模型检查点和物理So100 Lerobot机器人来运行策略。

In this tutorial session, we will show example scripts and code snippets to deploy a trained policy. We will use the So100 Lerobot arm as an example.

在本教程会话中，我们将展示部署训练好的策略的示例脚本和代码片段。我们将使用So100 Lerobot机械臂作为示例。

![alt text](../media/so100_eval_demo.gif)

### 1. Load the policy
### 1. 加载策略

Run the following command to start the policy server.
运行以下命令启动策略服务器。

```bash
python scripts/inference_service.py --server \
    --model_path <PATH_TO_YOUR_CHECKPOINT> \
    --embodiment-tag new_embodiment \
    --data-config so100_dualcam \
    --denoising-steps 4
```

 - Model path is the path to the checkpoint to use for the policy, user should provide the path to the checkpoint after finetuning
 - Denoising steps is the number of denoising steps to use for the policy, we noticed that having a denoising step of 4 is on par with 16
 - Embodiment tag is the tag of the embodiment to use for the policy, user should use new_embodiment when finetuning on a new robot
 - Data config is the data config to use for the policy. Users should use `so100`. If you want to use a different robot, implement your own `ModalityConfig` and `TransformConfig`

 - 模型路径是策略使用的检查点路径，用户应提供微调后的检查点路径
 - 去噪步数是策略使用的去噪步数，我们注意到去噪步数为4与16相当
 - 具身标签是策略使用的具身标签，用户在新机器人上微调时应使用new_embodiment
 - 数据配置是策略使用的数据配置。用户应使用`so100`。如果您想使用不同的机器人，请实现自己的`ModalityConfig`和`TransformConfig`

### 2. Client node
### 2. 客户端节点

To deploy the finetuned model, you can use the `scripts/inference_policy.py` script. This script will start a policy server.
要部署微调后的模型，您可以使用`scripts/inference_policy.py`脚本。此脚本将启动策略服务器。

The client node can be implemented using the `from gr00t.eval.service import ExternalRobotInferenceClient` class. This class is a standalone client-server class that can be used to communicate with the policy server, with a `get_action()` endpoint as the only interface. 

客户端节点可以使用`from gr00t.eval.service import ExternalRobotInferenceClient`类来实现。这个类是一个独立的客户端-服务器类，可以用来与策略服务器通信，以`get_action()`端点作为唯一接口。

```python
from gr00t.eval.service import ExternalRobotInferenceClient
from typing import Dict, Any

raw_obs_dict: Dict[str, Any] = {} # fill in the blanks
                                 # 填写空白

policy = ExternalRobotInferenceClient(host="localhost", port=5555)
raw_action_chunk: Dict[str, Any] = policy.get_action(raw_obs_dict)
```

User can just copy the class and implement their own client node in a separate isolated environment.
用户可以直接复制该类并在单独的隔离环境中实现自己的客户端节点。

### Example with So100/So101 Lerobot arm
### So100/So101 Lerobot机械臂示例

We provide a sample client node implementation for the So100 Lerobot arm. Please refer to the example script `scripts/eval_lerobot.py` for more details.
我们为So100 Lerobot机械臂提供了示例客户端节点实现。更多详情请参考示例脚本`scripts/eval_lerobot.py`。


User can run the following command to start the client node. This example demonstrate with 2 cameras:
用户可以运行以下命令启动客户端节点。此示例演示了使用2个摄像头：
```bash
python eval_lerobot.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 9, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 15, width: 640, height: 480, fps: 30}}" \
    --policy_host=10.112.209.136 \
    --lang_instruction="Grab pens and place into pen holder."
```

For task that uses single camera, change the `--robot.cameras` to:
```bash
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 9, width: 640, height: 480, fps: 30}}}" \
```

Change the language instruction to the task you want to perform by changing the `--lang_instruction` argument.

This will activate the robot, and call the `action = get_action(obs)` endpoint of the policy server to get the action, then execute the action on the robot.
