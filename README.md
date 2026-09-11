# robot VLA client

机上推理客户端：从 ROS 取观测，经 WebSocket 调远端策略，再把动作发回机械臂。

## 布局

| 路径 | 作用 |
|---|---|
| `inference_xtrainer.py` | 入口：组观测、推理、切动作、串行/并行发送 |
| `data_pipeline/tools/` | ROS 客户端、观测/动作结构、消息转换 |
| `vla/third_party/openpi_client/` | OpenPI WebSocket + msgpack 协议 |
| `data_msgs/` 等 `*_msgs` | ROS 消息（需在有 ROS 的机器上使用） |
| `vendor/` | 本进程自带的 `websockets` / `msgpack` |
| `_bootstrap.py` | 仅把本目录和 `vendor` 加入当前进程 `sys.path` |

## 同步到模型仓库时拷这些

```
inference_xtrainer.py
vla/third_party/openpi_client/
data_pipeline/tools/robot_action.py
data_pipeline/tools/robot_observation.py
data_pipeline/tools/robot_ros_client.py
data_pipeline/tools/robot_message_conversion.py
```

不要拷 `vendor/`、`_bootstrap.py`、ROS `*_msgs`。模型环境自己装依赖；真机仍用本仓库完整目录。

## 运行

需要已 source 的 ROS 环境，以及可达的策略服务地址（改 `inference_xtrainer.py` 里的 `url`，或后续改为命令行参数）。

```bash
python3 inference_xtrainer.py --rate 10
python3 inference_xtrainer.py --parallel --rate 10
```
