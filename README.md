# robot VLA client

机上推理客户端：从 ROS 取观测，经 WebSocket 调远端 StarVLA 策略服务，再把动作发回机械臂。

## 布局

| 路径 | 作用 |
|---|---|
| `inference_xtrainer.py` | 入口：ROS 控制循环、串行/并行发送 |
| `vla/xtrainer2_contract.py` | 与服务端约定的数据布局（纯 numpy，可脱机测试） |
| `vla/xtrainer2_policy.py` | WebSocket 桥：组请求、解响应、安全检查（不依赖 ROS） |
| `data_pipeline/tools/` | ROS 客户端、观测/动作结构、消息转换 |
| `vla/third_party/openpi_client/` | OpenPI WebSocket + msgpack 协议 |
| `data_msgs/` 等 `*_msgs` | ROS 消息（需在有 ROS 的机器上使用） |
| `vendor/` | 本进程自带的 `websockets` / `msgpack` |
| `_bootstrap.py` | 仅把本目录和 `vendor` 加入当前进程 `sys.path` |
| `tests/` | 脱机测试：布局、相机匹配、握手校验、WebSocket 回环 |

本仓库以 git submodule 的形式被 StarVLA 引用，不需要把文件拷到模型仓库。

## 与服务端的约定

服务端负责全部归一化与四元数换序；客户端只发机器人自己的数值。

- 图像：3 路 `cam_high, cam_left_wrist, cam_right_wrist`，224×224，**HWC RGB uint8**
  （`cv_bridge` 给的是 BGR，本客户端会转成 RGB）
- 状态：16 维 `[左 xyz(3) + 四元数(4) + 夹爪(1), 右同]`，四元数为 **xyzw**，
  位姿取 `multibody_pose`（末端位姿），不是 `multibody_state`（关节角）
- 动作：服务端返回同样布局的 16 维绝对末端位姿，已反归一化、已是 xyzw
- 指令：必须用训练时的英文句子

连接时会用握手 metadata 校验以上各项，不匹配就直接报错退出。
细节见 StarVLA 仓库 `examples/Xtrainer2/README.md` 的 "Deployment contract"。

## 运行

完整步骤（GPU 服务端 + 本客户端、参数、上机顺序、常见失败）见 StarVLA 仓库
`examples/Xtrainer2/INFERENCE.md`。

需要已 source 的 ROS 环境，以及可达的策略服务地址。

```bash
python3 inference_xtrainer.py --url ws://<server-ip>:10093 --rate 10
python3 inference_xtrainer.py --url ws://<server-ip>:10093 --parallel --rate 10
```

上机前需要确认的两个值（脱机无法验证）：

- `--pose-frame`：默认 `TorsoEe`，另一个可选值是 `TorsoTool`
- 相机 frame id：默认按 `high/head`、`left`、`right` 关键字匹配 `chain_images`；
  匹配不出来会报错并列出实际 frame id，此时用
  `--cam-high / --cam-left-wrist / --cam-right-wrist` 显式指定（frame id 或下标）

`--max-first-step-jump`（默认 0.15 m）会在动作块第一步离当前位姿过远时拒绝下发，
用于兜住位姿坐标系、四元数顺序、模型选错这类错误。首次上机不要调大它。

## 脱机测试

```bash
python3 -m pytest tests/ -q
```

不需要 ROS，不需要 GPU；只需 `numpy`、`pytest`、`opencv-python`。
