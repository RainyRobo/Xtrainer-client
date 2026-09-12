# robot VLA client

机上推理客户端：从 ROS 取观测，经 WebSocket 调远端 StarVLA 策略服务，再把动作发回机械臂。

## 布局

```
inference_xtrainer.py   正式推理入口
_bootstrap.py           进程内 sys.path 设置
tools/                  辅助脚本（预检、手写控制、假服务端、看相机）
scripts/                起本体：run_collection.sh + dobot 配置快照
vla/                    与服务端的数据约定、组包、WebSocket 桥
deps/                   不是我们写的：ROS 消息包、data_pipeline、vendor
tests/                  脱机测试
docs/                   ROS 控制入门与踩坑记录
```

| 路径 | 作用 |
|---|---|
| `inference_xtrainer.py` | 入口：ROS 控制循环、串行/并行发送 |
| `tools/probe_xtrainer.py` | 上机前预检：模拟/真机观测、可选一次推理，不发动作 |
| `tools/simple_control.py` | 手写 `/robot/actions`：读状态、夹爪、小幅移臂（学 SDK 用） |
| `tools/fake_policy_server.py` | 假策略服务端：真 WebSocket 协议，只做微小夹爪动作 |
| `tools/view_cameras.py` | 实时看三路 chain 相机（需要 DISPLAY） |
| `scripts/run_collection.sh` | **宿主机桌面**起本体（GUI + `xtrainer_main`），不依赖别人的 checkout |
| `scripts/usbreset.sh` | 宿主机 USB 复位三台 Orbbec Gemini 305；复位后要重新起本体 |
| `scripts/dobot_config/` | 本机 dobot 配置快照；启动时写入 `/data/robotics.install` |
| `vla/xtrainer2_contract.py` | 16 维布局、相机顺序、握手校验（纯 numpy） |
| `vla/xtrainer2_actions.py` | 16 维 → 四部件 `RobotAction` 组包 |
| `vla/xtrainer2_policy.py` | WebSocket 桥：组请求、解响应、首步跳变保护 |
| `deps/data_pipeline/` | 厂商 ROS 客户端、观测/动作结构、消息转换 |
| `deps/*_msgs/` | ROS 消息包；容器里会被 `.live_ros_msgs` 链到安装树 |
| `deps/vendor/` | 兜底的 `websockets` / `msgpack`（容器 3.8 用系统自带版本） |
| `docs/ros_control.md` | 新人：ROS 话题、消息格式、上手命令、已踩过的坑 |

`deps/` 里的包仍按自己的顶层名导入（`import data_msgs`、`import data_pipeline...`）：
`_bootstrap.py` 把 `deps/` 本身加进 `sys.path`，所以目录可以收起来而不改任何 import。

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

## 这台机上的工作流（xtrainer2）

三套东西不要混：

| 东西 | 是什么 | 不是什么 |
|---|---|---|
| 本仓库 `scripts/run_collection.sh` | 机械臂 / 相机 / `/robot/observations` 的 **ROS1 Noetic 运行时** | 不是 StarVLA；不要再用别人家里的同名脚本 |
| `run_inference_sdk.sh` | 站内旧 HyVLA/π0 推理 | 不要用来测本仓库 |
| 本仓库 `tools/probe_xtrainer.py` / `inference_xtrainer.py` | StarVLA 的机上 client | 宿主机 Humble 里跑不了 rospy |

Docker 和机械臂一起升。容器把 **`/data` 挂进去**，把 **`/home/x` 盖成空 tmpfs**，所以 client 必须在 `/data/...`，不能只放在 `~/arianliu/client`。

当前拷贝位置：`/data/arianliu/client`。

### 终端怎么分

**终端 A（有桌面 DISPLAY，起本体）**

```bash
cd ~/arianliu/client
./scripts/run_collection.sh
```

不要加 `-i`（那会顺带拉起站内 `inference_service`）。需要同步安装包时再显式加 `--sync`。

`collect_agent` 要 Tk 窗口；Cursor 终端没有 DISPLAY 时 GUI 挂掉，launch 里它是 `required`，会把整组（含 master）一起杀掉。必须在本机桌面会话里起。

手写控臂、话题含义和已踩过的坑见 **[docs/ros_control.md](docs/ros_control.md)**。GUI 需切到评测 / `server` 模式（日志出现 `Enable EE pose control mode`）。夹爪命令要按观测顺序一次发齐四个部件，frame 用 `TorsoEE`。

**终端 B（还是那个 Noetic 容器，跑本 client）**

```bash
cd /data/arianliu/client
python3 tools/view_cameras.py            # 实时看三路相机，窗口里按 q 退出
python3 tools/simple_control.py state --debug
python3 tools/simple_control.py gripper --left 0.0 --right 1.0 --hold-arms --debug --dry-run
python3 tools/simple_control.py gripper --left 0.0 --right 1.0 --hold-arms --seconds 2 --debug
python3 tools/probe_xtrainer.py --robot
python3 tools/probe_xtrainer.py --robot --url ws://<starvla-ip>:10093
python3 inference_xtrainer.py --url ws://<starvla-ip>:10093 --rate 10 --send-rate 100
```

宿主机改了 `~/arianliu/client` 之后，同步再进容器：

```bash
rsync -a --exclude .git ~/arianliu/client/ /data/arianliu/client/
# 或在宿主机直接 python3 tools/probe_xtrainer.py --robot ，它会 rsync 并 docker exec 进已有 master
```

### 预检看什么

- `chain images` 应是 `head_orbbec / left_hand_orbbec / right_hand_orbbec`
- 16 维 state 来自 `multibody_pose` + 夹爪，不是 6 维关节角
- 夹爪是否真听命令，用 `tools/simple_control.py gripper ... --hold-arms`（见 `docs/ros_control.md`），不要只用 `probe --send-gripper`（缺部件/缺位姿会把 `xtrainer_main` 打挂）
- `--url` 只握手 + 推一帧，**不发动作**；真正控臂才是 `inference_xtrainer.py`

`--pose-frame` 默认应与观测一致：`TorsoEE`。`--max-first-step-jump` 默认 0.15 m，首次上机不要调大。

GPU 服务端、参数和常见失败见 StarVLA 仓库 `examples/Xtrainer2/INFERENCE.md`。

## 脱机测试

```bash
python3 -m pytest tests/ -q
```

不需要 ROS，不需要 GPU；只需 `numpy`、`pytest`、`opencv-python`。
