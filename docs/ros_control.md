# xtrainer2：用 ROS 控制机械臂（新人入门）

一句话：**机器人大约 50 Hz 往 `/robot/observations` 广播当前状态；你大约 100 Hz 往 `/robot/actions` 持续发目标；节点 `xtrainer_main` 收到后驱动电机。**

`rostopic list` 能看见话题，只说明 ROS 总线通了，**不等于**手臂会动。

配套脚本：`tools/simple_control.py`（在 Noetic 容器里跑）。本仓库在宿主机是 `~/arianliu/client`，容器里必须用 **`/data/arianliu/client`**（`/home/x` 在容器里是空的 tmpfs）。

---

## 1. 先分清三套东西

| 东西 | 是什么 | 常见误用 |
|---|---|---|
| 有显示器的终端 + 本仓库 `scripts/run_collection.sh` | 起本体：`xtrainer_main`、相机、采集 GUI | Cursor 无头终端起 GUI 会把整栈杀掉；不要用别人家里的同名脚本 |
| 本仓库 `tools/simple_control.py` | 手写 `/robot/actions`，用来学 SDK | 不要在宿主机 Humble 里跑 rospy |
| `inference_xtrainer.py` | 连远端 VLA，再把动作发回臂 | 本体没稳之前不要上 |

容器一般叫 `master`，里面是 **ROS1 Noetic**。宿主机是 ROS2 Humble，**不能**在宿主机直接 `rostopic` 这套 `data_msgs`。

---

## 2. 控制回路长什么样

```
相机 / 臂 / 夹爪
        │
        ▼
  xtrainer_main          ──发布──▶  /robot/observations   ◀── 你的脚本读状态
        │
        └──订阅──  /robot/actions  ◀── 你的脚本 100 Hz 写目标
                        │
                        ▼
                   伺服 / 夹爪驱动
```

`xtrainer_main` 在 launch 里是 **`required="true"`**：它一崩（abort 或 segfault），roslaunch 会把 master 和其余节点全部收掉，看起来像“整台机器人被 kill”。

GUI 里还有一个服务 `/recording_command`（不是话题），用来切模式：

| `mode` | 日志里会出现 | 含义 |
|---|---|---|
| `server` | `Enable EE pose control mode` | 听 `/robot/actions` 的末端位姿 |
| `teleoperation` | `Disable EE pose control mode` | 听遥操作主臂，**忽略**你发的 EE 位姿 |

手写控制前，在采集 GUI 里启动**评测任务**，确认日志有 `req.mode=server`。切模式可以自己在 GUI 做，脚本不去调这个服务。

进了 server 之后，`xtrainer_main` 会每隔约 30 秒打：

```text
Waiting for commands from rosbag...
```

名字沿用了“播 rosbag”那条代码路径，实际意思是：**在等 `/robot/actions` 上的命令**。

---

## 3. 这几个话题是干什么的

### `/robot/observations`（读）

机器人的仪表盘，类型 `data_msgs/RobotObservation`，约 50 Hz。

里面按**固定顺序**有 4 个部件（不要改顺序去猜，以线上观测为准）：

| 下标 | `header.frame_id` | 末端位姿 | 关节 |
|---|---|---|---|
| 0 | `left_arm` | `TorsoEE`，7 个数：xyz + xyzw | 6 |
| 1 | `left_hand` | 空（没有自己的位姿） | 1（夹爪 0 合 / 1 开） |
| 2 | `right_arm` | 同左 | 6 |
| 3 | `right_hand` | 空 | 1 |

手臂控制用的是 **`multibody_pose`（末端位姿）**，不是 `multibody_state`（关节角）。

坐标系字面量是 **`TorsoEE`（两个大写 E）**。`TorsoEe` / `TorsoTool` 在 `xtrainer_main` 二进制里不存在。

### `/robot/actions`（写）

你的油门，类型 `data_msgs/RobotAction`。正常时**唯一订阅者**是 `/xtrainer_main`。

一条消息里是数组 `component_actions[]`。每一块：

- 部件名在 **`header.frame_id`**（`left_arm` / `left_hand` / …），不是消息最外层
- 手臂：填 `pose_command`（xyz + xyzw），并把 `pose_command.header.frame_id` 设为 `TorsoEE`
- 夹爪：填 `joint_commands = [0.0 ~ 1.0]`（驱动层再乘 255）
- 厂商推理是 **100 Hz 连续发同一条目标**；只 pub 一次很容易被丢掉

C++ 侧大致按观测的下标 `i` 去取 `component_actions[i]`，**不是**“按名字缺谁跳过谁”。安全做法：

**四个部件一次发齐，顺序与观测完全一致。**

夹爪那一块在 Python 里如果 `pose_command = None`，序列化出来的四元数是默认的 `(0,0,0,0)`。`xtrainer_main` 仍会调用 `QuaternionToRotationMatrix()`，全零会抛异常把进程打死。所以夹爪也要带上**对应手臂当前测到的位姿**（相当于原地保持）。

### `/robot/status/*`（读，辅助）

例如 `/robot/status/agent`。黄键没长按完时常见 `level: 3` / `initializing`。server 模式下动作走 `/robot/actions`，不依赖黄键；黄键起来后主臂会调 `/xtrainer/execute_command`。若此时 `xtrainer_main` 已经死了，日志会刷 `Failed to call service`——那是**崩完的连带现象**，不是死因。

### `rostopic list` vs `echo` / `pub`

| 命令 | 要不要 `data_msgs` |
|---|---|
| `rostopic list` | 不要，只问 master 要名字 |
| `rostopic echo` / `pub` | 要，必须能 `import data_msgs.msg` |

在容器里用纯 ROS CLI 时：

```bash
source /opt/ros/noetic/setup.bash
cd /data/robotics.install && source env.sh   # 必须在这个目录 source，脚本用了 $PWD
```

`tools/simple_control.py` **不需要** `env.sh`：`_bootstrap.py` 会把 `/data/robotics.install/lib/python3/dist-packages` 里的 `data_msgs` 链进当前进程。

---

## 4. 上手步骤（手写控制）

在**已经进入 Noetic 容器**、当前目录为 `/data/arianliu/client` 的终端：

**0. 本体**

有显示器的那台终端：

```bash
cd ~/arianliu/client
./scripts/run_collection.sh
```

不要加 `-i`，除非你确实要跑站内推理。GUI 里切到评测 / server，确认 glog 有 `Enable EE pose control mode`。

**1. 只读状态（不会动）**

```bash
python3 tools/simple_control.py state --debug
```

应打出左右臂 xyz、四元数、夹爪开度，以及 4 个部件的布局。

**2. 先检查将要发出的消息（仍不会动）**

```bash
python3 tools/simple_control.py gripper --left 0.0 --right 1.0 --hold-arms --debug --dry-run
```

`--debug` 会对照：全零四元数、部件名字/顺序、`joint_commands` 长度是否等于观测里的 dof。对不上会拒绝发送；要强行发才加 `--force`。

**3. 真发：合左爪、右爪保持开**

周围确认安全后：

```bash
python3 tools/simple_control.py gripper --left 0.0 --right 1.0 --hold-arms --seconds 2 --debug
```

张开左爪把 `--left 0.0` 改成 `1.0`。`--hold-arms` 会把没有夹爪指令的那一侧手臂当前位姿也带上，从而凑齐 4 块。

**4. 小幅平移末端（必须 `--yes`，单轴不超过 5 cm）**

```bash
python3 tools/simple_control.py move --arm left --dz 0.02 --yes --debug
```

公共参数写在**子命令后面**：`--seconds`、`--rate`、`--frame`、`--debug`。

---

## 5. 日志在哪

| 内容 | 路径 |
|---|---|
| `xtrainer_main` glog | `/data/roboticsx/log/xtrainer_main/master/<时间戳>/xtrainer_main.INFO` |
| 主臂 agent | `/data/roboticsx/log/xtrainer_agent_ros_main/master/<时间戳>/` |
| roslaunch | `/home/x/.ros/log/` |
| 屏幕上的 `terminate called` / `REQUIRED process has died` | 只在**有显示器的那个终端**，不一定落盘 |

看本次是否进了 server：

```bash
grep -E "req.mode|EE pose control" /data/roboticsx/log/xtrainer_main/master/*/xtrainer_main.INFO | tail
```

---

## 6. 我们已经踩过的坑（按优先级记）

1. **看见话题 ≠ 能控制**  
   `list` 不需要消息定义；`echo`/`pub` 需要。脚本能跑、`rostopic pub` 报 `invalid message type`，就是没 `source env.sh`。

2. **没进 server，夹爪完全不动**  
   teleop 会关掉 EE pose 控制。日志里要有 `Enable EE pose control mode`。

3. **只 pub 一次**  
   厂商是 100 Hz 流式发送。`tools/simple_control.py` 的 gripper/move 会按 `--rate` 持续发 `--seconds` 秒。

4. **全零四元数 → SIGABRT（signal 6）**  
   屏幕原文：`QuaternionToRotationMatrix(): All the elements in a quaternion are zero.`  
   只发 `left_hand` 且不填 pose 就会炸。夹爪必须带对应臂的实测位姿。

5. **frame 写成 `TorsoEe`**  
   观测和二进制都是 `TorsoEE`。

6. **只发一半部件 → SIGSEGV（signal 11）**  
   只发 `left_arm` + `left_hand` 时进程段错误。按观测顺序发齐 4 块后夹爪能动。`--debug` 会把顺序不一致直接拦下来。

7. **左夹爪回零失败（另一类死法）**  
   `J_L_GRIPPER ... Default position is not reached` 会写 glog **FATAL**，然后 required 节点退出。这是标定/回默认位，不是 action 格式问题。

8. **容器里 `TypeError: 'ABCMeta' object is not subscriptable`**
   `deps/vendor/` 里的 websockets 是给 Python 3.10+ 打的包，容器是 3.8。`_bootstrap.py`
   现在把它**追加**在 `sys.path` 末尾当兜底，不再盖住容器自带的可用版本。

9. **黄键 / inference 是旁支**  
   agent `initializing` 不等于 `/robot/actions` 没人听。  
   `inference_service` 自己还可能 `Not enough camera views` / `NoneType.copy`；它通常不是 required，不影响手写 action，但 GUI “部署模型”会失败。

10. **`--debug` 流式发送时订阅者突然变 0**  
    会打印 `lost its subscriber after N messages`：N=0 多半第一条就不合法；N>0 再对照屏幕上的 `what():`。

11. **三路相机 `initializing` / `No color image`**  
    USB 上还能看见三台 Gemini 305，但 `/robot/observations` 里 `chain_images` 为空。在**宿主机**复位：

    ```bash
    cd ~/arianliu/client
    ./scripts/usbreset.sh
    ./scripts/run_collection.sh -i
    ```

    复位后必须重新起本体，驱动才会重新打开设备。`lsusb -d 2bc5:0840` 找不到设备才是线/供电问题。

---

## 7. 和 VLA 推理的关系

`tools/simple_control.py` 学的是 **ROS 这一跳**。`inference_xtrainer.py` 只是在中间多了一跳 WebSocket：观测 → 策略服务 → 再填进同一套 `/robot/actions`（四个部件、`TorsoEE`、夹爪带手臂位姿、100 Hz 重复每个 10 Hz 步）。末端位姿约定、四元数 xyzw、夹爪 0/1，与这里相同；图像和 16 维 state 的合同见仓库根目录 `README.md`。

## 8. 上真模型之前：假策略服务端

想连真机、但还不想用真模型时，用 `tools/fake_policy_server.py`。它是一个**真正的
WebSocket 服务端**，说与 StarVLA 相同的 msgpack 协议，所以 `inference_xtrainer.py`
一行都不用改、也不知道对面是假的：观测采集、握手校验、组包、100 Hz 发布，整条
链路都被真实地跑了一遍。

它的“策略”极其简单：**双臂位姿原样返回（绝对保持不动），只让夹爪做一个很小的动作。**

### 步骤

**0. 安全前提**：夹爪里**不要夹着东西**（5% 的闭合作用在物体上就是挤压），手放在急停上。

**1. 确认本体在跑**

```bash
docker exec -it master bash
source /opt/ros/noetic/setup.bash
rosnode list | grep xtrainer_main
```

**2. 看当前夹爪开度**，好预判范围（子命令是 `state`）

```bash
cd /data/arianliu/client
python3 tools/simple_control.py state
```

`gripper=1.000` 表示全开，那么 `--delta 0.05` 会在 1.00 ↔ 0.95 之间来回。

**3. 在 GUI 里启动评测任务**，把机器人切到 **server** 模式。
不切就是 teleop，EE pose 控制是关的，**夹爪不会动**。日志里应出现
`Enable EE pose control mode`。

**4. 终端 A：起假服务端**

```bash
cd /data/arianliu/client
python3 tools/fake_policy_server.py --delta 0.05
```

**5. 终端 B（同一容器）：起客户端**

```bash
cd /data/arianliu/client
source /opt/ros/noetic/setup.bash
python3 inference_xtrainer.py --url ws://127.0.0.1:8000
```

**6. 对照两边输出**。服务端每次请求打印图像张数/尺寸和夹爪目标：

```
[fake] #1 3 views ['224x224x3'] lang='...' | left:1.000->0.950
[fake] #2 3 views ['224x224x3'] lang='...' | left:0.950->1.000
```

方向逐次翻转，且目标从**当次实测值**算起，所以锁在起始值 ±5% 内。
爪子实际动作应与打印一致。

**7. 停止**：两个终端各 Ctrl-C（先停客户端）。

### 参数

`--delta`（夹爪幅度，默认 0.05）、`--hands left|right|both`（默认 `left`）、
`--chunk`（每次返回多少步）、`--no-oscillate`、`--host` / `--port`。
`--delta` 不影响位姿，位姿永远原样返回。

### 这次演练验证不到什么

手臂指令值等于实测值，所以 `check_first_step` 的 0.15 m 跳变保护每次都是 0 距离、
必然通过——它只证明了协议、组包、发布这条链路，证明不了那道防线。
