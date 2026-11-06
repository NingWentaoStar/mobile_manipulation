# ROS2 接口

命名空间 `/mobile_panda`。坐标系 `map` 是仿真世界坐标系：
地面 z=0，+x 朝东、+y 朝北。

## 运行方式

```text
ros2 launch ──> mobile_panda_bridge（系统 Python，rclpy）
                   │  stdin/stdout，逐行 JSON
                   └─> mobile_panda_sim.ros2.worker（装了 mujoco 的 Python）
                robot_state_publisher（URDF + joint_states → TF）
```

- 启动后只加载场景、发布话题。**物理时间不推进**，直到收到控制命令。
- 每收到一条控制命令，仿真按实时速度推进，并在最后一条命令之后再推进 1 s（让机械臂到位、底盘停下），然后再次暂停。
- 控制周期 0.05 s（20 Hz）。状态、图像、位姿以 10 Hz 发布；暂停时继续发布最后一帧。

## 控制（订阅）

| 话题 | 类型 | 含义 | 限制 |
|---|---|---|---|
| `/mobile_panda/cmd_vel` | `geometry_msgs/Twist` | 底盘速度，只用 `linear.x`（m/s）和 `angular.z`（rad/s） | \|v\| ≤ 0.25，\|ω\| ≤ 0.7；其余分量必须为 0；**0.3 s 内没有新消息就停车** |
| `/mobile_panda/arm_joint_delta` | `std_msgs/Float64MultiArray` | 7 个关节增量 `panda_joint1..7`（rad） | 每个 \|Δ\| ≤ 0.1；每个控制周期最多执行一条，最新的覆盖旧的 |
| `/mobile_panda/gripper` | `std_msgs/Float64` | 每个手指的目标开口（m），保持到下次修改 | [-0.01, 0.04]；0.04 全开，0 闭合，负值用于夹紧细物体 |

超出限制的命令会被拒绝，原因发布在 `/mobile_panda/status`（`"type": "rejected"`），同时打印在终端。

**关节增量的含义**：
每一步的目标是"**当前实测**关节位置 + 增量"，由 PD 控制器跟踪；一个周期内跟不上的部分不会累计。
所以连续发送 N 条 Δ，实际转动通常小于 N·Δ（实测：20 条 0.02 rad 约转 0.09 rad）。需要到达指定角度时，按 `joint_states` 闭环控制。

底盘：

- `base:=wheels`（默认）：真实差速驱动，左右轮速 `(v ∓ ω·L/2)/r`，轮半径 r=0.10 m，轮距 L=0.58 m。
  轮胎与地面靠摩擦行驶，有打滑和动态误差（实测 0.15 m/s 持续 2 s 前进约 0.29–0.32 m）。
- `base:=planar`：根固定在世界坐标系，底盘由虚拟的 x/y/yaw 关节驱动，同样只接受 `(v, ω)`，没有打滑。

## 感知与状态（发布）

| 话题 | 类型 | 内容 |
|---|---|---|
| `/mobile_panda/base_pose` | `geometry_msgs/PoseStamped` | `chassis_link` 在 `map` 中的真值位姿 |
| `/mobile_panda/base_camera/rgb` | `sensor_msgs/Image`（rgb8，640×480） | 固定在机械臂安装板上、朝向前方工作区的相机 |
| `/mobile_panda/hand_camera/rgb` | `sensor_msgs/Image`（rgb8，320×240） | 腕部相机（RealSense 的安装位置） |
| `/mobile_panda/{base,hand}_camera/camera_info` | `sensor_msgs/CameraInfo` | 内参，无畸变；`frame_id` 为 `<相机>_optical_frame` |
| `/mobile_panda/joint_states` | `sensor_msgs/JointState` | 全部主动关节（轮子/虚拟底盘关节 + 7 个臂关节 + 2 个手指） |
| `/tf` | | `map → chassis_link`（wheels）或 `map → world_root`（planar），其余由 robot_state_publisher 根据 URDF 发布 |
| `/robot_description` | `std_msgs/String` | 与仿真加载的同一份 URDF |
| `/mobile_panda/state` | `std_msgs/String`（JSON） | 以上信息汇总，外加夹爪中心 `tcp_pose`、手指开口、步数、仿真时间、`running`；`objects`：所有可移动物品的真值位姿；`articulations`：每件家具各关节（抽屉、门）的开合位置 |
| `/mobile_panda/scene_info` | `std_msgs/String`（JSON，latched） | 静态布局：房间范围、起点，每件家具和物品的占地范围（`footprint`：xmin, xmax, ymin, ymax）和高度 |
| `/mobile_panda/status` | `std_msgs/String`（JSON，latched） | `starting`、`ready`、`rejected`、`stopped`、`reset_complete`、`fatal` |

`/mobile_panda/state` 里的物品位姿和关节位置是仿真真值，方便调试和评估。
如果作业要求用视觉感知，请在任务书里说明只允许使用相机话题。

相机坐标系：URDF 中 `<相机>_link` 的约定是（x 向前、y 向左、z 向上），仿真相机就挂在这个坐标系上；
`<相机>_optical_frame` 是 ROS 光学约定（z 向前、x 向右、y 向下），图像的 `frame_id` 用的是它。

## 服务

| 服务 | 类型 | 作用 |
|---|---|---|
| `/mobile_panda/stop` | `std_srvs/Trigger` | 清除速度和关节增量命令（保持夹爪目标） |
| `/mobile_panda/reset` | `std_srvs/Trigger` | 重置场景：物品回到初始位置，抽屉和门关上，机器人回到起点 |

## 例子

```bash
# 前进（需要持续发布，否则 0.3 s 后停车）
ros2 topic pub -r 10 /mobile_panda/cmd_vel geometry_msgs/Twist "{linear: {x: 0.15}}"
# 第 1 个关节转一点
ros2 topic pub --once /mobile_panda/arm_joint_delta std_msgs/Float64MultiArray "{data: [0.05, 0, 0, 0, 0, 0, 0]}"
# 闭合 / 张开夹爪
ros2 topic pub --once /mobile_panda/gripper std_msgs/Float64 "{data: 0.0}"
ros2 topic pub --once /mobile_panda/gripper std_msgs/Float64 "{data: 0.04}"
# 当前底盘位姿、物品位置和抽屉 / 门的开合
ros2 topic echo --once /mobile_panda/base_pose
ros2 topic echo --once /mobile_panda/state --field data
ros2 service call /mobile_panda/reset std_srvs/srv/Trigger
```

完整的 Python 示例：`ros2_ws/src/mobile_panda_bridge/mobile_panda_bridge/client.py`（`ros2 run mobile_panda_bridge teleop_check`）。
