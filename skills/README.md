# 技能库

技能代码在 `mobile_panda_sim/skills/`，每个技能一个模块，用同一个入口调用：

```python
from mobile_panda_sim.mj.sim import MujocoScene
from mobile_panda_sim.skills import SkillLibrary

scene = MujocoScene("wheels")
skills = SkillLibrary(scene)
skills.reset((-2.95, -1.15, 3.1416))                       # 底盘停靠位 (x, y, yaw)

result = skills.run("pick", object="lemon")                # 返回 SkillResult
print(result.success, result.message, result.info)
if result:
    skills.run("place", target=(0.40, -0.60))              # 相对底盘：前 0.40 m、右 0.60 m
print(skills.list())                                       # 所有技能和它们的适用范围
```

```text
mobile_panda_sim/skills/
├── base.py            技能基类、SkillResult、SkillError
├── registry.py        SkillLibrary：按名字查找和运行
├── context.py         技能共用的：仿真、规划器、夹爪、手里的物体、基本动作
├── pick.py            pick
├── place.py           place
├── articulate.py      open、close，以及它们共用的拖动规划
├── base_pose.py       suggest_base_pose
├── reset_arm.py       reset_arm
├── navigate.py        navigate
└── learned/
    ├── tasks.py       训练任务
    ├── common.py      策略怎么在机器人上运行
    ├── pick_move.py   rl_pick_move
    └── place.py       rl_place
```

| 其他内容 | 位置 |
|---|---|
| 训练好的权重 | `skills/policies/` |
| 目录 | `skills/catalog.json` |
| 外部技能参考 | [EXTERNAL_SKILLS.md](EXTERNAL_SKILLS.md)、`external_skills.json` |
| 传统技能的原理 | [docs/manipulation.md](../docs/manipulation.md) |
| 训练方法 | [training/README.md](../training/README.md) |

要抓放物品、开关抽屉和柜门，先用传统方法技能。强化学习技能用来学习和对比训练方法。

## 新增一个技能

新建一个模块，写一个类：

```python
from .base import Skill, SkillError
from .registry import register


@register
class Wave(Skill):
    name = "wave"
    applies_to = "手臂周围没有障碍物时"
    not_for = "手里有东西时"
    preconditions = "底盘停稳"

    def execute(self, times=2):
        ctx = self.context            # ctx.planner、ctx.world、ctx.follow(...)、ctx.set_gripper(...)
        if ctx.held:
            raise SkillError("holding something")
        return dict(times=times)      # 成功时返回细节，失败时抛 SkillError
```

然后在 `mobile_panda_sim/skills/__init__.py` 里导入这个模块，在 `skills/catalog.json` 里加一条，就能 `skills.run("wave", times=3)`。`tests/test_catalog.py` 和 `tests/test_mujoco.py` 会检查目录和代码是否一致。

## 适用范围

选技能前先看这张表。所有技能都读仿真真值，不经过相机；执行时底盘必须停稳。

| 技能 | 能做 | 不能做 | 实测 |
|---|---|---|---|
| `navigate` | 空旷地面上开到目标位姿 `(x, y, yaw)`，yaw 可省略 | 需要绕开家具的路径；没有避障，只做直线反馈控制 | 默认起点到玄关台 / 台柜 / 冰箱 3/3 |
| `pick` | 从上方抓窄边小于 7.5 cm、放在够得着的台面上的物体 | 从侧面抓；抓抽屉、冰箱里的东西；台面靠墙一侧太远的物品 | 糖盒、梨、柠檬 24/24 |
| `place` | 把手中物体放到平整台面上 | 放进抽屉或冰箱；放到倾斜或很小的表面；圆的物体（如柠檬）松手后可能滚动几厘米 | 24/24 |
| `open` / `close` | 台柜上层左、中抽屉；冰箱上门开到约 35° 再关上 | 台柜右上和所有下层抽屉、冰箱下门、矮柜推拉门（把手低于约 0.55 m）；从 70° 关冰箱门 | 开 24/24，关 24/24 |
| `suggest_base_pose` | 选一个开和关都够得着的停靠位 | 不考虑底盘受力，假设底盘不动 | 两个抽屉和冰箱门，开 12/12，关 12/12 |
| `reset_arm` | 手臂沿无碰撞路径回到默认姿态 | 手臂被卡住时 | 每轮都用到 |
| `rl_pick_move` | 玄关台上的柠檬、梨、汤罐，底盘停在 `DOCK`，物体在工作区内 | 其他物体；其他台面；其他停靠位 | 500 回合 83.6%（柠檬 88%、梨 93%、汤罐 70%） |
| `rl_place` | 把这三种物体放到玄关台工作区内的指定位置 | 其他物体和其他台面 | 500 回合 86.6%（柠檬 86%、梨 90%、汤罐 83%） |

要抓住宅场景里的物品，用 `pick` 和 `place`，它们更通用，成功率也更高。强化学习技能的用途是教学和训练例程：它们只为这三种物体和这个台面训练，不要指望换物体也能用。

## 在 ROS 仿真里运行

技能也可以在已经启动的 ROS 仿真里执行，和该仿真共用同一个场景，不会另起一个仿真：

```bash
ros2 run mobile_panda_bridge run_skill navigate --param target='[-2.95, -1.15, 3.1416]'
ros2 run mobile_panda_bridge run_skill pick --param object=lemon
```

请求和结果分别是 `/mobile_panda/skill_request`、`/mobile_panda/skill_result`（JSON 字符串），详见 [docs/ros2_interface.md](../docs/ros2_interface.md#技能skill)。

## 例程

```bash
.venv/bin/python scripts/demo_manipulation.py all --viewer              # 传统方法：抓放物品、开关抽屉和冰箱门
.venv/bin/python scripts/demo_manipulation.py navigate --viewer         # 导航到玄关台，再抓取、放置
.venv/bin/python scripts/demo_rl_skills.py --object pear                # 强化学习：开到玄关台，拿起，移到指定位置，放下
.venv/bin/python scripts/demo_rl_skills.py --no-viewer --episodes 20
```

```python
from mobile_panda_sim.skills.learned.tasks import DOCK, TABLE_Z

skills.reset(DOCK)
if skills.run("rl_pick_move", object="pear", goal=(-3.40, -1.50, TABLE_Z + 0.15)):   # 目标是物体中心的世界坐标
    skills.run("rl_place", target=(-3.34, -1.56))                                    # 台面上的 (x, y)
```

轮式底盘从充电位出发，导航、拿起、移动、放下全流程的实测，每个物体 20 次，物体和目标位置随机：

| 物体 | 拿起并移动 | 全流程 | 碰动旁边物品的回合 |
|---|---|---|---|
| 柠檬 | 19/20 | 17/20 | 0 |
| 梨 | 19/20 | 16/20 | 1 |
| 汤罐 | 15/20 | 11/20 | 3 |

这是一个训练种子的结果。汤罐最不稳定，训练时成功率波动也较大。

## catalog.json 字段

- `id`：`skills.run()` 用的名字
- `kind`：`classical`（传统方法）或 `rl`（强化学习）
- `call`、`code`、`demo`、`docs`：怎么调用、代码、例程、说明
- `checkpoint`、`training`：权重和训练脚本，只有强化学习技能有
- `applies_to`、`not_for`、`preconditions`：能做的事、不能做的事、前提条件
- `validated`：实测结果、判据和运行环境

