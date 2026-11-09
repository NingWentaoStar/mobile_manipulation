# 强化学习

在玄关台上训练两个技能，物体是柠檬、梨、汤罐，旁边的盒子和罐子都在：

| 技能 | 任务 | 权重 |
|---|---|---|
| 拿起并移动 | 从台面拿起物体，移到空中的指定点并停稳 | `skills/policies/pick_move.pt` |
| 放下 | 手里握着物体，放到台面的指定位置，松手，抬手 | `skills/policies/place.pt` |

只依赖 `mujoco`、`numpy`、`torch`。物理在 CPU 上多线程并行，网络在 GPU 上（没有 GPU 时用 CPU）。

```bash
.venv/bin/python scripts/demo_rl_skills.py --object pear                 # 看效果：开到玄关台，拿起，放下
.venv/bin/python training/train.py pick --name pick                      # 训练，权重在 training/runs/pick/
.venv/bin/python training/train.py place --name place --pick-policy skills/policies/pick_move.pt
.venv/bin/python training/train.py evaluate pick skills/policies/pick_move.pt --episodes 500
.venv/bin/python training/train.py evaluate place skills/policies/place.pt --pick-policy skills/policies/pick_move.pt --episodes 500
```

放下的训练起点取自拿起技能成功时的真实状态，所以两个技能能直接衔接，也所以要先有拿起的权重。

## 任务

任务在 `skills/library/learned/tasks.py`，部署在 `common.py`、`pick_move.py` 和 `place.py`。

| 项目 | 内容 |
|---|---|
| 观测 | 45 维：关节位置和速度、是否夹住、夹爪位姿、目标点、物体位姿、两个相对位置、物体尺寸 |
| 动作 | 8 维：7 个关节增量（±0.1 rad）+ 夹爪 |
| 底盘 | 训练时固定在停靠位 `DOCK`，部署时开到这里，观测换算到训练坐标系 |
| 回合 | 最多 60 步 |
| 物体位置 | 随机放在台面的工作区内，朝向随机 |
| 拿起的成功 | 物体中心距目标不超过 2.5 cm，夹住，手臂基本静止 |
| 放下的成功 | 物体距目标不超过 2.5 cm 并停稳，手指张开，手抬到物体上方 7 cm |
| 失败 | 碰动旁边的物品超过 2 cm；放下时中途掉落 |

网络是 3 层 256 单元，PPO 超参数见 `train.py` 的默认值。

## 实测

Ubuntu 22.04，MuJoCo 3.5.0，RTX 5070 Ti Laptop，32 线程 CPU，一个训练种子。

| 技能 | 500 回合成功率 | 柠檬 | 梨 | 汤罐 |
|---|---|---|---|---|
| 拿起并移动 | 83.6% | 87.8% | 92.6% | 69.6% |
| 放下 | 86.6% | 86.3% | 89.6% | 83.4% |

训练速度约 2,800 步/秒，放下 600 万步约 40 分钟。拿起调了几轮奖励，权重来自多段接续训练，共约 1200 万步，一次从头训练的结果可能不同。

轮式底盘上的完整流程（导航、拿起、移动、放下）见 [skills/README.md](../skills/README.md#强化学习技能)，全流程柠檬 17/20、梨 16/20、汤罐 11/20。这比 `pick` 和 `place` 低，强化学习技能主要用于教学。

## 训练时踩过的坑

这些问题都不报错，只表现为成功率上不去。看奖励曲线常常看不出来，要回放策略，看它实际在做什么。

| 现象 | 原因 | 修法 |
|---|---|---|
| 奖励平了，成功率 0 | "夹住"的判定太松，策略把合拢的指尖压在物体顶上刷分 | 判定时要求物体在两指之间 |
| 成功率到 43% 后掉到 0，奖励还在涨 | 成功后的未来价值算成 0，悬停在目标旁边比成功更划算 | 成功结束时也用价值网络估计后续回报 |
| 夹得住，却抬不起来 | 目标都在 8 cm 以上，抬一点奖励几乎不变 | 目标高度下限降到 2 cm，形成难度梯度 |
| 物体被挤出去 | 手夹在物体上半部，圆物体一收紧就滑脱 | 加一项对"手正对物体中心"很敏感的奖励 |
| 拿着物体，移动得很慢 | 靠近目标的奖励太小，策略不敢动 | 加大靠近目标的权重，再加一项尖锐的奖励 |

奖励上升而成功率下降，通常说明奖励和目标不一致。

## 换成自己的任务

在 `tasks.py` 里继承 `ConsoleTask`，写 `reset_one`（初始状态）和 `score`（奖励、成功、失败）。`train.py` 只用到 `reset`、`observe`、`step`。部署时继承 `PolicySkill`，步骤和普通技能一样，见[新增一个技能](../skills/README.md#新增一个技能)。
