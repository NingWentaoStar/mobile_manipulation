# 外部技能参考（Panda）

整理于 2026-11-03。机器可读版本：`external_skills.json`。
"已核对"表示打开过官方仓库或模型页确认；"未核对"表示只在搜索结果中看到。使用前请再看一次原仓库的许可证和说明。

这些项目都不在本仓库里，也没有接进本平台。表中的"训练仿真器"决定了迁移成本：
**抓取位姿模型**只依赖点云，换仿真器影响小；**策略类**的观测和控制器都和训练环境绑定，迁到 MuJoCo 通常要重新适配甚至重训。

## 1. 技能策略（直接输出动作）

| 项目 | 技能 | 训练仿真器 | 权重 | 许可 | 核对 | 备注 |
|---|---|---|---|---|---|---|
| [ManipGen](https://github.com/mihdalal/manipgen) | 抓取、放置、抓把手、打开、关闭 | IsaacGym | HF `minliu01/ManipGen` | MIT | 已核对 | 局部策略，要先把手移到目标附近；用腕部深度相机。训练约 300 块 A100 一周 |
| [IndustRealLib](https://github.com/NVlabs/industreallib) | 到位、抓销钉/齿轮、放置、销钉插孔、齿轮装配 | IsaacGym | 随仓库 | 见仓库 | 已核对 | 面向真机 frankapy，任务空间阻抗控制 |
| [InsertAnything](https://huggingface.co/tOMRmA/InsertAnything-checkpoints) | L 形孔、六边形孔插入 | Isaac Lab | HF，两个 `.pth`，观测 26 维 | BSD-3 | 已核对 | 只找到权重，没找到代码仓库 |
| [Neural MP](https://github.com/mihdalal/neuralmotionplanner) | 点云输入的避障到位 | 自建场景 | HF `mihdalal/NeuralMP` | 未确认 | 已核对 | 公开的是基础策略，不含测试时优化 |
| [MPiNets](https://github.com/NVlabs/motion-policy-networks) | 点云输入的避障运动 | — | [Zenodo](https://zenodo.org/records/8319949) `mpinets_hybrid_expert.ckpt` | MIT | 已核对 | |
| [robomimic 模型库](https://robomimic.github.io/docs/sources/model_zoo/robomimic_v0.1.md.txt) | Lift、Can、Square 等 | robosuite | 官方 `.pth` 直链 | 见仓库 | 未核对 | 需匹配 robomimic、robosuite 版本 |

## 2. 抓取位姿（输出 6D 夹爪位姿，执行还要 IK 和运动规划）

| 项目 | 功能 | 权重 | 许可 | 核对 | 备注 |
|---|---|---|---|---|---|
| [GraspGen](https://github.com/NVlabs/GraspGen) | 扩散模型生成抓取并评分 | HF `adithyamurali/GraspGenModels`，有 Panda 配置 | 见仓库 | 已核对 | 只支持 Panda、Robotiq-2f-140、吸盘；后续版本 [GraspGenX](https://github.com/NVlabs/GraspGenX) |
| [M2T2](https://github.com/NVlabs/M2T2) | 抓取位姿 + 放置位姿，有语言条件版本 | HF `wentao-yuan/m2t2` | 见仓库 | 已核对 | 能回答"放在哪" |
| [Contact-GraspNet](https://github.com/NVlabs/contact_graspnet) | 点云 6D 抓取 | Google Drive | 见 `License.pdf` | 已核对 | TF 2.2；有非官方 [PyTorch 移植](https://github.com/elchun/contact_graspnet_pytorch) |
| [VGN](https://github.com/ethz-asl/vgn) | 杂乱场景抓取（TSDF） | Google Drive，在 data 包里 | BSD-3 | 已核对 | PyBullet，最轻量 |
| [GIGA](https://github.com/UT-Austin-RPL/GIGA) | VGN 的改进版 | UT Box `data.zip`，含 GIGA 和 VGN | MIT | 已核对 | |
| [Edge-Grasp-Network](https://github.com/HaojHuang/Edge-Grasp-Network) | 6D 抓取检测 | 随仓库 | MIT | 已核对 | 训练用 Panda 夹爪 |
| [AnyGrasp](https://github.com/graspnet/anygrasp_sdk) | 通用抓取检测 | 按机器申请授权 | 闭源 | 已核对 | 不适合分发给学生 |
| TRG | — | — | — | 未找到 | 需要论文全名或 arXiv 号 |

本平台目前只发布 RGB 图像。要用这一类模型，需要在 `skills/mj/sim.py` 的 `capture_rgb()` 旁边再加深度输出（相机已经支持 `depth=True`），并在 ROS 端发布深度图或点云。

## 3. 运动规划

| 项目 | 说明 | 许可 |
|---|---|---|
| [cuRobo](https://github.com/NVlabs/curobo) | GPU 加速无碰撞运动生成，原生支持 Franka | Apache-2.0 |
| mplib | 本项目的运动规划层 | 已集成 |

## 4. 端到端 VLA 与多任务模型

| 项目 | 机器人 | 权重 | 核对 | 备注 |
|---|---|---|---|---|
| [RoboCasa365](https://github.com/robocasa/robocasa) | PandaOmron（移动底盘 Panda） | [LeRobot SmolVLA](https://huggingface.co/docs/lerobot/robocasa.md)；官方 π0.5、GR00T、DP 权重未核实 | 未核对 | 365 个厨房任务，代码 MIT、数据 CC-BY-4.0；跑在 robosuite（MuJoCo），不是 ManiSkill |
| [openpi](https://github.com/Physical-Intelligence/openpi) `pi05_droid` / `pi0_droid` | Franka（DROID） | GCS `gs://openpi-assets` | 未核对 | 需要大显存 GPU |
| OpenVLA-OFT | LIBERO 中的 Panda | HF `moojink/...` | 未核对 | 只在 LIBERO 上微调过 |

## 选型建议

1. **抓取**：GraspGen 或 M2T2 出位姿，cuRobo 或 mplib 规划，替代只能在训练范围内工作的 PickCube PPO。
2. **接触类技能**（推、开关、插入）：用 PPO 自己训练，见 [training/README.md](../training/README.md)，比迁移 IsaacGym 策略省事。
3. **RoboCasa** 适合当数据来源和对照，不适合直接接进本平台。
