# DAMA-DDPG 观测延迟适配版

本实现保留 MADDPG 和动作序列扩展输入。训练使用当前观测，测试冻结同一模型并施加观测延迟；动作立即执行。它是对论文的适配，不是原论文动作延迟实验的原样复现。

参考：[论文](https://arxiv.org/abs/2005.05441)、[作者代码](https://github.com/baimingc/delay-aware-MARL)。原方法将待执行动作队列加入 actor 输入，集中式 critic 使用所有智能体的扩展观测和联合动作。这里将队列转换为已执行的本地动作历史，没有新增动力学预测器。

## 输入与时间关系

每个 actor 独立，使用两层 ReLU MLP，不使用 RNN。输入按以下顺序拼接：

1. 当前已经交付的观测，维度 `obs_dim`。
2. 最近 H 个本地已执行动作的 one-hot，按最旧到最新排列，维度 `H * n_actions`。
3. 历史位置有效标记，维度 H；episode 开头不足的历史补零。
4. 观测生成之后的动作标记，维度 H。
5. 观测是否已经到达，维度 1。
6. 智能体 one-hot ID（可关），维度 `n_agents`。

令当前决策时刻为 t，已收到观测的生成时刻为 τ，动作位置 s 的对齐标记为 `valid(s) and τ >= 0 and s >= τ`。观测在一步开始时生成，动作在该步随后执行，因此包含 s=τ。输入动作严格来自 `[t-H, t)`，不读取当前动作或未来观测。每个 actor 只能读取自己的观测和动作历史；critic 可以读取所有智能体扩展输入与联合动作，不额外读取 batch 的真实全局 state。

例如 t=10、τ=7、H=4，输入动作是 `[a6,a7,a8,a9]`，对齐标记为 `[0,1,1,1]`。训练时 τ=t，完整动作历史仍参与训练，对齐标记为零。没有任何观测到达时 τ=-1、观测补零、到达标记为零。

默认 H=16，训练后不会随测试延迟改变网络维度。超过 H 的观测年龄只保留最近 H 步历史，评估输出 `history_overflow_fraction`。`obs_delay` 是交付观测的当前年龄，不是每份观测最初采样的传输时间；本算法用 `obs_gen_t` 进行对齐。

**适配限制：** 无延迟训练不能教会网络如何处理非零对齐标记；测试有分布变化。动作历史并不保证多智能体局部观测恢复 Markov 性质，也不保证延迟鲁棒性。需要以下消融与实测验证，不能把短训练当作收敛证据。

## 学习与模型选择

- 独立 actor 和独立集中式 critic；合作奖励展开到各智能体，竞争奖励分别保留。
- 离散训练动作使用 straight-through Gumbel-Softmax；评估与目标动作使用 argmax。全部路径应用可用动作掩码。
- 一步 TD，终止步奖励参与训练，真正终止时关闭 bootstrap；时间截断保留 bootstrap。无效回放补齐位置不参与损失。
- actor i 只通过 critic i 获得梯度，其他 actor 的动作在该目标内 detach。
- 软更新目标网络；保存 actor、critic、两个目标网络、optimizer 和更新计数。
- learner 默认拒绝包含旧观测的训练 batch；不进行延迟数据增强。
- 训练脚本使用普通 `sc2` / `mpe`。因此周期性验证也是无延迟的，best_model 不根据延迟测试选择。
- SMAC 根据现有胜率指标选模，spread 根据共享回报。tag 双方都训练，默认选择捕食者索引 `[0,1,2]` 的平均回报选模，双方回报分别报告。`dama_parallel` 仅扩展日志选模指标，复用公共 ParallelRunner 的交互逻辑。

超参数为移植起点，不是原论文结果的已验证最优设置。学习率 5e-4，gamma=0.99，tau=0.01，隐藏层128；SMAC回放500个episode，MPE脚本覆盖为2000个episode。MPE固定步长25；地图训练预算和场景参数都在脚本中。

## 训练

只有一份算法配置：`src/config/algs/dama_ddpg.yaml`。Linux/集群上激活环境后，从任意目录运行：

```bash
bash scripts/train_scripts/dama_ddpg_5m_vs_6m.sh
bash scripts/train_scripts/dama_ddpg_8m_vs_9m.sh
bash scripts/train_scripts/dama_ddpg_mmm2.sh
bash scripts/train_scripts/dama_ddpg_spread.sh
bash scripts/train_scripts/dama_ddpg_tag.sh
```

脚本支持 `PYTHON`、`SEED` 环境变量和末尾 Sacred 参数覆盖：

```bash
SEED=2 bash scripts/train_scripts/dama_ddpg_spread.sh hidden_dim=128
bash scripts/train_scripts/dama_ddpg_spread.sh use_cuda=False device=cpu
```

Windows PowerShell 可直接使用相同入口，避免依赖 Bash：

```powershell
& E:\Users\guo\Anaconda3\envs\bcrbc\python.exe -u src\main.py --config=dama_ddpg --env-config=mpe with env_args.map_name=simple_spread_v3 seed=1
```

tag 必须保留独立奖励并指定用于选模的队伍索引：

```powershell
& E:\Users\guo\Anaconda3\envs\bcrbc\python.exe -u src\main.py --config=dama_ddpg --env-config=mpe with env_args.map_name=simple_tag_v3 common_reward=False 'dama_checkpoint_agents=[0,1,2]' seed=1
```

改动 tag 智能体数量后要同步队伍索引。模型目录为 `results/models/<unique_token>/best_model`，对应配置为 `results/sacred/<unique_token>/1/config.json`。比较不同算法时，场景参数、奖励聚合、种子、预算、验证频率和选模规则应保持一致。

## 冻结观测延迟评估

提供独立入口，无需构造 learner。复用公共 DelayedEpisodeRunner、EpisodeBatch 和 EvaluationDelay，按每个观测的到达时间交付最新已到达观测。全局 state 和 avail_actions 保持当前值，不额外设置动作延迟。每个条件复用同一模型，使用相同的环境初始种子序列；延迟采样有独立随机流。

```bash
python scripts/eval_scripts/dama_obs_delay.py \
  --config results/sacred/<unique_token>/1/config.json \
  --checkpoint results/models/<unique_token>/best_model \
  --fixed 0 1 2 4 8 --means -2 -1 0 1 2 --stds 0 0.5 1 1.5 2 \
  --cap 16 --episodes 64 --device cuda
```

`--means × --stds` 展开高斯网格，固定延迟另由 `--fixed` 指定，始终包含零延迟对照。高斯采样先截断到零、向上取整，再限制到 cap。本命令生成30个条件，1920个episode（固定与高斯标签分别保留，即使分布相同）。每个训练种子分别调用该入口。

输出 `manifest.json`（原训练配置、条件、模型SHA256）、`episodes.csv`、`summary.csv`。包含 SMAC 胜率、回报、未到达比例和历史超窗比例。tag 额外包含各agent、捕食者平均和猎物平均回报；总回报仅作记录，不解释为竞争任务整体性能。回报标准差使用样本标准差（单episode记为零）。评估结束核对参数与checkpoint文件不变。

当前专用入口为单环境顺序评估；现有 `delay_study` 的工厂加载也能加载 `dama_mac`，但其通用汇总会将tag双方回报相加，竞争结果应使用这里的分队输出。

SMAC 的底层 `seed()` 是读取方法，游戏随机种子在构造与启动时指定。专用评估入口为每个episode重新创建游戏，确保不同条件使用相同的实际游戏种子序列；因此包含游戏启动开销。MPE则通过正常seed接口重置。未修改公共SMAC封装。

## 消融与调试

消融每种输入结构需要各自无延迟训练；冻结模型不能切换结构开关。

```bash
# 无动作历史的 MADDPG 对照
bash scripts/train_scripts/dama_ddpg_spread.sh dama_use_history=False dama_use_alignment=False
# 有历史、无观测时间对齐标记
bash scripts/train_scripts/dama_ddpg_spread.sh dama_use_alignment=False
# 完整适配版默认开启两项
```

```powershell
& E:\Users\guo\Anaconda3\envs\bcrbc\python.exe -m pytest tests\test_dama.py -q -p no:cacheprovider
# 实际MPE的公共main/run训练 + 冻结评估
& E:\Users\guo\Anaconda3\envs\bcrbc\python.exe -u scripts\smoke_tests\dama_smoke.py
# 实际三个SMAC地图，需要SC2PATH和已安装地图
& E:\Users\guo\Anaconda3\envs\bcrbc\python.exe -u scripts\smoke_tests\dama_smoke.py --smac-only
```

短训练只检查接口、有限损失、保存加载和延迟交付。不会自动运行百万步实验或声称获得优秀策略。

## 本次实际验证（2026-10-05）

| 检查 | 结果 |
| --- | --- |
| DAMA专用测试 | 15项通过：历史因果性、时间对齐、终止/截断目标、补齐掩码、独立梯度、目标软更新、保存恢复、真实MPE与冻结评估 |
| SMAC 5m_vs_6m / 8m_vs_9m / MMM2 | 各完成真实短训练，有限actor/critic损失，保存best_model，同一模型测试固定0/2步观测延迟 |
| MPE spread | CPU真实短训练；同一模型固定0/2/8步及高斯(1,1)评估 |
| MPE tag | GPU真实短训练，独立奖励与捕食者选模；同一模型固定0/2/8步及高斯(1,1)评估，分队输出 |
| 五个Bash启动脚本 | Bash语法检查通过 |
| 公共main.py / run.py / ParallelRunner / 环境封装 | 无修改 |

SMAC日志、模型路径与损失记录位于 `results/dama_smoke/20261005_104359_609574/report.json`。MPE评估分别在 `results/dama_smoke_eval_spread/` 和 `results/dama_smoke_eval_tag/`；manifest记录对应配置和checkpoint。

全量回归结果为30项通过、1项已有失败：`tests/test_delay_study.py::test_mpe_grid_uses_its_own_delays` 断言70个条件，当前未修改的 `delay_study.py` 实际生成71个条件（网格参数也与测试预期不同）。隔离运行同样失败。该测试及公共网格代码均未修改。

单episode runner 的胜率日志使用 `running/test_battle_won_mean`，专用评估入口同时识别此键与parallel runner的metric前缀。实际5m冻结胜率输出复查位于 `results/dama_smoke_winrate_5m/`。
