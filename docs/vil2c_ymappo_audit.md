# VIL2C-YMAPPO 参数与实现检查

检查日期：2026-09-17。只做检查与 CPU 数值诊断，未修改训练代码或参数，未中断正在运行的实验。
此前的 SMAC/MPE 冒烟测试验证运行链路，不等同于复现官方 MAPPO 的学习性能。

## 当前实验的证据

本机找到的长程实验：
`results/sacred/vil2c_ymappo__MMM2__2026-09-17_17-10-35/1/`。
读取时最后一条评估在 1,486,400 环境步；训练仍在继续。

| 指标 | 1,600 步 | 1,486,400 步 |
|---|---:|---:|
| 评估胜率 | 0 | 0 |
| 评估平均回报 | 3.2529 | 8.9390 |
| 评估平均击杀数 | 0.71875 | 3.125 |
| 评估平均回合长度 | 33.40625 | 171.59375 |

最近读取的训练 entropy 约 1.17，pi_max 约 0.53，critic_loss 约 0.0076，均未发现 NaN。
因此证据支持“学到部分伤害/存活行为但未转化为胜利”，不支持“奖励始终为零”或“完全没有更新”。
接近 180 步上限可能表示拖延或未能结束战斗；需要回放、超时比例、兵种行为来确认，不能仅凭这些指标断定策略具体行为。
本机当前 results/sacred 下没有找到用户所述 5m_vs_6m 75% 胜率和 8m_vs_9m 长程日志。

## 官方参数对照

对照来源为 marlbenchmark/on-policy 当前 main 分支的三个训练脚本及 config.py；这是参考配置，不是 VIL2C 的最优参数证明。

| 参数 | 本地 VIL2C-YMAPPO | 官方 5m_vs_6m | 官方 MMM2 | 官方 8m_vs_9m |
|---|---:|---:|---:|---:|
| rollout 环境数 | 4 | 8 | 8 | 8 |
| rollout 长度 | 400 | 400 | 400 | 400 |
| 每轮环境转移数 | 1600 | 3200 | 3200 | 3200 |
| PPO epochs | 10 | 10 | 5 | 15 |
| mini-batches / epoch | 1 | 1 | 2 | 1 |
| policy/value clip | 0.05 | 0.05 | 0.2 | 0.05 |
| actor 输出层 gain | 0.01 | 0.01 | 1 | 0.01 |
| use_value_active_masks | True | False | False | False |
| 环境步预算 | 6M | 10M | 10M | 10M |

官方 config.py 的 `--use_value_active_masks` 使用 `action='store_false', default=True`；
三个脚本都传了这个开关，因此实际值为 False。不要根据参数名字或注释判断实际布尔值。
本地 `use_death_mask=True` 是“死亡 critic 输入变为零向量加 agent id”，
与 `use_value_active_masks` 的“是否屏蔽死亡样本 value loss”是两个不同功能。

lr=5e-4、Adam eps=1e-5、entropy_coef=0.01、gamma=0.99、GAE lambda=0.95、
hidden_dim=64、chunk=10、grad_norm_clip=10、Huber delta=10、ValueNorm、
不额外标准化奖励等设置与官方常用默认值一致。
target_update_interval_or_tau 和 q_nstep 在这条 YMAPPO 路径不参与目标计算。

## 优先检查的实现差异

### 1. 死亡状态价值学习与后续 GAE 的组合

`src/learners/ymappo_learner.py` 使用 active mask 屏蔽死亡样本的 value loss，
但 GAE 的 done_mask 只在整个环境回合结束时归零，仍会使用死亡后的预测值。
因此死亡输入的值没有直接监督，却可能参与存活阶段的优势估计。
这不是仅凭代码就能证明的掉分根因，但比直接调学习率更值得先做对照实验。
官方三个地图脚本都不屏蔽这些 value loss。

此外，当前 ValueNorm 始终只用 `returns[policy_mask > 0]` 更新；即使把
`use_value_active_masks=False`，这个统计口径也不会自动改变。
官方则在 value loss 的每个 mini-batch 更新中使用 return_batch 更新归一化统计。
因此只切换一个 YAML 开关还不是完整的官方复现。

### 2. recurrent critic 在 rollout 边界无条件清零

runner 会延续未结束环境及 actor hidden，但 learner 在每批次计算 old_values 时调用
`critic.init_hidden(bs)`，从全零重算 critic。官方同时保存并延续 actor/critic recurrent state。
因此同一回合在第 400 步跨 batch 后，critic 的历史被截断；下一次真正 reset 之前的估值与连续推理不一致。
回合更长时受影响的片段可能更多，但具体幅度需要量化。

CPU 合成输入诊断：相同 critic、相同当前输入，延续 20 步历史和清零历史的输出最大差异为 1.21753
（随机初始化测试，只证明两个计算不等价，不代表实际训练误差幅度）。

### 3. 这是带通信瓶颈的 MAPPO 变体，不是仅增加辅助损失

actor 路径为 obs+id → MessageEncoder → own message + attention aggregate → YMAPPOBase → policy。
原始 obs 没有直接拼到 actor backbone；feature normalization 位于消息拼接之后。
随着 5→8→10 个智能体、不同单位类型和观测维度增加，固定 64 维消息的压缩与注意力难度可能改变。

训练时 recv_mask 为全 1，死亡智能体也会作为发送者参加注意力；
死亡观测为零不意味着消息为零，因为还有 id、线性偏置等。
CPU 合成输入检查得到零观测+id的消息平均范数 0.90747。
这属于需要消融的设计，不足以直接判定实现错误。

### 4. 不能把当前 critic 输入称为官方 AS 的逐项复现

本地 critic 使用 EP 全局 state + 本地 obs + id；官方 StarCraft2 实现默认
use_state_agent=True，构造各 agent 的相对距离、相对位置、可攻击信息等 agent-specific 全局特征。
信息表达不同，直接复制超参数也不能保证官方胜率。

## 已排查与边界

- 当前 MMM2 是 sc2，comm delay mean/std 都为 0；观测延迟未开启。默认延迟不是这次掉分的解释。
- 零延迟下，VIL2C rollout actor 与 learner 直接调用 agent 的前向结果，在 CPU 合成批次上最大 logits 差异为 0。
- 当前 buffer_size=batch_size=batch_size_run=4，sample_times_per_run=1，新 rollout 覆盖 buffer；未发现当前配置在重复抽旧 rollout。
- GAE 当前把 timeout 当终止。官方脚本默认 use_proper_time_limits=False，也采用这一类处理；不能单独把它列为偏离官方的错误。
- runner 跨 rollout 的训练 return/length 统计会从零重新累计，所以跨边界的训练回报日志不是完整回合回报；完整评估日志不受此项影响。
- loss 变小和平均 ratio 接近 1 不能证明性能正常；还应记录 approx_KL、clip_fraction、explained_variance、死亡样本比例和 timeout 比例。

## 建议验证顺序

1. 先用普通 ymappo 在相同地图做基线，分清 MAPPO 骨干与 VIL2C 通信的影响。
2. 修正/对齐 critic recurrent state 流程，并明确死亡 value loss、ValueNorm 的语义，再做回归。
3. 用分地图参数作为新实验起点：MMM2 参考 epochs=5、mini_batch=2、clip=0.2、gain=1；
   8m_vs_9m 参考 epochs=15、mini_batch=1、clip=0.05、gain=0.01。保留对照，避免同时改动后无法归因。
4. 条件允许时恢复 8 个并行环境（batch_size_run、batch_size、buffer_size 同步），并比较相同环境步预算、多个随机种子。
5. 普通 MAPPO 能学而 VIL2C 不能时，再消融原始 obs 直连、死亡发送者掩码及消息维度。

没有进行新的长程消融，因此不能承诺改某一参数即可恢复胜率。

## 官方来源

- https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/scripts/train_smac_scripts/train_smac_5m_vs_6m.sh
- https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/scripts/train_smac_scripts/train_smac_MMM2.sh
- https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/scripts/train_smac_scripts/train_smac_8m_vs_9m.sh
- https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/config.py
- https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/r_mappo.py
- https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/runner/shared/smac_runner.py
- https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/envs/starcraft2/StarCraft2_Env.py
