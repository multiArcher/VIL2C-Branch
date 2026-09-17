# MPE 环境

本项目通过独立的 `MPEWrapper` 将 MPE2 的 Parallel API 接入原有
`EnvMaker → Runner → EpisodeBatch → Learner` 链路。SMAC 配置仍可照常使用。
MPE 已从 PettingZoo 迁移到 [MPE2](https://pettingzoo.farama.org/environments/mpe/)，
因此不依赖旧的 `pettingzoo.mpe` 或 `gymma` 注册方式。

## 安装与训练

在已有项目依赖的 Python 环境中，从仓库根目录执行：

```sh
python -m pip install -r mpe_requirements.txt
python src/main.py --config=vil2c_mappo --env-config=mpe with seed=1
```

默认是 `simple_spread_v3`（3 个智能体、3 个地标、每回合 25 步）。
另一个 VIL2C 实现也可以直接使用：

```sh
python src/main.py --config=vil2c_ymappo --env-config=mpe with seed=1
```

配置中的 `t_max` 仅是运行预算，不是经过调优的收敛保证。
CPU 运行时追加 `use_cuda=False device=cpu`。

## 场景参数

`env_args.map_name` 使用 MPE2 模块名；场景专属参数放在 `scenario_args` 中。
默认空字典让每个场景使用自己的默认值，避免切换场景时携带不适用的参数。

```sh
python src/main.py --config=vil2c_mappo --env-config=mpe with env_args.scenario_args.N=5 env_args.scenario_args.local_ratio=0.5 env_args.time_limit=50
python src/main.py --config=vil2c_mappo --env-config=mpe with env_args.map_name=simple_reference_v3
python src/main.py --config=vil2c_mappo --env-config=mpe with env_args.map_name=simple_speaker_listener_v4
```

适配器要求固定智能体集合和离散动作；不同智能体的观测补零到相同长度，
动作维度取最大值，并通过动作掩码屏蔽各自不存在的动作。
全局状态为按固定智能体顺序拼接的补零后即时观测。
默认 spread 的 `obs_shape=18`、`state_shape=54`、`n_actions=5`。

默认把各智能体奖励取平均作为团队奖励；可用 `reward_scalarisation=sum`
改为求和。支持个体奖励的算法可用 `common_reward=False` 返回奖励向量。
对抗场景中把双方奖励聚合会改变任务目标，不能作为标准对抗训练配置使用。
MPE 最佳模型按评估回报选择；SMAC 仍优先使用胜率。

## 观测延迟和通信延迟

```sh
python src/main.py --config=vil2c_mappo --env-config=delayed_mpe with env_args.delay_mean=2 env_args.delay_std=0 env_args.max_delay=2
```

复用现有 `DelayedObservationWrapper`：训练观测不延迟，评估按配置延迟；
全局状态、可用动作和奖励保持即时。尚未到达的观测填零，生成时间为 -1。
封装自身维护回合步数，不再访问 SMAC 私有字段。
通信延迟独立由算法配置的 `comm_*` 参数控制；使用 `mpe` 只表示没有观测延迟，
不代表关闭了算法中的通信延迟。

## 验证

真实 MPE2 合约测试（未安装 MPE2 时跳过）：

```sh
python -m pytest tests/test_mpe.py -q -p no:cacheprovider
```

短程训练，包含策略更新和评估：

```sh
python src/main.py --config=vil2c_mappo --env-config=mpe with use_cuda=False device=cpu batch_size_run=2 batch_size=2 buffer_size=2 t_max=50 test_nepisode=2 test_interval=50 epochs=1 use_tensorboard=False save_model=False seed=7
python src/main.py --config=vil2c_ymappo --env-config=delayed_mpe with use_cuda=False device=cpu batch_size_run=2 batch_size=2 buffer_size=2 rollout_length=30 data_chunk_length=5 t_max=60 test_nepisode=2 test_interval=60 epochs=1 use_tensorboard=False save_model=False seed=7
```

原有训练循环按完整批次运行，可能超过 `t_max` 一个批次。
原有最佳模型保存逻辑独立于 `save_model`，以上命令仍可能保存最佳模型。
日志位于 `results/sacred/`，应同时检查 `run.json` 的 `COMPLETED` 状态、
`metrics.json` 中有限的训练损失和 `metric/test_return_mean`。
这些短测只验证接口和训练链路，不代表策略已收敛。

2026-09-17 本机实测：Windows / bcrbc / Python 3.10 / MPE2 1.1.1，CPU。
7 项真实环境测试通过；VIL2C-MAPPO + MPE、VIL2C-YMAPPO + delayed MPE、
VIL2C-MAPPO + 5 智能体 delayed MPE 均完成训练更新与评估，损失为有限值。
最后一项还确认评估回报从 -37.1406 到 -32.3306 时最佳模型再次保存。
未执行长程收敛实验。后续 SMAC / MPE GPU 实机回归见
[环境兼容性验证](environment_compatibility.md)。

渲染需同时设置 `env_args.render_mode=human` 和评估时的 `render=True`；
适配器不提供 `save_replay` 文件导出。
