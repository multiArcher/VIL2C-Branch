# SMAC / MPE 兼容性验证（2026-09-17）

验证环境：Windows，Conda `bcrbc`，PyTorch 2.6.0+cu124，CUDA 可用。
StarCraft II 路径为 `E:\StarCraft2\StarCraft II`，3m 和 MMM2 地图均已安装。
本轮使用相同的 `vil2c_ymappo` 控制器、学习器和 GPU 训练链路。

## 发现并修复的问题

YMAPPO 按固定步数采样，会在完整回合尚未结束时调用 `get_stats()`。
SMAC 原实现直接计算 `battles_won / battles_game`；首回合前分母为零，
导致环境子进程退出、主训练进程等待响应。

`SMACWrapper.get_stats()` 现在在 `battles_game == 0` 时返回零胜率和现有计数；
完成回合后继续调用 SMAC 原统计接口。此修复不影响 MPE 适配器。
回归测试覆盖首次回合前和完成回合后的两个分支。

## 实测结果

| 环境配置 | 地图 / 场景 | 训练步数 | 评估 | critic_loss |
|---|---|---:|---|---:|
| sc2 | 3m | 40 | 2 个完整回合，完成 | 0.949964 |
| delayed_sc2 | MMM2，固定 2 步观测延迟 | 20 | 1 个完整回合，完成 | 0.666890 |
| mpe | simple_tag_v3 | 30 | 1 个完整回合，完成 | 0.998746 |
| delayed_mpe | simple_spread_v3 | 30 | 1 个完整回合，完成 | 0.469855 |

以上训练预算很短，只证明启动、采样、梯度更新、评估与模型保存的兼容性，
不证明收敛或胜率。SMAC 使用胜率选择最佳模型，MPE 使用评估回报。
四次运行的状态均为 `COMPLETED`，actor/critic 损失均为有限值，最佳模型文件存在。
测试集结果为 9 passed（7 项真实 MPE 环境测试、2 项 SMAC 统计边界单元测试）。
MPE 的 30 步采样跨越 25 步回合上限，也覆盖了训练过程的自动重置。

注意：当前 `mpe.yaml` 使用 `simple_tag_v3`，`common_reward=True` 会平均
双方奖励。这验证了运行兼容性，但不是标准的双方独立对抗训练目标。
合作任务可显式选择 `simple_spread_v3`。

日志（仓库根目录下）：

- `results/sacred/vil2c_ymappo__3m__2026-09-17_17-07-13/1/`
- `results/sacred/vil2c_ymappo__MMM2__2026-09-17_17-08-04/1/`
- `results/sacred/vil2c_ymappo__simple_tag_v3__2026-09-17_17-08-35/1/`
- `results/sacred/vil2c_ymappo__simple_spread_v3__2026-09-17_17-08-53/1/`

每个目录内的 `config.json` 保存完整参数，`run.json` 保存执行状态，
`metrics.json` 保存损失与评估指标。

## 复现

从仓库根目录在 PowerShell 中执行：

```powershell
conda activate bcrbc
$env:SC2PATH = 'E:\StarCraft2\StarCraft II'
python -m pytest tests -q -p no:cacheprovider

python src/main.py --config=vil2c_ymappo --env-config=sc2 with env_args.map_name=3m batch_size_run=1 batch_size=1 buffer_size=1 rollout_length=20 data_chunk_length=10 t_max=20 test_nepisode=1 test_interval=20 epochs=1 use_tensorboard=False save_model=False seed=7

python src/main.py --config=vil2c_ymappo --env-config=delayed_sc2 with env_args.map_name=MMM2 env_args.delay_mean=2 env_args.delay_std=0 env_args.max_delay=2 batch_size_run=1 batch_size=1 buffer_size=1 rollout_length=20 data_chunk_length=10 t_max=0 test_nepisode=1 test_interval=20 epochs=1 use_tensorboard=False save_model=False seed=7

python src/main.py --config=vil2c_ymappo --env-config=mpe with env_args.map_name=simple_tag_v3 batch_size_run=1 batch_size=1 buffer_size=1 rollout_length=30 data_chunk_length=10 t_max=0 test_nepisode=1 test_interval=30 epochs=1 use_tensorboard=False save_model=False seed=7

python src/main.py --config=vil2c_ymappo --env-config=delayed_mpe with env_args.map_name=simple_spread_v3 batch_size_run=1 batch_size=1 buffer_size=1 rollout_length=30 data_chunk_length=10 t_max=0 test_nepisode=1 test_interval=30 epochs=1 use_tensorboard=False save_model=False seed=7
```

本项目训练循环条件为 `t_env <= t_max`，所以 `t_max=0` 仍执行一个采样批次和更新。
最佳模型保存独立于周期保存开关，`save_model=False` 仍会产生最佳模型文件。
SMAC 测试需要能够启动游戏和建立本地进程通信，受限沙箱可能拒绝这些操作。
