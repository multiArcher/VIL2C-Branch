# 固定无延迟模型的观测延迟评估

同一地图先完成一次无延迟训练，再固定一个 checkpoint，在多个观测延迟条件下各运行若干完整 episode。无需为每组延迟重新训练 6M/10M。此流程评估模型对测试时观测陈旧的鲁棒性，不进行延迟环境中的再训练。

脚本 `scripts/eval_scripts/evaluate_obs_delay.py` 自动定位与模型目录同名的 Sacred `config.json`，保留网络结构、观测特征和地图配置，只加载 `agent.th`；不创建 learner、critic、optimizer，不执行训练或保存模型。支持 Windows PowerShell 和 Linux。

## MMM2 示例

在仓库根目录、已安装项目依赖的 Python 环境中运行：

```powershell
conda activate bcrbc
python scripts/eval_scripts/evaluate_obs_delay.py results/models/vil2c_ymappo__MMM2__2026-09-17_23-30-44 --delays 0:0 1:0 2:0 4:0 --episodes 32 --device cuda
```

每组运行 32 局，共 128 局。默认串行环境；可加 `--batch-size 4`，要求 `--episodes` 能被 batch size 整除，不会悄悄舍弃或增加局数。`--dry-run` 仅检查配置和 checkpoint、打印计划，不启动游戏。若 `SC2PATH` 未配置，使用 `--sc2-path "E:\StarCraft2\StarCraft II"` 指定安装目录。

## 延迟含义

- `--delay-type gaussian`（默认）：`--delays` 的每项是 `均值:标准差`。`0:0` 是无延迟基线，`2:0` 是固定 2 个环境决策步；`2:1` 表示高斯采样的均值为 2、标准差为 1。
- `--delay-type fixed --delays 0 1 2 4`：依次测试固定 0、1、2、4 步延迟，参数必须是非负整数，也接受 `2:0` 写法。
- `--delay-type uniform --delays 0:0 0:2 0:4 2:6`：每项为包含两端的整数下界和上界。例如 `0:4` 在 0、1、2、3、4 上等概率采样，`2:6` 在 2、3、4、5、6 上等概率采样。使用离散均匀分布，避免连续均匀采样再取整造成端点概率不同；不使用 `--discretization`。
- 随机延迟对每个环境、智能体和决策时刻分别采样。高斯延迟将负数截为 0，再按 `--discretization round|floor|ceil` 离散化，默认 round（PyTorch 的半整数向偶数舍入）。截断后的实际均值不保证等于参数均值。
- 时刻 t 的策略输入使用 `obs[max(0,t-delay)]`。episode 开始时历史不足则使用初始观测，每局重新开始历史与 RNN 状态。
- 只延迟局部观测；全局 state、可用动作 mask、上一动作和 agent ID 保持原来的时间语义。通信延迟固定为 0，VIL2C 原有 progressive reception 参数保留。
- 延迟在 MAC 输入端施加，环境与 EpisodeBatch 保存真实当前观测；这与 `delayed_sc2` / `delayed_mpe` 包装器是不同的路径，不能叠加。本脚本支持 `env=sc2` 或 `env=mpe`，且要求 `mac=vil2c_mac`。
- 对 SMAC，1 个决策步对应训练配置中的 `step_mul` 个游戏步，不等于 1 秒。该 MMM2 配置的 `step_mul=8`。

## Checkpoint 与配置

### MPE / simple_spread

脚本从原训练配置自动识别 MPE，不需要另加环境参数；固定延迟、均匀延迟和高斯均值/标准差网格的用法相同。例如测试已有 simple_spread 的 BestModel：

```powershell
python scripts/eval_scripts/evaluate_obs_delay.py results/models/vil2c_ymappo__simple_spread_v3__2026-09-17_09-45-27 --checkpoint best_model --means -2 -1 0 1 2 --stds 0 0.5 1 1.5 2 --episodes 100 --seeds 101 201 301 --batch-size 4 --device cuda
```

共 25 × 3 × 100 = 7500 局。先做小规模检查可将网格替换为 `--delays 0:0 2:0`，配合 `--episodes 4 --seeds 1 --batch-size 2`，共 8 局。MPE 不需要 StarCraft II，依赖见 `mpe_requirements.txt`。

simple_spread 的当前包装器没有胜负指标，因此 `win_rate` 列留空，比较 `return_mean`（越大越好）、`return_std` 和 episode 长度。保留原训练的 `time_limit`、`scenario_args`、`common_reward` 和 `reward_scalarisation`。该示例模型每局为 25 步，共享奖励使用智能体奖励的均值；回报是该共享奖励沿 episode 累加，而不是单步平均奖励。若配置是个体奖励，汇总表使用各智能体 episode 回报之和，原始 JSON 也保存各智能体指标。CSV 的 `return_metric`、`common_reward`、`reward_scalarisation` 明确记录统计口径。

默认 `--checkpoint latest` 在启动时选择最大的数字 checkpoint，整个评估矩阵固定使用它。也可指定 `--checkpoint 5942400`（必须精确存在）、`--checkpoint best_model`，或直接传入包含 `agent.th` 的目录。`best_model` 是训练期间选出的模型，并不代表各延迟条件下的最优模型。建议跨条件始终使用同一个 checkpoint。

若移动了模型但没有对应 Sacred 目录，用 `--config /path/to/training/config.json` 指定原训练配置，不能用不匹配的网络结构替代。脚本会拒绝配置中显式开启了训练观测延迟的模型；配置只能说明运行设置，不能单独证明权重的训练来源。

## 多种子和输出

高斯延迟可分别给出均值和标准差列表，自动按 `for mean: for std:` 的顺序展开笛卡尔积，免写 25 个参数：

```powershell
python scripts/eval_scripts/evaluate_obs_delay.py results/models/vil2c_ymappo__MMM2__2026-09-17_23-30-44 --checkpoint best_model --delay-type gaussian --means -2 -1 0 1 2 --stds 0 0.5 1 1.5 2 --episodes 100 --seeds 101 201 301 --batch-size 4 --device cuda
```

这里是 25 组延迟 × 3 个种子 × 100 局 = 7500 局。两列表必须同时提供，仅用于高斯分布，不能与 `--delays` 混用。相同列表值会去重，保留首次出现的顺序。原来的 `--delays` 写法继续支持。

高斯均值允许负数，标准差必须非负。均值不会先截为 0，而是先从指定高斯分布采样，再将负样本截为 0、离散化。因此 `mean=-2,std=0` 的实际延迟全为 0，而 `mean=-2,std=2` 仍可能产生正延迟；有效延迟的平均值不等于输入均值。这些参数不同的组仍分别评估。可追加 `--dry-run` 先检查 25 组参数与总局数。

选择训练时保存的最佳模型，使用 `--checkpoint best_model`（目录实际名称是小写）。例如：

```powershell
$model = "results/models/vil2c_ymappo__MMM2__2026-09-17_23-30-44"
python scripts/eval_scripts/evaluate_obs_delay.py $model --checkpoint best_model --delay-type gaussian --delays 0:0 1:1 2:1 4:1 --episodes 100 --seeds 101 201 301 --batch-size 4 --device cuda
python scripts/eval_scripts/evaluate_obs_delay.py $model --checkpoint best_model --delay-type fixed --delays 0 1 2 4 --episodes 100 --seeds 101 201 301 --batch-size 4 --device cuda
python scripts/eval_scripts/evaluate_obs_delay.py $model --checkpoint best_model --delay-type uniform --delays 0:0 0:2 0:4 2:6 --episodes 100 --seeds 101 201 301 --batch-size 4 --device cuda
```

每条命令有 4 组延迟 × 3 个测试种子 × 每组每种子 100 局 = 1200 局，三条全部运行共 3600 局。每个种子可理解为一轮重复评估；没有额外的训练轮数。`--seeds` 中相同种子会去重，重复传同一个种子不会增加轮数。每组中 `--batch-size 4` 表示同时运行 4 个环境，100 局分成 25 个批次，不改变总局数。各条件使用相同种子列表和 batch size 便于比较，但不同策略轨迹并不保证后续随机事件完全配对。并行环境使用 seed+i，跨轮种子可选间隔大于 batch size 的值，避免环境初始种子重叠。

```powershell
python scripts/eval_scripts/evaluate_obs_delay.py results/models/vil2c_ymappo__MMM2__2026-09-17_23-30-44 --checkpoint 5942400 --delays 0:0 1:1 2:1 4:1 --seeds 1 2 3 --episodes 100 --batch-size 4 --device cuda
```

默认写入独立的 `results/obs_delay_eval/<时间戳>/`：

- `manifest.json`：固定 checkpoint、原训练配置位置、每组完整有效配置。
- `summary.csv`：每个延迟和测试种子的胜率（0–1）、平均回报、回报标准差、平均 episode 长度与局数。
- `metrics_*.json`：各组 runner 原始指标。

每组完成即写盘；中途失败保留已完成结果。可用 `--output` 指定尚不存在的新目录。不同测试种子评估同一组模型权重，不代表多个独立训练种子。正式报告建议使用更多测试局数，并区分这两种随机性。
