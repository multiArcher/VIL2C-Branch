# 无延迟模型的观测延迟评估

同一地图先在无延迟下训练，再固定 checkpoint，一次跑完整条延迟网格。延迟加在环境的观测到达时间上，和 `epymarl_based` 的 `EvaluationDelay` 相同：每个新数据包只采样一次延迟，查询步收到已经到达的最新观测。策略内部的 `ObservationDelayModel` 在这次评估里关闭，避免对同一观测再延迟一次。通信延迟保持训练配置，不在这里改写。

## 条件

训练配置里的环境决定用哪一套网格。`sc2` 用 SMAC 网格，`mpe` 用 MPE 网格。同一个 STUDY 不能混放两种环境。实际延迟都是 `min(cap, ceil(max(0, sample)))`。标准差为 0 的高斯和均匀格点合并成同一次固定延迟测试。均匀分布仍与对应高斯共用原始均值和标准差，全宽是 `√12 σ`。

SMAC，`cap=8`，去重后 49 个条件：

- 高斯与均匀：均值 `-2,-1,0,1,2`，标准差 `0,0.5,1,1.5,2`
- 另外固定 `4` 和 `8`
- `mixture_balanced`：50% `N(0,1)` + 50% `N(2,1)`
- `mixture_rare_severe`：90% `N(0,1)` + 10% `N(4,1)`
- `periodic_16`：`N(0,1)` 与 `N(2,1)` 每 16 步交替，从低延迟开始
- `markov_09`：同样两个高斯，保持概率 0.9

MPE，`cap=16`：

- 高斯与均匀：均值 `-2,0,2,4,6,8,10`，标准差 `0,2,4,6,8`
- 固定延迟：`0,1,2,4,6,8,10,12,14,16`
- `mixture_balanced`：50% `N(0,4)` + 50% `N(4,4)`
- `mixture_rare_severe`：90% `N(0,4)` + 10% `N(8,4)`
- `periodic_16`：`N(0,4)` 与 `N(4,4)` 每 16 步交替，从低延迟开始
- `markov_09`：`N(0,2)` 与 `N(4,4)`，保持概率 0.9

混合和动态条件里，一个环境的所有智能体共享状态，条件高斯噪声各自独立。

## 运行

把 `scripts/eval_scripts/delay_study_template.py` 复制为同目录的 `my_study.local.py`（`*.local.py` 不会进 git），改 `STUDY`、`MODELS` 和局数，然后在仓库根目录执行：

```powershell
python -u scripts/eval_scripts/my_study.local.py
```

`config` 指向训练 Sacred 的 `config.json`，`checkpoint` 指向含 `agent.th` 的目录。路径相对仓库根目录。一个 STUDY 只放同一种算法的多次独立训练。已完成的 batch 会跳过；改模型、网格、局数或评估代码后换一个 STUDY 名称。

只重画图、不启动环境：

```powershell
python scripts/eval_scripts/draw_all.py results/evaluate/<STUDY>
python scripts/eval_scripts/draw_all.py results/evaluate/<STUDY> <model_id> <condition_id> <episode> <agent>
```

输出在研究目录的 `tables/`、`figures/evaluation/` 和 `figures/training/`。图包括高斯/均匀热力图、按实测延迟均值的分布曲线、周期与 Markov 的时间响应、动作选择耗时，以及训练日志曲线。重建误差、动作一致率和误差-动作关系图需要 BCRBC 的 latent 诊断；当前 VIL2C 记录里没有这些字段时，对应脚本不画图。

MPE 图使用回合回报，SMAC 图使用胜率。跨 run 的阴影是等权标准差；只有一个 run 时不画波动带。

MPE 热力图的颜色不随本图的分数范围拉伸。浅橙色是下界，深橙色是上界，格子里的数字为黑色。`simple_spread` 固定为 -100 到 0，`simple_tag` 固定为 0 到 300，因此同一地图上不同算法的相同分数颜色相同。SMAC 胜率仍是 0–100%。
