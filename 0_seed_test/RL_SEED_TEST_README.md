# 无人机 RL 多种子测试

分别直接运行 `p4_quadrotor3d_rl_test.py` 和
`p4_quadrotor3d_constrained_rl_test.py`，无需手动设置参数。
默认依次训练种子 0、1、2、3、4，不同时运行多个种子，避免影响计时。
每个种子内部保持原程序的多进程并行采样。

## 默认设置

导入原程序的 `Args`、`RLNet` 和 `train_parallel_ppo`，不修改原训练逻辑。
目前为 8 个环境、每环境每轮 2000 步、minibatch 2048、每轮更新 256 次。
无约束问题请求 1500000 次交互，按完整 rollout 实际执行 1504000 次；
约束问题请求及实际执行均为 2000000 次。不是按回报阈值提前停止。
训练环境保持 `theoretic_mode=False, rl_mode=True`，回报含奖励塑形，
不代表蒙特卡洛测试时的真实控制代价。本入口不执行蒙特卡洛测试。

## 保存内容

结果位于本目录 `results/对应入口程序名/`，原模型和原实验结果不被覆盖。

- `seed_N/model/model.pth`：最终策略与价值网络。
- `seed_N/results/training.json`：回报曲线、实际交互数、三项时间及末段稳定性统计。
- `seed_N/return_curve.npz`：每轮交互数、该轮完成回合的平均回报、最近 50 个完成回合的平均回报、累计回合数。
- `seed_N/configuration.json`：超参数及源程序、训练公共代码、环境代码的指纹。
- `summary.json`：逐种子结果和三项时间的均值、标准差。
- `aggregate_curves.npz`：跨种子曲线、均值、标准差和有效种子数量。
- `return_individual`、`return_mean`、`contributing_seeds`：分别保存 PDF、PNG、SVG。

均值和标准差曲线使用最近 50 个已完成回合的平均训练回报。
标准差为种子间总体标准差（ddof=0），不是置信区间。
`return_mean` 图先对每个种子应用因果 EMA（默认 alpha=0.25），再计算均值及种子间 SD；
该图的停止点也使用平滑后的末值。原始回报、原始均值/SD、时间和稳定性统计不变，
`return_individual` 仍展示未经额外平滑的曲线。
`plot_settings.json` 记录绘图设置，`aggregate_curves.npz` 同时保存原始及平滑曲线。
可加 `--smooth-alpha 0.15` 增强平滑，或 `--smooth-alpha 1` 关闭额外平滑。
没有完成回合的记录保存为缺失值，不当作 0，也不插值。
`Stop Points` 标注固定交互预算的结束位置，不表示已满足收敛判据。

三项时间沿用原训练公共代码的口径：数据采集时间包括策略采样和环境交互；
训练时间为总时间减去采集时间，包括网络更新和其余中间计算。
总时间统计训练循环至环境关闭，不包括环境/网络初始化、保存模型和绘图。

## 重用结果

再次直接运行时，配置一致的已完成种子会被复用，不再训练。
仅重新画图：

```powershell
python p4_quadrotor3d_rl_test.py --plot-only
python p4_quadrotor3d_constrained_rl_test.py --plot-only
```

绘图无需导入原环境或创建并行采样进程。配置或源代码改变时，
程序会拒绝混用旧结果；使用 `--output-dir 新目录` 保存新实验，
或用 `--retrain` 覆盖指定种子的已有实验结果。
