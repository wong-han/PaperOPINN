# 新版五方法无人机比较

分别直接运行，无需手动设置参数：

- `p4_compare_five_quadratic.py`：二次代价无人机问题。
- `p4_compare_five_nonquadratic.py`：非二次代价/约束无人机问题。

比较 LQR、SIREN、MLP、RL、OPINN，不包含 MPC。不训练网络。
默认加载各原训练目录 `model` 中的正式模型，不从种子结果中自动挑选表现最好的模型：

| 方法 | 二次问题模型 | 非二次问题模型 |
| --- | --- | --- |
| MLP | `sup_learn_ex_plot/model/mlp_quadrotor3d.pth` | `sup_learn_ex_plot/model/mlp_quadrotor3d_constrained.pth` |
| SIREN | `sup_learn_ex_plot/model/siren_quadrotor3d.pth` | `sup_learn_ex_plot/model/siren_quadrotor3d_constrained.pth` |
| RL | `reinforce_learn_ex_plot/model/rl_quadrotor3d.pth` | `reinforce_learn_ex_plot/model/rl_quadrotor3d_constrained.pth` |
| OPINN | `pihnn_experiments_plot/model/hnn_quadrotor3d_TR_v2_paperplot.pth` | `pihnn_constrained_ex_plot/model/chnn_quadrotor3d_TR_v1.pth` |

网络类从对应最新版训练程序导入。模型必须存在且结构完全匹配，否则报错。
可以用 `--mlp-model`、`--siren-model`、`--rl-model`、`--opinn-model` 指定其他检查点，
例如某个已完成的种子模型。实际模型路径和 SHA256 会记录在结果中。

## 保留的设置和指标

- 50 个蒙特卡洛回合，测试随机种子 2222。先生成独立的回合种子列表，五种控制器共享每回合初始状态。
- 初始状态由原环境的 reset 生成，不另外修改采样范围。
- 环境使用 `theoretic_mode=True, rl_mode=False`，关闭训练奖励塑形。
- 沿用完整固定时域 `int(T/dt)`（当前 800 步）；记录但不按 terminated/truncated 提前停止。
  这是旧比较程序的测试口径。非有限状态/代价会停止计算并记为失败，不伪造成功轨迹。
- 最终位置范数不超过 0.1 判为成功，打印并保存每种方法的成功率。
- 代价 J 为环境累计回报的负值。非二次问题保留原环境的约束惩罚项。
- 代价均值/标准差及图仅使用各方法自己的成功回合，样本数可能不同。
- 约束图保留旧指标 `log10(abs(min_y)+1e-8)`，也仅使用成功回合。
  该量不是严格的约束违反深度；原始 min_y 一并保存。

## 输出和重新绘图

输出分别位于 `results/quadratic_five_methods/` 和
`results/nonquadratic_five_methods/`，不会覆盖旧程序的 `data` 和 `image`。

- `data/summary.json`：设置、模型来源、成功率及成功回合的指标均值/标准差。
- `data/all_episode_metrics.csv`：所有回合的指标和测试种子。
- `data/all_trajectories_and_metrics.npz`：所有状态、控制、原始回报和成功标记。
- `data/return_boxplot_data.csv`：成功回合的 J（沿用旧字段名 Return）。
- `data/logy_boxplot_data.csv`：约束问题的成功回合约束指标。
- `image/return_boxplot.*`、`image/min_y_boxplot.*`：小提琴图、箱线图、散点叠加，保存 PDF/SVG/PNG。

颜色、方法顺序和 12 x 7 英寸布局沿用原 SIREN 比较版本；y 轴标签使用
`usetex=True`，需要本机 LaTeX 环境。其他文字仍由 Matplotlib 渲染。
若某种方法没有成功回合，该方法没有统计点，不填入虚假零值。

仿真完成后，仅重新画图，不加载模型、不重新仿真：

```powershell
python p4_compare_five_quadratic.py --plot-only
python p4_compare_five_nonquadratic.py --plot-only
```

可选用 `--episodes 2 --horizon-steps 20 --device cpu --output-dir 临时目录` 做流程检查，
短时域检查结果不能作为正式性能比较结果。
