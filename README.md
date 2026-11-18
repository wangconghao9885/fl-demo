# 联邦学习演示（Flower + PyTorch）

本仓库提供一个简短且可运行的演示，聚焦于**非独立同分布（non-IID）数据**下的**联邦学习**。

演示包含：
- 使用 Dirichlet 分区进行非 IID 数据可视化
- 本地训练基线（每个客户端训练一个独立模型）
- 使用 Flower 和 FedAvg 进行联邦训练，并生成一个全局模型
- 对比图：本地模型、联邦学习和集中式参考方法

## 仓库结构

- `commands.txt`  
  用于配置环境和运行演示的命令清单，可直接复制粘贴。

- `run_demo.sh`  
  按以下顺序运行演示：非 IID 分布图 → 本地基线 → 联邦训练 → 集中式对比与结果绘图。

- `prepare_data.py`  
  将 MNIST 数据集下载至 `./data/`。

- `common.py`  
  通用工具：
  - 模型定义（`SmallCNN`）
  - Dirichlet 非 IID 分区
  - 数据加载器构建（`make_loaders`）
  - 训练和评估工具
  - 参数转换（PyTorch ↔ NumPy）
  - 可选警告配置（启用时）

- `client.py`  
  Flower 客户端实现，负责 PyTorch 本地训练和评估。

- `server.py`  
  Flower 服务端实现，使用 FedAvg，并在全局测试集上评估全局模型。

- `local_baselines.py`  
  为每个客户端训练一个本地模型，并输出：
  - 每个客户端各类别（0–9）的样本数量（每客户端占一行）
  - 本地模型在全局测试集上的准确率
  将结果写入 `logs/local_baselines.json`。

- `plot_non_iid.py`  
  绘制非 IID 分区图（堆叠柱状图：每个客户端的类别分布）。

- `plot_compare.py`  
  绘制：
  - 联邦学习准确率随轮次变化的曲线（如果日志中有相关数据）
  - 最终对比柱状图（各本地模型、本地平均值、联邦学习最终结果、集中式参考结果）

- `plot_logs.py`  
  快速绘图工具，读取 `logs/server_metrics.jsonl`（如果存在）并展示全局准确率。

- `logs/`  
  生成的输出，包括服务端/客户端日志、JSON 日志和历史记录转储。

## 快速开始（推荐）

所有命令都列在 `commands.txt` 中，执行顺序如下：

1. 创建并激活虚拟环境
2. 安装依赖
3. 下载 MNIST（推荐）
4. 运行演示
5. 绘制全局准确率曲线

为避免在此重复命令内容，请打开 `commands.txt` 并从上到下执行其中的命令。

## 仓库中的 `commands.txt`

该文件包含：

- 1）创建环境
  `python3 -m venv .venv`，再执行 `source .venv/bin/activate` 激活环境

- 2）安装依赖
  `pip install -r requirements.txt`

- 3）下载 MNIST（推荐）
  `python prepare_data.py`

- 4）运行演示
  先执行 `chmod +x run_demo.sh`，再执行 `bash run_demo.sh`

- 5）快速绘制全局准确率曲线
  `python plot_logs.py logs/server_metrics.jsonl`

## 配置（环境变量）

默认值已在 `run_demo.sh` 中设置。运行脚本前，可以通过环境变量覆盖这些默认值。

### 非 IID 分区参数（图表、本地基线和联邦学习共用）
- `NUM_CLIENTS`（默认值：3）
- `DIRICHLET_ALPHA`（默认值：0.1）
  注意：**alpha 越小，非 IID 程度越强**，客户端的数据分布也越不均衡。
  推荐值：
  - 0.05：异构性极强
  - 0.1：异构性强（推荐用于短演示）
  - 0.5：异构性中等
  - 1.0及以上：接近 IID
- `DIRICHLET_SEED`（默认值：42）
- `MAX_SAMPLES_PER_CLIENT`（默认值：12000）

### 联邦训练参数
- `NUM_ROUNDS`（默认值：15）
- `LOCAL_EPOCHS`（默认值：1）
- `LR`（默认值：0.05）
- `BATCH_SIZE`（默认值：64）

### 本地基线和集中式参考参数
- `LOCAL_BASELINE_EPOCHS`（默认值：2）
- `CENTRALIZED_EPOCHS`（默认值：2）

## 推荐的公平对比（本地基线与联邦学习）

要在相近的**每客户端计算预算**下比较本地基线和联邦学习，应尽量使每个客户端的本地训练轮数相同：

- 联邦学习的每客户端训练轮数 ≈ `NUM_ROUNDS × LOCAL_EPOCHS`
- 本地基线的每客户端训练轮数 = `LOCAL_BASELINE_EPOCHS`

推荐设置：
- `LOCAL_BASELINE_EPOCHS = NUM_ROUNDS × LOCAL_EPOCHS`

## 预期输出

- 一张堆叠柱状图，展示每个客户端的类别分布（非 IID 分区）
- 每个本地基线客户端的控制台输出：
  - 在一行中显示各类别（0–9）的样本数量
  - 本地模型在全局测试集上的准确率
- 联邦训练日志（服务端和客户端）写入 `logs/`
- 对比图：
  - 联邦学习准确率随轮次变化的曲线（如果能从日志中提取）
  - 最终柱状图：各本地客户端准确率、本地平均值、联邦学习最终结果、集中式参考结果

## 故障排查

- 如果找不到 MNIST，请先运行 `prepare_data.py`（即 `commands.txt` 中的第 3 步）。
- 如果 `plot_logs.py` 或 `plot_compare.py` 找不到联邦学习准确率序列：
  - 检查 `logs/server_metrics.jsonl`
  - 检查 `logs/history_dump.json`
  - 检查 `logs/server_stdout.log`
  日志结构可能因所安装的 Flower 版本而略有不同。
