# 2026年浙江大学数学建模校赛

**赛题 B：浙超联赛赛事设计的运筹优化模型**

## 项目简介

本项目为浙江省省市县足球联赛（"浙超"）的赛事设计提供运筹优化方案，依次求解四个子问题：

| 问题 | 方法 | 核心模块 |
|---|---|---|
| P1 多约束分组 | 混合整数线性规划 (ILP) + 熵权法 | `p1_grouping.py` |
| P2 无死锁随机抽签 | Hall 婚姻定理 + 二分图最大匹配 + 蒙特卡洛 | `p2_draw.py` |
| P3 赛区多目标选址 | 容量受限 p-中值问题 (CPMP) + 模拟退火 | `p3_venue.py` |
| P4 赛制合理化建议 | Bradley-Terry 三态模型 + 比赛重要性 (MI) | `p4_collusion.py` |

## 环境配置

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv sync
```

**Python 版本：** 3.12  
**主要依赖：** PuLP + CBC, NetworkX, NumPy, SciPy, pandas, Matplotlib

## 数据准备

运行前需确保 `../data/clean/` 目录下有以下文件：

| 文件 | 来源 | 说明 |
|---|---|---|
| `teams_master.csv` | 《浙江统计年鉴 2024》 | 64 支参赛队的 GDP、人口、坐标 |
| `coords.csv` | 高德地图 API | 64 个行政中心的经纬度 |
| `distance_matrix.csv` | 高德地图 Web API | 64×64 实际公路驾车距离矩阵 |

原始数据及获取脚本见 `../data/raw/`。

## 运行方式

各模块按问题编号独立运行，但存在顺序依赖（P1 分组→ P2 抽签 → P3 选址 → P4 赛制）：

```bash
# 1. 问题一：多约束分组（约 30~90 s/方案）
uv run python p1_grouping.py

# 2. 问题二：无死锁随机抽签 + 10,000 次蒙特卡洛（约 120 s）
uv run python p2_draw.py

# 3. 问题三：赛区选址模拟退火（约 45 s/次）
uv run python p3_venue.py

# 4. 问题四：串谋风险量化 + 淘汰赛公平性（约 300~600 s）
uv run python p4_collusion.py

# 5. 生成全部图表
uv run python visualize.py
```

结果输出至 `output/` 目录（JSON 格式），图表输出至 `../paper/figures/`。

## 模块说明

| 文件 | 职责 |
|---|---|
| `utils.py` | 公共数据加载、熵权法计算、常量定义 |
| `p1_grouping.py` | ILP 建模（C1--C5 约束）、CBC 求解、no-good cut 多方案生成 |
| `p2_draw.py` | 分档抽签算法、二分图 Hall 检验、蒙特卡洛均匀性验证 |
| `p3_venue.py` | CPMP 建模、SA 求解器（含可行性优先约束处理）、灵敏度分析 |
| `p4_collusion.py` | BT 三态模型模拟、MI 计算、Stakeless Match 统计、淘汰赛抽签公平性 |
| `visualize.py` | 所有数据图表的生成 |
| `main.py` | 一键运行全部模块 |
