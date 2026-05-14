"""共享数据加载与实力评分（熵权法）"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT       = Path(__file__).parent.parent
DATA_CLEAN = ROOT / "data" / "clean"
OUT_DIR    = ROOT / "code" / "output"
OUT_DIR.mkdir(exist_ok=True)

# -- 64支参赛队伍（顺序与距离矩阵一致）--------------------------------------
TEAMS = [
    "杭州市","建德市","桐庐县","淳安县",
    "宁波市","余姚市","慈溪市","象山县","宁海县",
    "温州市","瑞安市","乐清市","龙港市","永嘉县","平阳县","苍南县","文成县","泰顺县",
    "嘉兴市","海宁市","平湖市","桐乡市","嘉善县","海盐县",
    "湖州市","德清县","长兴县","安吉县",
    "绍兴市","诸暨市","嵊州市","新昌县",
    "金华市","兰溪市","义乌市","东阳市","永康市","武义县","浦江县","磐安县",
    "衢州市","江山市","常山县","开化县","龙游县",
    "舟山市","岱山县","嵊泗县",
    "台州市","温岭市","临海市","玉环市","三门县","天台县","仙居县",
    "丽水市","龙泉市","青田县","缙云县","遂昌县","松阳县","云和县","庆元县","景宁畲族自治县",
]
TEAM_IDX = {t: i for i, t in enumerate(TEAMS)}

CITIES = [
    "杭州市","宁波市","温州市","嘉兴市","湖州市",
    "绍兴市","金华市","衢州市","舟山市","台州市","丽水市",
]
CITY_IDX = [TEAM_IDX[c] for c in CITIES]

# 地级市 --> 代管县级队
PREFECTURE: dict[str, list[str]] = {
    "杭州市": ["建德市","桐庐县","淳安县"],
    "宁波市": ["余姚市","慈溪市","象山县","宁海县"],
    "温州市": ["瑞安市","乐清市","龙港市","永嘉县","平阳县","苍南县","文成县","泰顺县"],
    "嘉兴市": ["海宁市","平湖市","桐乡市","嘉善县","海盐县"],
    "湖州市": ["德清县","长兴县","安吉县"],
    "绍兴市": ["诸暨市","嵊州市","新昌县"],
    "金华市": ["兰溪市","义乌市","东阳市","永康市","武义县","浦江县","磐安县"],
    "衢州市": ["江山市","常山县","开化县","龙游县"],
    "舟山市": ["岱山县","嵊泗县"],
    "台州市": ["温岭市","临海市","玉环市","三门县","天台县","仙居县"],
    "丽水市": ["龙泉市","青田县","缙云县","遂昌县","松阳县","云和县","庆元县","景宁畲族自治县"],
}

# 每支队伍所属地级市名称（包括市级队本身）
CITY_OF: dict[str, str] = {}
for city in CITIES:
    CITY_OF[city] = city
for city, counties in PREFECTURE.items():
    for c in counties:
        CITY_OF[c] = city


def _entropy_weight(x: np.ndarray) -> float:
    """单指标熵权（已归一化的 x，取值 [0,1]）"""
    p = x / (x.sum() + 1e-12)
    p = np.where(p < 1e-12, 1e-12, p)
    H = -np.sum(p * np.log(p)) / np.log(len(p))
    return 1.0 - H


def load_master() -> tuple[pd.DataFrame, float, float]:
    """
    返回：
      df        带 strength 列的 DataFrame（以地区为索引）
      w_gdp     GDP 的熵权
      w_pop     人口的熵权
    """
    df = pd.read_csv(DATA_CLEAN / "teams_master.csv").set_index("地区")
    df = df.reindex(TEAMS)          # 确保顺序与 TEAMS 一致

    gdp = df["GDP_亿元"].values.astype(float)
    pop = df["人口_万人"].values.astype(float)

    gdp_norm = (gdp - gdp.min()) / (gdp.max() - gdp.min())
    pop_norm = (pop - pop.min()) / (pop.max() - pop.min())

    w_gdp = _entropy_weight(gdp_norm)
    w_pop = _entropy_weight(pop_norm)
    total = w_gdp + w_pop
    w_gdp /= total
    w_pop /= total

    df["strength"] = w_gdp * gdp_norm + w_pop * pop_norm
    # theta_i = s_i / s̄
    df["theta"] = df["strength"] / df["strength"].mean()
    return df, w_gdp, w_pop


def load_distance() -> np.ndarray:
    """返回 64×64 numpy 矩阵（行=出发地，列=目的地，单位 km）"""
    dist = pd.read_csv(DATA_CLEAN / "distance_matrix.csv", index_col=0)
    dist = dist.reindex(index=TEAMS, columns=TEAMS)
    return dist.values.astype(float)
