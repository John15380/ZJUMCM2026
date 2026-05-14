"""
问题一：多约束分组方案（ILP，PuLP/CBC）

模型：
  min  V = sum_{同市县级队对(i,j)} y_{ij}          （软约束违反次数）
  s.t. C1: 每队恰好分入一组
       C2: 每组恰好 4 支队伍
       C3: 任意两支市级队不同组
       C4: 市级队与其代管县级队不同组
       C5: y_{ij} >= x_{ig} + x_{jg} - 1  forallg（软约束线性化）

生成多个方案：通过不同 CBC 随机种子求解，取前 N_SOL 个近优解。
"""

import json
import pulp
import numpy as np
import pandas as pd
from itertools import combinations

from utils import (TEAMS, TEAM_IDX, CITIES, CITY_IDX, PREFECTURE,
                   load_master, OUT_DIR)

G        = 16   # 小组数
N_TEAMS  = 64
N_SOL    = 3    # 生成方案数


def _build_county_pairs() -> list[tuple[int, int]]:
    pairs = []
    for counties in PREFECTURE.values():
        idxs = [TEAM_IDX[c] for c in counties]
        pairs.extend(combinations(idxs, 2))
    return pairs


def solve(seed: int = 42, time_limit: int = 180, verbose: bool = False,
          forbid_assignments: list[dict[int, int]] | None = None
          ) -> tuple[list[list[str]], int] | None:
    """
    返回：
      groups      16个小组，每组为队伍名称列表
      violations  软约束违反总对数
      若添加禁止约束后不可行，返回 None

    forbid_assignments: 已生成的方案列表（{team_idx: group_idx}），
                        通过 no-good cuts 禁止重复生成。
    """
    n = N_TEAMS
    county_pairs = _build_county_pairs()

    prob = pulp.LpProblem(f"grouping_s{seed}", pulp.LpMinimize)

    # 决策变量
    x = [[pulp.LpVariable(f"x{i}g{g}", cat="Binary")
          for g in range(G)] for i in range(n)]
    y = {(i, j): pulp.LpVariable(f"y{i}_{j}", cat="Binary")
         for i, j in county_pairs}

    # 目标
    prob += pulp.lpSum(y[p] for p in county_pairs)

    # C1：每队恰好一组
    for i in range(n):
        prob += pulp.lpSum(x[i][g] for g in range(G)) == 1

    # C2：每组恰好 4 队
    for g in range(G):
        prob += pulp.lpSum(x[i][g] for i in range(n)) == 4

    # C3：任意两支市级队不同组
    for g in range(G):
        prob += pulp.lpSum(x[ci][g] for ci in CITY_IDX) <= 1

    # C4：市级队与代管县级队不同组
    for city, counties in PREFECTURE.items():
        ci = TEAM_IDX[city]
        for county in counties:
            cj = TEAM_IDX[county]
            for g in range(G):
                prob += x[ci][g] + x[cj][g] <= 1

    # C5：软约束线性化
    for (i, j) in county_pairs:
        for g in range(G):
            prob += y[(i, j)] >= x[i][g] + x[j][g] - 1

    # 禁止已有方案（no-good cut）：每个已有方案至少改 3 个县级队的组别
    for prev in (forbid_assignments or []):
        prob += pulp.lpSum(
            x[i][g] for i, g in prev.items() if i not in CITY_IDX
        ) <= len(prev) - len(CITY_IDX) - 3

    solver = pulp.PULP_CBC_CMD(
        msg=int(verbose),
        timeLimit=time_limit,
        options=["-randomCbcSeed", str(seed)],
    )
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Feasible"):
        if verbose:
            print(f"  seed={seed} 无可行解或超时: {status}")
        return None

    # 提取分组
    assignment = [0] * n
    for i in range(n):
        for g in range(G):
            if pulp.value(x[i][g]) > 0.5:
                assignment[i] = g

    groups: list[list[str]] = [[] for _ in range(G)]
    for i, g in enumerate(assignment):
        groups[g].append(TEAMS[i])

    violations = int(round(pulp.value(prob.objective)))
    return groups, violations


def _metrics(groups: list[list[str]], strength: np.ndarray) -> dict:
    """计算分组方案的均衡性与多样性指标"""
    from utils import CITY_OF, CITIES

    group_strengths = []
    city_free = 0
    for g in groups:
        idxs = [TEAM_IDX[t] for t in g]
        group_strengths.append(float(np.sum(strength[idxs])))
        has_city = any(t in CITIES for t in g)
        if not has_city:
            city_free += 1

    s_arr = np.array(group_strengths)
    return {
        "strength_std":   float(np.std(s_arr)),
        "strength_range": float(s_arr.max() - s_arr.min()),
        "city_free_groups": city_free,
    }


def _quasiA_analysis(groups: list[list[str]], strength: np.ndarray) -> list[str]:
    """识别'准一档县级队'：实力最高的5支位于无市级队小组的县级队伍"""
    from utils import CITIES

    city_free_groups = [
        g for g in groups if not any(t in CITIES for t in g)
    ]
    county_in_free = [
        t for g in city_free_groups for t in g if t not in CITIES
    ]
    # 按实力降序
    county_in_free.sort(key=lambda t: strength[TEAM_IDX[t]], reverse=True)
    return county_in_free[:5]


def main():
    master, w_gdp, w_pop = load_master()
    strength = master["strength"].values

    print(f"熵权法：w_GDP={w_gdp:.4f}  w_pop={w_pop:.4f}\n")

    seeds   = [42, 123, 2026]
    results = []
    prev_assignments: list[dict[int, int]] = []

    for i, seed in enumerate(seeds[:N_SOL]):
        print(f"-- 求解 seed={seed}（禁止前 {len(prev_assignments)} 个方案）--")
        result = solve(seed=seed, forbid_assignments=prev_assignments, verbose=False)
        if result is None:
            print(f"  seed={seed} 在多样性约束下无可行解，跳过")
            continue
        groups, viol = result
        m = _metrics(groups, strength)
        quasi = _quasiA_analysis(groups, strength)
        results.append({
            "seed": seed, "violations": viol,
            "groups": groups, "metrics": m, "quasi_A": quasi,
        })
        # 记录当前方案的县级队分配（用于禁止后续方案重复）
        assignment = {}
        for g_idx, grp in enumerate(groups):
            for t in grp:
                assignment[TEAM_IDX[t]] = g_idx
        prev_assignments.append(assignment)
        print(f"  软约束违反: {viol} 对")
        print(f"  实力标准差: {m['strength_std']:.4f}")
        print(f"  无市级队小组数: {m['city_free_groups']}")
        print(f"  准一档县级队: {quasi}")

    # 打印最优方案（违反最少的）
    best = min(results, key=lambda r: r["violations"])
    print(f"\n最优方案（seed={best['seed']}，违反={best['violations']}）:")
    for g_idx, grp in enumerate(best["groups"], 1):
        strengths = [f"{strength[TEAM_IDX[t]]:.3f}" for t in grp]
        print(f"  第{g_idx:02d}组: {grp}  实力={strengths}")

    # 保存
    out = []
    for r in results:
        out.append({
            "seed": r["seed"],
            "violations": r["violations"],
            "metrics": r["metrics"],
            "quasi_A": r["quasi_A"],
            "groups": r["groups"],
        })
    with open(OUT_DIR / "p1_grouping.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至 {OUT_DIR / 'p1_grouping.json'}")


if __name__ == "__main__":
    main()
