"""
问题三：赛区选址（容量受限 p-中值，模拟退火）

目标：min  sum_{g=1}^{16} sum_{iingroup_g} D[i, venue(g)]
约束：
  · 恰好选 8 个赛区，每赛区承办 2 个小组
  · 覆盖至少 K_MIN=6 个地级市（赛区所在地级市）
  · 各组出行总距离极差 <= Delta_MAX = mean * 0.5

输入：problem1 的分组结果 + 64×64 非对称距离矩阵
"""

import json
import math
import random
import numpy as np

from utils import (TEAMS, TEAM_IDX, CITIES, CITY_OF,
                   load_master, load_distance, OUT_DIR)

P        = 8       # 赛区数
CAP      = 2       # 每赛区承办小组数
K_MIN    = 6       # 最少覆盖地级市数
DELTA_F  = 0.5     # 公平性极差上限（均值的比例）
PENALTY_BASE = 100000   # 约束违反基础惩罚（1 km 或 1 个地级市违反对应此惩罚）
                          # 约为总出行均值的 10 倍，保证可行解远优于不可行解
                          # 同时在高温下允许少量不可行解被接受（exp(-1e5/5000) ~ 2e-9，极小但非零）

# SA 超参数
T_INIT   = 5000.0
T_FINAL  = 0.1
ALPHA    = 0.9998      # 冷却系数；温度在约 54000 步内降至 T_FINAL
PATIENCE = 10_000      # 早停：连续 10000 步无改善则终止
MAX_ITER = 80_000


def precompute_delta(groups: list[list[str]], dist: np.ndarray,
                     delta_factor: float = DELTA_F) -> float:
    """预计算固定的公平性阈值 Delta（km）。

    策略：计算所有小组-赛区配对的总出行距离的全局均值，乘以 delta_factor。
    该值在 SA 运行前确定，不随当前解变化（符合 epsilon-约束法的标准做法）。
    """
    all_costs = []
    for grp in groups:
        for v in range(len(TEAMS)):
            cost = sum(dist[TEAM_IDX[t], v] for t in grp)
            all_costs.append(cost)
    return delta_factor * float(np.mean(all_costs))


def _group_travel(groups: list[list[str]], venues: list[int],
                  group_venue: list[int], dist: np.ndarray) -> np.ndarray:
    """返回 shape=(16,) 的各组出行总距离"""
    costs = np.zeros(len(groups))
    for g_idx, grp in enumerate(groups):
        v = venues[group_venue[g_idx]]
        for team in grp:
            costs[g_idx] += dist[TEAM_IDX[team], v]
    return costs


def _city_coverage(venues: list[int]) -> int:
    """赛区覆盖的地级市数"""
    cities_covered = set()
    for v in venues:
        team_name = TEAMS[v]
        cities_covered.add(CITY_OF.get(team_name, team_name))
    return len(cities_covered)


def _energy(groups: list[list[str]], venues: list[int],
            group_venue: list[int], dist: np.ndarray,
            delta_fixed: float) -> float:
    """目标函数 + 约束惩罚（使用预计算的固定 Delta 阈值）"""
    costs = _group_travel(groups, venues, group_venue, dist)
    total = costs.sum()

    # 公平性约束：使用固定 Delta（不依赖当前解）
    range_viol = max(0.0, costs.max() - costs.min() - delta_fixed)

    # 覆盖约束
    cov = _city_coverage(venues)
    cov_viol = max(0, K_MIN - cov)

    return total + PENALTY_BASE * range_viol + PENALTY_BASE * cov_viol


def _random_state(n: int, rng: random.Random) -> tuple[list[int], list[int]]:
    """随机初始解：随机选8个赛区，将16组随机分配（每赛区2组）"""
    venues = rng.sample(range(n), P)
    # group_venue[g] in {0,...,7}，满足每赛区2组
    assignment = list(range(P)) * CAP          # [0,0,1,1,...,7,7]
    rng.shuffle(assignment)
    return venues, assignment


def _neighbor(venues: list[int], group_venue: list[int],
              n: int, rng: random.Random) -> tuple[list[int], list[int]]:
    """邻域移动（三种等概率）：
    1. 交换两个不同赛区的组的分配
    2. 将一个赛区位置换为另一个未选地点
    3. 同时做 1+2（组交换 + 赛区替换），扩大跳跃距离
    """
    move = rng.randint(0, 2)
    v = venues[:]
    gv = group_venue[:]

    if move == 0:
        # 交换两个不同赛区的组（保证有效变更）
        a = rng.randrange(len(gv))
        diff = [i for i in range(len(gv)) if gv[i] != gv[a]]
        if diff:
            b = rng.choice(diff)
            gv[a], gv[b] = gv[b], gv[a]

    elif move == 1:
        # 换一个赛区位置
        v_idx = rng.randrange(P)
        non_venues = [i for i in range(n) if i not in v]
        if non_venues:
            new_loc = rng.choice(non_venues)
            v[v_idx] = new_loc

    else:
        # 组合移动：交换两个不同赛区的组 + 替换一个赛区
        a = rng.randrange(len(gv))
        diff = [i for i in range(len(gv)) if gv[i] != gv[a]]
        if diff:
            b = rng.choice(diff)
            gv[a], gv[b] = gv[b], gv[a]
        v_idx = rng.randrange(P)
        non_venues = [i for i in range(n) if i not in v]
        if non_venues:
            new_loc = rng.choice(non_venues)
            v[v_idx] = new_loc

    return v, gv


def simulated_annealing(groups: list[list[str]], dist: np.ndarray,
                        delta_fixed: float, seed: int = 42,
                        record_every: int = 500
                        ) -> tuple[list[int], list[int], float, list]:
    """
    返回：
      best_venues      8 个赛区的位置索引（对应 TEAMS）
      best_gv          16 组的赛区分配（0-7）
      best_energy      目标值
      history          [(iteration, best_energy), ...] 收敛曲线
    """
    n = len(TEAMS)
    rng = random.Random(seed)

    venues, gv = _random_state(n, rng)
    E = _energy(groups, venues, gv, dist, delta_fixed)
    best_v, best_gv, best_E = venues[:], gv[:], E
    history = [(0, best_E)]

    T = T_INIT
    no_improve = 0   # 连续无改善步数（用于早停）
    for it in range(MAX_ITER):
        nv, ngv = _neighbor(venues, gv, n, rng)
        nE = _energy(groups, nv, ngv, dist, delta_fixed)
        dE = nE - E
        if dE < 0 or rng.random() < math.exp(-dE / T):
            venues, gv, E = nv, ngv, nE
            if E < best_E:
                best_v, best_gv, best_E = venues[:], gv[:], E
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        T = max(T_FINAL, T * ALPHA)
        if (it + 1) % record_every == 0:
            history.append((it + 1, best_E))

        if no_improve >= PATIENCE:   # 早停
            history.append((it + 1, best_E))
            break

    return best_v, best_gv, best_E, history


def report(groups: list[list[str]], venues: list[int],
           group_venue: list[int], dist: np.ndarray,
           delta_fixed: float) -> dict:
    costs = _group_travel(groups, venues, group_venue, dist)
    venue_names = [TEAMS[v] for v in venues]
    cov = _city_coverage(venues)

    group_info = []
    for g_idx, (grp, cost) in enumerate(zip(groups, costs)):
        group_info.append({
            "group": g_idx + 1,
            "teams": grp,
            "venue": venue_names[group_venue[g_idx]],
            "travel_km": round(float(cost), 1),
        })

    return {
        "venues": venue_names,
        "city_coverage": cov,
        "total_travel_km": round(float(costs.sum()), 1),
        "mean_travel_km":  round(float(costs.mean()), 1),
        "max_travel_km":   round(float(costs.max()), 1),
        "min_travel_km":   round(float(costs.min()), 1),
        "range_travel_km": round(float(costs.max() - costs.min()), 1),
        "delta_fixed_km":  round(delta_fixed, 1),
        "feasible": bool(
            _city_coverage(venues) >= K_MIN and
            (costs.max() - costs.min()) <= delta_fixed
        ),
        "groups": group_info,
    }


def main():
    import json

    # 加载问题一结果
    p1_path = OUT_DIR / "p1_grouping.json"
    if not p1_path.exists():
        print("请先运行 p1_grouping.py")
        return
    with open(p1_path, encoding="utf-8") as f:
        p1_results = json.load(f)

    dist = load_distance()
    best_sol = min(p1_results, key=lambda r: r["violations"])
    groups   = best_sol["groups"]

    # 预计算固定 Delta（不依赖 SA 解，符合 epsilon-约束法标准做法）
    delta_fixed = precompute_delta(groups, dist, delta_factor=DELTA_F)
    print(f"使用问题一方案（seed={best_sol['seed']}，违反={best_sol['violations']}）")
    print(f"预计算固定 Delta = {DELTA_F} × 全局均值 = {delta_fixed:.1f} km")
    print(f"运行模拟退火（{MAX_ITER} 次迭代）…")

    solutions = []
    all_histories = {}
    for seed in [42, 123, 2026]:
        vn, gv, E, hist = simulated_annealing(groups, dist, delta_fixed, seed=seed)
        r = report(groups, vn, gv, dist, delta_fixed)
        r["seed"] = seed
        r["objective"] = round(E, 1)
        solutions.append(r)
        all_histories[seed] = hist
        print(f"  seed={seed}: 总出行={r['total_travel_km']} km  "
              f"极差={r['range_travel_km']} km（Delta={delta_fixed:.0f} km）"
              f"  覆盖{r['city_coverage']}市  可行={r['feasible']}")

    best = min(solutions, key=lambda r: r["total_travel_km"]
               if r["feasible"] else float("inf"))
    print(f"\n最优方案（seed={best['seed']}）:")
    print(f"  赛区: {best['venues']}")
    print(f"  覆盖地级市: {best['city_coverage']}")
    print(f"  总出行距离: {best['total_travel_km']} km")
    print(f"  极差: {best['range_travel_km']} km  "
          f"（固定阈值 Delta={best['delta_fixed_km']} km）")

    out_path = OUT_DIR / "p3_venue.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(solutions, f, ensure_ascii=False, indent=2)

    hist_path = OUT_DIR / "p3_convergence.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in all_histories.items()}, f, indent=2)
    print(f"\n结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
