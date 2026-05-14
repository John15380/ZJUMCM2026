"""
问题二：无死锁随机抽签算法 + 公平性验证（蒙特卡洛）

算法（对应论文 Algorithm 1）：
  1. Shuffle 市级队 T0，依次分入不同小组（满足 C3）
  2. 对县级队按实力分为 T1/T2/T3 三档，每档：
     repeat:
       pi_l = Shuffle(T_l)
       for i in pi_l:
         S_i = 合法小组集（不违反 C3/C4）
         if HallCheck(剩余队伍, 剩余容量, 约束):
           g* ~ Uniform(S_i)；分配 i --> g*
         else:
           撤销本轮 T_l 分配；break
     until 无回溯

公平性验证：运行 N_SIM 次，记录每支市级队进入各小组的频率分布。
"""

import json
import random
import numpy as np
import networkx as nx

from utils import (TEAMS, TEAM_IDX, CITIES, CITY_IDX, PREFECTURE,
                   load_master, OUT_DIR)

G       = 16
N_SIM   = 10_000   # 蒙特卡洛次数
MAX_RETRY = 500    # 单档最大回溯次数


def _legal_groups(
    team: int,
    assigned: dict[int, int],   # team_idx -> group
    group_sizes: list[int],     # group_sizes[g] = 当前组内人数
) -> list[int]:
    """计算 team 的合法小组集合 S_i（不违反 C3/C4，且组未满）"""
    legal = []
    # 找 team 所属地级市的市级队已在哪组
    from utils import CITY_OF, CITIES
    city_name = CITY_OF.get(TEAMS[team])
    city_team = TEAM_IDX.get(city_name) if city_name in CITIES else None
    city_group = assigned.get(city_team, None) if city_team is not None else None

    # 找已分配的市级队所在组（用于 C3）
    city_occupied_groups = {
        assigned[ci] for ci in CITY_IDX if ci in assigned
    }

    for g in range(G):
        if group_sizes[g] >= 4:
            continue
        # C3：若 team 是市级队，该组不能已有其他市级队
        if team in CITY_IDX and g in city_occupied_groups:
            continue
        # C4：team 的市级队若已在 g，则 team 不能进 g
        if city_group is not None and g == city_group:
            continue
        legal.append(g)
    return legal


def _hall_check(
    current_remaining: list[int],
    future_teams: list[int],
    group_sizes: list[int],
    assigned: dict[int, int],
) -> bool:
    """
    Hall 条件：构建二分图（当前档剩余 + 所有后续档队伍 × 小组槽位），
    判断是否存在将所有剩余队伍分配完的完美匹配。
    """
    all_remaining = list(current_remaining) + list(future_teams)
    if not all_remaining:
        return True

    B = nx.Graph()
    left = all_remaining
    # 右侧：将每个小组按剩余容量展开为 cap 个槽位节点
    right = []
    for g in range(G):
        cap = 4 - group_sizes[g]
        for slot in range(cap):
            right.append((g, slot))

    B.add_nodes_from(left, bipartite=0)
    B.add_nodes_from(right, bipartite=1)

    for team in left:
        legal = _legal_groups(team, assigned, group_sizes)
        for g in legal:
            cap = 4 - group_sizes[g]
            for slot in range(cap):
                B.add_edge(team, (g, slot))

    matching = nx.bipartite.maximum_matching(B, top_nodes=left)
    matched = sum(1 for t in left if t in matching)
    return matched == len(left)


def _tier_split(teams_idx: list[int], strength: np.ndarray) -> tuple[list, list, list]:
    """将县级队按实力均分为 T1/T2/T3（18/18/17）"""
    sorted_idx = sorted(teams_idx, key=lambda i: strength[i], reverse=True)
    n = len(sorted_idx)
    s1 = (n + 2) // 3  # ceiling: 18
    s2 = (n + 1) // 3  # mid: 18
    t1 = sorted_idx[:s1]
    t2 = sorted_idx[s1:s1 + s2]
    t3 = sorted_idx[s1 + s2:]
    return t1, t2, t3


def draw_once(strength: np.ndarray, rng: random.Random) -> dict[int, int] | None:
    """
    执行一次抽签，返回 {team_idx: group_idx} 或 None（失败）。
    """
    assigned: dict[int, int] = {}
    group_sizes = [0] * G

    # -- T0：市级队随机排列，分入随机选取的 11 个不同小组 ------------
    city_order = CITY_IDX[:]
    rng.shuffle(city_order)
    city_groups = rng.sample(range(G), len(CITY_IDX))
    for ci, g in zip(city_order, city_groups):
        assigned[ci] = g
        group_sizes[g] += 1

    # -- T1/T2/T3：县级队分档抽签 ------------------------------------------
    county_idx = [TEAM_IDX[c] for c in TEAMS if c not in CITIES]
    t1, t2, t3 = _tier_split(county_idx, strength)

    tiers = [t1, t2, t3]
    for ti, tier in enumerate(tiers):
        backtrack_count = 0
        success = False
        # 后续档位的所有队伍（用于 Hall 检验全量剩余）
        future_teams = [t for future_tier in tiers[ti + 1:] for t in future_tier]
        while backtrack_count < MAX_RETRY:
            tier_assigned: dict[int, int] = {}
            tier_sizes = group_sizes[:]
            tier_order = tier[:]
            rng.shuffle(tier_order)
            failed = False

            for team in tier_order:
                # 本档未分配队伍（不含当前 team）
                current_rem = [
                    t for t in tier if t not in tier_assigned and t != team
                ]
                legal = _legal_groups(team, {**assigned, **tier_assigned}, tier_sizes)
                if not legal:
                    failed = True
                    break
                # Hall 检验（全量剩余 = 本档未分配 + 后续档所有队伍）
                if not _hall_check(
                    current_rem,
                    future_teams,
                    tier_sizes,
                    {**assigned, **tier_assigned},
                ):
                    failed = True
                    break
                g_star = rng.choice(legal)
                tier_assigned[team] = g_star
                tier_sizes[g_star] += 1

            if not failed:
                assigned.update(tier_assigned)
                group_sizes[:] = tier_sizes
                success = True
                break
            backtrack_count += 1

        if not success:
            return None

    return assigned


def monte_carlo(strength: np.ndarray, n_sim: int = N_SIM, seed: int = 0
                ) -> dict:
    """
    运行 n_sim 次抽签，统计：
      - 成功率
      - 每支市级队进入各小组的频率（均匀性检验）
      - 县级队入组频率（按地级市汇总，均匀性检验）
    """
    rng = random.Random(seed)
    success = 0
    # 记录市级队进组频率：city_freq[city_idx][group] = count
    city_freq: dict[int, list[int]] = {ci: [0]*G for ci in CITY_IDX}
    # 记录县级队进组频率：county_freq[county_idx][group] = count（按地级市汇总用）
    county_freq: dict[int, list[int]] = {}
    for i in range(len(TEAMS)):
        if i not in CITY_IDX:
            county_freq[i] = [0] * G
    deadlocks = 0

    for _ in range(n_sim):
        result = draw_once(strength, rng)
        if result is None:
            deadlocks += 1
            continue
        success += 1
        for ci in CITY_IDX:
            city_freq[ci][result[ci]] += 1
        for ci in county_freq:
            county_freq[ci][result[ci]] += 1

    # 均匀性：计算各市级队进组概率的最大偏差（理论值 1/16）
    max_dev = 0.0
    mean_dev = 0.0
    dev_count = 0
    all_devs = []
    for ci in CITY_IDX:
        total = sum(city_freq[ci])
        if total == 0:
            continue
        probs = [c / total for c in city_freq[ci]]
        for p in probs:
            d = abs(p - 1/G)
            all_devs.append(d)
            max_dev = max(max_dev, d)
            mean_dev += d
            dev_count += 1
    mean_dev /= dev_count if dev_count > 0 else 1

    # 县级队均匀性：按地级市汇总，统计同市县级队的入组偏差
    from utils import CITY_OF, CITIES
    county_dev_by_pref: dict[str, dict] = {}
    for city in CITIES:
        counties = PREFECTURE.get(city, [])
        if not counties:
            continue
        c_indices = [TEAM_IDX[c] for c in counties]
        # 对于该市代管的县级队，排除被管辖市级队所在组后的理论合法组数 = G-1 = 15
        theory_p = 1.0 / (G - 1)
        all_c_devs = []
        for ci in c_indices:
            total = sum(county_freq[ci])
            if total == 0:
                continue
            probs = [c / total for c in county_freq[ci]]
            for p in probs:
                all_c_devs.append(abs(p - theory_p))
        county_dev_by_pref[city] = {
            "n_counties": len(counties),
            "max_dev": float(max(all_c_devs)) if all_c_devs else 0.0,
            "mean_dev": float(np.mean(all_c_devs)) if all_c_devs else 0.0,
        }

    return {
        "n_sim": n_sim,
        "success": success,
        "deadlocks": deadlocks,
        "success_rate": success / n_sim,
        "max_prob_deviation": max_dev,
        "mean_prob_deviation": mean_dev,
        "city_freq": {
            TEAMS[ci]: city_freq[ci] for ci in CITY_IDX
        },
        "county_dev_by_pref": county_dev_by_pref,
    }


def main():
    import json
    master, _, _ = load_master()
    strength = master["strength"].values

    print("运行蒙特卡洛抽签模拟…")
    stats = monte_carlo(strength, n_sim=N_SIM, seed=42)

    print(f"模拟次数:     {stats['n_sim']}")
    print(f"成功次数:     {stats['success']}")
    print(f"死锁次数:     {stats['deadlocks']}")
    print(f"成功率:       {stats['success_rate']:.4f}")
    print(f"市级队入组概率最大偏差（vs 1/{G}={1/G:.4f}）: {stats['max_prob_deviation']:.4f}")
    print(f"市级队入组概率平均偏差: {stats.get('mean_prob_deviation', float('nan')):.4f}")

    # 按地级市展示县级队入组偏差
    print("\n县级队入组均匀性（按地级市汇总，理论基准 1/15 ~= 0.0667）:")
    county_dev = stats.get("county_dev_by_pref", {})
    for city in sorted(county_dev.keys(),
                       key=lambda c: county_dev[c]["max_dev"], reverse=True):
        d = county_dev[city]
        print(f"  {city}（{d['n_counties']}支县级队）: "
              f"max_dev={d['max_dev']:.4f}  mean_dev={d['mean_dev']:.4f}")

    out_path = OUT_DIR / "p2_draw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
