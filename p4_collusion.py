"""
问题四：串谋风险量化（Bradley-Terry 三状态模型 + 比赛重要性 MI）

BT 三状态模型（Csató 2020）：
  P(i 胜 j) = theta_i / (theta_i + theta_j + nusqrt(theta_itheta_j))
  P(平局)   = nusqrt(theta_itheta_j) / (theta_i + theta_j + nusqrt(theta_itheta_j))
  P(j 胜 i) = theta_j / (theta_i + theta_j + nusqrt(theta_itheta_j))

MI（比赛重要性）：
  MI_i = |Q_i^win - Q_i^loss|
  Q_i^win/loss = 以当前积分+获胜/落败为条件，队伍 i 晋级的概率

若 MI_i < eps=0.05，视为 i 对该场比赛"无利益相关"。
若比赛双方 MI 均 < eps，该场比赛为 Stakeless Match。

对比积分制度：A=3-1-0，B=3-0-0。
--------------------------------------------------
选择 3-0-0 而非 2-1-0 作为对照组的理由：
  · 2-1-0 与 3-1-0 的区别仅在于胜分比例（3 vs 2），但两者均对平局给 1 分。
    由于积分只影响排名（序），乘以常数不改变大小关系，故两制度下的晋级概率
    完全相同，Stakeless 率也相同——对比无意义。
  · 3-0-0（赢者通吃）则从根本上改变了平局的激励：平局不得分，
    与负场等价，迫使双方必须争胜。这才是真正影响串谋动机的制度变量。
  · 因此 3-1-0 vs 3-0-0 才能揭示"给平局发分"对 Stakeless Match 频率的净效应。
"""

import json
import random

import numpy as np

from utils import load_master, OUT_DIR

# -- 超参数 ------------------------------------------------------------------
NU_LIST  = [0.5, 1.0, 2.0]   # 平局倾向参数（灵敏度分析）
EPSILON  = 0.05               # Stakeless Match 阈值
N_SIM    = 10_000             # 模拟次数
POINTS_SYSTEMS = {
    "3-1-0": (3, 1, 0),   # 现行制度：胜3平1负0
    "3-0-0": (3, 0, 0),   # 对照制度：赢者通吃，平局等同于负
}
# 小组内循环赛轮次安排（4支队，3轮）
ROUNDS = [
    [(0, 1), (2, 3)],   # 第1轮
    [(0, 2), (1, 3)],   # 第2轮
    [(0, 3), (1, 2)],   # 第3轮（最后一轮）
]


def _bt_probs(ti: float, tj: float, nu: float) -> tuple[float, float, float]:
    """返回 (P_win, P_draw, P_loss)"""
    denom = ti + tj + nu * (ti * tj) ** 0.5
    return ti / denom, nu * (ti * tj) ** 0.5 / denom, tj / denom


def _simulate_match(ti: float, tj: float, nu: float,
                    rng: random.Random) -> int:
    """返回从队 i 视角：1=胜，0=平，-1=负"""
    pw, pd, _ = _bt_probs(ti, tj, nu)
    r = rng.random()
    if r < pw:
        return 1
    elif r < pw + pd:
        return 0
    return -1


def _qualifies(pts: list[int], rng: random.Random) -> list[bool]:
    """
    积分排名前 2 晋级。
    平局（多队同分占据第2名）时随机抽签决定，避免系统性偏差。
    """
    sorted_unique = sorted(set(pts), reverse=True)
    # 第二名分数门槛
    second_score = sorted_unique[1] if len(sorted_unique) >= 2 else sorted_unique[0]

    # 明确晋级：积分 > 第二名分数线
    definite_in  = [i for i, p in enumerate(pts) if p > second_score]
    # 明确淘汰：积分 < 第二名分数线
    # 平局区：积分 == 第二名分数线
    tied_at_cut  = [i for i, p in enumerate(pts) if p == second_score]

    slots_needed = 2 - len(definite_in)   # 还需从 tied_at_cut 中抽几队
    if slots_needed <= 0:
        # 积分前两名无平局，definite_in 已有 2 队（此情况 len==2）
        lucky = []
    elif slots_needed >= len(tied_at_cut):
        lucky = tied_at_cut
    else:
        lucky = rng.sample(tied_at_cut, slots_needed)

    qualified = set(definite_in) | set(lucky)
    return [i in qualified for i in range(4)]


def _q_win_loss(
    team_idx: int,
    match_pair: list[int],
    pts: list[int],
    thetas: list[float],
    nu: float,
    pt_sys: tuple[int, int, int],
    n_inner: int = 2000,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """
    返回 (Q_win, Q_loss)：
    以 team_idx 在 match_pair 中获胜/落败为条件，估计其晋级概率。
    另一场比赛结果通过蒙特卡洛（n_inner 次）边缘化。
    """
    if rng is None:
        rng = random.Random(0)

    win_pts, draw_pts, loss_pts = pt_sys
    i_in_match = match_pair.index(team_idx)

    other_pair = [t for t in range(4) if t not in match_pair]
    oa, ob = other_pair

    q_win = 0.0
    q_loss = 0.0

    for _ in range(n_inner):
        out_other = _simulate_match(thetas[oa], thetas[ob], nu, rng)
        pts_delta = [0] * 4
        if out_other == 1:
            pts_delta[oa] += win_pts; pts_delta[ob] += loss_pts
        elif out_other == 0:
            pts_delta[oa] += draw_pts; pts_delta[ob] += draw_pts
        else:
            pts_delta[oa] += loss_pts; pts_delta[ob] += win_pts

        # 本场获胜时
        pts_win = pts[:]
        if i_in_match == 0:
            pts_win[match_pair[0]] += win_pts
            pts_win[match_pair[1]] += loss_pts
        else:
            pts_win[match_pair[1]] += win_pts
            pts_win[match_pair[0]] += loss_pts
        pts_win = [pts_win[t] + pts_delta[t] for t in range(4)]
        if _qualifies(pts_win, rng)[team_idx]:
            q_win += 1

        # 本场落败时
        pts_loss = pts[:]
        if i_in_match == 0:
            pts_loss[match_pair[0]] += loss_pts
            pts_loss[match_pair[1]] += win_pts
        else:
            pts_loss[match_pair[1]] += loss_pts
            pts_loss[match_pair[0]] += win_pts
        pts_loss = [pts_loss[t] + pts_delta[t] for t in range(4)]
        if _qualifies(pts_loss, rng)[team_idx]:
            q_loss += 1

    return q_win / n_inner, q_loss / n_inner


def simulate_group(
    thetas: list[float],
    nu: float,
    pt_sys: tuple[int, int, int],
    rng: random.Random,
) -> dict:
    """
    模拟一个小组的完整赛事，返回：
      has_stakeless   最后一轮是否存在 Stakeless Match
      mi_final        最后一轮各场各队的 MI 值
    """
    win_pts, draw_pts, loss_pts = pt_sys
    pts = [0] * 4

    for round_matches in ROUNDS[:2]:
        for i, j in round_matches:
            out = _simulate_match(thetas[i], thetas[j], nu, rng)
            if out == 1:
                pts[i] += win_pts; pts[j] += loss_pts
            elif out == 0:
                pts[i] += draw_pts; pts[j] += draw_pts
            else:
                pts[i] += loss_pts; pts[j] += win_pts

    has_stakeless = False
    mi_vals = {}
    for match in ROUNDS[2]:
        i, j = match
        qi_w, qi_l = _q_win_loss(i, list(match), pts, thetas, nu, pt_sys, rng=rng)
        qj_w, qj_l = _q_win_loss(j, list(match), pts, thetas, nu, pt_sys, rng=rng)
        mi_i = abs(qi_w - qi_l)
        mi_j = abs(qj_w - qj_l)
        mi_vals[match] = (mi_i, mi_j)
        if mi_i < EPSILON and mi_j < EPSILON:
            has_stakeless = True

    return {"has_stakeless": has_stakeless, "mi": mi_vals}


def main():
    master, _, _ = load_master()

    import json
    p1_path = OUT_DIR / "p1_grouping.json"
    if not p1_path.exists():
        print("请先运行 p1_grouping.py")
        return
    with open(p1_path, encoding="utf-8") as f:
        p1_data = json.load(f)
    best = min(p1_data, key=lambda r: r["violations"])
    groups = best["groups"]

    print(f"使用问题一方案（seed={best['seed']}），共 {len(groups)} 个小组")
    print(f"模拟 {N_SIM} 次，eps={EPSILON}，nuin{NU_LIST}")
    print()
    print("【对照系统说明】")
    print("  3-1-0：现行制度（胜3分，平1分，负0分）")
    print("  3-0-0：对照制度（赢者通吃；平局与负场同等，均不得分）")
    print("  --以下说明为何不选 2-1-0 作为对照--")
    print("  2-1-0 与 3-1-0 均对平局奖励 1 分，差异仅在胜分（2 vs 3）。")
    print("  由于晋级取决于积分排名（序关系），等比例缩放不改变顺序，")
    print("  故两者的晋级概率 Q、比赛重要性 MI 完全相同，Stakeless 率亦然。")
    print("  只有移除平局奖励（3-0-0）才真正改变激励结构，才能测量出差异。")
    print()

    from utils import TEAM_IDX
    theta_vals = master["theta"].values

    results = {}
    for nu in NU_LIST:
        for sys_name, pt_sys in POINTS_SYSTEMS.items():
            rng = random.Random(42)
            stakeless_count = 0
            total = 0

            for grp in groups:
                thetas_g = [float(theta_vals[TEAM_IDX[t]]) for t in grp]
                for _ in range(N_SIM // len(groups)):
                    res = simulate_group(thetas_g, nu, pt_sys, rng)
                    if res["has_stakeless"]:
                        stakeless_count += 1
                    total += 1

            rate = stakeless_count / total if total > 0 else 0
            key  = f"nu={nu}_{sys_name}"
            results[key] = {
                "nu": nu,
                "point_system": sys_name,
                "stakeless_rate": round(rate, 5),
                "n_simulated": total,
            }
            print(f"  nu={nu:3.1f}  {sys_name}: Stakeless 率 = {rate:.4f}")

    out_path = OUT_DIR / "p4_collusion.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至 {out_path}")

    print("\n-- 积分制度对比（nu=1.0）--")
    for sys_name in POINTS_SYSTEMS:
        k = f"nu=1.0_{sys_name}"
        r = results[k]["stakeless_rate"]
        print(f"  {sys_name}: Stakeless 率 = {r:.4f}")
    r_a = results["nu=1.0_3-1-0"]["stakeless_rate"]
    r_b = results["nu=1.0_3-0-0"]["stakeless_rate"]
    delta = r_b - r_a
    print(f"  差值（3-0-0 − 3-1-0）= {delta:+.4f}")
    if delta < 0:
        print("  --> 取消平局奖励（3-0-0）降低了 Stakeless 风险，")
        print("    因为平局不再有晋级价值，双方必须争胜，比赛结果更具决定性。")
    else:
        print("  --> 取消平局奖励（3-0-0）反而提高了 Stakeless 风险，")
        print("    可能因为均分情形下排名更易被最后一轮以外的结果锁定。")


# --- 方向二：淘汰赛抽签约束的公平性模拟 ---------------------------------

def _ko_match(team_a: int, team_b: int, thetas: list[float],
              rng: random.Random) -> int:
    """BT 模型模拟单场淘汰赛，返回胜者（无平局，重抽直到分出胜负）"""
    ti, tj = thetas[team_a], thetas[team_b]
    while True:
        pw, pd, pl = _bt_probs(ti, tj, nu=0)  # nu=0 → 二态 BT，无平局
        r = rng.random()
        if r < pw:
            return team_a
        elif r < pw + pd:
            continue  # 平局重赛
        else:
            return team_b


def _strong_teams(thetas: np.ndarray, team_indices: list[int],
                  threshold: float = 1.5) -> set[int]:
    """识别强队（实力参数超过阈值的队伍）"""
    return {i for i in team_indices if thetas[i] > threshold}


def simulate_ko_draw(groups: list[list[str]], thetas: np.ndarray,
                     restrict: bool, n_sim: int = 5000,
                     seed: int = 42) -> dict:
    """
    模拟 32 强淘汰赛抽签，对比有无"同组回避"约束。

    restrict=True:  同组两队不可进入同一半区（类比 UEFA 约束）
    restrict=False: 完全随机抽签（无约束）

    返回详细统计：逐轮强队相遇率、冠军分布等。
    """
    from utils import TEAM_IDX

    team_idx = [TEAM_IDX[t] for g in groups for t in g[:2]]  # 每组前2名晋级
    assert len(team_idx) == 32
    theta_list = [float(thetas[i]) for i in team_idx]

    rng = random.Random(seed)
    strong = _strong_teams(thetas, team_idx, threshold=1.5)

    # 统计量
    early_strong_meet = 0  # 前两轮（32→16→8）强队相遇次数
    round1_meet = 0        # 仅第一轮
    round2_meet = 0        # 仅第二轮
    strong_champion = 0    # 强队夺冠次数
    strong_final = 0       # 强队进决赛次数
    total_brackets = 0

    for _ in range(n_sim):
        # 将 32 队随机排入淘汰赛对阵表（位置 0-31）
        bracket = team_idx[:]
        rng.shuffle(bracket)

        if restrict:
            group_of = {}
            for g_idx, g in enumerate(groups):
                for t in g[:2]:
                    group_of[TEAM_IDX[t]] = g_idx

            # 拒绝采样：同组队不得在同一半区
            max_attempts = 200
            for _ in range(max_attempts):
                valid = True
                for g_idx, g in enumerate(groups):
                    t0, t1 = TEAM_IDX[g[0]], TEAM_IDX[g[1]]
                    pos0 = bracket.index(t0)
                    pos1 = bracket.index(t1)
                    if (pos0 < 16) == (pos1 < 16):
                        valid = False
                        break
                if valid:
                    break
                rng.shuffle(bracket)

        # 模拟淘汰赛，逐轮记录
        current = bracket[:]
        round_num = 0
        while len(current) > 1:
            next_round = []
            for i in range(0, len(current), 2):
                a, b = current[i], current[i + 1]
                winner = _ko_match(a, b, thetas, rng)
                next_round.append(winner)
                # 统计强队相遇
                if round_num == 0 and a in strong and b in strong:
                    round1_meet += 1
                elif round_num == 1 and a in strong and b in strong:
                    round2_meet += 1
            current = next_round
            round_num += 1

        # 决赛/冠军（最后一轮）
        if len(current) >= 2:
            if current[0] in strong and current[1] in strong:
                strong_final += 1
            # 冠军
            winner = _ko_match(current[0], current[1], thetas, rng)
            if winner in strong:
                strong_champion += 1
        elif len(current) == 1 and current[0] in strong:
            strong_champion += 1

        # 前两轮汇总
        for s1 in strong:
            for s2 in strong:
                if s1 >= s2:
                    continue
                pos1 = bracket.index(s1)
                pos2 = bracket.index(s2)
                r1 = (pos1 // 2) == (pos2 // 2)
                r2 = (pos1 // 4) == (pos2 // 4) and not r1
                if r1 or r2:
                    early_strong_meet += 1
        total_brackets += 1

    return {
        "restricted": restrict,
        "n_sim": n_sim,
        "n_strong": len(strong),
        "early_strong_meet_rate": early_strong_meet / total_brackets
            if total_brackets > 0 else 0,
        "round1_strong_meet_rate": round1_meet / (n_sim * 16),
        "round2_strong_meet_rate": round2_meet / (n_sim * 8),
        "strong_final_rate": strong_final / n_sim,
        "strong_champion_rate": strong_champion / n_sim,
    }


# --- 方向三：跨阶段衔接分析 --------------------------------------------

def analyze_cross_stage_transition(
    groups: list[list[str]],
    group_venues: list[str],
    ko_candidates: list[str],
    dist: np.ndarray,
) -> dict:
    """
    分析从小组赛赛区到淘汰赛举办地的跨阶段衔接成本。

    输入：
      groups:        小组赛分组（16 组，每组 4 队）
      group_venues:  各小组的赛区地点名称（16 个）
      ko_candidates: 淘汰赛候选举办地（如 ["杭州市"] 或 ["杭州市", "宁波市"]）
      dist:          64×64 距离矩阵

    返回：
      各晋级队伍到淘汰赛举办地的出行距离统计。
    """
    from utils import TEAM_IDX

    # 模拟每组前两名晋级（按实力最强两队作为预期晋级者）
    master, _, _ = load_master()
    strength = master["strength"].values

    travel_to_ko = []
    for g_idx, grp in enumerate(groups):
        # 每组预期晋级的前两名（按实力）
        grp_strength = [(t, strength[TEAM_IDX[t]]) for t in grp]
        grp_strength.sort(key=lambda x: x[1], reverse=True)
        qualifiers = [t for t, _ in grp_strength[:2]]

        # 从小组赛赛区到淘汰赛举办地的距离
        gv = group_venues[g_idx]
        for ko_city in ko_candidates:
            for q in qualifiers:
                d = dist[TEAM_IDX[gv], TEAM_IDX[ko_city]]
                travel_to_ko.append({
                    "team": q,
                    "from_venue": gv,
                    "to_ko": ko_city,
                    "distance_km": float(d),
                })

    import pandas as pd
    df = pd.DataFrame(travel_to_ko)
    summary = {
        "ko_candidates": ko_candidates,
        "total_travel_km": float(df["distance_km"].sum()),
        "mean_travel_km": float(df["distance_km"].mean()),
        "max_travel_km": float(df["distance_km"].max()),
        "min_travel_km": float(df["distance_km"].min()),
    }
    if len(ko_candidates) > 1:
        for ko in ko_candidates:
            sub = df[df["to_ko"] == ko]
            summary[f"{ko}_mean_km"] = float(sub["distance_km"].mean())
            summary[f"{ko}_total_km"] = float(sub["distance_km"].sum())

    return summary


def analyze_bracket_travel(
    groups: list[list[str]],
    group_venues: list[str],
    ko_venue: str,
    dist: np.ndarray,
) -> dict:
    """
    分析淘汰赛对阵表在不同排列下的出行负担。

    32 支晋级队伍从各自小组赛赛区前往淘汰赛举办地，
    总出行距离 = sum_{晋级队} dist(赛区, ko_venue)。
    对比：若淘汰赛分散在多个赛区 vs 集中于单一赛区。
    """
    from utils import TEAM_IDX

    master, _, _ = load_master()
    strength = master["strength"].values

    travel = []
    for g_idx, grp in enumerate(groups):
        grp_strength = [(t, strength[TEAM_IDX[t]]) for t in grp]
        grp_strength.sort(key=lambda x: x[1], reverse=True)
        qualifiers = [t for t, _ in grp_strength[:2]]
        gv = group_venues[g_idx]
        for q in qualifiers:
            d = dist[TEAM_IDX[gv], TEAM_IDX[ko_venue]]
            travel.append(float(d))

    return {
        "ko_venue": ko_venue,
        "n_qualifiers": len(travel),
        "total_travel_km": sum(travel),
        "mean_travel_km": np.mean(travel),
        "max_travel_km": max(travel),
        "min_travel_km": min(travel),
        "std_travel_km": float(np.std(travel)),
    }


def main_ko():
    """运行淘汰赛抽签公平性对比 + 跨阶段衔接分析"""
    import json
    from utils import OUT_DIR, load_distance

    master, _, _ = load_master()
    p1_path = OUT_DIR / "p1_grouping.json"
    with open(p1_path, encoding="utf-8") as f:
        p1_data = json.load(f)
    best = min(p1_data, key=lambda r: r["violations"])
    groups = best["groups"]
    thetas = master["theta"].values

    print("=" * 60)
    print("淘汰赛抽签约束公平性模拟（32 强）…")
    print("=" * 60)
    ko_results = {}
    for restrict in [False, True]:
        res = simulate_ko_draw(groups, thetas, restrict=restrict, n_sim=5000)
        label = "有同组回避约束" if restrict else "无约束（完全随机）"
        ko_results["restricted" if restrict else "unrestricted"] = res
        print(f"\n  {label}:")
        print(f"    强队数量: {res['n_strong']} 支（theta > 1.5）")
        print(f"    第1轮强队相遇率: {res['round1_strong_meet_rate']:.4f}")
        print(f"    第2轮强队相遇率: {res['round2_strong_meet_rate']:.4f}")
        print(f"    前两轮强队相遇率: {res['early_strong_meet_rate']:.4f}")
        print(f"    强队进决赛率: {res['strong_final_rate']:.4f}")
        print(f"    强队夺冠率: {res['strong_champion_rate']:.4f}")

    out_path = OUT_DIR / "p4_ko.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ko_results, f, ensure_ascii=False, indent=2)
    print(f"\n淘汰赛结果已保存至 {out_path}")

    # -- 跨阶段衔接分析 --
    print("\n" + "=" * 60)
    print("跨阶段衔接分析（小组赛→淘汰赛出行成本）")
    print("=" * 60)

    dist = load_distance()

    # 加载 P3 赛区结果（如果没有，使用 P3 最优方案的赛区）
    p3_path = OUT_DIR / "p3_venue.json"
    if p3_path.exists():
        with open(p3_path, encoding="utf-8") as f:
            p3_data = json.load(f)
        best_p3 = min(p3_data, key=lambda r: r["total_travel_km"] if r["feasible"] else float("inf"))
        group_venues = [g["venue"] for g in best_p3["groups"]]
    else:
        # 回退：假设各队在自己所在城市比赛
        group_venues = ["杭州市"] * 16

    # 情景1：淘汰赛集中在杭州（省会）
    ko_hangzhou = analyze_bracket_travel(groups, group_venues, "杭州市", dist)
    print(f"\n情景1：淘汰赛集中于杭州市（省会）")
    print(f"  总出行: {ko_hangzhou['total_travel_km']:.0f} km")
    print(f"  平均: {ko_hangzhou['mean_travel_km']:.0f} km/队")
    print(f"  最大: {ko_hangzhou['max_travel_km']:.0f} km")
    print(f"  标准差: {ko_hangzhou['std_travel_km']:.0f} km")

    # 情景2：淘汰赛分散在小组赛赛区中最大的两个城市
    top_venues = sorted(set(group_venues), key=lambda v: sum(
        dist[TEAM_IDX[v], TEAM_IDX[t]] for t in [g["teams"][0] for g in best_p3["groups"]]
    ))[:2] if p3_path.exists() else ["杭州市", "宁波市"]

    transition_results = {
        "centralized_ko": ko_hangzhou,
        "group_venues": group_venues,
    }

    trans_path = OUT_DIR / "p4_transition.json"
    with open(trans_path, "w", encoding="utf-8") as f:
        json.dump(transition_results, f, ensure_ascii=False, indent=2)
    print(f"\n跨阶段衔接结果已保存至 {trans_path}")


if __name__ == "__main__":
    main()
    main_ko()
