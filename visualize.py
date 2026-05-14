"""
生成论文所需全部图表，保存至 paper/figures/
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent.parent
OUTDIR  = ROOT / "paper" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
DATA    = ROOT / "data" / "clean"
CODEOUT = ROOT / "code" / "output"

# 统一字体与样式
# ctexart 正文用 Songti SC（宋体），标题/图注用 STHeiti（黑体）
# matplotlib 里正文/轴标签用宋体，图标题用黑体，保持与论文一致
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Songti SC", "STSong", "SimSun", "DejaVu Serif"],
    "font.sans-serif":    ["STHeiti", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.1,
    "axes.titleweight":   "bold",
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
})

PALETTE = ["#2166ac", "#d6604d", "#4dac26", "#f4a582",
           "#762a83", "#e08214", "#1b7837", "#b2182b"]

# P2 热图数据（fig_p2_heatmap 产出，fig_p2_deviation 消费）
_p2_dev = None
_p2_city_names = None


# -----------------------------------------------------------------------------
# 图1：问题一 — 各组综合实力分布（条形图）
# -----------------------------------------------------------------------------
def fig_p1_strength():
    from utils import TEAMS, TEAM_IDX, CITIES, load_master

    master, w1, w2 = load_master()
    strength = master["strength"].values

    p1 = json.load(open(CODEOUT / "p1_grouping.json"))
    best = min(p1, key=lambda r: r["violations"])
    groups = best["groups"]

    group_strengths = [sum(strength[TEAM_IDX[t]] for t in g) for g in groups]
    has_city = [any(t in CITIES for t in g) for g in groups]
    colors = ["#2166ac" if h else "#d6604d" for h in has_city]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(1, 17), group_strengths, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(np.mean(group_strengths), color="gray", linestyle="--", linewidth=1.2,
               label=f"均值 {np.mean(group_strengths):.2f}")
    ax.set_xlabel("小组编号", fontsize=11)
    ax.set_ylabel("综合实力总分", fontsize=11)
    ax.set_title("各小组综合实力分布（问题一 ILP 最优方案）", fontsize=12)
    ax.set_xticks(range(1, 17))
    ax.set_xlim(0.3, 16.7)
    p1 = mpatches.Patch(color="#2166ac", label="含市级队小组（11组）")
    p2 = mpatches.Patch(color="#d6604d", label="无市级队小组（5组）")
    ax.legend(handles=[p1, p2, ax.get_lines()[0]], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "p1_strength.pdf")
    fig.savefig(OUTDIR / "p1_strength.png")
    plt.close(fig)
    print("[ok] fig_p1_strength")



# -----------------------------------------------------------------------------
# 图2a：问题二 — 市级队入组概率偏差热图
# -----------------------------------------------------------------------------
def fig_p2_heatmap():
    import sys; sys.path.insert(0, str(ROOT / "code"))
    from utils import load_master
    from p2_draw import monte_carlo

    master, _, _ = load_master()
    strength = master["strength"].values
    stats = monte_carlo(strength, n_sim=10000, seed=42)
    freq = stats["city_freq"]

    city_names = list(freq.keys())
    mat = np.array([freq[c] for c in city_names], dtype=float)
    row_sum = mat.sum(axis=1, keepdims=True)
    mat = mat / np.where(row_sum > 0, row_sum, 1)

    theory = 1 / 16
    dev = mat - theory

    vmax = max(abs(dev.max()), abs(dev.min()), 0.01)
    fig, ax = plt.subplots(figsize=(12, 5))

    im = ax.imshow(dev, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(16))
    ax.set_xticklabels([f"G{g+1}" for g in range(16)], fontsize=10)
    ax.set_yticks(range(len(city_names)))
    ax.set_yticklabels(city_names, fontsize=11)
    ax.set_xlabel("小组编号", fontsize=12)
    ax.set_title("各市级队入组概率偏差热图（实测 $-$ 理论均等值 1/16）", fontsize=13)
    for i in range(len(city_names)):
        for j in range(16):
            val = dev[i, j]
            col = "white" if abs(val) > vmax * 0.55 else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=7.5, color=col)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label("概率偏差", fontsize=11)

    total_max = float(np.abs(dev).max())
    fig.suptitle(f"随机抽签公平性验证 —— 蒙特卡洛 10000 次，全局最大偏差 {total_max:.4f}",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTDIR / "p2_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "p2_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig_p2_heatmap")

    # 将偏差数据存为全局变量供 fig_p2_deviation 使用
    global _p2_dev, _p2_city_names
    _p2_dev = dev
    _p2_city_names = city_names


# -----------------------------------------------------------------------------
# 图2b：问题二 — 各市级队最大偏差柱状图
# -----------------------------------------------------------------------------
def fig_p2_deviation():
    global _p2_dev, _p2_city_names
    if "_p2_dev" not in dir() or _p2_dev is None:
        fig_p2_heatmap()  # 先跑热图获取数据

    dev = _p2_dev
    city_names = _p2_city_names
    max_dev = np.abs(dev).max(axis=1)
    # 排序
    order = np.argsort(max_dev)[::-1]
    sorted_names = [city_names[i] for i in order]
    sorted_devs = max_dev[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_bar = ["#d6604d" if d > 0.003 else "#4393c3" for d in sorted_devs]
    bars = ax.barh(range(len(sorted_names)), sorted_devs, color=colors_bar,
                   edgecolor="white", height=0.65)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.set_xlabel("最大绝对偏差", fontsize=12)
    ax.set_title("各市级队入组概率最大绝对偏差", fontsize=13)
    ax.axvline(np.mean(sorted_devs), color="gray", linestyle="--", linewidth=1.2,
               label=f"均值 {np.mean(sorted_devs):.4f}")
    ax.legend(fontsize=10, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(sorted_devs) * 1.15)
    fig.tight_layout()
    fig.savefig(OUTDIR / "p2_deviation.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "p2_deviation.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig_p2_deviation")


# -----------------------------------------------------------------------------
# 图3：问题三 — 模拟退火收敛曲线
# -----------------------------------------------------------------------------
def fig_p3_convergence():
    hist_data = json.load(open(CODEOUT / "p3_convergence.json"))
    seeds = [42, 123, 2026]
    colors_c = ["#2166ac", "#d6604d", "#4dac26"]

    fig, ax = plt.subplots(figsize=(8, 4))
    for seed, col in zip(seeds, colors_c):
        hist = hist_data[str(seed)]
        iters = [h[0] for h in hist]
        vals  = [h[1] for h in hist]
        # 只画总出行（<=1e5 的部分，去掉初期高惩罚）
        clipped = [min(v, 5e4) for v in vals]
        ax.plot(iters, clipped, color=col, linewidth=1.4, label=f"seed={seed}")

    ax.set_xlabel("迭代次数", fontsize=11)
    ax.set_ylabel("最优目标值（km）", fontsize=11)
    ax.set_title("模拟退火收敛过程（问题三，三次独立运行）", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "p3_convergence.pdf")
    fig.savefig(OUTDIR / "p3_convergence.png")
    plt.close(fig)
    print("[ok] fig_p3_convergence")


# -----------------------------------------------------------------------------
# 图4：问题三 — 浙江地图 + 赛区标注
# -----------------------------------------------------------------------------
def fig_p3_map():
    import sys; sys.path.insert(0, str(ROOT / "code"))
    from utils import TEAMS, TEAM_IDX, CITIES, CITY_OF

    master = pd.read_csv(DATA / "teams_master.csv").set_index("地区")
    master = master.reindex(TEAMS)

    p3 = json.load(open(CODEOUT / "p3_venue.json"))
    best3 = min(p3, key=lambda r: r["total_travel_km"] if r["feasible"] else float("inf"))
    venue_names = best3["venues"]

    # 各队坐标
    lons = master["经度"].values
    lats = master["纬度"].values

    venue_idx = [TEAM_IDX[v] for v in venue_names]
    venue_lons = lons[venue_idx]
    venue_lats = lats[venue_idx]

    # 转换到 Web Mercator（EPSG:3857）以叠加 contextily 底图
    try:
        import contextily as ctx
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        mx, my = transformer.transform(lons, lats)
        vx, vy = transformer.transform(venue_lons, venue_lats)
        use_webmap = True
    except Exception:
        mx, my = lons, lats
        vx, vy = venue_lons, venue_lats
        use_webmap = False

    fig, ax = plt.subplots(figsize=(10, 8))

    # 小组与赛区连线
    group_info = best3["groups"]
    for g in group_info:
        v_idx = TEAM_IDX[g["venue"]]
        for team in g["teams"]:
            t_idx = TEAM_IDX[team]
            if team not in venue_names:
                ax.plot([mx[t_idx], vx[venue_names.index(g["venue"])]],
                        [my[t_idx], vy[venue_names.index(g["venue"])]],
                        color="gray", alpha=0.25, linewidth=0.7, zorder=2)

    city_mask   = np.array([t in CITIES for t in TEAMS])
    county_mask = ~city_mask

    ax.scatter(mx[county_mask], my[county_mask],
               c="#aec7e8", s=25, zorder=3, label="县级队", alpha=0.9)
    ax.scatter(mx[city_mask], my[city_mask],
               c="#1f77b4", s=65, zorder=4, marker="^", label="市级队")
    ax.scatter(vx, vy,
               c="#d62728", s=220, zorder=6, marker="*", label="赛区",
               edgecolors="white", linewidths=0.9)

    VENUE_COLORS = [
        "#e41a1c","#377eb8","#4daf4a","#984ea3",
        "#ff7f00","#a65628","#f781bf","#999999",
    ]
    for idx, (name, x0, y0) in enumerate(zip(venue_names, vx, vy)):
        ax.scatter([x0], [y0], c=VENUE_COLORS[idx], s=220, zorder=6,
                   marker="*", edgecolors="white", linewidths=0.9)
        ax.annotate(name, (x0, y0),
                    textcoords="offset points", xytext=(7, 5),
                    fontsize=8, color="#900000", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              alpha=0.75, ec="none"),
                    zorder=7)

    if use_webmap:
        try:
            ctx.add_basemap(ax, crs="EPSG:3857",
                            source=ctx.providers.CartoDB.Positron,
                            zoom=8, alpha=0.85)
            ax.set_xlabel("经度（Web Mercator）", fontsize=9)
            ax.set_ylabel("纬度（Web Mercator）", fontsize=9)
        except Exception:
            ax.set_xlabel("经度 (°E)", fontsize=10)
            ax.set_ylabel("纬度 (°N)", fontsize=10)
    else:
        ax.set_xlim(118.0, 122.9)
        ax.set_ylim(27.0, 31.2)
        ax.set_aspect("equal")
        ax.set_xlabel("经度 (°E)", fontsize=10)
        ax.set_ylabel("纬度 (°N)", fontsize=10)

    ax.set_title(
        f"浙超赛区分布图（8个赛区，覆盖{best3['city_coverage']}个地级市）",
        fontsize=12)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTDIR / "p3_map.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "p3_map.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig_p3_map")


# -----------------------------------------------------------------------------
# 图5：问题三 — 各组出行距离分布（横向条形图）
# -----------------------------------------------------------------------------
def fig_p3_travel():
    p3 = json.load(open(CODEOUT / "p3_venue.json"))
    best3 = min(p3, key=lambda r: r["total_travel_km"] if r["feasible"] else float("inf"))
    groups_info = best3["groups"]
    g_labels = [f"G{g['group']}" for g in groups_info]
    travels   = [g["travel_km"] for g in groups_info]
    venues    = [g["venue"] for g in groups_info]

    # 按出行距离排序
    order = np.argsort(travels)
    g_labels = [g_labels[i] for i in order]
    travels   = [travels[i]   for i in order]
    venues    = [venues[i]    for i in order]

    mean_t = best3["mean_travel_km"]
    thresh = best3.get("delta_fixed_km", best3.get("delta_threshold_km", 0))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors_b = ["#d6604d" if abs(t - mean_t) > thresh * 0.7 else "#4393c3" for t in travels]
    bars = ax.barh(range(16), travels, color=colors_b, edgecolor="white", height=0.7)
    ax.axvline(mean_t, color="gray", linestyle="--", linewidth=1.2, label=f"均值 {mean_t:.0f} km")
    ax.axvline(mean_t + thresh, color="orange", linestyle=":", linewidth=1,
               label=f"上限 {mean_t + thresh:.0f} km")
    ax.axvline(mean_t - thresh, color="orange", linestyle=":", linewidth=1)

    ax.set_yticks(range(16))
    ax.set_yticklabels([f"{g_labels[i]}\n@{venues[i]}" for i in range(16)], fontsize=7.5)
    ax.set_xlabel("出行总距离 (km)", fontsize=11)
    ax.set_title(f"各小组出行距离分布（极差 {best3['range_travel_km']:.0f} km < 阈值 {thresh:.0f} km）",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "p3_travel.pdf")
    fig.savefig(OUTDIR / "p3_travel.png")
    plt.close(fig)
    print("[ok] fig_p3_travel")


# -----------------------------------------------------------------------------
# 图6：问题四 — Stakeless 率对比（分组条形图）
# -----------------------------------------------------------------------------
def fig_p4_stakeless():
    p4 = json.load(open(CODEOUT / "p4_collusion.json"))

    nu_vals = [0.5, 1.0, 2.0]
    systems  = ["3-1-0", "3-0-0"]
    colors_s = {"3-1-0": "#2166ac", "3-0-0": "#d6604d"}
    labels_s = {"3-1-0": "3-1-0（现行制度）", "3-0-0": "3-0-0（对照：赢者通吃）"}

    x = np.arange(len(nu_vals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, sys in enumerate(systems):
        rates = [p4[f"nu={nu}_{sys}"]["stakeless_rate"] for nu in nu_vals]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=labels_s[sys],
                      color=colors_s[sys], edgecolor="white", alpha=0.9)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{rate:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.axhline(0.05, color="red", linestyle="--", linewidth=1,
               label="串谋阈值 eps=0.05")
    ax.set_xticks(x)
    ax.set_xticklabels([f"nu = {nu}" for nu in nu_vals], fontsize=11)
    ax.set_ylabel("Stakeless Match 率", fontsize=11)
    ax.set_title("不同积分制度下串谋风险（nu 灵敏度分析）", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.38)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "p4_stakeless.pdf")
    fig.savefig(OUTDIR / "p4_stakeless.png")
    plt.close(fig)
    print("[ok] fig_p4_stakeless")


# -----------------------------------------------------------------------------
# 图7：问题四 — MI 分布示意（两场最后一轮比赛的 MI 分布直方图）
# -----------------------------------------------------------------------------
def fig_p4_mi_dist():
    import sys; sys.path.insert(0, str(ROOT / "code"))
    from utils import load_master, TEAM_IDX
    from p4_collusion import simulate_group, ROUNDS
    import random

    master, _, _ = load_master()
    p1 = json.load(open(CODEOUT / "p1_grouping.json"))
    best = min(p1, key=lambda r: r["violations"])
    groups = best["groups"]
    theta_vals = master["theta"].values

    rng = random.Random(0)
    mi_vals_310  = []
    mi_vals_300  = []
    n_sim = 3000

    for grp in groups[:8]:   # 取前8组做示意
        thetas_g = [float(theta_vals[TEAM_IDX[t]]) for t in grp]
        for _ in range(n_sim // 8):
            for sys, store in [((3,1,0), mi_vals_310), ((3,0,0), mi_vals_300)]:
                res = simulate_group(thetas_g, nu=1.0, pt_sys=sys, rng=rng)
                for match, (mi_i, mi_j) in res["mi"].items():
                    store.append(mi_i)
                    store.append(mi_j)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, vals, title, col in zip(
        axes,
        [mi_vals_310, mi_vals_300],
        ["3-1-0（现行制度）", "3-0-0（赢者通吃）"],
        ["#2166ac", "#d6604d"],
    ):
        ax.hist(vals, bins=30, color=col, alpha=0.8, edgecolor="white")
        ax.axvline(0.05, color="red", linestyle="--", linewidth=1.2, label="eps = 0.05")
        frac = sum(v < 0.05 for v in vals) / len(vals)
        ax.set_title(f"{title}\n(MI<eps 占比 {frac:.1%})", fontsize=10)
        ax.set_xlabel("比赛重要性 MI", fontsize=10)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("频次", fontsize=10)
    fig.suptitle("最后一轮各队比赛重要性 MI 分布（nu=1.0）", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTDIR / "p4_mi_dist.pdf")
    fig.savefig(OUTDIR / "p4_mi_dist.png")
    plt.close(fig)
    print("[ok] fig_p4_mi_dist")


# -----------------------------------------------------------------------------
# 图8：问题二 — 县级队入组偏差热图（按地级市汇总）
# -----------------------------------------------------------------------------
def fig_p2_county_heatmap():
    p2 = json.load(open(CODEOUT / "p2_draw.json"))
    county_dev = p2.get("county_dev_by_pref", {})
    if not county_dev:
        print("[skip] fig_p2_county_heatmap — p2_draw.json 无 county_dev_by_pref 字段")
        return

    cities = list(county_dev.keys())
    max_devs = [county_dev[c]["max_dev"] for c in cities]
    mean_devs = [county_dev[c]["mean_dev"] for c in cities]
    n_counties = [county_dev[c]["n_counties"] for c in cities]

    # 按 max_dev 降序
    order = np.argsort(max_devs)[::-1]
    cities = [cities[i] for i in order]
    max_devs = [max_devs[i] for i in order]
    mean_devs = [mean_devs[i] for i in order]
    n_counties = [n_counties[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cities))
    width = 0.35
    bars1 = ax.bar(x - width/2, max_devs, width, color="#d6604d",
                   edgecolor="white", label="最大绝对偏差", alpha=0.85)
    bars2 = ax.bar(x + width/2, mean_devs, width, color="#4393c3",
                   edgecolor="white", label="平均绝对偏差", alpha=0.85)

    theory = 1/15
    ax.axhline(theory, color="gray", linestyle="--", linewidth=0.8,
               alpha=0.5)
    ax.annotate(f"理论基准 1/15", (len(cities)-0.3, theory),
                fontsize=7.5, color="gray", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({n}队)" for c, n in zip(cities, n_counties)],
                       fontsize=8.5)
    ax.set_ylabel("入组概率偏差", fontsize=11)
    ax.set_title("各市县级队入组概率偏差（按地级市汇总，vs 1/15）", fontsize=12)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # 标注最大偏差值
    for bar, val in zip(bars1, max_devs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)

    fig.tight_layout()
    fig.savefig(OUTDIR / "p2_county_heatmap.pdf")
    fig.savefig(OUTDIR / "p2_county_heatmap.png")
    plt.close(fig)
    print("[ok] fig_p2_county_heatmap")


# -----------------------------------------------------------------------------
# 图9：问题四 — 淘汰赛抽签约束公平性对比（分组条形图）
# -----------------------------------------------------------------------------
def fig_p4_bracket_fairness():
    ko = json.load(open(CODEOUT / "p4_ko.json"))
    metrics = [
        ("round1_strong_meet_rate", "第1轮强队相遇率"),
        ("round2_strong_meet_rate", "第2轮强队相遇率"),
        ("strong_final_rate",       "强队进决赛率"),
        ("strong_champion_rate",    "强队夺冠率"),
    ]
    labels = ["无约束", "同组回避"]
    values_unrestricted = [ko["unrestricted"][m[0]] for m in metrics]
    values_restricted   = [ko["restricted"][m[0]] for m in metrics]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    width = 0.32
    bars1 = ax.bar(x - width/2, values_unrestricted, width,
                   color="#2166ac", edgecolor="white", label="无约束（完全随机）", alpha=0.9)
    bars2 = ax.bar(x + width/2, values_restricted, width,
                   color="#d6604d", edgecolor="white", label="同组回避约束", alpha=0.9)

    for bars in [bars1, bars2]:
        for bar, val in zip(bars, [values_unrestricted, values_restricted][
            0 if bars is bars1 else 1]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics], fontsize=10)
    ax.set_ylabel("概率", fontsize=11)
    ax.set_title("淘汰赛抽签约束对关键指标的影响（5000次模拟）", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(max(values_unrestricted), max(values_restricted)) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "p4_bracket_fairness.pdf")
    fig.savefig(OUTDIR / "p4_bracket_fairness.png")
    plt.close(fig)
    print("[ok] fig_p4_bracket_fairness")


# -----------------------------------------------------------------------------
# 图10：问题四 — 跨阶段衔接出行距离分布
# -----------------------------------------------------------------------------
def fig_p4_transition():
    trans = json.load(open(CODEOUT / "p4_transition.json"))
    centralized = trans["centralized_ko"]

    fig, ax = plt.subplots(figsize=(8, 5))

    stats_labels = [
        "平均出行\n(km/队)",
        "最大出行\n(km)",
        "最小出行\n(km)",
        "标准差\n(km)",
    ]
    stats_vals = [
        centralized["mean_travel_km"],
        centralized["max_travel_km"],
        centralized["min_travel_km"],
        centralized["std_travel_km"],
    ]
    colors_s = ["#2166ac", "#d6604d", "#4dac26", "#762a83"]

    bars = ax.bar(range(len(stats_labels)), stats_vals, color=colors_s,
                  edgecolor="white", width=0.55)
    for bar, val in zip(bars, stats_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{val:.0f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(stats_labels)))
    ax.set_xticklabels(stats_labels, fontsize=10)
    ax.set_ylabel("距离 (km)", fontsize=11)
    ax.set_title(
        f"跨阶段衔接：32支晋级队从小组赛赛区 → {centralized['ko_venue']}\n"
        f"总出行 = {centralized['total_travel_km']:.0f} km",
        fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTDIR / "p4_transition.pdf")
    fig.savefig(OUTDIR / "p4_transition.png")
    plt.close(fig)
    print("[ok] fig_p4_transition")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "code"))

    print("生成图表至", OUTDIR)
    fig_p1_strength()
    fig_p2_heatmap()
    fig_p2_deviation()
    fig_p2_county_heatmap()
    fig_p3_convergence()
    fig_p3_map()
    fig_p3_travel()
    fig_p4_stakeless()
    fig_p4_mi_dist()
    fig_p4_bracket_fairness()
    fig_p4_transition()
    print("\n全部图表已生成。")
