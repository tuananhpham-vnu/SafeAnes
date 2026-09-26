"""Figures for reports/EDA (matplotlib, static PNG). Palette: validated reference instance
(dataviz skill) — slot 1 blue, slot 2 orange, slot 3 aqua; text in ink tones, recessive grid."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .eda import LOG_SIGNALS, OFFSETS, TRAJ

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABELS = {
    "map": "MAP (Solar8000)", "hr": "HR", "bt_sv_lz": "SV proxy (LZ)", "bt_co_lz": "CO proxy",
    "bt_svr_lz": "SVR proxy", "bt_hr": "HR (beat)", "bt_ppv": "PPV (%)", "bt_dpdt": "dP/dt max",
    "bt_pp": "Pulse pressure", "bis": "BIS", "ppf_ce": "Propofol Ce (µg/mL)",
    "rftn_ce": "Remifentanil Ce (ng/mL)", "mac": "MAC", "cvp": "CVP (mmHg)", "nibp_mbp": "NIBP MAP",
    "ev_sv": "EV1000 SV (mL)", "ev_svr": "EV1000 SVR", "ev_svv": "EV1000 SVV (%)", "ev_co": "EV1000 CO",
    "vg_sv": "Vigileo SV", "vg_svv": "Vigileo SVV (%)", "phen_rate": "Phenylephrine rate",
    "nepi_rate": "Norepinephrine rate", "peep": "PEEP (mbar)",
}


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 130, "font.size": 9, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False,
        "axes.titlesize": 9.5, "axes.titleweight": "bold", "figure.facecolor": "white"})


def availability(table, path):
    style()
    t = table.sort_values("% eligible")
    colors = {g: SERIES[i % len(SERIES)] for i, g in enumerate(dict.fromkeys(table["nhóm"]))}
    fig, ax = plt.subplots(figsize=(7.5, max(4, len(t) * .17)))
    ax.barh(t.track, t["% eligible"], color=[colors[g] for g in t["nhóm"]], height=.7)
    ax.set_xlabel("% ca eligible có track")
    ax.tick_params(axis="y", labelsize=6.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), loc="lower right", fontsize=7)
    ax.set_title("Mức sẵn có của track liên quan UC04 (3.626 ca eligible)")
    _save(fig, path)


def events_overview(events, cases, path):
    style()
    ev = events[events.kind.eq("event")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    ax = axes[0, 0]
    ax.hist(cases.n_events.clip(upper=15), bins=np.arange(0, 17) - .5, color=BLUE, rwidth=.85)
    ax.set(title="Số biến cố IOH mỗi ca (≥15 gộp)", xlabel="biến cố", ylabel="số ca")
    ax = axes[0, 1]
    ax.hist(np.log10(ev.duration_s), bins=40, color=BLUE)
    ax.set_xticks(np.log10([60, 120, 300, 600, 1800, 3600]), ["1", "2", "5", "10", "30", "60"])
    ax.set(title="Thời lượng biến cố", xlabel="phút (log)", ylabel="biến cố")
    ax = axes[0, 2]
    ax.hist(ev.min_after_anestart.clip(upper=360), bins=72, color=BLUE)
    ax.axvline(ev.min_after_anestart.sub(ev.min_after_opstart).median(), color=INK2, lw=1, ls="--")
    ax.set(title="Thời điểm khởi phát sau bắt đầu gây mê", xlabel="phút (--- = median lúc rạch da)", ylabel="biến cố")
    ax = axes[1, 0]
    ax.hist(ev.min_map.clip(30, 65), bins=35, color=BLUE)
    ax.set(title="MAP thấp nhất trong biến cố", xlabel="mmHg", ylabel="biến cố")
    ax = axes[1, 1]
    ax.hist(np.log10(ev.auc65_mmHg_min.clip(lower=.5)), bins=40, color=BLUE)
    ax.set_xticks(np.log10([1, 3, 10, 30, 100, 300]), ["1", "3", "10", "30", "100", "300"])
    ax.set(title="Độ nặng: diện tích dưới 65", xlabel="mmHg·phút (log)", ylabel="biến cố")
    ax = axes[1, 2]
    ax.hist(cases.pct_time_map_lt65.clip(upper=50), bins=50, color=BLUE)
    ax.set(title="% thời gian MAP<65 mỗi ca", xlabel="% (≥50 gộp)", ylabel="số ca")
    _save(fig, path)


def trajectories(events, traj, path, signals=None):
    """Median and IQR, events vs controls. Ratio signals are normalised to the case's own
    baseline (-15..-10 min) so they are comparable across patients."""
    style()
    signals = signals or TRAJ
    kinds = events.kind.to_numpy()
    base_idx = (OFFSETS >= -900) & (OFFSETS < -600)
    n = len(signals)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.6 * rows), squeeze=False)
    minutes = OFFSETS / 60
    for ax, name in zip(axes.flat, signals):
        i = TRAJ.index(name)
        data = traj[:, i, :].astype(float)
        ratio = name in LOG_SIGNALS
        if ratio:
            with np.errstate(all="ignore"):
                base = np.nanmedian(data[:, base_idx], axis=1, keepdims=True)
                data = data / base
        for kind, color, label in (("event", BLUE, "trước IOH"), ("control", ORANGE, "đối chứng")):
            d = data[kinds == kind]
            d = d[np.isfinite(d).sum(axis=1) > len(OFFSETS) // 2]
            if len(d) < 20:
                continue
            with np.errstate(all="ignore"):
                q1, q2, q3 = np.nanpercentile(d, [25, 50, 75], axis=0)
            ax.fill_between(minutes, q1, q3, color=color, alpha=.15, lw=0)
            ax.plot(minutes, q2, color=color, label=f"{label} (n={len(d)})")
        ax.axvline(0, color=INK2, lw=.8, ls="--")
        ax.set_title(LABELS.get(name, name) + (" — tỉ lệ so với nền" if ratio else ""))
        ax.legend(fontsize=6.5, loc="best")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("phút so với khởi phát (0) / mốc đối chứng")
    _save(fig, path)


def decomposition(dec, path):
    style()
    order = ["giảm SVR", "giảm SVR + tăng thuốc mê", "giảm SV + PPV cao/tăng (gợi ý tiền tải)",
             "giảm SV + giảm dP/dt (gợi ý co bóp)", "giảm SV, không rõ", "giảm HR", "thay đổi nhỏ (<5%)"]
    known = dec[~dec.pattern.eq("thiếu dữ liệu beat")]
    share = known.groupby("kind").pattern.value_counts(normalize=True).unstack(0).reindex(order).fillna(0) * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    y = np.arange(len(order))
    for j, (kind, color, label) in enumerate((("event", BLUE, "trước IOH"), ("control", ORANGE, "đối chứng"))):
        if kind in share:
            a1.barh(y + (j - .5) * .38, share[kind], height=.36, color=color, label=label)
            for yy, v in zip(y + (j - .5) * .38, share[kind]):
                a1.text(v + .5, yy, f"{v:.0f}%", va="center", fontsize=7, color=INK2)
    a1.set_yticks(y, order, fontsize=7.5)
    a1.invert_yaxis()
    a1.set(xlabel="% biến cố / mốc có dữ liệu beat", title="Thành phần giảm mạnh nhất (−15→0 phút), mẫu hình heuristic")
    a1.legend(fontsize=7)
    for kind, color in (("control", ORANGE), ("event", BLUE)):
        d = known[known.kind.eq(kind)]
        a2.scatter(d.dlog_bt_svr_lz.clip(-1, 1), d.dlog_bt_sv_lz.clip(-1, 1), s=8, color=color, alpha=.35,
                   edgecolors="white", linewidths=.3, label="trước IOH" if kind == "event" else "đối chứng")
    a2.axhline(0, color=INK2, lw=.7)
    a2.axvline(0, color=INK2, lw=.7)
    a2.set(xlabel="Δlog SVR proxy", ylabel="Δlog SV proxy", title="Thay đổi SV vs SVR trước mốc")
    a2.legend(fontsize=7, markerscale=2)
    _save(fig, path)


def proxy_validation(table, path):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    pairs = [("r_bt_sv_lz_ev_sv", "SV proxy vs EV1000 SV", BLUE), ("r_bt_svr_lz_ev_svr", "SVR proxy vs EV1000 SVR", ORANGE),
             ("r_bt_ppv_ev_svv", "PPV vs EV1000 SVV", AQUA)]
    for col, label, color in pairs:
        if col in table and table[col].notna().sum() >= 5:
            axes[0].hist(table[col].dropna(), bins=np.linspace(-1, 1, 41), histtype="step", lw=2, color=color,
                         label=f"{label} (n={table[col].notna().sum()}, median {table[col].median():.2f})")
    axes[0].set(title="Spearman r theo ca (lưới 30 s)", xlabel="r", ylabel="số ca")
    axes[0].legend(fontsize=6.5, loc="upper left")
    for col, label, color in [("concord_" + c[len("r_"):], l, k) for c, l, k in pairs]:
        if col in table and table[col].notna().sum() >= 5:
            axes[1].hist(table[col].dropna(), bins=np.linspace(0, 1, 21), histtype="step", lw=2, color=color,
                         label=f"{label} (median {table[col].median():.2f})")
    axes[1].set(title="Tỉ lệ đồng hướng xu hướng 5 phút (vùng loại 10%)", xlabel="concordance", ylabel="số ca")
    axes[1].legend(fontsize=6.5, loc="upper left")
    _save(fig, path)


def nibp_art(pairs, path):
    style()
    d = pairs.sample(min(len(pairs), 20000), random_state=0)
    mean, diff = (d.nibp + d.art) / 2, d.nibp - d.art
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(mean, diff, s=3, color=BLUE, alpha=.2, edgecolors="none")
    for v, ls in ((diff.mean(), "-"), (diff.mean() - 1.96 * diff.std(), "--"), (diff.mean() + 1.96 * diff.std(), "--")):
        ax.axhline(v, color=INK2, lw=1, ls=ls)
    ax.set(xlim=(30, 160), ylim=(-60, 60), xlabel="(NIBP + ART)/2 MAP", ylabel="NIBP − ART (mmHg)",
           title="Bland–Altman NIBP vs ART MAP (—— bias, -- LoA 95%)")
    _save(fig, path)


def bleeding(table, path):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    d = table.dropna(subset=["hb_drop"])
    axes[0].scatter(d.hb_drop, d.events_per_h.clip(upper=6), s=8, color=BLUE, alpha=.4, edgecolors="none")
    axes[0].set(xlabel="Hb trước mổ − Hb thấp nhất trong mổ (g/dL)", ylabel="biến cố IOH / giờ (≤6)",
                title=f"Giảm Hb vs gánh nặng IOH (n={len(d)} ca có Hb trong mổ)")
    d = table.dropna(subset=["intraop_ebl"])
    axes[1].scatter(np.log10(d.intraop_ebl.clip(lower=10)), d.events_per_h.clip(upper=6), s=8, color=BLUE,
                    alpha=.4, edgecolors="none")
    axes[1].set_xticks(np.log10([10, 100, 1000, 10000]), ["≤10", "100", "1.000", "10.000"])
    axes[1].set(xlabel="EBL (mL, log)", ylabel="biến cố IOH / giờ (≤6)", title=f"EBL vs gánh nặng IOH (n={len(d)})")
    _save(fig, path)


def clusters(centroids, path):
    style()
    cols = ["dlog_bt_sv_lz", "dlog_bt_hr", "dlog_bt_svr_lz", "d_bt_ppv", "dlog_bt_dpdt"]
    names = ["ΔlogSV", "ΔlogHR", "ΔlogSVR", "ΔPPV/10", "Δlog dP/dt"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(cols))
    k = len(centroids)
    for j, (cid, row) in enumerate(centroids.iterrows()):
        vals = [row[c] / (10 if c == "d_bt_ppv" else 1) for c in cols]
        ax.bar(x + (j - (k - 1) / 2) * .8 / k, vals, width=.8 / k * .9, color=SERIES[j % 8],
               label=f"cụm {cid} (n={int(row['số biến cố'])})")
    ax.axhline(0, color=INK2, lw=.8)
    ax.set_xticks(x, names)
    ax.set(title="Median thay đổi trước IOH theo cụm k-means", ylabel="thay đổi")
    ax.legend(fontsize=7, ncol=3)
    _save(fig, path)


def beat_snippets(paths, path):
    style()
    paths = paths[:6]
    fig, axes = plt.subplots(len(paths), 1, figsize=(10, 1.8 * len(paths)), squeeze=False)
    for ax, p in zip(axes[:, 0], paths):
        z = np.load(p)
        ax.plot(z["t"] - z["t"][0], z["p"], color=INK2, lw=.8)
        for onset, sbp, dbp, valid in zip(z["onset"], z["sbp"], z["dbp"], z["valid"]):
            ax.plot([onset - z["t"][0]] * 2, [dbp, sbp], color=BLUE if valid else RED, lw=1.2)
        ax.set_title(f"ca {p.stem}: xanh = beat hợp lệ (DBP→SBP), đỏ = loại", loc="left", fontsize=8)
        ax.set_ylabel("mmHg")
    axes[-1, 0].set_xlabel("giây")
    _save(fig, path)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- insight figures (reports/EDA/INSIGHTS.md)
GROUP_COLORS = {"MAP": BLUE, "SBP/DBP": AQUA, "HR": ORANGE, "SpO2": MAGENTA, "EtCO2/RR": YELLOW,
                "Beat (ART waveform)": VIOLET, "Độ mê / thuốc mê": RED, "Thở máy / CVP": GREEN,
                "Static": "#8a8984", "Khác": "#8a8984"}


def _group_legend(ax, groups, loc="lower right"):
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in groups]
    ax.legend(handles, groups, fontsize=7, loc=loc)


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div", [BLUE, "#eeeeea", RED])


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#7fb0ea", BLUE, "#0d3a73"])


def hbar_by_group(table, value, label, path, title, top=25, fmt="{:.1f}", xlim=None, signed=False):
    style()
    t = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, max(3, .26 * len(t) + 1)))
    ax.barh(t.feature, t[value], color=[GROUP_COLORS.get(g, INK2) for g in t["nhóm"]], height=.7)
    for y, v in enumerate(t[value]):
        ax.text(v, y, f" {fmt.format(v)} " , va="center", ha="left" if v >= 0 else "right", fontsize=6.5, color=INK2)
    if signed:
        ax.axvline(0, color=INK2, lw=.8)
    if xlim:
        ax.set_xlim(*xlim)
    ax.tick_params(axis="y", labelsize=7)
    ax.set(xlabel=label, title=title)
    _group_legend(ax, list(dict.fromkeys(t["nhóm"][::-1])))
    _save(fig, path)


def noise_bars(summary, path):
    style()
    s = summary.iloc[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    panels = [("% thời gian mổ thiếu/cũ (median ca)", "% thời gian mổ không có giá trị mới\n(median theo ca)", BLUE),
              ("% ca thiếu >20% thời gian", "% ca có >20% thời gian mổ thiếu", ORANGE),
              ("% mẫu bất thường", "% mẫu ngoài ngưỡng sinh lý (xem bảng)", RED)]
    for ax, (col, label, color) in zip(axes, panels):
        ax.barh(s["tín hiệu"], s[col].fillna(0), color=color, height=.65)
        for y, v in enumerate(s[col].fillna(0)):
            ax.text(v, y, f" {v:.2f}" if v < 10 else f" {v:.1f}", va="center", fontsize=6.5, color=INK2)
        ax.set_xlabel(label)
    axes[0].set_title("Thiếu dữ liệu", loc="left")
    axes[1].set_title("Ca bị ảnh hưởng", loc="left")
    axes[2].set_title("Giá trị bất thường", loc="left")
    _save(fig, path)


def corr_heatmap(corr, path):
    style()
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    c = corr.fillna(0)
    order = leaves_list(linkage(squareform(1 - c.abs().values, checks=False), "average"))
    c = corr.iloc[order, order]
    fig, ax = plt.subplots(figsize=(10.5, 9))
    im = ax.imshow(c.values, cmap=diverging_cmap(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(c)), c.index, fontsize=6.5)
    ax.grid(False)
    for i in range(len(c)):
        for j in range(len(c)):
            v = c.values[i, j]
            if np.isfinite(v) and abs(v) >= .5 and i != j:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=4.5, color="white" if abs(v) > .75 else INK)
    fig.colorbar(im, ax=ax, shrink=.6, label="Spearman ρ")
    ax.set_title("Tương quan Spearman giữa các feature chính (sắp theo cụm; số hiển thị khi |ρ| ≥ 0,5)")
    _save(fig, path)


def risk_curves(curves, path, base_rate):
    style()
    n = len(curves)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5 * rows), squeeze=False)
    for ax, (name, c) in zip(axes.flat, curves.items()):
        if c.empty:
            ax.set_visible(False)
            continue
        ax.plot(c.x, c.risk * 100, color=BLUE, marker="o", ms=4)
        ax.axhline(base_rate * 100, color=INK2, lw=.8, ls="--")
        ax.set_title(name, fontsize=8.5)
        ax.set_ylabel("% y5 dương")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle("Nguy cơ IOH trong 5 phút theo decile của từng feature (--- = tỉ lệ nền)", fontsize=10, y=1.0)
    _save(fig, path)


def level_trend(risk, path):
    style()
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(risk.values * 100, cmap=sequential_cmap(), aspect="auto", origin="lower")
    ax.set_xticks(range(risk.shape[1]), [f"{iv.left:.0f}–{iv.right:.0f}" for iv in risk.columns], rotation=45, fontsize=7)
    ax.set_yticks(range(risk.shape[0]), [f"{iv.left:+.0f}…{iv.right:+.0f}" for iv in risk.index], fontsize=7)
    ax.grid(False)
    for i in range(risk.shape[0]):
        for j in range(risk.shape[1]):
            v = risk.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=6.5, color="white" if v > .25 else INK)
    fig.colorbar(im, ax=ax, label="% y5 dương")
    ax.set(xlabel="MAP hiện tại (mmHg)", ylabel="xu hướng MAP 5 phút (mmHg/phút)",
           title="Nguy cơ IOH 5 phút theo mức × xu hướng MAP (ô trống: < 200 decision)")
    _save(fig, path)


def shap_panels(imp, path):
    style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.6, 1]})
    t = imp.head(20).iloc[::-1]
    a1.barh(t.feature, t["share_%"], color=[GROUP_COLORS.get(g, INK2) for g in t["nhóm"]], height=.7)
    a1.tick_params(axis="y", labelsize=7)
    a1.set(xlabel="% tổng mean|SHAP|", title="20 feature đóng góp nhiều nhất (LightGBM, y5, VALIDATION)")
    g = imp.groupby("nhóm")["share_%"].sum().sort_values()
    a2.barh(g.index, g.values, color=[GROUP_COLORS.get(x, INK2) for x in g.index], height=.65)
    for y, v in enumerate(g.values):
        a2.text(v, y, f" {v:.1f}%", va="center", fontsize=7, color=INK2)
    a2.set(xlabel="% tổng mean|SHAP|", title="Theo nhóm tín hiệu")
    _save(fig, path)


def ablation_bars(table, path):
    style()
    t = table.copy()
    t["kind"] = np.where(t["cấu hình"].str.startswith("MAP +"), "MAP + nhóm",
                         np.where(t["cấu hình"].str.startswith("Tất cả −"), "Tất cả − nhóm", "tham chiếu"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    order = t.sort_values(["kind", "AUROC"])["cấu hình"]
    t = t.set_index("cấu hình").loc[order]
    colors = {"MAP + nhóm": BLUE, "Tất cả − nhóm": ORANGE, "tham chiếu": INK2}
    for ax, metric in zip(axes, ("AUROC", "AP")):
        ax.barh(t.index, t[metric], color=[colors[k] for k in t.kind], height=.7)
        lo = t[metric].min()
        ax.set_xlim(lo - (t[metric].max() - lo) * .6, t[metric].max() + (t[metric].max() - lo) * .25)
        for y, v in enumerate(t[metric]):
            ax.text(v, y, f" {v:.3f}", va="center", fontsize=6.5, color=INK2)
        ax.set(xlabel=f"{metric} (VALIDATION, y5)")
    axes[0].tick_params(axis="y", labelsize=7.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    axes[1].legend(handles, colors.keys(), fontsize=7, loc="lower right")
    fig.suptitle("Ablation nhóm tín hiệu: thêm vào MAP (xanh) và bỏ khỏi mô hình đầy đủ (cam)", fontsize=10)
    _save(fig, path)
