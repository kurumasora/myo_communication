"""
k-fold 交差検証結果の統計的有意性検定
対応ありt検定 (paired t-test) + Cohen's d によるペアワイズ比較
scipy 不要: math / numpy のみで t 分布 p 値を計算
"""
import sys
import math
import numpy as np
from itertools import combinations
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
SRC_DIR = EXPERIMENT_DIR.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR))

from kfold_validation import load_raw_data, run_kfold

RESULTS_MD = EXPERIMENT_DIR / "kfold_results.md"
N_FOLDS = 5
ALPHA = 0.05
SHORT = ["A", "B", "C", "D"]
PATTERNS = [
    ("パターンA：ベースライン",           {}),
    ("パターンB：データ拡張",             {"use_augment": True}),
    ("パターンC：特徴量抽出",             {"use_features": True}),
    ("パターンD：特徴量抽出＋データ拡張", {"use_features": True, "use_augment": True}),
]


# ── t 分布 p 値 (Numerical Recipes の betacf を移植) ──────────────────

def _betacf(a, b, x, max_iter=500, eps=3e-7, tiny=1e-30):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny: d = tiny
        if abs(c) < tiny: c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny: d = tiny
        if abs(c) < tiny: c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= eps:
            break
    return h


def _betainc(a, b, x):
    """正則化不完全ベータ関数 I_x(a, b)。"""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    bt = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    else:
        return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_pvalue(t_stat, df):
    """対応ありt検定の両側 p 値。"""
    x = df / (df + t_stat ** 2)
    return _betainc(df / 2.0, 0.5, x)


def t_ppf(p, df, lo=0.0, hi=20.0, tol=1e-7):
    """両側 p 値が p になる t 値を二分法で計算（信頼区間用）。"""
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if t_pvalue(mid, df) > p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


# ── 統計量 ────────────────────────────────────────────────────────────

def cohens_d_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(np.mean(d) / np.std(d, ddof=1))


def paired_ttest(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    mean_d = float(np.mean(d))
    std_d = float(np.std(d, ddof=1))
    t = mean_d / (std_d / math.sqrt(n))
    p = t_pvalue(t, df=n - 1)
    t_crit = t_ppf(ALPHA, df=n - 1)
    se = std_d / math.sqrt(n)
    ci = (mean_d - t_crit * se, mean_d + t_crit * se)
    return t, p, mean_d, ci


def sig_star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def effect_label(d):
    a = abs(d)
    if a >= 0.8: return "大"
    if a >= 0.5: return "中"
    if a >= 0.2: return "小"
    return "微小"


# ── 実行 ──────────────────────────────────────────────────────────────

def run_all(data_raw, labels):
    accs_dict = {}
    for name, kwargs in PATTERNS:
        _, _, accs = run_kfold(name, data_raw, labels, **kwargs)
        accs_dict[name] = np.array(accs)
    return accs_dict


def pairwise_tests(accs_dict):
    names = [p[0] for p in PATTERNS]
    rows = []
    for (i, n1), (j, n2) in combinations(enumerate(names), 2):
        t, p, diff, ci = paired_ttest(accs_dict[n1], accs_dict[n2])
        d = cohens_d_paired(accs_dict[n1], accs_dict[n2])
        rows.append({
            "pair":   f"{SHORT[i]} vs {SHORT[j]}",
            "diff":   diff,
            "ci":     ci,
            "t":      t,
            "p":      p,
            "star":   sig_star(p),
            "d":      d,
            "effect": effect_label(d),
        })
    return rows


def print_table(rows):
    header = f"{'ペア':^10} {'平均差':>8} {'95% CI':^24} {'t 値':>8} {'p 値':>8} {'有意':^6} {'Cohen d':>8} {'効果量':^6}"
    print(f"\n=== 統計的有意性検定 (対応ありt検定, df={N_FOLDS-1}) ===")
    print(header)
    print("-" * len(header))
    for r in rows:
        ci_str = f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"
        print(f"{r['pair']:^10} {r['diff']:>+8.4f} {ci_str:^24} {r['t']:>+8.3f} {r['p']:>8.4f} {r['star']:^6} {r['d']:>+8.3f} {r['effect']:^6}")
    print("\n* p<0.05  ** p<0.01  *** p<0.001  n.s.: 有意差なし")
    print("Cohen's d: |d| 微小 < 0.2 ≤ 小 < 0.5 ≤ 中 < 0.8 ≤ 大")


def build_md_section(rows):
    sig = [r for r in rows if r["p"] < ALPHA]
    ns  = [r for r in rows if r["p"] >= ALPHA]

    lines = [
        "",
        "---",
        "",
        "## 統計的有意性検定",
        "",
        f"**検定方法**: 対応ありt検定（paired t-test）、df = {N_FOLDS - 1}",
        "**有意水準**: α = 0.05（両側検定）",
        "",
        "### 前提",
        "",
        "5-fold の各 fold は全パターンで同一インデックス（seed=42）を共有しているため、fold 単位でのペア比較が有効。",
        "",
        "### ペアワイズ比較",
        "",
        "| ペア | 平均差 | 95% CI | t 値 | p 値 | 有意 | Cohen's d | 効果量 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in rows:
        ci_str = f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"
        lines.append(
            f"| {r['pair']} | {r['diff']:+.4f} | {ci_str} "
            f"| {r['t']:+.3f} | {r['p']:.4f} | {r['star']} "
            f"| {r['d']:+.3f} | {r['effect']} |"
        )

    lines += [
        "",
        "\\* p<0.05 &nbsp;\\*\\* p<0.01 &nbsp;\\*\\*\\* p<0.001 &nbsp; n.s.: 有意差なし  ",
        "Cohen's d の目安: 微小 < 0.2 ≤ 小 < 0.5 ≤ 中 < 0.8 ≤ 大",
        "",
        "### 考察",
        "",
    ]

    n_total, n_sig = len(rows), len(sig)
    if n_sig == n_total:
        lines.append(f"全 {n_total} ペアで統計的に有意な差が確認された（すべて p < {ALPHA}）。")
    else:
        lines.append(f"全 {n_total} ペア中 {n_sig} ペアで有意差あり（p < {ALPHA}）。")
        if ns:
            lines.append(f"有意差なし（n.s.）: {', '.join(r['pair'] for r in ns)}。")
    lines.append("")

    d_vs_a = next((r for r in rows if r["pair"] == "A vs D"), None)
    if d_vs_a:
        if d_vs_a["p"] < ALPHA:
            lines.append(
                f"パターン D（特徴量抽出＋データ拡張）はベースライン A に対して有意に精度が高く"
                f"（{d_vs_a['star']}, t = {d_vs_a['t']:+.3f}, p = {d_vs_a['p']:.4f}）、"
                f"効果量は{d_vs_a['effect']}（Cohen's d = {d_vs_a['d']:+.3f}）。"
            )
        else:
            lines.append(
                f"パターン D とベースライン A の差は有意でない（n.s., p = {d_vs_a['p']:.4f}）。"
            )
        lines.append("")

    c_vs_d = next((r for r in rows if r["pair"] == "C vs D"), None)
    if c_vs_d:
        if c_vs_d["p"] < ALPHA:
            lines.append(
                f"特徴量抽出単体（C）へのデータ拡張の上乗せ（C vs D）も有意"
                f"（{c_vs_d['star']}, t = {c_vs_d['t']:+.3f}, p = {c_vs_d['p']:.4f}）。"
                "特徴量空間でのデータ拡張が汎化性能の向上に寄与していると言える。"
            )
        else:
            lines.append(
                f"特徴量抽出単体（C）とパターン D の差は有意でない（n.s., p = {c_vs_d['p']:.4f}）。"
                "n = 5 の検定力の限界も考慮が必要。"
            )
        lines.append("")

    lines.append(
        "n = 5（df = 4）は検定力が低く、有意差なしの結果は「差がない」の証明ではない。"
        "fold 数を増やすことで検定力を高められる。"
    )
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    print("データを読み込み中...")
    data_raw, labels = load_raw_data()
    print(f"データ形状: {data_raw.shape}, ラベル数: {len(labels)}")

    print("\nk-fold 検証を実行中（4パターン）...")
    accs_dict = run_all(data_raw, labels)

    rows = pairwise_tests(accs_dict)
    print_table(rows)

    md = build_md_section(rows)
    with open(RESULTS_MD, "a", encoding="utf-8") as f:
        f.write(md)
    print(f"\n結果を {RESULTS_MD} に追記しました。")
