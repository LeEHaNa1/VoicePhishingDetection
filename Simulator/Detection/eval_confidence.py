# Simulator/Detection/eval_confidence.py
# Confidence Calibration(isotonic)의 효과를 Track B/C별로 정량 평가:
#   - ECE(quantile / uniform bin), Brier score, LogLoss, 평균 예측확률 vs 실제 피싱 비율, PR-AUC
#   - Reliability Diagram(3-panel): 선형 / log-log / 예측확률 분포 히스토그램
# 참고: 유의성 검정(부트스트랩 등)은 보고서 범위에서 제외하여 이 스크립트에도 포함하지 않음
#      (표본 크기 한계는 보고서에 별도로 명시).
# 결과: calibration_results/reliability_diagram_track{B,C}.png, confidence_metrics.json
#
# 실행: python -X utf8 -m Simulator.Detection.eval_confidence
#
# 해석 주의:
#  - ECE만으로는 부족함. 피싱 1% 데이터에서는 "항상 base rate만 예측"해도 ECE≈0이므로,
#    Brier/LogLoss(확률의 질)와 PR-AUC(변별력)를 함께 봐야 "보정이 변별력을 해치지 않으면서
#    확률을 정직하게 만들었는가"를 판단할 수 있음.
#  - 보정 전(raw) 확률은 scale_pos_weight로 부풀려져 있어 ECE가 큰 게 정상이고, 이 스크립트가
#    보려는 핵심은 "보정 후 test set에서 ECE/Brier가 실제로 낮아졌는가 + 그 차이가 유의한가"임.

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter, NullFormatter
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

from Simulator.main import (
    CALIBRATION_VAL_SIZE,
    TRACK_SCENARIOS,
    get_or_generate_final_dataset,
    load_tuned_hyperparams,
)
from Simulator.Detection.train_eval import (
    cast_object_columns,
    fit_calibrator,
    prepare_categorical,
    split_data,
    train_model,
)
from Simulator.schema_utils import get_feature_columns

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OUT_DIR = _PROJECT_ROOT / "calibration_results"

N_BINS = 10            # ECE/reliability용 bin 개수
GAP_TOLERANCE = 0.10   # 평균 예측확률이 실제 비율의 ±10% 이내면 "평균 정렬"로 판정
_EPS = 1e-15
NEAR_ZERO = 1e-6      # 이 값 이하의 예측확률은 사실상 0으로 취급(isotonic의 0 plateau 진단용)
# 점수 내림차순 기준 상위 x% 구간 경계. 피싱이 1%라 양성이 전부 상위 몇 %에 몰리므로, 10-분위 bin보다
# 촘촘하게 나눠 '판단에 실제로 쓰이는 구간'의 보정을 확인하기 위함.
TAIL_EDGES = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 1.0)


def _setup_korean_font():
    """한글 라벨 깨짐 방지. 실패해도 조용히 넘어감(그래프 생성은 막지 않음)."""
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",                               # Windows 맑은 고딕
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",                # macOS
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",           # Linux NanumGothic
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",    # Linux Noto CJK
    ]
    for path in candidates:
        try:
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue


_setup_korean_font()


# ------------------------------------------------------------
# ECE 계산
# ------------------------------------------------------------
def _bin_stats(y_true, proba, n_bins, strategy):
    """bin별 (평균 예측확률, 실제 양성 비율, 표본 수)를 반환. 빈 bin은 제외.
    sklearn.calibration_curve와 달리 표본 수를 함께 돌려줘서 ECE를 bin 크기로 정확히 가중할 수 있음
    (isotonic은 계단형 동점 값을 만들어 quantile 경계가 중복되고 bin 수가 줄어들기 때문)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)

    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))  # 동점 경계 병합
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    n_eff = max(len(edges) - 1, 1)
    bin_ids = np.searchsorted(edges[1:-1], p, side="right")
    counts = np.bincount(bin_ids, minlength=n_eff)
    sum_p = np.bincount(bin_ids, weights=p, minlength=n_eff)
    sum_y = np.bincount(bin_ids, weights=y, minlength=n_eff)

    keep = counts > 0
    return sum_p[keep] / counts[keep], sum_y[keep] / counts[keep], counts[keep]


def _ece(mean_pred, frac_pos, counts):
    weights = counts / counts.sum()
    return float(np.sum(weights * np.abs(frac_pos - mean_pred)))


def _ece_quantile(y_true, proba, n_bins=N_BINS):
    return _ece(*_bin_stats(y_true, proba, n_bins, "quantile"))


def compute_calibration_metrics(y_true, proba, n_bins=N_BINS):
    """한 세트의 확률에 대한 정량 지표 + reliability 곡선용 quantile bin 통계를 반환."""
    y = np.asarray(y_true)
    p = np.asarray(proba, dtype=float)

    q_bins = _bin_stats(y, p, n_bins, "quantile")
    u_bins = _bin_stats(y, p, n_bins, "uniform")

    mean_pred, base_rate = float(p.mean()), float(y.mean())
    rel_gap = (mean_pred - base_rate) / base_rate if base_rate > 0 else float("nan")
    if abs(rel_gap) <= GAP_TOLERANCE:
        status = "평균 정렬 (평균 예측확률 ≈ 실제 피싱 비율)"
    elif rel_gap > 0:
        status = "과신 (Overconfidence: 확률을 실제보다 높게 평가)"
    else:
        status = "과소신 (Underconfidence: 확률을 실제보다 낮게 평가)"

    metrics = {
        "ece_quantile": _ece(*q_bins),
        "ece_uniform": _ece(*u_bins),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, _EPS, 1 - _EPS), labels=[0, 1])),
        "mean_predicted": mean_pred,
        "base_rate": base_rate,
        "mean_gap_ratio": float(rel_gap),
        "pr_auc": float(average_precision_score(y, p)),
        "n_unique_proba": int(len(np.unique(p))),                       # 서로 다른 확률값 개수(isotonic은 계단형이라 작음)
        "frac_near_zero_proba": float((p <= NEAR_ZERO).mean()),         # 확률이 사실상 0인 표본 비율
        "positives_in_near_zero": int(y[p <= NEAR_ZERO].sum()),         # 그 안에 들어간 실제 피싱 수(LogLoss에 치명적)
        "status": status,
    }
    return metrics, q_bins


def bins_to_rows(bins):
    """(mean_pred, frac_pos, counts) -> JSON 저장용 행 목록. 그래프의 각 점이 무엇인지 표로 확인하기 위함."""
    mean_pred, frac_pos, counts = bins
    return [
        {"mean_pred": float(m), "frac_pos": float(f), "count": int(c), "n_pos": int(round(f * c))}
        for m, f, c in zip(mean_pred, frac_pos, counts)
    ]


def tail_bin_table(y_true, proba, edges=TAIL_EDGES):
    """점수 내림차순으로 상위 구간(TAIL_EDGES)별 (표본 수, 양성 수, 평균 예측확률, 실제 양성 비율).
    참고: isotonic은 동점이 많아 같은 확률 안에서의 순서는 임의(안정 정렬)이므로 구간 경계 근처는 근사임."""
    y = np.asarray(y_true)
    p = np.asarray(proba, dtype=float)
    order = np.argsort(-p, kind="stable")
    n = len(p)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = order[int(round(n * lo)):int(round(n * hi))]
        if len(idx) == 0:
            continue
        rows.append({
            "top_pct": f"{lo * 100:g}-{hi * 100:g}%",
            "n": int(len(idx)),
            "n_pos": int(y[idx].sum()),
            "mean_pred": float(p[idx].mean()),
            "frac_pos": float(y[idx].mean()),
        })
    return rows



# ------------------------------------------------------------
# 시각화
# ------------------------------------------------------------
def _plain_log_ticks(ax, axis="both"):
    """로그 축 눈금을 10^-4 같은 mathtext 대신 0.0001 같은 일반 숫자로 표시.
    맑은 고딕 등 일부 한글 폰트는 mathtext의 지수 마이너스 글리프가 없어 눈금이 깨져 보임."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    for name in ("x", "y"):
        if axis in (name, "both"):
            a = ax.xaxis if name == "x" else ax.yaxis
            a.set_major_formatter(fmt)
            a.set_minor_formatter(NullFormatter())


def _plot_loglog(ax, bins, color, marker, label, floor):
    """log 축에는 0을 그릴 수 없으므로 양성 0개 bin은 선으로 잇지 않고 하한에 빈 삼각형으로 따로 표시
    (하한에 점을 이어 그리면 '확률 1e-4로 평평한 곡선'처럼 오해됨)."""
    mean_pred, frac_pos = np.asarray(bins[0]), np.asarray(bins[1])
    nz = frac_pos > 0
    ax.plot(np.clip(mean_pred[nz], floor, 1), frac_pos[nz], marker + "-", color=color, label=label)
    if (~nz).any():
        ax.scatter(np.clip(mean_pred[~nz], floor, 1), np.full((~nz).sum(), floor), marker="v",
                   facecolors="none", edgecolors=color, label=f"{label}: zero-positive bins (plotted at floor)")


def plot_reliability(track_name, raw_m, raw_bins, cal_m, cal_bins, raw_proba, calib_proba, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    floor = 1e-4  # log 축에서 0을 그릴 수 없어 하한으로 표시

    # (1) 선형 reliability
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect (y=x)")
    ax.plot(raw_bins[0], raw_bins[1], "s-", color="crimson", label=f"Before (ECE {raw_m['ece_quantile']:.4f})")
    ax.plot(cal_bins[0], cal_bins[1], "o-", color="royalblue", label=f"After (ECE {cal_m['ece_quantile']:.4f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted Probability (Confidence)")
    ax.set_ylabel("Fraction of Positives (Empirical Frequency)")
    ax.set_title("Reliability (linear, quantile bins)")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)

    # (2) log-log reliability: 피싱 1%라 낮은 확률 구간이 선형 축에서는 뭉개져 보임
    ax = axes[1]
    ax.plot([floor, 1], [floor, 1], "k--", alpha=0.7, label="Perfect")
    _plot_loglog(ax, raw_bins, "crimson", "s", "Before", floor)
    _plot_loglog(ax, cal_bins, "royalblue", "o", "After", floor)
    ax.set_xscale("log"); ax.set_yscale("log")
    _plain_log_ticks(ax)
    ax.set_xlabel("Mean Predicted Probability (log)")
    ax.set_ylabel("Fraction of Positives (log)")
    ax.set_title("Reliability (log-log)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3, which="both")

    # (3) 예측확률 분포: 각 bin의 신뢰도를 얼마나 많은 표본이 뒷받침하는지 확인
    ax = axes[2]
    bins = np.logspace(-6, 0, 40)
    ax.hist(np.clip(raw_proba, 1e-6, 1), bins=bins, alpha=0.55, color="crimson", label="Before")
    ax.hist(np.clip(calib_proba, 1e-6, 1), bins=bins, alpha=0.55, color="royalblue", label="After")
    ax.set_xscale("log"); ax.set_yscale("log")
    _plain_log_ticks(ax)
    ax.set_xlabel("Predicted Probability (log)")
    ax.set_ylabel("Count (log)")
    ax.set_title("Predicted probability distribution")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(f"[Track {track_name}] Confidence Calibration Diagnosis", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / f"reliability_diagram_track{track_name}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_tail_reliability(tail_by_track, out_dir):
    """상위 구간별 신뢰도 다이어그램(보고서 본문용). 10-분위 bin은 피싱이 상위 10% bin 하나에 몰려서 점이
    2~3개(Track C 보정 후는 1개)뿐이라 정보가 없음 -> 점수 상위 0-0.5%, 0.5-1%, 1-2%, 2-5%, 5-10%, 나머지로
    나눠 각 구간의 (평균 예측확률, 실제 피싱 비율)을 선형 축에 찍음. 구간은 두 모델이 같은 순위 범위를 쓰므로
    Before/After 점이 같은 구간끼리 대응됨. 라벨은 Before 점에만 표기."""
    n = len(tail_by_track)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.6), squeeze=False)
    for ax, (track_name, rows) in zip(axes[0], tail_by_track.items()):
        ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect (y=x)")
        for key, color, marker, label in (("raw", "crimson", "s", "Before"), ("calibrated", "royalblue", "o", "After")):
            xs = [r["mean_pred"] for r in rows[key]]
            ys = [r["frac_pos"] for r in rows[key]]
            ax.plot(xs, ys, marker + "-", color=color, label=label, alpha=0.9)
        for r in rows["raw"]:
            # 원점 근처(평균 예측확률 < 0.2)에 몰린 구간은 라벨이 서로 겹쳐 읽히지 않으므로 라벨 생략
            if r["mean_pred"] < 0.2:
                continue
        ax.text(0.98, 0.02, "Bins with mean prediction < 0.2 are not labeled", ha="right", va="bottom",
                fontsize=7, color="gray", transform=ax.transAxes)
        ax.set_xlim(-0.02, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Mean Predicted Probability (per bin)")
        ax.set_ylabel("Fraction of Positives (per bin)")
        ax.set_title(f"[Track {track_name}] Tail Reliability Diagram (bins by score rank)")
        ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "tail_reliability_diagram.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ------------------------------------------------------------
# 실행
# ------------------------------------------------------------
def _print_report(track_name, raw_m, cal_m):
    rows = [
        ("ECE (quantile)", "ece_quantile"), ("ECE (uniform)", "ece_uniform"),
        ("Brier", "brier"), ("LogLoss", "log_loss"),
        ("평균 예측확률", "mean_predicted"), ("실제 피싱 비율", "base_rate"),
        ("PR-AUC", "pr_auc"),
    ]
    print("\n" + "=" * 66)
    print(f"📊 [Track {track_name}] 확신도(Confidence) 정량 진단")
    print("=" * 66)
    print(f"{'지표':<16}{'보정 전(Raw)':>16}{'보정 후(Isotonic)':>20}")
    for label, key in rows:
        print(f"{label:<16}{raw_m[key]:>16.5f}{cal_m[key]:>20.5f}")
    print(f"상태(Raw)       : {raw_m['status']}")
    print(f"상태(Calibrated): {cal_m['status']}")
    print(f"확률≈0(≤{NEAR_ZERO:g}) 표본 비율  Raw {raw_m['frac_near_zero_proba']:.1%} / Calibrated {cal_m['frac_near_zero_proba']:.1%}"
          f"  | 그 안의 실제 피싱 수 {raw_m['positives_in_near_zero']} / {cal_m['positives_in_near_zero']}")
    print(f"서로 다른 확률값 개수         Raw {raw_m['n_unique_proba']} / Calibrated {cal_m['n_unique_proba']}")
    print("=" * 66)


def _print_tail_table(raw_rows, cal_rows):
    print("상위 구간별 보정 확인 (평균 예측확률 vs 실제 피싱 비율)")
    print(f"{'구간':<12}{'n':>7}{'양성':>6}{'Raw 예측':>11}{'Cal 예측':>11}{'실제 비율':>11}")
    for r, c in zip(raw_rows, cal_rows):
        print(f"{r['top_pct']:<12}{r['n']:>7}{r['n_pos']:>6}{r['mean_pred']:>11.4f}{c['mean_pred']:>11.4f}{r['frac_pos']:>11.4f}")


def run_confidence_analysis():
    df = get_or_generate_final_dataset()
    df = prepare_categorical(df)
    train_set, test_set = split_data(df)

    # main.py::run_final()의 버전 C와 동일한 분할(CALIBRATION_VAL_SIZE)을 써야 같은 모델을 평가하게 됨.
    train_sub, val_calib = train_test_split(
        train_set,
        test_size=CALIBRATION_VAL_SIZE,
        random_state=42,
        stratify=train_set["is_phishing"],
    )

    model_type, hyperparams = load_tuned_hyperparams()
    model_type = model_type or "XGBoost"
    y_test = test_set["is_phishing"].to_numpy()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    tail_by_track = {}
    payload = {"n_bins": N_BINS, "calibration_val_size": CALIBRATION_VAL_SIZE, "tracks": {}}

    for track_name, tracks in TRACK_SCENARIOS.items():
        feature_cols = get_feature_columns(track_filter=tracks)

        model = train_model(train_sub, feature_cols=feature_cols, hyperparams=hyperparams, model_type=model_type)
        calibrator = fit_calibrator(model, val_calib, feature_cols=feature_cols, method="isotonic")

        X_test, _ = cast_object_columns(test_set[feature_cols].copy())
        raw_proba = model.predict_proba(X_test)[:, 1]
        calib_proba = calibrator.predict_proba(X_test)[:, 1]

        raw_m, raw_bins = compute_calibration_metrics(y_test, raw_proba)
        cal_m, cal_bins = compute_calibration_metrics(y_test, calib_proba)

        _print_report(track_name, raw_m, cal_m)
        raw_tail, cal_tail = tail_bin_table(y_test, raw_proba), tail_bin_table(y_test, calib_proba)
        tail_by_track[track_name] = {"raw": raw_tail, "calibrated": cal_tail}
        _print_tail_table(raw_tail, cal_tail)
        out_path = plot_reliability(track_name, raw_m, raw_bins, cal_m, cal_bins, raw_proba, calib_proba, _OUT_DIR)
        print(f"신뢰도 다이어그램 저장: {out_path}")

        payload["tracks"][track_name] = {
            "raw": raw_m, "calibrated": cal_m,
            "quantile_bins": {"raw": bins_to_rows(raw_bins), "calibrated": bins_to_rows(cal_bins)},
            "tail_bins": {"raw": raw_tail, "calibrated": cal_tail},
        }

    print(f"상위 구간 신뢰도 다이어그램 저장: {plot_tail_reliability(tail_by_track, _OUT_DIR)}")

    json_path = _OUT_DIR / "confidence_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"지표 JSON 저장: {json_path}")
    return payload


if __name__ == "__main__":
    run_confidence_analysis()