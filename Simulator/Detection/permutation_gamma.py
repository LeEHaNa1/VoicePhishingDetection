# γ(calibration 적용) 모델 기준 permutation importance 재계산 파일.
# 역할: 보고서의 permutation importance는 α(train_set 100% 학습, 보정 없음) 기준이고 탐지 성능 표는 γ 기준이라,
#       같은 설정(test_set 30,000건, scoring="average_precision", n_repeats=10, random_state=42 —
#       borderline_recheck.py::step3_permutation()과 동일)으로 α를 재현하고 γ도 계산해서 나란히 비교.
#       γ는 CalibratedClassifierCV(XGBoost -> isotonic)를 estimator로 넘김 -> 섞은 입력이 XGBoost와
#       보정기를 차례로 통과한 확률로 점수를 냄(main.py::run_final(compute_permutation=True)와 같은 방식).
# 실행: python -X utf8 -m Simulator.Detection.permutation_gamma
# 결과: permutation_results/ (Track별 csv/md, 그림 png, json)
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from Simulator.main import (
    CALIBRATION_VAL_SIZE,
    TRACK_SCENARIOS,
    get_or_generate_final_dataset,
    load_tuned_hyperparams,
)
from Simulator.Detection.train_eval import fit_calibrator, prepare_categorical, split_data, train_model
from Simulator.Detection.eval_confidence import _setup_korean_font  # import 시점에 한글 글꼴도 설정됨
from Simulator.schema_utils import get_feature_columns

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OUT_DIR = _PROJECT_ROOT / "permutation_results"

PERM_N_REPEATS = 10   # borderline_recheck.py / main.py와 동일
PERM_RANDOM_STATE = 42
TOP_K_PLOT = 7

# 보고서에 적힌 α 기준 상위 3개(재현 확인용)
REPORTED_ALPHA = {
    "B": {"number_type": 0.3962, "sms_to_call_gap": 0.1319, "hour_bucket": 0.0547},
    "C": {"is_reliable_url": 0.0823, "number_type": 0.0545, "sms_to_call_gap": 0.0397},
}

KOREAN_NAMES = {
    "number_type": "발신번호 종류·대역",
    "hour_bucket": "활동 시간대",
    "first_contact_type": "접촉 개시 채널",
    "has_prior_history": "과거 통화 이력",
    "in_contacts": "연락처 저장 여부",
    "has_repeat_contact": "재연락 여부",
    "repeat_gap": "재연락 간격",
    "sms_to_call_gap": "문자→통화 전환 간격",
    "is_sequential_callers": "순차적 복수번호 접촉",
    "is_num_in_msg": "문자 내 번호 포함 여부",
    "inner_num_differs": "발신번호-문자 내 번호 불일치",
    "msg_number_official_match": "문자 내 번호-대표번호 일치",
    "is_url_in_msg": "문자 내 URL 포함 여부",
    "is_reliable_url": "문자 내 URL 신뢰 여부",
    "has_appinstall_link": "앱 설치 유도 신호",
    "cold_contact": "낯선 접촉",
    "is_sms_initiated_unreg": "미등록 발신자의 문자 개시",
    "repeat_pressure_intensity": "재연락 압박 강도",
    "structural_phishing_score": "문자 구조 위험 점수",
}


def _perm(estimator, X, y):
    r = permutation_importance(
        estimator, X, y, scoring="average_precision",
        n_repeats=PERM_N_REPEATS, random_state=PERM_RANDOM_STATE,
    )
    return r.importances_mean, r.importances_std


def compute_track(scenario_name, tracks, train_set, train_sub, val_calib, test_set, model_type, hyperparams):
    cols = get_feature_columns(track_filter=tracks)
    X_test, y_test = test_set[cols], test_set["is_phishing"]

    # α: train_set 100%로 학습, 보정 없음
    model_alpha = train_model(train_set, feature_cols=cols, hyperparams=hyperparams, model_type=model_type)
    # γ: train_sub로 학습 + val_calib으로 isotonic 보정
    model_gamma = train_model(train_sub, feature_cols=cols, hyperparams=hyperparams, model_type=model_type)
    calibrator = fit_calibrator(model_gamma, val_calib, feature_cols=cols, method="isotonic")

    a_mean, a_std = _perm(model_alpha, X_test, y_test)
    g_mean, g_std = _perm(calibrator, X_test, y_test)

    table = pd.DataFrame({
        "변수명": cols,
        "한글명": [KOREAN_NAMES.get(c, c) for c in cols],
        "α 평균": a_mean,
        "α 표준편차": a_std,
        "γ 평균": g_mean,
        "γ 표준편차": g_std,
    })
    table["α 순위"] = table["α 평균"].rank(ascending=False, method="min").astype(int)
    table["γ 순위"] = table["γ 평균"].rank(ascending=False, method="min").astype(int)
    table = table.sort_values("γ 순위").reset_index(drop=True)

    baseline = {
        "alpha_test_AP": float(average_precision_score(y_test, model_alpha.predict_proba(X_test)[:, 1])),
        "gamma_test_AP": float(average_precision_score(y_test, calibrator.predict_proba(X_test)[:, 1])),
    }
    return table, baseline


def check_reproduction(scenario_name, table):
    rows = []
    alpha = dict(zip(table["변수명"], table["α 평균"]))
    for feat, reported in REPORTED_ALPHA[scenario_name].items():
        rows.append({"feature": feat, "reported": reported, "recomputed": round(float(alpha[feat]), 4),
                     "match_4dp": round(float(alpha[feat]), 4) == reported})
    return rows


def write_tables(scenario_name, table, out_dir):
    out = table[["변수명", "한글명", "α 평균", "γ 평균", "γ 표준편차", "γ 순위", "α 순위"]].copy()
    out.to_csv(out_dir / f"permutation_track{scenario_name}.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    lines = [
        f"### Track {scenario_name} (test_set, scoring=average_precision, n_repeats={PERM_N_REPEATS}, random_state={PERM_RANDOM_STATE})",
        "",
        "| 변수명 | 한글명 | α 평균 | γ 평균 | γ 표준편차 | γ 순위 | α 순위 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in out.itertuples(index=False):
        lines.append(f"| `{r[0]}` | {r[1]} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.4f} | {r[5]} | {r[6]} |")
    md = "\n".join(lines) + "\n"
    (out_dir / f"permutation_track{scenario_name}.md").write_text(md, encoding="utf-8")
    return md


def plot_top_k(tables, out_path):
    _setup_korean_font()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    tops = {name: t.head(TOP_K_PLOT) for name, t in tables.items()}
    x_max = max(float((t["γ 평균"] + t["γ 표준편차"]).max()) for t in tops.values()) * 1.12
    titles = {"B": "Track B (행동 메타데이터)", "C": "Track C (행동 메타데이터 + 구조적 신호)"}

    for ax, (name, top) in zip(axes, tops.items()):
        top = top.iloc[::-1]  # 1위가 맨 위로 오도록
        y = np.arange(len(top))
        ax.barh(y, top["γ 평균"], xerr=top["γ 표준편차"], color="#8c8c8c", edgecolor="#333333",
                linewidth=0.6, height=0.6, error_kw={"ecolor": "#000000", "elinewidth": 0.9, "capsize": 3})
        ax.set_yticks(y)
        ax.set_yticklabels(top["한글명"], fontsize=9.5)
        ax.set_xlim(0, x_max)
        ax.set_title(titles.get(name, f"Track {name}"), fontsize=11)
        ax.set_xlabel("Permutation Importance (PR-AUC 감소량)", fontsize=9.5)
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def summarize(name, table):
    top3_a = list(table.sort_values("α 순위")["변수명"].head(3))
    top3_g = list(table["변수명"].head(3))
    changed = table[table["α 순위"] != table["γ 순위"]]
    return {
        "top3_alpha": top3_a,
        "top3_gamma": top3_g,
        "top3_same_order": top3_a == top3_g,
        "top3_same_set": set(top3_a) == set(top3_g),
        "rank_changed": [
            {"feature": r["변수명"], "alpha_rank": int(r["α 순위"]), "gamma_rank": int(r["γ 순위"])}
            for _, r in changed.iterrows()
        ],
    }


def main():
    df = prepare_categorical(get_or_generate_final_dataset())
    train_set, test_set = split_data(df)
    train_sub, val_calib = train_test_split(
        train_set, test_size=CALIBRATION_VAL_SIZE, random_state=42, stratify=train_set["is_phishing"]
    )
    model_type, hyperparams = load_tuned_hyperparams()
    model_type = model_type or "XGBoost"
    print(f"model_type={model_type}, train_set={len(train_set)}, train_sub={len(train_sub)}, "
          f"val_calib={len(val_calib)}, test_set={len(test_set)}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables, payload = {}, {
        "settings": {
            "model_type": model_type, "data": f"test_set {len(test_set)}건", "scoring": "average_precision",
            "n_repeats": PERM_N_REPEATS, "random_state": PERM_RANDOM_STATE,
            "alpha": f"train_set {len(train_set)}건 학습, 보정 없음",
            "gamma": f"train_sub {len(train_sub)}건 학습 + val_calib {len(val_calib)}건 isotonic 보정",
        },
        "tracks": {},
    }
    for scenario_name, tracks in TRACK_SCENARIOS.items():
        print(f"[Track {scenario_name}] 계산 중...")
        table, baseline = compute_track(scenario_name, tracks, train_set, train_sub, val_calib, test_set,
                                        model_type, hyperparams)
        tables[scenario_name] = table
        repro = check_reproduction(scenario_name, table)
        md = write_tables(scenario_name, table, _OUT_DIR)
        summary = summarize(scenario_name, table)
        payload["tracks"][scenario_name] = {
            "baseline": baseline, "alpha_reproduction": repro, "summary": summary,
            "table": table.to_dict(orient="records"),
        }
        print(md)
        print("  α 재현:", repro)
        print("  요약:", summary)

    plot_top_k(tables, _OUT_DIR / "permutation_gamma_top7.png")
    with open(_OUT_DIR / "permutation_gamma.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"저장: {_OUT_DIR}")
    return payload


if __name__ == "__main__":
    main()
