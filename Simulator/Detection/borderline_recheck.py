# -m final 결과에서 경계선으로 남은 3가지를 같은 dataset/split로 재확인하는 스크립트.
#
# 1) 약한 파생 2개(is_sms_initiated_unreg, repeat_pressure_intensity)를 빼고 재학습
# 2) is_global 빈도 및 number_type과의 중복
# 3) Track C permutation importance (gain에서 sms_to_call이 사라진 것이 착시인지)
#
# 실행 (프로젝트 루트):
#   python -X utf8 -m Simulator.Detection.borderline_recheck

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from Simulator.schema import Track
from Simulator.schema_utils import get_feature_columns
from Simulator.Detection.train_eval import (
    SCALAR_METRIC_KEYS,
    evaluate,
    prepare_categorical,
    split_data,
    train_model,
)
from Simulator.main import get_or_generate_final_dataset, load_tuned_hyperparams, TRACK_SCENARIOS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OUT_DIR = _PROJECT_ROOT / "borderline_recheck_results"

WEAK_DERIVED = ["is_sms_initiated_unreg", "repeat_pressure_intensity"]
PERM_N_REPEATS = 10
SCALAR_KEYS = SCALAR_METRIC_KEYS


def _scalar_metrics(result):
    return {k: float(result[k]) if isinstance(result[k], (np.floating, np.integer)) else result[k]
            for k in SCALAR_KEYS}


def _gain_importance(result):
    return {k: float(v) for k, v in result["feature_importance"].items()}


def _drop_weak(cols):
    return [c for c in cols if c not in WEAK_DERIVED]


def _fit_eval(train_set, test_set, cols, hyperparams, model_type):
    model = train_model(train_set, feature_cols=cols, hyperparams=hyperparams, model_type=model_type)
    result = evaluate(
        model, test_set, feature_cols=cols, threshold_df=train_set, pr_auc_method="average_precision"
    )
    return model, _scalar_metrics(result), _gain_importance(result)


def _print_metrics(title, metrics):
    print(f"  [{title}] F1={metrics['f1']:.4f}  PR-AUC={metrics['PR-AUC']:.4f}  "
          f"precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  "
          f"Lift@5%={metrics['Lift@Top5%']}")


def step1_drop_weak(train_set, test_set, hyperparams, model_type):
    print("\n" + "=" * 80)
    print("[1] 약한 파생 2개 제거 재학습 (동일 train/test, 동일 하이퍼파라미터)")
    print(f"    제거 대상: {WEAK_DERIVED}")
    print("=" * 80)

    payload = {}
    for scenario_name, tracks in TRACK_SCENARIOS.items():
        full_cols = get_feature_columns(track_filter=tracks)
        reduced_cols = _drop_weak(full_cols)
        print(f"\n  Track {scenario_name}: full={len(full_cols)} → reduced={len(reduced_cols)}")

        _, full_metrics, full_gain = _fit_eval(train_set, test_set, full_cols, hyperparams, model_type)
        _, reduced_metrics, reduced_gain = _fit_eval(
            train_set, test_set, reduced_cols, hyperparams, model_type
        )
        _print_metrics("full   ", full_metrics)
        _print_metrics("reduced", reduced_metrics)

        diff = {k: round(reduced_metrics[k] - full_metrics[k], 6) for k in SCALAR_KEYS
                if isinstance(full_metrics[k], float)}
        print("  diff(reduced - full): "
              f"F1={diff['f1']:+.4f}  PR-AUC={diff['PR-AUC']:+.4f}  "
              f"recall={diff['recall']:+.4f}")

        payload[scenario_name] = {
            "full_n_features": len(full_cols),
            "reduced_n_features": len(reduced_cols),
            "dropped": [c for c in WEAK_DERIVED if c in full_cols],
            "full": full_metrics,
            "reduced": reduced_metrics,
            "diff(reduced - full)": diff,
            "full_gain_importance": full_gain,
            "reduced_gain_importance": reduced_gain,
        }
    return payload


def step2_is_global(df):
    print("\n" + "=" * 80)
    print("[2] is_global 빈도 및 number_type 중복")
    print("=" * 80)

    n = len(df)
    vc = df["is_global"].value_counts(dropna=False).to_dict()
    n_global = int((df["is_global"] == 1).sum())
    rate = n_global / n

    crosstab_type = pd.crosstab(df["number_type"].astype(str), df["is_global"])
    # is_global=1 이어야 하는 대역
    international = df["number_type"].astype(str) == "00X(국제)"
    match_rate = float((international == (df["is_global"] == 1)).mean())

    phishing_given_global = None
    if n_global > 0:
        phishing_given_global = float(df.loc[df["is_global"] == 1, "is_phishing"].mean())
    phishing_base = float(df["is_phishing"].mean())

    print(f"  전체 {n}건 중 is_global=1: {n_global}건 ({rate:.4%})")
    print(f"  is_global value_counts: {vc}")
    print(f"  number_type=='00X(국제)' 와 is_global==1 일치율: {match_rate:.4%}")
    print(f"  전체 피싱 비율: {phishing_base:.4%}")
    if phishing_given_global is not None:
        print(f"  is_global=1 중 피싱 비율: {phishing_given_global:.4%}")
    else:
        print("  is_global=1 표본이 없어 피싱 비율을 계산할 수 없음")
    print("  number_type × is_global:")
    print(crosstab_type.to_string())

    return {
        "n": n,
        "is_global_value_counts": {str(k): int(v) for k, v in vc.items()},
        "n_is_global_1": n_global,
        "is_global_1_rate": rate,
        "match_rate_with_00X": match_rate,
        "phishing_base_rate": phishing_base,
        "phishing_rate_given_is_global_1": phishing_given_global,
        "number_type_by_is_global": {
            str(idx): {str(col): int(crosstab_type.loc[idx, col]) for col in crosstab_type.columns}
            for idx in crosstab_type.index
        },
    }


def step3_permutation(train_set, test_set, hyperparams, model_type):
    print("\n" + "=" * 80)
    print("[3] Track B/C permutation importance (test_set, scoring=average_precision)")
    print("    gain에서 Track C의 sms_to_call이 사라진 것이 착시인지 확인")
    print("=" * 80)

    payload = {}
    for scenario_name, tracks in TRACK_SCENARIOS.items():
        cols = get_feature_columns(track_filter=tracks)
        model = train_model(train_set, feature_cols=cols, hyperparams=hyperparams, model_type=model_type)
        result = evaluate(
            model, test_set, feature_cols=cols, threshold_df=train_set, pr_auc_method="average_precision"
        )
        gain = _gain_importance(result)

        X_test = test_set[cols]
        y_test = test_set["is_phishing"]
        perm = permutation_importance(
            model, X_test, y_test, scoring="average_precision",
            n_repeats=PERM_N_REPEATS, random_state=42,
        )
        perm_mean = {col: float(v) for col, v in zip(cols, perm.importances_mean)}
        perm_std = {col: float(v) for col, v in zip(cols, perm.importances_std)}
        ranked = sorted(perm_mean.items(), key=lambda kv: kv[1], reverse=True)

        print(f"\n  Track {scenario_name} permutation 상위 8개 (mean ± std):")
        for name, val in ranked[:8]:
            print(f"    {name:32s} {val: .6f}  ± {perm_std[name]:.6f}   gain={gain.get(name, 0):.4f}")

        for focus in ["sms_to_call", "is_reliable_url", "cold_contact", "is_global",
                      "is_sms_initiated_unreg", "repeat_pressure_intensity"]:
            if focus in perm_mean:
                print(f"  [focus] {focus}: perm={perm_mean[focus]:.6f}  gain={gain.get(focus, 0):.4f}")

        payload[scenario_name] = {
            "gain_importance": gain,
            "permutation_mean": perm_mean,
            "permutation_std": perm_std,
            "permutation_ranked": [
                {"rank": i + 1, "feature": name, "mean": val, "std": perm_std[name],
                 "gain": gain.get(name, 0.0)}
                for i, (name, val) in enumerate(ranked)
            ],
        }
    return payload


def main():
    df = get_or_generate_final_dataset()
    df = prepare_categorical(df)
    train_set, test_set = split_data(df)
    model_type, hyperparams = load_tuned_hyperparams()
    model_type = model_type or "XGBoost"

    print(f"model_type={model_type}")
    print(f"train={len(train_set)}, test={len(test_set)}")

    step1 = step1_drop_weak(train_set, test_set, hyperparams, model_type)
    step2 = step2_is_global(df)
    step3 = step3_permutation(train_set, test_set, hyperparams, model_type)

    payload = {
        "model_type": model_type,
        "weak_derived_dropped": WEAK_DERIVED,
        "perm_n_repeats": PERM_N_REPEATS,
        "step1_drop_weak_derived": step1,
        "step2_is_global": step2,
        "step3_permutation_importance": step3,
    }
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "borderline_recheck.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")
    return payload


if __name__ == "__main__":
    main()
