# run_feature_selection_pipeline() 내에서 진행되는 파생 feature 13개의 선정 파이프라인.
#
# [0단계: 기본 작업]
#   1. dataset 생성 (1회)
#   2. train/test 고정 분할 (1회, 1·4단계가 test_set 재사용)
#   3. train_set 내부에서 추가 분할 (1회, 3단계 전용)
#   4. feature 목록 확정 (1회, 1단계에서 바로 사용)
#
# [1단계: baseline vs full 비교]
#   5. 모델 학습 #1 — baseline_model: train_set 전체를 "원본 feature만"(baseline_cols)으로 학습
#   6. 모델 학습 #2 — full_model: train_set 전체를 "원본+파생 13개 전부"(full_cols)로 학습
#   7. 둘 다 test_set으로 평가 -> baseline_metrics vs full_metrics 비교표 산출
#
# [2단계: 다중공선성 점검 + 데이터 누수 플래그]
#   8. 다중공선성 점검 및 제거
#   9. 데이터 누수 플래그
#
# [3단계: Embedded method(train_sub) + SHAP(참고) + Permutation Importance(val_sub, 실제 기준)]
#   10. 모델 학습 #3 — step3_model: train_sub(train_set의 80%)만으로 "원본 + 2단계 생존 feature"를 학습
#   11. SHAP 계산(참고용)
#   12. Permutation Importance 계산
#   13. Permutation Importance ≤ 0인 feature 제거
#
# [4단계: 최종 재학습]
#   14. 모델 학습 #4 — final_model: train_set 전체(70%)를 "원본 + 최종 확정 feature"로 재학습
#   15. test_set으로 최종 평가 -> 이게 진짜 "이 feature 조합으로 나온 최종 성능"
#
# run_sensitivity()/run_stress_sensitivity()가 K-Fold로 "생성 파라미터"를 스윕하는 것과 달리,
# 이건 dataset 1개를 고정 분할해서 "feature 선정" 자체를 진단하는 용도라 K-Fold를 쓰지 않음.
# (다중공선성/누수/permutation importance 기준으로 "어떤 feature를 넣을지" 결정하는 과정 자체를
#  K번 반복하면 반복마다 결정이 달라질 수 있어, "최종 조합이 무엇인가"라는 질문에 답할 수 없게 됨)
#
# run_stability_check() — [선정 이후] 사후 검증 전용 함수 (K-Fold 사용)
#   위 선정 과정(run_feature_selection_pipeline())으로 "이미 확정된" feature 조합의 test 성능
#   차이가 실제 효과인지, 단일 실행에서 나온 우연(노이즈)인지 확인하는 별도 함수.
#   "무엇을 고를지"를 다시 정하는 게 아니라 "이미 고른 것이 fold가 바뀌어도 일관된가"만
#   사후 확인하는 것이라, 위 원칙(선정 과정=1회 고정 분할)과 모순되지 않음.
#   dataset은 동일하게 1개만 생성하고(재생성 X), run_sensitivity()와 동일하게
#   StratifiedKFold로 나눈 K개 fold에서 baseline/final 모델을 반복 학습·평가함.
import json
from pathlib import Path
from typing import Optional

import numpy as np
import shap
import shap.explainers._tree as _shap_tree_mod
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, StratifiedKFold

from Simulator.Generation.dataset_builder import build_dataset
from Simulator.schema_utils import get_feature_columns
from Simulator.Detection.train_eval import (
    SCALAR_METRIC_KEYS,
    evaluate,
    prepare_categorical,
    split_data,
    train_model,
)
from Simulator.Detection.derived_features import add_candidate_features
from Simulator.Detection.sensitivity_analysis import summarize_fold_results

# 결과 저장 폴더: 실행 위치(cwd)와 무관하게 항상 프로젝트 루트 기준으로 고정.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RESULTS_DIR = _PROJECT_ROOT / "feature_selection_results"
# run_stability_check() 결과 저장 폴더.
_STABILITY_RESULTS_DIR = _PROJECT_ROOT / "feature_stability_results"

# add_candidate_features()가 만드는 13개 파생 컬럼 중, 아직 SCHEMA에 미등록인 9개(=여전히 ablation
# 검증 후보인 것들)만 나열한 목록. 나머지 4개(FINAL_CANDIDATE_FEATURES)는 검증 통과 후
# schema_columns.py의 SCHEMA에 확정 등록되어 이제 get_feature_columns()(=baseline_cols)에 이미
# 포함되므로, 여기 다시 넣으면 baseline_cols와 full_cols에서 같은 컬럼이 중복 선택됨 -> 제외함.
CANDIDATE_DERIVED_FEATURES = [
    "cross_channel_urgency_score",
    "is_malicious_bait_sms",
    "urgency_path",
    "unreg_sender_with_url",
    "is_rapid_s2c_contact",
    "is_zero_gap_repeat",
    "has_any_msg_bait",
    "is_high_risk_number_type",
    "suspicious_unreg_number_combo",
]

DEFAULT_MULTICOLLINEARITY_THRESHOLD = 0.8  # 다중공선성(Multicollinearity) 중복 제거를 위해 사용하는 임계값.
DEFAULT_LEAKAGE_THRESHOLD = 0.95            # 데이터 누수(Data Leakage) 탐지하기 위해 사용하는 임계값.
DEFAULT_VAL_SIZE = 0.2                      # train_set 중 permutation importance 검증용으로 사용할 데이터로 분리할 비율
DEFAULT_PERM_N_REPEATS = 10                 # permutation importance 반복 횟수(우연에 따른 흔들림 줄이기 위해 반복 진행)

# run_feature_selection_pipeline() 실행 결과(2026-09-19, N=100,000)로 확정된 최종 파생 feature 4개.
# 참고용 상수. SCHEMA에 이미 등록되어 get_feature_columns()(=baseline)에 포함되므로
# run_stability_check()의 feature_sets에 다시 넣으면 안 됨(컬럼 중복). 별도 인자 없이
# run_stability_check()를 호출하면 기본값은 아래 FINAL이 아니라 CANDIDATE_DERIVED_FEATURES(잔여 9개)다.
FINAL_CANDIDATE_FEATURES = [
    "structural_phishing_score",
    "is_sms_initiated_unreg",
    "cold_contact",
    "repeat_pressure_intensity",
]

DEFAULT_STABILITY_K = 10  # run_stability_check() 기본 K. 재검증 성격이라 민감도분석의 DEFAULT_K(5)보다 크게 잡음.


def _metrics_summary(m):
    """evaluate() 반환값에서 threshold/accuracy/precision/recall/f1/PR-AUC/Lift@Top5~20%만 추려서
    저장용 요약 dict으로 만드는 함수. run_feature_selection_pipeline()과 run_stability_check()가 공유."""
    return {k: m[k] for k in SCALAR_METRIC_KEYS}


def _select_by_correlation(train_df, candidate_features, target_col="is_phishing", threshold=0.8):
    """train set 기준 다중공선성 점검을 진행하는 함수. 
    후보 파생 feature끼리의 상관계수가 threshold를 넘는 쌍이 있으면,
    "is_phishing"과의 상관계수가 더 높은(=더 예측력 있는) 쪽만 남기고 나머지를 제거."""

    # corr_with_target : 각 feature와 is_phishing과의 상관계수. (각 feature가 혼자서 얼마나 예측력이 있는지)
    corr_with_target = train_df[candidate_features + [target_col]].corr()[target_col].drop(target_col)

    # 상관계수가 높은 순서대로 정렬 진행. (예측력이 더 좋은 feature를 남기는 게 합리적이기 때문)
    ordered = corr_with_target.abs().sort_values(ascending=False).index.tolist()

    # feature_corr : 파생 feature 후보들끼리의 상관계수 행렬. (feature들끼리 서로 얼마나 겹치는지 = 다중공선성)
    feature_corr = train_df[candidate_features].corr()

    # kept : 다중공선성 점검 후 남은 파생 feature 후보 목록
    # dropped : 다중공선성 점검으로 제거된 파생 feature 후보 목록
    kept, dropped = [], []
    for feat in ordered:
        conflict = None
        for kept_feat in kept:
            r = feature_corr.loc[feat, kept_feat]
            if abs(r) > threshold:
                conflict = (kept_feat, r)
                break
        if conflict is not None:
            dropped.append({
                "feature": feat,
                "correlated_with": conflict[0],
                "correlation": round(float(conflict[1]), 4),
            })
        else:
            kept.append(feat)
    return kept, dropped


def _check_leakage(train_df, candidate_features, target_col="is_phishing", threshold=0.95):
    """train set 기준 데이터 누수 "의심" 점검을 위한 함수. 
    파생 feature가 타깃과 지나치게(threshold 이상) 상관되면 플래그만 함(자동 제거 안 함) 
    -> 누수처럼 보여도 실제로는 핵심 유효 feature일 수 있으므로, 최종 판단은 permutation importance를 통해 결정.
    기록 및 참고용으로 진행."""
    corr = train_df[candidate_features + [target_col]].corr()[target_col].drop(target_col)
    return [
        {"feature": f, "correlation_with_target": round(float(corr[f]), 4)}
        for f in candidate_features if abs(corr[f]) > threshold
    ]


_SHAP_PATCHED = False


def _patch_shap_xgboost_base_score_bug():
    """SHAP이 xgboost에서 TreeExplainer 생성 시 ValueError가 나는 알려진 호환성 버그를 우회하기 위한 함수. 
    decode_ubjson_buffer()가 반환한 값에서 base_score의 대괄호만 벗겨서 되돌려줌. 
    모듈 전체에 한 번만 적용."""
    global _SHAP_PATCHED
    if _SHAP_PATCHED:
        return
    _orig_decode = _shap_tree_mod.decode_ubjson_buffer

    def _patched_decode(fd):
        jmodel = _orig_decode(fd)
        bs = jmodel["learner"]["learner_model_param"]["base_score"]
        if isinstance(bs, str) and bs.startswith("["):
            jmodel["learner"]["learner_model_param"]["base_score"] = bs.strip("[]")
        return jmodel

    _shap_tree_mod.decode_ubjson_buffer = _patched_decode
    _SHAP_PATCHED = True


def _compute_shap_importance(model, X):
    """train_sub에서 |SHAP value| 평균(feature별 기여도)을 계산. 
    참고/리포트용이며, 실제 가지치기 기준(permutation importance)과는 별개."""
    _patch_shap_xgboost_base_score_bug()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        # 이진분류에서 클래스별 리스트로 나오는 shap 버전 대응 -> 양성(1) 클래스 값 사용.
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    mean_abs = np.abs(shap_values).mean(axis=0)
    return {col: float(v) for col, v in zip(X.columns, mean_abs)}


def _compute_permutation_importance(model, X_val, y_val, n_repeats=DEFAULT_PERM_N_REPEATS, random_state=42):
    """val_sub(학습에 안 쓰인 데이터)에서 Permutation Importance 계산 진행하는 함수. 
    scoring="average_precision"(~PR-AUC)을 씀 -> threshold 선택에 의존하지 않는 지표라, evaluate() 로직 없이 판단 가능"""
    result = permutation_importance(
        model, X_val, y_val, scoring="average_precision",
        n_repeats=n_repeats, random_state=random_state,
    )
    return {col: float(v) for col, v in zip(X_val.columns, result.importances_mean)}


def run_feature_selection_pipeline(
    fixed_config,                       # config.py 모듈. build_dataset()에 그대로 전달.
    n: int = 100000,                    # build_dataset()에 그대로 전달.
    phishing_rate: Optional[float] = None, # build_dataset()에 그대로 전달.
    subgroup_ratio_key: str = "D",     # build_dataset()에 그대로 전달.
    sophistication="mid",               # build_dataset()에 전달되어 진행될 땐, FINAL_SOPHISTICATION이 사용됨.
    gen_seed: int = 42,                 # build_dataset()에 그대로 전달.
    multicollinearity_threshold: float = DEFAULT_MULTICOLLINEARITY_THRESHOLD,
    leakage_threshold: float = DEFAULT_LEAKAGE_THRESHOLD,
    val_size: float = DEFAULT_VAL_SIZE,
    out_dir: Path = _RESULTS_DIR,
) -> dict:

    # --- 0단계 : 기본 작업
    # 1. dataset 생성 (1회) => 원본 23개의 column + 13개의 파생 column인 총 36개의 column으로 만들어진 dataset 생성.
    df = build_dataset(
        n, phishing_rate=phishing_rate, subgroup_ratio_key=subgroup_ratio_key,
        random_state=gen_seed, config=fixed_config, sophistication=sophistication,
    ) # 원본 23개의 column으로 구성된 dataset 생성.
    df = add_candidate_features(df) # 생성한 df에 13개의 후보 파생 column 추가.
    df = prepare_categorical(df) # category dtype 변경 진행. (데이터 분할 전에 진행해야 함.)

    # 2. train/test 고정 분할
    train_set, test_set = split_data(df)  # train_set(70%), test_set(30%)로 분할.

    # 3. train_set 내부에서 추가 분할 (permutation importance 계산을 위한 과정)
    train_sub, val_sub = train_test_split(
        train_set, test_size=val_size, random_state=42, stratify=train_set["is_phishing"]
    ) # train_set을 train_sub(80%), val_sub(20%)로 분할.

    # 4. feature 목록 확정
    # baseline_cols : SCHEMA 기준 is_feature=True인 column 목록(원본 21개 + 이미 확정된 파생 4개 = 25개).
    # 확정된 4개(FINAL_CANDIDATE_FEATURES)는 SCHEMA에 등록돼 있어 자동으로 여기 포함됨 -> "지금 production이
    # 실제로 쓰는 feature 조합"이 baseline이 되는 셈(향후 새 파생 feature 후보를 검증할 때도 동일한 의미).
    baseline_cols = get_feature_columns()

    # full_cols : baseline_cols(25개) + 아직 미확정인 파생 feature 9개(CANDIDATE_DERIVED_FEATURES) => 총 34개
    full_cols = baseline_cols + CANDIDATE_DERIVED_FEATURES

    # --- 1단계 : baseline vs full 
    # 5. 모델 학습 #1 — baseline_model: train_set 전체를 "원본 feature만"(baseline_cols)으로 학습
    # pr_auc_method="average_precision": baseline/full/final 모델끼리 비교하는 게 목적이라,
    # 직선 보간 편향이 없는 average_precision_score를 씀(민감도 분석과는 다른 기준).
    baseline_model = train_model(train_set, feature_cols=baseline_cols)
    baseline_metrics = evaluate(baseline_model, test_set, feature_cols=baseline_cols, threshold_df=train_set, pr_auc_method="average_precision")

    # 6. 모델 학습 #2 — full_model: train_set 전체를 "원본+파생 13개 전부"(full_cols)로 학습
    full_model = train_model(train_set, feature_cols=full_cols)

    # 7. 둘 다 test_set으로 평가 -> baseline_metrics vs full_metrics 비교표 산출 ("기본 feature vs 파생 feature 포함 모델" 성능 차이 확인 가능.)
    full_metrics = evaluate(full_model, test_set, feature_cols=full_cols, threshold_df=train_set, pr_auc_method="average_precision")

    # --- 2단계 : 다중공선성 점검(제거) + 데이터 누수 점검
    # 8. 다중공선성 점검 및 제거
    # kept_after_corr: 다중공선성 점검 후 통과한 파생 feature 목록, dropped_corr: 다중공선성 점검 후 제거된 파생 feature 목록
    kept_after_corr, dropped_corr = _select_by_correlation(
        train_set, CANDIDATE_DERIVED_FEATURES, threshold=multicollinearity_threshold
    )

    # 9. 데이터 누수 플래그
    #   (자동 제거 안 함 — 핵심 유효 feature일 수 있어서 최종 판단은 3단계 객관적 지표에 맡김)
    leakage_flags = _check_leakage(train_set, CANDIDATE_DERIVED_FEATURES, threshold=leakage_threshold)

    # step2_survivors : 8단계에서 다중공선성 점검 진행 후 통과된 feature 목록. (9단계에서 진행한 데이터 누수 플래그는 반영 X, 기록만)
    step2_survivors = kept_after_corr

    # --- 3단계 : Embedded method + SHAP + Permutation Importance
    # 10. 모델 학습 #3 — step3_model: train_sub만으로 "원본(baseline_cols) + 2단계 생존 feature(step2_survivors)"를 학습
    step3_cols = baseline_cols + step2_survivors
    step3_model = train_model(train_sub, feature_cols=step3_cols)

    # 11. SHAP 계산(일반화되는지 보장하지 않으므로 참고용으로 사용하여 결과 리포트에만 남김.)
    X_train_sub_step3 = train_sub[step3_cols]
    shap_importance = _compute_shap_importance(step3_model, X_train_sub_step3)

    # 12. Permutation Importance 계산
    #   (val_sub - 학습에 안 쓰인 처음 보는 데이터 사용하여 계산 진행.)
    X_val_step3 = val_sub[step3_cols]
    y_val_step3 = val_sub["is_phishing"]
    perm_importance = _compute_permutation_importance(step3_model, X_val_step3, y_val_step3)

    # 13. Permutation Importance ≤ 0인 feature 제거
    #   (섞어도 성능이 안 떨어지거나 오히려 좋아짐 = 기여 없는 feature) -> 여기까지 살아남은 것들이 "최종 확정 feature"(=final_candidates)
    final_candidates = [f for f in step2_survivors if perm_importance.get(f, 0.0) > 0]
    pruned_by_permutation = [f for f in step2_survivors if perm_importance.get(f, 0.0) <= 0]

    # --- 4단계: 최종 feature set으로 train_set 전체를 다시 학습 -> test_set 최종 평가
    # 14. 모델 학습 #4 — final_model: train_set 전체를 "원본(baseline_cols) + 최종 확정 feature(final_candidates)"로 재학습
    final_cols = baseline_cols + final_candidates # 원본 21개의 column + 13개의 파생 feature 목록 중 검증에서 통과한 목록
    final_model = train_model(train_set, feature_cols=final_cols)

    # 15. test_set으로 최종 평가 -> 이게 진짜 "이 feature 조합으로 나온 최종 성능"
    final_metrics = evaluate(final_model, test_set, feature_cols=final_cols, threshold_df=train_set, pr_auc_method="average_precision")

    payload = {
        "kind": "feature_selection_pipeline",
        "n": n,
        # baseline/full/final 세 모델의 test_set 성능을 한곳에 모아둔 요약(별도 계산 없이 비교만 편하게 하려고 추가).
        "metrics_comparison": {
            "baseline": _metrics_summary(baseline_metrics),
            "full": _metrics_summary(full_metrics),
            "final": _metrics_summary(final_metrics),
        },
        "step1_baseline_vs_full": {
            "baseline_feature_count": len(baseline_cols),
            "full_feature_count": len(full_cols),
            "baseline": _metrics_summary(baseline_metrics),
            "full": _metrics_summary(full_metrics),
        },
        "step2_multicollinearity": {
            "threshold": multicollinearity_threshold,
            "dropped_pairs": dropped_corr,
            "survivors": kept_after_corr,
        },
        "step2_leakage_check": {
            "threshold": leakage_threshold,
            "flagged_features(참고용, 자동 제거 안 함)": leakage_flags,
        },
        "step3_embedded_importance": {
            "candidates_evaluated": step2_survivors,
            "shap_importance(참고용)": {k: round(v, 6) for k, v in shap_importance.items()},
            "permutation_importance(실제 가지치기 기준, val_sub 기준)": {
                k: round(v, 6) for k, v in perm_importance.items()
            },
            "pruned_by_permutation(<=0)": pruned_by_permutation,
            "final_candidate_features": final_candidates,
        },
        "step4_final_retrain": {
            "final_feature_set": final_cols,
            "test_metrics": _metrics_summary(final_metrics),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "feature_selection_pipeline.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


def _diff_summary(baseline_summary: dict, final_summary: dict) -> dict:
    """final_summary - baseline_summary의 지표별 평균(mean) 차이만 따로 뽑아 요약.
    양수면 final(파생 feature 포함)이 baseline보다 평균적으로 더 좋았다는 뜻, 음수면 더 나빴다는 뜻.
    -> mean 차이가 baseline/final 각각의 sem(표준오차)보다 훨씬 작으면 "노이즈 범위 안"으로 판단 가능."""
    return {
        key: round(final_summary[key]["mean"] - baseline_summary[key]["mean"], 4)
        for key in final_summary
        if key in baseline_summary
    }


def run_stability_check(
    fixed_config,                              # config.py 모듈. build_dataset()에 그대로 전달.
    feature_sets=None,                         # {조합명: [파생 feature 목록]} 형태. baseline과 각각 비교됨.
                                                # None이면 {"remaining_candidates": CANDIDATE_DERIVED_FEATURES}(미확정 9개 묶음 1개)로 진행.
    k: int = DEFAULT_STABILITY_K,
    n: int = 100000,                           # build_dataset()에 그대로 전달.
    phishing_rate: Optional[float] = None,     # build_dataset()에 그대로 전달.
    subgroup_ratio_key: str = "D",             # build_dataset()에 그대로 전달.
    sophistication="mid",                      # build_dataset()에 전달되어 진행될 땐, FINAL_SOPHISTICATION이 사용됨.
    gen_seed: int = 42,                        # build_dataset()용 random_state (dataset은 1회만 생성).
    fold_seed: int = 7,                        # StratifiedKFold용 random_state. run_sensitivity()의 fold_seed와 동일한 역할.
    out_dir: Path = _STABILITY_RESULTS_DIR,
) -> dict:
    """run_feature_selection_pipeline()으로 이미 확정된 파생 feature 조합이 baseline 대비 보인 성능
    차이가 "실제 효과"인지 "단일 실행에서 나온 우연(노이즈)"인지 확인하는 사후 검증 함수.
    run_feature_selection_pipeline()과 달리 "무엇을 고를지"를 다시 결정하지 않고, "이미 고른"
    조합(들)만 대상으로 함 -> dataset은 동일하게 1개만 생성(재생성 X)하고, StratifiedKFold로 나눈
    K개 fold에서 baseline 모델 + feature_sets에 담긴 조합별 모델을 K번씩 반복 학습·평가해
    mean/std/sem을 비교함(run_sensitivity()와 동일한 K-Fold 패턴).

    feature_sets는 {"조합명": [파생 feature 목록]} 형태의 dict라 4개 전체 묶음뿐 아니라, 개별
    feature 1개씩이나 부분집합(예: 강한 신호 2개 vs 약한 신호 2개)도 한 번의 실행(같은 fold 분할)
    안에서 함께 비교할 수 있음 -> 조합끼리 서로 다른 fold로 평가되는 걸 방지해 공정한 비교가 됨.

    주의: feature_sets에 넣는 feature는 baseline_cols(=get_feature_columns())에 아직 없는 것이어야
    함. FINAL_CANDIDATE_FEATURES(구 4개 후보)는 이미 SCHEMA에 등록되어 baseline_cols에 포함되므로
    여기 다시 넣으면 컬럼이 중복 선택됨 -> 기본값은 아직 SCHEMA 미등록인 CANDIDATE_DERIVED_FEATURES
    (9개, 1단계 선정에서 탈락/대기 중인 후보)를 하나의 묶음으로 재검증하도록 되어 있음."""
    feature_sets = feature_sets or {"remaining_candidates": CANDIDATE_DERIVED_FEATURES}

    # dataset은 run_feature_selection_pipeline()과 동일하게 1회만 생성. split만 K번 바뀜.
    df = build_dataset(
        n, phishing_rate=phishing_rate, subgroup_ratio_key=subgroup_ratio_key,
        random_state=gen_seed, config=fixed_config, sophistication=sophistication,
    )
    df = add_candidate_features(df)
    df = prepare_categorical(df)

    baseline_cols = get_feature_columns()

    # feature_sets 안의 feature가 이미 baseline_cols에 있으면(예: SCHEMA에 이미 등록된 확정
    # feature를 실수로 다시 넣은 경우) train_model()에서 컬럼 중복 선택으로 알아보기 힘든
    # 에러(XGBoost 내부 dtype AttributeError 등)가 나므로, 미리 명확한 메시지로 막음.
    for name, feats in feature_sets.items():
        overlap = set(feats) & set(baseline_cols)
        if overlap:
            raise ValueError(
                f"feature_sets[{name!r}]에 이미 baseline에 포함된 feature가 있음: {sorted(overlap)} "
                "-> 이미 SCHEMA에 등록된 feature는 다시 비교 대상으로 넣지 않아도 됨."
            )

    # combo_cols: {조합명: baseline_cols + 해당 조합의 파생 feature 목록}
    combo_cols = {name: baseline_cols + feats for name, feats in feature_sets.items()}

    y = df["is_phishing"]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=fold_seed)

    # baseline_folds: baseline 모델의 fold별 evaluate() 결과(K개).
    # combo_folds: {조합명: [fold별 evaluate() 결과(K개)]}.
    baseline_folds = []
    combo_folds = {name: [] for name in feature_sets}
    for train_idx, val_idx in skf.split(df, y):
        train_fold = df.iloc[train_idx]
        val_fold = df.iloc[val_idx]

        # threshold는 train_fold에서 고르고 val_fold에 고정 적용 (run_sensitivity()와 동일한 낙관 편향 방지 방식).
        # pr_auc_method="average_precision": baseline vs 조합끼리 비교하는 게 목적이라 민감도 분석과는
        # 다른 기준(직선 보간 편향 없는 average_precision_score)을 씀.
        baseline_model = train_model(train_fold, feature_cols=baseline_cols)
        baseline_folds.append(evaluate(baseline_model, val_fold, feature_cols=baseline_cols, threshold_df=train_fold, pr_auc_method="average_precision"))

        # 같은 fold(train_fold/val_fold) 안에서 조합별 모델도 함께 학습·평가 -> baseline과 동일한
        # 분할을 공유해야 diff 비교가 공정함(조합마다 다른 fold를 쓰면 fold 차이가 diff에 섞여 들어감).
        for name, cols in combo_cols.items():
            model = train_model(train_fold, feature_cols=cols)
            combo_folds[name].append(evaluate(model, val_fold, feature_cols=cols, threshold_df=train_fold, pr_auc_method="average_precision"))

    baseline_summary = summarize_fold_results(baseline_folds)

    # combos: {조합명: {features, feature_count, summary, diff(vs baseline)}}
    combos = {}
    for name, feats in feature_sets.items():
        summary = summarize_fold_results(combo_folds[name])
        combos[name] = {
            "features": feats,
            "feature_count": len(combo_cols[name]),
            "summary": summary,
            # summary - baseline_summary의 mean 차이만 따로 뽑은 요약. 음수면 이 조합이 평균적으로
            # 더 낮았다는 뜻이지만, sem(표준오차) 대비 작은 차이면 노이즈 범위로 봐야 함(자동 판정은 안 함).
            "diff(vs baseline, mean 기준)": _diff_summary(baseline_summary, summary),
        }

    payload = {
        "kind": "feature_stability_check",
        "k": k,
        "n": n,
        "baseline_feature_count": len(baseline_cols),
        "baseline_summary": baseline_summary,
        "combos": combos,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "feature_stability_check.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload
