# 이 파일은 Simulator/Detection/plot_curve.py 위치에 두는 것을 전제로 작성됨
# (eval_confidence.py와 동일한 위치/import 컨벤션 -> _PROJECT_ROOT = parent.parent.parent).
#
# Track B(단말만) / Track C(단말+구조적신호) 총 2개 조합에 대해 아래 2종 그래프를 생성:
#   1. Learning Curve      : train_sub 크기별 train/test PR-AUC 추이 (Raw/Calibrated 구분)
#   2. Cumulative Lift Curve: 예측 확률 상위 %별 누적 lift (Raw/Calibrated 구분)
# PR Curve/SHAP은 보고서에서 제외되어 이 스크립트에서도 뺌.
#
# 실행 방법 (프로젝트 루트에서):
#   python -X utf8 -m Simulator.Detection.plot_curve
# 결과는 plot_results/ 폴더에 저장됨.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

from Simulator.main import (
    get_or_generate_final_dataset,
    load_tuned_hyperparams,
    CALIBRATION_VAL_SIZE,
    TRACK_SCENARIOS,
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
_OUT_DIR = _PROJECT_ROOT / "plot_results"

# Learning Curve에서 스윕할 train 크기 비율(train_sub 기준). 촘촘할수록 정확하지만 매 지점마다
# 모델을 새로 학습(calibration=True면 calibrator까지)하므로 계산 비용이 배로 늘어남 -> 10개로 제한.
LEARNING_CURVE_FRACTIONS = np.linspace(0.1, 1.0, 10)

# Cumulative Lift Curve에서 스윕할 population 비율(상위 x%). 1% 단위로 100개 지점.
LIFT_CURVE_FRACTIONS = np.linspace(0.01, 1.0, 100)

def _prepare_splits():
    """main.py::run_final()과 동일한 분할 방식(train_sub/val_calib/test_set)을 재사용.
    이 파일의 모든 plot 함수가 동일한 분할을 공유해야 서로 비교 가능한 그래프가 나옴."""
    df = get_or_generate_final_dataset()
    df = prepare_categorical(df)
    train_set, test_set = split_data(df)
    train_sub, val_calib = train_test_split(
        train_set,
        test_size=CALIBRATION_VAL_SIZE,
        random_state=42,
        stratify=train_set["is_phishing"],
    )
    return train_sub, val_calib, test_set


def _get_feature_cols(track_name):
    return get_feature_columns(track_filter=TRACK_SCENARIOS[track_name])


def _get_model_type_and_hyperparams():
    model_type, hyperparams = load_tuned_hyperparams()
    return model_type or "XGBoost", hyperparams


def _get_proba(model, calibrator, X, use_calibration):
    """use_calibration 여부에 따라 원본 model 확률 또는 calibrator 확률 반환.
    train_eval.py::evaluate()의 분기 방식과 동일하게 맞춤."""
    if use_calibration:
        return calibrator.predict_proba(X)[:, 1]
    return model.predict_proba(X)[:, 1]


# ------------------------------------------------------------
# 1. Learning Curve
# ------------------------------------------------------------
def compute_learning_curve_data(track_name, use_calibration, train_sub, val_calib, test_set):
    """train_sub를 여러 크기로 잘라가며(LEARNING_CURVE_FRACTIONS) 각 크기에서 모델을 새로 학습하고,
    train subset 자기 자신과 test_set에 대한 PR-AUC(average_precision)를 기록.
    use_calibration=True면 매 크기마다 val_calib으로 calibrator도 새로 적합해서 그 확률로 채점
    (학습 데이터가 줄어들수록 calibration 품질 자체도 같이 나빠지는지까지 반영하기 위함)."""
    feature_cols = _get_feature_cols(track_name)
    model_type, hyperparams = _get_model_type_and_hyperparams()

    n_total = len(train_sub)
    sizes, train_scores, val_scores = [], [], []

    for frac in LEARNING_CURVE_FRACTIONS:
        n_sub = max(int(n_total * frac), 20)  # 표본이 너무 적으면(양성 0개 등) 불안정해지므로 최소 20개 보장
        if n_sub >= n_total:
            subset = train_sub
        else:
            subset, _ = train_test_split(
                train_sub,
                train_size=n_sub,
                random_state=42,
                stratify=train_sub["is_phishing"],
            )

        model = train_model(subset, feature_cols=feature_cols, hyperparams=hyperparams, model_type=model_type)
        calibrator = (
            fit_calibrator(model, val_calib, feature_cols=feature_cols, method="isotonic")
            if use_calibration else None
        )

        X_train, _ = cast_object_columns(subset[feature_cols].copy())
        X_test, _ = cast_object_columns(test_set[feature_cols].copy())

        proba_train = _get_proba(model, calibrator, X_train, use_calibration)
        proba_test = _get_proba(model, calibrator, X_test, use_calibration)

        train_scores.append(average_precision_score(subset["is_phishing"], proba_train))
        val_scores.append(average_precision_score(test_set["is_phishing"], proba_test))
        sizes.append(n_sub)

    return np.array(sizes), np.array(train_scores), np.array(val_scores)


def plot_learning_curve(track_name, use_calibration, train_sub, val_calib, test_set, out_dir):
    sizes, train_scores, val_scores = compute_learning_curve_data(
        track_name, use_calibration, train_sub, val_calib, test_set
    )

    calib_label = "Calibrated" if use_calibration else "Raw"
    plt.figure(figsize=(7, 6))
    plt.plot(sizes, train_scores, "o-", color="darkorange", label="Train PR-AUC")
    plt.plot(sizes, val_scores, "o-", color="royalblue", label="Test PR-AUC")
    plt.title(f"[Track {track_name} | {calib_label}] Learning Curve", fontsize=12, pad=12)
    plt.xlabel("Training Set Size")
    plt.ylabel("PR-AUC (Average Precision)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / f"learning_curve_track{track_name}_{calib_label.lower()}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


# ------------------------------------------------------------
# 2. Cumulative Lift Curve
# ------------------------------------------------------------
def plot_cumulative_lift(track_name, use_calibration, model, calibrator, test_set, out_dir):
    feature_cols = _get_feature_cols(track_name)
    X_test, _ = cast_object_columns(test_set[feature_cols].copy())
    y_test = test_set["is_phishing"].to_numpy()

    proba = _get_proba(model, calibrator, X_test, use_calibration)
    order = np.argsort(proba)[::-1]  # 예측 확률 내림차순 정렬
    y_sorted = y_test[order]

    n = len(y_sorted)
    base_rate = y_sorted.mean()

    cum_lift = []
    for frac in LIFT_CURVE_FRACTIONS:
        cutoff = max(int(np.ceil(n * frac)), 1)
        top_rate = y_sorted[:cutoff].mean()
        cum_lift.append(top_rate / base_rate if base_rate > 0 else 0.0)

    calib_label = "Calibrated" if use_calibration else "Raw"

    plt.figure(figsize=(7, 6))
    plt.plot(LIFT_CURVE_FRACTIONS * 100, cum_lift, color="royalblue", label="Cumulative Lift")
    plt.hlines(1.0, 0, 100, colors="gray", linestyles="--", label="Baseline (Lift=1)")
    plt.title(f"[Track {track_name} | {calib_label}] Cumulative Lift Curve", fontsize=12, pad=12)
    plt.xlabel("Top % of Population")
    plt.ylabel("Lift")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / f"cumulative_lift_track{track_name}_{calib_label.lower()}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path



# ------------------------------------------------------------
# 전체 실행
# ------------------------------------------------------------
def run_all_plots():
    """Track B/C별로 Learning Curve와 Cumulative Lift Curve를 Calibration 유/무(Raw/Calibrated)로
    구분해서 plot_results/ 폴더에 저장. model/calibrator는 Track당 한 번만 학습해서 Lift Curve가
    공유함(Raw/Calibrated가 같은 model 기반이므로). Learning Curve는 크기별로 자체적으로 다시
    학습하므로 별도 루프."""
    train_sub, val_calib, test_set = _prepare_splits()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for track_name in TRACK_SCENARIOS:
        feature_cols = _get_feature_cols(track_name)
        model_type, hyperparams = _get_model_type_and_hyperparams()

        # Lift Curve용 model+calibrator는 Track당 한 번만 학습(Raw/Calibrated가 같은 model 공유).
        model = train_model(train_sub, feature_cols=feature_cols, hyperparams=hyperparams, model_type=model_type)
        calibrator = fit_calibrator(model, val_calib, feature_cols=feature_cols, method="isotonic")

        for use_calibration in (False, True):
            saved_paths.append(
                plot_learning_curve(track_name, use_calibration, train_sub, val_calib, test_set, _OUT_DIR)
            )
            saved_paths.append(
                plot_cumulative_lift(track_name, use_calibration, model, calibrator, test_set, _OUT_DIR)
            )

    print(f"총 {len(saved_paths)}개 그래프 저장 완료 -> {_OUT_DIR}")
    for p in saved_paths:
        print(f"  - {p.name}")
    return saved_paths


if __name__ == "__main__":
    run_all_plots()