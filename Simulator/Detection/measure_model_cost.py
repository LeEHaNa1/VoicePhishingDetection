# 최종 탐지 모델의 크기·연산량 측정 파일.
# 역할: run_final()과 동일한 절차(dataset/분할/하이퍼파라미터/calibration)로 Track B/C 모델을 다시 학습한 뒤,
#       파라미터 수(트리 노드 수)·모델 파일 크기·1회 추론 연산량(근사)·입력 크기·추론/전처리 시간을 측정.
#       run_final()은 학습된 model을 반환/저장하지 않기 때문에, 측정용으로 여기서 같은 단계를 그대로 재현함.
#       재현이 맞는지는 출력되는 test PR-AUC를 -m final 결과와 비교해서 확인.
# 실행: python -X utf8 -m Simulator.Detection.measure_model_cost
# 결과: model_cost_results/ (모델 파일, model_cost.json, model_cost_table.md)
import json
import math
import os
import pickle
import platform
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from Simulator.main import (
    CALIBRATION_VAL_SIZE,
    TRACK_SCENARIOS,
    get_or_generate_final_dataset,
    load_tuned_hyperparams,
)
from Simulator.Detection.train_eval import (
    evaluate,
    fit_calibrator,
    prepare_categorical,
    split_data,
    train_model,
)
from Simulator.Detection.derived_features import add_candidate_features
from Simulator.schema_utils import get_feature_columns

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OUT_DIR = _PROJECT_ROOT / "model_cost_results"

N_TIMING_REPEATS = 1000   # 1건 추론을 반복할 횟수(서로 다른 test row 1,000건을 1건씩 예측)
N_WARMUP = 50             # 측정 전 워밍업 횟수(첫 호출의 초기화 비용 제외)
MIB = 2 ** 20             # 모델 크기 단위. 비교 논문 수치(92.188M x 4B = 351.7)와 맞추려고 2^20 bytes를 "MB"로 씀

# 파생 feature 4개(SCHEMA 등록분). 전처리 시간 측정 때 "원시 row"를 만들려고 이 컬럼만 지움.
DERIVED_FINAL_FEATURES = [
    "structural_phishing_score", "is_sms_initiated_unreg", "cold_contact", "repeat_pressure_intensity",
]


# ------------------------------------------------------------
# 측정 환경
# ------------------------------------------------------------
def _cpu_name():
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def environment_info():
    return {
        "os": platform.platform(),
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "xgboost": xgb.__version__,
        "lightgbm": lgb.__version__,
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


# ------------------------------------------------------------
# 트리 구조: 노드 수 / 깊이 / 샘플별 탐색 경로 길이
# ------------------------------------------------------------
def _xgb_leaf_depths(booster):
    """XGBoost booster -> ({(tree, leaf_node_id): 깊이(=루트~리프까지 비교 횟수)}, 노드 통계)."""
    tdf = booster.trees_to_dataframe()
    children = {row.ID: (row.Yes, row.No) for row in tdf.itertuples() if row.Feature != "Leaf"}
    leaf_depth = {}
    for t in tdf["Tree"].unique():
        queue = deque([(f"{t}-0", 0)])
        while queue:
            node_id, depth = queue.popleft()
            if node_id in children:
                for child in children[node_id]:
                    queue.append((child, depth + 1))
            else:
                leaf_depth[(int(t), int(node_id.split("-")[1]))] = depth
    internal = tdf[tdf["Feature"] != "Leaf"]
    stats = {
        "n_trees": int(tdf["Tree"].nunique()),
        "n_internal_nodes": int(len(internal)),
        "n_leaves": int((tdf["Feature"] == "Leaf").sum()),
        "n_categorical_splits": int(internal["Split"].isna().sum()),  # 범주형 split은 Split(임계값) 대신 Category(집합)를 가짐
    }
    return leaf_depth, stats


def _lgb_leaf_depths(booster):
    """LightGBM booster -> ({(tree, leaf_index): 깊이}, 노드 통계). node_depth는 루트=1이라 비교 횟수는 -1."""
    tdf = booster.trees_to_dataframe()
    is_leaf = tdf["split_feature"].isna()
    leaf_depth = {}
    for row in tdf[is_leaf].itertuples():
        leaf_idx = int(row.node_index.split("-L")[1])
        leaf_depth[(int(row.tree_index), leaf_idx)] = int(row.node_depth) - 1
    internal = tdf[~is_leaf]
    stats = {
        "n_trees": int(tdf["tree_index"].nunique()),
        "n_internal_nodes": int(len(internal)),
        "n_leaves": int(is_leaf.sum()),
        "n_categorical_splits": int((internal["decision_type"] == "==").sum()),
    }
    return leaf_depth, stats


def tree_structure_and_paths(model, model_type, X):
    """트리 수/노드 수/최대 깊이 + X 각 행이 실제로 지나간 경로 길이(비교 횟수 합) 분포."""
    if model_type == "XGBoost":
        booster = model.get_booster()
        leaf_depth, stats = _xgb_leaf_depths(booster)
        leaves = booster.predict(xgb.DMatrix(X, enable_categorical=True), pred_leaf=True).astype(int)
    else:
        booster = model.booster_
        leaf_depth, stats = _lgb_leaf_depths(booster)
        leaves = booster.predict(X, pred_leaf=True).astype(int)

    n_trees = leaves.shape[1]
    depth_lookup = np.vectorize(lambda t, leaf: leaf_depth[(t, leaf)])
    tree_idx = np.broadcast_to(np.arange(n_trees), leaves.shape)
    comparisons = depth_lookup(tree_idx, leaves).sum(axis=1)  # 행별: 모든 트리에서 지나간 내부 노드 수 합

    stats["n_total_nodes"] = stats["n_internal_nodes"] + stats["n_leaves"]
    stats["max_depth_observed"] = int(max(leaf_depth.values()))
    stats["comparisons_upper_bound"] = int(sum(
        max(d for (t, _), d in leaf_depth.items() if t == tree) for tree in range(n_trees)
    ))  # 트리별 최대 깊이의 합(어떤 입력이 와도 이보다 많이 비교하지 않음)
    return stats, comparisons


def inference_ops(comparisons, n_trees):
    """1회 추론 근사 연산 수(ops). 정의:
       비교 = 실제 지나간 내부 노드 수(노드 1개 = 결측 확인 포함 비교 1회로 셈, 범주형 split도 1회)
       덧셈 = 트리 리프값 합산 n_trees회 + base_score 1회
       sigmoid = exp 1 + 덧셈 1 + 나눗셈 1 = 3"""
    return comparisons + (n_trees + 1) + 3


def isotonic_info(calibrator):
    """CalibratedClassifierCV 안의 isotonic 보정기 개수/구간 수와 근사 연산 수.
    np.interp 1회 = 이진 탐색 ceil(log2 K) 비교 + 선형 보간 5 ops(뺄셈 2, 나눗셈 1, 곱셈 1, 덧셈 1)."""
    members = calibrator.calibrated_classifiers_
    thresholds = [len(m.calibrators[0].X_thresholds_) for m in members]
    ops_per_member = [math.ceil(math.log2(k)) + 5 if k > 1 else 0 for k in thresholds]
    return {
        "n_calibrated_members": len(members),
        "isotonic_thresholds_per_member": thresholds,
        "isotonic_params_total": int(sum(2 * k for k in thresholds)),  # (x, y) 쌍
        "isotonic_ops_per_member": ops_per_member,
        # member가 여러 개면 각 member가 base model을 다시 호출한 뒤 평균을 냄 -> 아래 calibrated 총 연산에 반영
    }


# ------------------------------------------------------------
# 모델 파일 크기
# ------------------------------------------------------------
def save_and_size(model, model_type, calibrator, out_dir, tag):
    # 라이브러리 save_model()은 C++ 쪽에서 파일을 열어 한글 경로에 못 씀 -> 직렬화 결과를 받아 Python으로 저장.
    # (save_model()로 저장한 파일과 내용이 같음)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    if model_type == "XGBoost":
        booster = model.get_booster()
        for ext in ("json", "ubj"):  # json: 텍스트, ubj: Universal Binary JSON(float32 그대로 저장)
            path = out_dir / f"{tag}.{ext}"
            path.write_bytes(bytes(booster.save_raw(raw_format=ext)))
            files[ext] = path
    else:
        path = out_dir / f"{tag}.txt"  # LightGBM 기본 텍스트 포맷(값은 double을 10진 문자열로 저장)
        path.write_bytes(model.booster_.model_to_string().encode("utf-8"))
        files["txt"] = path
    path = out_dir / f"{tag}_sklearn.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    files["sklearn_pickle"] = path
    if calibrator is not None:
        path = out_dir / f"{tag}_calibrated.pkl"  # 최종 추론 단위(model + isotonic 보정기)
        with open(path, "wb") as f:
            pickle.dump(calibrator, f)
        files["calibrated_pickle"] = path
    return {k: {"bytes": p.stat().st_size, "MB": round(p.stat().st_size / MIB, 4)} for k, p in files.items()}


# ------------------------------------------------------------
# 시간 측정
# ------------------------------------------------------------
def time_per_row(fn, rows):
    """rows의 각 원소로 fn을 1회씩 호출해 1건당 시간(ms)의 평균/표준편차를 반환."""
    for r in rows[:N_WARMUP]:
        fn(r)
    times = np.empty(len(rows))
    for i, r in enumerate(rows):
        t0 = time.perf_counter_ns()
        fn(r)
        times[i] = (time.perf_counter_ns() - t0) / 1e6
    return {"mean_ms": round(float(times.mean()), 4), "std_ms": round(float(times.std(ddof=1)), 4),
            "median_ms": round(float(np.median(times)), 4), "n": len(rows)}


def measure_inference_time(model, model_type, calibrator, X_test):
    rows = [X_test.iloc[[i]] for i in range(N_TIMING_REPEATS)]  # 1행 DataFrame(dtype은 학습 때와 동일)
    result = {}
    for n_threads in (None, 1):  # None: 라이브러리 기본(전체 코어), 1: 단일 스레드
        label = "threads=default" if n_threads is None else "threads=1"
        model.set_params(n_jobs=n_threads)
        entry = {"model.predict_proba": time_per_row(lambda r: model.predict_proba(r), rows)}
        if model_type == "XGBoost":
            booster = model.get_booster()
            entry["booster.inplace_predict"] = time_per_row(lambda r: booster.inplace_predict(r), rows)
        if calibrator is not None:
            entry["calibrated.predict_proba"] = time_per_row(lambda r: calibrator.predict_proba(r), rows)
        result[label] = entry
    model.set_params(n_jobs=None)
    return result


def measure_preprocess_time(raw_rows, cols, train_dtypes):
    """원시 1행(파생 feature 없음, 범주형은 문자열) -> 모델 입력 1행까지의 시간.
    단계: add_candidate_features() -> feature 컬럼 선택 -> 학습 때와 같은 category 코드북 적용."""
    cat_cols = [c for c in cols if isinstance(train_dtypes[c], pd.CategoricalDtype)]

    def _prep(row):
        x = add_candidate_features(row)[cols]
        for c in cat_cols:
            x[c] = x[c].astype(train_dtypes[c])
        return x

    return time_per_row(_prep, raw_rows), _prep


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------
def measure():
    env = environment_info()
    df_raw = get_or_generate_final_dataset()
    df = prepare_categorical(df_raw)
    train_set, test_set = split_data(df)
    train_sub, val_calib = train_test_split(
        train_set, test_size=CALIBRATION_VAL_SIZE, random_state=42, stratify=train_set["is_phishing"]
    )

    winner, winner_params = load_tuned_hyperparams()
    with open(_PROJECT_ROOT / "optuna_results" / "best_params.json", "r", encoding="utf-8") as f:
        all_params = json.load(f)

    # 최종 모델(winner) + 참고용으로 Optuna 탐색에서 비교한 다른 모델도 같은 기준으로 측정
    model_types = [winner] + [m for m in ("XGBoost", "LightGBM") if m != winner]

    # 전처리 시간용 원시 row: 파생 feature 4개를 지우고 범주형을 문자열(object)로 되돌림(생성기 출력 형태)
    raw_test = df_raw.loc[test_set.index].drop(columns=DERIVED_FINAL_FEATURES)
    for c in raw_test.columns:
        if isinstance(raw_test[c].dtype, pd.CategoricalDtype):
            raw_test[c] = raw_test[c].astype(object)
    raw_rows = [raw_test.iloc[[i]] for i in range(N_TIMING_REPEATS)]

    results = {"environment": env, "models": []}
    for model_type in model_types:
        hyperparams = all_params[model_type]["tuned_hyperparams"]
        for scenario_name, tracks in TRACK_SCENARIOS.items():
            cols = get_feature_columns(track_filter=tracks)
            tag = f"{model_type}_track{scenario_name}"
            print(f"[{tag}] 학습/측정 중...")

            model = train_model(train_sub, feature_cols=cols, hyperparams=hyperparams, model_type=model_type)
            calibrator = fit_calibrator(model, val_calib, feature_cols=cols, method="isotonic")
            pr_auc = evaluate(model, test_set, feature_cols=cols, threshold_df=train_sub,
                              pr_auc_method="average_precision", calibrator=calibrator)["PR-AUC"]

            X_test = test_set[cols]
            stats, comparisons = tree_structure_and_paths(model, model_type, X_test)
            n_trees = stats["n_trees"]
            ops = inference_ops(comparisons, n_trees)
            iso = isotonic_info(calibrator)
            # calibrated 추론 = member마다 (base model 1회 + isotonic 1회) 후 평균(member 수 - 1 덧셈 + 1 나눗셈)
            n_mem = iso["n_calibrated_members"]
            calibrated_ops_mean = n_mem * float(ops.mean()) + sum(iso["isotonic_ops_per_member"]) + (n_mem if n_mem > 1 else 0)

            # 입력 길이 의존성: 입력 벡터 길이는 schema로 고정. 사건 내 접촉이 많은(반복 접촉/순차 사칭) 사건과
            # 아닌 사건에서 실제 연산 수가 달라지는지도 같이 기록(통화 기록 길이의 대리 지표).
            groups = {}
            for gcol in ("has_repeat_contact", "is_sequential_callers"):
                for v in (0, 1):
                    mask = (test_set[gcol] == v).to_numpy()
                    if mask.any():
                        groups[f"{gcol}={v}"] = {"n": int(mask.sum()), "ops_mean": round(float(ops[mask].mean()), 2)}

            # 1행 예측 = 배치 예측 확인(1건씩 넣어도 category 코드북이 어긋나지 않는지)
            batch_p = calibrator.predict_proba(X_test.iloc[:20])[:, 1]
            single_p = np.array([calibrator.predict_proba(X_test.iloc[[i]])[0, 1] for i in range(20)])
            prep_time, prep_fn = measure_preprocess_time(raw_rows, cols, X_test.dtypes)
            prep_p = np.array([calibrator.predict_proba(prep_fn(raw_rows[i]))[0, 1] for i in range(20)])

            entry = {
                "model": tag,
                "is_final_model": model_type == winner,
                "model_type": model_type,
                "track": scenario_name,
                "hyperparams": hyperparams,
                "test_PR-AUC(average_precision, calibrated)": round(float(pr_auc), 6),
                "input": {
                    "n_features": len(cols),
                    "features": cols,
                    "categorical_features": [c for c in cols if isinstance(X_test[c].dtype, pd.CategoricalDtype)],
                    "dtypes": {c: str(X_test[c].dtype) for c in cols},
                },
                "tree_structure": stats,
                "params": {
                    "total_nodes": stats["n_total_nodes"],
                    "alt_2xinternal_plus_leaves": 2 * stats["n_internal_nodes"] + stats["n_leaves"],
                },
                "ops_per_inference_model_only": {
                    "comparisons_mean": round(float(comparisons.mean()), 2),
                    "comparisons_min": int(comparisons.min()),
                    "comparisons_max": int(comparisons.max()),
                    "comparisons_upper_bound": stats["comparisons_upper_bound"],
                    "ops_mean": round(float(ops.mean()), 2),
                    "ops_min": int(ops.min()),
                    "ops_max": int(ops.max()),
                    "ops_upper_bound": inference_ops(stats["comparisons_upper_bound"], n_trees),
                    "n_test_rows": int(len(ops)),
                },
                "calibrator": iso,
                "ops_per_inference_calibrated_mean": round(calibrated_ops_mean, 2),
                "ops_by_contact_history": groups,
                "model_files": save_and_size(model, model_type, calibrator, _OUT_DIR / "models", tag),
                "single_row_matches_batch": bool(np.allclose(batch_p, single_p) and np.allclose(batch_p, prep_p)),
                "inference_time_per_row": measure_inference_time(model, model_type, calibrator, X_test),
                "preprocess_time_per_row(raw row -> model input)": prep_time,
            }
            results["models"].append(entry)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(_OUT_DIR / "model_cost.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    table = render_table(results)
    with open(_OUT_DIR / "model_cost_table.md", "w", encoding="utf-8") as f:
        f.write(table)
    return results, table


def _fmt_sci(x):
    mant, exp = f"{x:.2e}".split("e")
    return f"{mant}×10^{int(exp)}"


def render_table(results):
    env = results["environment"]
    lines = [
        f"측정 환경: {env['cpu']} (logical {env['logical_cpus']}), {env['os']}, Python {env['python']}, "
        f"xgboost {env['xgboost']}, lightgbm {env['lightgbm']}, scikit-learn {env['scikit-learn']}, "
        f"pandas {env['pandas']}, numpy {env['numpy']}",
        "",
        "| 모델 | 입력 | 입력 크기 | 파라미터 | 모델 크기 (MB, 정밀도) | 1회 추론 연산량 | 입력 길이에 따라 증가 | 추론 시간 (선택) | 측정 기준·환경 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in results["models"]:
        s, o, files = m["tree_structure"], m["ops_per_inference_model_only"], m["model_files"]
        if m["model_type"] == "XGBoost":
            size = (f"{files['ubj']['MB']:.3f} (UBJ, FP32) / {files['json']['MB']:.3f} (JSON 텍스트) / "
                    f"{files['calibrated_pickle']['MB']:.3f} (보정기 포함 pickle); INT8 버전 없음")
        else:
            size = (f"{files['txt']['MB']:.3f} (텍스트, FP64 값) / "
                    f"{files['calibrated_pickle']['MB']:.3f} (보정기 포함 pickle); INT8 버전 없음")
        t = m["inference_time_per_row"]["threads=1"]
        time_key = "calibrated.predict_proba"
        tm = t[time_key]
        name = f"{m['model_type']} (Track {m['track']}){' ★최종' if m['is_final_model'] else ' (참고)'}"
        lines.append(
            f"| {name} "
            f"| 사건 1건 요약 feature 벡터 "
            f"| {m['input']['n_features']} (고정) "
            f"| {s['n_total_nodes'] / 1e6:.4f}M 노드 ({s['n_trees']} trees, 내부 {s['n_internal_nodes']} + 리프 {s['n_leaves']}) "
            f"| {size} "
            f"| 평균 {_fmt_sci(o['ops_mean'])} ops (범위 {o['ops_min']}~{o['ops_max']}, 상한 {o['ops_upper_bound']}); "
            f"보정기 포함 평균 {_fmt_sci(m['ops_per_inference_calibrated_mean'])} "
            f"| 아니오 "
            f"| {tm['mean_ms']:.3f} ± {tm['std_ms']:.3f} ms (보정기 포함, 1 thread, n={tm['n']}) "
            f"| 트리 노드 수 기준 / 근사 ops = 지나간 내부 노드 수 + (트리 수+1) 덧셈 + sigmoid 3 |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    res, table = measure()
    print(json.dumps(res["environment"], ensure_ascii=False, indent=2))
    print(table)
    print(f"상세 결과: {_OUT_DIR / 'model_cost.json'}")
