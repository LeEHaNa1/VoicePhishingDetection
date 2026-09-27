# XGBoost/LightGBM 하이퍼파라미터 Optuna 탐색 + 두 모델 비교.
# 목적: 최종 모델을 XGBoost/LightGBM 각각 따로 탐색한 뒤, val set PR-AUC로 비교해서
#      더 나은 쪽을 최종 모델 종류로 선택하기 위함(main.py::run_tuning_mode() 참고).
import optuna

# 하이퍼파라미터 중요도 분석
import optuna.importance
import optuna.visualization as vis
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from Simulator.Detection.train_eval import lift_at_top_k


def _lgb_pr_auc_feval(y_true, y_pred):
    """LightGBM 조기종료를 PR-AUC 기준으로 하기 위한 커스텀 평가 함수.
    LGBMClassifier.fit()의 eval_metric에 문자열이 아니라 이 콜러블을 직접 넘겨야
    실제로 이 지표 기준으로 조기종료가 동작함(문자열 'custom' 등은 조용히 무시되고
    기본 metric으로 대체됨 -> 실제 재현해서 확인한 문제)."""
    return "pr_auc", average_precision_score(y_true, y_pred), True  # True = 높을수록 좋음


def create_objective(X_train, y_train, X_val, y_val, model_type="XGBoost"):
    """
    Optuna 목적함수를 생성하는 클로저(Closure) 함수.
    미리 분할된 train_sub/val_sub를 입력받아 데이터 누수를 방지함(호출부에서 test_set은
    절대 넘기면 안 됨 -> main.py::run_tuning_mode() 참고).
    model_type: "XGBoost" 또는 "LightGBM".
    """
    # 양성 클래스(피싱) 가중치 기준점 계산 (train_model()의 동적 scale_pos_weight 계산과 동일한 근거)
    base_scale = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        if model_type == "XGBoost":
            params = {
                'eval_metric': 'aucpr',  # XGBoost가 PR-AUC를 기본 메트릭으로 지원 -> 조기종료도 이 기준으로 동작
                'early_stopping_rounds': 50,
                'n_estimators': 1000,    # 충분히 크게 주고 조기종료로 실제 라운드 수를 제어
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', base_scale * 0.5, base_scale * 1.5),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_child_weight': trial.suggest_float('min_child_weight', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'tree_method': 'hist',
                'enable_categorical': True,  # number_type(category dtype) 컬럼 지원에 필수
                'random_state': 42,
                'n_jobs': -1,
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict_proba(X_val)[:, 1]

        elif model_type == "LightGBM":
            params = {
                'objective': 'binary',
                'metric': 'None',        # 아래 커스텀 feval(pr_auc)만으로 조기종료 판단(기본 logloss 안 섞이게)
                'n_estimators': 1000,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', base_scale * 0.5, base_scale * 1.5),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1,
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=_lgb_pr_auc_feval,  # 콜러블 직접 전달(문자열 X) -> 실제로 PR-AUC 기준 조기종료됨
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            preds = model.predict_proba(X_val)[:, 1]

        else:
            raise ValueError(f"알 수 없는 model_type: {model_type!r}. 'XGBoost' 또는 'LightGBM'이어야 함.")

        # Optuna가 실제로 최적화하는 값: 조기종료 기준과 별개로, sklearn의
        # average_precision_score로 val set에서 직접 재계산 -> 항상 명확한 기준으로 비교.
        return average_precision_score(y_val, preds)

    return objective


def run_optuna_tuning(X_train, y_train, X_val, y_val, model_type="XGBoost", n_trials=50, random_state=42):
    """model_type 1개에 대해 Optuna 탐색을 돌려서 최종 하이퍼파라미터 dict를 반환.
    train_model()에 그대로 넘길 수 있는 형태(n_estimators까지 확정된 고정값)로 만들어서 반환함
    -> 최종 재학습 때는 조기종료 없이 이 n_estimators로 고정 학습.
    """
    objective = create_objective(X_train, y_train, X_val, y_val, model_type=model_type)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction='maximize', sampler=sampler)  # PR-AUC는 클수록 좋음
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_params)
    base_scale = (y_train == 0).sum() / (y_train == 1).sum()

    # best_params로 한 번 더 학습해서 조기종료가 실제로 멈춘 지점(best_iteration)을 확정.
    # 최종 재학습(train_model())은 val set 없이 돌기 때문에, n_estimators를 여기서
    # 고정값으로 못박아 둬야 함.
    if model_type == "XGBoost":
        probe_params = {
            'eval_metric': 'aucpr', 'early_stopping_rounds': 50, 'n_estimators': 1000,
            'tree_method': 'hist', 'enable_categorical': True,
            'random_state': random_state, 'n_jobs': -1,
            **best_params,
        }
        probe_model = xgb.XGBClassifier(**probe_params)
        probe_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        best_n_estimators = int(probe_model.best_iteration) + 1  # best_iteration은 0-indexed

        # val PR-AUC(=study.best_value)와는 별개로 top-5 lift 지표 확인
        probe_proba_val = probe_model.predict_proba(X_val)[:, 1]
        val_lift_top1 = lift_at_top_k(y_val, probe_proba_val, k=0.01)

        tuned_hyperparams = {
            'n_estimators': best_n_estimators,
            'learning_rate': best_params['learning_rate'],
            'scale_pos_weight': best_params['scale_pos_weight'],
            'max_depth': best_params['max_depth'],
            'min_child_weight': best_params['min_child_weight'],
            'subsample': best_params['subsample'],
            'colsample_bytree': best_params['colsample_bytree'],
            'reg_alpha': best_params['reg_alpha'],
            'reg_lambda': best_params['reg_lambda'],
        }
    else:  # LightGBM
        probe_params = {
            'objective': 'binary', 'metric': 'None', 'n_estimators': 1000,
            'random_state': random_state, 'verbose': -1, 'n_jobs': -1,
            **best_params,
        }
        probe_model = lgb.LGBMClassifier(**probe_params)
        probe_model.fit(
            X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=_lgb_pr_auc_feval,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)],
        )
        best_n_estimators = int(probe_model.best_iteration_)

        probe_proba_val = probe_model.predict_proba(X_val)[:, 1]
        val_lift_top1 = lift_at_top_k(y_val, probe_proba_val, k=0.01)

        tuned_hyperparams = {
            'n_estimators': best_n_estimators,
            'learning_rate': best_params['learning_rate'],
            'scale_pos_weight': best_params['scale_pos_weight'],
            'max_depth': best_params['max_depth'],
            'num_leaves': best_params['num_leaves'],
            'min_child_samples': best_params['min_child_samples'],
            'subsample': best_params['subsample'],
            'colsample_bytree': best_params['colsample_bytree'],
            'reg_alpha': best_params['reg_alpha'],
            'reg_lambda': best_params['reg_lambda'],
        }

    # fANOVA를 통해, 하이퍼파리미터 중요도 확인. 
    # trial 수가 너무 적거나 특정 하이퍼파라미터가 한 값으로 고정되면 계산 실패할 수 있기에 방어적으로 처리
    try:
        evaluator = optuna.importance.FanovaImportanceEvaluator(seed=42)
        raw_importance = optuna.importance.get_param_importances(study, evaluator=evaluator)
        hyperparameter_importance_ranked = [
            {
                "rank": i + 1,
                "param": name,
                "importance": round(float(value), 4),
                "importance_pct": f"{value * 100:.1f}%",  # 전체 중요도 합 대비 이 하이퍼파라미터가 차지하는 비율
            }
            for i, (name, value) in enumerate(raw_importance.items())
        ]
    except Exception as e:
        hyperparameter_importance_ranked = [{"error": f"중요도 계산 실패: {e}"}]


    return {
        "model_type": model_type,
        "tuned_hyperparams": tuned_hyperparams,
        "best_value(val PR-AUC)": study.best_value,
        "val_lift_top1%": val_lift_top1,
        "hyperparameter_importance_ranked(val PR-AUC 기준, fANOVA, 순위·비율순)": hyperparameter_importance_ranked,
        "n_trials": n_trials,
        "base_scale_pos_weight(참고용, 실제 데이터 neg/pos 비율)": base_scale,
        "study" : study, # 시각화 및 등고선 분석을 위함
    }


def run_optuna_tuning_and_compare(X_train, y_train, X_val, y_val, n_trials=50, random_state=42):
    """XGBoost/LightGBM 둘 다 탐색해서 val PR-AUC가 더 높은 쪽을 승자로 선택.
    반환값에 "winner"(model_type 문자열)와 두 모델 각각의 전체 결과가 다 들어있어서,
    진 쪽 결과도 보고서용으로 남길 수 있음."""
    xgb_result = run_optuna_tuning(X_train, y_train, X_val, y_val, model_type="XGBoost",
                                    n_trials=n_trials, random_state=random_state)
    lgb_result = run_optuna_tuning(X_train, y_train, X_val, y_val, model_type="LightGBM",
                                    n_trials=n_trials, random_state=random_state)

    winner = "XGBoost" if xgb_result["best_value(val PR-AUC)"] >= lgb_result["best_value(val PR-AUC)"] else "LightGBM"

    xgb_pr_auc = xgb_result["best_value(val PR-AUC)"]
    lgb_pr_auc = lgb_result["best_value(val PR-AUC)"]
    xgb_lift = xgb_result["val_lift_top1%"]
    lgb_lift = lgb_result["val_lift_top1%"]

    comparison = {
        "PR-AUC": {
            "XGBoost": xgb_pr_auc,
            "LightGBM": lgb_pr_auc,
            "diff(LightGBM - XGBoost)": round(lgb_pr_auc - xgb_pr_auc, 4),
            "higher": "XGBoost" if xgb_pr_auc >= lgb_pr_auc else "LightGBM",
        },
        "Lift@Top1%": {
            "XGBoost": xgb_lift,
            "LightGBM": lgb_lift,
            "diff(LightGBM - XGBoost)": round(lgb_lift - xgb_lift, 4),
            "higher": "XGBoost" if xgb_lift >= lgb_lift else "LightGBM",
        },
        "hyperparameter_importance_ranked": {
            # 두 모델은 하이퍼파라미터 종류 자체가 다르므로(예: XGBoost의 min_child_weight vs
            # LightGBM의 num_leaves) 항목별 diff는 의미가 없음 -> 각자의 순위·비율 리스트를 나란히만 배치.
            "XGBoost": xgb_result["hyperparameter_importance_ranked(val PR-AUC 기준, fANOVA, 순위·비율순)"],
            "LightGBM": lgb_result["hyperparameter_importance_ranked(val PR-AUC 기준, fANOVA, 순위·비율순)"],
        },
    }

    
    # 1. 튜닝 결과에서 승자 모델(또는 분석하고 싶은 모델)의 study 객체 추출
    # 일단 주석 -> 계속 나오니깐 확실해지면 풀기(보고서에 작성하기 위함)
    models_to_visualize = [
        ("XGBoost", xgb_result),
        ("LightGBM", lgb_result),
    ]

    for model_name, res in models_to_visualize:
        try:
            target_study = res.get("study")
            if target_study is None:
                continue

            # 난수고정
            evaluator = optuna.importance.FanovaImportanceEvaluator(seed=42)

            # 1. fANOVA 기반 하이퍼파라미터 중요도 막대그래프
            fig_imp = vis.plot_param_importances(target_study, evaluator=evaluator)
            fig_imp.update_layout(title=f"[{model_name}] Hyperparameter Importances (fANOVA)")
            fig_imp.show()

            # 2. 파라미터 값 변화에 따른 PR-AUC 분포 (Slice Plot)
            fig_slice = vis.plot_slice(target_study)
            fig_slice.update_layout(title=f"[{model_name}] Slice Plot (PR-AUC vs Params)")
            fig_slice.show()

            # 3. 모델별 핵심 규제 파라미터 등고선 (Contour Plot)
            # XGBoost: max_depth, LightGBM: num_leaves (각 모델의 핵심 복잡도 변수 반영)
            contour_params = (
                ["colsample_bytree", "max_depth"]
                if model_name == "XGBoost"
                else ["colsample_bytree", "min_child_samples"]
            )
            fig_contour = vis.plot_contour(target_study, params=contour_params)
            fig_contour.update_layout(title=f"[{model_name}] Contour Plot ({contour_params[0]} vs {contour_params[1]})")
            fig_contour.show()

        except Exception as e:
            print(f"[{model_name}] 시각화 출력 중 오류 발생 (무시하고 진행): {e}")

    # 시각화 완료 후 JSON 저장을 위해 Study 객체 제거
    xgb_result.pop("study", None)
    lgb_result.pop("study", None)


    return {
        # val PR-AUC 기준 자동 승자. run_final()의 load_tuned_hyperparams()가 이 필드를 읽음.
        # Lift 등 다른 지표까지 보고 바꾸려면 best_params.json의 "winner"만 "XGBoost"/"LightGBM"으로
        # 고치면 되고, 해당 모델의 tuned_hyperparams가 이어서 반영됨.
        "winner": winner,
        "comparison": comparison,
        "XGBoost": xgb_result,
        "LightGBM": lgb_result,
    }
