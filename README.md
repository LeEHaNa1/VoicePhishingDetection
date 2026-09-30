# VoicePhishingDetection

대화 내용이 아닌 메타데이터와 구조적 신호만을 사용한 보이스피싱 탐지 모델 학습을 위해
합성(시뮬레이션) 데이터 생성기 + 탐지 pipeline 구현.
실측 통신 로그 없이도 재현 가능한 학습 데이터를 만들고, 그 위에서 feature 선정, 모델 학습 및 평가까지 한 번에 검증한다.

---

## 배경 / 왜 이 방식인가

보이스피싱 탐지 모델을 학습하려면 실제 통화/문자 로그가 필요하지만, 이런 데이터는 통신사·수사기관 등 극히 제한된 주체만 접근할 수 있고 개인정보 문제로 외부 공유가 사실상 불가능하다. 그래서 이 프로젝트는 실측 자료 대신, 공개된 판결문·통계·전문가 서술(마스터표) 등 확보 가능한 근거를 바탕으로 **정상/피싱 이벤트를 확률적으로 합성하는 시뮬레이터**를 만들고, 그 위에서 탐지 모델을 학습·검증하는 방식을 택했다.

이때 핵심 위험은 "생성 규칙을 그대로 암기하는 모델"이 나오는 것이다. 데이터를 만들 때 쓴 파라미터(확률값, 지인:기관 비중 등)를 학습 코드가 알고 있으면, 그 정보가 feature처럼 새어 들어가 실제로는 아무 의미 없는 규칙을 "학습"해버릴 수 있다. 이를 구조적으로 차단하기 위해 **Generation(데이터 생성)과 Detection(학습/평가)을 완전히 분리**했다 — `Detection/` 안의 어떤 코드도 `Generation/config.py`를 import하지 않는다.

또한 실제 탐지 시스템은 소속 기관에 따라 접근 가능한 데이터 범위가 다르다. 이 프로젝트는 이를 3개 Track으로 구분해서 모델링한다:

| Track                | 의미                                                                | 이 프로젝트에서의 상태                   |
| -------------------- | ------------------------------------------------------------------- | ---------------------------------------- |
| A (통신사)           | 발신번호 변작 여부, 번호 클러스터링 등 통신사 실측자료 기반 feature | 실측자료 접근 불가 → 전부 결측(NaN) 처리 |
| B (단말)             | 이용자 단말/통화 이력만으로 확인 가능한 feature                     | 시뮬레이터로 생성 가능                   |
| C (단말+구조적 신호) | 문자 내용의 구조적 패턴(URL, 번호 불일치 등)까지 포함               | 시뮬레이터로 생성 가능                   |

즉 이 프로젝트가 실제로 검증하는 것은 "통신사 협조 없이, 일반 단말/앱 단에서 확보 가능한 정보만으로 보이스피싱을 얼마나 탐지할 수 있는가"이다.

---

## 프로젝트 구조

```
(프로젝트 루트)
├── README.md
├── Structure.md                  # 폴더/파일별 역할, 함수 목록, 최신 결정 사항 기록
├── DataSet/                      # 최종 dataset 저장 (-m generate/-m final 산출물)
└── Simulator/
    ├── main.py                   # 전체 파이프라인 진입점 (CLI)
    ├── schema.py                 # Track/ValueType/ColumnSchema 정의
    ├── schema_columns.py         # 실제 컬럼 목록 (SCHEMA)
    ├── schema_utils.py           # SCHEMA 조회 헬퍼 함수
    ├── Generation/                       # 합성 데이터 생성 (Detection에서 import 금지)
    │   ├── config.py                     # 정상값/피싱값 마스터 설정
    │   ├── generator_utils.py            # 번호 합성·sophistication 조회 공용 유틸
    │   ├── normal_generator.py           # 정상 이벤트 1건 생성
    │   ├── phishing_generator.py         # 피싱 이벤트 1건 생성
    │   └── dataset_builder.py            # Algorithm 1 전체 구현 (n건 생성 + 검증)
    └── Detection/                        # 학습/평가/검증 (config.py 미참조)
        ├── derived_features.py           # 파생 feature 13개 후보 계산
        ├── feature_ablation.py           # 파생 feature 선정 + 안정성 재검증 파이프라인
        ├── optuna_apply.py               # XGBoost/LightGBM 하이퍼파라미터 탐색
        ├── train_eval.py                 # 분할 → 학습 → 평가(+ confidence calibration)
        ├── eval_confidence.py            # confidence calibration 정량 평가(ECE/Brier/신뢰도 다이어그램)
        ├── plot_curve.py                 # Track B/C × Raw/Calibrated Learning Curve·Cumulative Lift Curve
        ├── sensitivity_analysis.py       # 파라미터별 민감도/스트레스 분석
        ├── borderline_recheck.py         # -m final 경계선 재확인 (약한 파생 제거·is_global·permutation)
        └── url_reliability_recheck.py    # is_reliable_url stress의 gain 해석을 permutation으로 재검증
```

각 파일의 상세 역할과 선언된 함수 목록은 [Structure.md](Structure.md)에 정리돼 있다.

---

## 파이프라인 단계

### Generation — 합성 데이터 생성

1. `config.py`에 정의된 정상/피싱 확률값(sophistication별 low/mid/high)을 바탕으로,
2. 사건마다 클래스 불균형 비율(`CLASS_IMBALANCE`, 기본 1%)에 따라 정상/피싱을 확률적으로 결정하고,
3. 정상이면 `normal_generator.py`, 피싱이면 `phishing_generator.py`가 해당 사건 1건을 생성하며,
4. `dataset_builder.py::build_dataset()`이 이 과정을 n번 반복해 표 형태 dataset을 만든다.

### Detection — 학습 · 평가 · 검증

1. **feature 선정**: `derived_features.py`가 만드는 13개 파생 feature 후보를 `feature_ablation.py`가 baseline 비교 → 다중공선성 제거 → SHAP/Permutation Importance 기반 가지치기 → 최종 재학습 순으로 검증한다. 확정된 조합은 다시 K-Fold로 안정성을 재검증한다.
2. **하이퍼파라미터 탐색**: `optuna_apply.py`가 XGBoost/LightGBM 각각 Optuna로 탐색하고, validation PR-AUC가 더 높은 모델을 채택한다.
3. **민감도/스트레스 분석**: `sensitivity_analysis.py`가 sophistication(low/mid/high) 및 절대값 가정을 하나씩 흔들어가며 성능이 얼마나 민감한지 확인한다.
4. **최종 학습/평가**: `train_eval.py`가 확정된 feature·하이퍼파라미터로 train/test 분할 학습을 수행하고, threshold는 train에서만 결정해 test에 고정 적용한다(낙관 편향 방지). 기본 설정('CALIBRATION_APPLY="True")에서는 이 단계에서 클래스 가중치로 왜곡된 예측확률을 `fit_calibrator()`(isotonic)로 함께 보정.

---

## 사용법 (Quick Start)

**1. 필요 패키지 설치**

```
pip install pandas numpy scikit-learn xgboost lightgbm shap optuna pyarrow
```

**2. 최종 dataset 생성** (`DataSet/`에 저장. 실행할 때마다 항상 새로 생성 — 재현성 확인용으로 강제 재생성하고 싶을 때 사용)

```
python -X utf8 -m Simulator.main -m generate
```

**3. 최종 모델 학습 + 평가** (저장된 dataset이 있고 파라미터가 그대로면 재사용, 없거나 다르면 자동 생성 후 학습)

```
python -X utf8 -m Simulator.main -m final
```

**4. (선택) 그 외 모드**

| 모드                                  | 용도                                                                                                                                                          |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-m sen` / `-m each_sen` / `-m ratio` | sophistication·지인:기관 비중 민감도분석                                                                                                                      |
| `-m stress`                           | 절대값 가정 자체가 틀렸을 때의 스트레스테스트                                                                                                                 |
| `-m ablation`                         | 파생 feature 후보 선정 파이프라인 실행                                                                                                                        |
| `-m ablation_stability`               | SCHEMA 미등록 잔여 후보(현재 9개)를 baseline에 더했을 때 성능 차이가 노이즈인지 K-Fold 재검증. 이미 SCHEMA에 넣은 확정 4개는 비교 대상에 넣지 않음(컬럼 중복) |
| `-m tune`                             | XGBoost/LightGBM Optuna 하이퍼파라미터 탐색                                                                                                                   |

| `-m calib_compare` | 원본(α)/데이터 축소(β)/보정 적용(γ) 3버전 결과 도출 — `-m final`의 최종 출력(보정 적용)과 별개로 보정 자체의 효과만 분리해서 확인 |

`-m final`의 최종 출력은 기본적으로 보정된 확률(γ) 기준이다. 보정 자체의 세부 지표(ECE·Brier score·신뢰도 다이어그램)까지 보려면 `python -X utf8 -m Simulator.Detection.eval_confidence`를 쓴다.

각 모드의 상세 파라미터는 `Simulator/main.py` 상단 상수와 하단 CLI 사용법 주석을 참고.

---

## 결과

### Track별 최종 성능 (`-m final`, 보정 적용 후 γ 기준)

N=100,000, train 70,000 / test 30,000 (피싱 298건). 모델은 Optuna 탐색 후 **XGBoost**(val PR-AUC 0.9941 > LightGBM 0.9935)와 `optuna_results/best_params.json`의 `tuned_hyperparams`를 사용. PR-AUC는 `average_precision_score`. threshold는 train에서만 고르고 test에 고정 적용. `-m final`은 기본 설정(`CALIBRATION_APPLY=True`)에서 confidence calibration이 적용된 확률(γ)로 평가하므로, 아래는 그 결과다. 보정 전(α/β)과의 비교는 아래 "confidence calibration의 적용 여부" 참고.

| Track               | precision | recall | f1     | PR-AUC | Lift@Top5% | Lift@Top10% | FN (미탐) |
| ------------------- | --------- | ------ | ------ | ------ | ---------- | ----------- | --------- |
| B (단말만)          | 0.9856    | 0.9161 | 0.9496 | 0.9695 | 19.66      | 9.87        | 25/298    |
| C (단말+구조적신호) | 1.0000    | 0.9698 | 0.9847 | 0.9857 | 19.80      | 9.90        | 9/298     |

**메인 결과는 Track C.** B 대비 F1 +0.035, PR-AUC +0.016, 미탐 25건→9건.

혼동행렬(test): B는 FP 4 / FN 25, C는 FP 0 / FN 9.

#### Confidence Calibration 비교(`-m calib_compare`)

`-m final`은 기본 설정(`CALIBRATION_APPLY=True`)에서 이미 보정(isotonic)된 확률로 평가한다. 위 표(보정, γ)와 비교해 보정 자체의 효과만 분리해서 보려면 학습 데이터를 동일하게 맞춘 γ(보정 후)-β(보정 전) 비교가 필요하다(`-m calib_compare`).

클래스 가중치로 학습하면 예측확률이 실제 발생률(0.99%)보다 부풀려지는 문제(overconfidence)가 있어, isotonic regression으로 후처리 보정을 적용했다. 보정은 판별 성능(PR-AUC)을 거의 그대로 두면서(변화 0.005 이내) 확률 자체의 정직성만 크게 개선한다.

| Track | 평균 예측확률(실제 0.99%) 보정 전→후 | ECE(quantile) 보정 전→후 | Brier score 보정 전→후 | PR-AUC 변화(보정 자체 효과, γ−β) |
| ----- | ------------------------------------ | ------------------------ | ---------------------- | -------------------------------- |
| B     | 3.61% → 0.97%                        | 0.0262 → 0.0002          | 0.0068 → 0.0008        | −0.0048                          |
| C     | 2.26% → 0.99%                        | 0.0126 → 0.0001          | 0.0024 → 0.0004        | −0.0042                          |

**해석 시 주의**: 위 ECE·Brier는 학습 데이터가 동일하고 보정 유무만 다른 γ(보정 후)-β(보정 전) 비교다(`calibration_results/confidence_metrics.json`, `eval_confidence.py` 산출). Track C 보정 후 ECE는 확률이 동점으로 뭉쳐 분위수 bin이 1개까지 줄어들어 사실상 "평균 예측확률 − 실제 비율"과 같은 값이라, 구간별 정밀 보정이 아니라 평균 수준의 정렬로 해석해야 한다(구간별 확인은 `tail_reliability_diagram.png` 참고).

### 피처 중요도 (보고 기준 = permutation, PR-AUC)

`-m final`이 출력하는 `feature_importance`는 XGBoost **gain**(합=1)이라 트리 분할에 자주 쓰인 피처가 과대 평가된다. 최종 해석은 같은 모델·같은 test_set에서 `scoring="average_precision"` permutation importance(반복 10회)를 기준으로 한다.

Track B 상위: `number_type`(0.345) > `sms_to_call_gap`(0.114) > `repeat_gap`(0.061).  
Track C 상위: `number_type`(0.056) > `is_reliable_url`(0.050) > `sms_to_call_gap`(0.036).

gain만 보면 Track B에서 `sms_to_call`이 25.3%로 1위처럼 보이지만, permutation은 0.0009로 거의 0이다. Track C에서 이 값이 0.8%로 줄어든 것을 “C에서만 쓸모없어졌다”고 읽으면 안 된다 — **B/C 모두 랭킹 기여가 작다.**

재학습으로 Track B에서 12개의 변수 Track C에서 19개의 변수를 적용하여 Permutation Importance를 통해

- Track B 상위: `number_type`(0.3962) > `sms_to_call_gap`(0.1319) > `hour_bucket`(0.0547).
- Track C 상위: `is_reliable_url`(0.0823) > `number_type`(0.0545) > `sms_to_call_gap`(0.0397).

`is_global`/`sms_to_call`을 뺀 뒤에도 상위권 구성 자체는 크게 안 바뀌었지만, Track C에서는 `is_reliable_url`(문자 URL 신뢰도)이 1위로 올라서며 `number_type`(발신번호 대역) 하나에 대한 의존도가 줄고 판단 근거가 여러 변수에 분산됐다.

### 경계선 재확인 (`borderline_recheck.py`)

같은 dataset·같은 split·같은 하이퍼파라미터로 세 가지를 재확인했다. SCHEMA/최종 학습 feature는 **바꾸지 않고 유지**한다(해석만 확정).

1. **약한 파생 2개** (`is_sms_initiated_unreg`, `repeat_pressure_intensity`)를 빼고 재학습: Track B는 F1/PR-AUC가 사실상 동일(F1 +0.002, PR-AUC −0.0007). Track C는 PR-AUC 동일(−0.0001), F1 −0.005·recall −0.010. 이전 K=10 `weak_pair` 결론과 같다. 랭킹 성능은 유지되므로 4개 파생을 SCHEMA에 남겨 둔다.
2. **`is_global`**: 10만 건 중 2,114건(2.11%)이라 희귀 피처가 아니다. 그러나 `number_type == "00X(국제)"`와 **100% 일치**하고, gain·permutation 모두 0이다. `number_type`이 있으면 완전 중복.
3. **gain vs permutation**: 위 피처 중요도 절 참고. `cold_contact`는 gain이 B에서 15.8%로 크지만 permutation은 0.010으로 작다. 파생 4개 중 상대적으로 강한 쪽은 `cold_contact`이고, 약한 2개는 최종 모델에서도 기여가 작다.

### `is_reliable_url` stress 재검증 (`url_reliability_recheck.py`)

Track C 우위가 `is_reliable_url`(정상/피싱 URL 신뢰도) 가정에 얼마나 의존하는지, gain이 아니라 permutation importance 기준으로 재확인하는 스크립트. 상세 결과·표는 [Structure.md](Structure.md) 참고.

```
python -X utf8 -m Simulator.Detection.url_reliability_recheck
```

### 파생 feature 검증 (`-m ablation` / `-m ablation_stability`)

**방식**: `derived_features.py`가 원본 컬럼만으로 계산한 13개 파생 feature 후보를,

1. baseline(원본 21개) vs full(원본+13개 전부) 성능 비교로 "무분별하게 다 넣으면 손해인지" 먼저 확인
2. 후보끼리 상관계수 0.8을 넘는 쌍은 `is_phishing`과의 상관계수가 더 낮은 쪽을 다중공선성으로 제거
3. train_sub로 학습한 모델의 SHAP(참고용)과 Permutation Importance(held-out val_sub 기준, 실제 판단 기준)를 계산해 permutation importance ≤0인 feature 제거
4. 남은 feature + 원본으로 최종 재학습 → test_set 평가

순서로 검증한다.

**결과**: 13개 중 다중공선성 제거로 3개, permutation importance 기준으로 6개가 추가로 탈락하고 **4개만 최종 확정**됐다.

| column명                    | 정의                                         | Track              |
| --------------------------- | -------------------------------------------- | ------------------ |
| `structural_phishing_score` | 문자 구조적 위협 누적 지수(0–5점)            | 단말+구조적신호(C) |
| `is_sms_initiated_unreg`    | 미등록 발신자의 문자 개시 여부               | 단말(B)            |
| `cold_contact`              | 완전 낯선 접촉 여부(미저장+이력 없음)        | 단말(B)            |
| `repeat_pressure_intensity` | 재연락 압박 강도(재연락 간격 기반 연속 점수) | 단말(B)            |

이 4개를 추가해도 baseline 대비 f1/PR-AUC는 거의 그대로였는데(뚜렷한 향상도, 손해도 아닌 수준), 이 차이가 실제 효과인지 단일 실행의 우연인지 확인하기 위해 `run_stability_check()`로 같은 K=10 fold 안에서 baseline과 반복 비교했다(당시 비교 대상은 확정 직전의 4개 조합). **모든 지표에서 diff가 baseline/final의 결합 표준오차보다 작아, 통계적으로 baseline과 구별되지 않음을 확인**했다(= 4개를 추가해도 손해가 없다는 것이 재검증으로 뒷받침됨). 추가로 4개 중 `cold_contact`/`structural_phishing_score` 2개만으로도 4개 전체 효과의 대부분을 재현하고, 나머지 2개는 기여가 거의 없다는 것도 확인했다(자세한 수치는 [Structure.md](Structure.md) 참고).

확정된 4개는 이제 `schema_columns.py`의 SCHEMA에 등록되어 production feature(총 25개)에 포함됐다. 그래서 `feature_ablation.py`의 후보 pool(`CANDIDATE_DERIVED_FEATURES`)도 13개에서, 검증에 탈락했거나 아직 미확정인 나머지 9개로 줄었다 — 이후 `-m ablation`을 다시 돌리면 이 4개는 이미 baseline에 포함된 채로, 남은 9개(또는 새로 추가하는 아이디어)만 후보로 검증하게 된다.

**현재 `-m ablation_stability`의 의미(SCHEMA 등록 이후):** 확정 4개는 이미 baseline에 들어가 있어서 다시 “더하기” 비교를 하면 컬럼이 중복된다. 그래서 기본 비교 대상은 **아직 SCHEMA에 없는 잔여 9개**(`remaining_candidates`)다. “확정 4개가 노이즈인지”를 보던 과거 실행과 목적이 다르다.

실제로 SCHEMA 반영 이후 이 9개를 25개짜리(4개 포함) baseline 대비로 다시 검증해봤는데, 다중공선성 제거 후 남은 7개 전부 permutation importance ≤0으로 다시 탈락했다(`final_candidate_features: []`) — 이미 탈락했던 후보들이 더 강해진 baseline에서도 여전히 기여가 없다는 뜻이라, 4개 확정 결정을 다시 한번 뒷받침한다.

### 민감도 분석 (`-m sen` / `-m each_sen` / `-m ratio` / `-m stress`)

**방식**: 근거가 약해 low/mid/high 3단계로 관리하는 feature 12개를 하나씩, 나머지는 "mid"로 고정한 채 값만 바꿔가며 K-Fold(기본 K=5)로 학습·평가해 어느 단계가 나은지 비교한다(`sen`/`each_sen`). 지인:기관 하위집단 비중(A–E)도 같은 방식으로 별도 비교한다(`ratio`). 결과가 애매하면 K=10으로 재검증하고, 근거가 특히 약한 feature는 sophistication이 아니라 확률/θ **절대값 자체**를 낮음부터 극단까지 흔들어보는 stress 모드로 한 번 더 검증한다.

**결과 예시**:

| 검증 대상                               | 결과                                                                                                     | 확정                                   |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `repeat_gap` sophistication (K=10)      | low(f1=0.9549) > mid(0.9521) > high(0.9466) — low가 뚜렷하게 우수                                        | **low**                                |
| `sms_to_call_gap` sophistication (K=10) | low/mid/high f1 0.9550–0.9573로 사실상 무차이(노이즈 범위)                                               | **mid**(중간값으로 안전하게 채택)      |
| 지인:기관 비중 A–E                      | D가 PR-AUC=0.9776로 최상위, f1(0.9576)도 상위권(E가 f1=0.9589로 근소하게 더 높지만 PR-AUC는 D가 더 높음) | **B → D로 변경**                       |
| `url_rate` stress 검증                  | 확률을 very_low부터 extreme까지 흔들어도 f1 0.949–0.962, PR-AUC 0.975–0.989 범위 — 급격한 성능 붕괴 없음 | mid 가정이 다소 어긋나도 안전함을 확인 |

전체 스윕 결과는 전부 `sensitivity_results/*.json`에 저장되며, 상세 방법과 최종 확정값 전체 목록은 `Simulator/main.py`의 `FINAL_SOPHISTICATION`과 [Structure.md](Structure.md)에서 확인할 수 있다.

---

## 결론 (보고서용)

통신사 실측 없이 단말·구조적 신호만으로도, 합성 데이터 기준 보이스피싱을 높은 정밀도로 걸러낼 수 있었다. 최종 모델은 Optuna로 고른 XGBoost이며, 문자 URL·번호 불일치 등 구조적 신호를 더한 Track C가 단말만 쓰는 Track B보다 F1 0.9496->0.9847로 +0.0351, PR-AUC 0.9695->0.9857로 +0.0162 증가하여 Track C의 성능이 앞선 것을 확인하였다. 두 Track에서의 성능지표는 원본(α)의 결과 보다 보정(γ)의 값이 떨어졌지만 Confidecne Calibration을 적용을 통해 예측 확률을 실제 발생 빈도에 맞게 재조정하였다. 보고용 피처 중요도는 XGBoost gain이 아니라 permutation(PR-AUC)을 쓴다. 핵심 신호는 발신번호 대역(`number_type`)과 문자→통화 간격(`sms_to_call_gap`)이고, Track C에서는 비신뢰 URL(`is_reliable_url`)이 추가로 큰 역할을 한다. gain에서 커 보이던 `sms_to_call`은 섞어도 순위가 거의 바뀌지 않았고, `is_global`은 `number_type`과 완전 중복이며, 파생 4개 중 약한 2개는 빼도 랭킹 성능이 유지된다. 다만 이 수치는 합성 데이터 내부 평가이므로 실제 통화·문자 분포와의 차이는 아직 검증하지 못했고, Confidecne Calibration의 개선폭도 적은 표본(보정용 검증셋 피싱 약 209건)에서 관찰된 것이라 통계적 유의성은 확인하지 않았다.

## 방법론 핵심 원칙

- **Generation-Detection 분리**: `Detection/`은 `Generation/config.py`를 참조하지 않는다. 생성 규칙 암기 방지가 목적.
- **sophistication 3단계 체계**: 근거가 불확실한 feature 값은 확정값 하나로 못 박지 않고 low/mid/high 3단계로 나눠 관리한다. 최종값 확정 전에는 민감도분석(`sen`/`each_sen`)으로 세 단계 간 성능 차이를 비교해 어느 수준이 타당한지 확인하고, 애매한 결과는 K=10으로 재검증한 뒤 확정한다.
- **값의 근거 수준 표기**: `config.py`의 각 확률/θ 값에는 그 값이 어디서 왔는지(실측 θ값·판결문 집계 n·마스터표 서술 근거·근거 없어 무정보사전확률 등)를 주석으로 함께 남긴다. 근거가 약한 값일수록 stress 모드로 절대값 자체가 틀렸을 때의 성능 흔들림을 별도로 검증한다.
- **클래스 불균형 반영**: 실제 보이스피싱 발생률에 맞춰 전체의 약 1%만 피싱으로 생성하고, `scale_pos_weight`로 학습 시 양성 오분류에 더 큰 패널티를 준다. accuracy 대신 PR-AUC/Lift@Top-K%를 주요 지표로 사용.
- **목적에 따라 다른 PR-AUC 계산 방식 사용**: `precision_recall_curve()`+`auc()`(사다리꼴 적분)는 PR 곡선을 직선으로 보간해서 값이 과장될 수 있다는 게 알려진 한계라, 모델/조합끼리 비교하는 곳(파생 feature 검증, Track B/C 최종 비교, Optuna 모델 비교)은 보간 편향이 없는 `average_precision_score()`를 쓰고, 같은 feature의 sophistication 단계 간 상대적 우열만 비교하면 되는 민감도분석은 기존 사다리꼴 방식을 그대로 쓴다(`Detection/train_eval.py::evaluate(pr_auc_method=...)`).
- **결측치는 구조적 게이트로만 발생**: 예를 들어 "사건 내 반복 접촉이 없으면 재연락 간격은 결측"처럼, 결측은 항상 선행 조건에 의해 명시적으로 발생하며 `schema.py`의 `depends_on`에 그 규칙을 텍스트로 남긴다. 근거 없는 임의 결측은 없다.
- **낙관 편향 방지**: 판정 threshold는 항상 train_set에서만 결정하고 test_set에는 고정 적용만 한다. feature 선정 단계의 3단계(Embedded/Permutation Importance)도 train_set 내부의 별도 val_sub로만 계산해 test_set을 건드리지 않는다.
- **재현성 vs 실행 간 흔들림의 구분**: 같은 코드·같은 설정이라도 XGBoost의 멀티스레드 히스토그램 학습 특성상 완전히 새 프로세스로 실행하면 결과가 미세하게 달라질 수 있음을 직접 확인했다. 그래서 permutation importance가 0에 가까운 경계선 feature는 단일 실행 결과만으로 판단하지 않고, 같은 fold로 baseline과 반복 비교하는 안정성 재검증(`run_stability_check`) 단계를 별도로 거친다.
- **최종 보고 시 gain이 아니라 permutation**: `-m final`의 `feature_importance`는 XGBoost **gain**(합=1)이라 트리 분할에 자주 쓰인 피처가 과대 평가될 수 있다. 최종 해석은 같은 모델·같은 test_set에서 피처를 섞어 PR-AUC가 얼마나 떨어지는지 보는 permutation importance를 쓴다(`borderline_recheck.py`).
- **Calibration 효과와 데이터 축소 효과의 분리**: calibration을 사용하기 위해 보정기를 학습에 안 쓴 별도 데이터(`val_calib`)로 적합해야 해서, 그만큼 모델 학습 데이터가 줄어든다(`train_set` 100% → `train_sub`). 이 둘을 한 번에 켜고 끄면 성능 차이가 "보정 때문"인지 "데이터가 줄어서"인지 구분이 안 되므로, 원본(α)/데이터만 축소(β)/보정 적용(γ) 3버전으로 나눠 비교한다(`run_final(split_for_calibration, apply_calibrator)`, `-m calib_compare`).

---

## 향후 계획 / 미해결 사항

- **Track A(통신사) 데이터 부재**: 발신번호 변작·번호 클러스터링·사용 기간 등 통신사 실측자료 기반 feature는 현재 전부 결측 처리돼 있다. 통신사 협조나 대체 자료 확보 시 반영 필요.
- **파생 feature 잔여 후보**: 13개 중 4개만 확정 반영됐고, 나머지 9개는 검증 탈락(다중공선성/낮은 permutation importance) 또는 대기 상태다. 새로운 파생 feature 아이디어가 나오면 동일한 ablation 파이프라인으로 재검증 가능.
- **실데이터 대비 검증 공백**: 현재 모든 성능 지표는 합성 데이터 내부의 train/test 분할 기준이다. 실제 피해 사례·정상 통화 표본과의 분포 차이(synthetic-to-real gap) 검증은 아직 진행되지 않았다.
- **모델 비교 확장**: 현재는 XGBoost/LightGBM만 Optuna로 비교한다. 향후 다른 계열 모델과의 비교도 고려 가능.
- **`is_global` 중복**: `number_type`과 100% 일치해 최종 모델에서 쓰이지 않는다. SCHEMA에서는 남겨 두었고, 보고 시에는 중복 피처로 명시한다.
- **Confidence Calibration 평가의 표본 크기**: 평가는 보정용 검증셋의 피싱 약 209건, 테스트셋의 피싱 약 298건이라는 적은 표본으로 진행되어, ECE·Brier score의 개선이 실제 효과인지 표본에 따른 우연인지 확인하기 어렵다. 이후 전체 N의 수를 늘려가며 재학습.
