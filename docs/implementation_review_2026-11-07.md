# 구현 타당성 점검 — 2026-11-07

후속 업데이트: [Raw 파이프라인](raw_pipeline.md)을 추가했다. 이제 `infant-prepare`는
raw 전용이며 기존 어댑터는 `infant-prepare-preprocessed-pilot`로 분리했다.
무작위 초기화 강제, 관측 전용 voxel 정규화, 독립 공간 처리의 불변성 테스트를
포함한다. 아래 내용은 앞선 점검 당시의 기록이며, raw 경로의 **코드 미구현**은
해결했다. 실데이터의 시간·annotation 대응과 ROI 검증은 아직 완료되지 않았다.
Raw 설정의 checkpoint 선택 지표는 pooled normalized MSE로 명시적으로 변경했고,
legacy 설정은 기존 MSE를 유지한다. CV와 가족별 macro 지표는 계속 남은 과제다.

## 판정과 범위

불규칙한 관측 마스크를 조건으로 기존 TR 격자의 BOLD를 예측하는 구현은
feasibility pilot으로 타당하다. 그러나 현재 코드를 실행해 얻은 성능만으로
raw motion 오염 복원 또는 기존 전처리 대체의 유효성을 주장할 수는 없다.
독립 저장소의 데이터 어댑터, 분할 검사, 마스크 생성, STSR/INR, 학습 루프,
baseline, 지표, EDA와 테스트를 점검했다. 이번 점검은 새로운 실데이터 학습이나
raw 전체 영상 QC가 아니다. 모델 성능을 확인한 결과도 아니다.

## 논문용 실험 전에 해결할 사항

### P1 — 입력은 raw가 아니라 이미 처리된 `_Z` 신호

`stsr/imputation/prepare_dryad.py`는
`preprocessed_native/linear_alignment/{ppt}_Z.nii.gz`만 읽는다.
표시된 motion 값을 NaN으로 지우는 조치는 올바르지만, 이전 보간·despiking·
정규화 등의 효과를 되돌리지는 않는다. 보간 적용은 원 논문 Methods에서 확인된다.
기존 문서의 "potentially including motion interpolation" 표현을 수정했다.

- 근거: https://elifesciences.org/reviewed-preprints/92119v1 의 Preprocessing.
- 코드: https://github.com/ntblab/infant_neuropipe/blob/1d9d586655ca57080b4b3cbdb51b66825f87f6b9/prototype/link/scripts/interpolate_TRs.m
- 필요 작업: 복원 목표 신호의 정의, raw에서 출발하는 처리 경로,
  holdout 정답에 접근하지 않는 입력 생성 절차 및 불변성 검증.
- 상태: 미구현. 이 단계에서는 preprocessed-space pilot으로만 해석한다.

### P1 — 시간 대응과 제외 구간의 의미가 확정되지 않음

현재 `validity()`는 CSV `num_TR`까지를 분석 prefix로 사용하고,
`eye_shift_tr`을 사용자가 지정한다. 원본 run, burn-in, movie onset,
SRM의 시간 이동과 gaze 시각을 자동 연결하거나 검증하는 코드는 없다.
`padding`/`acquired`라는 기존 필드는 실제 무촬영 여부를 입증하는 이름이 아니다.
관측값이 남아 있는 excluded tail도 발견됐다.

- 필요 작업: 원본 frame ↔ 처리 frame ↔ 영화 시간 대응표와 clip 경계 검증.
- `Child_Play` 거부는 이 문제가 해결되기 전의 타당한 보수적 처리다.
- 이번 수정은 이미 지정된 분석 prefix 바깥의 계산을 차단한다.
  CSV prefix 자체가 정답이라는 것을 검증하거나 시간을 재정렬한 수정은 아니다.

### P1 — 논의한 5-fold 교차검증과 최종 재학습은 아직 없음

`audit_masks.family_split()`은 전체 가족 ID에 대한 70/15/remainder 단일 분할이다.
현재 적격 자료는 train 19가족/46기록, val 2가족/6기록,
test 5가족/19기록이다. `runner.train()`은 비어 있지 않은 train과 val을 요구하고
val MSE로 checkpoint를 선택한다. Fold 생성, fold 간 결과 집계,
val 없이 개발용 전체를 고정 step만큼 재학습하는 실행 경로가 없다.

- 필요 작업: 기존 test 가족을 보존하고 나머지 개발용 가족에 대한 fold manifest,
  fold별 train-only mask profile, 설정 선택 및 최종 고정-step 재학습 경로.
- 상태: 이번 점검에서 분할·기존 EDA·test 가족을 변경하지 않았다.

### P2 — 점수와 학습 표본의 가중 단위를 명시해야 함

`evaluate()`와 `stress_evaluation()`은 평가한 voxel×time 값 수로 오차를 가중한다.
Checkpoint 선택도 이 pooled MSE를 따른다. 가족별 평균이나 가족 단위 불확실성은
구현되지 않았다. `train()`의 recipient 선택 역시 record 단위이며, donor mask를
가족 균등으로 뽑는 것과 구분해야 한다. 기록이 많은 가족의 영향이 커질 수 있다.

- 필요 작업: 가족별 결과, 동일 가족 내 기록 집계 규칙, 가족 평균의 primary metric,
  동일 target에서의 paired baseline 차이. Pooled 결과는 보조 지표로 보존 가능하다.
- 상태: 이번 수정에서 primary metric을 조용히 바꾸지 않았다.

### P1/P2 — 해부학적 ROI와 주장 범위

어댑터는 명시적으로 주어진 최대 256개 voxel의 native-space 좌표를 읽는다.
Voxel이 뇌/시각피질 안에 있는지, 참가자 간 해부학적 대응이 적절한지는 검증하지 않는다.
EDA의 중앙 격자 probe는 학습용 visual ROI가 아니다. 전체 뇌 실험,
movie decoder, 실제 오염 구간의 깨끗한 정답은 현재 구현에 없다.

- 필요 작업: 해부학 정보로 정한 ROI, 위치·coverage·registration QC,
  downstream 평가 계획. Native affine의 mm 좌표만으로 공통 해부학적 위치가
  보장되지는 않는다.
- 임의 위치의 native-TR 결측 예측과 sub-TR superresolution 정확도는 다른 주장이다.
  이 구현은 앞뒤 관측을 사용하는 offline 복원이며 online causal 모델이 아니다.

## 재현 후 수정한 코드 문제

### 1. 제외된 tail이 예측에 영향을 줌

기존 runner는 `acquired=False`인 뒷부분까지 attention/FFT 입력에 포함했다.
16프레임의 동일 신호 뒤에 제외된 7프레임을 붙인 재현에서, 24개 복원값이 달라졌다
(해당 초기화에서 최대 절대 차이 0.0006256). 영향 크기는 학습 성능의 추정치가 아니다.

`Record.analysis_view()`와 `predict_record()`를 추가했다. Train, validation,
test, inference에서 모델은 지정된 분석 prefix만 읽고, 결과 export는 원래 길이를
보존하며 제외된 tail은 NaN으로 둔다. 내부 motion/gaze 결측은 삭제하지 않는다.
모델 입력의 시간 시작점과 TR도 보존한다.

### 2. 평가 마스크가 manifest 순서와 training seed에 따라 변함

기존 primary seed는 `seed + 100000 + row_index * 1000 + repeat`였다.
같은 기록의 평가 순서를 바꾸거나 모델 초기화 seed를 바꾸면 가림 위치가 바뀌었다.
이 경우 비교 성능 차이에 평가 문제의 차이가 섞인다.

이제 별도 `evaluation_seed`(기본 42), record ID, case, repeat의 SHA256으로
평가 seed를 만든다. Stress case도 같은 원칙이다. 결과에는 protocol version 2,
seed, family를 기록한다. 동일 마스크를 보장하려면 profile·validity·mask 설정도
동일해야 한다. 다른 training fold의 profile까지 동일하게 만든다는 의미는 아니다.

### 3. Reserved test 가족을 validation으로 다시 표시할 수 있었음

기존 `check_evaluation_split(..., 'val')`은 train과의 중복만 차단했다.
Checkpoint manifest의 test 가족을 val로 바꾸어 보내도 통과했다.
이제 평가하려는 split 외의 기존 모든 split과 family/ID/path 중복을 검사한다.
원래 test 가족의 다른 run을 val로 추가하는 경우도 차단한다.
이는 신뢰할 수 있는 ID/가족 metadata를 전제로 한 검사다.

### 4. Inference 출력의 유한성 검사 보강

복원 대상에 NaN/Inf 예측이 생기면 결과를 저장하지 않고 명시적으로 실패한다.
실제 관측값은 기존처럼 그대로 복사한다.

## 확인한 타당한 부분

- 내부 결측 시점은 삭제하지 않고 균일한 원래 시간 격자를 유지한다.
- 실제 invalid/gaze/제외 tail은 지도학습 정답이 되지 않는다.
- Holdout 값을 NaN 또는 큰 수로 바꿔도 입력 feature/예측이 변하지 않으며,
  숨긴 입력값으로 흐르는 gradient는 0이다. 이 보장은 모델 입력 이후에 한정된다.
- 평균·scale은 남아 있는 관측값만으로 계산한다.
- Routing용 선형 보간도 남아 있는 관측값만 사용한다. 원본 정답 누출과는 구분한다.
- Fitted donor profile은 train에서만 만들고 recipient 가족을 제외한다.
- STSR, linear, mean_included는 동일한 알려진 target 값에서 비교한다.
- Stress 구간의 일부만 정답이 있는 경우 scoring coverage를 기록한다.
  전체 구간을 정확하게 복원했다는 증거로 해석해서는 안 된다.
- 원래 관측값을 보존하고 명시된 recoverable motion만 export에서 채운다.
- Checkpoint provenance는 imputation 외 trunk/INR/utils 코드 hash도 포함한다.

## 검증과 호환성

수정 전 기존 테스트 25개는 통과했다. 새 점검 5개 중 4개는 실제 실패했고,
코드 hash 확인 1개는 기존 구현도 통과했다. 수정 후 회귀 테스트 9개와 기존 25개를
함께 검증한다. 회귀 테스트는 순서/seed 불변성, test 재분류 차단, tail 불변성,
학습·검증의 tail 제외, 내부 시간 보존, 비유한 복원값 거부를 포함한다.

평가 protocol과 모델의 tail 처리가 바뀌므로 과거 결과와 직접 숫자를 섞지 않는다.
새 실험은 같은 버전에서 모든 baseline을 다시 평가하고 모델도 다시 학습해야 한다.
가중치 tensor 형식은 바뀌지 않았지만 과거 checkpoint의 예측이 동일하다고 보장하지 않는다.
실제 infant 본학습 결과는 아직 없다. 최종 테스트 실행 결과는
`outputs/implementation_review_2026-11-07/validation.json`에 기록한다.
