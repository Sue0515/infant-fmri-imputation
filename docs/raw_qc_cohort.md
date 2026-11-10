# Raw BOLD 복원 실험용 코호트

이 경로는 영화·gaze 정렬이 확정되기 전에 시작하는 **탐색적 raw BOLD temporal
imputation** 실험이다. STSR는 무작위 초기화하며, 신호는 공개 raw NIfTI에서만
읽는다. `_Z` 신호를 학습 입력이나 정답으로 사용하지 않는다.

정확한 movie/gaze 대응이 필요한 [기존 raw 경로](raw_pipeline.md)는 그대로 유지한다.
새 코호트의 source kind는 `raw_framewise_qc_v1`이며, 기존
`raw_framewise_spatial_v1`와 혼용할 수 없다. 원 논문의 inclusion 기준을
재현하거나 gaze가 정상인 구간만 골랐다고 해석하지 않는다.

## 누락된 시간 metadata의 처리

현재 배포본에는 README에 적힌 run별 burn-in 파일이 없다. 원본 코드에서도
예외가 가능하므로 각 run의 burn-in을 3이라고 채우지 않는다.
새 실험은 raw 촬영 시간만 사용하고 앞 11프레임을 학습 구간에서 제외한다.
Raw index 10을 공간 정렬의 고정 reference로 예약하며, index 11부터 최대
256프레임을 분석한다. 이는 이번 실험의 고정 prefix 제외 규칙이며, 원래
burn-in을 알아냈거나 항상 충분한 warm-up을 입증했다는 뜻이 아니다.

Movie onset과 gaze 지연을 추정하지 않는다. 모든 출력의 `gaze_known`은 false,
receipt의 `timing_verified`는 false로 유지한다. Gaze는 이 코호트의 inclusion
조건이 아니다. 영화 외 휴식이나 eyes-closed 구간이 포함될 수 있다.
따라서 이 자료로 movie decoding 또는 시각 주의 특이적 복원 성능을 주장하지 않는다.

## 공간 처리와 ROI

공개 후두엽 ROI와 raw-to-highres 행렬을 사용한다. FLIRT 행렬은 NIfTI world-mm와
다른 scaled-mm 좌표를 사용하므로, voxel 크기와 handedness에 따른 x 반전을
명시적으로 변환한다. [FSL 공식 규약](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/flirt/faq.html).
ROI는 nearest-neighbor로 raw 격자에 옮기고 0.5로 이진화한다. 범위가 [0,1]을
벗어나거나 비유한 값인 mask는 자동 승인하지 않는다.

해부학 ROI를 한 voxel 침식한 뒤 고정 reference의 intensity support와 교차한다.
이 support는 skull stripping 결과가 아니며, 밝기 기준으로 만든 보조 mask다.
그 안에서 공간적으로 퍼진 최대 256개 voxel을 결정론적으로 선택한다. 선택 과정은
분석 구간의 시간 신호를 사용하지 않는다. 각 프레임은 reference에 독립적으로
6-DOF rigid 정렬하고 선형 공간 resampling한다. Temporal filtering, despiking,
시간 보간, 전체 run z-scoring은 하지 않는다.

공개 affine는 고정된 외부 공간 metadata다. 원래 정렬 과정까지 hidden target에
독립적이었다고 주장하지 않는다. 자동 coverage 검사와 대표 영상 overlay를
검사하지만, 전체 코호트의 전문가 해부학 검토는 별도 검증 과제다.

## 이번 실험의 QC 규칙

아래 값은 원 논문의 censor 기준을 복제한 값이 아니라 이번 탐색 실험의 고정 규칙이다.
Test 성능을 보고 threshold를 조정하지 않는다.

| 검사 | 제외 기준 |
| --- | --- |
| 연속 프레임 간 RMS displacement | 1mm 초과 |
| Reference에서의 RMS displacement | 3mm 초과 |
| 정렬 후 reference와의 spatial correlation | 0.8 미만 |
| Reference voxel 크기로 정규화한 fractional DVARS | 0.10 초과 |
| 선택한 ROI의 공간 coverage | 98% 미만 |
| 등록 실패·비유한 값 | 해당 프레임 제외; 차분 QC가 불명확한 다음 프레임도 제외 |

RMS displacement는 35mm 반경의 6개 점에서 계산한다. **Power FD 또는 FSL의
기존 motion 지표와 같지 않다.** 실제 motion으로 분류되고 transform의 앞뒤 대응이
알려진 프레임만 recoverable로 표시한다. Signal QC만 실패한 프레임은 자동 복원
대상으로 취급하지 않는다. 최소 30개 프레임, 분석 구간의 40% 이상이 QC를 통과해야
record를 포함한다. ROI 내부에서도 관측이 유한한 값만 학습 정답 후보가 된다.

QC와 ROI 선택을 고정한 후 정상 관측을 인위적으로 가려 학습한다. QC 자체는
분석 구간의 신호를 사용하므로, 입력 불변성 검증은 이 고정된 eligibility를
조건으로 한다. 실제 motion 구간에는 깨끗한 정답이 없으며, QC 통과가 신경 신호의
완전한 무오염을 보증하지도 않는다.

## 실행과 재현

```bash
infant-prepare-raw-qc --support SOURCE_SUPPORT_DIR --data-root DRYAD_DIR \
  --out NEW_OUTPUT_DIR --workers 8

infant-impute train --manifest NEW_OUTPUT_DIR/manifest.json \
  --config configs/infant_raw_qc_imputation.yaml --device cuda --out NEW_TRAINING_DIR
```

`protocol.json`, 기록별 `qc.json`, `summary.json`에 처리 규칙, raw SHA256,
원본 보조자료 출처, 제외 이유, 프레임 QC를 기록한다. Prepared NPZ는 원래 raw
프레임 인덱스와 TR을 유지하며 실제 오염값을 NaN으로 지운다.
가족별 train/val/test 분할은 기존 `cohort_split.json`을 보존한다. QC로 제외된
가족을 다른 split으로 이동하지 않는다. Train 가족의 motion gap으로만 fitted
mask profile을 만들고, val/test에서 학습용 donor를 얻지 않는다.

Checkpoint 선택은 관측값만으로 정규화한 pooled MSE다. 여러 seed, 가족별 통계,
교차검증과 downstream decoding은 별도의 실험 단계이며 이 데이터 준비로
완료되었다고 해석하지 않는다.

## 생성된 데이터 — 2026-11-08

72기록을 처리해 51기록을 포함했다. 나머지 21기록은 고정한 관측 수/비율 기준을
충족하지 못했다. 실행 중 예외로 실패한 기록은 없었다. 원래 분할은 train
28기록/16가족, val 4기록/1가족, test 19기록/5가족이다. 원래 validation이
한 가족만 남았으므로 본격적인 설정 비교에는 개발용 32기록/17가족의 5-fold
입력 manifest를 사용한다. Test 19기록/5가족은 모든 fold에서 그대로 예약한다.
이는 fold 입력 준비이며 5개 모델 학습이나 결과 집계가 완료됐다는 뜻은 아니다.

데이터 위치:
`/pscratch/sd/s/suincho/data/infant_movies_dryad_jm63xsjm3/derived/raw_qc_v1_20261108/`

- `manifest.json`: 원래 가족 분할을 유지한 전체 manifest.
- `development_cv/fold_0/manifest.json`부터 `fold_4/manifest.json`: 개발용 교차검증 입력.
- `readiness.json`: 입력·시간 격자·split·학습 마스크·실데이터 gradient 검사.
- `summary.json`: 포함/제외 기록과 QC 결과.
- `qc_overview.png`, `roi_gallery_*.png`: 시간별 QC와 ROI 확인 그림.

51기록의 대표 sagittal/axial ROI 그림을 점검했다. Selected voxel이 후두부 영상
신호 영역에 놓이는 것을 확인했지만, 여러 calibration 영상에 줄무늬 아티팩트가
보인다. 일관된 프레임 내부 아티팩트는 현재 QC만으로 제거가 보장되지 않는다.
이 자료는 탐색적 복원 실험을 실행할 수 있는 입력이며, artifact-free neural
ground truth 또는 전문가가 승인한 cortical ROI라는 의미가 아니다.

```bash
infant-make-folds --manifest PREPARED_DIR/manifest.json --out NEW_CV_DIR --folds 5

infant-impute train \
  --manifest /pscratch/sd/s/suincho/data/infant_movies_dryad_jm63xsjm3/derived/raw_qc_v1_20261108/development_cv/fold_0/manifest.json \
  --config configs/infant_raw_qc_imputation.yaml --device cuda --out outputs/raw_qc_fold0
```
