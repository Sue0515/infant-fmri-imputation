# Raw 기반 temporal imputation — 2026-11-07

이 프로젝트의 주 실험 경로는 raw NIfTI에서 시작하고 STSR 구조를 무작위 초기화해
학습한다. 기존 STSR 사전학습 가중치는 사용하지 않는다. 공개 `_Z` 데이터 어댑터는
초기 feasibility 분석용으로 남겨 두었다. **코드 경로의 구현·검증은 완료했지만,
실제 영아 자료의 학습은 아직 시작하지 않았다.** 시간 대응과 annotation, ROI가
검증되지 않은 자료는 새 어댑터가 거부한다.

## 이번에 확인한 자료

`outputs/raw_pipeline_2026-11-07/raw_audit/audit.json`에 전체 조사 결과가 있다.
5개 movie의 영아 기록 94개, 연결된 raw NIfTI 98개를 확인했다. `Child_Play`를
제외한 연속 movie 후보는 72기록이다. 6개 ZIP 전체에서 README에 적힌 burn-in
파일을 찾지 못했다. `ready_records: 0`은 새 경로에 필요한 검증이 완료된 기록이
없다는 뜻이며, raw 데이터가 모두 불량하거나 관측률이 0%라는 뜻이 아니다.

기존 processed 신호의 높은 관측률은 raw 축의 학습 가능 비율과 다르다.
원 논문은 motion 오염값 보간과 다른 신호 처리를 기술한다.
[eLife Methods](https://elifesciences.org/reviewed-preprints/92119v1).
고정된 원본 코드에서는 pseudorun의 사용 가능한 앞부분 길이에 따라 burn-in이
달라질 수 있다. 따라서 모든 기록에서 3 TR을 빼는 규칙을 자동 적용하지 않았다.
[pseudorun_divide.m, 고정 커밋](https://github.com/ntblab/infant_neuropipe/blob/1d9d586655ca57080b4b3cbdb51b66825f87f6b9/prototype/link/scripts/Analysis_Timing_functions/pseudorun_divide.m).

기존 가족 단위 train/val/test 분할을 그대로 사용하며 split 파일 hash를 기록한다.
기존 processed 기준 71기록과 raw 후보 72기록은 eligibility 기준이 다르므로
동일한 코호트라고 가정하지 않는다.

## 입력에서 학습까지

1. Raw NIfTI의 공간 단위와 TR 단위를 검사한다. mm와 sec/msec/usec만 허용한다.
2. 분석 구간 밖의 고정 calibration 프레임 하나를 지정한다. 이 프레임은 확인된
   burn-in 이후여야 하고 motion 불량으로 표시되지 않아야 한다.
3. 각 프레임을 calibration에 독립적으로 6-DOF rigid registration한다.
   선형 **공간** resampling은 사용한다. 이전 프레임의 transform이나 전체 run
   평균을 reference로 사용하지 않는다.
4. 미리 지정한 해부학적 ROI에서 값을 추출하고 원래 TR과 내부 결측 위치를 유지한다.
   motion 불량, gaze 불량/미확인, 공간 coverage 밖의 값은 관측/정답에서 제외한다.
   실제 오염값은 NaN으로 지운다. motion 불량이면서 gaze가 확인된 부분만
   recoverable로 표시하며, 추론은 같은 voxel에 관측이 최소 2개 있는 위치만 채운다.
5. 남은 정상 관측에서 불규칙한 구간을 추가로 가리고 이를 학습 정답으로 쓴다.
   donor gap profile은 train 가족에서만 추정한다. recipient 자신의 가족을
   제외하며, val/test의 gap을 학습용 donor로 사용하지 않는다.
6. 각 voxel의 평균·표준편차는 해당 마스킹 이후 남은 관측에서만 계산한다.
   학습과 raw checkpoint 선택은 observed-only scale로 정규화한 MSE를 사용한다.
   실제 단위 MSE도 별도로 보존한다. 거의 상수인 voxel은 작은 scale 때문에
   normalized MSE가 커질 수 있어 ROI와 신호 QC가 필요하다.

시간 보간, slice timing correction, temporal filtering, despiking, 전체 run
z-scoring은 이 raw 어댑터에서 수행하지 않는다. 모델 내부의 observation-only
선형 보간과 비교 baseline의 보간은 마스킹 이후 허용된 관측만 사용한다.
`initialization: random` 외 설정과 pretrained/resume 경로를 거부한다.
학습된 이 실험의 checkpoint를 평가·추론용으로 읽는 것은 허용한다.

Calibration은 추가로 제공되는 관측 영상이다. 모든 방법에서 같은 calibration과
공간 처리를 사용하고, 모든 scored interval 밖에 예약해야 한다. 참조 영상,
ROI, annotation이 고정된 조건에서 한 hidden 프레임을 바꿔도 다른 프레임의
준비 값이 변하지 않는 것을 테스트했다. **전체 metadata 생성 과정이나 실제
뇌 신호 복원 정확도까지 검증했다는 의미는 아니다.**

## 실행 절차

```bash
python -m pip install -e '.[dev,raw]'

infant-audit-raw \
  --data-root /pscratch/sd/s/suincho/data/infant_movies_dryad_jm63xsjm3 \
  --split outputs/mask_audit/cohort_split.json \
  --out outputs/new_raw_audit

# 아래 두 명령은 selection과 annotation/ROI 검증이 끝난 후에 실행한다.
infant-prepare \
  --selection verified_raw_selection.json \
  --data-root /pscratch/sd/s/suincho/data/infant_movies_dryad_jm63xsjm3 \
  --out outputs/new_raw_prepared

infant-impute train \
  --manifest outputs/new_raw_prepared/manifest.json \
  --config configs/infant_raw_imputation.yaml \
  --device cuda --out outputs/new_raw_training
```

출력 폴더는 새 경로여야 한다. 감사 도구가 만든 `selection_unverified.json`은
**미검증 초안**이므로 바로 prepare에 넣으면 실패하는 것이 정상이다.
`verified: true`는 사용자의 검증 선언이지 코드가 외부 증거를 독립적으로
확인했다는 보증이 아니다. 아래 항목의 출처와 대응표를 실제로 확인해야 한다.

Selection은 `spatial` 설정과 `records` 목록을 가진 JSON이다. 각 record의 필드는:

| 필드 | 의미 / 요구사항 |
| --- | --- |
| `movie`, `ppt`, `family`, `split`, `raw_member` | 기존 가족 분할과 정확한 raw ZIP member |
| `burn_in_frames` | 해당 raw run의 검증된 burn-in 수; 기본값 없음 |
| `analysis_start_frame`, `analysis_stop_frame` | 0부터 시작하는 raw 인덱스, stop 제외; 연속 2–256프레임 |
| `reference_frame` | post-burn-in이며 분석 구간 밖인 고정 raw 프레임 |
| `timing` | `verified`, `evidence`, `movie_onset_raw_seconds`; raw 촬영 시작 기준 초 |
| `roi_voxels` | 고정 reference 격자의 정수 `V × 3` 좌표 .npy, 1–256 voxel |
| `registration_mask` | 같은 3D 격자의 bool .npy; ROI를 포함 |
| `raw_motion_bad`, `raw_gaze_bad`, `raw_gaze_known` | **전체 raw run 길이**의 bool .npy |
| `annotation_evidence`, `roi_evidence` | annotation 대응과 해부학적 ROI 정의의 근거 |

외부 .npy 경로는 selection 파일 위치 기준이다. 준비 결과에는 raw 프레임 인덱스,
raw 시간, movie 시간, 좌표, mask, transform, source receipt를 보존한다.
원본 압축 해제 NIfTI, annotation, selection, 출력 record의 SHA256도 기록한다.
Raw 설정으로 학습·평가·추론하면 legacy `_Z` record와 필수 receipt가 없는
record를 거부한다. 이 검사는 잘못된 파일 혼용을 막으며, provenance 위조를
암호학적으로 막는 장치는 아니다.

## 본학습 전에 필요한 확인

- 각 raw/pseudorun의 burn-in 수와 exported timing의 시간 원점.
- 공개 motion/eye confound의 각 행이 어느 raw frame인지 나타내는 대응표.
  Crop, padding, movie onset, gaze의 BOLD 지연 적용 여부를 함께 확인한다.
  길이가 비슷하다는 이유로 processed mask를 raw에 복사하면 안 된다.
- 해부학적으로 정의한 visual ROI, reference 프레임 선택 근거,
  reference/ROI 위치와 registration·coverage QC.
- 실제 오염의 censor 기준과 인접 guard 구간을 적용할지에 대한 사전 규칙.
  현재는 제공된 raw annotation을 그대로 사용하며 추가 guard를 자동 적용하지 않는다.

기존 annotation을 정확히 대응시키거나, raw에서 새 QC 프로토콜로 annotation을
만들어 검증하는 두 경로가 가능하다. 현재 저장하는 motion 진단은 정해진 반경의
6개 점에서 측정한 transform 간 RMS displacement다. **Power FD, FSL 지표 또는
원 연구의 3mm 기준과 동일하지 않다.** 기본 반경 35mm/진단 임계값 3mm가
원 연구 기준을 재현한다는 뜻이 아니며, 이 값으로 학습 mask를 자동 생성하지 않는다.

## 검증과 해석의 범위

50개 테스트가 통과했다. 시간 단위, raw ZIP 전체 변환, 실제 payload overlap
거부, 독립 공간 변환, hidden 입력 불변성, observed-only 정규화,
pretrained 거부, checkpoint round trip과 기존 회귀 검사를 포함한다.
ZIP의 32-bit wrapped offset을 메모리에서 복구하며 원본 ZIP은 변경하지 않는다.

실제 Aeronaut raw 한 run에서 raw index 4–15의 12프레임을 처리했다.
`outputs/raw_pipeline_2026-11-07/real_raw_smoke/summary.json`과 `spatial_qc.png`에
결과가 있다. 모두 reference MSE가 감소했다. 그러나 reference index 3은
**공학적 확인용 선택**이며 burn-in이 3이라는 증거가 아니다. Reference intensity로
만든 mask도 해부학적 visual ROI가 아니다. 이 결과에서 학습 manifest를 만들지 않았다.
QC 그림에는 줄무늬 형태의 아티팩트가 보이며, rigid 정렬만으로 프레임 내부의
오염까지 제거했다고 해석할 수 없다. MSE 감소는 복원 성능 지표가 아니다.

이 경로는 최소한의 공간 처리 구현이다. Susceptibility distortion correction,
전체 코호트의 해부학적 정렬 및 영상 QC를 마친 완성형 infant preprocessing이 아니다.
SimpleITK registration은 [공식 예제](https://simpleitk.readthedocs.io/en/master/link_ImageRegistrationMethod4_docs.html)의
API를 사용하고 버전 2.5.7을 고정한다.

현재 checkpoint 선택은 target 수로 가중한 normalized MSE다. 가족별 macro 지표,
5-fold CV와 최종 개발용 전체 재학습은 여전히 미구현이다. 실제 motion 구간에는
깨끗한 정답이 없으므로 성능은 정상 관측의 holdout과 별도의 downstream 평가로
검증해야 한다. Native TR 결측 복원과 sub-TR superresolution의 정확도는 다른
주장이다. 기존 전처리 전체를 대체하거나 movie decoding을 개선했다는 claim은
아직 뒷받침되지 않는다.
