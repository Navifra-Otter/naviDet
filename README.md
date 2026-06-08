# navidet — YOLOv11 기반 통합 검출 프레임워크

하나의 백본·넥·데이터 파이프라인을 공유하면서 두 계열 모델을 제공한다.
`model.task`(config) 또는 `build_model(task=...)`로 선택한다.

- **6dof** — 1-stage 6DoF Object Pose Estimation. 2D 검출 + 회전(6D) + 크기를 회귀하고
  Depth Head의 dense depth를 unprojection하여 translation을 복원 (`YOLO6DoF`).
- **pose / detect** — 2D 멀티태스크. 표준 YOLO11 방식의 박스 + 키포인트(+세그) 회귀
  (`YOLOPose` + `MultiTaskHead`). Edge PC 실시간 추론을 가정해 nano/small 스케일 기본.

> segment task는 head/loss에 구현돼 있으나, 현재 데이터셋에 마스크 라벨이 없어 학습은
> 미배선(마스크 라벨 확보 시 dataset GT만 추가하면 동작).

## 패키지 구조

```
navidet/
├── core/        # 신경망 코어: blocks, backbone, head(Pose6DoFHead/DepthHead/MultiTaskHead/Proto), model
├── module/      # 처리 모듈: loss(6DoF), loss_mt(멀티태스크), dataset, pose_label, trainer, metrics
├── utils/       # geometry, camera, nms, visualize, config
├── messaging/   # 결과 직렬화/발행 (publisher: R,t→quat/yaw, JSONL)
├── tools/       # 실행 스크립트 (train, predict, eval, build_labels, viz_gt, demo)
└── config/
    ├── default_6dof.yaml   # ★ 6DoF 학습/추론 제어
    └── default_pose.yaml   # ★ 2D pose 학습/추론 제어
```

## 설치
```bash
pip install -r requirements.txt
```

## 1. 설정
`navidet/config/default_6dof.yaml`(6DoF) 또는 `default_pose.yaml`(pose)에서 데이터 경로·
intrinsics·하이퍼파라미터·손실 가중치·추론 옵션을 제어한다. CLI로 즉석 override도 가능:
```bash
--set train.epochs=50 data.imgsz=512 train.lr=1e-3
```
`model.task`로 계열을 고른다: `6dof` | `detect` | `pose`(+`segment`).

## 2. 학습

모든 명령은 repo 루트(이 README가 있는 디렉토리)에서 실행한다. 학습 계열은 config의
`model.task`로 결정되므로, 항상 `--config`로 프리셋을 지정하는 것을 권장한다.

### 6DoF
```bash
# 기본
python -m navidet.tools.train --config navidet/config/default_6dof.yaml

# 하이퍼파라미터 즉석 override
python -m navidet.tools.train --config navidet/config/default_6dof.yaml \
       --set train.epochs=50 train.batch=16 data.imgsz=640 train.lr=5e-4
```

### 2D Pose

```bash
# 기본 (model.task=pose)
python -m navidet.tools.train --config navidet/config/default_pose.yaml

# override 예시 (스케일·키포인트 형상 등)
python -m navidet.tools.train --config navidet/config/default_pose.yaml \
       --set train.epochs=100 train.batch=16 model.scale=s model.kpt_shape="[4, 3]"
```

### 스모크 테스트 (소량 프레임으로 빠르게 동작 확인)

```bash
# 6dof — 8프레임, 1 epoch, 워커 0(에러 즉시 확인)
python -m navidet.tools.train --config navidet/config/default_6dof.yaml \
       --set train.epochs=1 train.limit=8 train.batch=2 train.workers=0

# pose — 8프레임, 1 epoch
python -m navidet.tools.train --config navidet/config/default_pose.yaml \
       --set train.epochs=1 train.limit=8 train.batch=2 train.workers=0
```

### 출력물 · 동작

- 출력: `runs/<train.out>/`에 `best.pt`·`last.pt`, `history.json`, **`curves.png`**(학습 곡선),
  6DoF는 `epoch_NNN/val_viz/`(검증 시각화)도 저장.
- GT 생성: 6DoF는 학습 중 on-the-fly(keypoint+depth→RANSAC→6DoF), pose는 YOLO keypoint
  라벨에서 직접 생성(depth 불필요). 별도 전처리 불필요.
- task는 체크포인트 meta에 기록되어 추론/평가 시 자동 분기된다.

### 주요 override 키 (`--set key=value`)

| 키 | 의미 |
| --- | --- |
| `train.epochs` / `train.batch` / `train.lr` | 에폭 / 배치 / 학습률 |
| `train.limit` | 사용할 프레임 수 제한(0=전체, 디버그용) |
| `train.workers` | DataLoader 워커 수(스모크 시 0 권장) |
| `train.amp` | mixed precision on/off |
| `train.out` | 출력 디렉토리(`runs/...`) |
| `data.imgsz` | 입력 해상도(얇은 박스라 ≥640 권장) |
| `data.train_root` / `data.val_root` / `data.ini` | 데이터·intrinsics 경로 |
| `model.scale` | 백본 스케일 `n/s/m/l/x` |
| `model.task` | `6dof` \| `detect` \| `pose`(\| `segment`) |
| `model.kpt_shape` | pose 키포인트 형상 `[nk, dim]`(dim=3이면 xy+vis) |

> ⚠️ `model.task`를 `--set`으로 바꾸는 것보다 **프리셋 파일 전환**을 권장한다(데이터 경로·
> 손실 가중치가 프리셋마다 다름). 실제 학습은 `data.*` 경로가 유효해야 동작한다.

## 3. 추론 + 시각화
```bash
python -m navidet.tools.predict --config navidet/config/default_6dof.yaml \
       --set predict.ckpt=runs/exp/best.pt predict.num=12
```
- 6dof: `_pred/*_pred.png`(예측 청록 vs GT 초록 쿼드 + 좌표축), `_pred/preds.jsonl`(t/quat/yaw/size)
- pose: `_pred/*_pred.png`(박스 + 키포인트 오버레이)
- task는 체크포인트 meta에 기록돼 자동 분기된다.

## 4. 평가 / GT 검수 (선택)
```bash
python -m navidet.tools.eval --set predict.ckpt=runs/exp/best.pt       # 6DoF 정량평가 전용
python -m navidet.tools.build_labels --root ".../train" --ini ".../CameraParam_....ini"
python -m navidet.tools.viz_gt --root ".../train" --num 20
```

## 데이터 / GT 생성 원리 (6DoF)
- 입력: YOLO 4-keypoint(TR→BR→BL→TL) + Orbbec depth(mm) + intrinsics(.ini)
- 쿼드 내부 그리드 다수점을 depth로 3D 언프로젝션 → **X–Z 평면 RANSAC 직선 피팅**으로
  yaw 강건 추정, 좌/우 코너 레이∩직선으로 폭 복원 (운영 `pallet.py` 기법 차용)
- `size = (가로, 세로, 0)` — 두께(z)는 단면 미관측 + 팔레트마다 달라 0

## Loss
- **6dof** — CIoU(box) + DFL + BCE(obj/cls) + **1-cos(회전)** + SmoothL1(size/translation)
  + L1(dense depth). 라벨 할당은 TaskAlignedAssigner.
- **pose** — CIoU + DFL + BCE(cls) + keypoint L1(박스정규화) + visibility BCE. 동일한
  TaskAlignedAssigner를 재사용해 할당 방식을 6dof와 일관 유지.

## 안정성 노트
- **imgsz ≥ 640**: 팔레트 전면 박스가 ~9px로 얇아 작은 해상도에선 양성 anchor=0.
- depth head는 `sigmoid×max_depth`로 bound, 회전 손실은 `1-cos`(유한 그래디언트) →
  AMP 포함 발산 없음. 학습 루프에 non-finite 가드도 내장.
