# navidet 6DoF Pose 파이프라인

YOLO 4-keypoint 라벨 + Depth로부터 **6DoF 물체 자세(R, t, size)** 를 학습/추론하는
단일 파이프라인. 데이터 GT 생성 → 데이터셋 → 학습(+증류) → 추론·발행의 4단계로 구성된다.

> 다이어그램: [`pipeline.drawio`](pipeline.drawio) (draw.io / VSCode Draw.io Integration 확장으로 열기)

```
① GT 생성        ② 데이터셋/로더        ③ 학습                ④ 추론·평가·발행
pose_label.py  →  dataset.py        →  train.py             →  predictor.py
(mode="face")     PoseDataset          train_distill.py        eval.py
                  DataLoader           YOLO6DoF + Loss         publisher.py
```

---

## 0. 좌표 / 데이터 규약

- **카메라 좌표계**: x 오른쪽, y 아래, z 전방(광축).
- **오브젝트 축**: x = 가로(전면 폭), y = 세로(아래), z = 전면 법선. `R = camera_R_object`.
- **입력 데이터** (예: `mando_aug` cart 데이터셋)
  - `images/*.png` — RGB (color 해상도, 예 1920×1080)
  - `depth/*.png` — uint16 PNG, color 해상도로 정합(D2C=identity). `× depth_scale(0.001)` → meter
  - `labels/*.txt` — YOLO keypoint: `cls cx cy w h kx1 ky1 ... (4 keypoints, 순서 TR,BR,BL,TL)`
  - `CameraParam_*.ini` — `[ColorIntrinsic]` 의 K (fx,fy,cx,cy) 사용 → [`utils/camera.py`](../navidet/utils/camera.py)

---

## ① 6DoF GT 생성 — [`module/pose_label.py`](../navidet/module/pose_label.py)

키포인트 + depth → `ObjectPose(R, t, size, kpts3d, rmse, valid)`.
`keypoints_to_pose(..., mode=...)` 가 진입점이며 **`mode="face"` 가 권장(full 6DoF)**.

### `mode="face"` 흐름 (테두리 평면 → GT 코너 복원)

| 단계 | 함수 | 입력 → 출력 |
|---|---|---|
| 1. 테두리 샘플 | `sample_face_border` | 쿼드 4점 → 테두리 띠 픽셀(좌·우·하, **상단 제외**, 모양유지 `expand`) |
| 2. 언프로젝션 | `rgbd_unproject` | 픽셀 + depth + K → 3D 점군(유효 depth만) |
| 3. 평면 피팅 | `ransac_plane_full` | 3D 점군 → **3D RANSAC 평면** `(c, n)` + 인라이어/RMSE |
| 4. 코너 복원 | `ray_plane_intersect` | GT 키포인트 레이 ∩ 평면 → 4코너 `P4` (**사이드 depth 결손도 복원**) |
| 5. 자세 구성 | `R_from_plane_corners` | 평면 법선 + 코너 → `R, t, size`, `kpts3d=P4` |

**설계 근거**
- 외곽 **프레임(테두리)** 은 내부(트레이/구멍/적재물)보다 depth가 robust →
  테두리 띠만 샘플하면 RANSAC 평면이 깨끗(인라이어율 ↑, RMSE ↓).
- Femto Bolt가 **사이드에 depth를 못 줘도**, 코너는 평면 위에서 레이로 복원된다.
- 직사각형을 강제(Kabsch)하지 않고 복원 코너(`P4`)를 그대로 써서 **코너가 2D 키포인트(=객체 edge)에 정확히 붙는다**.
- 평면 기울기에서 **roll/pitch/yaw 전체**가 나온다(직립 가정 불필요).

### 그 외 mode (비교/레거시)

- `full` — 쿼드 전체 샘플 + X–Z 2D 직선(yaw). 얇은 팔레트 전면용.
- `bottom` — 바닥 변 한 줄 + 2D 직선(yaw). 세로로 긴 cart용(레거시).
- `corners` — 4코너 직접 역투영 + 평면복원 + Kabsch.

### 주요 파라미터 (`keypoints_to_pose`)
`max_depth`, `ransac_thresh`, `border_band`(띠 두께, 기본 0.12), `border_expand`(확장, 기본 0.04),
`face_rigid`(True면 직사각형 강제), `rmse`로 품질 판단.

### 배치 생성 — [`tools/build_labels.py`](../navidet/tools/build_labels.py)
프레임별 `pose_labels/*.npz` 캐시 생성(검수/속도용). `--mode {face,full,bottom,corners}` (기본 `face`).

### 시각화 / 예제
- [`tools/viz_gt.py`](../navidet/tools/viz_gt.py) — npz GT를 이미지에 재투영(쿼드/축).
- [`tools/example_face_ransac.py`](../navidet/tools/example_face_ransac.py) — face 방식 데모(복원 점/평면 시각화, bottom 방식과 비교).

---

## ② 데이터셋 / 로더 — [`module/dataset.py`](../navidet/module/dataset.py)

`PoseDataset` 가 프레임을 읽어 학습 텐서로 변환.

- **GT 소스 분기 (`use_cache`)**
  - `use_cache=true` + `pose_labels/` 존재 → 캐시 npz 로드(RANSAC 재계산 생략, 빠름)
  - `use_cache=false` → 매 epoch **즉석** `keypoints_to_pose(mode=pose_mode)` 계산
  - ⚠️ **캐시가 있으면 `pose_mode`를 무시**한다. mode를 바꾼 직후엔 `use_cache=false`로 두거나 캐시를 다시 굽는다.
- **전처리**: `letterbox`(RGB) + `letterbox_depth`(NEAREST) — 동일 변환으로 정합 유지.
- **GT 필터**: `fit_rmse > rmse_thresh` 또는 `valid=False` 인 객체 제외.
- 출력: `(이미지[3,H,W], targets{gt_labels, gt_bboxes, R, t, size, mask_gt, img_size, ...})`.
- `collate_fn` + `DataLoader` → 배치(shuffle, pin_memory).

---

## ③ 학습 — [`tools/train.py`](../navidet/tools/train.py) · [`tools/train_distill.py`](../navidet/tools/train_distill.py)

### 모델 — [`core/model.py`](../navidet/core/model.py) `YOLO6DoF`
```
입력 [B,3,H,W]
  → Backbone(YOLOv11)  → PANNeck  → Pose6DoFHead : box/dfl · cls · rot(6d|quat) · size
                                  → DepthHead     : dense depth [B,1,H/2,W/2]
forward → {"det": head, "depth": depth_map}
decode_pose(K, depth) → {R, t, size}   # 추론 시 자세 복원
```

### 손실 — [`module/loss.py`](../navidet/module/loss.py) `Pose6DoFLoss`
`TaskAlignedAssigner`로 라벨 할당 후 항목별 손실:
`box · dfl · obj · cls · rot · size · depth · trans`.

### 최적화 / 리포트
- AdamW + cosine LR(+warmup), AMP, `clip_grad_norm`.
- [`module/trainer.py`](../navidet/module/trainer.py) `EpochReporter` → `train.out/` 에 저장:
  `best.pt`(val 종합점수 기준), `last.pt`, `epoch_NNN/model.pt`, `curves.png`, `history.json`, `val_viz/`.

### 증류 경로 (`train_distill.py`)
Student `YOLO6DoF` + `FeatureProjector` + **Teacher DINOv3(frozen)**.
`distill loss(cosine|mse)` + task loss를 α/β 스케줄로 가중(`teacher_ckpt` 미지정 시 MockDINOv3 스모크).

---

## ④ 추론 · 평가 · 발행

- [`module/predictor.py`](../navidet/module/predictor.py) `YOLO6DoFPredictor`
  `_preprocess → forward → _postprocess`(top-k 프리필터 + NMS) → `decode_pose(K, depth)` → `R, t, size`.
- [`tools/predict.py`](../navidet/tools/predict.py) / [`tools/eval.py`](../navidet/tools/eval.py) — `predict.ckpt` 로드 → 시각화 / 지표(rot°, t mm).
- [`messaging/publisher.py`](../navidet/messaging/publisher.py) `PosePublisher` → **JSONL**(quaternion + t) 발행.

> ⚠️ `train.out`(쓰기 디렉토리)과 `predict.ckpt`(읽기 파일)는 별개다. 학습 직후 추론하려면
> `--set predict.ckpt=<train.out>/best.pt` 로 맞춘다.

---

## 설정 — [`config/default_6dof.yaml`](../navidet/config/default_6dof.yaml)

| 키 | 값(예) | 의미 |
|---|---|---|
| `data.train_root/val_root/ini` | `mando_aug/...` | 데이터 경로 + 카메라 .ini |
| `data.depth_scale` | `0.001` | depth PNG(uint16) → meter |
| `data.rmse_thresh` | `0.05` | RANSAC 잔차 초과 객체 학습 제외 |
| `data.pose_mode` | `face` | GT 생성 모드 (face/full/bottom/corners) |
| `data.use_cache` | `false` | 캐시 사용 여부(mode 변경 직후 false 권장) |
| `data.nc / class_names` | `3 / [cart_white, cart_blue, cart_gray]` | 클래스 |
| `model.scale / rot_repr / light_head` | `s / 6d / true` | 모델 구성 |
| `train.epochs/batch/lr/out` | `100 / 8 / 5e-4 / runs/6dof/mando_cart` | 학습 |
| `loss.*` | box 7.5 · dfl 1.5 · rot 2.0 · trans 2.0 … | 손실 가중치 |
| `distill.*` | teacher_ckpt, hub_name … | 증류 |
| `predict.ckpt/conf/iou` | `best.pt / 0.25 / 0.5` | 추론 |

CLI override: `--set train.epochs=200 data.imgsz=640 data.use_cache=true`

---

## 실행 방법

```bash
# (선택) face GT 캐시 굽기 — 대량/반복 학습 시 속도 ↑
INI="/media/otter/otterHD/mando_aug/CameraParam_Orbbec Femto BoltCL8855300X5_Color1920x1080_Depth640x576.ini"
python -m navidet.tools.build_labels --root /media/otter/otterHD/mando_aug/train --ini "$INI" --mode face
python -m navidet.tools.build_labels --root /media/otter/otterHD/mando_aug/valid --ini "$INI" --mode face

# 일반 학습 (즉석 face GT)
python -m navidet.tools.train --config navidet/config/default_6dof.yaml

# 캐시 사용 학습
python -m navidet.tools.train --config navidet/config/default_6dof.yaml --set data.use_cache=true

# 증류 학습 (미세조정 DINOv3 가중치 지정)
python -m navidet.tools.train_distill --config navidet/config/default_6dof.yaml \
  --set distill.teacher_ckpt=<dinov3_finetuned.pth>

# 추론 / 평가 (학습 산출물 지정)
python -m navidet.tools.predict --config navidet/config/default_6dof.yaml \
  --set predict.ckpt=runs/6dof/mando_cart/best.pt
python -m navidet.tools.eval    --config navidet/config/default_6dof.yaml \
  --set predict.ckpt=runs/6dof/mando_cart/best.pt

# GT 검수 시각화 / face 방식 데모
python -m navidet.tools.viz_gt --root /media/otter/otterHD/mando_aug/train --num 12
python -m navidet.tools.example_face_ransac --stem frame_978 --out asset/_face_demo.png
```

---

## 파일 맵

```
navidet/
├─ config/default_6dof.yaml      ① 단일 진입 설정
├─ module/
│  ├─ pose_label.py              ① keypoints_to_pose(mode=face) + 헬퍼
│  ├─ dataset.py                 ② PoseDataset / collate / letterbox
│  ├─ loss.py                    ③ Pose6DoFLoss + TaskAlignedAssigner
│  ├─ trainer.py                 ③ EpochReporter / Progress
│  ├─ predictor.py               ④ YOLO6DoFPredictor (NMS + decode)
│  └─ distill.py                 ③ FeatureProjector / teacher
├─ core/
│  ├─ model.py                   ③ YOLO6DoF / YOLOPose
│  ├─ backbone.py · blocks.py    ③ YOLOv11 backbone
│  └─ head.py                    ③ Pose6DoFHead / DepthHead / MultiTaskHead
├─ messaging/publisher.py        ④ PosePublisher (JSONL)
├─ utils/{camera,geometry,visualize,nms,config}.py
└─ tools/
   ├─ build_labels.py            ① GT npz 캐시 생성
   ├─ viz_gt.py                  ① GT 재투영 시각화
   ├─ example_face_ransac.py     ① face 방식 데모(복원/평면 시각화)
   ├─ train.py · train_distill.py③ 학습 진입점
   └─ predict.py · eval.py       ④ 추론 / 평가

asset/
├─ pipeline.drawio              파이프라인 다이어그램
└─ pipeline.md                  본 문서
```

---

## 설계 노트

- **테두리 띠 샘플링이 핵심**: 내부 균등 샘플(인라이어 35~60%)보다 테두리 띠(62~86%)가 평면을 훨씬 깨끗이 잡아 RANSAC이 안정적.
- **복원 코너 vs 직사각형 강제**: 복원 코너(`P4`)는 GT 키포인트에 0.0px로 재투영(edge에 붙음). Kabsch 직사각형 강제는 2~22px 어긋남 → 기본은 복원 코너 사용(`face_rigid=False`).
- **캐시 주의**: `pose_mode`를 바꾸면 기존 `pose_labels/` 캐시를 다시 굽거나 `use_cache=false`로 둬야 새 GT가 학습에 반영된다.
