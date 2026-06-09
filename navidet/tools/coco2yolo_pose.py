"""
COCO person_keypoints(json) → YOLO-pose 포맷 변환.

PoseDataset(task=pose)가 읽는 레이아웃을 만든다:
    <out>/images/<stem>.jpg   (원본 이미지로의 심볼릭 링크, 복사 안 함)
    <out>/labels/<stem>.txt   (cls cx cy w h  x1 y1 v1 ... x17 y17 v17, 모두 정규화)

person 단일 클래스(cls=0). iscrowd 인스턴스는 제외. 키포인트 1개 이상이 라벨된
이미지만 출력(표준 COCO-pose 학습 관행). 좌표는 [0,1] 정규화, vis는 COCO 그대로(0/1/2).

사용 예 (fiftyone COCO-2017 기준)
--------------------------------
  # val
  python -m navidet.tools.coco2yolo_pose \
      --ann /home/otter/fiftyone/coco-2017/raw/person_keypoints_val2017.json \
      --img-dir /home/otter/fiftyone/coco-2017/validation/data \
      --out /home/otter/datasets/coco-pose/val
  # train
  python -m navidet.tools.coco2yolo_pose \
      --ann /home/otter/fiftyone/coco-2017/raw/person_keypoints_train2017.json \
      --img-dir /home/otter/fiftyone/coco-2017/train/data \
      --out /home/otter/datasets/coco-pose/train
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="person_keypoints_*.json 경로")
    ap.add_argument("--img-dir", required=True, help="원본 이미지 디렉토리(data/)")
    ap.add_argument("--out", required=True, help="출력 루트(images/ labels/ 생성)")
    ap.add_argument("--copy", action="store_true",
                    help="심볼릭 링크 대신 이미지를 복사")
    ap.add_argument("--keep-empty", action="store_true",
                    help="키포인트가 0개인 person 인스턴스도 box로 포함(기본: 제외)")
    args = ap.parse_args()

    img_out = os.path.join(args.out, "images")
    lbl_out = os.path.join(args.out, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    print(f"[load] {args.ann}")
    coco = json.load(open(args.ann))
    images = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    n_img = n_inst = n_skip_crowd = n_skip_empty = 0
    for img_id, anns in anns_by_img.items():
        im = images[img_id]
        W, H = im["width"], im["height"]
        fname = im["file_name"]                       # 예: 000000397133.jpg
        stem = os.path.splitext(fname)[0]
        lines = []
        for a in anns:
            if a.get("iscrowd", 0):
                n_skip_crowd += 1
                continue
            if a["num_keypoints"] == 0 and not args.keep_empty:
                n_skip_empty += 1
                continue
            x, y, w, h = a["bbox"]                    # top-left x,y,w,h (px)
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            bw, bh = w / W, h / H
            kp = a["keypoints"]                       # [x1,y1,v1, ...] len 51
            parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
            for i in range(0, len(kp), 3):
                kx, ky, v = kp[i], kp[i + 1], kp[i + 2]
                if v > 0:
                    parts.append(f"{kx / W:.6f} {ky / H:.6f} {int(v)}")
                else:
                    parts.append("0.000000 0.000000 0")   # 미라벨 → 0
            lines.append(" ".join(parts))
            n_inst += 1

        if not lines:
            continue

        with open(os.path.join(lbl_out, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

        # 이미지 링크/복사
        src = os.path.join(args.img_dir, fname)
        dst = os.path.join(img_out, fname)
        if not os.path.exists(dst):
            if args.copy:
                import shutil
                shutil.copy2(src, dst)
            else:
                os.symlink(os.path.abspath(src), dst)
        n_img += 1

    print(f"[done] images={n_img}  instances={n_inst}  "
          f"(skip crowd={n_skip_crowd}, empty-kpt={n_skip_empty})")
    print(f"  images → {img_out}")
    print(f"  labels → {lbl_out}")


if __name__ == "__main__":
    main()
