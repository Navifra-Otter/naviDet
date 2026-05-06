"""
YOLO-pose 데이터셋의 이미지를 미리 고정 크기로 resize 해서 새 디렉토리에 저장.

학습 시 cv2.resize 비용을 제거하고, HDD I/O 부담도 1/N로 줄임.
라벨은 정규화된 좌표라 변환 불필요 → 단순 복사.

사용 예:
    python tools/preprocess_resize.py \\
        --src /media/otter/otterHD/pallet_data/data_yolo/pose/pallet_260310_total/pallet_260310 \\
        --dst /media/otter/otterHD/pallet_data/data_yolo/pose/pallet_260310_total/pallet_260310_640 \\
        --size 640 --jobs 8
"""
import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import cv2
from tqdm import tqdm


def resize_one(args):
    src_img, dst_img, size, quality = args
    if os.path.exists(dst_img):
        return True
    img = cv2.imread(src_img)
    if img is None:
        return False
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    ext = os.path.splitext(dst_img)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        cv2.imwrite(dst_img, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(dst_img, img)
    return True


def process_split(src_split, dst_split, size, jobs, quality):
    src_img_dir = os.path.join(src_split, "images")
    src_lbl_dir = os.path.join(src_split, "labels")
    dst_img_dir = os.path.join(dst_split, "images")
    dst_lbl_dir = os.path.join(dst_split, "labels")
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    img_files = sorted(glob(os.path.join(src_img_dir, "*.jpg")) +
                       glob(os.path.join(src_img_dir, "*.png")))
    print(f"[{os.path.basename(src_split)}] images: {len(img_files)} -> {dst_img_dir}")

    tasks = [(p, os.path.join(dst_img_dir, os.path.basename(p)), size, quality)
             for p in img_files]

    if jobs <= 1:
        for t in tqdm(tasks, desc=f"resize {os.path.basename(src_split)}"):
            resize_one(t)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for _ in tqdm(ex.map(resize_one, tasks, chunksize=16),
                          total=len(tasks),
                          desc=f"resize {os.path.basename(src_split)}"):
                pass

    if os.path.isdir(src_lbl_dir):
        lbl_files = glob(os.path.join(src_lbl_dir, "*.txt"))
        print(f"[{os.path.basename(src_split)}] labels: {len(lbl_files)} -> {dst_lbl_dir}")
        for p in tqdm(lbl_files, desc=f"copy labels {os.path.basename(src_split)}"):
            dst = os.path.join(dst_lbl_dir, os.path.basename(p))
            if not os.path.exists(dst):
                shutil.copy2(p, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="dataset root (train/, valid/ 서브디렉토리 포함)")
    ap.add_argument("--dst", required=True, help="출력 dataset root")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality (1-100)")
    ap.add_argument("--splits", nargs="+", default=["train", "valid"])
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    if src == dst:
        raise SystemExit("src와 dst가 같으면 안 됩니다 (원본 덮어쓰기 위험).")

    for split in args.splits:
        src_split = os.path.join(src, split)
        dst_split = os.path.join(dst, split)
        if not os.path.isdir(src_split):
            print(f"skip: {src_split} not found")
            continue
        process_split(src_split, dst_split, args.size, args.jobs, args.quality)

    print("done.")


if __name__ == "__main__":
    main()
