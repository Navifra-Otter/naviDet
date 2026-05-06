import torch 
import os, glob, cv2
import numpy as np 
from torch.utils.data import Dataset
from tqdm import tqdm
from pose.utils import printS, colored_msg, MASTER_RANK

class YoloPoseDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, nkpts=4, transform=None, cache_ram=False):
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
        self.label_dir = label_dir
        self.img_size = img_size
        self.nkpts = nkpts
        self.transform = transform

        # RAM 캐시: 디코딩+리사이즈된 uint8 numpy (CHW) 만 저장 (텐서 변환은 매번 cheap).
        # persistent_workers=True 면 워커별로 첫 에폭 후 자기 인덱스를 모두 캐싱 → 2에폭부터 IO 0.
        self.cache_ram = cache_ram
        self._ram_cache = {} if cache_ram else None
        
        # 라벨 미리 읽기 (기존 최적화 유지)
        printS(f"Loading labels for {len(self.img_files)} images...")
        self.cached_labels = []
        pbar = tqdm(self.img_files, desc=f" {colored_msg('[SYSTEMS]', 'blue')} Parsing Labels") if MASTER_RANK else self.img_files 
        for img_path in pbar:
            label_path = os.path.join(self.label_dir, os.path.basename(img_path).rsplit('.', 1)[0] + ".txt")
            targets = np.zeros((0, 6 + self.nkpts * 3))
            
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                try:
                    labels = np.loadtxt(label_path, ndmin=2)
                    if labels.shape[0] > 0:
                        ncols = labels.shape[1]
                        expected_xy = 5 + self.nkpts * 2
                        expected_xyv = 5 + self.nkpts * 3
                        if ncols == expected_xy:
                            kpts = labels[:, 5:].reshape(-1, self.nkpts, 2)
                            vis = np.ones((len(labels), self.nkpts, 1))
                            kpts = np.concatenate([kpts, vis], axis=2).reshape(-1, self.nkpts * 3)
                        elif ncols == expected_xyv:
                            kpts = labels[:, 5:]
                        else:
                            printS(f"Skipping {label_path}: expected {expected_xy} or {expected_xyv} cols, got {ncols}")
                            self.cached_labels.append(targets)
                            continue
                        boxes = labels[:, 1:5].copy()
                        cls = labels[:, 0:1].copy()
                        zeros = np.zeros((len(labels), 1))
                        targets = np.concatenate([zeros, cls, boxes, kpts], axis=1)
                except:
                    pass
            self.cached_labels.append(targets)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if self._ram_cache is not None and index in self._ram_cache:
            img = self._ram_cache[index]
        else:
            img_path = self.img_files[index]
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            # HWC BGR -> CHW RGB, contiguous uint8
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            if self._ram_cache is not None:
                self._ram_cache[index] = img

        img_tensor = torch.from_numpy(img)

        targets = self.cached_labels[index]
        labels_out = torch.from_numpy(targets).float()
        return img_tensor, labels_out
        
    @staticmethod
    def collate_fn(batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs, 0)
        new_labels = []
        for i, label in enumerate(labels):
            if label.shape[0] > 0:
                l = label.clone()
                l[:, 0] = i
                new_labels.append(l)
        labels = torch.cat(new_labels, 0) if new_labels else torch.zeros((0, 6 + 4*3))
        return imgs, labels