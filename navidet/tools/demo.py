"""
스모크 테스트: forward + Loss + Pose 디코딩이 끝까지 동작하는지 확인 (더미 데이터).

    python -m navidet.tools.demo
"""

import torch

from navidet import Pose6DoFLoss, YOLO6DoF


def random_rotations(B, M, device):
    q, _ = torch.linalg.qr(torch.randn(B, M, 3, 3, device=device))
    det = torch.det(q)
    q[..., 0] *= det.unsqueeze(-1)
    return q


def make_dummy_targets(B, M, img=640, depth_hw=(320, 320), device="cpu"):
    cx = torch.rand(B, M, 2, device=device) * img * 0.6 + img * 0.2
    wh = torch.rand(B, M, 2, device=device) * 120 + 40
    boxes = torch.cat([cx - wh / 2, cx + wh / 2], -1).clamp(0, img)
    labels = torch.randint(0, 3, (B, M, 1), device=device).float()
    rot = random_rotations(B, M, device)
    size = torch.rand(B, M, 3, device=device) * 0.3 + 0.05
    trans = torch.cat([torch.randn(B, M, 2, device=device) * 0.1,
                       torch.rand(B, M, 1, device=device) * 1.5 + 0.5], -1)
    mask = torch.ones(B, M, 1, device=device)
    mask[:, M // 2:] = 0
    K = torch.tensor([[500., 0., img / 2], [0., 500., img / 2], [0., 0., 1.]],
                     device=device).unsqueeze(0).repeat(B, 1, 1)
    gt_depth = torch.rand(B, 1, *depth_hw, device=device) * 2.0 + 0.3
    return {"gt_labels": labels, "gt_bboxes": boxes, "gt_rot": rot,
            "gt_size": size, "gt_trans": trans, "mask_gt": mask,
            "K": K, "img_size": (img, img),
            "gt_depth": gt_depth, "depth_mask": torch.ones_like(gt_depth)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, nc = 2, 3
    model = YOLO6DoF(nc=nc, scale="n", rot_repr="6d").to(device)
    loss_fn = Pose6DoFLoss(nc=nc, rot_repr="6d").to(device)
    x = torch.randn(B, 3, 640, 640, device=device)

    model.train()
    out = model(x)
    print("[train] det keys:", list(out["det"].keys()), "| depth:", tuple(out["depth"].shape))
    targets = make_dummy_targets(B, M=6, device=device)
    loss, items = loss_fn(out, targets)
    print("[train] loss:", {k: round(v, 4) for k, v in items.items()})
    loss.backward()
    print("[train] backward OK")

    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"[eval] det: {tuple(out['det'].shape)}  depth: {tuple(out['depth'].shape)}")


if __name__ == "__main__":
    main()
