"""실습 02 — ResNet18 전이학습 Baseline 학습.

모든 팀은 옵션 없이 이 파일을 한 번 돌려 같은 출발점을 만듭니다.
그 다음 개선 실험에서는 '한 번에 하나의 옵션만' 바꿉니다.

기본 실행 (Baseline):
    python train_baseline.py --data-root data --output-dir outputs/baseline

개선 실험 예시:
    python train_baseline.py --data-root data --augment           --output-dir outputs/exp_aug
    python train_baseline.py --data-root data --augment --rotation-degrees 15 \
        --brightness 0.3 --contrast 0.3 --output-dir outputs/exp_aug_custom
    python train_baseline.py --data-root data --use-class-weights --output-dir outputs/exp_weight
    python train_baseline.py --data-root data --lr 1e-4            --output-dir outputs/exp_lr_control
    python train_baseline.py --data-root data --unfreeze --lr 1e-4 --output-dir outputs/exp_ft
    python train_baseline.py --data-root data --optimizer sgd      --output-dir outputs/exp_sgd
"""

import argparse
import csv
import hashlib
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from battery_dataset import BatteryDataset
from common import (CLASS_NAMES, NUM_CLASSES, evaluate, format_confusion_matrix,
                    get_transforms, resolve_device, save_json, seed_worker,
                    set_seed)
from check_data import validate_data_root


def parse_args():
    ap = argparse.ArgumentParser(description="배터리 열화상 3-class 분류 Baseline")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--output-dir", default="outputs/baseline")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="옵티마이저의 L2 정규화 강도 (옵티마이저 비교 시 기본 0으로 동일)")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42,
                    help="공정한 비교를 위해 42 고정을 권장합니다")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--augment", action="store_true",
                    help="학습 데이터에 crop/flip/rotation/밝기 변화를 추가")
    ap.add_argument("--crop-scale-min", type=float, default=0.85,
                    help="RandomResizedCrop 최소 영역 비율 (기본 0.85, 범위 0 초과~1 이하)")
    ap.add_argument("--flip-prob", type=float, default=0.5,
                    help="좌우 반전 확률 (기본 0.5, 범위 0~1)")
    ap.add_argument("--rotation-degrees", type=float, default=10.0,
                    help="무작위 회전 최대 각도 (기본 10도, 0 이상)")
    ap.add_argument("--brightness", type=float, default=0.2,
                    help="ColorJitter 밝기 변화 강도 (기본 0.2, 0 이상)")
    ap.add_argument("--contrast", type=float, default=0.2,
                    help="ColorJitter 대비 변화 강도 (기본 0.2, 0 이상)")
    ap.add_argument("--use-class-weights", action="store_true",
                    help="적은 클래스의 오답에 더 큰 손실을 부여")
    ap.add_argument("--unfreeze", action="store_true",
                    help="backbone 까지 전체 미세조정 (--lr 1e-4 정도를 함께 쓰세요)")
    ap.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw",
                    help="가중치를 어떤 규칙으로 업데이트할지 선택 (기본 adamw)")
    args = ap.parse_args()
    if not 0.0 < args.crop_scale_min <= 1.0:
        ap.error("--crop-scale-min은 0보다 크고 1 이하여야 합니다.")
    if not 0.0 <= args.flip_prob <= 1.0:
        ap.error("--flip-prob은 0 이상 1 이하여야 합니다.")
    if args.rotation_degrees < 0.0:
        ap.error("--rotation-degrees는 0 이상이어야 합니다.")
    if args.brightness < 0.0:
        ap.error("--brightness는 0 이상이어야 합니다.")
    if args.contrast < 0.0:
        ap.error("--contrast는 0 이상이어야 합니다.")
    return args


def build_model(unfreeze: bool, device, pretrained: bool = True):
    """ImageNet 사전학습 ResNet18 을 불러와 마지막 분류층만 3-class 로 교체합니다."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if not unfreeze:
        # backbone 고정: 사전학습된 특징 추출기를 그대로 쓰고 새 head 만 학습합니다.
        for param in model.parameters():
            param.requires_grad = False

    # fc 를 새로 만들면 requires_grad=True 인 새 파라미터가 됩니다.
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0):
    """--optimizer 옵션에 따라 옵티마이저를 만듭니다.

    이 프로젝트는 backbone을 고정한 채 head(Linear 하나)만 학습하는 경우가 기본이라
    옵티마이저 차이가 CNN을 처음부터 학습할 때만큼 극적이지는 않습니다. 그래도 수렴
    속도와 최종 점수가 조금씩 달라지므로 개선 실험의 한 축으로 다뤄볼 수 있습니다.
    """
    if name == "sgd":
        # SGD는 관성(momentum)을 줘야 Adam 계열과 비슷한 속도로 수렴합니다.
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def freeze_bn(module):
    """BatchNorm 레이어를 eval 모드로 고정합니다.

    requires_grad=False 는 가중치 업데이트만 막을 뿐, model.train() 상태에서
    BatchNorm 의 running_mean/running_var 통계는 계속 갱신됩니다 (PyTorch 기본 동작).
    "backbone 고정"이 이름 그대로 동작하려면 BatchNorm 도 eval 모드로 묶어야 합니다.
    """
    if isinstance(module, nn.BatchNorm2d):
        module.eval()


def save_learning_curves(history: list[dict], output_path: Path) -> None:
    """epoch별 학습/검증 변화를 한눈에 볼 수 있는 그래프를 저장합니다."""
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="train loss")
    axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label="validation loss")
    axes[0].set_title("Loss by epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, [row["val_accuracy"] for row in history], marker="o", label="validation accuracy")
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], marker="o", label="validation macro F1")
    axes[1].set_title("Validation metrics by epoch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_figure(cm: list[list[int]], output_path: Path) -> None:
    """행=실제, 열=예측인 혼동행렬을 이미지로 저장합니다."""
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    image = ax.imshow(cm, cmap="Blues")
    # Colab 기본 폰트에서도 깨지지 않도록 그래프 축은 영문으로 표시합니다.
    plot_labels = ["0 (early)", "1 (middle)", "2 (late)"]
    ax.set_xticks(range(NUM_CLASSES), plot_labels)
    ax.set_yticks(range(NUM_CLASSES), plot_labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Public validation confusion matrix")

    threshold = max(max(row) for row in cm) / 2
    for true_idx, row in enumerate(cm):
        for pred_idx, value in enumerate(row):
            ax.text(pred_idx, true_idx, value, ha="center", va="center",
                    color="white" if value > threshold else "black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def current_code_state() -> dict:
    """커밋과 로컬 수정 내용을 함께 기록해 실행 코드를 재현할 수 있게 합니다."""
    repo_dir = Path(__file__).resolve().parent
    state = {
        "git_commit": None,
        "git_dirty": None,
        "git_status": None,
        "git_diff": None,
        "source_sha256": {},
    }
    try:
        state["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, check=True, capture_output=True, text=True,
        ).stdout.strip()
        state["git_status"] = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_dir, check=True, capture_output=True, text=True,
        ).stdout
        state["git_diff"] = subprocess.run(
            ["git", "diff", "--no-ext-diff", "HEAD"],
            cwd=repo_dir, check=True, capture_output=True, text=True,
        ).stdout
        state["git_dirty"] = bool(state["git_status"].strip())
    except (OSError, subprocess.CalledProcessError):
        pass

    for name in ["train_baseline.py", "common.py", "battery_dataset.py", "check_data.py"]:
        path = repo_dir / name
        if path.is_file():
            state["source_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return state


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.unfreeze and args.lr > 5e-4:
        print(f"[주의] --unfreeze 에 lr={args.lr} 는 큽니다. 1e-4 정도를 권장합니다.")

    if args.optimizer == "sgd" and args.lr <= 5e-4:
        print(f"[주의] --optimizer sgd 에 lr={args.lr} 는 작습니다. "
              f"SGD는 보통 adam 계열보다 큰 lr(예: 1e-2)이 필요합니다.")

    # ---------- 데이터 ----------
    problems = validate_data_root(Path(args.data_root))
    if problems:
        raise RuntimeError(
            "데이터 검사를 통과하지 못했습니다. 먼저 check_data.py 결과를 해결하세요.\n"
            + "\n".join(f"- {msg}" for msg in problems)
        )

    train_set = BatteryDataset(
        args.data_root, "train",
        transform=get_transforms(
            args.image_size,
            train=True,
            augment=args.augment,
            crop_scale_min=args.crop_scale_min,
            flip_prob=args.flip_prob,
            rotation_degrees=args.rotation_degrees,
            brightness=args.brightness,
            contrast=args.contrast,
        ))
    val_set = BatteryDataset(
        args.data_root, "public_val",
        transform=get_transforms(args.image_size, train=False))

    loader_generator = torch.Generator().manual_seed(args.seed)
    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_pin_memory, drop_last=False,
        generator=loader_generator, worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=use_pin_memory,
                            worker_init_fn=seed_worker)

    counts = train_set.class_counts()
    print(f"Training {len(train_set)}장  {dict(zip(CLASS_NAMES, counts))}")
    print(f"Public Validation {len(val_set)}장")

    # ---------- 모델 / 손실 / 최적화 ----------
    model = build_model(args.unfreeze, device)

    if args.use_class_weights:
        # 적은 클래스일수록 큰 가중치. 전체수 / (클래스수 * 해당클래스수)
        total = sum(counts)
        weights = torch.tensor(
            [total / (NUM_CLASSES * c) if c > 0 else 0.0 for c in counts],
            dtype=torch.float32, device=device)
        print(f"class weights = {[round(w, 3) for w in weights.tolist()]}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(args.optimizer, trainable, args.lr, args.weight_decay)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except (AttributeError, TypeError):  # PyTorch 2.1 호환
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    n_train = sum(p.numel() for p in trainable)
    print(f"학습 대상 파라미터 {n_train:,}개 "
          f"({'전체 미세조정' if args.unfreeze else 'head 만 학습'})  "
          f"optimizer={args.optimizer}\n")

    # ---------- 학습 루프 ----------
    history_path = out_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"])

    best_f1, best_report = -1.0, None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        if not args.unfreeze:
            model.apply(freeze_bn)  # backbone 고정 시 BatchNorm 통계도 함께 고정
        running, seen, t0 = 0.0, 0, time.time()

        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(images)          # (1) 예측: 클래스별 점수 3개
                loss = criterion(logits, targets)  # (2) 정답과의 차이 측정

            scaler.scale(loss).backward()       # (3) 역전파: 각 가중치의 책임 계산
            scaler.step(optimizer)              # (4) 업데이트: 가중치 수정
            scaler.update()

            running += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running / seen
        report = evaluate(model, val_loader, device, criterion)

        print(f"epoch {epoch}/{args.epochs}  "
              f"train_loss {train_loss:.4f}  val_loss {report['loss']:.4f}  "
              f"val_acc {report['accuracy']:.4f}  val_macroF1 {report['macro_f1']:.4f}  "
              f"({time.time() - t0:.0f}s)")

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, round(train_loss, 6),
                                    round(report["loss"], 6),
                                    round(report["accuracy"], 6),
                                    round(report["macro_f1"], 6)])

        # CSV와 그래프가 정확히 같은 값을 사용하도록 한 곳에 함께 기록합니다.
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": report["loss"],
            "val_accuracy": report["accuracy"],
            "val_macro_f1": report["macro_f1"],
        })

        # 최고 Macro F1 시점만 저장합니다. 마지막 epoch 가 최고라는 보장은 없습니다.
        if report["macro_f1"] > best_f1:
            best_f1 = report["macro_f1"]
            best_report = {**report, "best_epoch": epoch}
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "macro_f1": best_f1,
                "args": vars(args),
            }, out_dir / "best_model.pt")

    # ---------- 결과 정리 ----------
    save_json(best_report, out_dir / "validation_report.json")
    code_state = current_code_state()
    run_config = {
        **vars(args),
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": getattr(__import__("torchvision"), "__version__", "unknown"),
        **code_state,
    }
    save_json(run_config, out_dir / "run_config.json")
    save_learning_curves(history, out_dir / "learning_curves.png")
    save_confusion_matrix_figure(
        best_report["confusion_matrix"], out_dir / "confusion_matrix.png")

    print("\n" + "=" * 60)
    print(f"최고 Public Validation Macro F1: {best_f1:.4f} (epoch {best_report['best_epoch']})")
    print("Class F1: " + "  ".join(
        f"{n} {v:.4f}" for n, v in zip(CLASS_NAMES, best_report["class_f1"])))
    print("\n혼동행렬 (행=실제, 열=예측)")
    print(format_confusion_matrix(best_report["confusion_matrix"]))
    print("=" * 60)
    print(f"저장 위치: {out_dir.resolve()}")
    print("그래프: learning_curves.png, confusion_matrix.png")


if __name__ == "__main__":
    main()
