"""실습 01 — 학습 전에 데이터가 깨지지 않았는지 검사합니다.

왜 먼저 돌리는가:
    모델 코드가 완벽해도 이미지와 라벨의 대응이 어긋나 있으면
    학습은 조용히 돌아가고 점수만 이상하게 나옵니다.
    특히 experiment_id 누수는 Validation 점수를 실제보다 훨씬 높게 만들어
    '잘 되는 줄 알았는데 비공개 Test에서 무너지는' 대표 원인입니다.

사용법:
    python check_data.py --data-root data
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

SPLITS_WITH_LABELS = ["train", "public_val"]
SPLIT_TEST = "private_test"
VALID_TARGETS = {0, 1, 2}


def check_split(data_root: Path, split: str, has_labels: bool):
    """한 분할을 검사하고 (문제 목록, 요약 정보)를 돌려줍니다."""
    problems = []
    split_dir = data_root / split
    image_dir = split_dir / "images"

    if not image_dir.is_dir():
        return [f"[{split}] 이미지 폴더 없음: {image_dir}"], None

    files = {p.name for p in image_dir.glob("*.png")}
    info = {"split": split, "n_images": len(files), "experiment_ids": set()}

    if not has_labels:
        print(f"[{split}] 이미지 {len(files)}장 (정답 비공개)")
        return problems, info

    label_path = split_dir / "labels.csv"
    if not label_path.is_file():
        return [f"[{split}] labels.csv 없음: {label_path}"], info

    df = pd.read_csv(label_path)

    for col in ["image_name", "target"]:
        if col not in df.columns:
            problems.append(f"[{split}] labels.csv 에 '{col}' 열이 없습니다")
    if problems:
        return problems, info

    # 1) 라벨에는 있는데 파일이 없는 경우
    listed = set(df["image_name"])
    missing = listed - files
    if missing:
        problems.append(
            f"[{split}] labels.csv 에 있으나 파일이 없는 이미지 {len(missing)}장 "
            f"(예: {sorted(missing)[:3]})"
        )

    # 2) 파일은 있는데 라벨이 없는 경우 (학습에서 그냥 무시되어 티가 안 납니다)
    orphan = files - listed
    if orphan:
        problems.append(
            f"[{split}] 파일은 있으나 labels.csv 에 없는 이미지 {len(orphan)}장 "
            f"(예: {sorted(orphan)[:3]})"
        )

    # 3) 파일명 중복
    dup = df["image_name"].duplicated().sum()
    if dup:
        problems.append(f"[{split}] labels.csv 에 중복된 image_name {dup}건")

    # 4) target 값 범위
    bad = set(df["target"].unique()) - VALID_TARGETS
    if bad:
        problems.append(f"[{split}] target 에 허용되지 않은 값: {sorted(bad)}")

    counts = Counter(df["target"])
    info["n_rows"] = len(df)
    info["class_counts"] = {int(k): int(counts.get(k, 0)) for k in sorted(VALID_TARGETS)}
    if "experiment_id" in df.columns:
        info["experiment_ids"] = set(df["experiment_id"].unique())

    total = len(df)
    dist = "  ".join(
        f"{name} {info['class_counts'][i]:>6} ({info['class_counts'][i]/total*100:5.1f}%)"
        for i, name in enumerate(["초기", "중기", "후기"])
    )
    print(f"[{split}] 이미지 {len(files)}장 / 라벨 {total}행")
    print(f"         {dist}")
    print(f"         experiment_id {len(info['experiment_ids'])}개")

    return problems, info


def check_leakage(infos):
    """분할 사이에 같은 experiment_id 가 섞였는지 확인합니다.

    같은 실험의 연속 프레임은 파일명이 달라도 장면이 거의 같습니다.
    이 프레임들이 train 과 public_val 에 나뉘어 들어가면 모델은
    '새로운 실험을 이해한 것'이 아니라 '본 장면을 기억한 것'이 됩니다.
    """
    problems = []
    named = {i["split"]: i["experiment_ids"] for i in infos if i and i["experiment_ids"]}
    keys = sorted(named)
    for a_idx in range(len(keys)):
        for b_idx in range(a_idx + 1, len(keys)):
            a, b = keys[a_idx], keys[b_idx]
            overlap = named[a] & named[b]
            if overlap:
                problems.append(
                    f"[누수] {a} 와 {b} 가 experiment_id {len(overlap)}개를 공유합니다 "
                    f"(예: {sorted(overlap)[:3]})"
                )
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        print(f"데이터 폴더가 없습니다: {data_root.resolve()}")
        sys.exit(1)

    print(f"데이터 검사 시작: {data_root.resolve()}\n" + "-" * 60)

    problems, infos = [], []
    for split in SPLITS_WITH_LABELS:
        p, info = check_split(data_root, split, has_labels=True)
        problems += p
        infos.append(info)

    p, info = check_split(data_root, SPLIT_TEST, has_labels=False)
    problems += p
    infos.append(info)

    problems += check_leakage([i for i in infos if i])

    print("-" * 60)
    if problems:
        print(f"문제 {len(problems)}건이 발견되었습니다. 학습 전에 해결하세요.\n")
        for msg in problems:
            print("  - " + msg)
        sys.exit(1)

    print("문제 없음. train_baseline.py 를 실행해도 좋습니다.")


if __name__ == "__main__":
    main()
