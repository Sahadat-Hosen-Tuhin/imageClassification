import os
import shutil
import random

PROCESSED_DIRS = [
    "data/processed/restored",
    "data/processed/frames"
]

CLASS_BASE = "data/classification"
TRAIN_RATIO = 0.8


def collect_all_images():
    all_images = []

    for folder in PROCESSED_DIRS:
        if not os.path.exists(folder):
            continue

        for img in os.listdir(folder):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                all_images.append(os.path.join(folder, img))

    return all_images


def prepare_dataset():
    images = collect_all_images()

    if len(images) == 0:
        print("❌ No processed images found in any folder")
        return

    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for split_name, img_list in [("train", train_imgs), ("val", val_imgs)]:
        for idx, img_path in enumerate(img_list):
            class_name = "effective" if idx % 2 == 0 else "ineffective"

            dest_dir = os.path.join(CLASS_BASE, split_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)

            shutil.copy(
                img_path,
                os.path.join(dest_dir, os.path.basename(img_path))
            )

    print("Classification dataset ready")


if __name__ == "__main__":
    prepare_dataset()
