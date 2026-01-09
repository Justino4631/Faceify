import os
import re
import shutil

source = "screenshots"
dest = "dataset"
splits = {"train": 0.65, "val": 0.2, "test": 0.15}

classes = os.listdir(source)

for cls in classes:
    cls_path = os.path.join(source, cls)

    imgs = sorted(
        os.listdir(cls_path),
        key=lambda x: int(re.search(r'(\d+)(?=\.\w+$)', x).group())
    )

    n = len(imgs)
    t = int(n * splits["train"])
    v = int(n * splits["val"])

    split_imgs = {
        "train": imgs[:t],
        "val": imgs[t:t+v],
        "test": imgs[t+v:]
    }

    for split, files in split_imgs.items():
        os.makedirs(os.path.join(dest, split, cls), exist_ok=True)
        for f in files:
            shutil.copy(
                os.path.join(cls_path, f),
                os.path.join(dest, split, cls, f)
            )
