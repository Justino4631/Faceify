import os, shutil, random

source = "edited_images"
dest = "dataset"
splits = {"train": 0.8, "val": 0.1, "test": 0.1}

classes = os.listdir(source)

for cls in classes:
    imgs = os.listdir(os.path.join(source, cls))
    random.shuffle(imgs)

    n = len(imgs)
    t = int(n * splits["train"])
    v = int(n * splits["val"])

    split_imgs = {
        "train": imgs[:t],
        "val": imgs[t:t+v],
        "test": imgs[t+v:]
    }

    for split, files in split_imgs.items():
        os.makedirs(f"{dest}/{split}/{cls}", exist_ok=True)
        for f in files:
            shutil.copy(
                f"{source}/{cls}/{f}",
                f"{dest}/{split}/{cls}/{f}"
            )
