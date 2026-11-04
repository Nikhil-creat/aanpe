"""AI4Mars data pipeline. Column names (image, label_mask, rover_mask, range_mask) are
assumed from the HF card; adjust COLS if the schema differs."""
import numpy as np, cv2, torch, albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

IGNORE, N_CLS, SIZE = 255, 4, 512
COLS = dict(img="image", lab="label_mask", rov="rover_mask", rng="range_mask")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def build_tf(train):
    t = [A.Resize(SIZE, SIZE, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST)]
    if train:
        t += [A.HorizontalFlip(), A.VerticalFlip(), A.RandomBrightnessContrast(),
              A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(16, 48),
                              hole_width_range=(16, 48), fill=0, fill_mask=IGNORE)]
    return A.Compose(t + [A.Normalize(MEAN, STD), ToTensorV2()])

def clean_mask(row):
    lab = np.array(row[COLS["lab"]].convert("L"), dtype=np.uint8)
    lab[lab >= N_CLS] = IGNORE
    for k in ("rov", "rng"):  # nonzero = excluded region (rover body, range > 30 m)
        if row.get(COLS[k]) is not None:
            m = np.array(row[COLS[k]].convert("L").resize(lab.shape[::-1], 0))
            lab[m > 0] = IGNORE
    return lab

class Mars(Dataset):
    def __init__(self, split, train=False):
        self.ds, self.tf = load_dataset("hassanjbara/AI4MARS", split=split), build_tf(train)
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        row = self.ds[i]
        img = np.array(row[COLS["img"]].convert("RGB"))
        lab = clean_mask(row)
        if lab.shape[:2] != img.shape[:2]:
            lab = cv2.resize(lab, img.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        o = self.tf(image=img, mask=lab)
        return o["image"], o["mask"].long()

def class_weights(ds, limit=500):
    c = np.zeros(N_CLS)
    for i in range(min(limit, len(ds))):
        m = clean_mask(ds.ds[i]); c += np.bincount(m[m != IGNORE], minlength=N_CLS)[:N_CLS]
    return torch.tensor(1.0 - c / max(c.sum(), 1), dtype=torch.float32)  # Weight_c = 1 - Count_c/Total

def loaders(bs=8, workers=4):
    kw = dict(num_workers=workers, pin_memory=True, prefetch_factor=2, persistent_workers=workers > 0)
    tr = Mars("train", True)
    return (DataLoader(tr, bs, shuffle=True, drop_last=True, **kw),
            {s: DataLoader(Mars(s), bs, **kw) for s in ("test_min1", "test_min3")}, class_weights(tr))

if __name__ == "__main__":
    tr, tests, w = loaders(2, 0); x, y = next(iter(tr)); print(x.shape, y.shape, w)
