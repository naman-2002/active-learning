# active_learning_streamlit.py
import streamlit as st
from pathlib import Path
import os
import random
import pandas as pd
import numpy as np
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from PIL import Image
from pathlib import Path

# ---------------- USER CONFIG ----------------
IMAGE_DIR = Path(r"C:\Users\2002n\OneDrive\Desktop\Fast-API\NEU-DET_1\train\images")   # <-- set your ImageFolder root
LABELS_CSV = Path(r"C:\Users\2002n\OneDrive\Desktop\Fast-API\labels.csv")
WARM_FRAC = 0.30
ACQ_SIZE = 10            # 5 / 10 / 15 as you like
MAX_ITERS = 20
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_PER_ITER = 2    # keep small for fast interactivity; increase if you want accuracy
WARM_EPOCHS = 3
SEED = 42
# ---------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

st.set_page_config(layout="wide", page_title="Interactive Active Learning")

st.title("Streamline Interactive Active Learning")

# ---------- transforms ----------
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
infer_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

import random
import streamlit as st

def safe_rerun():
    """
    Try to call streamlit's experimental rerun. If not available in this Streamlit
    version, fall back to changing the query params (forces a rerun) then stop.
    """
    try:
        # Preferred API (may not exist on some versions)
        st.experimental_rerun()
    except Exception:
        try:
            # Fallback: change query params so the app reloads in the browser
            params = st.query_params()
            params["_refresh"] = str(random.random())
            st.query_params(**params)
            # Stop current run so the new run starts
            st.stop()
        except Exception:
            # Last resort: tell the user to manually refresh
            st.warning("Please refresh the page to continue (fallback).")
            st.stop()


# ---------- small model ----------
# ...existing code...
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__()
        # Very small / efficient backbone: low channel counts + strided convs
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),  # 224 -> 112
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 112 -> 56
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 56 -> 28
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.AdaptiveAvgPool2d(1)  # global pooling -> (N, 32, 1, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(x)
        return self.fc(x)
# ...existing code...

# ...existing code...
def build_resnet18(num_classes):
    m = models.resnet18(pretrained=True)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# New: EfficientNet-B0 (good accuracy / speed) and MobileNetV3 (fast)
def build_efficientnet_b0(num_classes, pretrained=True):
    try:
        m = models.efficientnet_b0(pretrained=pretrained)
    except TypeError:
        # older torchvision API
        m = models.efficientnet_b0(pretrained=pretrained)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m

def build_mobilenet_v3_large(num_classes, pretrained=True):
    m = models.mobilenet_v3_large(pretrained=pretrained)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m

# helper to optionally freeze backbone for initial warm epochs
def maybe_freeze_backbone(model, freeze=True):
    if not freeze:
        return
    for name, p in model.named_parameters():
        if 'fc' in name or 'classifier' in name or 'head' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
# ...existing code...

# ---------- ImageFolder wrapper that returns filename ----------
class ImageFolderWithPaths(datasets.ImageFolder):
    # returns (image, label, relative_path_or_basename)
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        # return basename to keep labels.csv simple; if non-unique use relative path
        return sample, target, os.path.basename(path)

# ---------- helper I/O ----------
def load_labels_csv(path: str) -> Dict[str,int]:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    mapping = {}
    for _, r in df.iterrows():
        mapping[str(r['filename'])] = int(r['label'])
    return mapping

def append_labels_to_csv(path: str, rows: List[List]):
    df = pd.DataFrame(rows, columns=['filename','label'])
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)

def write_warm_start_labels(path: str, imagefolder: ImageFolderWithPaths, idxs: List[int]):
    rows = []
    for i in idxs:
        p, cls = imagefolder.samples[i]
        rows.append([os.path.basename(p), int(cls)])
    pd.DataFrame(rows, columns=['filename','label']).to_csv(path, index=False)

# ---------- dataset index builders ----------
def build_fname2idx_map(imagefolder: ImageFolderWithPaths) -> Dict[str,int]:
    mapping = {}
    for i, (p, cls) in enumerate(imagefolder.samples):
        mapping[os.path.basename(p)] = i
    return mapping

def build_subsets_from_labels(imagefolder: ImageFolderWithPaths, labels_map: Dict[str,int]):
    fname2idx = build_fname2idx_map(imagefolder)
    labeled_idx = []
    unlabeled_idx = []
    for fname, idx in fname2idx.items():
        if fname in labels_map:
            labeled_idx.append(idx)
        else:
            unlabeled_idx.append(idx)
    return labeled_idx, unlabeled_idx

if 'metrics' not in st.session_state:
    st.session_state.metrics = []   # list of dicts: {'iter': int, 'loss': float, 'acc': float}

# ...existing code...
def train_model(model, loader, device, epochs=3, lr=1e-3):
    """
    Trains model and returns the model plus a history dict with per-epoch losses/accs.
    """
    model.to(device)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()
    epoch_losses = []
    epoch_accs = []
    for ep in range(epochs):
        running = 0.0; total=0; corr=0
        for batch in loader:
            if len(batch)==3:
                imgs, labels, _ = batch
            else:
                imgs, labels = batch
            imgs = imgs.to(device); labels = labels.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward(); opt.step()
            running += loss.item()*imgs.size(0)
            preds = logits.argmax(1)
            corr += (preds==labels).sum().item()
            total += labels.size(0)
        if total:
            avg_loss = running/total
            acc = 100.*corr/total
            st.write(f"  Epoch {ep+1}/{epochs} - loss: {avg_loss:.4f} - acc: {acc:.2f}%")
        else:
            avg_loss = 0.0; acc = 0.0
        epoch_losses.append(avg_loss)
        epoch_accs.append(acc)
    history = {'epoch_losses': epoch_losses, 'epoch_accs': epoch_accs}
    return model, history
# ...existing code...

def run_one_iteration():
    # Train a model on current labeled set, score unlabeled, pick ACQ_SIZE, set session_state.selected_list
    if len(unlabeled_idx) == 0:
        st.info("No unlabeled samples left.")
        st.session_state.running = False
        return

    # Build datasets / loaders
    labeled_subset = Subset(full_dataset_train, labeled_idx)
    unlabeled_subset = Subset(full_dataset_infer, unlabeled_idx)

    if len(labeled_subset) == 0:
        st.error("No labeled data to train on. Add warm-start or create labels.csv manually.")
        st.session_state.running = False
        return

    labeled_loader = DataLoader(labeled_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    unlabeled_loader = DataLoader(unlabeled_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # create & train model
    st.write("Training model on labeled set...")
    model = SimpleCNN(num_classes).to(DEVICE)  # lightweight model for interactivity
    # model = build_resnet18(num_classes).to(DEVICE)  # alternative - stronger but slower
    epochs_to_run = WARM_EPOCHS if st.session_state.iter == 0 else EPOCHS_PER_ITER
    with st.spinner("Training..."):
        model, history = train_model(model, labeled_loader, device=DEVICE, epochs=epochs_to_run, lr=1e-3)

    # record iteration-level metric (use last epoch values)
    last_loss = history['epoch_losses'][-1] if history['epoch_losses'] else 0.0
    last_acc = history['epoch_accs'][-1] if history['epoch_accs'] else 0.0
    st.session_state.metrics.append({'iter': st.session_state.iter, 'loss': last_loss, 'acc': last_acc})

    # score unlabeled

    st.write("Scoring unlabeled pool by entropy...")
    fnames, entropies = score_unlabeled_by_entropy(model, unlabeled_loader, device=DEVICE)

    # top-k selection (highest entropy)
    k = min(ACQ_SIZE, len(fnames))
    topk = np.argsort(-entropies)[:k]
    selected = [fnames[i] for i in topk]
    st.session_state.selected_list = selected
    st.session_state.selected_idx = 0

    # save to a to_label CSV for record (optional)
    existing = sorted(Path('.').glob('to_label_iteration_*.csv'))
    next_iter = len(existing) + 1
    recs = []
    for s in selected:
        # find full path
        fullpath = None
        for p, _ in full_dataset_train.samples:
            if os.path.basename(p) == s:
                fullpath = p; break
        recs.append({'filename': s, 'fullpath': fullpath})
    if recs:
        pd.DataFrame(recs).to_csv(f"to_label_iteration_{next_iter}.csv", index=False)

    st.success(f"Selected {len(selected)} uncertain images for labeling (iteration saved as to_label_iteration_{next_iter}.csv).")

# # ---------- training & scoring ----------
# def train_model(model, loader, device, epochs=3, lr=1e-3):
#     model.to(device)
#     opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#     criterion = nn.CrossEntropyLoss()
#     model.train()
#     for ep in range(epochs):
#         running = 0.0; total=0; corr=0
#         for batch in loader:
#             if len(batch)==3:
#                 imgs, labels, _ = batch
#             else:
#                 imgs, labels = batch
#             imgs = imgs.to(device); labels = labels.to(device)
#             opt.zero_grad()
#             logits = model(imgs)
#             loss = criterion(logits, labels)
#             loss.backward(); opt.step()
#             running += loss.item()*imgs.size(0)
#             preds = logits.argmax(1)
#             corr += (preds==labels).sum().item()
#             total += labels.size(0)
#         if total:
#             st.write(f"  Epoch {ep+1}/{epochs} - loss: {running/total:.4f} - acc: {100*corr/total:.2f}%")
#     return model

@torch.no_grad()
def score_unlabeled_by_entropy(model, loader, device):
    model.eval()
    scores = []
    fnames = []
    for batch in loader:
        if len(batch)==3:
            imgs, _, names = batch
        else:
            imgs, _ = batch
            names = None
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        ent = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
        scores.extend(ent.cpu().numpy().tolist())
        if names is not None:
            fnames.extend(names)
    if not fnames:
        # fallback: reconstruct from subset dataset
        try:
            ds = loader.dataset
            if isinstance(ds, Subset):
                base = ds.dataset
                idxs = ds.indices
                fnames = [os.path.basename(base.samples[i][0]) for i in idxs]
            else:
                fnames = [os.path.basename(p[0]) for p in ds.samples]
        except:
            fnames = [None]*len(scores)
    return fnames, np.array(scores)

# ---------- Streamlit session-state keys ----------
if 'iter' not in st.session_state:
    st.session_state.iter = 0
if 'selected_list' not in st.session_state:
    st.session_state.selected_list = []     # basenames selected for current iteration
if 'selected_idx' not in st.session_state:
    st.session_state.selected_idx = 0      # pointer to index in selected_list
if 'running' not in st.session_state:
    st.session_state.running = False       # if true, active loop is running

# ---------- UI: control panel ----------
col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("Start / Resume"):
        st.session_state.running = True
with col2:
    if st.button("Pause"):
        st.session_state.running = False
with col3:
    if st.button("Reset (clear labels.csv and state)"):
        if os.path.exists(LABELS_CSV):
            os.remove(LABELS_CSV)
        st.session_state.iter = 0
        st.session_state.selected_list = []
        st.session_state.selected_idx = 0
        st.session_state.running = False
        safe_rerun()

st.markdown("---")

# ---------- load datasets ----------
if not Path(IMAGE_DIR).exists():
    st.error(f"IMAGE_DIR '{IMAGE_DIR}' not found. Fix config at top of script.")
    st.stop()

full_dataset_train = ImageFolderWithPaths(IMAGE_DIR, transform=train_transform)
full_dataset_infer = ImageFolderWithPaths(IMAGE_DIR, transform=infer_transform)
num_classes = len(full_dataset_train.classes)
st.write(f"Dataset: {len(full_dataset_train)} images, {num_classes} classes.")

# ---------- prepare labels (warm start if needed) ----------
labels_map = load_labels_csv(LABELS_CSV)
if len(labels_map) == 0:
    # Stratified / balanced warm-start: pick initial WARM_FRAC images
    n_warm = max(1, int(len(full_dataset_train) * WARM_FRAC))
    num_classes = len(full_dataset_train.classes)
    per_class = n_warm // num_classes

    rng = random.Random(SEED)
    # build class -> indices map
    cls_to_idxs = {c: [] for c in range(num_classes)}
    for i, (_, c) in enumerate(full_dataset_train.samples):
        cls_to_idxs[c].append(i)

    selected = []
    # sample up to `per_class` examples per class (no replacement)
    for c, idxs in cls_to_idxs.items():
        if not idxs:
            continue
        if len(idxs) <= per_class:
            selected.extend(idxs)
        else:
            selected.extend(rng.sample(idxs, per_class))

    # if we still need more (due to rounding or empty classes), fill from remaining pool
    if len(selected) < n_warm:
        remaining = list(set(range(len(full_dataset_train))) - set(selected))
        need = n_warm - len(selected)
        if len(remaining) <= need:
            selected.extend(remaining)
        else:
            selected.extend(rng.sample(remaining, need))

    # final safety: trim to exactly n_warm
    warm_idx = selected[:n_warm]

    # persist warm-start labels (basename + class id)
    write_warm_start_labels(LABELS_CSV, full_dataset_train, warm_idx)
    labels_map = load_labels_csv(LABELS_CSV)
    st.success(f"Warm-start (stratified): wrote {len(warm_idx)} labels to {LABELS_CSV}")

# ...existing code...

labeled_idx, unlabeled_idx = build_subsets_from_labels(full_dataset_train, labels_map)
st.write(f"Labeled: {len(labeled_idx)}  |  Unlabeled: {len(unlabeled_idx)}  | Iteration: {st.session_state.iter}/{MAX_ITERS}")


# ---------- orchestrator ----------
if st.session_state.running:
    # run iterations until MAX_ITERS or until paused or unlabeled exhausted
    while st.session_state.running and st.session_state.iter < MAX_ITERS:
        # if no current selected list, obtain one
        if not st.session_state.selected_list:
            run_one_iteration()
            if not st.session_state.selected_list:
                break  # nothing to label
        # Now we are in labeling stage for the current selected_list
        st.session_state.running = False  # switch to manual-interaction mode (so app doesn't loop forever); UI will let user resume after each save
        safe_rerun()

# Show metrics plot if we have any iterations run
# Show metrics plot if we have any iterations run
if st.session_state.metrics:
    dfm = pd.DataFrame(st.session_state.metrics).set_index('iter').sort_index()
    st.subheader("Training metrics per iteration")
    # show numeric table
    st.dataframe(dfm.style.format({"loss":"{:.4f}", "acc":"{:.2f}"}))

    # Plot loss and accuracy separately (side-by-side)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Loss per iteration**")
        st.line_chart(dfm['loss'])
    with c2:
        st.markdown("**Accuracy per iteration (%)**")
        st.line_chart(dfm['acc'])

def export_labeled_full_csv(outname="labeled_dataset_full.csv"):
    """
    Export a consolidated labeled CSV that:
      - reads LABELS_CSV (may contain multiple appends / duplicates)
      - keeps the last label entry per filename (most recent human decision)
      - attaches fullpath when available and writes out consolidated CSV
    """
    if not os.path.exists(LABELS_CSV):
        st.warning("No labels.csv found to export.")
        return
    df = pd.read_csv(LABELS_CSV)

    # Ensure expected columns
    if 'filename' not in df.columns or 'label' not in df.columns:
        st.error(f"{LABELS_CSV} missing required columns 'filename','label'.")
        return

    # Keep the last label for each filename (preserve human corrections)
    df_consol = df.copy()
    # preserve original order then drop duplicates keeping last occurrence
    df_consol['__order__'] = np.arange(len(df_consol))
    df_consol = df_consol.drop_duplicates(subset='filename', keep='last').drop(columns='__order__')

    # attach full path if available (match by basename)
    fname2path = {os.path.basename(p): p for p, _ in full_dataset_train.samples}
    df_consol['fullpath'] = df_consol['filename'].map(lambda x: fname2path.get(x, None))

    # save consolidated CSV
    df_consol.to_csv(outname, index=False)
    st.success(f"Exported consolidated labeled dataset to {outname} ({len(df_consol)} rows)")
    return outname

if st.button("Export labeled dataset CSV (with full paths)"):
    out = export_labeled_full_csv()
    if out:
        st.write(f"Saved {out}")
        st.write(pd.read_csv(out).head(20))
# ...existing code...

# ---------- Labeling UI (runs always) ----------
st.header("Labeling panel")

if not st.session_state.selected_list:
    st.info("No active batch to label. Click **Start / Resume** to create/select a batch.")
else:
    idx = st.session_state.selected_idx
    if idx >= len(st.session_state.selected_list):
        st.write("Batch labeling finished for this iteration.")
        # increment iter, clear selected_list, continue if user clicks Resume
        if st.button("Commit iteration and continue to next iteration"):
            st.session_state.iter += 1
            st.session_state.selected_list = []
            st.session_state.selected_idx = 0
            st.session_state.running = True
            safe_rerun()
    else:
        fname = st.session_state.selected_list[idx]
        # locate full path
        fullpath = None
        for p,_ in full_dataset_train.samples:
            if os.path.basename(p) == fname:
                fullpath = p; break

        cols = st.columns([1,2])
        with cols[0]:
            st.subheader(f"File {idx+1}/{len(st.session_state.selected_list)}")
            if fullpath and Path(fullpath).exists():
                im = Image.open(fullpath)
                st.image(im, use_column_width=True)
            else:
                st.warning("Image path not found; showing filename only.")
                st.write(fname)

        with cols[1]:
            st.write("Choose class id (or mark Unsure = -1).")
            # try to show class names if exist
            class_dirs = sorted([p.name for p in Path(IMAGE_DIR).iterdir() if p.is_dir()]) if Path(IMAGE_DIR).exists() else []
            prev_label_df = pd.read_csv(LABELS_CSV) if Path(LABELS_CSV).exists() else pd.DataFrame(columns=['filename','label'])
            prev = prev_label_df[prev_label_df['filename']==fname]
            default_val = int(prev.iloc[-1]['label']) if (not prev.empty) else 0
            if class_dirs:
                options = [f"{i} - {nm}" for i,nm in enumerate(class_dirs)]
                chosen = st.selectbox("Class", options, index=min(default_val, len(options)-1))
                chosen_id = int(chosen.split(" - ")[0])
            else:
                chosen_id = st.number_input("Class id (integer)", min_value=-1, value=default_val, step=1)

            if st.button("Save label for this image"):
                append_labels_to_csv(LABELS_CSV, [[fname, int(chosen_id)]])
                st.success(f"Saved: {fname} -> {chosen_id}")
                st.session_state.selected_idx += 1
                safe_rerun()

            if st.button("Mark as Unsure (-1)"):
                append_labels_to_csv(LABELS_CSV, [[fname, -1]])
                st.success(f"Saved: {fname} -> -1")
                st.session_state.selected_idx += 1
                safe_rerun()

        # show small batch preview below
        st.markdown("---")
        st.write("Batch preview:")
        preview_cols = st.columns(min(6, len(st.session_state.selected_list)))
        for i, f in enumerate(st.session_state.selected_list):
            col = preview_cols[i % len(preview_cols)]
            # thumbnail
            pth = None
            for p,_ in full_dataset_train.samples:
                if os.path.basename(p) == f:
                    pth = p; break
            if pth and Path(pth).exists():
                try:
                    thumb = Image.open(pth)
                    thumb.thumbnail((140,140))
                    col.image(thumb, caption=f"{i+1}: {f}")
                except:
                    col.write(f"{i+1}: {f}")
            else:
                col.write(f"{i+1}: {f}")

st.markdown("---")
st.caption("Workflow: Click Start/Resume → app trains & selects uncertain images → label images one-by-one in the UI (Save) → when batch done, click 'Commit iteration and continue' to proceed to next iteration. Repeat until MAX_ITERS reached.")
