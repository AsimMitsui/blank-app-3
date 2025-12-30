# streamlit_app.py
# EL PV Cell Extractor — Build image+mask dataset for training (Streamlit)
# (This is the app code. If you already have a working app file from previous messages,
#  use that instead; this file is the same dataset-builder variant.)
import io
import math
import json
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------
# Helpers: conversion & IO
# ---------------------------
def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def zip_bytes_from_dict(filedict: Dict[str, bytes]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, data in filedict.items():
            zf.writestr(fname, data)
    bio.seek(0)
    return bio.read()

# ---------------------------
# Geometry & rectification
# ---------------------------
def perspective_warp(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        I = np.eye(3, dtype=np.float32)
        return img_bgr.copy(), I, I
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) != 4:
        I = np.eye(3, dtype=np.float32)
        return img_bgr.copy(), I, I
    pts = approx.reshape(4,2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
    rect = np.array([tl, tr, br, bl], dtype=np.float32)
    widthA = np.linalg.norm(br - bl); widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br); heightB = np.linalg.norm(tl - bl)
    maxW = int(max(1, max(widthA, widthB))); maxH = int(max(1, max(heightA, heightB)))
    if maxW < 50 or maxH < 50:
        I = np.eye(3, dtype=np.float32)
        return img_bgr.copy(), I, I
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    Minv = cv2.getPerspectiveTransform(dst, rect)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH), flags=cv2.INTER_LINEAR)
    return warped, M, Minv

# ---------------------------
# EL-specific normalization + line extraction
# ---------------------------
def normalize_el_gray(img_bgr: np.ndarray, clahe_clip: float = 2.5, tile: int = 8, blur: int = 3) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(tile, tile))
    gray = clahe.apply(gray)
    if blur and blur > 0:
        k = blur if blur % 2 == 1 else blur + 1
        gray = cv2.GaussianBlur(gray, (k,k), 0)
    return gray

def detect_line_maps(gray: np.ndarray, polarity: str, binarize: str, k_v: int, k_h: int) -> Tuple[np.ndarray, np.ndarray]:
    if binarize == "adaptive":
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if polarity == "auto":
        use = 255 - bw if gray.mean() > 127 else bw
    elif polarity == "dark":
        use = 255 - bw
    else:
        use = bw
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, k_v)))
    vert = cv2.dilate(cv2.erode(use, kernel_v, iterations=1), kernel_v, iterations=1)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, k_h), 1))
    horiz = cv2.dilate(cv2.erode(use, kernel_h, iterations=1), kernel_h, iterations=1)
    small = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    vert = cv2.morphologyEx(vert, cv2.MORPH_CLOSE, small)
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE, small)
    return vert, horiz

def project_peaks(line_map: np.ndarray, axis:int=0, min_dist:int=20, min_strength:float=0.12) -> List[int]:
    proj = line_map.sum(axis=axis).astype(np.float32)
    if proj.max() <= 0:
        return []
    p = (proj - proj.min()) / (proj.max() - proj.min())
    peaks = []
    last = -min_dist
    L = len(p)
    for i in range(1, L-1):
        if p[i] > min_strength and p[i] > p[i-1] and p[i] > p[i+1]:
            if i - last >= min_dist:
                peaks.append(i); last = i
    return peaks

def cuts_from_peaks(peaks: List[int], length:int) -> List[int]:
    if len(peaks) < 2:
        return [0, length]
    cuts = [0]
    for i in range(len(peaks)-1):
        cuts.append((peaks[i] + peaks[i+1]) // 2)
    cuts.append(length)
    return sorted(list(dict.fromkeys(cuts)))

def grid_cells_from_maps(warped: np.ndarray, vert_map: np.ndarray, horiz_map: np.ndarray,
                         min_w:int=30, min_h:int=30) -> List[Dict[str,Any]]:
    H,W = vert_map.shape
    xs = project_peaks(vert_map, axis=0, min_dist=max(10, W//40), min_strength=0.12)
    ys = project_peaks(horiz_map, axis=1, min_dist=max(10, H//40), min_strength=0.12)
    xcuts = cuts_from_peaks(xs, W)
    ycuts = cuts_from_peaks(ys, H)
    cells = []
    for r in range(len(ycuts)-1):
        y0,y1 = ycuts[r], ycuts[r+1]
        for c in range(len(xcuts)-1):
            x0,x1 = xcuts[c], xcuts[c+1]
            w,h = x1-x0, y1-y0
            if w >= min_w and h >= min_h:
                crop = warped[y0:y1, x0:x1].copy()
                cells.append({"row": r, "col": c, "bbox_warp": (x0,y0,x1,y1), "image_warp": crop})
    return cells

# ---------------------------
# Manual split helpers for enforcing expected count
# ---------------------------
def factor_pairs(n:int) -> List[Tuple[int,int]]:
    pairs = []
    for r in range(1, int(math.sqrt(n))+1):
        if n % r == 0:
            pairs.append((r, n//r))
    pairs_full = []
    for (r,c) in pairs:
        pairs_full.append((r,c))
        if r != c:
            pairs_full.append((c,r))
    pairs_full = [ (min(a,b), max(a,b)) for a,b in pairs_full ]
    pairs_unique = sorted(list(set(pairs_full)))
    return pairs_unique

def choose_grid_for_count(total:int, warped_shape:Tuple[int,int]) -> Tuple[int,int]:
    H,W = warped_shape
    ar = W / H if H>0 else 1.0
    pairs = factor_pairs(total)
    if pairs:
        best = min(pairs, key=lambda rc: abs((rc[1]/rc[0]) - ar))
        return best
    r = int(math.sqrt(total)); c = int(math.ceil(total / r))
    return (r, c)

def manual_split_warp(warped: np.ndarray, rows:int, cols:int, margin:int=0) -> List[Dict[str,Any]]:
    h,w = warped.shape[:2]
    cells = []
    cell_w = (w - 2*margin) // cols
    cell_h = (h - 2*margin) // rows
    for r in range(rows):
        for c in range(cols):
            x0 = margin + c*cell_w
            y0 = margin + r*cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            crop = warped[y0:y1, x0:x1].copy()
            cells.append({"row": r, "col": c, "bbox_warp": (x0,y0,x1,y1), "image_warp": crop})
    return cells

# ---------------------------
# Mapping to original & mask building
# ---------------------------
def warp_bbox_to_original(bbox_warp: Tuple[int,int,int,int], Minv: np.ndarray, clip_shape: Tuple[int,int]) -> Tuple[int,int,int,int]:
    x0,y0,x1,y1 = bbox_warp
    corners = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.float32).reshape(-1,1,2)
    if Minv is None:
        pts = corners.reshape(-1,2)
    else:
        pts = cv2.perspectiveTransform(corners, Minv).reshape(-1,2)
    xs = pts[:,0]; ys = pts[:,1]
    xi0 = int(max(0, math.floor(xs.min()))); yi0 = int(max(0, math.floor(ys.min())))
    xi1 = int(min(clip_shape[1], math.ceil(xs.max()))); yi1 = int(min(clip_shape[0], math.ceil(ys.max())))
    if xi1 <= xi0 or yi1 <= yi0:
        return xi0, yi0, 0, 0
    return xi0, yi0, xi1 - xi0, yi1 - yi0

def build_mask_for_bbox(warped_gray: np.ndarray, bbox_warp: Tuple[int,int,int,int]) -> np.ndarray:
    x0,y0,x1,y1 = bbox_warp
    crop = warped_gray[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((0,0), dtype=np.uint8)
    _, m = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(1, min(7, (min(crop.shape)//20)|1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return (m>0).astype(np.uint8)

# ---------------------------
# Pipeline per-image
# ---------------------------
def process_module_image(img_orig_bgr: np.ndarray,
                         enforce_expected: bool,
                         expected_total: int,
                         margin: int,
                         polarity: str,
                         binarize: str,
                         k_v: int,
                         k_h: int,
                         min_cell_w: int,
                         min_cell_h: int,
                         do_warp: bool) -> Dict[str, Any]:
    H_orig, W_orig = img_orig_bgr.shape[:2]
    if do_warp:
        warped, M, Minv = perspective_warp(img_orig_bgr)
    else:
        warped = img_orig_bgr.copy(); M = np.eye(3, dtype=np.float32); Minv = np.eye(3, dtype=np.float32)

    warped_gray = normalize_el_gray(warped, blur=3)

    if enforce_expected and expected_total > 0:
        rows, cols = choose_grid_for_count(expected_total, (warped.shape[0], warped.shape[1]))
        cells_warp = manual_split_warp(warped, rows, cols, margin=margin)
        if len(cells_warp) > expected_total:
            cells_warp = cells_warp[:expected_total]
    else:
        vert_map, horiz_map = detect_line_maps(warped_gray, polarity=polarity, binarize=binarize, k_v=k_v, k_h=k_h)
        cells_warp = grid_cells_from_maps(warped, vert_map, horiz_map, min_w=min_cell_w, min_h=min_cell_h)

    warped_gray_for_masks = normalize_el_gray(warped, blur=1)
    outputs = []
    for idx, c in enumerate(cells_warp):
        bbox_w = c["bbox_warp"]
        mask_w = build_mask_for_bbox(warped_gray_for_masks, bbox_w)
        bbox_o = warp_bbox_to_original(bbox_w, Minv, clip_shape=(H_orig, W_orig))
        x_o,y_o,w_o,h_o = bbox_o
        img_orig_crop = None
        if w_o>0 and h_o>0:
            img_orig_crop = img_orig_bgr[y_o:y_o+h_o, x_o:x_o+w_o].copy()
        outputs.append({
            "index": idx,
            "row": c["row"],
            "col": c["col"],
            "bbox_warp": bbox_w,
            "bbox_orig": bbox_o,
            "image_warp": c["image_warp"],
            "image_orig": img_orig_crop,
            "mask_warp": mask_w
        })

    overlay_warp = warped.copy()
    for out in outputs:
        x0,y0,x1,y1 = out["bbox_warp"]
        cv2.rectangle(overlay_warp, (x0,y0), (x1,y1), (0,255,0), 2)

    return {
        "warped": warped,
        "Minv": Minv,
        "outputs": outputs,
        "overlay_warp": overlay_warp
    }

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="EL PV Cell Extractor — Dataset Builder", layout="wide")
st.title("EL PV Cell Extractor — Build image+mask dataset for training")

st.markdown("""
Upload EL PV module images and extract per-cell crops + masks suitable for training segmentation/classification models.
You can either enforce an expected total cell count (e.g., 144) to force a grid split, or let the app auto-detect grid lines.
After detection you can preview and include/exclude cells before exporting a ZIP with images, masks and annotations.json.
""")

# Controls
expected_total = st.number_input("Expected total cells (0 = auto-detect)", min_value=0, value=144, step=1)
enforce_expected = st.checkbox("Enforce expected total (force rows×cols)", value=True)
do_warp = st.checkbox("Try perspective warp (rectify module)", value=True)
polarity = st.selectbox("Line polarity", ["auto","dark","bright"], index=0)
binarize = st.selectbox("Binarization", ["otsu","adaptive"], index=0)
k_v = st.slider("Vertical kernel size", 5, 75, 25)
k_h = st.slider("Horizontal kernel size", 5, 75, 25)
min_cell_w = st.slider("Min cell width (px)", 10, 400, 30)
min_cell_h = st.slider("Min cell height (px)", 10, 400, 30)
margin = st.number_input("Grid margin (px, used when enforcing grid)", min_value=0, max_value=200, value=0)
uploads = st.file_uploader("Upload EL module image(s)", type=["jpg","jpeg","png","tif","tiff","bmp"], accept_multiple_files=True)
run_detect = st.button("Detect cells")
export_zip_btn = st.button("Export selected cells as ZIP")

# Storage for results across reruns: keep in session_state
if "results" not in st.session_state:
    st.session_state["results"] = {}  # key = filename -> result dict including outputs, selection booleans

# Run detection
if run_detect:
    if not uploads:
        st.warning("Upload at least one image first.")
    else:
        st.session_state["results"].clear()
        for upl in uploads:
            img_pil = Image.open(io.BytesIO(upl.read())).convert("RGB")
            img_bgr = pil_to_cv(img_pil)
            res = process_module_image(
                img_bgr,
                enforce_expected=enforce_expected,
                expected_total=int(expected_total),
                margin=int(margin),
                polarity=polarity,
                binarize=binarize,
                k_v=int(k_v),
                k_h=int(k_h),
                min_cell_w=int(min_cell_w),
                min_cell_h=int(min_cell_h),
                do_warp=do_warp
            )
            st.session_state["results"][upl.name] = {
                "orig_name": upl.name,
                "orig_size": img_bgr.shape[:2],
                "warped": res["warped"],
                "overlay_warp": res["overlay_warp"],
                "outputs": res["outputs"],
                "Minv": res["Minv"],
                # initialize inclusion flags (all True)
                "include": [True] * len(res["outputs"])
            }
        st.success("Detection finished for uploaded images.")

# Display detected images & previews
for fname, data in list(st.session_state["results"].items()):
    st.header(f"Image: {fname} — {len(data['outputs'])} detected cells")
    st.image(cv_to_pil(data["overlay_warp"]), caption=f"{fname} — overlay (warped plane)", use_column_width=True)

    cols = st.columns(min(6, max(1, len(data["outputs"]))))
    # show cells with checkboxes
    for i, out in enumerate(data["outputs"]):
        img_show = out["image_orig"] if out["image_orig"] is not None else out["image_warp"]
        if img_show is None:
            continue
        with cols[i % len(cols)]:
            st.image(cv_to_pil(img_show), caption=f"idx {i} r{out['row']} c{out['col']}", use_column_width=True)
            key = f"include_{fname}_{i}"
            checked = st.checkbox("Include", value=data["include"][i] if i < len(data["include"]) else True, key=key)
            data["include"][i] = checked

# Export selected cells into a ZIP for training
if export_zip_btn:
    if not st.session_state["results"]:
        st.warning("No detected results to export. Run detection first.")
    else:
        files: Dict[str, bytes] = {}
        annotations = {"items": []}
        img_idx = 0
        for fname, data in st.session_state["results"].items():
            for i, out in enumerate(data["outputs"]):
                if not data["include"][i]:
                    continue
                base_name = f"{Path(fname).stem}_cell_{img_idx:05d}"
                # original-space image (preferred)
                if out["image_orig"] is not None:
                    pil_img = cv_to_pil(out["image_orig"])
                else:
                    pil_img = cv_to_pil(out["image_warp"])
                b = io.BytesIO(); pil_img.save(b, format="PNG"); files[f"images/{base_name}.png"] = b.getvalue()
                # mask (warp-space) — mapped to crop size? We save the warp mask as provided (same size as warp crop)
                if out["mask_warp"] is not None and out["mask_warp"].size != 0:
                    pil_mask = Image.fromarray((out["mask_warp"]*255).astype("uint8"))
                    b = io.BytesIO(); pil_mask.save(b, format="PNG"); files[f"masks/{base_name}_mask.png"] = b.getvalue()
                else:
                    # create empty mask matching image size
                    img_arr = np.array(pil_img)
                    h,w = img_arr.shape[:2]
                    empty = np.zeros((h,w), dtype=np.uint8)
                    pil_mask = Image.fromarray(empty)
                    b = io.BytesIO(); pil_mask.save(b, format="PNG"); files[f"masks/{base_name}_mask.png"] = b.getvalue()
                annotations["items"].append({
                    "file_image": f"images/{base_name}.png",
                    "file_mask": f"masks/{base_name}_mask.png",
                    "source_module": fname,
                    "index_in_module": i,
                    "row": out["row"],
                    "col": out["col"],
                    "bbox_orig": out["bbox_orig"],
                    "bbox_warp": out["bbox_warp"]
                })
                img_idx += 1

        files["annotations.json"] = json.dumps(annotations, indent=2).encode("utf-8")
        files["README.txt"] = b"Dataset exported by EL PV Cell Extractor. Images under images/, masks under masks/, annotations.json lists entries."
        zipb = zip_bytes_from_dict(files)
        st.success(f"Export ready: {img_idx} images")
        st.download_button("Download dataset ZIP", data=zipb, file_name="el_pv_cells_dataset.zip", mime="application/zip")

st.markdown("---")
st.caption("Tips: For regular modules with known total cell count (e.g., 144), enable enforcement so the app will split the rectified module into an appropriate rows×cols grid (best-matching aspect ratio). For more robust masks consider replacing the simple Otsu local mask step with a trained segmentation model; this app exports per-cell crops and masks so you can use them to train such a model.")
