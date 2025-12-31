"""
Streamlit app: Extract EL PV cells from EL PV module images

Features:
- Upload EL module images
- Option: enter expected total number of cells (e.g. 144)
- If "Enforce expected" is checked the app will split the (warped) module into that grid
  (chooses best rows x cols factor pair). If exact divisor not available it chooses
  an approximate grid and returns the first N cells.
- Otherwise the app will auto-detect gridlines and infer cells.
- Exports per-cell crops (original image coordinates), warp-space crops and per-cell masks in an in-memory ZIP.
- Shows a preview gallery.

Run:
pip install streamlit opencv-python-headless numpy pillow
streamlit run app.py
"""
import io
import math
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------
# Helpers: image conversions
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
# Geometry & warp
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
# Grid detection (morph + peaks)
# ---------------------------
def normalize_el_gray(img_bgr: np.ndarray, clahe_clip: float = 2.5, tile: int = 8, blur: int = 3) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(tile, tile))
    gray = clahe.apply(gray)
    if blur and blur > 0:
        k = blur if blur % 2 == 1 else blur + 1
        gray = cv2.GaussianBlur(gray, (k,k), 0)
    return gray

def detect_line_maps(gray: np.ndarray, polarity: str = "auto", binarize: str = "otsu", k_v: int = 25, k_h: int = 25):
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

def project_peaks(line_map: np.ndarray, axis:int=0, min_dist:int=20, min_strength:float=0.15) -> List[int]:
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
                         min_w:int=40, min_h:int=40) -> List[Dict[str,Any]]:
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
# Manual splitting & factorization
# ---------------------------
def factor_pairs(n:int) -> List[Tuple[int,int]]:
    pairs = []
    for r in range(1, int(math.sqrt(n))+1):
        if n % r == 0:
            c = n // r
            pairs.append((r,c))
    # include both orientations
    pairs_full = []
    for (r,c) in pairs:
        pairs_full.append((r,c))
        if r != c:
            pairs_full.append((c,r))
    # sort so rows <= cols
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
    # fallback for prime or no divisors: choose near-square grid with product >= total
    r = int(math.sqrt(total))
    c = int(math.ceil(total / r))
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
# Map warp bboxes -> original and mask
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
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Extract EL PV Cells", layout="wide")
st.title("Extract EL PV module → individual PV cells")

st.markdown("""
Upload EL PV module images, optionally enter expected total cells (e.g. 144).
If you check "Enforce expected", the app will split the rectified module into that grid.
Otherwise it will attempt automatic gridline detection.
""")

# Controls
expected_cells = st.number_input("Expected total cells (0 = unknown)", min_value=0, value=144, step=1)
enforce_expected = st.checkbox("Enforce expected grid (force split into rows × cols)", value=True)
deskew = st.checkbox("Deskew / rotate to align grid (Hough)", value=False)
do_warp = st.checkbox("Perspective warp (rectify module)", value=True)
polarity = st.selectbox("Line polarity", ["auto","dark","bright"], index=0)
binarize = st.selectbox("Binarization", ["otsu","adaptive"], index=0)
k_v = st.slider("Vertical kernel size", 5, 75, 25)
k_h = st.slider("Horizontal kernel size", 5, 75, 25)
min_cell_w = st.slider("Min cell width (px)", 10, 400, 30)
min_cell_h = st.slider("Min cell height (px)", 10, 400, 30)
margin = st.number_input("Manual grid margin (px)", min_value=0, max_value=200, value=0)
uploads = st.file_uploader("Upload EL module image(s)", type=["jpg","jpeg","png","tif","tiff","bmp"], accept_multiple_files=True)
run = st.button("Run")

# Simple deskew using Hough lines (optional)
def simple_deskew(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 120)
    if lines is None:
        return img_bgr
    angles = []
    for l in lines[:300]:
        theta = l[0][1]
        deg = np.rad2deg(theta); deg = ((deg+90)%180)-90
        angles.append(deg)
    if not angles:
        return img_bgr
    angle = float(np.median(angles))
    if abs(angle) < 0.25:
        return img_bgr
    h,w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), -angle, 1.0)
    return cv2.warpAffine(img_bgr, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# Process
if run:
    if not uploads:
        st.warning("Upload at least one image.")
    else:
        for upl in uploads:
            img_pil = Image.open(io.BytesIO(upl.read())).convert("RGB")
            img_orig = pil_to_cv(img_pil)
            H_orig, W_orig = img_orig.shape[:2]

            # Warp / rectify
            if do_warp:
                warped, M, Minv = perspective_warp(img_orig)
            else:
                warped = img_orig.copy(); M = np.eye(3, dtype=np.float32); Minv = np.eye(3, dtype=np.float32)

            if deskew:
                warped = simple_deskew(warped)

            warped_gray = normalize_el_gray(warped, blur=3)

            cells_warp = []
            if enforce_expected and expected_cells > 0:
                rows, cols = choose_grid_for_count(expected_cells, (warped.shape[0], warped.shape[1]))
                st.info(f"Enforcing grid: {rows} rows × {cols} cols (product {rows*cols})")
                cells_warp = manual_split_warp(warped, rows, cols, margin=margin)
                # If product > expected, trim to expected
                if len(cells_warp) > expected_cells:
                    cells_warp = cells_warp[:expected_cells]
            else:
                vert_map, horiz_map = detect_line_maps(warped_gray, polarity=polarity, binarize=binarize, k_v=k_v, k_h=k_h)
                cells_warp = grid_cells_from_maps(warped, vert_map, horiz_map, min_w=min_cell_w, min_h=min_cell_h)
                st.info(f"Auto-detected {len(cells_warp)} cells (warp-plane)")

            # Build outputs: map to original coords, make masks, package
            files: Dict[str, bytes] = {}
            summary = {"n_cells": len(cells_warp), "cells": []}
            warped_gray_for_masks = normalize_el_gray(warped, blur=1)
            overlay = warped.copy()
            for i, c in enumerate(cells_warp):
                r = c["row"]; col = c["col"]
                bbox_w = c["bbox_warp"]
                mask_w = build_mask_for_bbox(warped_gray_for_masks, bbox_w)
                bbox_o = warp_bbox_to_original(bbox_w, Minv, clip_shape=(H_orig, W_orig))
                x_o,y_o,w_o,h_o = bbox_o
                img_orig_crop = None
                if w_o>0 and h_o>0:
                    img_orig_crop = img_orig[y_o:y_o+h_o, x_o:x_o+w_o].copy()
                # Save warp-space crop
                pil_w = cv_to_pil(c["image_warp"])
                buf = io.BytesIO(); pil_w.save(buf, format="PNG"); files[f"cell_{i:03d}_warp.png"] = buf.getvalue()
                # Save orig-space crop if available
                if img_orig_crop is not None:
                    pil_o = cv_to_pil(img_orig_crop); buf = io.BytesIO(); pil_o.save(buf, format="PNG"); files[f"cell_{i:03d}_orig.png"] = buf.getvalue()
                # Save mask (warp)
                if mask_w.size != 0:
                    pil_m = Image.fromarray((mask_w*255).astype("uint8")); buf = io.BytesIO(); pil_m.save(buf, format="PNG"); files[f"cell_{i:03d}_mask.png"] = buf.getvalue()
                # overlay rect on warped for preview
                x0,y0,x1,y1 = bbox_w
                cv2.rectangle(overlay, (x0,y0), (x1,y1), (0,255,0), 2)
                summary["cells"].append({"index": i, "row": r, "col": col, "bbox_warp": bbox_w, "bbox_orig": bbox_o})

            # store overlay
            buf = io.BytesIO(); cv_to_pil(overlay).save(buf, format="PNG"); files["overlay_warp.png"] = buf.getvalue()
            files["summary.json"] = (str(summary)).encode("utf-8")
            zipb = zip_bytes_from_dict(files)

            st.success(f"{upl.name}: extracted {len(cells_warp)} cells")
            st.image(cv_to_pil(overlay), caption=f"Warp-plane overlay: {upl.name}", use_column_width=True)

            # Show first 12 crops (prefer original-space)
            preview = cells_warp[:min(len(cells_warp), 12)]
            cols_show = st.columns(min(6, max(1, len(preview))))
            for i, c in enumerate(preview):
                bbox_w = c["bbox_warp"]
                bbox_o = warp_bbox_to_original(bbox_w, Minv, clip_shape=(H_orig,W_orig))
                x_o,y_o,w_o,h_o = bbox_o
                img_show = None
                if w_o>0 and h_o>0:
                    img_show = img_orig[y_o:y_o+h_o, x_o:x_o+w_o]
                else:
                    img_show = c["image_warp"]
                cols_show[i % len(cols_show)].image(cv_to_pil(img_show), caption=f"cell {i}", use_column_width=True)

            st.download_button(f"Download cells ZIP ({upl.name})", data=zipb, file_name=f"{Path(upl.name).stem}_cells.zip", mime="application/zip")

st.markdown("---")
st.caption("If you know the exact total cells (e.g. 144) enable 'Enforce expected' — the app will split the (warped) module into a best-matching rows×cols grid and return exactly that many cells (trimming the last ones if needed). For difficult images, try adjusting kernel sizes or use the manual option.")
