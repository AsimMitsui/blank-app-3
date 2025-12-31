"""
Streamlit app: PV EL Module → Cell Segregation with batch processing and save options.

Features added compared to previous version:
- Option to save segmented cell crops to server disk (per-image directory).
- Option to produce a single combined ZIP with all crops (in-memory) for download.
- Batch processing of uploaded images (existing behavior) with progress bar.
- Optional processing of a server-side folder path (process all images in that folder).
- Per-cell simple mask generation (local Otsu + small morphology) saved alongside crops.
- Status/log output for every processed image.

Usage:
- Run locally or deploy to Streamlit Cloud.
- Upload multiple images, tune parameters, check "Save outputs" to persist crops to `Output directory`.
- Click "Run (process uploads)" to process uploaded files.
- Or provide a local folder path on the server and press "Process folder" to batch-process images already on disk.
"""
import os
import io
import cv2
import time
import json
import zipfile
import glob
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ---------------------------------------
# Utilities
# ---------------------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def save_image(path: Path, image: np.ndarray, quality: int = 95):
    ensure_dir(path.parent)
    # If image is a mask (single channel) convert to 3-channel before saving as JPG
    if image.ndim == 2:
        vis = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(path), vis, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])

def zip_bytes_from_dict(filedict: Dict[str, bytes]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, data in filedict.items():
            zf.writestr(fname, data)
    bio.seek(0)
    return bio.read()

def allowed_image(path: Path) -> bool:
    return path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ---------------------------------------
# EL-specific preprocessing
# ---------------------------------------
def normalize_el(img_bgr: np.ndarray, clahe_clip: float = 2.5, tile: int = 8, blur_ksize: int = 3) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(tile, tile))
    gray_norm = clahe.apply(gray)
    if blur_ksize > 0:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        gray_norm = cv2.GaussianBlur(gray_norm, (k, k), 0)
    return gray_norm

def auto_deskew(img_bgr: np.ndarray, gray: np.ndarray, hough_thresh: int = 120) -> np.ndarray:
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, hough_thresh)
    if lines is None:
        return img_bgr
    angles = []
    for l in lines[:200]:
        theta = l[0][1]
        deg = np.rad2deg(theta)
        deg = ((deg + 90) % 180) - 90
        angles.append(deg)
    if len(angles) == 0:
        return img_bgr
    mean_angle = float(np.median(angles))
    if abs(mean_angle) < 0.25:
        return img_bgr
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), -mean_angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def perspective_warp(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) != 4:
        return img_bgr
    pts = approx.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    rect = np.array([tl, tr, br, bl], dtype=np.float32)
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    if maxW < 100 or maxH < 100:
        return img_bgr
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH), flags=cv2.INTER_LINEAR)
    return warped

# ---------------------------------------
# Grid detection + cells
# ---------------------------------------
def detect_grid_lines(gray: np.ndarray,
                      polarity: str = "auto",
                      binarize: str = "otsu",
                      ksize_v: int = 25,
                      ksize_h: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    if binarize == "adaptive":
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 31, 5)
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if polarity == "auto":
        use = 255 - bw if np.mean(gray) > 127 else bw
    elif polarity == "dark":
        use = 255 - bw
    else:
        use = bw

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, ksize_v)))
    vert = cv2.erode(use, kernel_v, iterations=1)
    vert = cv2.dilate(vert, kernel_v, iterations=1)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, ksize_h), 1))
    horiz = cv2.erode(use, kernel_h, iterations=1)
    horiz = cv2.dilate(horiz, kernel_h, iterations=1)

    return vert, horiz

def project_peaks(line_map: np.ndarray, axis: int = 0, min_dist: int = 30, min_strength: float = 0.3) -> List[int]:
    proj = line_map.sum(axis=axis)
    proj_norm = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
    peaks = []
    last_idx = -min_dist
    for i in range(1, len(proj_norm) - 1):
        if proj_norm[i] > min_strength and proj_norm[i] > proj_norm[i - 1] and proj_norm[i] > proj_norm[i + 1]:
            if i - last_idx >= min_dist:
                peaks.append(i)
                last_idx = i
    return peaks

def cuts_from_peaks(peaks: List[int], maxlen: int) -> List[int]:
    if len(peaks) < 2:
        return [0, maxlen - 1]
    cuts = [0]
    for i in range(len(peaks) - 1):
        cuts.append((peaks[i] + peaks[i + 1]) // 2)
    cuts.append(maxlen - 1)
    cuts = sorted(list(set(cuts)))
    return cuts

def build_cells_from_grid(img_bgr: np.ndarray,
                          vert_map: np.ndarray,
                          horiz_map: np.ndarray,
                          min_cell_w: int = 40,
                          min_cell_h: int = 40) -> List[Dict[str, Any]]:
    H, W = vert_map.shape
    xs = project_peaks(vert_map, axis=0, min_dist=max(20, W // 30))
    ys = project_peaks(horiz_map, axis=1, min_dist=max(20, H // 30))

    xcuts = cuts_from_peaks(xs, W)
    ycuts = cuts_from_peaks(ys, H)

    cells = []
    for r in range(len(ycuts) - 1):
        y0, y1 = ycuts[r], ycuts[r + 1]
        for c in range(len(xcuts) - 1):
            x0, x1 = xcuts[c], xcuts[c + 1]
            w, h = x1 - x0, y1 - y0
            if w >= min_cell_w and h >= min_cell_h:
                crop = img_bgr[y0:y1, x0:x1].copy()
                cells.append({
                    "row": r,
                    "col": c,
                    "bbox": (x0, y0, x1, y1),
                    "image": crop
                })
    return cells

def overlay_grid(img_bgr: np.ndarray, cells: List[Dict[str, Any]], color=(0, 255, 0), thickness=2) -> np.ndarray:
    vis = img_bgr.copy()
    for cell in cells:
        x0, y0, x1, y1 = cell["bbox"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, thickness)
    return vis

def manual_split(img_bgr: np.ndarray, n_rows: int, n_cols: int, margin: int = 0) -> List[Dict[str, Any]]:
    h, w = img_bgr.shape[:2]
    x0, y0 = margin, margin
    x1, y1 = w - margin, h - margin
    cell_w = (x1 - x0) // n_cols
    cell_h = (y1 - y0) // n_rows
    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            cx0 = x0 + c * cell_w
            cy0 = y0 + r * cell_h
            cx1 = cx0 + cell_w
            cy1 = cy0 + cell_h
            crop = img_bgr[cy0:cy1, cx0:cx1].copy()
            cells.append({
                "row": r,
                "col": c,
                "bbox": (cx0, cy0, cx1, cy1),
                "image": crop
            })
    return cells

# ---------------------------------------
# Simple per-cell mask builder
# ---------------------------------------
def build_mask_from_crop(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return np.zeros((0,0), dtype=np.uint8)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Heuristic: cells often bright; keep mask as bright regions
    # Small morphological clean
    k = max(1, min(7, (min(gray.shape)//20)|1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return (m > 0).astype(np.uint8)

# ---------------------------------------
# Streamlit UI
# ---------------------------------------
st.set_page_config(page_title="PV EL Module → Cell Segregation (Batch + Save)", layout="wide")
st.title("🔬 PV EL Module → Cell Segregation (Batch + Save)")

st.markdown("""
Upload EL PV module images to detect the cell grid and export per-cell crops.
New features:
- Save segmented cells to server disk (per-image).
- Create a combined ZIP containing all crops & masks.
- Batch process a server-side folder path (process many images already on disk).
""")

# Sidebar controls
st.sidebar.header("⚙️ Settings")

# Preprocessing
clahe_clip = st.sidebar.slider("CLAHE clipLimit", 1.0, 4.0, 2.5, 0.1)
clahe_tile = st.sidebar.slider("CLAHE tile size", 4, 16, 8, 1)
blur_ksize = st.sidebar.slider("Gaussian blur (ksize)", 0, 7, 3, 1)

# Orientation / detection
do_warp = st.sidebar.checkbox("Perspective warp (rectify module)", True)
do_deskew = st.sidebar.checkbox("Auto deskew (align grid)", True)
polarity = st.sidebar.selectbox("Line polarity", ["auto", "dark", "bright"], index=0)
binarize = st.sidebar.selectbox("Binarization", ["otsu", "adaptive"], index=0)
ksize_v = st.sidebar.slider("Vertical kernel size", 5, 75, 25, 1)
ksize_h = st.sidebar.slider("Horizontal kernel size", 5, 75, 25, 1)
min_cell_w = st.sidebar.slider("Min cell width (px)", 20, 400, 40, 10)
min_cell_h = st.sidebar.slider("Min cell height (px)", 20, 400, 40, 10)

# Manual fallback grid
use_manual = st.sidebar.checkbox("Use manual rows × cols fallback", False)
n_rows = st.sidebar.number_input("Rows", min_value=1, max_value=40, value=6)
n_cols = st.sidebar.number_input("Cols", min_value=1, max_value=40, value=10)
manual_margin = st.sidebar.number_input("Manual margin (px)", min_value=0, max_value=500, value=0)

# Save / batch options
save_outputs = st.sidebar.checkbox("Save segmented cells to disk", value=True)
out_dir_str = st.sidebar.text_input("Output directory (server)", "output")
create_combined_zip = st.sidebar.checkbox("Create combined ZIP for all processed images", value=True)
process_folder_path = st.sidebar.text_input("Server folder path for batch processing (optional)", "")

# Actions
start_btn = st.sidebar.button("🚀 Run (process uploaded)")
process_folder_btn = st.sidebar.button("📁 Process server folder")

# Uploader
uploads = st.file_uploader("Upload EL module image(s)", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], accept_multiple_files=True)

# ---------------------------------------
# Processing function
# ---------------------------------------
def process_single_image(img_pil: Image.Image,
                         settings: Dict[str, Any],
                         save_root: Path = None) -> Dict[str, Any]:
    t0 = time.time()
    img_bgr = pil_to_cv(img_pil)
    orig_name = settings.get("name", "image")

    # Optionally warp (we operate on the warped/rectified image for detection)
    processed_img = img_bgr.copy()
    if settings["do_warp"]:
        processed_img = perspective_warp(processed_img)

    # Preprocess
    gray_norm = normalize_el(processed_img,
                             clahe_clip=settings["clahe_clip"],
                             tile=settings["clahe_tile"],
                             blur_ksize=settings["blur_ksize"])

    if settings["do_deskew"]:
        processed_img = auto_deskew(processed_img, gray_norm)
        gray_norm = normalize_el(processed_img,
                                 clahe_clip=settings["clahe_clip"],
                                 tile=settings["clahe_tile"],
                                 blur_ksize=settings["blur_ksize"])

    # Detection
    if not settings["use_manual"]:
        vert_map, horiz_map = detect_grid_lines(gray_norm,
                                                polarity=settings["polarity"],
                                                binarize=settings["binarize"],
                                                ksize_v=settings["ksize_v"],
                                                ksize_h=settings["ksize_h"])
        cells = build_cells_from_grid(processed_img, vert_map, horiz_map,
                                      min_cell_w=settings["min_cell_w"],
                                      min_cell_h=settings["min_cell_h"])
    else:
        cells = manual_split(processed_img, n_rows=settings["n_rows"], n_cols=settings["n_cols"], margin=settings["manual_margin"])

    overlay = overlay_grid(processed_img, cells)
    outputs = []
    for i, cell in enumerate(cells):
        x0, y0, x1, y1 = cell["bbox"]
        crop = cell["image"]
        mask = build_mask_from_crop(crop)
        outputs.append({
            "index": i,
            "row": cell["row"],
            "col": cell["col"],
            "bbox": cell["bbox"],
            "crop": crop,
            "mask": mask
        })

    # Optionally save to disk
    if save_root is not None:
        save_root = Path(save_root)
        ensure_dir(save_root)
        # save overlay
        save_image(save_root / f"{orig_name}_overlay.jpg", overlay)
        cells_dir = save_root / "cells"
        ensure_dir(cells_dir)
        for out in outputs:
            idx = out["index"]
            crop = out["crop"]
            mask = out["mask"]
            # save crop (jpg) and mask (png)
            save_image(cells_dir / f"{orig_name}_cell_{idx:03d}.jpg", crop)
            # mask as png
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_buf = io.BytesIO()
            mask_pil.save(mask_buf, format="PNG")
            with open(cells_dir / f"{orig_name}_cell_{idx:03d}_mask.png", "wb") as mf:
                mf.write(mask_buf.getvalue())

        # summary json
        meta = {"n_cells": len(outputs), "cells": [{"index": o["index"], "row": o["row"], "col": o["col"], "bbox": o["bbox"]} for o in outputs]}
        with open(save_root / "summary.json", "w") as f:
            json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    return {"n_cells": len(outputs), "overlay": overlay, "outputs": outputs, "elapsed": elapsed}

# ---------------------------------------
# Batch helpers
# ---------------------------------------
def process_uploaded_files(uploaded_files: List[Any], settings: Dict[str, Any]):
    combined_files: Dict[str, bytes] = {}
    overall_summary = {}
    total = len(uploaded_files)
    progress = st.progress(0)
    status = st.empty()
    processed_count = 0

    out_base = Path(out_dir_str) if save_outputs else None
    for i, upl in enumerate(uploaded_files):
        status.text(f"Processing {i+1}/{total}: {upl.name}")
        try:
            img_pil = Image.open(io.BytesIO(upl.read())).convert("RGB")
        except Exception as e:
            st.warning(f"Failed to open {upl.name}: {e}")
            continue

        settings_local = settings.copy()
        settings_local["name"] = Path(upl.name).stem
        res = process_single_image(img_pil, settings_local, save_root=(out_base / Path(upl.name).stem) if out_base else None)

        # Add overlay and crops to combined_files (if requested)
        if create_combined_zip:
            # overlay
            buf = io.BytesIO()
            cv_to_pil(res["overlay"]).save(buf, format="PNG")
            combined_files[f"{Path(upl.name).stem}/overlay.png"] = buf.getvalue()

            # crops and masks
            for out in res["outputs"]:
                idx = out["index"]
                # crop
                buf = io.BytesIO()
                cv_to_pil(out["crop"]).save(buf, format="PNG")
                combined_files[f"{Path(upl.name).stem}/cell_{idx:03d}.png"] = buf.getvalue()
                # mask
                bufm = io.BytesIO()
                Image.fromarray((out["mask"] * 255).astype(np.uint8)).save(bufm, format="PNG")
                combined_files[f"{Path(upl.name).stem}/cell_{idx:03d}_mask.png"] = bufm.getvalue()

            # summary per image
            combined_files[f"{Path(upl.name).stem}/summary.json"] = json.dumps({"n_cells": res["n_cells"]}).encode("utf-8")

        overall_summary[upl.name] = {"n_cells": res["n_cells"], "elapsed": res["elapsed"]}
        processed_count += 1
        progress.progress(int((i+1)/total * 100))

    status.text(f"Finished: processed {processed_count}/{total}")
    progress.empty()
    # create combined zip if requested
    zip_bytes = None
    if create_combined_zip and combined_files:
        # add top-level summary
        combined_files["overall_summary.json"] = json.dumps(overall_summary, indent=2).encode("utf-8")
        zip_bytes = zip_bytes_from_dict(combined_files)
    return zip_bytes, overall_summary

def process_server_folder(folder_path: str, settings: Dict[str, Any]):
    p = Path(folder_path)
    if not p.exists() or not p.is_dir():
        st.error("Provided folder path does not exist or is not a directory on server.")
        return None, {}
    # collect image files
    files = [f for f in sorted(p.iterdir()) if allowed_image(f)]
    if not files:
        st.warning("No image files found in folder.")
        return None, {}
    combined_files: Dict[str, bytes] = {}
    overall_summary = {}
    total = len(files)
    progress = st.progress(0)
    status = st.empty()
    out_base = Path(out_dir_str) if save_outputs else None
    for i, fp in enumerate(files):
        status.text(f"Processing {i+1}/{total}: {fp.name}")
        try:
            img_pil = Image.open(str(fp)).convert("RGB")
        except Exception as e:
            st.warning(f"Failed to open {fp}: {e}")
            continue

        settings_local = settings.copy()
        settings_local["name"] = fp.stem
        res = process_single_image(img_pil, settings_local, save_root=(out_base / fp.stem) if out_base else None)

        if create_combined_zip:
            buf = io.BytesIO(); cv_to_pil(res["overlay"]).save(buf, format="PNG")
            combined_files[f"{fp.stem}/overlay.png"] = buf.getvalue()
            for out in res["outputs"]:
                idx = out["index"]
                buf = io.BytesIO(); cv_to_pil(out["crop"]).save(buf, format="PNG")
                combined_files[f"{fp.stem}/cell_{idx:03d}.png"] = buf.getvalue()
                bufm = io.BytesIO(); Image.fromarray((out["mask"]*255).astype(np.uint8)).save(bufm, format="PNG")
                combined_files[f"{fp.stem}/cell_{idx:03d}_mask.png"] = bufm.getvalue()
            combined_files[f"{fp.stem}/summary.json"] = json.dumps({"n_cells": res["n_cells"]}).encode("utf-8")

        overall_summary[fp.name] = {"n_cells": res["n_cells"], "elapsed": res["elapsed"]}
        progress.progress(int((i+1)/total * 100))

    status.text(f"Finished: processed {len(files)}/{total}")
    progress.empty()
    zip_bytes = None
    if create_combined_zip and combined_files:
        combined_files["overall_summary.json"] = json.dumps(overall_summary, indent=2).encode("utf-8")
        zip_bytes = zip_bytes_from_dict(combined_files)
    return zip_bytes, overall_summary

# ---------------------------------------
# Actions
# ---------------------------------------
if start_btn:
    if not uploads:
        st.warning("Upload at least one image.")
    else:
        st.info("Starting batch processing of uploaded files...")
        settings = {
            "clahe_clip": clahe_clip,
            "clahe_tile": clahe_tile,
            "blur_ksize": blur_ksize,
            "do_warp": do_warp,
            "do_deskew": do_deskew,
            "polarity": polarity,
            "binarize": binarize,
            "ksize_v": ksize_v,
            "ksize_h": ksize_h,
            "min_cell_w": min_cell_w,
            "min_cell_h": min_cell_h,
            "use_manual": use_manual,
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "manual_margin": int(manual_margin)
        }
        zip_bytes, summary = process_uploaded_files(uploads, settings)
        st.success("Batch processing complete.")
        st.json(summary)
        if zip_bytes:
            st.download_button("📦 Download combined ZIP of all processed images", data=zip_bytes, file_name="all_cells.zip", mime="application/zip")

if process_folder_btn:
    if not process_folder_path:
        st.warning("Enter a server-side folder path to process.")
    else:
        st.info(f"Starting batch processing of folder: {process_folder_path}")
        settings = {
            "clahe_clip": clahe_clip,
            "clahe_tile": clahe_tile,
            "blur_ksize": blur_ksize,
            "do_warp": do_warp,
            "do_deskew": do_deskew,
            "polarity": polarity,
            "binarize": binarize,
            "ksize_v": ksize_v,
            "ksize_h": ksize_h,
            "min_cell_w": min_cell_w,
            "min_cell_h": min_cell_h,
            "use_manual": use_manual,
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "manual_margin": int(manual_margin)
        }
        zip_bytes, summary = process_server_folder(process_folder_path, settings)
        st.success("Folder batch processing complete.")
        st.json(summary)
        if zip_bytes:
            st.download_button("📦 Download combined ZIP of folder processing", data=zip_bytes, file_name="folder_cells.zip", mime="application/zip")

st.markdown("---")
st.caption("Notes: 'Save segmented cells to disk' writes per-image directories under the Output directory. The combined ZIP (in-memory) collects overlay images, per-cell crops and masks for every processed image. Processing a server folder requires that Streamlit has access to that path (useful when testing on a server with images already uploaded).")
