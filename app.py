"""Automatic chessboard photo -> digital position/FEN.

Version 7.0
- Restores ChessSight RT-DETR as the PRIMARY piece classifier (the better-performing
  model from the previous version).
- Removes the per-square ONNX piece classifier from the decision path.
- Uses ChessSight's dedicated Swin corner-heatmap model for automatic board geometry,
  with OpenCV and the older pose model as fallbacks.
- Uses the upstream ChessSight three-pass recipe: full frame, square board crop,
  and mirrored crop, with weighted per-square score fusion.

Run with:
    .\\.venv\\Scripts\\python.exe -m streamlit run app.py

The ChessSight weights are labelled CC BY-NC 4.0; this version is intended for
non-commercial educational/research use. See the model card linked in the UI.
"""
from __future__ import annotations

import itertools
import math
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import chess

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
POSE_CORNER_PATH = MODEL_DIR / "photochess_corner.pt"
POSE_CORNER_URL = "https://raw.githubusercontent.com/2T1F/photochess/main/corner.pt"
CHESSSIGHT_ID = "tchauffi/chesssight-rtdetr-v0.2.0"
CORNER_MODEL_ID = "tchauffi/chesssight-corner-heatmap-v1"
CORNER_MIN_SCORE = 0.30
BOARD_SIZE = 800
# Published calibration from the ChessSight v0.2.0 model card.
CALIBRATION_SCALE = 2.9502
CALIBRATION_BIAS = 10.5962
POSITION_CONFIDENCE = 0.05

FEN_MAP = {
    "white_pawn": "P", "white_knight": "N", "white_bishop": "B",
    "white_rook": "R", "white_queen": "Q", "white_king": "K",
    "black_pawn": "p", "black_knight": "n", "black_bishop": "b",
    "black_rook": "r", "black_queen": "q", "black_king": "k",
}
WHITE_NAMES = {"white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king"}
BLACK_NAMES = {"black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king"}

st.set_page_config(page_title="Automatikus sakkállás-felismerő", page_icon="♟️", layout="wide")


def normalise_label(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    # Accept common label styles from model cards/configs.
    aliases = {
        "white_pawn": "white_pawn", "whitepawn": "white_pawn", "wp": "white_pawn",
        "white_knight": "white_knight", "whiteknight": "white_knight", "wn": "white_knight",
        "white_bishop": "white_bishop", "whitebishop": "white_bishop", "wb": "white_bishop",
        "white_rook": "white_rook", "whiterook": "white_rook", "wr": "white_rook",
        "white_queen": "white_queen", "whitequeen": "white_queen", "wq": "white_queen",
        "white_king": "white_king", "whiteking": "white_king", "wk": "white_king",
        "black_pawn": "black_pawn", "blackpawn": "black_pawn", "bp": "black_pawn",
        "black_knight": "black_knight", "blackknight": "black_knight", "bn": "black_knight",
        "black_bishop": "black_bishop", "blackbishop": "black_bishop", "bb": "black_bishop",
        "black_rook": "black_rook", "blackrook": "black_rook", "br": "black_rook",
        "black_queen": "black_queen", "blackqueen": "black_queen", "bq": "black_queen",
        "black_king": "black_king", "blackking": "black_king", "bk": "black_king",
    }
    if text in aliases:
        return aliases[text]
    # Handle names that prefix a class with a dataset id.
    for color in ("white", "black"):
        for piece in ("pawn", "knight", "bishop", "rook", "queen", "king"):
            if color in text and piece in text:
                return f"{color}_{piece}"
    return text


def calibrate_score(raw: float) -> float:
    raw = float(np.clip(raw, 1e-6, 1 - 1e-6))
    logit = math.log(raw / (1.0 - raw))
    z = CALIBRATION_SCALE * logit + CALIBRATION_BIAS
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


@st.cache_resource(show_spinner=False)
def load_chesssight_model():
    """Load the stronger, documented ChessSight pretrained detector."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    processor = AutoImageProcessor.from_pretrained(CHESSSIGHT_ID)
    model = AutoModelForObjectDetection.from_pretrained(CHESSSIGHT_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device


def run_chesssight(image_bgr: np.ndarray, processor, model, device) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Infer detections and apply the model's published score calibration."""
    import torch
    from PIL import Image

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    target_sizes = torch.tensor([[h, w]], device=device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.0)[0]

    id2label = getattr(model.config, "id2label", {}) or {}
    all_dets: list[dict[str, Any]] = []
    for box, score, label_id in zip(results["boxes"], results["scores"], results["labels"]):
        raw_score = float(score.detach().cpu().item())
        calibrated = calibrate_score(raw_score)
        label_index = int(label_id.detach().cpu().item())
        label = id2label.get(label_index, id2label.get(str(label_index), str(label_index)))
        label = normalise_label(label)
        xyxy = box.detach().cpu().numpy().astype(float).tolist()
        # Keep low-scoring runner-up class proposals for score diagnostics and
        # per-square second opinions. They never override the top fused score.
        if calibrated < 0.001:
            continue
        x1, y1, x2, y2 = xyxy
        all_dets.append({"label": label, "score": calibrated, "raw_score": raw_score,
                         "box": (x1, y1, x2, y2)})
    pieces = [d for d in all_dets if d["label"] in FEN_MAP and d["score"] >= POSITION_CONFIDENCE]
    return all_dets, pieces


@st.cache_data(show_spinner=False, max_entries=24)
def cached_chesssight_image(image_png: bytes):
    """Cache detections by image bytes so repeated clicks give the same outputs."""
    image_array = np.frombuffer(image_png, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError("Nem sikerült dekódolni a modell bemeneti képét.")
    processor, model, device = load_chesssight_model()
    all_dets, pieces = run_chesssight(image_bgr, processor, model, device)
    return all_dets, pieces


def detect_cached(image_bgr: np.ndarray):
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Nem sikerült előkészíteni a képet a felismeréshez.")
    return cached_chesssight_image(encoded.tobytes())


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four corners clockwise, starting at the visually top-left corner."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    ordered = points[np.argsort(angles)]
    top_left_reference = np.array([points[:, 0].min(), points[:, 1].min()], dtype=np.float32)
    first = int(np.argmin(np.linalg.norm(ordered - top_left_reference, axis=1)))
    return np.roll(ordered, -first, axis=0).astype(np.float32)


def quad_area(points: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2))))


def get_board_bbox(detections: list[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    boards = [d for d in detections if d["label"] in {"board", "chessboard", "chess_board"}]
    if not boards:
        return None
    # Prefer the largest high-confidence board instance if the model emits duplicates.
    best = max(boards, key=lambda d: d["score"] * max(1.0, (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])) ** 0.15)
    return tuple(best["box"])


def _quad_geometry_ok(quad: np.ndarray, width: int, height: int) -> bool:
    quad = order_corners(quad)
    if not cv2.isContourConvex(quad.reshape(-1, 1, 2).astype(np.int32)):
        return False
    area_ratio = quad_area(quad) / max(1.0, width * height)
    if not 0.035 <= area_ratio <= 0.96:
        return False
    sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    if sides.min() < 0.12 * min(width, height):
        return False
    if np.any(quad[:, 0] < -0.10 * width) or np.any(quad[:, 0] > 1.10 * width):
        return False
    if np.any(quad[:, 1] < -0.10 * height) or np.any(quad[:, 1] > 1.10 * height):
        return False
    return True


def best_quad_from_candidates(points: list[tuple[float, float]], scores: list[float],
                              shape: tuple[int, ...], board_bbox: tuple[float, float, float, float] | None = None) -> np.ndarray | None:
    """Select four spatially distinct corners instead of naively taking top four scores."""
    height, width = shape[:2]
    if len(points) < 4:
        return None
    pts = np.asarray(points, dtype=np.float32)
    vals = np.asarray(scores, dtype=np.float32)
    # Greedy spatial clustering prevents several detections of the same corner winning.
    order = np.argsort(vals)[::-1]
    clusters: list[dict[str, Any]] = []
    merge_dist = 0.022 * math.hypot(width, height)
    for idx in order:
        point = pts[idx]
        match = None
        for cluster in clusters:
            if np.linalg.norm(point - cluster["point"]) < merge_dist:
                match = cluster
                break
        if match is None:
            clusters.append({"point": point.copy(), "score": float(vals[idx]), "weight": float(vals[idx]), "members": 1})
        else:
            weight = max(0.01, float(vals[idx]))
            match["point"] = (match["point"] * match["weight"] + point * weight) / (match["weight"] + weight)
            match["score"] = max(match["score"], float(vals[idx]))
            match["weight"] += weight
            match["members"] += 1
    if len(clusters) < 4:
        return None
    # Bound combinatorics if the detector emits too many candidates.
    clusters.sort(key=lambda item: item["score"], reverse=True)
    clusters = clusters[:14]
    best_score = -1e9
    best_quad = None
    image_area = max(1.0, width * height)
    for combo in itertools.combinations(clusters, 4):
        raw_points = np.array([item["point"] for item in combo], dtype=np.float32)
        quad = order_corners(raw_points)
        if not _quad_geometry_ok(quad, width, height):
            continue
        area_ratio = quad_area(quad) / image_area
        score_sum = sum(item["score"] for item in combo) / 4.0
        # A valid board should be a reasonably square quadrilateral, while allowing a steep camera angle.
        edges = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        horiz = (edges[0] + edges[2]) / 2.0
        vert = (edges[1] + edges[3]) / 2.0
        ar = max(horiz, vert) / max(1.0, min(horiz, vert))
        shape_penalty = max(0.0, ar - 2.6) * 0.12
        # If the detector also found a board box, prefer corners inside its frame and fairly centred.
        bbox_score = 0.0
        if board_bbox is not None:
            bx1, by1, bx2, by2 = board_bbox
            bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
            qx1, qy1 = quad.min(axis=0)
            qx2, qy2 = quad.max(axis=0)
            margins = np.array([(qx1 - bx1) / bw, (bx2 - qx2) / bw,
                                (qy1 - by1) / bh, (by2 - qy2) / bh])
            if np.all(margins >= -0.08) and np.all(margins <= 0.42):
                bbox_score = 0.15 - 0.9 * abs(margins[0] - margins[1]) - 0.9 * abs(margins[2] - margins[3])
            else:
                bbox_score = -0.35
        total = score_sum + 0.10 * math.log(max(area_ratio, 1e-3)) - shape_penalty + bbox_score
        if total > best_score:
            best_score, best_quad = total, quad
    return best_quad


def _score_quad_against_board_bbox(quad: np.ndarray, board_bbox: tuple[float, float, float, float] | None) -> float:
    if board_bbox is None:
        return 0.0
    bx1, by1, bx2, by2 = board_bbox
    bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
    qx1, qy1 = quad.min(axis=0)
    qx2, qy2 = quad.max(axis=0)
    margins = np.array([(qx1 - bx1) / bw, (bx2 - qx2) / bw,
                        (qy1 - by1) / bh, (by2 - qy2) / bh])
    if np.any(margins < -0.05) or np.any(margins > 0.45):
        return -2.0
    return -abs(margins[0] - margins[1]) - abs(margins[2] - margins[3]) - 0.25 * abs(margins.mean() - 0.09)


def detect_corners_from_checker_grid(image_bgr: np.ndarray,
                                     board_bbox: tuple[float, float, float, float] | None) -> tuple[np.ndarray | None, str]:
    """Find visible inner checker intersections with OpenCV and extrapolate to the 8x8 border.

    This is the main corner detector. It often succeeds even when pieces hide the first/last
    row of intersections, because it tests partial 7xN patterns and extrapolates the plane.
    """
    height, width = image_bgr.shape[:2]
    gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variants = [gray_full, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_full)]
    rois: list[tuple[np.ndarray, int, int]] = [(im, 0, 0) for im in variants]
    if board_bbox is not None:
        x1, y1, x2, y2 = board_bbox
        pad_x, pad_y = 0.04 * (x2 - x1), 0.04 * (y2 - y1)
        x1i = max(0, int(x1 - pad_x)); y1i = max(0, int(y1 - pad_y))
        x2i = min(width, int(x2 + pad_x)); y2i = min(height, int(y2 + pad_y))
        if x2i - x1i > 120 and y2i - y1i > 120:
            for im in variants:
                rois.append((im[y1i:y2i, x1i:x2i], x1i, y1i))

    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
    candidates: list[tuple[float, np.ndarray, str]] = []
    # Rows/columns are internal intersection points. The usual start position hides the first
    # and last rows with pieces, so a 7x5 inner pattern is especially useful.
    patterns = [(7, 7), (7, 6), (7, 5), (7, 4), (6, 6), (6, 5), (6, 4)]
    for roi_index, (gray, off_x, off_y) in enumerate(rois):
        if gray.size == 0:
            continue
        for cols, rows in patterns:
            try:
                found, raw_corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=flags)
            except cv2.error:
                found, raw_corners = False, None
            if not found or raw_corners is None or len(raw_corners) != cols * rows:
                continue
            image_points = raw_corners.reshape(rows, cols, 2).astype(np.float32)
            image_points[:, :, 0] += off_x
            image_points[:, :, 1] += off_y
            flat_points = image_points.reshape(-1, 2)

            for grid_x_start in range(1, 9 - cols):
                for grid_y_start in range(1, 9 - rows):
                    grid_points = np.array(
                        [[grid_x_start + c, grid_y_start + r] for r in range(rows) for c in range(cols)],
                        dtype=np.float32,
                    )
                    H, inliers = cv2.findHomography(grid_points, flat_points, 0)
                    if H is None or not np.isfinite(H).all():
                        continue
                    board_grid_corners = np.array([[[0, 0], [8, 0], [8, 8], [0, 8]]], dtype=np.float32)
                    quad = cv2.perspectiveTransform(board_grid_corners, H).reshape(4, 2)
                    quad = order_corners(quad)
                    if not _quad_geometry_ok(quad, width, height):
                        continue
                    # Reward bigger detected patterns and a well-centred play area inside the board box.
                    pattern_score = (cols * rows) / 49.0
                    bbox_score = _score_quad_against_board_bbox(quad, board_bbox)
                    centered_offset = abs((grid_x_start + (cols - 1) / 2.0) - 4.0) + abs((grid_y_start + (rows - 1) / 2.0) - 4.0)
                    score = 1.6 * pattern_score + 0.6 * bbox_score - 0.06 * centered_offset
                    # Full-image detection wins ties over a cropped detection; a crop is still useful when the full image is busy.
                    if roi_index >= 2:
                        score -= 0.015
                    candidates.append((score, quad, f"OpenCV grid {cols}x{rows}"))
    if not candidates:
        return None, ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, quad, method = candidates[0]
    return quad, method


def detect_corners_from_model_boxes(detections: list[dict[str, Any]], image_shape,
                                    board_bbox: tuple[float, float, float, float] | None) -> np.ndarray | None:
    candidates = [d for d in detections if d["label"] in {"corner", "board_corner", "chessboard_corner"}]
    pts, scores = [], []
    for det in candidates:
        x1, y1, x2, y2 = det["box"]
        pts.append(((x1 + x2) / 2, (y1 + y2) / 2))
        scores.append(det["score"])
    return best_quad_from_candidates(pts, scores, image_shape, board_bbox)


def ensure_download(path: Path, url: str, description: str) -> None:
    if path.exists() and path.stat().st_size > 100_000:
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response, tmp_path.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        if not tmp_path.exists() or tmp_path.stat().st_size < 100_000:
            raise RuntimeError("A letöltött modellfájl túl kicsi vagy üres.")
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


@st.cache_resource(show_spinner=False)
def load_optional_photochess_corner_model():
    from ultralytics import YOLO
    ensure_download(POSE_CORNER_PATH, POSE_CORNER_URL, "fallback sarokmodell")
    return YOLO(str(POSE_CORNER_PATH))


def detect_corners_from_pose_model(image_bgr: np.ndarray, board_bbox=None) -> np.ndarray | None:
    """Fallback only: run the older pose model at lower confidence and high resolution."""
    try:
        model = load_optional_photochess_corner_model()
    except Exception:
        return None
    points: list[tuple[float, float]] = []
    scores: list[float] = []
    for imgsz, conf in ((1280, 0.04), (960, 0.025)):
        try:
            result = model.predict(source=image_bgr, imgsz=imgsz, conf=conf, iou=0.35,
                                   augment=True, verbose=False)[0]
        except Exception:
            continue
        if getattr(result, "keypoints", None) is None or result.keypoints.data is None:
            continue
        keypoints = result.keypoints.data.detach().cpu().numpy().reshape(-1, 3)
        if result.boxes is not None and result.boxes.conf is not None and len(result.boxes.conf) == len(keypoints):
            box_conf = result.boxes.conf.detach().cpu().numpy()
        else:
            box_conf = np.ones(len(keypoints), dtype=np.float32)
        for kp, det_conf in zip(keypoints, box_conf):
            x, y, kp_conf = map(float, kp[:3])
            score = float(np.clip(kp_conf * det_conf, 0, 1))
            if np.isfinite(x) and np.isfinite(y) and score > 0.005:
                points.append((x, y))
                scores.append(score)
    return best_quad_from_candidates(points, scores, image_bgr.shape, board_bbox)


@st.cache_resource(show_spinner=False)
def load_chesssight_corner_model():
    """Load the dedicated ChessSight Swin-T corner heatmap model from Hugging Face."""
    import importlib.util
    import sys
    import torch
    from huggingface_hub import snapshot_download

    model_dir = Path(snapshot_download(
        repo_id=CORNER_MODEL_ID,
        allow_patterns=["config.json", "model.safetensors", "modeling_corner_heatmap.py"],
    ))
    module_path = model_dir / "modeling_corner_heatmap.py"
    if not module_path.is_file():
        raise RuntimeError(f"Hiányzik a sarokmodell kódja: {module_path}")

    module_name = "_chesssight_corner_heatmap_v1"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Nem sikerült betölteni a ChessSight sarokmodell modulját.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = module.load_corner_model(model_dir, device=device)
    return module, model, config, device


def detect_corners_from_heatmap(image_bgr: np.ndarray) -> np.ndarray | None:
    """Use the model author's dedicated Swin heatmap model to predict all four corners."""
    from PIL import Image

    module, model, config, device = load_chesssight_corner_model()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    quad = module.predict_quad(
        model, pil_image, config, device=device, min_score=CORNER_MIN_SCORE
    )
    if quad is None:
        return None
    quad = order_corners(np.asarray(quad, dtype=np.float32))
    h, w = image_bgr.shape[:2]
    if not _quad_geometry_ok(quad, w, h):
        return None
    return quad


def detect_board_corners(image_bgr: np.ndarray, detections: list[dict[str, Any]]) -> tuple[np.ndarray, str]:
    board_bbox = get_board_bbox(detections)
    # Primary corner model: trained specifically for chessboard geometry.
    heatmap_error = None
    try:
        quad = detect_corners_from_heatmap(image_bgr)
        if quad is not None:
            return quad, "ChessSight Swin corner heatmap v1"
    except Exception as exc:
        heatmap_error = str(exc)

    # Independent fallbacks are retained in case the heatmap model cannot load
    # or the photograph is outside its training distribution.
    quad, method = detect_corners_from_checker_grid(image_bgr, board_bbox)
    if quad is not None:
        return quad, method
    quad = detect_corners_from_model_boxes(detections, image_bgr.shape, board_bbox)
    if quad is not None:
        return quad, "ChessSight corner-box proposals"
    quad = detect_corners_from_pose_model(image_bgr, board_bbox)
    if quad is not None:
        return quad, "fallback photochess pose model"
    detail = f" A ChessSight sarokmodell jelzése: {heatmap_error}" if heatmap_error else ""
    raise RuntimeError(
        "Nem sikerült automatikusan meghatározni a sakktábla négy sarkát. "
        "A program kipróbálta a ChessSight Swin sarokmodellt, az OpenCV rácsdetektálást, "
        "a detektor sarokjavaslatait és a photochess tartalékmodellt. "
        "Készíts olyan képet, amelyen a teljes tábla és mind a négy sarka látszik. "
        "Kézi sarokkijelölést nem kérünk." + detail
    )


def make_fen(grid: list[list[str]], side_to_move: str = "w") -> str:
    rows = []
    for row in grid:
        out, empty_count = "", 0
        for piece in row:
            if piece == ".":
                empty_count += 1
            else:
                if empty_count:
                    out += str(empty_count)
                    empty_count = 0
                out += piece
        if empty_count:
            out += str(empty_count)
        rows.append(out)
    return f"{'/'.join(rows)} {side_to_move} - - 0 1"


def draw_original(image_bgr: np.ndarray, corners: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    view = image_bgr.copy()
    quad = order_corners(corners).astype(int)
    cv2.polylines(view, [quad.reshape(-1, 1, 2)], True, (0, 255, 255), 4, cv2.LINE_AA)
    for i, (x, y) in enumerate(quad):
        cv2.circle(view, (int(x), int(y)), 8, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(view, f"C{i+1}", (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, f"C{i+1}", (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)
    for det in detections:
        if det["label"] not in FEN_MAP or det["score"] < POSITION_CONFIDENCE:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in det["box"]]
        cv2.rectangle(view, (x1, y1), (x2, y2), (40, 220, 50), 2, cv2.LINE_AA)
        caption = f"{det['label']} {det['score']:.2f}"
        cv2.putText(view, caption, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, caption, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (40, 220, 50), 1, cv2.LINE_AA)
    return view


# ChessSight's published board reader uses the contact point near the bottom
# of the detection box, not the box centre or the exact lower edge.
FOOT_X = 0.50
FOOT_Y = 0.96
POSITION_THRESHOLD = 0.08
CANDIDATE_FLOOR = 0.01
CROP_SIDE_MARGIN = 0.10
CROP_TOP_MARGIN = 0.30


def _board_crop(corners: np.ndarray, width: int, height: int) -> tuple[int, int, int]:
    """Square crop around the board, leaving extra room above the far rank."""
    points = np.asarray(corners, dtype=np.float64)
    x0, x1 = float(points[:, 0].min()), float(points[:, 0].max())
    y0, y1 = float(points[:, 1].min()), float(points[:, 1].max())
    bw, bh = x1 - x0, y1 - y0
    x0 -= CROP_SIDE_MARGIN * bw
    x1 += CROP_SIDE_MARGIN * bw
    y0 -= CROP_TOP_MARGIN * bh
    y1 += CROP_SIDE_MARGIN * bh
    side = int(round(min(max(x1 - x0, y1 - y0), width, height)))
    side = max(32, side)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    left = int(round(min(max(cx - side / 2.0, 0), max(0, width - side))))
    top = int(round(min(max(cy - side / 2.0, 0), max(0, height - side))))
    return left, top, side


def _square_scores(
    detections: list[dict[str, Any]],
    image_to_board: np.ndarray,
    *,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    mirror_side: float | None = None,
) -> dict[tuple[int, int, str], float]:
    """Return the strongest proposal for every (square, class) in one view."""
    scores: dict[tuple[int, int, str], float] = {}
    square_size = BOARD_SIZE / 8.0
    for det in detections:
        label = str(det.get("label", ""))
        letter = FEN_MAP.get(label)
        if letter is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in det["box"]]
        # Undo a horizontal mirror before shifting the crop back into the photo.
        if mirror_side is not None:
            x1, x2 = mirror_side - x2, mirror_side - x1
        x1 += offset_x; x2 += offset_x
        y1 += offset_y; y2 += offset_y
        px = x1 + FOOT_X * (x2 - x1)
        py = y1 + FOOT_Y * (y2 - y1)
        point = cv2.perspectiveTransform(
            np.asarray([[[px, py]]], dtype=np.float32), image_to_board
        )[0, 0]
        bx, by = float(point[0]), float(point[1])
        if not (0.0 <= bx < BOARD_SIZE and 0.0 <= by < BOARD_SIZE):
            continue
        col = min(7, int(bx // square_size))
        row = min(7, int(by // square_size))
        key = (row, col, letter)
        scores[key] = max(scores.get(key, 0.0), float(det["score"]))
    return scores


def _fuse_square_scores(
    passes: list[tuple[dict[tuple[int, int, str], float], float]]
) -> dict[tuple[int, int, str], float]:
    """Weighted mean; a class missing from a pass receives zero in that pass."""
    total_weight = sum(weight for _, weight in passes)
    fused: dict[tuple[int, int, str], float] = {}
    for score_map, weight in passes:
        for key, score in score_map.items():
            fused[key] = fused.get(key, 0.0) + weight * score
    if total_weight <= 0:
        return {}
    return {key: value / total_weight for key, value in fused.items()}


def _grid_from_scores(scores: dict[tuple[int, int, str], float]):
    """Choose the highest weighted/fused model score on each square.

    Deliberately no chess-material cap is applied here. A photograph may show a
    promoted piece, a captured-piece set, or an unusual legal position; image-model
    scores must not be silently overwritten by an assumed starting-piece inventory.
    """
    grid = [["." for _ in range(8)] for _ in range(8)]
    best_score = [[0.0 for _ in range(8)] for _ in range(8)]
    candidates: dict[tuple[int, int], dict[str, float]] = {}
    for (row, col, letter), score in scores.items():
        score = float(score)
        if score >= CANDIDATE_FLOOR:
            candidates.setdefault((row, col), {})[letter] = score
        if score >= POSITION_THRESHOLD and score > best_score[row][col]:
            best_score[row][col] = score
            grid[row][col] = letter
    return grid, candidates


def _top_prediction_per_square(scores: dict[tuple[int, int, str], float]):
    """Top class and score for each occupied square in one detector view."""
    result: dict[tuple[int, int], tuple[str, float]] = {}
    for (row, col, letter), score in scores.items():
        key = (row, col)
        old = result.get(key)
        if old is None or float(score) > old[1]:
            result[key] = (letter, float(score))
    return result


def _square_luminance(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    samples = np.zeros((8, 8), dtype=np.float32)
    # Sample the interior 50% of each square, avoiding borders and piece bases.
    offsets = (0.30, 0.50, 0.70)
    size = BOARD_SIZE / 8.0
    for row in range(8):
        for col in range(8):
            values = []
            for dy in offsets:
                for dx in offsets:
                    x = int(np.clip((col + dx) * size, 0, BOARD_SIZE - 1))
                    y = int(np.clip((row + dy) * size, 0, BOARD_SIZE - 1))
                    values.append(float(gray[y, x]))
            samples[row, col] = float(np.mean(values))
    return samples


def _orientation_evidence(grid: list[list[str]], luminance: np.ndarray):
    """Resolve 90-degree ambiguity with square parity and 180-degree with pawns."""
    candidates = []
    for turns in range(4):
        rotated_grid = np.rot90(np.asarray(grid), turns).tolist()
        rotated_lum = np.rot90(luminance, turns)
        ranks, files = np.indices((8, 8))
        even = (ranks + files) % 2 == 0
        colour = float(rotated_lum[even].mean() - rotated_lum[~even].mean())
        near = far = 0
        total_pawns = own_pawns = 0
        white_seen = black_seen = False
        for r in range(8):
            for c in range(8):
                piece = rotated_grid[r][c]
                if piece == ".":
                    continue
                bottom = r >= 4
                if piece.isupper():
                    near += int(bottom); far += int(not bottom)
                else:
                    near += int(not bottom); far += int(bottom)
                if piece == "P":
                    white_seen = True; total_pawns += 1; own_pawns += int(bottom)
                elif piece == "p":
                    black_seen = True; total_pawns += 1; own_pawns += int(not bottom)
        total_pieces = near + far
        pieces_score = (near - far) / total_pieces if total_pieces else 0.0
        pawns_score = ((2 * own_pawns - total_pawns) / total_pawns
                       if total_pawns and white_seen and black_seen else 0.0)
        candidates.append((turns, colour, pieces_score + 2.0 * pawns_score, pieces_score, pawns_score))
    best_colour = max(item[1] for item in candidates)
    survivors = [item for item in candidates if item[1] >= best_colour - 0.02]
    chosen = max(survivors, key=lambda item: item[2])
    runner_up = max((item[2] for item in survivors if item[0] != chosen[0]), default=None)
    evidence = {
        "square_colour": chosen[1], "piece_position": chosen[3], "pawn_position": chosen[4],
        "orientation_margin": 0.0 if runner_up is None else chosen[2] - runner_up,
        "orientation_candidates": len(survivors),
    }
    return int(chosen[0]), evidence


def _rotate_square(row: int, col: int, turns: int) -> tuple[int, int]:
    if turns == 0:
        return row, col
    if turns == 1:
        return 7 - col, row
    if turns == 2:
        return 7 - row, 7 - col
    return col, 7 - row


def _rotate_point(x: float, y: float, turns: int) -> tuple[float, float]:
    if turns == 0:
        return x, y
    if turns == 1:
        return y, BOARD_SIZE - 1 - x
    if turns == 2:
        return BOARD_SIZE - 1 - x, BOARD_SIZE - 1 - y
    return BOARD_SIZE - 1 - y, x


def run_recognition(image_bgr: np.ndarray):
    """ChessSight's three-pass reading recipe: full frame, board crop, mirrored crop.

    Uses calibrated detector scores, contact-point anchoring, weighted three-view
    fusion and image-based orientation. The highest fused model score is preserved;
    chess-material assumptions never rewrite a visual prediction.
    """
    height, width = image_bgr.shape[:2]
    original_all, original_pieces = detect_cached(image_bgr)
    corners, corner_method = detect_board_corners(image_bgr, original_all)

    destination = np.array(
        [[0, 0], [BOARD_SIZE - 1, 0], [BOARD_SIZE - 1, BOARD_SIZE - 1], [0, BOARD_SIZE - 1]],
        dtype=np.float32,
    )
    image_to_board = cv2.getPerspectiveTransform(order_corners(corners), destination)
    if not np.isfinite(image_to_board).all():
        raise RuntimeError("A sakktábla perspektívakorrekciója nem sikerült.")
    rectified = cv2.warpPerspective(
        image_bgr, image_to_board, (BOARD_SIZE, BOARD_SIZE), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Pass 1: original photo. Passes 2/3: square crop with room for tall pieces,
    # and its horizontal mirror. Mirroring creates a genuinely different model view;
    # sharpening the same rectified pixels is not an equivalent independent pass.
    left, top, side = _board_crop(corners, width, height)
    crop = image_bgr[top:top + side, left:left + side]
    if crop.size == 0 or crop.shape[0] != crop.shape[1]:
        raise RuntimeError("Nem sikerült négyzetes kivágást készíteni a sakktábláról.")
    mirrored_crop = cv2.flip(crop, 1)
    crop_all, _ = detect_cached(crop)
    mirror_all, _ = detect_cached(mirrored_crop)

    original_scores = _square_scores(original_all, image_to_board)
    crop_scores = _square_scores(crop_all, image_to_board, offset_x=left, offset_y=top)
    mirror_scores = _square_scores(
        mirror_all, image_to_board, offset_x=left, offset_y=top, mirror_side=float(side)
    )
    full_weight = (side / float(min(width, height))) ** 2
    fused_scores = _fuse_square_scores([
        (original_scores, full_weight), (crop_scores, 1.0), (mirror_scores, 1.0)
    ])

    # Keep the best model-proposed class on every square. No material correction:
    # promotions and arbitrary mid-game positions make starting-material caps unsafe.
    grid, candidates = _grid_from_scores(fused_scores)
    per_view_maps = [
        _top_prediction_per_square(original_scores),
        _top_prediction_per_square(crop_scores),
        _top_prediction_per_square(mirror_scores),
    ]

    # Resolve board rotation using the checkerboard's light/dark parity first,
    # then piece placement and pawn direction for the remaining 180-degree tie.
    turns, orientation_info = _orientation_evidence(grid, _square_luminance(rectified))
    grid = np.rot90(np.asarray(grid), turns).tolist()
    rectified = np.rot90(rectified, turns).copy()
    # Re-map candidate and confidence information to the chosen orientation.
    oriented_candidates: dict[tuple[int, int], dict[str, float]] = {}
    for (row, col), class_scores in candidates.items():
        new_key = _rotate_square(row, col, turns)
        oriented_candidates[new_key] = class_scores

    oriented_view_maps: list[dict[tuple[int, int], tuple[str, float]]] = []
    for view_map in per_view_maps:
        rotated_view: dict[tuple[int, int], tuple[str, float]] = {}
        for (row, col), prediction in view_map.items():
            rotated_view[_rotate_square(row, col, turns)] = prediction
        oriented_view_maps.append(rotated_view)

    annotated = rectified.copy()
    square_size = BOARD_SIZE / 8.0
    for i in range(9):
        point = min(BOARD_SIZE - 1, int(round(i * square_size)))
        cv2.line(annotated, (point, 0), (point, BOARD_SIZE - 1), (255, 220, 0), 1, cv2.LINE_AA)
        cv2.line(annotated, (0, point), (BOARD_SIZE - 1, point), (255, 220, 0), 1, cv2.LINE_AA)

    uncertain_details = []
    uncertain_keys = set()
    for row in range(8):
        for col in range(8):
            piece = grid[row][col]
            if piece == ".":
                continue
            class_scores = oriented_candidates.get((row, col), {})
            ranked = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
            top_score = ranked[0][1] if ranked else 0.0
            selected_score = class_scores.get(piece, 0.0)
            second_piece, second_score = ranked[1] if len(ranked) > 1 else ("—", 0.0)
            view_predictions = [view.get((row, col)) for view in oriented_view_maps]
            view_labels = [item[0] if item and item[1] >= POSITION_THRESHOLD else "—"
                           for item in view_predictions]
            view_agreement = sum(label == piece for label in view_labels)
            view_summary = []
            for view_name, prediction in zip(("Teljes kép", "Nagyított kép", "Tükrözött kép"), view_predictions):
                if prediction and prediction[1] >= POSITION_THRESHOLD:
                    view_summary.append(f"{view_name}: {prediction[0]} ({prediction[1]:.2f})")
                else:
                    view_summary.append(f"{view_name}: nincs találat")
            # Highlight low margin, low confidence, or disagreement between views.
            is_uncertain = (
                top_score - second_score < 0.07
                or selected_score < 0.20
                or view_agreement < 2
            )
            if is_uncertain:
                uncertain_keys.add((row, col))
                uncertain_details.append({
                    "Mező": f"{chr(97 + col)}{8 - row}",
                    "Kiválasztott bábu": piece,
                    "Összesített pont": round(selected_score, 3),
                    "Második tipp": second_piece,
                    "Második pont": round(second_score, 3),
                    "Nézetegyezés": f"{view_agreement}/3",
                    "Nézetenkénti tippek": " | ".join(view_summary),
                })
            x = int(round((col + 0.5) * square_size))
            y = int(round((row + FOOT_Y) * square_size))
            colour = (0, 190, 0) if piece.isupper() else (255, 130, 10)
            cv2.circle(annotated, (x, y), 6, colour, -1, cv2.LINE_AA)
            caption = f"{piece} {chr(97 + col)}{8 - row} {selected_score:.2f}"
            cv2.putText(annotated, caption, (min(BOARD_SIZE - 150, x + 7), max(18, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, caption, (min(BOARD_SIZE - 150, x + 7), max(18, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1, cv2.LINE_AA)
            if (row, col) in uncertain_keys:
                cv2.circle(annotated, (x, y), 14, (0, 220, 255), 3, cv2.LINE_AA)

    fen = make_fen(grid)
    board_obj = chess.Board(fen)
    original_annotated = draw_original(image_bgr, corners, original_pieces)
    occupied = [piece for row in grid for piece in row if piece != "."]
    metrics = {
        "corner_method": corner_method,
        "piece_detections": sum(d["label"] in FEN_MAP for d in original_all)
                            + sum(d["label"] in FEN_MAP for d in crop_all)
                            + sum(d["label"] in FEN_MAP for d in mirror_all),
        "occupied_squares": len(occupied),
        "uncertain_squares": len(uncertain_keys),
        "orientation_turns": turns,
        "orientation_info": orientation_info,
        "fen_valid": board_obj.is_valid(),
        "grid": grid,
        "confidence_mean": float(np.mean([oriented_candidates.get((r, c), {}).get(grid[r][c], 0.0)
                                           for r in range(8) for c in range(8) if grid[r][c] != "."]))
                            if occupied else 0.0,
        "view_agreement_mean": float(np.mean([
            sum((view.get((r, c), ("—", 0.0))[0] == grid[r][c])
                and view.get((r, c), ("—", 0.0))[1] >= POSITION_THRESHOLD
                for view in oriented_view_maps)
            for r in range(8) for c in range(8) if grid[r][c] != "."
        ])) if occupied else 0.0,
        "uncertain_details": uncertain_details,
        "views_used": 3,
        "full_frame_weight": full_weight,
        "classifier_name": "ChessSight RT-DETR v0.2.0; official three-pass fusion recipe",
    }
    return original_annotated, annotated, fen, metrics


PIPELINE_REVISION = "2026-11-09-chesssight-threepass-no-material-override-v7"


@st.cache_data(show_spinner=False, max_entries=8)
def cached_full_recognition(image_bytes: bytes, pipeline_revision: str):
    """Cache the complete result for the exact same uploaded file, including corner geometry."""
    image_bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError("Nem sikerült beolvasni a feltöltött képet.")
    return run_recognition(image_bgr)


st.title("♟️ Automatikus sakkállás-felismerő")
st.write(
    "Tölts fel egy fényképet a fizikai sakktábláról. A program automatikusan megtalálja a négy sarkot, "
    "majd három, egymást kiegészítő képi nézetből olvassa le a pozíciót."
)
st.caption("ChessSight RT-DETR v0.2.0 + Swin sarokmodell. Három nézet súlyozott pontszám-fúziója. A végső mezőcímkét mindig a legmagasabb összesített modellpontszám adja; sakkanyag-korlát nem írhatja felül a felismerést.")
st.warning("Oktatási prototípus: a ChessSight detector és corner modell CC BY-NC 4.0 licencű, nem kereskedelmi használatra. Saját fényképeken az eredményt ellenőrizni kell.")

uploaded = st.file_uploader("Sakktábla-fotó feltöltése", type=["jpg", "jpeg", "png", "webp", "bmp"])
run_button = st.button("Automatikus felismerés", type="primary", disabled=uploaded is None)

if uploaded is not None:
    image_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("Nem sikerült beolvasni a feltöltött képet.")
        st.stop()
    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Feltöltött kép", use_container_width=True)

if run_button and uploaded is not None:
    image_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    status = st.status("A modell betöltése és a felismerés fut…", expanded=True)
    try:
        status.write("ChessSight RT-DETR bábufelismerő betöltése…")
        status.write("ChessSight Swin sarokmodell betöltése (első alkalommal kb. 111 MB)…")
        status.write("Automatikus sarokkeresés, perspektívakorrekció és háromnézetes bábufelismerés…")
        with st.spinner("Automatikus felismerés folyamatban…"):
            original_annotated, board_annotated, fen, metrics = cached_full_recognition(uploaded.getvalue(), PIPELINE_REVISION)
        status.update(label="Felismerés befejezve", state="complete", expanded=False)

        left, right = st.columns(2)
        with left:
            st.subheader("Automatikusan megtalált tábla az eredeti képen")
            st.image(cv2.cvtColor(original_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        with right:
            st.subheader("Kiegyenesített digitális tábla")
            st.image(cv2.cvtColor(board_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.subheader("FEN-kód")
        st.code(fen, language="text")
        st.download_button("FEN letöltése", data=fen + "\n", file_name="position.fen", mime="text/plain")
        cols = st.columns(6)
        cols[0].metric("Elfoglalt mezők", f"{metrics['occupied_squares']} / 64")
        cols[1].metric("Bábujelöltek a 3 nézetben", metrics["piece_detections"])
        cols[2].metric("Átlagos bizalom", f"{metrics['confidence_mean']:.2f}")
        cols[3].metric("Bizonytalan mezők", metrics["uncertain_squares"])
        cols[4].metric("FEN szabályossága", "Érvényes" if metrics["fen_valid"] else "Ellenőrzendő")
        cols[5].metric("Tájolás", f"{metrics['orientation_turns'] * 90}°")
        st.caption(f"Sarokdetektálási módszer: {metrics['corner_method']} · teljes fotó súlya: {metrics['full_frame_weight']:.2f}")
        st.caption(
            "Tájolási bizonyíték – mezőszín: "
            f"{metrics['orientation_info']['square_colour']:.3f}, "
            f"bábuelhelyezkedés: {metrics['orientation_info']['piece_position']:.3f}, "
            f"gyalogok: {metrics['orientation_info']['pawn_position']:.3f}."
        )
        st.caption(
            f"Három nézet átlagos egyezése: {metrics['view_agreement_mean']:.2f}/3. "
            "A mező címkéjét kizárólag a három nézet súlyozott modellpontszáma határozza meg; "
            "sakkanyag-alapú címkecsere nincs."
        )
        st.subheader("Felismerési rács")
        for row in metrics["grid"]:
            st.code(" ".join(row), language="text")
        if metrics["uncertain_squares"]:
            st.warning("A sárga karikás mezőkön kicsi az első és második osztálypontszám különbsége, alacsony az összesített bizalom, vagy a három képi nézet nem ért egyet. A táblázat megmutatja mindhárom nézet külön tippjét.")
            st.dataframe(metrics["uncertain_details"], use_container_width=True, hide_index=True)
        if metrics["occupied_squares"] < 12:
            st.warning("Kevés bábu került a digitális táblára. Ellenőrizd a sarokdetektálást és próbálj élesebb, teljes táblás fotót.")
        if not metrics["fen_valid"]:
            st.warning("A felismert állás sakk-szabályok szerint nem teljesen érvényes. A modell kihagyhatott vagy félreosztályozhatott bábukat.")
        st.info("A lépő fél alapértelmezés szerint fehér. Egyetlen fényképből a lépő fél, a sáncolási jogok és az en-passant jog nem állapíthatók meg megbízhatóan.")
    except Exception as exc:
        status.update(label="A felismerés nem sikerült", state="error", expanded=True)
        st.error(str(exc))
        st.markdown(
            "**Hibaelhárítás:** ellenőrizd, hogy telepítetted-e a `requirements_chess_app.txt` csomagjait, "
            "és hogy az első modellletöltés alatt van-e internetkapcsolat. A program nem kér kézi sarokkijelölést."
        )

with st.expander("Modellek, módszer és korlátok"):
    st.markdown(
        "- **Elsődleges bábufelismerő:** [ChessSight RT-DETR v0.2.0](https://huggingface.co/tchauffi/chesssight-rtdetr-v0.2.0). "
        "A három nézet súlyozott pontszámait egyesítjük. A jelenlegi alkalmazás szándékosan nem alkalmaz sakkanyag-korlátot, mert az felülírhatja a képi modell magasabb pontszámú döntését, és promóciós vagy szokatlan állásoknál hibát okozhat.\n"
        "- **Sarokkeresés:** [ChessSight Swin corner heatmap v1](https://huggingface.co/tchauffi/chesssight-corner-heatmap-v1), automatikus 4 pont; OpenCV és photochess tartalék eljárással.\n"
        "- **Döntés:** ChessSight RT-DETR pontszámok súlyozott fúziója három nézetből (teljes kép, négyzetes board crop, tükrözött crop), `FOOT_Y=0.96` bábu-talppont. Minden mezőn a legmagasabb összesített modellpontszám nyer; nincs sakkanyag-alapú címkecsere. A bizonytalan mezők táblázata nézetenként megmutatja a modell tippjét.\n"
        "- **Régi `detector.py` és `chess_board.py`:** nem szükséges módosítani őket.\n"
        "- **Licenc:** mindkét ChessSight modell CC BY-NC 4.0; kereskedelmi felhasználásra nem használhatók."
    )
