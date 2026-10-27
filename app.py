"""Automatic chessboard photo -> digital position/FEN.

Version 2.0
- Uses ChessSight RT-DETR as the primary pretrained piece detector.
- Finds board corners automatically: OpenCV checker-grid detection first,
  the detector's corner proposals second, and the older photochess pose model
  as a final learned fallback. There is no manual corner selection.

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
        if calibrated < 0.02:
            continue
        x1, y1, x2, y2 = xyxy
        all_dets.append({"label": label, "score": calibrated, "raw_score": raw_score,
                         "box": (x1, y1, x2, y2)})
    pieces = [d for d in all_dets if d["label"] in FEN_MAP and d["score"] >= POSITION_CONFIDENCE]
    return all_dets, pieces


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return TL, TR, BR, BL in image coordinates."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    first = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -first, axis=0)
    # If it is not convex, caller should reject it.
    return ordered.astype(np.float32)


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


def detect_board_corners(image_bgr: np.ndarray, detections: list[dict[str, Any]]) -> tuple[np.ndarray, str]:
    board_bbox = get_board_bbox(detections)
    quad, method = detect_corners_from_checker_grid(image_bgr, board_bbox)
    if quad is not None:
        return quad, method
    quad = detect_corners_from_model_boxes(detections, image_bgr.shape, board_bbox)
    if quad is not None:
        return quad, "ChessSight corner-box proposals"
    quad = detect_corners_from_pose_model(image_bgr, board_bbox)
    if quad is not None:
        return quad, "fallback photochess pose model"
    raise RuntimeError(
        "Nem sikerült automatikusan meghatározni a sakktábla négy sarkát. "
        "A program már több módszert kipróbált (OpenCV rácsdetektálás, ChessSight sarokjelöltek, "
        "fallback pose-modell). Készíts olyan képet, amelyen a teljes 8×8-as rács élesen látszik, "
        "és a tábla nincs erősen levágva. Kézi sarokkijelölést nem kérünk."
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


def run_recognition(image_bgr: np.ndarray, processor, model, device):
    all_dets, piece_dets = run_chesssight(image_bgr, processor, model, device)
    corners, corner_method = detect_board_corners(image_bgr, all_dets)
    dest = np.array([[0, 0], [BOARD_SIZE - 1, 0], [BOARD_SIZE - 1, BOARD_SIZE - 1], [0, BOARD_SIZE - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(order_corners(corners).astype(np.float32), dest)
    rectified = cv2.warpPerspective(image_bgr, H, (BOARD_SIZE, BOARD_SIZE), flags=cv2.INTER_CUBIC)

    size = BOARD_SIZE / 8.0
    candidates: list[dict[str, Any]] = []
    for det in piece_dets:
        x1, y1, x2, y2 = det["box"]
        # Piece-to-square anchor: bottom-centre is more stable than the visual centre of tall pieces.
        foot = np.array([[[0.5 * (x1 + x2), y2]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(foot, H).reshape(2)
        x, y = float(warped[0]), float(warped[1])
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            continue
        col = min(7, int(x // size))
        row = min(7, int(y // size))
        candidates.append({**det, "row": row, "col": col, "xy": (x, y)})

    if not candidates:
        raise RuntimeError("A bábufelismerő nem talált olyan bábut, amely a felismert tábla területére esik.")

    # Keep the best detection per square; this is intentionally recall-oriented as the model card recommends.
    best: dict[tuple[int, int], dict[str, Any]] = {}
    for det in candidates:
        key = (det["row"], det["col"])
        if key not in best or det["score"] > best[key]["score"]:
            best[key] = det
    # A normal chess position has at most 32 pieces on the board. Drop the weakest spill-over detections.
    if len(best) > 32:
        keep_keys = {key for key, _ in sorted(best.items(), key=lambda item: item[1]["score"], reverse=True)[:32]}
        best = {key: val for key, val in best.items() if key in keep_keys}

    # Detect whether the board must be rotated 180 degrees so White is at the bottom.
    white_bottom = sum(d["score"] for d in best.values() if d["label"] in WHITE_NAMES and d["row"] >= 4)
    white_bottom += sum(d["score"] for d in best.values() if d["label"] in BLACK_NAMES and d["row"] < 4)
    white_top = sum(d["score"] for d in best.values() if d["label"] in WHITE_NAMES and d["row"] < 4)
    white_top += sum(d["score"] for d in best.values() if d["label"] in BLACK_NAMES and d["row"] >= 4)
    flipped_180 = white_top > white_bottom
    if flipped_180:
        rectified = cv2.rotate(rectified, cv2.ROTATE_180)
        best = {(7 - row, 7 - col): {**det, "row": 7 - row, "col": 7 - col,
                                    "xy": (BOARD_SIZE - 1 - det["xy"][0], BOARD_SIZE - 1 - det["xy"][1])}
                for (row, col), det in best.items()}

    grid = [["." for _ in range(8)] for _ in range(8)]
    annotated = rectified.copy()
    for i in range(9):
        p = min(BOARD_SIZE - 1, int(round(i * size)))
        cv2.line(annotated, (p, 0), (p, BOARD_SIZE - 1), (255, 220, 0), 1, cv2.LINE_AA)
        cv2.line(annotated, (0, p), (BOARD_SIZE - 1, p), (255, 220, 0), 1, cv2.LINE_AA)
    for (row, col), det in best.items():
        piece = FEN_MAP[det["label"]]
        grid[row][col] = piece
        x, y = [int(round(v)) for v in det["xy"]]
        color = (30, 210, 30) if piece.isupper() else (255, 150, 20)
        square = f"{chr(97 + col)}{8 - row}"
        cv2.circle(annotated, (x, y), 7, color, -1, cv2.LINE_AA)
        cv2.putText(annotated, f"{piece} {square} {det['score']:.2f}",
                    (min(BOARD_SIZE - 150, x + 7), max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, color, 2, cv2.LINE_AA)

    fen = make_fen(grid)
    chess_board = chess.Board(fen)
    original_annotated = draw_original(image_bgr, corners, all_dets)
    metrics = {
        "corner_method": corner_method,
        "piece_detections": len(piece_dets),
        "occupied_squares": len(best),
        "flipped_180": flipped_180,
        "fen_valid": chess_board.is_valid(),
        "grid": grid,
        "confidence_mean": float(np.mean([d["score"] for d in best.values()])) if best else 0.0,
    }
    return original_annotated, annotated, fen, metrics


st.title("♟️ Automatikus sakkállás-felismerő")
st.write(
    "Tölts fel egy fényképet a fizikai sakktábláról. A program kézi kijelölés nélkül keresi a rácsot/sarkokat, "
    "majd egy újabb, előre betanított modell osztályozza a bábukat és FEN-kódot készít."
)
st.caption("ChessSight RT-DETR v0.2.0 + automatikus OpenCV-rácsdetektálás. Első indításkor az új modell letöltődik a Hugging Face gyorsítótárába.")
st.warning("Oktatási prototípus: a ChessSight modell CC BY-NC 4.0 licencű, nem kereskedelmi használatra. A modell szintetikus adatokon tanult; valós fényképeken a felismerést ellenőrizni kell.")

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
        status.write("ChessSight pretrained modell betöltése (első alkalommal internetről letöltődik)…")
        processor, model, device = load_chesssight_model()
        status.write("Sakktábla-rács keresése és bábufelismerés…")
        with st.spinner("Automatikus felismerés folyamatban…"):
            original_annotated, board_annotated, fen, metrics = run_recognition(image_bgr, processor, model, device)
        status.update(label="Felismerés befejezve", state="complete", expanded=False)

        left, right = st.columns(2)
        with left:
            st.subheader("Detektálás az eredeti képen")
            st.image(cv2.cvtColor(original_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        with right:
            st.subheader("Kiegyenesített digitális tábla")
            st.image(cv2.cvtColor(board_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.subheader("FEN-kód")
        st.code(fen, language="text")
        st.download_button("FEN letöltése", data=fen + "\n", file_name="position.fen", mime="text/plain")
        cols = st.columns(5)
        cols[0].metric("Elfoglalt mezők", f"{metrics['occupied_squares']} / 64")
        cols[1].metric("Modell találatai", metrics["piece_detections"])
        cols[2].metric("Átlagos bizalom", f"{metrics['confidence_mean']:.2f}")
        cols[3].metric("FEN szabályossága", "Érvényes" if metrics["fen_valid"] else "Ellenőrzendő")
        cols[4].metric("180° forgatás", "Igen" if metrics["flipped_180"] else "Nem")
        st.caption(f"Sarokdetektálási módszer: {metrics['corner_method']}")
        st.subheader("Felismerési rács")
        for row in metrics["grid"]:
            st.code(" ".join(row), language="text")
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
        "- **Bábufelismerés:** [ChessSight RT-DETR v0.2.0 modellkártya](https://huggingface.co/tchauffi/chesssight-rtdetr-v0.2.0). "
        "A modellkártya 306 valós fotóból álló ChessReD teszthalmazon 71,24% teljes állás-pontosságot közöl, de a kis bábuk felismerése nehezebb.\n"
        "- **Sarokkeresés:** először OpenCV sakkrács-illesztés (részleges 7×N mintákból a teljes 8×8-as tábla extrapolálása), "
        "utána a detektor sarokjavaslatai, végül a régi photochess pose modell.\n"
        "- **Régi `detector.py` és `chess_board.py`:** ezekhez nem nyúlunk; az új `app.py` önálló pipeline.\n"
        "- **Licenc:** a ChessSight súlyai CC BY-NC 4.0 licencűek; kereskedelmi felhasználásra más megoldás szükséges."
    )
