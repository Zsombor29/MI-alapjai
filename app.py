"""Automatic chessboard photo -> FEN prototype.

Uses two pretrained YOLO models from 2T1F/photochess:
- corner.pt: detects the four board corners without manual clicks
- pieces.pt: detects and classifies chess pieces

Run: python -m streamlit run app.py
The model files are downloaded on first use into models/ and should not be committed to Git.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import chess
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
CORNER_MODEL_PATH = MODEL_DIR / "photochess_corner.pt"
PIECE_MODEL_PATH = MODEL_DIR / "photochess_pieces.pt"
CORNER_MODEL_URL = "https://raw.githubusercontent.com/2T1F/photochess/main/corner.pt"
PIECE_MODEL_URL = "https://raw.githubusercontent.com/2T1F/photochess/main/pieces.pt"
MODEL_IMG_SIZE = 800
BOARD_SIZE = 800
PIECE_CONF = 0.25
CORNER_CONF = 0.20

FEN_MAP = {
    "white_pawn": "P", "white_knight": "N", "white_bishop": "B",
    "white_rook": "R", "white_queen": "Q", "white_king": "K",
    "black_pawn": "p", "black_knight": "n", "black_bishop": "b",
    "black_rook": "r", "black_queen": "q", "black_king": "k",
}
WHITE_NAMES = {"white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king"}
BLACK_NAMES = {"black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king"}

st.set_page_config(page_title="Automatikus sakkállás-felismerő", page_icon="♟️", layout="wide")


def download_if_missing(path: Path, url: str, label: str, progress_slot: Any) -> None:
    """Download one model using a temporary file so partial downloads aren't reused."""
    if path.exists() and path.stat().st_size > 1_000_000:
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    progress_slot.write(f"Modell letöltése: {label}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response, tmp_path.open("wb") as out:
            total = int(response.headers.get("Content-Length", "0") or 0)
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    progress_slot.progress(min(downloaded / total, 1.0), text=f"{label}: {downloaded / total:.0%}")
        if tmp_path.stat().st_size < 1_000_000:
            raise RuntimeError("A letöltött fájl túl kicsi; a modell letöltése valószínűleg nem sikerült.")
        tmp_path.replace(path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Nem sikerült letölteni a(z) {label} modellt. Ellenőrizd az internetkapcsolatot, "
            f"majd próbáld újra. Részlet: {exc}"
        ) from exc


@st.cache_resource(show_spinner=False)
def load_models(corner_path: str, piece_path: str) -> tuple[YOLO, YOLO]:
    return YOLO(corner_path), YOLO(piece_path)


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order a convex four-corner set clockwise from the visually top-left point."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]  # TL, TR, BR, BL for ordinary image coordinates
    first = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -first, axis=0).astype(np.float32)


def detect_board_corners(image_bgr: np.ndarray, corner_model: YOLO) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Find 4 keypoints automatically. Returns ordered corners, all keypoint candidates, confidences."""
    result = corner_model.predict(
        source=image_bgr, imgsz=MODEL_IMG_SIZE, conf=CORNER_CONF, iou=0.45, verbose=False
    )[0]
    if result.keypoints is None or result.keypoints.data is None:
        raise RuntimeError(
            "A sarokfelismerő nem adott vissza sarokpontokat. Próbálj olyan képet, amelyen a teljes tábla látható."
        )

    keypoints = result.keypoints.data.detach().cpu().numpy().reshape(-1, 3)
    points = keypoints[:, :2]
    kp_conf = keypoints[:, 2]
    if result.boxes is not None and result.boxes.conf is not None and len(result.boxes.conf) == len(points):
        det_conf = result.boxes.conf.detach().cpu().numpy()
        scores = kp_conf * det_conf
    else:
        scores = kp_conf

    valid = np.isfinite(points).all(axis=1) & np.isfinite(scores) & (scores >= 0.05)
    points, scores = points[valid], scores[valid]
    if len(points) < 4:
        raise RuntimeError(
            f"Csak {len(points)} sakktáblasarkot sikerült megtalálni a szükséges 4 helyett. "
            "Használj élesebb képet, amin a teljes sakktábla és minden sarka látszik."
        )

    # If the model returns more than 4 candidates, retain the four most confident ones.
    chosen = np.argsort(scores)[-4:]
    selected_points = points[chosen]
    selected_scores = scores[chosen]
    # Reject near-duplicate keypoints (e.g., the same corner detected twice).
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(selected_points[i] - selected_points[j]) < min(image_bgr.shape[:2]) * 0.08:
                raise RuntimeError(
                    "A sarokmodell két egymáshoz közeli pontot jelölt meg ugyanannak a saroknak. "
                    "Próbálj élesebb, a teljes táblát mutató képet."
                )
    corners = order_corners(selected_points)
    return corners, points, selected_scores.tolist()


def get_homography(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    destination = np.array(
        [[0, 0], [BOARD_SIZE - 1, 0], [BOARD_SIZE - 1, BOARD_SIZE - 1], [0, BOARD_SIZE - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    if not np.isfinite(homography).all():
        raise RuntimeError("Nem sikerült kiszámítani a perspektívakorrekciót.")
    return homography, destination


def normalize_class_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def build_placement(grid: list[list[str]]) -> str:
    ranks: list[str] = []
    for row in grid:
        encoded = ""
        empties = 0
        for piece in row:
            if piece == ".":
                empties += 1
                continue
            if empties:
                encoded += str(empties)
                empties = 0
            encoded += piece
        if empties:
            encoded += str(empties)
        ranks.append(encoded)
    return "/".join(ranks)


def make_fen(grid: list[list[str]], side_to_move: str = "w") -> str:
    # Castling/en-passant rights and move counters cannot be inferred from a single still image.
    return f"{build_placement(grid)} {side_to_move} - - 0 1"


def put_piece_name(name_value: Any, class_id: int) -> str:
    if isinstance(name_value, dict):
        raw = name_value.get(class_id, name_value.get(str(class_id), str(class_id)))
    else:
        raw = name_value[class_id] if isinstance(name_value, (list, tuple)) and class_id < len(name_value) else str(class_id)
    return normalize_class_name(raw)


def detect_pieces(image_bgr: np.ndarray, piece_model: YOLO):
    result = piece_model.predict(
        source=image_bgr, imgsz=MODEL_IMG_SIZE, conf=PIECE_CONF, iou=0.35, max_det=100, verbose=False
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return [], result

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confs = result.boxes.conf.detach().cpu().numpy()
    dets: list[dict[str, Any]] = []
    for box, cls, conf in zip(xyxy, classes, confs):
        name = put_piece_name(result.names, int(cls))
        if name not in FEN_MAP:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        # The bottom centre is where the piece meets its square; box centre is too high for tall pieces.
        foot = np.array([[[((x1 + x2) / 2.0), y2]]], dtype=np.float32)
        dets.append({"name": name, "fen": FEN_MAP[name], "conf": float(conf), "box": (x1, y1, x2, y2), "foot": foot})
    return dets, result


def draw_original(image_bgr: np.ndarray, corners: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    canvas = image_bgr.copy()
    for i, (x, y) in enumerate(corners):
        cv2.circle(canvas, (int(round(x)), int(round(y))), 9, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, f"C{i + 1}", (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 210, 40), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{det['name']} {det['conf']:.2f}", (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 210, 40), 2, cv2.LINE_AA)
    return canvas


def run_recognition(image_bgr: np.ndarray, corner_model: YOLO, piece_model: YOLO):
    corners, _, corner_scores = detect_board_corners(image_bgr, corner_model)
    homography, _ = get_homography(corners)
    rectified = cv2.warpPerspective(image_bgr, homography, (BOARD_SIZE, BOARD_SIZE), flags=cv2.INTER_CUBIC)

    detections, _ = detect_pieces(image_bgr, piece_model)
    if not detections:
        raise RuntimeError("A bábufelismerő egyetlen bábut sem talált. Próbálj világos, éles képet a teljes tábláról.")

    foot_points = np.concatenate([d["foot"] for d in detections], axis=0).astype(np.float32)
    warped_points = cv2.perspectiveTransform(foot_points, homography).reshape(-1, 2)

    candidate_rows: list[tuple[int, int, dict[str, Any], tuple[float, float]]] = []
    square_size = BOARD_SIZE / 8.0
    for det, (x, y) in zip(detections, warped_points):
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            continue
        col = min(7, int(x // square_size))
        row = min(7, int(y // square_size))
        candidate_rows.append((row, col, det, (float(x), float(y))))

    if not candidate_rows:
        raise RuntimeError("A bábuk felismerése sikerült, de egyik sem esett a felismert sakktábla területére.")

    # Automatically orient the board so that White's pieces are at the bottom and Black's at the top.
    white_bottom_score = sum(d["conf"] for r, _, d, _ in candidate_rows if d["name"] in WHITE_NAMES and r >= 4)
    white_bottom_score += sum(d["conf"] for r, _, d, _ in candidate_rows if d["name"] in BLACK_NAMES and r < 4)
    white_top_score = sum(d["conf"] for r, _, d, _ in candidate_rows if d["name"] in WHITE_NAMES and r < 4)
    white_top_score += sum(d["conf"] for r, _, d, _ in candidate_rows if d["name"] in BLACK_NAMES and r >= 4)
    flipped_180 = white_top_score > white_bottom_score

    if flipped_180:
        rectified = cv2.rotate(rectified, cv2.ROTATE_180)
        candidate_rows = [
            (7 - row, 7 - col, det, (BOARD_SIZE - 1 - xy[0], BOARD_SIZE - 1 - xy[1]))
            for row, col, det, xy in candidate_rows
        ]

    # Prefer the highest-confidence detection if multiple boxes land on the same square.
    best_per_square: dict[tuple[int, int], tuple[float, str, float, tuple[float, float]]] = {}
    for row, col, det, xy in candidate_rows:
        key = (row, col)
        old = best_per_square.get(key)
        if old is None or det["conf"] > old[0]:
            best_per_square[key] = (det["conf"], det["fen"], det["conf"], xy)

    grid = [["." for _ in range(8)] for _ in range(8)]
    annotated_board = rectified.copy()
    for step in range(9):
        pos = min(BOARD_SIZE - 1, int(round(step * square_size)))
        cv2.line(annotated_board, (pos, 0), (pos, BOARD_SIZE - 1), (255, 240, 0), 1, cv2.LINE_AA)
        cv2.line(annotated_board, (0, pos), (BOARD_SIZE - 1, pos), (255, 240, 0), 1, cv2.LINE_AA)

    for (row, col), (_, piece, confidence, xy) in best_per_square.items():
        grid[row][col] = piece
        x, y = int(round(xy[0])), int(round(xy[1]))
        square = f"{chr(ord('a') + col)}{8 - row}"
        color = (30, 200, 30) if piece.isupper() else (255, 140, 20)
        cv2.circle(annotated_board, (x, y), 7, color, -1, cv2.LINE_AA)
        cv2.putText(annotated_board, f"{piece} {square} {confidence:.2f}", (min(BOARD_SIZE - 130, x + 6), max(18, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)

    fen = make_fen(grid)
    board = chess.Board(fen)
    original_annotated = draw_original(image_bgr, corners, detections)
    metrics = {
        "corner_scores": corner_scores,
        "detections": len(detections),
        "occupied_squares": len(best_per_square),
        "flipped_180": flipped_180,
        "fen_valid": board.is_valid(),
        "grid": grid,
    }
    return original_annotated, annotated_board, fen, metrics


st.title("♟️ Automatikus sakkállás-felismerő")
st.write("Tölts fel egy fotót a fizikai sakktábláról. A program automatikusan megkeresi a tábla sarkait, felismeri a bábukat, majd FEN-kódot készít — nincs kézi sarokkijelölés.")
st.caption("Új, kétmodellű pipeline: automatikus sarokészlelés + bábuészlelés. Az első futáskor a modelleket letölti a models/ mappába.")

uploaded = st.file_uploader("Sakktábla-fotó feltöltése", type=["jpg", "jpeg", "png", "webp", "bmp"])
run_button = st.button("Automatikus felismerés", type="primary", disabled=uploaded is None, use_container_width=False)

if uploaded is not None:
    image_data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("Nem sikerült beolvasni a képet.")
        st.stop()
    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Feltöltött kép", use_container_width=True)

if run_button and uploaded is not None:
    image_data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    status = st.status("Az automatikus felismerés előkészítése…", expanded=True)
    progress = status.empty()
    try:
        download_if_missing(CORNER_MODEL_PATH, CORNER_MODEL_URL, "sarokfelismerő", progress)
        download_if_missing(PIECE_MODEL_PATH, PIECE_MODEL_URL, "bábufelismerő", progress)
        progress.empty()
        status.write("Modellek betöltése…")
        corner_model, piece_model = load_models(str(CORNER_MODEL_PATH), str(PIECE_MODEL_PATH))
        status.write("Sakktábla sarkainak automatikus keresése és bábufelismerés…")
        with st.spinner("Felismerés fut…"):
            original_annotated, board_annotated, fen, metrics = run_recognition(image_bgr, corner_model, piece_model)
        status.update(label="Felismerés kész", state="complete", expanded=False)

        left, right = st.columns(2)
        with left:
            st.subheader("Automatikus észlelések az eredeti képen")
            st.image(cv2.cvtColor(original_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        with right:
            st.subheader("Kiegyenesített tábla és mező-hozzárendelés")
            st.image(cv2.cvtColor(board_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.subheader("Digitális sakkállás")
        board = chess.Board(fen)
        st.code(fen, language="text")
        st.download_button("FEN letöltése", data=fen + "\n", file_name="position.fen", mime="text/plain")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Felismerett bábuk", metrics["detections"])
        col2.metric("Elfoglalt mezők", metrics["occupied_squares"], "/ 64")
        col3.metric("Állás szabályossága", "Érvényes" if metrics["fen_valid"] else "Ellenőrzendő")
        col4.metric("Automatikus 180° forgatás", "Igen" if metrics["flipped_180"] else "Nem")

        st.subheader("Felismerési tábla")
        for row in metrics["grid"]:
            st.code(" ".join(row), language="text")

        if metrics["occupied_squares"] < 20:
            st.warning("Kevés bábu került a táblára. Ez az új pipeline is lehet bizonytalan ezen a fotón; ellenőrizd a két képet fent. Próbálj teljes táblás, élesebb, kevésbé takart képet.")
        if not metrics["fen_valid"]:
            st.warning("A felismert FEN nem alkot teljesen szabályos sakkállást. Ez jellemzően hibás bábuosztályozásra, kihagyott bábura vagy rossz mező-hozzárendelésre utal.")
        st.caption("A FEN oldalon a lépő fél alapértelmezés szerint fehér, mert ezt egyetlen fotóból nem lehet megbízhatóan megállapítani. Sáncolási és en-passant jogok sem következtethetők ki biztosan egy állóképből.")
    except Exception as exc:
        status.update(label="A felismerés nem sikerült", state="error", expanded=True)
        st.error(str(exc))
        st.info("A legjobb eredményhez a teljes 8×8-as táblának, mind a négy sarkának és a bábuk tövének látszania kell. A sarokkijelölésre nincs szükség: ha a modell nem találja a sarkokat, a felület pontos hibát jelez.")

with st.expander("Mi változott? / modellforrás"):
    st.markdown(
        "- A régi, egyetlen bábufelismerő modell helyett két előre betanított modell fut.\n"
        "- Az egyik automatikusan megkeresi a négy táblasarkot, a másik osztályozza a bábukat.\n"
        "- A négy sarok kijelölése már nem szükséges.\n"
        "- A modellek első használatkor ide kerülnek: `models/photochess_corner.pt` és `models/photochess_pieces.pt`.\n"
        "- Modellforrás: [2T1F/photochess GitHub](https://github.com/2T1F/photochess). A projekt leírása szerint a két modell a ChessRender360 szintetikus adatállományon készült. A forrás repóban nem találtam egyértelmű licencfájlt, ezért ez jelenleg prototípus/oktatási célú integráció; nyilvános továbbterjesztés előtt a modellek felhasználási feltételeit tisztázni kell."
    )
