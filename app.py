"""Local Streamlit UI for the physical chessboard recognition prototype.

Run from the project root:
    python -m streamlit run app.py

The board-corner picker opens a small OpenCV window on the same computer.
Click the playable board's four corners in order: a8, h8, h1, a1.
"""
from pathlib import Path
import sys

import cv2
import numpy as np
import streamlit as st
import chess
from ultralytics import YOLO

# Reuse the existing project logic and model lookup/download behavior.
try:
    import chess_board as logic
except Exception as exc:
    st.error(f"Nem sikerült betölteni a chess_board.py modult: {exc}")
    st.stop()

st.set_page_config(page_title="Sakkállás-felismerő", page_icon="♟️", layout="wide")
st.title("♟️ Fizikai sakktábla felismerése")
st.caption("YOLO bábufelismerés → digitális sakkállás → FEN-kimenet")
st.info(
    "Tölts fel egy képet a tábláról. A felismerés után egy külön ablakban jelöld ki "
    "a 8×8-as játékmező négy sarkát ebben a sorrendben: a8 → h8 → h1 → a1. "
    "A külső fa keret helyett a sakktábla mezőinek külső határát válaszd."
)

@st.cache_resource(show_spinner="YOLO-modell betöltése…")
def load_model():
    path = logic.resolve_model()
    return YOLO(str(path)), str(path)


def to_square_data(image: np.ndarray, model: YOLO, corners: np.ndarray, conf: float):
    prediction = model.predict(source=image, imgsz=640, conf=conf, iou=0.45, verbose=False)[0]
    destination = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)

    # Each square keeps only its highest-confidence piece detection.
    square_best = {}
    boxes = prediction.boxes
    if boxes is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        for box, cls_id, confidence in zip(xyxy, classes, confidences):
            if cls_id not in logic.CLASS_TO_FEN:
                continue
            x1, y1, x2, y2 = map(float, box)
            # The base of a standing piece is usually a better square locator than its center.
            foot = np.array([[[((x1 + x2) / 2), y2 - max(2.0, (y2 - y1) * 0.03)]]], dtype=np.float32)
            grid = cv2.perspectiveTransform(foot, homography)[0, 0]
            col, row = int(np.floor(grid[0])), int(np.floor(grid[1]))
            if not (0 <= col < 8 and 0 <= row < 8):
                continue
            square = (row, col)
            old = square_best.get(square)
            if old is None or float(confidence) > old[0]:
                square_best[square] = (float(confidence), logic.CLASS_TO_FEN[cls_id],
                                       tuple(map(int, (x1, y1, x2, y2))))

    grid = [["." for _ in range(8)] for _ in range(8)]
    annotated = image.copy()

    # Draw the perspective-corrected 8x8 grid back onto the original image.
    inv_h = np.linalg.inv(homography)
    for step in range(9):
        p1 = cv2.perspectiveTransform(np.array([[[step, 0]]], np.float32), inv_h)[0, 0]
        p2 = cv2.perspectiveTransform(np.array([[[step, 8]]], np.float32), inv_h)[0, 0]
        p3 = cv2.perspectiveTransform(np.array([[[0, step]]], np.float32), inv_h)[0, 0]
        p4 = cv2.perspectiveTransform(np.array([[[8, step]]], np.float32), inv_h)[0, 0]
        cv2.line(annotated, tuple(np.rint(p1).astype(int)), tuple(np.rint(p2).astype(int)), (255, 220, 0), 1, cv2.LINE_AA)
        cv2.line(annotated, tuple(np.rint(p3).astype(int)), tuple(np.rint(p4).astype(int)), (255, 220, 0), 1, cv2.LINE_AA)

    corner_labels = ["a8", "h8", "h1", "a1"]
    for i, (x, y) in enumerate(corners):
        cv2.circle(annotated, (int(x), int(y)), 7, (0, 255, 255), -1)
        cv2.putText(annotated, corner_labels[i], (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    for (row, col), (score, piece, box) in square_best.items():
        grid[row][col] = piece
        x1, y1, x2, y2 = box
        sq = f"{chr(ord('a') + col)}{8-row}"
        color = (0, 210, 0) if piece.isupper() else (255, 150, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{piece} {sq} {score:.2f}", (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

    fen = logic.make_fen(grid)
    board = chess.Board(fen)
    return annotated, grid, fen, board, len(square_best), len(boxes) if boxes is not None else 0

uploaded = st.file_uploader("Sakktábla képe", type=["jpg", "jpeg", "png", "webp"])
conf = st.slider("YOLO konfidencia-küszöb", min_value=0.05, max_value=0.60, value=0.15, step=0.05,
                 help="Alacsonyabb értéknél több, de több téves felismerés is várható.")

if uploaded is not None:
    raw = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        st.error("A feltöltött képet nem sikerült beolvasni.")
        st.stop()
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Feltöltött kép", use_container_width=True)

    if st.button("Bábuk felismerése és sakkállás létrehozása", type="primary"):
        with st.spinner("Modell betöltése és felismerés…"):
            try:
                model, model_path = load_model()
            except Exception as exc:
                st.error(f"A modellt nem sikerült betölteni: {exc}")
                st.stop()
            st.caption(f"Használt modell: `{model_path}`")
            try:
                # The helper opens an OpenCV window on this computer for four corner clicks.
                corners = logic.select_board_corners(image)
                if corners is None:
                    st.warning("A sarokkijelölést megszakítottad.")
                    st.stop()
                annotated, grid, fen, board, squares, raw_detections = to_square_data(image, model, corners, conf)
            except Exception as exc:
                st.error(f"A felismerés nem sikerült: {exc}")
                st.stop()

        st.session_state["chess_annotated"] = annotated
        st.session_state["chess_grid"] = grid
        st.session_state["chess_fen"] = fen
        st.session_state["chess_valid"] = board.is_valid()
        st.session_state["chess_squares"] = squares
        st.session_state["chess_detections"] = raw_detections

if "chess_fen" in st.session_state:
    st.subheader("Felismerés eredménye")
    left, right = st.columns([3, 2])
    with left:
        st.image(cv2.cvtColor(st.session_state["chess_annotated"], cv2.COLOR_BGR2RGB),
                 caption="Rács, bábuk és mező-hozzárendelések", use_container_width=True)
        ok, encoded = cv2.imencode(".jpg", st.session_state["chess_annotated"])
        if ok:
            st.download_button("Annotált kép letöltése", encoded.tobytes(), "chess_position.jpg", "image/jpeg")
    with right:
        st.metric("Mezőre rendelt bábuk", f"{st.session_state['chess_squares']} / 64")
        st.metric("Nyers YOLO-találatok", st.session_state["chess_detections"])
        if st.session_state["chess_valid"]:
            st.success("A FEN egy szabályossági ellenőrzésen átment.")
        else:
            st.warning("A felismert állás valószínűleg javításra szorul (pl. hiányzó vagy téves bábuk).")
        st.markdown("**FEN**")
        st.code(st.session_state["chess_fen"], language="text")
        st.download_button("FEN letöltése", st.session_state["chess_fen"] + "\n", "position.fen", "text/plain")

    st.markdown("**Digitális tábla** — `.` jelöli az üres mezőt; a sorok a 8. sortól az 1. sorig haladnak.")
    st.code("\n".join(" ".join(row) for row in st.session_state["chess_grid"]), language="text")
    st.caption("Fehér bábuk: nagybetűk (K, Q, R, B, N, P); fekete bábuk: kisbetűk (k, q, r, b, n, p).")

st.divider()
st.caption("Helyi prototípus · Az automatikus felismerés hibázhat, ezért a FEN-t érdemes ellenőrizni.")
