"""Browser-based Streamlit UI for physical chessboard recognition.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -m streamlit run app.py

The user selects the board corners directly in the browser; no OpenCV window is opened.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import chess
from ultralytics import YOLO

st.set_page_config(page_title="Sakkállás-felismerő", page_icon="♟️", layout="wide")

# FEN character -> human-readable option shown in the manual position editor.
PIECE_CHOICES = [
    (".", "· Üres"),
    ("K", "K – fehér király"),
    ("Q", "Q – fehér vezér"),
    ("R", "R – fehér bástya"),
    ("B", "B – fehér futó"),
    ("N", "N – fehér huszár"),
    ("P", "P – fehér gyalog"),
    ("k", "k – fekete király"),
    ("q", "q – fekete vezér"),
    ("r", "r – fekete bástya"),
    ("b", "b – fekete futó"),
    ("n", "n – fekete huszár"),
    ("p", "p – fekete gyalog"),
]
FEN_TO_LABEL = {fen_char: label for fen_char, label in PIECE_CHOICES}
LABEL_TO_FEN = {label: fen_char for fen_char, label in PIECE_CHOICES}
PIECE_LABELS = [label for _, label in PIECE_CHOICES]

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    st.title("♟️ Fizikai sakktábla felismerése")
    st.error("Hiányzik a böngészős sarokkijelölő csomag.")
    st.code(r".\.venv\Scripts\python.exe -m pip install streamlit-image-coordinates")
    st.stop()

try:
    import chess_board as logic
except Exception as exc:
    st.title("♟️ Fizikai sakktábla felismerése")
    st.error(f"Nem sikerült betölteni a chess_board.py modult: {exc}")
    st.stop()

st.title("♟️ Fizikai sakktábla felismerése")
st.caption("Fénykép → YOLO bábufelismerés → digitális sakkállás → FEN")
st.info(
    "Tölts fel egy képet, majd kattints a játszható sakktábla négy sarkára "
    "közvetlenül ezen az oldalon. Nem nyílik meg külön ablak."
)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> YOLO:
    """Load the YOLO model once per model path and Streamlit session server."""
    return YOLO(model_path)


def clear_result_state() -> None:
    for key in (
        "chess_annotated",
        "chess_grid",
        "chess_detected_grid",
        "chess_fen",
        "chess_valid",
        "chess_squares",
        "chess_detections",
        "chess_model_path",
    ):
        st.session_state.pop(key, None)
    for row in range(8):
        for col in range(8):
            st.session_state.pop(f"piece_{row}_{col}", None)


def draw_corner_markers(
    image_rgb: np.ndarray,
    points_original: list[tuple[float, float]],
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Draw already selected corners on the scaled RGB preview."""
    view = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    labels = ["a8", "h8", "h1", "a1"]
    display_points: list[tuple[int, int]] = []
    for x, y in points_original:
        display_points.append((int(round(x * scale_x)), int(round(y * scale_y))))

    for i in range(1, len(display_points)):
        cv2.line(view, display_points[i - 1], display_points[i], (0, 220, 255), 2, cv2.LINE_AA)
    if len(display_points) == 4:
        cv2.line(view, display_points[-1], display_points[0], (0, 220, 255), 2, cv2.LINE_AA)

    for i, (x, y) in enumerate(display_points):
        cv2.circle(view, (x, y), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            view,
            f"{i + 1}. {labels[i]}",
            (min(max(5, x + 9), max(5, view.shape[1] - 100)), max(24, y - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(view, cv2.COLOR_BGR2RGB)


def detect_position(
    image_bgr: np.ndarray,
    model: YOLO,
    corners: np.ndarray,
    confidence_threshold: float,
):
    """Detect pieces and assign each detection to one of the 64 board squares."""
    prediction = model.predict(
        source=image_bgr,
        imgsz=640,
        conf=confidence_threshold,
        iou=0.45,
        verbose=False,
    )[0]

    # Selected points are ordered a8, h8, h1, a1; map them to an 8x8 grid.
    destination = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)

    square_best: dict[tuple[int, int], tuple[float, str, tuple[int, int, int, int]]] = {}
    boxes = prediction.boxes
    raw_detections = 0 if boxes is None else len(boxes)

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        for box, cls_id, score in zip(xyxy, classes, confidences):
            # Class zero is the board itself; only chess pieces should fill squares.
            if cls_id not in logic.CLASS_TO_FEN:
                continue
            x1, y1, x2, y2 = map(float, box)
            # The lower midpoint of a piece box generally projects more accurately
            # to its occupied square than the geometric center of the whole piece.
            foot_x = (x1 + x2) / 2
            foot_y = y2 - max(2.0, (y2 - y1) * 0.03)
            foot = np.array([[[foot_x, foot_y]]], dtype=np.float32)
            board_xy = cv2.perspectiveTransform(foot, homography)[0, 0]
            col, row = int(np.floor(board_xy[0])), int(np.floor(board_xy[1]))
            if not (0 <= col < 8 and 0 <= row < 8):
                continue
            square = (row, col)
            current = square_best.get(square)
            if current is None or float(score) > current[0]:
                square_best[square] = (
                    float(score),
                    logic.CLASS_TO_FEN[cls_id],
                    tuple(map(int, (x1, y1, x2, y2))),
                )

    grid = [["." for _ in range(8)] for _ in range(8)]
    annotated = image_bgr.copy()
    inverse_h = np.linalg.inv(homography)

    # Draw the perspective-corrected board grid onto the input image.
    for step in range(9):
        p1 = cv2.perspectiveTransform(np.array([[[step, 0]]], np.float32), inverse_h)[0, 0]
        p2 = cv2.perspectiveTransform(np.array([[[step, 8]]], np.float32), inverse_h)[0, 0]
        p3 = cv2.perspectiveTransform(np.array([[[0, step]]], np.float32), inverse_h)[0, 0]
        p4 = cv2.perspectiveTransform(np.array([[[8, step]]], np.float32), inverse_h)[0, 0]
        cv2.line(annotated, tuple(np.rint(p1).astype(int)), tuple(np.rint(p2).astype(int)), (255, 220, 0), 1, cv2.LINE_AA)
        cv2.line(annotated, tuple(np.rint(p3).astype(int)), tuple(np.rint(p4).astype(int)), (255, 220, 0), 1, cv2.LINE_AA)

    labels = ["a8", "h8", "h1", "a1"]
    for i, (x, y) in enumerate(corners):
        cv2.circle(annotated, (int(x), int(y)), 8, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(annotated, labels[i], (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    for (row, col), (score, piece, box) in square_best.items():
        grid[row][col] = piece
        x1, y1, x2, y2 = box
        square_name = f"{chr(ord('a') + col)}{8 - row}"
        color = (0, 210, 0) if piece.isupper() else (255, 150, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{piece} {square_name} {score:.2f}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

    fen = logic.make_fen(grid)
    board = chess.Board(fen)
    return annotated, grid, fen, board, len(square_best), raw_detections


uploaded = st.file_uploader("Sakktábla képe", type=["jpg", "jpeg", "png", "webp"])
conf = st.slider(
    "YOLO konfidencia-küszöb",
    min_value=0.05,
    max_value=0.60,
    value=0.15,
    step=0.05,
    help="Alacsonyabb értéknél több találat lehet, de a téves felismerések száma is nőhet.",
)

if uploaded is None:
    st.session_state.pop("active_image_hash", None)
    st.session_state.pop("board_corners", None)
    st.caption("Kezdéshez tölts fel egy fizikai sakktábláról készült képet.")
else:
    image_bytes = uploaded.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("A feltöltött képet nem sikerült beolvasni.")
        st.stop()

    if st.session_state.get("active_image_hash") != image_hash:
        st.session_state["active_image_hash"] = image_hash
        st.session_state["board_corners"] = []
        st.session_state["last_corner_event"] = None
        st.session_state["corner_picker_version"] = int(st.session_state.get("corner_picker_version", 0)) + 1
        clear_result_state()

    corners_list: list[tuple[float, float]] = st.session_state.setdefault("board_corners", [])
    generation = int(st.session_state.get("corner_picker_version", 1))

    left, right = st.columns([1, 3])
    with left:
        if st.button("Sarkok újrakezdése", disabled=(len(corners_list) == 0), use_container_width=True):
            st.session_state["board_corners"] = []
            st.session_state["last_corner_event"] = None
            st.session_state["corner_picker_version"] = generation + 1
            clear_result_state()
            st.rerun()
        st.markdown("**Kijelölési sorrend**")
        st.write("1. `a8` — bal felső")
        st.write("2. `h8` — jobb felső")
        st.write("3. `h1` — jobb alsó")
        st.write("4. `a1` — bal alsó")
        st.caption("A játszható mezők sarkát jelöld, ne a fa keret külső sarkát.")
        st.metric("Kijelölt sarkok", f"{len(corners_list)} / 4")

    original_h, original_w = image_bgr.shape[:2]
    max_display_w, max_display_h = 900, 650
    scale = min(1.0, max_display_w / original_w, max_display_h / original_h)
    display_w = max(1, int(round(original_w * scale)))
    display_h = max(1, int(round(original_h * scale)))
    scale_x = display_w / original_w
    scale_y = display_h / original_h
    preview_bgr = cv2.resize(image_bgr, (display_w, display_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
    marked_preview = draw_corner_markers(preview_rgb, corners_list, scale_x, scale_y)

    with right:
        st.markdown("### Jelöld ki a sakktábla sarkait a képen")
        if len(corners_list) < 4:
            st.caption(f"Kattintás következik: {['a8', 'h8', 'h1', 'a1'][len(corners_list)]}")
            click = streamlit_image_coordinates(
                marked_preview,
                width=display_w,
                key=f"corner_picker_{image_hash}_{generation}_{len(corners_list)}",
                cursor="crosshair",
            )
            if isinstance(click, dict) and click.get("x") is not None and click.get("y") is not None:
                token = click.get("event_time", (click.get("x"), click.get("y")))
                if token != st.session_state.get("last_corner_event"):
                    try:
                        click_x = float(click["x"])
                        click_y = float(click["y"])
                        if 0 <= click_x < display_w and 0 <= click_y < display_h:
                            updated = list(st.session_state["board_corners"])
                            updated.append((click_x / scale_x, click_y / scale_y))
                            st.session_state["board_corners"] = updated
                            st.session_state["last_corner_event"] = token
                            st.rerun()
                    except (KeyError, TypeError, ValueError):
                        st.warning("Nem sikerült feldolgozni a kattintás koordinátáit. Kattints újra a képre.")
        else:
            st.image(marked_preview, caption="A négy kijelölt sarok", use_container_width=False)
            st.success("Mind a négy sarok kijelölve. Indíthatod a felismerést.")

    if len(st.session_state["board_corners"]) == 4:
        if st.button("Bábuk felismerése és sakkállás létrehozása", type="primary", use_container_width=True):
            model_path: Path | None = None
            try:
                with st.spinner("YOLO-modell betöltése… első alkalommal ez eltarthat néhány másodpercig."):
                    model_path = logic.resolve_model()
                    model = load_model(str(model_path))
                with st.spinner("Bábuk felismerése és mezőkhöz rendelése…"):
                    corners_np = np.array(st.session_state["board_corners"], dtype=np.float32)
                    annotated, grid, fen, board, squares, raw_detections = detect_position(
                        image_bgr, model, corners_np, conf
                    )
                # Keep the original automatic result so the user can reset edits.
                st.session_state["chess_annotated"] = annotated
                st.session_state["chess_detected_grid"] = [row.copy() for row in grid]
                st.session_state["chess_grid"] = [row.copy() for row in grid]
                st.session_state["chess_fen"] = fen
                st.session_state["chess_valid"] = board.is_valid()
                st.session_state["chess_squares"] = squares
                st.session_state["chess_detections"] = raw_detections
                st.session_state["chess_model_path"] = str(model_path)
                for r in range(8):
                    for c in range(8):
                        st.session_state[f"piece_{r}_{c}"] = FEN_TO_LABEL[grid[r][c]]
            except Exception as exc:
                st.error(f"A felismerés nem sikerült: {type(exc).__name__}: {exc}")

    if "chess_fen" in st.session_state:
        st.divider()
        st.subheader("Felismerés eredménye")
        result_left, result_right = st.columns([3, 2])
        with result_left:
            annotated_rgb = cv2.cvtColor(st.session_state["chess_annotated"], cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Sakkrács, felismerések és mező-hozzárendelések", use_container_width=True)
            ok, encoded = cv2.imencode(".jpg", st.session_state["chess_annotated"])
            if ok:
                st.download_button("Annotált kép letöltése", encoded.tobytes(), "chess_position.jpg", "image/jpeg")
        with result_right:
            st.caption(f"Használt modell: `{st.session_state['chess_model_path']}`")
            st.metric("Mezőre rendelt bábuk", f"{st.session_state['chess_squares']} / 64")
            st.metric("YOLO-találatok összesen", st.session_state["chess_detections"])
            if st.session_state["chess_valid"]:
                st.success("A sakkállás átment a python-chess alapvető szabályossági ellenőrzésén.")
            else:
                st.warning("A felismert állás valószínűleg javításra szorul. Ellenőrizd a bábukat és a FEN-t.")
            st.markdown("**FEN**")
            st.code(st.session_state["chess_fen"], language="text")
            st.download_button("FEN letöltése", st.session_state["chess_fen"] + "\n", "position.fen", "text/plain")

        st.markdown("**Digitális tábla** — a sorok a 8. sortól az 1. sorig haladnak; `.` jelöli az üres mezőt.")
        st.code("\n".join(" ".join(row) for row in st.session_state["chess_grid"]), language="text")
        st.caption("Fehér bábuk: nagybetűk (K, Q, R, B, N, P); fekete bábuk: kisbetűk (k, q, r, b, n, p).")

        with st.expander("🛠️ Állás kézi javítása", expanded=True):
            st.write(
                "A felismerés tévedhet. Az egyes mezőknél kiválaszthatod a helyes bábut "
                "vagy az üres mezőt, majd az alsó gombbal frissítheted a digitális állást és a FEN-kódot."
            )
            if st.button("Automatikusan felismert állás visszaállítása", key="reset_piece_edits"):
                original_grid = st.session_state["chess_detected_grid"]
                st.session_state["chess_grid"] = [row.copy() for row in original_grid]
                reset_fen = logic.make_fen(original_grid)
                reset_board = chess.Board(reset_fen)
                st.session_state["chess_fen"] = reset_fen
                st.session_state["chess_valid"] = reset_board.is_valid()
                st.session_state["chess_squares"] = sum(ch != "." for row in original_grid for ch in row)
                for r in range(8):
                    for c in range(8):
                        st.session_state[f"piece_{r}_{c}"] = FEN_TO_LABEL[original_grid[r][c]]
                st.rerun()

            st.caption("Az oszlopok a–h, a sorok 8–1 sorrendben jelennek meg.")
            for r in range(8):
                st.markdown(f"**{8-r}. sor**")
                cols = st.columns(8)
                for c in range(8):
                    square = f"{chr(ord('a') + c)}{8-r}"
                    widget_key = f"piece_{r}_{c}"
                    # Streamlit requires widget state to be initialized before creating it.
                    if widget_key not in st.session_state:
                        st.session_state[widget_key] = FEN_TO_LABEL[st.session_state["chess_grid"][r][c]]
                    with cols[c]:
                        st.selectbox(
                            square,
                            options=PIECE_LABELS,
                            key=widget_key,
                            label_visibility="visible",
                        )
            if st.button("Kézi módosítások alkalmazása", type="primary", key="apply_piece_edits"):
                edited_grid = [
                    [LABEL_TO_FEN[st.session_state[f"piece_{r}_{c}"]] for c in range(8)]
                    for r in range(8)
                ]
                edited_fen = logic.make_fen(edited_grid)
                edited_board = chess.Board(edited_fen)
                st.session_state["chess_grid"] = edited_grid
                st.session_state["chess_fen"] = edited_fen
                st.session_state["chess_valid"] = edited_board.is_valid()
                st.session_state["chess_squares"] = sum(ch != "." for row in edited_grid for ch in row)
                st.success("A kézi módosítások alkalmazva. A FEN-kód frissült.")
                st.rerun()

st.divider()
st.caption("Helyi prototípus · A felismerőmodell hibázhat, ezért a létrehozott sakkállást/FEN-t ellenőrizd.")
