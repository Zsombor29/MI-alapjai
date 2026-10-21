"""Physical chessboard image -> YOLO detections -> digital board + FEN.

Expected project layout:
    weights/best.pt
    samples/test_board.jpg
    outputs/

When asked, click the four physical corner squares in this algebraic order:
    a8, h8, h1, a1
This gives the script the board orientation as well as its perspective.
"""
from pathlib import Path
import sys
import urllib.request

import cv2
import numpy as np
import chess
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_URL = "https://huggingface.co/NAKSTStudio/yolov8m-chess-piece-detection/resolve/main/best.pt"
MODEL_PATH = ROOT / "models" / "best.pt"
IMAGE_PATH = ROOT / "samples" / "test_board.jpg"
OUTPUT_DIR = ROOT / "outputs"


def resolve_model() -> Path:
    """Use an existing best.pt if found; otherwise download it into models/."""
    candidates = [
        ROOT / "weights" / "best.pt",
        ROOT / "models" / "best.pt",
        ROOT / "best.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Search project folders, without scanning the virtual environment or Git data.
    for candidate in ROOT.rglob("*.pt"):
        if ".venv" in candidate.parts or ".git" in candidate.parts:
            continue
        if candidate.is_file():
            print(f"Meglévő modell használata: {candidate}")
            return candidate

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print("A modellfájl nem található. Letöltés Hugging Face-ről (kb. 300 MB)...")
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=60) as response, MODEL_PATH.open("wb") as output:
                total = int(response.headers.get("Content-Length", "0") or 0)
                downloaded = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(f"\rLetöltve: {downloaded / total:.0%}", end="", flush=True)
            print("\nModell letöltve.")
        except Exception as exc:
            if MODEL_PATH.exists():
                MODEL_PATH.unlink()
            raise RuntimeError(f"Nem sikerült letölteni a modellt: {exc}") from exc
    return MODEL_PATH

# Class IDs from NAKSTStudio/yolov8m-chess-piece-detection.
CLASS_TO_FEN = {
    1: "K", 2: "Q", 3: "R", 4: "B", 5: "N", 6: "P",
    7: "k", 8: "q", 9: "r", 10: "b", 11: "n", 12: "p",
}
CLASS_TO_NAME = {
    1: "white_king", 2: "white_queen", 3: "white_rook",
    4: "white_bishop", 5: "white_knight", 6: "white_pawn",
    7: "black_king", 8: "black_queen", 9: "black_rook",
    10: "black_bishop", 11: "black_knight", 12: "black_pawn",
}


def select_board_corners(image: np.ndarray) -> np.ndarray | None:
    """Let user click the a8, h8, h1, a1 physical board corners."""
    h, w = image.shape[:2]
    scale = min(1.0, 1100 / w, 760 / h)
    display_w, display_h = int(w * scale), int(h * scale)
    base = cv2.resize(image, (display_w, display_h)) if scale < 1.0 else image.copy()
    points_display: list[tuple[int, int]] = []
    labels = ["a8", "h8", "h1", "a1"]
    window = "Select chessboard corners"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def mouse_callback(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points_display) < 4:
            points_display.append((x, y))

    cv2.setMouseCallback(window, mouse_callback)
    while True:
        view = base.copy()
        instruction = "Click corners in order: a8 -> h8 -> h1 -> a1 | R=reset | ESC=cancel"
        cv2.putText(view, instruction, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (0, 255, 255), 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(points_display):
            cv2.circle(view, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(view, labels[i], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, view)
        key = cv2.waitKey(20) & 0xFF
        if len(points_display) == 4:
            break
        if key == 27:  # ESC
            cv2.destroyWindow(window)
            return None
        if key in (ord("r"), ord("R")):
            points_display.clear()

    cv2.destroyWindow(window)
    # Convert display positions back to original image coordinates.
    points = np.array([(x / scale, y / scale) for x, y in points_display], dtype=np.float32)
    return points


def make_fen(placement: list[list[str]]) -> str:
    rows = []
    for row in placement:  # row 0 is rank 8; row 7 is rank 1
        encoded = ""
        empty = 0
        for piece in row:
            if piece == ".":
                empty += 1
            else:
                if empty:
                    encoded += str(empty)
                    empty = 0
                encoded += piece
        if empty:
            encoded += str(empty)
        rows.append(encoded)
    return "/".join(rows) + " w - - 0 1"


def main() -> int:
    if not IMAGE_PATH.exists():
        print(f"Nem találom a képet: {IMAGE_PATH}")
        print("Tedd a tesztképet a samples\\test_board.jpg útvonalra, vagy módosítsd az IMAGE_PATH értékét.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        print(f"Nem sikerült beolvasni a képet: {IMAGE_PATH}")
        return 1

    print("Modell betöltése...")
    try:
        model_path = resolve_model()
        model = YOLO(str(model_path))
    except Exception as exc:
        print(f"Nem sikerült betölteni a modellt: {exc}")
        return 1
    print("Bábuk felismerése...")
    # Ultralytics rescales detections back to the original image coordinates.
    result = model.predict(source=image, imgsz=640, conf=0.15, iou=0.45, verbose=False)[0]

    corners = select_board_corners(image)
    if corners is None:
        print("Sarokkijelölés megszakítva.")
        return 1

    # Map physical board corners (a8, h8, h1, a1) to chess-grid coordinates.
    destination = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(corners, destination)

    square_best: dict[tuple[int, int], tuple[float, str, tuple[int, int, int, int], float]] = {}
    detections = result.boxes
    if detections is not None:
        xyxy = detections.xyxy.cpu().numpy()
        classes = detections.cls.cpu().numpy().astype(int)
        confidences = detections.conf.cpu().numpy()
        for box, cls_id, confidence in zip(xyxy, classes, confidences):
            if cls_id not in CLASS_TO_FEN:
                continue  # Skip board class and unknown classes.
            x1, y1, x2, y2 = map(float, box)
            center = np.array([[[((x1 + x2) / 2), ((y1 + y2) / 2)]]], dtype=np.float32)
            grid = cv2.perspectiveTransform(center, homography)[0, 0]
            col, row = int(np.floor(grid[0])), int(np.floor(grid[1]))
            if not (0 <= col < 8 and 0 <= row < 8):
                continue  # Detection center falls outside the selected board.
            square = (row, col)
            piece = CLASS_TO_FEN[cls_id]
            previous = square_best.get(square)
            # If multiple detections land on one square, keep the most confident.
            if previous is None or confidence > previous[0]:
                square_best[square] = (float(confidence), piece,
                                       (int(x1), int(y1), int(x2), int(y2)), float(confidence))

    board_grid = [["." for _ in range(8)] for _ in range(8)]
    annotated = image.copy()
    for i, (x, y) in enumerate(corners):
        cv2.circle(annotated, (int(x), int(y)), 9, (0, 255, 255), -1)
        cv2.putText(annotated, ["a8", "h8", "h1", "a1"][i],
                    (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)

    for (row, col), (_, piece, box, confidence) in square_best.items():
        board_grid[row][col] = piece
        x1, y1, x2, y2 = box
        square_name = f"{chr(ord('a') + col)}{8 - row}"
        color = (0, 200, 0) if piece.isupper() else (255, 120, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{piece} {square_name} {confidence:.2f}",
                    (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)

    fen = make_fen(board_grid)
    board = chess.Board(fen)
    image_out = OUTPUT_DIR / "chess_position.jpg"
    fen_out = OUTPUT_DIR / "position.fen"
    cv2.imwrite(str(image_out), annotated)
    fen_out.write_text(fen + "\n", encoding="utf-8")

    print("\nFelismert állás (soronként a 8. sortól az 1. sorig):")
    for row in board_grid:
        print(" ".join(row))
    print(f"\nFEN: {fen}")
    print(f"Felismert mezők száma: {len(square_best)} / 64")
    print(f"Sakkállás szabályossága: {'érvényes' if board.is_valid() else 'nem teljesen érvényes'}")
    if not board.is_valid():
        print("Ez az első prototípus. Téves vagy hiányzó bábufelismerés esetén a FEN javításra szorulhat.")
    print(f"Annotált kép: {image_out}")
    print(f"FEN fájl: {fen_out}")

    # Show the result; press any key to close.
    cv2.imshow("Detected chess position - press any key to close", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
