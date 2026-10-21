
from pathlib import Path
from ultralytics import YOLO

MODEL = (
    "https://huggingface.co/NAKSTStudio/"
    "yolov8m-chess-piece-detection/resolve/main/best.pt"
)

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "samples" / "test_board.jpg"
OUTPUT_DIR = BASE_DIR / "outputs"


def main():
    if not IMAGE_PATH.exists():
        print(f"HIBA: Nem találom a képet: {IMAGE_PATH}")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("YOLO-modell betöltése...")
    model = YOLO(MODEL)

    print("Kép feldolgozása...")
    results = model.predict(
        source=str(IMAGE_PATH),
        conf=0.25,
        save=False,
        verbose=False
    )

    result = results[0]
    annotated_image = result.plot()

    output_path = OUTPUT_DIR / "detected_board.jpg"

    import cv2
    cv2.imwrite(str(output_path), annotated_image)

    print("\nFelismerések:")
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        name = model.names[class_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        print(
            f"{name}: {confidence:.2%} "
            f"| koordináták: ({x1}, {y1}, {x2}, {y2})"
        )

    print(f"\nÖsszes felismert objektum: {len(result.boxes)}")
    print(f"Eredmény mentve ide: {output_path}")


if __name__ == "__main__":
    main()
