Sakkfelismerő V5 – ChessSight visszaállítva elsődleges bábufelismerőnek

A V4 per-square ONNX piece classifier már NEM vesz részt a bábutípus végső kiválasztásában.
Az elsődleges bábufelismerő ismét a korábban jobb eredményt adó ChessSight RT-DETR v0.2.0.

A tábla sarkait a ChessSight saját Swin corner-heatmap v1 modellje keresi automatikusan.
Tartalék: OpenCV sakkrácsdetektálás, ChessSight corner-box jelöltek, régi photochess pose modell.

A bábukat három nézetben osztályozza:
1. teljes feltöltött fénykép (a boardon kívüli bábukat a perspektívatranszformáció kiszűri),
2. kiegyenesített sakktábla,
3. enyhén élesített kiegyenesített sakktábla.
A mezőnkénti döntésben a több nézetből egyező címke élvez elsőbbséget.

Telepítés PowerShellből a projekt gyökérmappájában:
.\.venv\Scripts\python.exe -m pip install -r requirements_chess_app.txt

Indítás:
.\.venv\Scripts\python.exe -m streamlit run app.py

Első futás:
- ChessSight RT-DETR súlyok: Hugging Face cache-be töltődnek (ha még nincsenek ott).
- ChessSight Swin corner heatmap: kb. 111 MB, models/cache mappába / Hugging Face cache-be kerül.
Internetkapcsolat szükséges az első letöltéshez.

Nem kell módosítani a detector.py vagy chess_board.py fájlokat.
A .venv, models, *.pt, *.safetensors maradjon a .gitignore-ban.

Licencek: ChessSight RT-DETR és corner heatmap CC BY-NC 4.0; nem kereskedelmi oktatási prototípushoz.
