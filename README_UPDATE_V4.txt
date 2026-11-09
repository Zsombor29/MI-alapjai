MI alapjai – sakkfelismerő v4

Mit váltottunk le?
- A ChessSight modellt már nem használjuk elsődleges bábuosztályozónak.
- A táblát először minden mezőben üres/foglalt osztályozó olvassa, majd a foglalt mezőkön egy külön 12 osztályos bábuosztályozó fut.
- A bábukropozás a chesscog fotó-feldolgozási módszerét követi: a bábu teteje a szomszédos mezők irányába is bekerül a képbe. Ez különösen a királyok és vezérek felismerését segítheti.
- A négy sarok továbbra is automatikusan keresett; nincs kézi kijelölés.

Telepítés:
1. Másold az app.py és requirements_chess_app.txt fájlokat a C:\Temp\MI alapjai mappába, az app.py-t felülírva.
2. A VS Code terminálban:
   .\.venv\Scripts\python.exe -m pip install -r requirements_chess_app.txt
3. Indítás:
   .\.venv\Scripts\python.exe -m streamlit run app.py

Az ONNX modellek első futáskor letöltődnek a models mappába (internet kell). Add a .gitignore-hoz a models/ és *.onnx szabályt, hogy ne töltsd fel a modellfájlokat GitHubra.

Korlátok:
- Ez továbbra is előre betanított modell, saját tanítás nélkül.
- A modell közzétett teszteredményei nem garantálják ugyanazt az eredményt minden sakk-készleten.
- A sarokkeresés fallbackként továbbra is ChessSight súlyokat használ, amelyek CC BY-NC 4.0 licencűek. Oktatási, nem kereskedelmi prototípushoz használd.
