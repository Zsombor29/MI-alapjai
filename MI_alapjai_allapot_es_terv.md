# Fizikai sakktábla-felismerő MI-projekt – állapot és fejlesztési terv

**Utolsó frissítés:** 2026-10-21  
**Cél:** egy rövid idő alatt bemutatható, előre betanított MI-modellt használó alkalmazás, amely egy fizikai sakktábla fényképéből digitális állást és FEN-kódot állít elő, később pedig Stockfish segítségével elemezheti az állást.

---

## 1. Rövid összefoglaló

A projekt egy Python-alapú prototípus. A bábu- és táblafelismeréshez előre betanított YOLO-modellt használunk; saját modellt egyelőre nem tanítunk. A felhasználó feltölt egy képet a Streamlit webes felületén, kijelöli a játszható sakktábla négy sarkát, a program pedig a felismerési dobozokat a tábla mezőihez rendeli, majd digitális állást és FEN-kódot készít.

**A projekt jelenlegi állapota: működő alapok, de a teljes folyamatot még ellenőrizni és tesztelni kell.** A detektor már futott, de a felismert bábuk pontossága nem volt megfelelő. A webes felület és a kézi állásjavítás kódja elkészült, de az aktuális helyi példányban végig kell tesztelni a felismerést, a sarokkijelölést és a mentéseket.

## 2. Mi készült el eddig?

### Fejlesztői környezet és Git

- Windows + PowerShell + VS Code környezetben dolgozunk.
- A projekt mappája: `C:\Temp\MI alapjai`.
- Készült `.gitignore`, amely kizárja legalább a `.venv/` mappát.
- A `git check-ignore -v .venv/Lib/site-packages/torch/lib/torch_cpu.dll` parancs igazolta, hogy a PyTorch nagy DLL-je ki van zárva.
- A GitHub-push során korábban a virtuális környezet nagy fájljai miatt hiba történt. A helyi Git-előzményeket újraépítettük, és a felhasználó jelezte, hogy a Git-probléma megoldódott.
- A GitHub-repozitórium címe: `https://github.com/Zsombor29/MI-alapjai.git`.

### YOLO-modell

- Modell: `NAKSTStudio/yolov8m-chess-piece-detection`, Ultralytics YOLO.
- Helyi modellútvonal, amely az alkalmazás képernyőképén megjelent: `C:\Temp\MI alapjai\models\best.pt`.
- A modell osztályai között van a tábla és a 12 bábutípus (fehér/fekete király, vezér, bástya, futó, huszár, gyalog).
- A modellt korábban sikerült betölteni és képfelismerést futtatni.
- Korábbi tesztben a kisebb konfidenciaküszöb sok téves/átfedő találatot adott, magasabb küszöbnél viszont bábuk maradtak ki.
- Egy korábbi javaslat szerint 640×640-es, képarányt megőrző előfeldolgozást is teszteltünk/javasoltunk, de a pontosság problémája ettől még nincs véglegesen megoldva.

### `chess_board.py` – feldolgozási logika / parancssoros prototípus

A fájl elérhető külön letöltésként, és a projekt gyökerébe kell kerülnie. Fő feladatai:

- modell megkeresése; ha nem található, letöltés a `models/best.pt` helyre;
- YOLO-bábufelismerés;
- a sakktábla négy sarkának kijelölése külön OpenCV-ablakban;
- perspektívatranszformáció és bábu-hozzárendelés a 64 mezőhöz;
- digitális állás és FEN előállítása;
- annotált kép mentése `outputs/chess_position.jpg` néven;
- FEN mentése `outputs/position.fen` néven.

**Fontos:** a Streamlit felület közvetlenül importálja ezt a modult (különösen a `CLASS_TO_FEN`, `resolve_model()` és `make_fen()` elemeket). Ezért a `chess_board.py` fájlt ne töröljük. A benne lévő `main()` külön futtatva továbbra is OpenCV-ablakot nyit a sarkok kijelöléséhez; az elsődleges felhasználói felület viszont az `app.py`.

### `app.py` – böngészős grafikus felület

Az aktuális, legutóbb elkészített `app.py` kódjában szerepel:

- Streamlit-felület és képfeltöltés (`jpg`, `jpeg`, `png`, `webp`);
- a négy tábla-sarok kijelölése közvetlenül a böngészőben a `streamlit-image-coordinates` csomaggal;
- a kijelölések sorrendje: `a8 → h8 → h1 → a1`;
- a YOLO-modell betöltése és a felismerések mezőkhöz rendelése perspektívatranszformációval;
- konfidenciaküszöb állítása a felületen;
- annotált kép és digitális, 8×8-as állás megjelenítése;
- FEN-kód generálása és letöltése;
- kézi állásjavító panel: mezőnként kiválasztható üres mező vagy bábu;
- az automatikusan felismert állás visszaállítása;
- alapvető állásérvényesség-ellenőrzés a `python-chess` könyvtárral.

A fájl letölthető volt a beszélgetésben `app.py` néven. Biztosítani kell, hogy a legutóbbi letöltött verzió kerüljön a helyi projektbe, mert korábbi változatok még külön OpenCV-ablakos sarokkijelölést használtak.

## 3. Javasolt projektmappaszerkezet

```text
MI alapjai/
├── .gitignore
├── requirements.txt
├── app.py
├── chess_board.py
├── models/
│   └── best.pt             # helyi modell; GitHubra ne kerüljön fel
├── samples/
│   └── test_board.jpg      # csak a parancssoros teszthez szükséges
├── outputs/                # generált eredmények; GitHubra ne kerüljenek fel
└── .venv/                  # Python virtuális környezet; GitHubra ne kerüljön fel
```

A böngészős `app.py` fájlba közvetlenül is fel lehet tölteni képet, ezért a `samples/test_board.jpg` nem kötelező a webes felület futtatásához. A külön parancssoros `chess_board.py` viszont ezt az útvonalat használja.

## 4. Környezet beállítása és indítás

Minden csomagtelepítésnél a virtuális környezet Pythonját hívjuk meg közvetlenül, így elkerüljük, hogy a csomagok a felhasználói Python-telepítésbe kerüljenek.

A projekt gyökerében PowerShellből:

```powershell
.\.venv\Scripts\python.exe -m pip install ultralytics opencv-python numpy python-chess streamlit streamlit-image-coordinates
```

A `python-chess` importneve `chess`, ezért ezzel ellenőrizzük:

```powershell
.\.venv\Scripts\python.exe -c "import chess; print('python-chess OK'); print(chess.__version__)"
```

A webes alkalmazás indítása:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

A futó Streamlit leállítása: `Ctrl + C` a terminálban.

A csomagok rögzítése a függőségi fájlba:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements.txt
```

Megjegyzés: a `pip freeze` az aktív `.venv` csomagjait mentse; ezért használjuk az explicit `python.exe -m pip` alakot. A `requirements.txt` fájlt ellenőrizni kell, és a ténylegesen szükséges függőségeket érdemes benne tartani.

## 5. `.gitignore` javasolt tartalma

Ellenőrizzük, hogy a `.gitignore` legalább az alábbiakat tartalmazza:

```gitignore
.venv/
venv/
__pycache__/
*.py[cod]
.env
outputs/
models/
weights/
*.pt
```

A `models/`, `weights/` és `*.pt` szabályok kizárják a nagy modellsúlyokat is. Ezután ellenőrizhető például:

```powershell
git check-ignore -v .venv/Lib/site-packages/torch/lib/torch_cpu.dll
git check-ignore -v models/best.pt
git status --short
```

A `git check-ignore` akkor ad megfelelő kimenetet, ha megmutatja a kizáró szabályt. A `.gitignore` önmagában nem szedi ki a korábban már commitolt fájlokat; a jelenlegi projektben a tiszta Git-előzmény újraépítése megtörtént, a felhasználó szerint a push-probléma megoldódott.

## 6. Ismert problémák és korlátok

1. **A felismerés pontossága nem elég jó.** Alacsony konfidencián sok a téves/átfedő találat, magas konfidencián több bábu kimarad. Ezt nem elég egyetlen küszöb átállításával megoldani.
2. **A sarokpontok pontossága számít.** A játszható mezők négy sarkát kell kijelölni, nem a fa keret külső sarkait. A sorrend legyen `a8`, `h8`, `h1`, `a1` a felület által jelzett koordinátázás szerint.
3. **A korábbi webes futtatás beragadt betöltést jelzett.** Egy képernyőképen a felhasználó a „Modell betöltése és felismerés…” szöveget látta. Akkor még nem volt tisztázva, hogy modellbetöltés, felismerés vagy a régebbi felület külön ablakos kijelölése okozta-e. A legutóbbi `app.py` a kijelölést a böngészőbe helyezi, de ezt a helyi példányban újra kell tesztelni.
4. **FEN-kód korlátai.** A fényképből a bábuelrendezés becsülhető. A FEN további mezői közül a program alapértelmezetten `w - - 0 1` értéket tesz meg (világos lép, nincs megadott sáncolási vagy en passant jog). Ezeket egyetlen képből általában nem lehet biztosan kikövetkeztetni, ezért a FEN teljes játékelőzményhez nem tekinthető automatikusan teljesnek.
5. **A `python-chess` érvényességellenőrzése nem garantálja a vizuális helyességet.** Egy formailag elfogadott állás még lehet rosszul felismert. A felhasználónak kell ellenőriznie a táblát.
6. **Nincs még Stockfish-integráció.** A program jelenleg állást állít elő, de a legjobb lépés és a motorelemzés még nincs beépítve.
7. **Nincs még automatikus játékmenet- vagy lépésfelismerés.** Egy fotóból csak az aktuális állás becslésével foglalkozunk.

## 7. Következő lépések – ajánlott sorrend

### 1. A jelenlegi webes prototípus működésének igazolása

- Ellenőrizni, hogy a helyi `app.py` a legutóbbi, böngészős sarokkijelölést használó változat.
- Elindítani az alkalmazást a fenti paranccsal.
- Feltölteni egy fotót.
- Kijelölni a négy sarkot.
- Megnyomni a felismerés gombot.
- Ha elakad: a PowerShell terminál teljes hibaüzenetét és a böngésző állapotát rögzíteni. A végtelen töltés okát nem szabad találgatással megállapítani.

### 2. Egy egyszerű, kontrollált tesztkép

Először olyan képpel teszteljünk, amelyen a teljes sakktábla jól látszik, nincs levágva, a bábuk nem takarják egymást túlzottan, és a fényviszonyok jók. A kezdeti tesztet lehetőleg közel felülnézeti fotóval végezzük.

### 3. Felismerési eredmény és kézi javítás ellenőrzése

- Vizsgálni, hogy a detekciós dobozok a megfelelő bábukra kerülnek-e.
- Ellenőrizni a 64 mezőre való hozzárendelést.
- Hibás felismerést kézzel kijavítani a szerkesztőben.
- Megnézni, hogy a módosítás után frissül-e a FEN, és a letöltött `.fen` állomány valóban a javított állást tartalmazza-e.

### 4. Stockfish hozzáadása

Ha a digitális állás létrehozása stabil, a következő funkció a Stockfish integrációja lehet:

- Stockfish motor telepítése / elérési út beállítása;
- a kézzel ellenőrzött FEN átadása a motornak;
- legjobb lépés és értékelés megjelenítése;
- a motorválasz hibakezelése.

Ezután már készíthető egy kerek, rövid egyetemi bemutató: fizikai tábla fotója → felismerés → kézi ellenőrzés → digitális állás → sakkmotor-javaslat.

## 8. Bemutatóhoz javasolt működési folyamat

1. A felhasználó feltölt egy jól látható sakktáblafotót.
2. Kijelöli a négy sarokpontot a böngészőben.
3. A YOLO felismeri a bábukat.
4. A perspektívatranszformáció alapján minden találat a legvalószínűbb mezőre kerül.
5. A felhasználó átnézi és szükség esetén kijavítja a 64 mezőt.
6. A program megjeleníti és letölthetővé teszi a FEN-t és az annotált képet.
7. A következő fejlesztési fázisban Stockfish kiszámítja a javasolt lépést.

## 9. Mit kell feltölteni GitHubra?

**Feltölthető:** `app.py`, `chess_board.py`, `.gitignore`, `requirements.txt`, dokumentáció (`README.md`, ez a fájl), valamint kis méretű saját tesztképek, ha azok megoszthatók.

**Ne töltsük fel:** `.venv/`, `models/best.pt`, `weights/best.pt`, a generált `outputs/` könyvtár, titkos kulcsok, `.env` fájlok.

A nagy modellfájlt a program helyileg használja; a dokumentációban leírható, hogyan szerezhető be. A modell licencét a projekt nyilvános bemutatása előtt ellenőrizni kell.

---

## Jelenlegi státusz egy mondatban

A böngészős felület, a YOLO-integráció, a perspektívaalapú mező-hozzárendelés, a FEN-előállítás és a kézi javítás kódja rendelkezésre áll; **a következő feladat a teljes helyi folyamat tesztelése, a beragadó betöltés okának diagnosztizálása és a felismerési pontosság javítása, majd a Stockfish hozzáadása**.
