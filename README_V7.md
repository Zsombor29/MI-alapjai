# Sakkállás-felismerő – V7

## Mi változott?

- A ChessSight RT-DETR és a sarokmodell változatlanul marad.
- Megmarad a három nézetes felismerés: teljes kép, nagyított táblakivágás, tükrözött kivágás.
- **Eltávolítottuk a sakkanyag-korlát szerinti címkecserét.** Egyetlen modell által javasolt bábut sem írunk át azért, mert a táblán több királyt, vezért, bástyát, futót, huszárt vagy gyalogot talált a program a kezdőállásban szokásos darabszámnál.
- Minden mezőn a három nézet súlyozott összevont pontszámának legmagasabb értékű osztálya marad.
- A bizonytalan mezők táblázata megmutatja a teljes fotó, a nagyított kivágás és a tükrözött kivágás külön tippjét és pontszámát.
- A Streamlit gyorsítótár verziókulcsa V7-re változott, így a korábbi eredmény nem fog visszajönni a gyorsítótárból.

## Miért vettük ki a sakkanyag-korlátot?

A szabály a ChessSight nyilvános teszthalmazán javította a teljes állás pontosságát, de heurisztika: a kezdőállás darabszámaiból indul ki, és egy vizuálisan nagyobb pontszámú címkét is lecserélhet egy gyengébb alternatívára. Promóció, nem szokványos állás vagy egy konkrét készlet felismerési hibája esetén ez félrevezető lehet. Ez a verzió ezért megőrzi a képi modell pontszámait, és a sakkállás érvényességét csak jelzi, nem javítja át automatikusan.

## Telepítés Windows / PowerShell alatt

A projekt gyökérmappájában (`C:\Temp\MI alapjai`):

```powershell
# Állítsd le a futó Streamlit alkalmazást Ctrl+C-vel.
# Másold felül az app.py fájlt a ZIP-ből.

.\.venv\Scripts\python.exe -m pip install -r requirements_chess_app.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

A `detector.py` és `chess_board.py` fájlokat nem kell módosítani. A modellfájlok továbbra is a `models` gyorsítótárba kerülnek, és ne töltsd fel őket GitHubra.

## Az új eredmény értelmezése

A bizonytalan mezők táblázatában:

- **Teljes kép:** az eredeti feltöltött fotó eredménye.
- **Nagyított kép:** a tábla körüli kivágás eredménye.
- **Tükrözött kép:** ugyanennek a kivágásnak a tükrözött eredménye.
- **Összesített pont:** a három menet súlyozott fúziójából kapott kiválasztott osztály pontszáma.
- **Nézetegyezés:** a három nézet közül hány választotta ugyanazt a bábut.

Ha mindhárom nézet ugyanazt a rossz bábut választja, akkor a hiba az alapmodell vizuális osztályozásában van. Ha a nézetek eltérnek, a táblázat megmutatja, melyik nézet okozza az eltérést. A modell még nem garantál hibamentes felismerést; további készletspecifikus finomhangoláshoz címkézett saját fotókra lesz szükség.
