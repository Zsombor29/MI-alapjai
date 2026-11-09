# Sakkfelismerő – V6

## Mi változott a V5-höz képest?

A V6 a ChessSight projekt hivatalosabb, hárommenetes pozícióolvasási módszerét követi:

1. teljes fotó felismerése;
2. a tábla köré készített négyzetes kivágás felismerése (a bábuknak extra felső margóval);
3. ugyanennek a kivágásnak a vízszintesen tükrözött felismerése.

A három menet osztálypontszámait súlyozottan összevonjuk. A teljes fotó súlya a tábla tényleges képméretéhez igazodik. A bábuk mezőhöz rendelésénél a doboz aljától 96%-nál számított talppontot használjuk (nem a doboz legalját és nem a középpontját). Ez a ChessSight szerzőinek dokumentált pozícióolvasási megoldása.

Az összevont pontszám után a program a modell által ugyanarra a mezőre felajánlott alternatívák között sakkanyag-konzisztenciát is alkalmaz. Nem talál ki új bábut, csak a modell saját jelöltjei között választhat. Ez javíthatja a futó–huszár és vezér–bástya jellegű tévesztéseket, ha a második osztálycímkét is felajánlotta a modell. Promóciós állásoknál az anyagkorlát csak heurisztika, ezért az ilyen módosítások külön megjelennek.

A tábla tájolását a mezők világos/sötét mintája, a bábuk elhelyezkedése és a gyalogok iránya együtt határozza meg.

## Telepítés

A projekt gyökerében:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements_chess_app.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Az app a meglévő `.venv`-et és a Hugging Face cache-t használja. A `detector.py` és `chess_board.py` nem módosul.

## Ismert korlát

A jelenlegi változat ugyanazt a ChessSight RT-DETR modellt futtatja három különböző képi nézetben. Ez nem független második neurális háló, hanem többnézetes modellfúzió. A „Második tipp” oszlopban ugyanannak a modellnek a másik osztályjavaslata látszik.

## Későbbi, célzott finomhangolás

A felhasználó saját bábukészletéhez külön chesscog alapú occupancy/piece klasszifikátor finomhangolható. A chesscog kutatás két, kezdőállást mutató fotóval dolgozik: egy kép a fehér, egy a fekete játékos perspektívájából. A két képből 128 négyzetminta származik, majd augmentációval finomhangolják a két osztályozót. A publikáció az adott tesztkészleten 0,17%-os négyzetenkénti hibát közölt, de ez nem garantálható másik készleten.

A chesscog hivatalos csomagja Python 3.8–3.10 közötti verziót kér, ezért a mostani V6 app `.venv` környezetébe nem szabad vakon telepíteni; külön, Python 3.10-es környezet és integráció kellene. A két fotót a saját fizikai készletről kell elkészíteni, nem vízjeles stockfotót használni.
