# Automatikus sakkállás-felismerő – végső javítási csomag (V9)

## Mi változott?

A V9 megtartja a V7-ben jól teljesítő ChessSight RT-DETR bábufelismerőt és a ChessSight Swin sarokmodellt. Nem cseréli le az alapmodellt, és nem igényel saját modelltréninget.

### 1. Öt képi nézet

A program most öt bemenetből gyűjt osztályjelölteket:

1. eredeti fénykép;
2. automatikusan kivágott, négyzetes tábla környezete;
3. a négyzetes kivágás vízszintesen tükrözött változata;
4. perspektívakorrekcióval kiegyenesített tábla;
5. a kiegyenesített tábla 180 fokkal elforgatott változata.

Az öt nézet osztálypontszámait súlyozottan egyesíti. A negyedik és ötödik nézet kifejezetten azt ellenőrzi, hogy a perspektíva eltávolítása, illetve a 180 fokos forgatás mellett stabil marad-e a bábuosztályozás.

### 2. Többségi másodvélemény a két ismert hibapárra

Csak a gyalog–futó és futó–huszár pároknál használja a következő szabályt:

- az alternatív osztály legalább 3/5 nézetben első helyen szerepel;
- az alternatív osztályt a detektor ténylegesen javasolta az adott mezőn;
- az összesített vizuális pontszámok közötti különbség legfeljebb 0,18;
- a bábu színe nem változhat.

Ha ezek nem teljesülnek, a súlyozott modellpontszám marad érvényben. A többségi korrekció megjelenik az eredmények táblázatában.

### 3. Gyenge kezdőállás-közeli prior

A másodvélemény után egy óvatos pozíciós szabály még segíthet a kezdőállásból kevés lépésre lévő képeken. Csak akkor aktiválódik, ha:

- legalább 24 bábu van a táblán;
- a 32 klasszikus kezdőmezőből legalább 18-on már eleve a megfelelő bábu van;
- legalább 22 kezdőmezőn a várt bábut a modell is támogatja.

A prior kizárólag az alábbi közeli osztálytévesztéseket oldhatja fel:

- saját gyalogmezőn gyalog helyett futó;
- eredeti futómezőn futó helyett huszár.

Az elvárt alternatívának már szerepelnie kell a modell jelöltjei között, a pontszámkülönbség legfeljebb 0,17 lehet, és a bábu színe nem módosulhat. Nem számolja a bábukat, nem kényszerít ki 16/16/4/4/4/2 készletet, nem talál ki hiányzó figurát, és nem ír felül egy egyértelmű vizuális pontszámkülönbséget.

## Telepítés

1. Állítsd le a futó Streamlit alkalmazást `Ctrl + C` billentyűvel.
2. Csomagold ki a ZIP-fájlt.
3. Másold az `app.py` fájlt a `C:\Temp\MI alapjai` mappába, és írd felül a régit.
4. A `requirements_chess_app.txt` fájl változatlan; ha a függőségek már telepítve vannak, nem kell újratelepíteni.
5. Indítás:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

A `detector.py`, `chess_board.py`, `.venv`, valamint a letöltött modellfájlok nem igényelnek módosítást.

## Mit kell ellenőrizni?

Elsőként ugyanazzal a fényképpel kell összehasonlítani a V7 és V9 kimenetét. A V9 eredménye megmutatja az öt nézet egyezését, a bizonytalan mezőket és az esetleges többségi/kezdőállás-közeli korrekciókat.

A kód szintaktikai ellenőrzése és a döntési logika célzott tesztjei sikeresek. A teljes RT-DETR és Swin modellfuttatást a felhasználó Windows környezetében kell ellenőrizni; a V9 nem ígér tökéletes felismerést minden fényképen.

## Licenc és korlátok

A ChessSight detector és corner model a modellkártya szerint CC BY-NC 4.0 licencű, tehát nem kereskedelmi használatra készült. A modell szintetikus Blender-képeken tanult, valós fényképeken is tesztelt, de eltérő sakk-készleteknél, vízjeleknél, erős árnyékoknál és részben takart bábuknál továbbra is lehetnek félreismerések.
