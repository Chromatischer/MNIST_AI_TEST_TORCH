<h1>Vorbereitung Informatik Klausur 1 Q2 </h1>

<h2>Arrays in Javascript </h2>
Zeile 1 leitet den Javascript part des html files ein.

Zeile 2 definiert und initialisiert ein Array mit den Daten: der Temperaturen

Zeile 3 beginnt die funktion minTemperatur
 - Die Variable ``m`` wird auf das erste Element des Temperaturarrays gesetzt
 - nun beginnt eine Schleife welche die interne Variable ``i`` hat welche von 1 bis zur länge der Temperatur erhöht wird
 - als nächstes folgt eine Kondition wenn die aktuell ausgewählte Temperatur ``temp[i]`` kleiner als die variable ``m`` ist, so wird ``m`` auf ``temp[i]`` gesetzt.
 - Am Ende der Schleife wird ``m`` zurückgegeben

```javascript
<script>
    temp =[12.4,14.1,9.3,12.0,15.9,16.5,15.7,15.1,11.4,8.0,13.1,13.4,15.8,12.2];
    function minTemperatur(){
        m = temp[0];
        for (let i = 1; i < temp.length; i++) {
        if (temp[i] < m){
            m = temp[i];
        }
    }
    return m;
    }
</script>
```
Die funktion ``avgTemperatur`` ist relativ einfach zu implementieren, da man nur alle Werte addieren muss und dann durch die Anzahl der Werte teilen muss
```javascript
    function avgTemperatur(array){
        avg = 0
        for (let i = 0; i < array.length; i++){
            avg += array[i];
        }
        return avg / array.length
    }
```
<h2>Suchverfahren</h2>

Binäre Suche:
```javascript
function binSuche(schluessel, feld){
    let min = 0;
    let mitte = 0;
    let max = feld.length - 1;
    while (min <= max) {
        mitte = Math.floor((max + min) / 2);
        if (feld[mitte] == schluessel) {
            return mitte;
        } else if (feld[mitte] < schluessel) { 
            min = mitte + 1;
        } else { 
            max = mitte - 1;
        }
        return - 1;
    }
}
```
Gegeben sei ein Array:

Gesucht wird nach 27

Die Länge des Datenfeldes ist 9

| 1   | 2   | 3   | 4   | 6   | 19  | 27  | 31  | 40  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| min |     |     |     | mid |     |     |     | max |
|     |     |     |     |     | min | mid |     | max |
|     |     |     |     |     |     | mid |     |     |

Im ersten Schritt wird min auf die jetzige Mitte gesetzt, da die gesuchte Zahl größer als mitte ist. Die neue mitte wird berechnet und liegt diesmal direkt auf dem gesuchten Schlüssel 27. Die Komplexität ist ``O(log(n))`` da man mit steigender Feldgröße auch mehr des Feldes eliminieren kann.

<h2>Sortierverfahren</h2>

Gegeben sei ein Sortieralgorythmus:
```javascript
function sort(feld) {
    let minindex = 0;
    let wert = 0;
    for(let i = 0; i <= feld.length - 2; i++){
        minindex = i;
        for(let j = i + 1; j <= feld.length - 1; j++){
            if(feld[minindex] > feld[j]){
                minindex = j;
            }
        }
        wert = feld[minindex];
        feld[minindex] = feld[i]; 
        feld[i] = wert;
    }
}
```
Und ein Array ``[15,22,4,7]``
| i   | j   | minIndex | feld[minindex] | feld[j] | feld        |
| --- | --- | -------- | -------------- | ------- | ----------- |
| 0   | 3   | 2        | 4              | 4       | [4,22,15,7] |
| 1   | 3   | 3        | 7              | 22      | [4,7,15,22] |
| 2   | 3   | 2        | 15             | 15      | [4,7,15,22] |

Denke ich...

Der Iterator ``i`` bewegt sich von 0 bis zu zwei werte vor dem Ende des Datenfeldes. Für jedes ``i`` wird weiterhin mit ``j`` iteriert. ``minindex`` ist eine Hilfsvariable, welches für jedes ``i`` auf ``i`` anfangs gesetzt wird, da das Array bis zu diesem Punkt bereits sortiert wurde. Für jedes ``j`` wird geprüft, ob der aktuelle wert des Datenfeldes an position ``minindex`` kleiner ist als der wert im Datenfeld an Position des iterators ``j``. Wenn dem nicht der fall ist, so wird ``minindex`` auf ``j`` gesetzt. Nach vollenden dieser werden die Werte an Position ``minindex`` und ``j`` vertauscht.

Bei beiden ist die Anzahl der Vergleiche 6. Für Datenfelder der Größe $n$ ist die Anzahl der Vergleiche: $(n \times (n-1))/2$. Daraus folgt: $O(n^2 / 2)$.

d)
| best-case | avg-case | worst-case |
|-----------|----------|------------|
|n-1        |nˆ2/4     |n*(n-1)/2   |
|n*(n-1)/2  |n*(n-1)/2 |n*(n-1)/2   |

e)
``feld[minindex] > feld[j]`` muss durch: ``feld[minindex] < feld[j]`` ersetzt werden.


## Rekursion

a)
```js
function whoAmI(n){
    if (n==1){
        return n;
    } else {
        return 2 * (whoAmI(n-1))+1;
    }
}
```
Rekursion ist wenn eine Funktion sich selbst aufruft. (Unter Rekursion versteht man eine Funktion, die sich selbst aufruft). Wenn die Funktion mit dem Wert n=1 aufgerufen wird, gibt sie n zurück. In allen anderen Fällen, ruft sie sich selbst mit dem Wert n-1 erneut auf. Das Ergebnis wird verdoppelt und 1 wird addiert.

Vorteile einer Problemlösung durch Rekursion sind:
1. kurz und leicht verständlich (je nach Problemstellung)
2. keine Iteration wird benötigt

Nachteile:
1. hohe Speicherbelastung durch viele Funktionsaufrufe
2. langsamer als iterative Problemlösung (problemabhängig)
   
| n | whoAmI(n) |
|---|-----------|
| 1 | 1 |
| 2 | 3 |
| 3 | 7 |
| 4 | 15|
| 5 | 31|
| 6 | 63|
| 7 | 127|
| n | (2^n)-1|

```js
function whoAmIIterativ(n){
    res = n;
    for (i=1; i < n; i++){
        res = res * res;
    }
    return res - 1
}
```



