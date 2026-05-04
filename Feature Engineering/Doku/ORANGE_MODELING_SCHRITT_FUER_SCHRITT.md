# Orange Modeling Schritt fuer Schritt

Dieses Dokument beschreibt den praktischen Ablauf in Orange, damit die Modelle fuer Klassifikation (CLS) und Regression (REG) sauber gebaut, fair verglichen und korrekt evaluiert werden.

Es ergaenzt die Datei `ORANGE_MODELING_CHECKLIST.md`.
Die Checkliste prueft den Import. Dieses Dokument beschreibt den eigentlichen Modellierungs- und Evaluationsablauf.

Wichtig fuer dieses Projekt: Die Split-Logik folgt eurer Unterrichtslogik und nicht der klassischen ML-Sprache.
Hier gilt also bewusst:

- `train` = Training
- `test` = Modellvergleich und Tuning
- `validation` / `val` = finaler Holdout ganz am Schluss

---

## 1. Zielbild

Am Ende sollst du fuer beide Stages jeweils genau einen finalen Gewinner haben:

- ein finales CLS-Modell fuer `order`
- ein finales REG-Modell fuer `quantity`

Dabei gilt immer dieselbe Logik:

1. Workflow technisch korrekt aufbauen.
2. Mit `train_sample` nur den Ablauf testen.
3. Mit `train_full` gegen `test` Modelle vergleichen.
4. Danach die konservative Variante ohne Gruppen-Features pruefen.
5. Erst ganz am Schluss den finalen Gewinner einmal auf `validation` / `val` messen.

---

## 2. Harte Regeln

Diese Regeln gelten fuer CLS und REG gleichermassen:

1. Verwende `test` in diesem Projekt fuer Modellwahl, Hyperparameter-Tuning und Widget-Vergleiche.
2. Verwende `validation` / `val` nicht fuer Modellwahl, sondern nur fuer die finale Endmessung.
3. Verwende `train_sample` nur fuer Rauchtest und schnellen Workflow-Aufbau.
4. Verwende `train_full` fuer das eigentliche Training in der Vergleichsphase.
5. Vergleiche immer nur Modelle, die auf demselben Datensatz und mit derselben Zielvariable laufen.
6. Wenn ein Modell wegen Missing Values scheitert, zaehlt der noetige Impute-Schritt zur Pipeline dazu.
7. Wenn du fuer CLS oder REG einen Schwellenwert aenderst, darf das in diesem Workflow auf Basis von `test` passieren. Der Schwellenwert bleibt fuer `validation` / `val` danach fix.

---

## 3. Welche Dateien gehoeren zusammen

### CLS

Basis-Workflow:

- `cls_train_sample.csv` fuer Rauchtest
- `cls_train_full.csv` fuer Haupttraining
- `cls_test.csv` fuer Modellvergleich und Tuning
- `cls_val.csv` fuer finale Endmessung

Konservative Sensitivitaetsvariante:

- `cls_train_full_no_groups.csv`
- `cls_test_no_groups.csv`
- `cls_val_no_groups.csv`

Zielvariable:

- `order`

### REG

Basis-Workflow:

- `reg_train_sample.csv` fuer Rauchtest
- `reg_train_full.csv` fuer Haupttraining
- `reg_test.csv` fuer Modellvergleich und Tuning
- `reg_val.csv` fuer finale Endmessung

Konservative Sensitivitaetsvariante:

- `reg_train_full_no_group34.csv`
- `reg_test_no_group34.csv`
- `reg_val_no_group34.csv`

Zielvariable:

- `quantity`

---

## 4. Vorbereitung in Orange

### 4.1 Vor dem Modellieren

1. Oeffne zuerst `Doku/ORANGE_MODELING_CHECKLIST.md` und arbeite die Import-Checks einmal sauber durch.
2. Lege in Orange zwei getrennte Workflows oder zwei getrennte Tabs an:
   - einen fuer CLS
   - einen fuer REG
3. Speichere die Workflows frueh ab, zum Beispiel als:
   - `orange_cls.ows`
   - `orange_reg.ows`

### 4.2 Grundaufbau eines sauberen Orange-Workflows

Der Grundaufbau ist fuer beide Stages gleich:

1. Ein `File`-Widget fuer die Trainingsdatei anlegen.
2. Ein zweites `File`-Widget fuer die Vergleichsdatei anlegen.
3. Beide Daten einmal mit `Data Table` kontrollieren.
4. Mehrere Learner-Widgets anschliessen.
5. Alle Learner in `Test & Score` einspeisen.
6. Im `Test & Score`-Widget die Methode `Test on test data` waehlen.
7. `train` an `Data` anschliessen.
8. `test` an `Test Data` anschliessen.

Die Logik dahinter ist einfach:

- Orange trainiert auf `train`
- Orange bewertet fuer den Modellvergleich auf `test`
- `validation` / `val` bleibt bis zur finalen Endmessung unberuehrt

### 4.3 Optionaler Impute-Schritt

Einige Modelle, vor allem lineare Modelle, koennen an Missing Values scheitern.

Wenn das passiert:

1. Fuege einen `Impute`-Schritt in die betroffene Pipeline ein.
2. Dokumentiere, dass dieses Modell nur als `Impute + Modell` bewertet wurde.
3. Vergleiche komplette Pipelines, nicht nur den Learner isoliert.

Pragmatische Regel:

- Baumbasierte Modelle zuerst ohne Zusatzschritte testen.
- Lineare Modelle bei Bedarf mit `Impute` erneut aufnehmen.

---

## 5. CLS Schritt fuer Schritt

## 5.1 CLS Rauchtest aufbauen

Ziel: Nur pruefen, ob der Workflow technisch funktioniert.

1. Lade `cls_train_sample.csv` im ersten `File`-Widget.
2. Lade `cls_test.csv` im zweiten `File`-Widget.
3. Oeffne beide Dateien mit `Data Table`.
4. Pruefe noch einmal kurz:
   - Zielspalte ist `order`
   - kategoriale Felder sind diskret
   - Spalten zwischen Train und Test passen zusammen
5. Fuege zuerst nur wenige Modelle hinzu, zum Beispiel:
   - `Logistic Regression`
   - `Tree`
   - `Random Forest`
6. Verbinde alle Learner mit `Test & Score`.
7. Stelle `Test & Score` auf `Test on test data`.
8. Verbinde `cls_train_sample.csv` mit `Data`.
9. Verbinde `cls_test.csv` mit `Test Data`.
10. Pruefe, ob alle Modelle ohne Fehler durchlaufen.

Wenn hier Fehler auftreten, loese sie jetzt. Noch nicht mit `train_full` weitermachen.

## 5.2 CLS Basisvergleich aufsetzen

Ziel: Die eigentliche Modellauswahl fuer CLS.

1. Ersetze `cls_train_sample.csv` durch `cls_train_full.csv`.
2. Lasse `cls_test.csv` als Vergleichsdatei bestehen.
3. Behalte zuerst dieselben Modelle bei, damit der Vergleich konsistent bleibt.
4. Fuege erst dann weitere Kandidaten hinzu, falls sie in Orange verfuegbar sind.
5. Achte darauf, dass alle Kandidaten auf exakt denselben Daten laufen.
6. Dokumentiere pro Lauf die Ergebnisse aus `Test & Score`.

Empfohlene erste CLS-Kandidaten:

- `Logistic Regression` als einfache lineare Baseline
- `Tree` als einfache interpretierbare Nichtlinearitaet
- `Random Forest` als starke baumbasierte Baseline

Wenn in deiner Orange-Installation weitere stabile CLS-Learner verfuegbar sind, kannst du sie danach ergaenzen. Entscheidend ist nicht die Anzahl der Modelle, sondern ein fairer Vergleich.

## 5.3 CLS korrekt bewerten

Verwende fuer CLS eine feste Metrik-Hierarchie.

Empfohlene Reihenfolge:

1. Primaere Vergleichsmetrik: `AUC`
2. Sekundaere Kontrollmetriken: `LogLoss`, `F1`, `Precision`, `Recall`, `CA`

Vorgehen:

1. Sortiere die Modelle zuerst nach `AUC`.
2. Wenn zwei Modelle sehr nah beieinander liegen, ziehe `LogLoss` und `F1` als Tiebreaker heran.
3. Oeffne fuer die besten 2 bis 3 Modelle zusaetzlich `Confusion Matrix`.
4. Pruefe, ob das Modell nur auf dem Papier gut aussieht oder auch eine brauchbare Fehlerstruktur hat.
5. Wenn du einen anderen Klassifikations-Schwellenwert benoetigst, stelle ihn in diesem Workflow anhand von `test` ein.

Wichtig:

- Den Gewinner noch nicht auf `validation` / `val` pruefen.
- Zuerst die Sensitivitaetsvariante ohne Gruppen laufen lassen.

## 5.4 CLS Sensitivitaetscheck ohne Gruppen

Ziel: pruefen, ob `group12` und `group34` wirklich noetig sind.

1. Kopiere den funktionierenden CLS-Basisworkflow.
2. Ersetze die Datendateien durch:
   - `cls_train_full_no_groups.csv`
   - `cls_test_no_groups.csv`
3. Lasse dieselben Modelle mit denselben Einstellungen erneut laufen.
4. Vergleiche die Ergebnisse mit dem Basisworkflow.

Entscheidungsregel:

1. Wenn die No-Group-Variante klar schlechter ist, bleibe bei der Basisvariante.
2. Wenn die No-Group-Variante fast gleich gut ist, darfst du die konservativere Variante bevorzugen.

Danach steht genau ein CLS-Finalist fest:

- entweder Basis: `CLS_FINAL`
- oder konservativ: `CLS_FINAL_NO_GROUPS`

---

## 6. REG Schritt fuer Schritt

## 6.1 REG Rauchtest aufbauen

Ziel: Nur den Workflow pruefen, nicht final entscheiden.

1. Lade `reg_train_sample.csv` im ersten `File`-Widget.
2. Lade `reg_test.csv` im zweiten `File`-Widget.
3. Oeffne beide Dateien mit `Data Table`.
4. Pruefe noch einmal kurz:
   - Zielspalte ist `quantity`
   - kategoriale Felder sind diskret
   - Spalten zwischen Train und Test passen zusammen
5. Fuege zuerst wenige robuste Modelle hinzu, zum Beispiel:
   - `Linear Regression`
   - `Tree`
   - `Random Forest`
6. Verbinde alles mit `Test & Score`.
7. Stelle wieder `Test on test data` ein.
8. Pruefe, ob der Ablauf technisch sauber durchlaeuft.

## 6.2 REG Basisvergleich aufsetzen

1. Ersetze `reg_train_sample.csv` durch `reg_train_full.csv`.
2. Lasse `reg_test.csv` als Vergleichsdatei bestehen.
3. Vergleiche die Modelle auf identischem Datensatz.
4. Dokumentiere die Kennzahlen aus `Test & Score` nach jedem Lauf.

Empfohlene erste REG-Kandidaten:

- `Linear Regression` als einfache Baseline
- `Tree` als interpretierbare Nichtlinearitaet
- `Random Forest` als starke baumbasierte Baseline

Wenn ein lineares Modell an Missing Values scheitert, nimm es als `Impute + Linear Regression` in die Vergleichsliste auf.

## 6.3 REG korrekt bewerten

Verwende fuer REG eine feste Metrik-Hierarchie.

Empfohlene Reihenfolge:

1. Primaere Kontrollmetriken: `MAE` und `RMSE`
2. Zusaetzliche Plausibilitaetsmetrik: `R2`

Vorgehen:

1. Bevorzuge Modelle mit kleinerem `MAE` und kleinerem `RMSE`.
2. Nutze `R2` nur als Zusatzsignal, nicht allein als Entscheidung.
3. Wenn zwei Modelle aehnlich sind, bevorzuge das stabilere und einfachere Modell.
4. Wenn moeglich, schaue dir fuer den Finalisten die Vorhersagen in einer Tabelle an, um grobe Fehlmuster zu erkennen.

Wichtig:

- Wieder gilt: noch nicht mit `validation` / `val` arbeiten.
- Erst danach die konservative Variante ohne `group34` pruefen.

## 6.4 REG Sensitivitaetscheck ohne `group34`

1. Kopiere den funktionierenden REG-Basisworkflow.
2. Ersetze die Datendateien durch:
   - `reg_train_full_no_group34.csv`
   - `reg_test_no_group34.csv`
3. Lasse dieselben Modelle mit denselben Einstellungen erneut laufen.
4. Vergleiche Basisvariante gegen konservative Variante.

Entscheidungsregel:

1. Wenn die No-Group34-Variante deutlich schlechter ist, bleibe bei `REG_FINAL`.
2. Wenn sie nahezu gleich gut ist, darfst du `REG_FINAL_NO_GROUP34` als robustere Endvariante bevorzugen.

Danach steht genau ein REG-Finalist fest:

- entweder Basis: `REG_FINAL`
- oder konservativ: `REG_FINAL_NO_GROUP34`

---

## 7. Finale Evaluationsphase

Erst jetzt beginnt die eigentliche Endmessung.

Bis hierhin wurden Modelle nur mit `train_full` gegen `test` verglichen.
Jetzt wird genau ein Gewinner pro Stage final auf `validation` / `val` bewertet.

## 7.1 Zuerst Gewinner fixieren

Vor der finalen Validation muessen diese Punkte feststehen:

1. Welcher Learner gewinnt.
2. Welche Hyperparameter verwendet werden.
3. Ob ein Impute-Schritt zur finalen Pipeline gehoert.
4. Ob die Basis- oder die konservative Feature-Variante gewinnt.
5. Bei CLS optional: welcher feste Schwellenwert verwendet wird.

Wenn einer dieser Punkte noch offen ist, darf `validation` / `val` noch nicht benutzt werden.

## 7.2 Empfohlener finaler Aufbau

Fuer die finale Messung gibt es zwei saubere Optionen.

### Option A: Einfach und konservativ

1. Trainiere den Gewinner auf `train_full`.
2. Messe einmal auf `validation` / `val`.

Vorteil:

- sehr einfach und sauber

Nachteil:

- `validation` / `val` wird nicht mehr zum finalen Training genutzt

### Option B: Empfohlen nach abgeschlossener Modellwahl

1. Fixiere den Gewinner komplett auf Basis von `test`.
2. Verbinde `train_full` und `test` mit `Concatenate`.
3. Nutze das zusammengefuehrte Dataset als finales Training.
4. Nutze `validation` / `val` genau einmal als finales Holdout.

Diese Option ist methodisch sauber, wenn wirklich vorher alles festgelegt wurde.

## 7.3 Welche Validierungsdatei gehoert zu welchem Gewinner

Wenn der Basisworkflow gewinnt:

- CLS: `cls_val.csv`
- REG: `reg_val.csv`

Wenn die konservative Variante gewinnt:

- CLS: `cls_val_no_groups.csv`
- REG: `reg_val_no_group34.csv`

## 7.4 Was du final berichten solltest

Fuer CLS:

1. Name des finalen Learners
2. Ob Basis oder No-Groups gewonnen hat
3. Finale Validation-Metriken: mindestens `AUC`, `LogLoss`, `F1`, `Precision`, `Recall`, `CA`
4. Falls verwendet: finaler Klassifikations-Schwellenwert

Fuer REG:

1. Name des finalen Learners
2. Ob Basis oder No-Group34 gewonnen hat
3. Finale Validation-Metriken: mindestens `MAE`, `RMSE`, `R2`

---

## 8. Konkrete Arbeitsreihenfolge

Wenn du einfach nur exakt wissen willst, was du nacheinander tun sollst, dann halte dich an diese Reihenfolge:

1. Import-Check mit `ORANGE_MODELING_CHECKLIST.md` abschliessen.
2. CLS-Rauchtest mit `cls_train_sample.csv` und `cls_test.csv` aufbauen.
3. REG-Rauchtest mit `reg_train_sample.csv` und `reg_test.csv` aufbauen.
4. CLS-Basisvergleich mit `cls_train_full.csv` gegen `cls_test.csv` durchfuehren.
5. REG-Basisvergleich mit `reg_train_full.csv` gegen `reg_test.csv` durchfuehren.
6. CLS-No-Groups-Sensitivitaetscheck durchfuehren.
7. REG-No-Group34-Sensitivitaetscheck durchfuehren.
8. Pro Stage genau einen Gewinner fixieren.
9. Finalen Gewinner pro Stage einmal auf dem passenden `validation`-Split messen.
10. Ergebnisse dokumentieren und danach nichts mehr anhand von `validation` / `val` nachoptimieren.

---

## 9. Typische Fehler, die du vermeiden musst

1. `validation` / `val` zu frueh oeffnen und dadurch unbewusst auf das finale Holdout optimieren.
2. CLS- und REG-Dateien in demselben Workflow vermischen.
3. Modelle auf unterschiedlichen Datensatzvarianten vergleichen.
4. No-Groups-Varianten mit der Basisvariante vermischen.
5. Nach dem ersten Blick auf `validation` / `val` doch noch einmal Modell oder Parameter aendern.
6. Nur eine Metrik anschauen, obwohl ein zweites Kontrollsignal einen Widerspruch zeigt.

---

## 10. Kurzentscheidung fuer dieses Projekt

Wenn du pragmatisch und sauber vorgehen willst, ist die beste Reihenfolge fuer dieses Projekt:

1. Zuerst mit baumbasierten Modellen starten.
2. Lineare Modelle als einfache Baseline mitlaufen lassen.
3. Die No-Groups-Varianten als Sensitivitaetscheck behandeln, nicht als Standard.
4. `validation` / `val` wirklich erst ganz am Schluss anfassen.

Damit ist der Orange-Teil methodisch sauber aufgebaut:

- Rauchtest auf Sample
- Modellwahl auf Test
- Sensitivitaetscheck auf konservativer Variante
- finale Endmessung auf Validation
