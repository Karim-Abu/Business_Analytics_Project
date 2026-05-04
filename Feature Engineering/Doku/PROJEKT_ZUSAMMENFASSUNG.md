# Projektstand – Data Preparation und Orange-Handoff

Diese Zusammenfassung beschreibt den belegten aktuellen Stand des Projekts.
Sie basiert auf dem Workspace, den erzeugten Outputs, der Datei `IMPLEMENTATION_STATUS.md`, den Audit-Ergebnissen, den Feature-Selection-Reports und den Orange-Export-Artefakten.

## 1. Projektziel

Das Projekt nutzt ein Two-Stage-Setup.

In **Stage 1** wird vorhergesagt, ob ein Produkt gekauft wird. Die Zielvariable ist `order` mit den Werten 0 oder 1.

In **Stage 2** wird nur für echte Käufe die Menge vorhergesagt. Die Zielvariable ist `quantity`.

Die Datenaufbereitung läuft in Python. Dazu gehören Laden, Join, Bereinigung, Feature Engineering, Sampling, Audits und Export.

Die eigentliche Modellierung ist für Orange vorgesehen. Python liefert dafür die vorbereiteten CSV-Dateien.

## 2. Datenbasis

Die Rohdaten kommen aus `Data/train.csv` und `Data/items.csv`.

Beide Dateien werden über `pid` zusammengeführt. Damit stehen Transaktionsdaten und Produktstammdaten in einem gemeinsamen Modellierungsdatensatz zur Verfügung.

Der Split ist **zeitbasiert** und nicht zufällig:

- Train: Tage 26–70
- Test: Tage 71–81
- Validation: Tage 82–92

Train Set: Modell lernt Muster.
Test Set: wird für Hyperparameter-Optimierung / Modellanpassung verwendet.
Validation Set: spätester Holdout-Split zur finalen Auswahl/Bewertung des optimierten Modells.

Ein Random Split wäre hier methodisch schwächer. Er würde frühere und spätere Tage mischen. Das wäre für ein zeitliches Vorhersageproblem unnatürlich und würde die Trennung von Vergangenheit und Zukunft aufweichen.

Zusätzlich ist in der Projektdokumentation ein Strukturbruch vor Tag 26 festgehalten. Deshalb beginnt das reguläre Training bewusst erst ab Tag 26.

## 3. Preprocessing

Das Preprocessing bereitet die Rohdaten so auf, dass daraus stabile Modellfeatures gebaut werden können.

Missing Values werden je nach Feldtyp unterschiedlich behandelt. Kategoriale Felder bekommen explizite Platzhalter wie `MISSING` oder `NONE`. Numerische Werte bleiben je nach Feld entweder fehlend oder bekommen nur einen klar dokumentierten Fallback.

Bei `competitorPrice` gilt eine harte Regel: Werte kleiner oder gleich 0 werden auf `NaN` gesetzt. Zusätzlich wird ein Missing-Flag gebaut. Eine eigene Imputation für `competitorPrice` wurde bewusst nicht in die Data-Preparation eingebaut.

Kategoriale Felder wie `category_norm`, `pharmForm_norm` und `campaignIndex_norm` werden normalisiert, damit Schreibweisen konsistent sind und die späteren Features stabil bleiben.

`unit`, `group` und `content` werden nur leicht normalisiert. Das Ziel ist nicht eine tiefe fachliche Interpretation, sondern eine saubere technische Basis. Aus `group` werden später `group12` und `group34` abgeleitet. Aus `content` werden Packungsmerkmale wie `is_multipack`, `pack_n`, `pack_size` und `pack_total_size` erzeugt.

`quantity` wird für Bestellzeilen aus `revenue / price` abgeleitet. Gleichzeitig wird `qty_suspicious` gebaut. Diese Markierung greift bei negativen Werten, nicht ganzzahligen Mengen oder fehlender Menge trotz `order = 1`.

Nach dem Preprocessing läuft ein Integritätscheck. Damit wird technisch abgesichert, dass die Grundregeln der Datenaufbereitung eingehalten wurden.

## 4. Feature Engineering

Das Projekt trennt klar zwischen **Safe Features** und **Conditional Features**.

Safe Features sind direkt aus den vorhandenen Feldern ableitbar. Sie brauchen keine Verlaufshistorie. Beispiele sind Zeitmerkmale wie `day`, Preismerkmale wie `price` oder `price_diff_bin`, Stammdatenmerkmale wie `category_norm`, `pharmForm_norm`, `group12`, `group34` und Packungsmerkmale.

Conditional Features nutzen historische Information, werden aber so gebaut, dass keine Zukunftsinformation in das Training einfließt. Dazu gehören zum Beispiel `pid_total_events`, `click_time`, `order_time`, `group12_order`, `group34_order`, `pid_prob`, `availability_likelihood`, `day_7_likelihood` und im Regressionsfall `day_7_qty_mean_oof`.

Diese Trennung ist methodisch wichtig. Safe Features sind robust und direkt verfügbar. Conditional Features sind aufwendiger, liefern aber oft deutlich mehr Signal.

Das sieht man besonders bei der Klassifikation. In den Embedded-Selection-Ergebnissen liegt `CLS_SAFE_PRUNED` nur bei einer PR-AUC von rund 0.25. `CLS_FULL_PRUNED` mit Conditional Features liegt bei rund 0.42. Für CLS helfen diese Features also klar.

Die in der Family Ablation verwendeten Sammelnamen bedeuten:

- `conditional_history`: kumulative Verlaufssignale plus Aggregationen wie `pid_total_events`, `order_time`, `group12_order` und `group34_order`
- `conditional_oof_segment`: zeitbewusste OOF-Signale plus Segmentinformation wie `pid_prob`, `availability_likelihood`, `day_7_likelihood` beziehungsweise `day_7_qty_mean_oof` sowie `pid_segment`

OOF bedeutet hier: Ein Trainingsblock darf nur mit Information aus früheren Blöcken kodiert werden. Es gibt also kein zufälliges K-Fold-Encoding.

## 5. Sampling

Sampling dient nur der schnelleren Entwicklung.

Es wird nur auf dem Train-Split angewendet. Validation und Test bleiben ungesampelt.

CLS und REG werden getrennt gesampelt.

Für CLS wird auf dem vollen Train-Datensatz gesampelt. Die Schichtung läuft über `week_block`, `order` und `pid_segment`.

Für REG wird zuerst die Stage-2-Maske angewendet. Es bleiben also nur Zeilen mit `order = 1`, vorhandener `quantity` und unauffälliger `qty_suspicious` übrig. Danach wird über `week_block`, `quantity_class` und `pid_segment` geschichtet.

Die Sampling-Audits zeigen nur sehr kleine Abweichungen zwischen Population und Sample. Der maximale absolute Unterschied liegt bei CLS bei 0.0005 und bei REG bei 0.0001.

Wichtig ist die Rolle des Samples: `train_sample` ist nur für schnelle Entwicklungsvergleiche gedacht. Für finale Modellfreigaben sollte `train_full` die Basis sein.

## 6. Feature Selection

Die Feature Selection wurde auf den vollen Train-Matrizen durchgeführt, nicht auf den gesampelten Daten.

Zuerst wurden Filter-Methoden angewendet. Dazu gehören Mutual Information, Near-Zero-Variance und Redundanz-Prüfungen über hohe Korrelationen.

Danach wurden bereinigte Kandidatensets aufgebaut. In diesem Schritt wurden schwache oder redundante Features systematisch entfernt.

Anschließend folgte die Embedded Selection. Dafür wurden lineare Modelle und baumbasierte Modelle verwendet. Die linearen Modelle liefen aus Praktikabilitätsgründen auf einem Subsample von 200.000 Zeilen. Die baumbasierten Modelle liefen auf dem vollen Train-Split.

Zum Schluss wurde eine Family Ablation durchgeführt. Dabei wurde geprüft, wie sich ganze Merkmalsfamilien auf die Güte auswirken.

Das Ergebnis sind zwei unterschiedliche finale Sets:

- `CLS_FINAL` mit 28 Features
- `REG_FINAL` mit 26 Features

Dass die Sets unterschiedlich sind, ist erwartbar. Beide Stufen haben andere Ziele. Außerdem gelten für REG strengere Regeln, weil dort nur echte Käufe modelliert werden und einige Features methodisch nicht zulässig sind. Ein Beispiel ist `num_pid_order`, das für REG als Leakage gilt und deshalb ausgeschlossen bleibt.

## 7. Finale Feature-Sets

`CLS_FINAL` ist das finale Set für die Vorhersage der Kaufwahrscheinlichkeit. Es kombiniert Zeitmerkmale, Preismerkmale, Produktstammdaten und starke Conditional Features.

`REG_FINAL` ist das finale Set für die Vorhersage der Kaufmenge. Es nutzt ebenfalls Zeit- und Preismerkmale, aber mit einer etwas anderen Auswahl. Rohes `group12` ist dort nicht enthalten, gruppenbasierte Aggregationen dagegen schon.

Beide Sets sind praktisch nutzbar. Sie wurden nicht nur aus einer einzelnen Metrik abgeleitet, sondern aus Filter-Reports, Embedded Selection und Family Ablation zusammengeführt.

Trotzdem sollte man einige Rohkategorien vorsichtig einordnen. Das gilt besonders für `group12` und `group34`.

Der Grund ist einfach: In den Filter-Ergebnissen sind die direkten Signale dieser Rohfelder eher schwach als viele Conditional Features. In den Embedded-Ergebnissen für CLS gehören sie ebenfalls nicht zu den stärksten Merkmalen. Genau deshalb wurden konservative Vergleichsvarianten gebaut:

- `CLS_FINAL_NO_GROUPS` mit 26 Features
- `REG_FINAL_NO_GROUP34` mit 25 Features

Diese Varianten sind kein Ersatz für die Hauptsets. Sie sind als Robustheitscheck gedacht.

## 8. Orange-Export

Der finale Export läuft im Modus `safe_plus_conditional`.

Unter `outputs/orange_exports/` liegen aktuell 14 Datendateien plus `export_manifest.csv`.

Es gibt vier Basisdateien für CLS:

- `cls_train_full.csv`
- `cls_train_sample.csv`
- `cls_val.csv`
- `cls_test.csv`

Es gibt vier Basisdateien für REG:

- `reg_train_full.csv`
- `reg_train_sample.csv`
- `reg_val.csv`
- `reg_test.csv`

Dazu kommen sechs konservative Varianten:

- `cls_train_full_no_groups.csv`
- `cls_val_no_groups.csv`
- `cls_test_no_groups.csv`
- `reg_train_full_no_group34.csv`
- `reg_val_no_group34.csv`
- `reg_test_no_group34.csv`

Die Rollen der Splits sind klar:

- `train_sample`: schneller Rauchtest und schneller Workflow-Aufbau in Orange
- `train_full`: eigentliches Training
- `val`: Vergleich von Modellen und Einstellungen
- `test`: letzte Prüfung ganz am Schluss

Im Manifest stehen pro Datei unter anderem Dateiname, Stage, Split, Zeilenzahl, Feature-Anzahl, Zielvariable, verwendetes Feature-Set, Sampling-Info, Build-Modus und die Information, ob die REG-Maske angewendet wurde.

Der Export ist für Orange zusätzlich gehärtet worden. Zielspalten stehen am Ende. Relevante kategoriale Felder werden als String exportiert. `category_norm` bekommt zusätzlich das Präfix `C_`, damit Orange numerisch aussehende Kategorien nicht fälschlich als numerisch behandelt.

Vor der eigentlichen Modellierung gibt es noch einen manuellen Orange-Check. Dafür liegt die Datei `Doku/ORANGE_MODELING_CHECKLIST.md` vor.

Die manuellen Checks sind:

- Zielspalte korrekt erkannt
- diskrete Felder in Orange wirklich als diskret erkannt
- Zeilenzahlen mit `export_manifest.csv` abgeglichen
- keine unerwarteten Spalten im Import

## 9. Aktueller Status

Der aktuelle Stand ist klar:

- Preprocessing: abgeschlossen genug
- Safe Feature Engineering: abgeschlossen genug
- Conditional Feature Engineering: abgeschlossen genug
- Sampling: abgeschlossen genug
- Feature Selection: abgeschlossen genug
- Orange-Export: bereit

Die technischen Artefakte für die Modellierung liegen vor. Die Exportdateien wurden erzeugt, das Manifest ist vorhanden und die konservativen Varianten sind mit exportiert.

Damit kann die Modellierung in Orange starten.

Offen sind nicht mehr die Data-Preparation-Schritte, sondern die manuellen Orange-Checks, der eigentliche Modellvergleich und die finale Bewertung.

## 10. Wichtige methodische Hinweise

Lineare Modelle in Orange können einen Impute-Schritt brauchen. Der Grund ist nicht Orange-spezifisch, sondern datenseitig: Im Train-Split gibt es weiterhin Missing Values, zum Beispiel bei `competitorPrice` mit 3.81 % Missingness und bei einigen davon abgeleiteten Preisfeldern. Wenn ein lineares Modell diese Werte nicht direkt akzeptiert, sollte davor ein Impute-Widget stehen.

Baummodelle sind ein guter Einstieg. In den Embedded-Selection-Ergebnissen waren die HistGradientBoosting-Modelle in beiden Stufen konkurrenzfähig oder stark. Für einen ersten Orange-Vergleich sind sie deshalb eine pragmatische Referenz.

Conditional Features helfen besonders stark bei CLS. Dort kommt ein großer Teil des Signals aus `pid_prob`, `order_time`, `click_time`, `basket_time` und ähnlichen Verlaufssignalen. Bei REG liefern einige Conditional Features ebenfalls wichtiges Signal, vor allem `pid_prob`, `group12_order` und `group34_order`, aber der Effekt ist weniger eindeutig als bei CLS.

Kaltstart-Fälle bleiben schwierig. Wenn für eine PID wenig oder keine Historie vorliegt, können Verlaufssignale nur eingeschränkt helfen. Das Projekt geht damit konservativ um. In `config.py` sind dafür Default-Werte hinterlegt: 0.5 für die OOF-Kaufwahrscheinlichkeit und 1.0 für die Mengenannahme.

`group12` und `group34` sollten vorsichtig interpretiert werden. Sie sind technisch verfügbar und teilweise nützlich, aber ihre direkte Aussagekraft ist schwächer abgesichert als bei mehreren Conditional Features. Deshalb sollten die No-Group-Varianten als Sensitivitätscheck mitlaufen.

## 11. Nächste Schritte

1. Zuerst einen Orange-Rauchtest mit `train_sample` und den passenden `val`-Dateien aufsetzen.
2. Danach mehrere Modelle auf `cls_val.csv` und `reg_val.csv` vergleichen.
3. Anschließend die konservativen Varianten `CLS_FINAL_NO_GROUPS` und `REG_FINAL_NO_GROUP34` testen.
4. Den Test-Split erst ganz am Schluss für die letzte Kontrollmessung verwenden.

## Kurzfazit

Die Data Preparation ist weit genug abgeschlossen, um mit der Modellierung zu beginnen.

Das Projekt hat jetzt saubere Splits, dokumentierte Feature-Sets, Audits, Feature-Selection-Ergebnisse und Orange-Exportdateien mit Manifest.

Der nächste Schritt ist nicht mehr neues Feature Engineering, sondern ein sauberer Modellvergleich in Orange.
