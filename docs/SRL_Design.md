# Structured Reinforcement Learning (SRL) — Konzept & Design

Status: Konzeptphase, noch keine Implementierung. Dieses Dokument hält fest, was
bisher überlegt wurde, bevor Code geschrieben wird.

## Prototyp in einfachen Worten

**Was gleich bleibt:** Das bestehende System — das Netz, das Trips bewertet
(`ScoringMLP`), der Solver, der daraus eine gültige Zuweisung baut
(`CO_ScoreMaximization`), und die Trainings-Schleife, die Batch für Batch
durch eine Li&Lim-Instanz läuft. Nichts davon wird angefasst.

**Was neu ist — ein einziges Puzzleteil:** Bisher hat das Training beim
Lernen "gespickt": es kannte die optimale Lösung und hat das Netz darauf
trainiert, diese nachzuahmen (`ImitationHandler`). Der Prototyp ersetzt genau
diese eine Stelle durch etwas, das ohne Spicken auskommt: Statt "hier ist die
richtige Antwort, lern das" macht das System jetzt "probier mehrere leicht
unterschiedliche Varianten deiner eigenen Einschätzung aus, schau welche
davon zu den meisten bedienten Fahrgästen führt, und lern von der besten
(bzw. einer Mischung der besten)" — das übernimmt der neue
`SRLTargetBuilder` (Details siehe Phase 1 unten).

**Was wir damit testen:** Ob das System auf diese Weise überhaupt lernt, gute
Trips zu bevorzugen — ganz ohne die optimale Lösung je gesehen zu haben. Am
Ende jeder Instanz läuft einmal der "echte" Check (`StatsParser`, komplette
Routen-Simulation), um die tatsächliche Service Rate zu prüfen.

**Was der Prototyp bewusst NICHT macht:** kein Kritiker-Netzwerk, keine
Kombination mit dem Pruner-GNN, kein Blick über den aktuellen Batch hinaus in
die Zukunft.

**Was der Kritiker (Phase 2) später bringt:** Der Prototyp lernt nur "was ist
gut *in diesem einen Moment*". Er kann nicht sehen, ob eine Pooling-
Entscheidung jetzt ein Fahrzeug später in eine schlechte Position bringt —
weil er dafür in die Zukunft schauen müsste, und die kennt er beim Training
noch nicht. Ein Kritiker ist ein zweites, kleines Netz, das genau das lernt:
"wenn ich diese Aktion jetzt wähle, wie gut wird es *im weiteren Verlauf*
laufen" — er schätzt die Zukunft, statt sie zu ignorieren. Damit kann der
Prototyp dann auch vorausschauend planen (z.B. ein Fahrzeug lieber in einem
Gebiet lassen, in dem später mehr Fahrgäste auftauchen), statt nur den
aktuellen Moment zu optimieren. Der Haken: die Zukunft lässt sich nicht
direkt ablesen wie die Service Rate des aktuellen Batches (Phase 1) — sie
muss gelernt/geschätzt werden, und genau dafür braucht es den Kritiker.

## Ziel

Bisher trainiert die COAML-Pipeline (`coaml_pipeline.py`) den Actor per
Imitation Learning: `y_star` kommt aus der bekannten optimalen Fahrer-Route
(`ImitationHandler`), was eine optimale Lösung voraussetzt.

SRL (nach Bouvier et al., "Structured Reinforcement Learning") ersetzt dieses
`y_star` durch ein online berechnetes Ziel, das keine optimale Lösung braucht.
Startpunkt: Li&Lim-Benchmark (`inputs/li_lim/pdp_100/`), da dort keine
optimalen Lösungen für das eigentliche Trainingsziel nötig sein sollen.

## Mapping Paper → bestehender Code

| Paper-Begriff | Code | Datei |
|---|---|---|
| State s | Aktueller Dispatch-Batch (Requests + Fahrzeuge) | `feat_builder.py`, `request_graph_feature_builder.py` |
| Statistisches Modell φ_w | `ScoringMLP` | `model_simpleScoring.py` |
| Scores θ | Output von `ScoringMLP`: ein Skalar pro Trip-Kandidat + ein Skalar pro Reject-Aktion pro Fahrzeug | — |
| CO-Layer f(θ,s) | `CO_ScoreMaximization.solve_ilp` (max θᵀa unter RTV-Constraints) | `co_scoreMaximization.py` |
| Aktion a ∈ A(s) | Binäre ILP-Lösung (x_t, x_r, x_reject) → `AssignmentResult` | `assignment_result.py` |
| Fenchel-Young-Loss | `FenchelYoungLoss` | `loss_FYscoring.py` |
| Target y* (aktuell, IL) | One-Hot auf den Trip, der zur bekannten optimalen Route passt | `imitation_handler.py` (`ImitationHandler`) |
| Kritiker Q_ψ(s,a) | existiert noch nicht | — |

Separates, vorgelagertes Modell (nicht φ_w): `RequestGraphEdgeGNN`
(`request_graph_gnn.py`) scored Request-Request-Kanten und verkleinert A(s)
(Pruning), bevor das ILP läuft. Wird für den ersten SRL-Test bewusst NICHT
kombiniert — erst alle Trips nehmen, um zu prüfen, ob die SRL-Actor-Loop an
sich funktioniert.

## Konkreter Injection-Punkt im bestehenden Code

`COAMLPipeline.solve_iteration()` (`coaml_pipeline.py`) macht pro Batch:

1. Features bauen → `ScoringMLP` → θ
2. `y_star = self._build_y_star_from_imitation_scores(...)` (Zeile ~518) —
   ruft `ImitationHandler` auf, braucht externe optimale Lösung
3. `self.fy_loss(scores, y_star, oracle)` — Oracle ist `make_map_oracle` →
   letztlich `CO_ScoreMaximization.solve_ilp`

Rolling-Horizon-Logik (Batches über eine Episode/Instanz) existiert bereits
in `COAMLTrainingLoop` (`training_loop.py`) und muss nicht neu gebaut werden.

**Korrektur (Recherche 2026-07-24): Es ist NICHT nur eine Stelle, die
ersetzt werden muss.** `solve_iteration` löst das ILP intern zweimal — einmal
mit θ (→ `score_result`, die eigene Einschätzung des Netzes), einmal mit den
Imitation-Scores aus der bekannten Optimallösung (→ `optimal_result`). Welches
Ergebnis tatsächlich die Fahrzeuge bewegt und die Simulation in den nächsten
Batch überführt, hängt vom `mode`-Parameter ab
([coaml_pipeline.py:614-626](../rtv_solver/coaml_pipeline.py)):
- `mode="train"` → `result = optimal_result` — die Simulation läuft beim
  Trainieren **immer entlang der bekannten optimalen Route weiter**,
  unabhängig davon, was das Netz vorhersagt (Teacher-Forcing)
- `mode="eval"` → `result = score_result` — hier bestimmt die eigene
  Vorhersage des Netzes den weiteren Simulationsverlauf

`result` wird nicht nur geloggt, sondern bewegt tatsächlich die Fahrzeuge
(`vehicle.apply_trip_insertion`, Zeile ~673-689) und bestimmt so den
Zustand des nächsten Batches. Der bisherige `"train"`-Modus ist also reines
Teacher-Forcing: das System erlebt beim Training nie die Konsequenzen seiner
eigenen Entscheidungen. Für SRL ist das ein Problem — wir wollen ja gerade
lernen, was die eigenen (ggf. noch schlechten) Aktionen für Konsequenzen
haben; mit Teacher-Forcing gäbe es dafür kein Signal, weil die Simulation nie
vom bekannten optimalen Pfad abweicht.

**Drei koordinierte Stellen statt einer:**
1. Neuer Modus-Zweig in `solve_iteration` (~Zeile 619): `result =
   score_result` (wie `eval`) — Rollout folgt der eigenen Policy, nicht der
   Optimallösung
2. Loss-Berechnung: neue Methode (ruft `SRLTargetBuilder` statt
   `ImitationHandler`/`_compute_fy_loss_from_optimal_solution`)
3. `solve_pdptw`s Gradienten-Trigger (aktuell fest an `mode=="train"`
   gekoppelt, [coaml_pipeline.py:152](../rtv_solver/coaml_pipeline.py)) muss
   den neuen Modus mit auslösen

Präzedenzfall im Code: `_compute_fy_loss_from_default_ilp()`
(`coaml_pipeline.py`, Zeile ~727) baut `y_star` bereits aus einem on-the-fly
gelösten ILP statt aus einer externen Datei (kein Kritiker, aber ein Beispiel
für "y_star ohne externe optimale Lösung"). Lohnt sich als Referenz beim
Entwurf des SRL-Target-Builders.

## Phase 1 — ohne Kritiker (myopisch)

Idee: Die Service Rate einer Kandidaten-Aktion ist direkt aus der gelösten
ILP-Lösung ablesbar (`AssignmentResult.request_assignment` /
`unassigned_trip_count`) — kein gelernter Kritiker nötig, weil der "wahre"
Wert einer Aktion für den aktuellen Batch nicht geschätzt, sondern exakt
berechnet werden kann.

Ablauf, konkret (`SRLTargetBuilder`, pro Batch, gegeben θ ∈ ℝⁿ mit n =
trip_count + reject_count):

1. m Perturbationen ziehen: `η_k = θ + σ_b · ε_k`, `ε_k ~ N(0, I)`, k=1..m
   (kein Gradient nötig).
2. Für jedes η_k das CO-Layer lösen: `a_k = CO_ScoreMaximization.solve_ilp(
   feature_scores=η_k, ...)` → m verschiedene, aber jeweils zulässige
   binäre Aktionen.
3. Für jedes a_k die Service Rate r_k ablesen (aus dem jeweiligen
   `AssignmentResult`, siehe oben).
4. Softmax-Gewichte: `w_k = exp(r_k/τ) / Σ_j exp(r_j/τ)` (τ = Temperatur,
   Hyperparameter).
5. Zielvektor bilden: `ā = Σ_k w_k · a_k`. **`ā` ist nicht binär und muss
   keine zulässige Aktion sein** — dient nur als Trainingssignal, wird nie
   ausgeführt (explizit so im Paper vorgesehen).
6. `loss = FenchelYoungLoss(θ, ā, oracle)` — Modul bleibt unverändert. Macht
   intern eine **eigene, zweite** Perturbation (eigenes `sigma`/
   `num_samples`) zur Gradientenschätzung — unabhängig von Schritt 1–5, zwei
   getrennte Perturbations-Mechanismen (einer zum Bauen von ā, einer in der
   Loss selbst).
7. `loss.backward()` → Adam-Step auf `ScoringMLP`-Parameter — exakt die
   bestehende Trainings-Schleife in `COAMLPipeline`/`COAMLTrainingLoop`,
   nur Schritt 1–5 ersetzt `_build_y_star_from_imitation_scores`.

Einschränkung: rein myopisch, sieht keine Konsequenzen für spätere Batches
innerhalb derselben Episode (Rolling-Horizon-Effekt).

### Trial-Lauf für eine Instanz (z.B. lc101) — was dafür nötig ist

Recherche 2026-07-24: Die Pipeline lädt für eine Li&Lim-Instanz nicht die
rohe `inputs/li_lim/pdp_100/lc101.txt`, sondern
`solutions/li_lim/manifests/lc101.json` — eine aufbereitete JSON-Payload
(Depot/Requests/Fahrzeuge). Diese Datei enthält zusätzlich die vorberechneten
optimalen Fahrer-Routen (menschlich lesbar auch in
`solutions/li_lim/txt_files/lc101.txt`). Siehe
[run_opt_single_instance.py:96-99](../rtv_solver/pipeline/run_opt_single_instance.py):

```
payload = PayloadParser.load_input_data(input_path)              # inkl. optimaler Routen
cleared_payload = PayloadParser.clear_vehicle_manifests(payload)  # Routen entfernt
```

`clear_vehicle_manifests` entfernt die Routen wieder — genau der
Sanitizing-Schritt, den ein SRL-Trial ohnehin braucht. Für den Trial-Lauf
NICHT benötigt: `solutions/li_lim/txt_files/lc101.txt` und die
Routen-Inhalte aus `lc101.json` (höchstens am Ende als Vergleichswert, nie
als Trainingsinput).

Bausteine für den Trial (Status):
1. `lc101.json` laden + `clear_vehicle_manifests` — **vorhanden**
2. `COAMLPipeline.solve_pdptw(payload, mode=...)` — durchläuft Batch für
   Batch die Instanz, ruft intern `solve_iteration` auf — **vorhanden**
   (`coaml_pipeline.py:93`)
3. `ScoringMLP` — **vorhanden**, für einen reinen Mechanik-Test reicht ein
   untrainiertes/zufällig initialisiertes Netz
4. `CO_ScoreMaximization` als Oracle — **vorhanden**
5. Perturb-und-Bewerten-Schleife (m Perturbationen → CO-Layer lösen →
   Service Rate ablesen) — **einziger neuer Teil**, entspricht dem
   `SRLTargetBuilder`-Entwurf oben

Für einen minimalen ersten Test (nur "läuft die Mechanik für einen Batch
durch") bietet sich `run_opt_single_instance.py` als Vorbild an — ein
schlanker Einzel-Instanz-Runner ohne volle `COAMLTrainingLoop`-Infrastruktur.

## Phase 2 — mit Kritiker (später)

Nötig, sobald über mehrere Dispatch-Zyklen einer Episode trainiert wird und
die kumulierte Service Rate über die Episode zählt, nicht nur die des
aktuellen Batches — eine Pooling-Entscheidung jetzt kann ein Fahrzeug für den
nächsten Batch schlecht positionieren; das sieht Phase 1 nicht.

Kritiker Q_ψ(s,a):
- Input: Trip-Features (wie φ_w) + Flag pro Trip "in dieser Aktion gewählt"
- Aggregation nur über die gewählten Trips (gewichtete Summe/Pooling) →
  variable Trip-Anzahl pro Batch kein Problem
- Output: ein Skalar Q(s,a)
- Training per TD: `y_t = r_t + γ · Q(s_{t+1}, a_{t+1})`, r_t = Service Rate
  des aktuellen Batches (gleiche Definition wie in Phase 1)
- Zielnetzwerk (langsam aktualisiert) + ggf. Double-Critic gegen
  Overestimation-Bias

## Neue Bausteine für Phase 1

| Baustein | Typ | Status |
|---|---|---|
| Service-Rate-Funktion (`AssignmentResult` → Skalar) | Funktion, evtl. kein eigenes File | zu prüfen: existiert sowas schon in einem der `evaluate_*`/`parse_eval_*`-Skripte? |
| `SRLTargetBuilder` | neue Datei (Vorschlag: `rtv_solver/pipeline/srl_target_builder.py`) | Pendant zu `ImitationHandler`, ohne externe Lösung: perturbiert θ, löst CO-Layer, liest Service Rate, baut softmax-gewichtetes ā |
| Neue Methode in `COAMLPipeline` | kein neues File | Pendant zu `_build_y_star_from_imitation_scores`, ruft `SRLTargetBuilder` |
| Config-Flag `TARGET_TYPE` (`imitation` \| `srl_phase1`) | Config-Erweiterung | erlaubt Vergleich IL vs. SRL, ersetzt bestehenden Pfad nicht |
| Prüfen: erzwingt `COAMLTrainingLoop`/Config irgendwo zwingend `IMITATION_SOLUTION_FILE`? | Prüfung, kein neues File | erst bei Umsetzung klären |

Unverändert wiederverwendet: `ScoringMLP`, `CO_ScoreMaximization`,
`FenchelYoungLoss`, `feat_builder.py`.

## Li&Lim-Instanzen (`inputs/li_lim/pdp_100/`)

Format: Zeile 1 = Fahrzeuganzahl/Kapazität/Geschwindigkeit; Zeile 2 = Depot
mit Zeitfenster `[0, 1236]`; jede weitere Zeile = ein Knoten
(`id, x, y, demand, ready_time, due_time, service_time, pickup_sibling,
delivery_sibling`). Ein Request = ein Pickup-Delivery-Knotenpaar (demand
positiv/negativ, sibling-Spalten verlinken die Partner).

Depot-Zeitfenster `[0, 1236]` = Gesamt-Simulationshorizont → **eine Episode =
eine komplette Instanz** (z.B. `lc101.txt`), abgespult von t=0 bis t=1236.

`lc101`/`lc102`/`lc103` haben identische Knoten-Koordinaten, aber
unterschiedlich enge Zeitfenster (lc101 eng, lc102/lc103 offener) → natürliche
Schwierigkeitsstufung für späteres Training.

**Instanzgröße (verifiziert 2026-07-23):** Trotz Ordnername `pdp_100` hat
`lc101.txt` nur **53 Pickup-Delivery-Paare** (53 Requests), nicht 100 (`wc -l`
+ Vorzeichen-Zählung der `demand`-Spalte). Der Ordner enthält 56 Instanzdateien
(lc1xx, lc2xx, lr1xx, lr2xx, ...), alle in ähnlicher Größenordnung. Wichtig für
die Proxy-vs-echt-Frage unten: bei dieser Größe ist `StatsParser.evaluate()`
einmal pro kompletter Episode (Instanz) rechnerisch trivial billig.

Vorschlag für erste Tests: batch_interval=200, step_size=40 (wie in
bestehenden Shell-Skripten, z.B. `run_baseline_data_bi200_ss40.sh`) — noch
nicht final festgelegt.

## Offene Fragen

1. ~~Existiert bereits eine wiederverwendbare Service-Rate-Berechnung aus
   `AssignmentResult`?~~ **Geklärt (Recherche 2026-07-23):** Braucht kein
   eigenes File. `AssignmentResult` trägt `unassigned_trip_count` und
   `request_assignment` bereits mit (gesetzt in `co_base.py:72-133`, direkt
   dort wo die Aktion aus der ILP-Lösung gebaut wird) — Service Rate einer
   einzelnen Aktion ist eine Ein-Zeilen-Berechnung aus vorhandenen Feldern.
   Was tatsächlich existiert, ist `StatsParser.evaluate()`
   (`rtv_solver/handlers/stats_parser.py`), genutzt in `training_loop.py` für
   die Val-Service-Rate. Diese ist aber **ungeeignet für den
   SRL-Target-Builder**: sie simuliert pro Fahrer die komplette Route
   (Wartezeiten, Umwege, TW-Verletzungen) über ein **komplettes gelöstes
   Payload** — zu teuer, um sie m-mal pro Trainingsschritt für perturbierte
   Kandidaten aufzurufen. Außerdem zählt sie "serviced" anders: nur wenn
   Pickup **und** Dropoff im simulierten Manifest vollständig auftauchen
   (`stats_parser.py:515-516`), nicht "wurde in diesem Batch zugewiesen".
   → **Aufgelöst (2026-07-23):** Phase 1 nutzt zwangsläufig die günstige
   Batch-Proxy-Metrik (assigned/rejected in diesem einen ILP-Solve) *während*
   des Trainings (pro Perturbation, oft pro Iteration — muss billig sein).
   Ob das mit der "echten" Service Rate übereinstimmt, muss nicht theoretisch
   bewiesen werden: Instanzen sind klein (~53 Requests), also ist
   `StatsParser.evaluate()` einmal pro kompletter Episode (Instanz) trivial
   billig — genau das Muster, das `training_loop.py` für die IL-Val-Service-
   Rate schon nutzt (einmal pro Epoche, nicht pro Iteration). Aufteilung also:
   Proxy-Metrik zum Trainieren, echte `StatsParser`-Metrik zum Validieren/
   Auswerten pro Instanz — Korrelation wird empirisch pro Instanz beobachtet,
   nicht angenommen.
2. Erzwingt `COAMLTrainingLoop` irgendwo zwingend eine
   `IMITATION_SOLUTION_FILE`, die für SRL-Phase-1-Runs entfernt werden müsste?
3. Decision-Epoch-Intervall für erste Tests: bi=200/ss=40 bestätigen oder
   anders?
4. Reward-Definition präzisieren: `served / total` vs. `-unassigned_trip_count`
   vs. etwas anderes? (Jetzt zusätzlich zu klären: Batch-Proxy vs. später
   ggf. gegen `StatsParser`-Ergebnis validieren.)
5. Wie viele Perturbationen m und welches σ_b für den SRL-Target-Builder als
   Startwert?

## Nächste Schritte (Diskussion, keine Implementierung)

- Offene Fragen 1–2 klären (Code-Recherche, kein Schreiben).
- Reward-Definition (Frage 4) konkret festlegen.
- Erst danach: `SRLTargetBuilder` entwerfen (Interface, nicht Code).
