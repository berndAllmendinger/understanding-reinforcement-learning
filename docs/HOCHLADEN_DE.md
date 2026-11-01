# Das Begleit-Repository auf GitHub hochladen

Das Repository ist lokal vorbereitet. Es wurde noch nicht online veröffentlicht.

## 1. Neues Repository auf GitHub anlegen

Melde dich bei GitHub als `berndAllmendinger` an. Wähle **New repository** und verwende den Namen:

`understanding-reinforcement-learning`

Vorgeschlagene Beschreibung:

> Step-by-step Python notebooks for Nim and Tic-Tac-Toe. Monte Carlo, SARSA, Q-learning, REINFORCE and PPO. Companion code for Understanding Reinforcement Learning by Bernd Allmendinger.

Wähle **Public**, damit Leser die Notebooks direkt öffnen können. Lasse die Optionen für eine zusätzliche README, .gitignore und Lizenz ausgeschaltet: Diese Dateien liegen bereits vor. Erstelle das Repository. Es wird eine leere Startseite angezeigt.

## 2. Die vorbereiteten Dateien hochladen

Wähle auf der leeren Startseite **uploading an existing file**. Alternativ findest du später **Add file → Upload files**.

Öffne auf deinem Computer den Ordner `github/understanding-reinforcement-learning` im Buchprojekt. Ziehe **dessen Inhalt** in die Upload-Fläche: README, LICENSE, ASSET_RIGHTS, requirements, Titelbild sowie die Unterordner `notebooks`, `docs` und `tests`. Nimm auch `.gitignore` mit, falls ausgeblendete Dateien sichtbar sind. Ein eventuell vorhandener lokaler `.git`-Ordner gehört nicht in den Browser-Upload.

Wichtig: Nicht den übergeordneten Buchordner und nicht die ZIP-Datei als einzelne Datei hochladen. Nach dem Upload müssen `README.md` und `book-cover.png` direkt auf der obersten Repository-Ebene liegen. Die beiden Notebooks gehören in `notebooks/`. Die Ordnerstruktur muss erhalten bleiben.

Trage als Beschreibung zum Beispiel `Add book companion notebooks` ein und bestätige den Commit auf `main`. GitHub erlaubt Datei- und Ordner-Uploads im Browser; [offizielle Anleitung](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository).

## 3. Colab-Links prüfen

Die vorbereiteten Buttons zeigen auf dein Konto `berndAllmendinger`, den Repository-Namen `understanding-reinforcement-learning` und den Branch `main`.

Öffne in der README beide **Open in Colab**-Buttons. In Colab: **Runtime → Run all**. Der Spielmodus ist zunächst ausgeschaltet. Am Ende `PLAY = True` setzen und diese Zelle ausführen. Beim Nim-Spiel lässt sich auch einstellen, wer beginnt; beim Tic-Tac-Toe spielt der trainierte Agent X und der Leser O.

Wenn du einen anderen Repository-Namen oder Branch wählst, passe die entsprechenden Links in README und beiden Notebooks an. Die relative Cover-Verknüpfung funktioniert unverändert, wenn die Ordnerstruktur stimmt.

## Schon vor dem GitHub-Upload ausprobieren

Öffne [Google Colab](https://colab.research.google.com/), wähle **File → Upload notebook** und lade eines der beiden `.ipynb`-Notebooks hoch. Es ist vollständig eigenständig. Du brauchst weder die übrigen Dateien noch das Buch-PDF für die Ausführung.

## Was gehört zum Buch, was zum Code?

Die README nennt Titel, Untertitel und Autor. ISBN und Kauf-Link bleiben bis zur Veröffentlichung offen. Die MIT-Lizenz gilt für den Begleitcode und seine Erklärungen. Titelbild und Buchtext sind ausdrücklich ausgenommen. Diese Trennung entspricht deinem Sports-Betting-Repository; die MIT-Lizenz ist hier vollständig ausgeschrieben.

## Spätere Änderungen

Die Notebooks sind die maßgebliche, direkt bearbeitbare Quelle. Kleine Änderungen lassen sich über Colab speichern und anschließend auf GitHub hochladen. Bei Änderungen an Formeln, Code oder Regeln immer das ganze Notebook neu ausführen und die Ausgaben aktualisieren. Bei größeren Änderungen empfiehlt sich GitHub Desktop; für die erste Veröffentlichung reicht der Browser.
