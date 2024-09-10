# Transkriptions- und Zusammenfassungstool

Dieses Python-Skript ermöglicht die Transkription von Audio- oder Videodateien mithilfe des Whisper-Modells und die anschließende Zusammenfassung des transkribierten Textes unter Verwendung eines lokalen Sprachmodells über Ollama.

## Funktionen

- Transkription von Audio- oder Videodateien mit verschiedenen Whisper-Modellgrößen
- Zusammenfassung des transkribierten Textes mit einem lokalen Sprachmodell
- Speicherung der Transkription und Zusammenfassung in einer Textdatei

## Voraussetzungen

- Python 3.x
- pip (Python Package Installer)
- [Ollama](https://ollama.com/)

## Installation

1. Klonen Sie das Repository oder laden Sie das Skript herunter.

2. Installieren Sie die erforderlichen Pakete:

   ```
   pip install -r requirements.txt
   ```

3. Stellen Sie sicher, dass Ollama auf Ihrem System installiert und konfiguriert ist und auf `http://localhost:11434` läuft.

## Verwendung

Führen Sie das Skript mit folgendem Befehl aus:

```
python script_name.py <file_path> [--model MODEL] [--output OUTPUT] [--summarize {true,false}]
```

### Argumente

- `file_path`: Pfad zur zu transkribierenden Audio- oder Videodatei (erforderlich)
- `--model`: Whisper-Modell zur Verwendung (optional, Standard: "large")
  - Verfügbare Optionen: "tiny", "base", "small", "medium", "large"
- `--output`: Ausgabedateipfad für die Transkription (optional)
- `--summarize`: Ob der transkribierte Text zusammengefasst werden soll (optional, Standard: "true")

### Beispiele

1. Transkribieren einer Datei mit Standardeinstellungen:
   ```
   python script_name.py path/to/your/file.mp3
   ```

2. Verwenden eines anderen Whisper-Modells:
   ```
   python script_name.py path/to/your/file.mp3 --model medium
   ```

3. Speichern der Ausgabe in einer bestimmten Datei:
   ```
   python script_name.py path/to/your/file.mp3 --output path/to/output.txt
   ```

4. Transkribieren ohne Zusammenfassung:
   ```
   python script_name.py path/to/your/file.mp3 --summarize false
   ```

## Ausgabe

Das Skript erzeugt eine Textdatei mit der Transkription und (falls aktiviert) der Zusammenfassung. Standardmäßig wird die Datei im Verzeichnis `transcriptions/` gespeichert, es sei denn, ein benutzerdefinierter Ausgabepfad wird angegeben.

## Hinweise

- Stellen Sie sicher, dass Sie über ausreichend Speicherplatz für die Whisper-Modelle verfügen.
- Die Verarbeitungszeit kann je nach Länge der Eingabedatei und der gewählten Modellgröße variieren.
- Für die Zusammenfassung muss Ollama auf Ihrem System installiert und konfiguriert sein.

## Fehlerbehebung

- Wenn Sie Probleme mit der Whisper-Installation haben, konsultieren Sie die offizielle Whisper-Dokumentation.
- Stellen Sie sicher, dass Ollama läuft und über `http://localhost:11434` erreichbar ist.

Bei weiteren Fragen oder Problemen erstellen Sie bitte ein Issue in diesem Repository.