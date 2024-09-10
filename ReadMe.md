# Meine tägliche Skriptsammlung

Dieses Repository enthält eine Sammlung von Python-Skripten, die ich für verschiedene tägliche Aufgaben und Automatisierungen verwende. Jedes Skript ist für einen spezifischen Zweck konzipiert und kann unabhängig voneinander ausgeführt werden.

## Übersicht

- **Transkriptions- und Zusammenfassungstool** [\[ReadMe\]](audio_summary_helper/audio_summary_helper.md): Transkribiert Audio/Video-Dateien und erstellt Zusammenfassungen
- mehr füge ich hinzu, wenn ich happy mit der Codequalität bin

## Voraussetzungen

- Python 3.x
- pip (Python Package Installer)
- Verschiedene Python-Bibliotheken (spezifisch für jedes Skript)

## Installation

1. Klonen Sie dieses Repository:
   ```
   git clone https://github.com/yourusername/daily-scripts.git
   cd daily-scripts
   ```

2. Es wird empfohlen, eine virtuelle Umgebung zu erstellen:
   ```
   python -m venv venv
   source venv/bin/activate  # Auf Windows: venv\Scripts\activate
   ```

3. Installieren Sie die erforderlichen Pakete:
   ```
   pip install -r requirements.txt
   ```

## Verwendung

Jedes Skript in diesem Repository kann unabhängig ausgeführt werden. Navigieren Sie zum Verzeichnis des gewünschten Skripts und führen Sie es mit Python aus. Beispiel:

```
python transcription_tool.py <arguments>
```

Für detaillierte Informationen zur Verwendung eines bestimmten Skripts, lesen Sie bitte die zugehörige README-Datei im entsprechenden Unterverzeichnis oder führen Sie das Skript mit der `-h` oder `--help` Option aus.

## Struktur des Repositories

```
daily-scripts/
│
├── transcription_tool/
│   ├── transcription_tool.py
│   └── README.md
│
├── [andere_skript_verzeichnisse]/
│   ├── [skript_name].py
│   └── README.md
│
├── requirements.txt
└── README.md
```