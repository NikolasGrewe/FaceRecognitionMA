# FaceRecognitionMA

Empfohlene Python-Version: 3.9.7

Weitere benötigte Pakete: Tensorflow 2.9.1 bzw. Keras 2.9, Matplotlib, Tkinter, Numpy

Raspberry Pi Funktionen getestet auf Raspberry Pi 3 Model B+ 1GB
Benötigte Raspberry Pi OS Version: 64-Bit
Ebenfalls benötigt: Angeschlossene Kamera

## Anleitung
### Benutzung der Software
main.py soll mit Python 3.9.7 ausgeführt werden. Sobald sich die Konsole geöffnet hat, können Befehle eingegeben werden. 

- help: Zeigt verfügbare Befehle an. 
- train: Trainiert ein Netzwerk.  
- predict: Verarbeitet ein Bild mit einem Netzwerk. 
- photo: Funktioniert nur auf dem Raspberry Pi mit angeschlossener Kamera. Macht ein Bild und wertet dieses mit einem Netzwerk aus. 
- continue training: Fährt mit dem Training eines Netzwerks fort. Braucht einen Ordner gleicher Struktur. 
- evaluate: Evaluiert ein Netzwerk mit einem Evaluationsdatensatz. Schreibt die Werte in eine Textdatei in ./Tabels/
- quit: Schliesst das Programm. 

### Eigene Datensätze
Um die Netzwerke zu trainieren, muss ein Datensatz in ./pictures/unspecific/ bzw. ./pictures/specific/ platziert werden. 
Dieser muss den folgenden Aufbau haben:

- Unspecific: Der Name des Sets soll TrainingSet sein. Zwei Ordner sollen hier vertreten sein: "No Faces" und "Faces". Welche Dateien in welchen Ordner gehören ist selbsterklärend.
- Specific: Der Name des Sets soll TrainingSet2 sein. Für jede Person soll ein Ordner mit den der Person entsprechenden Bilder vertreten sein. 
