# Setup & Environment Guide

## Python Version
Python 3.10 or 3.11 recommended.

## Windows
```powershell
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## Mac/Linux
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## GPU (Optional)
```bash
pip install tensorflow[and-cuda]==2.15.0
```

## Common Errors
| Error | Fix |
|---|---|
| No module flask_socketio | pip install flask-socketio eventlet |
| OOM GPU | Set BATCH_SIZE=16 in preprocessing.py |
| cv2 not found | pip install opencv-python-headless |
| Port 5000 busy | Change port=5001 in app.py last line |
