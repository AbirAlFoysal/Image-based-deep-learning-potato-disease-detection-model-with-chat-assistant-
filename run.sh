#!/bin/bash

if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    python -m venv env

    echo "Installing requirements..."
    source env/Scripts/activate
    pip install -r requirements.txt
    deactivate
fi

# Activate virtual environment
source env/Scripts/activate

# Start leaf API in background
cd detection_service || { echo "detection_service not found"; exit 1; }
python leaf_api_app.py &

# Start Django server
cd ..
python manage.py runserver &

# Optional: keep script alive (e.g., for logs)
# If you don't want to wait, remove the next line
wait