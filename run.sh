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

# Start Django server
python manage.py runserver &

# Optional: keep script alive (e.g., for logs)
# If you don't want to wait, remove the next line
wait