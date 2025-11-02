#!/bin/bash

# Start script for Django Genetic Algorithm Feature Selection
# This script sets up the environment and starts the Gunicorn server

echo "Starting Django Genetic Algorithm Feature Selection Server..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run Django migrations
echo "Running database migrations..."
python manage.py migrate

# Collect static files for production
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Start Gunicorn server
echo "Starting Gunicorn server..."
gunicorn feature_selection.wsgi:application --config gunicorn.conf.py

echo "Server started successfully!"
