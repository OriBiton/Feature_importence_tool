#!/bin/bash

echo "Running collectstatic..."
python manage.py collectstatic --noinput

echo "Starting Gunicorn..."
gunicorn myproject.wsgi --bind 0.0.0.0:$PORT
