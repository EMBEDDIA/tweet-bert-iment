#!/bin/sh
exec /sbin/tini -- venv/bin/gunicorn --bind=0.0.0.0:8000 "--workers=$WORKERS" "--timeout=$TIMEOUT" "--worker-class=$WORKER_CLASS" --worker-tmp-dir=/dev/shm "$@" elg_service:app
