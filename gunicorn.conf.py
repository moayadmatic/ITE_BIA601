# Gunicorn configuration file for Django Genetic Algorithm Feature Selection

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:" + str(os.environ.get("PORT", 8000))
backlog = 2048

# Worker processes
workers = 1  # Use single worker for free tier
worker_class = "sync"
worker_connections = 1000
timeout = 120  # Increased timeout
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "genetic_algorithm_app"

# Django WSGI module
wsgi_module = "feature_selection.wsgi:application"

# Preload application for better performance
preload_app = True

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# SSL (uncomment if using HTTPS)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

# Graceful timeout for worker shutdown
graceful_timeout = 30

# Enable stats if needed
# statsd_host = "localhost:8125"

def when_ready(server):
    server.log.info("Genetic Algorithm Feature Selection server is ready. Listening on: %s", server.address)

def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_worker_init(worker):
    worker.log.info("Worker initialized (pid: %s)", worker.pid)

def worker_abort(worker):
    worker.log.info("Worker aborted (pid: %s)", worker.pid)
