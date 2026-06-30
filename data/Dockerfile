FROM python:3.11-slim

CMD ["sh", "-c", \
     "mkdir -p /data/job_payloads && \
      ([ -f /data/runs.sqlite ] || python3 -c \"\
import sqlite3; \
c = sqlite3.connect('/data/runs.sqlite'); \
c.execute('PRAGMA journal_mode=WAL'); \
c.close(); \
print('SQLite initialized')\
\") && \
      echo 'Data directories ready'"]
