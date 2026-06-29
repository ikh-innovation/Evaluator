FROM alpine:3.19

RUN apk add --no-cache sqlite

CMD ["sh", "-c", \
     "mkdir -p /data/job_payloads && \
      [ -f /data/runs.sqlite ] || sqlite3 /data/runs.sqlite 'PRAGMA journal_mode=WAL;' && \
      echo 'Data directories ready'"]
