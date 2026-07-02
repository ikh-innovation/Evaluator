FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY auth/ ./auth/

EXPOSE 8502

CMD ["streamlit", "run", "auth/app.py", \
     "--server.port=8502", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
