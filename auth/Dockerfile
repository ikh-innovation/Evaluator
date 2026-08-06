FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY .streamlit/ ./.streamlit/
COPY backend/ ./backend/
# The login page shares the frontend's design system (and its logo) so both
# apps look like one product. Only those two files are needed here.
COPY frontend/theme.py frontend/ikh_logo.png ./frontend/
COPY auth/ ./auth/

EXPOSE 8502

CMD ["streamlit", "run", "auth/app.py", \
     "--server.port=8502", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
