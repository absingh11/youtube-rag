FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8501 8000
# Default: the chat UI. For the API instead:
#   docker run -p 8000:8000 --env-file .env yt-rag uvicorn yt_rag.api:app --host 0.0.0.0
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
