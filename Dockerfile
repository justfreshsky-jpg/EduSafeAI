FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser
ENV PORT 8080
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:$PORT app:app"]
