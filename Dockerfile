FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY requirements.txt requirements-ml.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt

COPY configs/ ./configs/
COPY src/ ./src/
COPY main.py ./
COPY scripts/ ./scripts/
COPY web/ ./web/

EXPOSE 5000

CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "5000"]
