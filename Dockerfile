FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
COPY --chown=user app ./app
COPY --chown=user data/kb ./data/kb
USER user

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','7860')+'/api/health')" || exit 1

# OPENROUTER_API_KEY must be supplied at runtime (never baked into the image)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
