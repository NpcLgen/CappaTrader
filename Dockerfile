FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY cappatrader ./cappatrader
RUN pip install --no-cache-dir .

COPY config ./config

RUN useradd --create-home --uid 1000 trader && mkdir -p state logs data reports && chown -R trader /app
USER trader

CMD ["python", "-m", "cappatrader", "paper"]
