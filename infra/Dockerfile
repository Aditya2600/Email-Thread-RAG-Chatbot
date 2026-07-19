FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr libgl1 antiword \
    && rm -rf /var/lib/apt/lists/*

# pyproject first, with just enough package for an editable install to resolve:
# deps land in their own layer and stay cached when only source changes. The
# later COPY fills in the real source over the editable link.
COPY pyproject.toml /app/
RUN mkdir -p /app/email_thread_rag && touch /app/email_thread_rag/__init__.py
RUN pip install --no-cache-dir -e '.[models,serve,postgres]'

COPY . /app

EXPOSE 8000
