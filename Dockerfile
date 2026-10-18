# AGBE endpoint. The model itself runs in a llama.cpp server container (see docker-compose.yml);
# this image holds the guard, the retriever and the API. Not yet built in CI: docker is not installed on the dev machine.
FROM python:3.12-slim
WORKDIR /app
COPY serve/requirements.txt serve/requirements.txt
RUN pip install --no-cache-dir -r serve/requirements.txt
COPY serve serve
COPY eval/claim_check.py eval/claim_check.py
COPY tools/extract_claims.py tools/extract_claims.py
ENV AGBE_PASSAGES=/data/passages.jsonl AGBE_BACKEND_URL=http://llama:8080
EXPOSE 8080
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8080"]
