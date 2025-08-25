FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH /app

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --deploy --ignore-pipfile --system

COPY data/ ./data/
COPY knowledge_base_assistant ./knowledge_base_assistant

EXPOSE 8000

CMD gunicorn --bind 0.0.0.0:8000 knowledge_base_assistant.app:app