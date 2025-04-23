# chatbot_api/Dockerfile

FROM python:3.12-slim

WORKDIR /app

COPY ./pyproject.toml ./README.md /app/

COPY ./src/chatbot_frontend/main.py /app/chatbot_frontend/main.py

RUN pip install poetry

COPY ./src/ /app/src/

RUN pip install .

RUN chmod +x /app/src/entrypoint.sh

CMD ["sh", "/app/src/entrypoint.sh"]
