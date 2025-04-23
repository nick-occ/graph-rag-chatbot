# chatbot_api/Dockerfile

FROM python:3.12-slim

WORKDIR /app

COPY ./pyproject.toml ./README.md /app/

RUN pip install poetry

COPY ./src/ /app/src/

RUN pip install .

RUN chmod +x /app/src/entrypoint.sh

EXPOSE 8000

CMD ["sh", "/app/src/entrypoint.sh"]
