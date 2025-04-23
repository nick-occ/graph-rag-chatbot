FROM python:3.12-slim

WORKDIR /app

COPY ./pyproject.toml ./README.md /app/

RUN pip install poetry

COPY ./src/ /app/src/

RUN pip install .

COPY ./entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["sh", "entrypoint.sh"]