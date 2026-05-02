FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml README.md requirements.txt ./
COPY configs ./configs
COPY src ./src
COPY tests ./tests

RUN pip install --no-cache-dir -e ".[paper]"

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "unittest", "discover", "-s", "tests"]

