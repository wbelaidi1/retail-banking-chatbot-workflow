FROM python:3.12-slim

WORKDIR /app

# mistralai-workflows >= 3.10 is required to read service-account tokens
RUN pip install --no-cache-dir 'mistralai-workflows[mistralai]>=3.10'

COPY src ./src
ENV PYTHONPATH=/app

# The image's CMD must start the worker
CMD ["python", "src/entrypoints/worker.py"]
