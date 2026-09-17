# Paper-trading harnesses. READ-ONLY market data, fake money, no order path.
FROM python:3.12-slim

WORKDIR /app
COPY requirements-harness.txt .
RUN pip install --no-cache-dir -r requirements-harness.txt

COPY src/ ./src/
COPY models/ ./models/
COPY run_paper.py run_paper_mlb.py run_harnesses.py ./

# Ledgers and observation logs belong on a mounted volume, or a redeploy erases the measurement.
ENV PAPER_DATA_DIR=/data
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "run_harnesses.py"]
