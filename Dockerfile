FROM python:3.12-slim

WORKDIR /app

# Install faultmap with user-facing extras (local embeddings + rich terminal output)
RUN pip install --no-cache-dir "faultmap[local,rich]"

# Copy example scripts
COPY examples/ ./examples/

CMD ["python", "-c", "import faultmap; print('faultmap', faultmap.__version__)"]
