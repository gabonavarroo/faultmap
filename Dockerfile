FROM python:3.12-slim

WORKDIR /app

# Install the local project with lightweight runtime extras by default.
RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml README.md LICENSE ./
COPY faultmap/ ./faultmap/
RUN pip install --no-cache-dir ".[rich]"

# Copy example scripts
COPY examples/ ./examples/

CMD ["python", "-c", "import faultmap; print('faultmap', faultmap.__version__)"]
