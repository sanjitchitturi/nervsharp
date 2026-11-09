FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN CMAKE_ARGS="-DGGML_CUDA=OFF" pip install --no-cache-dir '.[inference,privacy]'

RUN mkdir -p models

VOLUME ["/app/models"]

CMD ["python", "-c", "import nervsharp; print(f'nervsharp {nervsharp.__version__} ready')"]

