FROM python:3.11-slim

# System deps for pyarrow / h5py wheels + jupyter
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mnt

# Install the package in editable mode plus dev + notebook extras.
# The repo is bind-mounted at /mnt at runtime, so copy only the metadata
# needed to resolve dependencies at build time; the rest comes from the mount.
COPY pyproject.toml /mnt/pyproject.toml
COPY code /mnt/code

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[dev]" \
    && pip install --no-cache-dir jupyter

ENTRYPOINT ["/usr/bin/env"]
CMD ["bash"]
