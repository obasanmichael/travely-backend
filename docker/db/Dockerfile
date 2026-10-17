# postgis/postgis publishes amd64 only; build on the multi-arch official image
# so local dev works on Apple Silicon without emulation.
FROM postgres:17-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql-17-postgis-3 \
 && rm -rf /var/lib/apt/lists/*
