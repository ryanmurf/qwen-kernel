# qk-server: the qwen-kernel Vulkan engine + safe-Rust HTTP server.
# Runtime only — binaries are built on the host (cargo + cmake) and copied in.
FROM ubuntu:26.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        libvulkan1 mesa-vulkan-drivers libstdc++6 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY server   /app/server
COPY libqk.so /app/libqk.so
COPY shaders  /app/shaders
ENV QK_SHADER_DIR=/app/shaders
EXPOSE 8080
ENTRYPOINT ["/app/server"]
