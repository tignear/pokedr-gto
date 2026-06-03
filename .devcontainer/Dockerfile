FROM mcr.microsoft.com/devcontainers/rust:2-1-trixie

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libvulkan1 \
        libvulkan-dev \
        mesa-vulkan-drivers \
        pkg-config \
        vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/share/vulkan/icd.d \
    && printf '%s\n' \
        '{' \
        '  "file_format_version": "1.0.0",' \
        '  "ICD": {' \
        '    "library_path": "/usr/lib/x86_64-linux-gnu/libvulkan_dzn.so",' \
        '    "api_version": "1.2.318"' \
        '  }' \
        '}' \
        > /usr/share/vulkan/icd.d/dzn_icd.x86_64.json

ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/dzn_icd.x86_64.json
ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu
