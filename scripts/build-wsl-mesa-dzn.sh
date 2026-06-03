#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Build Ubuntu's mesa-vulkan-drivers package with WSL2 dzn Vulkan enabled.

Run this on the WSL Ubuntu host, not inside the devcontainer.

Usage:
  scripts/build-wsl-mesa-dzn.sh [--install] [--workdir DIR]

Options:
  --install      Install the rebuilt mesa-vulkan-drivers package after build.
  --workdir DIR  Build directory. Default: $HOME/mesa-wsl-dzn-build
USAGE
}

install_after_build=0
workdir="${HOME}/mesa-wsl-dzn-build"
icd_json="/usr/share/vulkan/icd.d/dzn_icd.x86_64.json"

repair_dzn_icd_json() {
    if [[ -d "$icd_json" ]]; then
        echo "removing broken ICD directory: $icd_json"
        sudo rm -rf "$icd_json"
    fi

    sudo mkdir -p "$(dirname "$icd_json")"
    printf '%s\n' \
        '{' \
        '  "file_format_version": "1.0.0",' \
        '  "ICD": {' \
        '    "library_path": "/usr/lib/x86_64-linux-gnu/libvulkan_dzn.so",' \
        '    "api_version": "1.2.318"' \
        '  }' \
        '}' \
        | sudo tee "$icd_json" >/dev/null
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install)
            install_after_build=1
            shift
            ;;
        --workdir)
            workdir="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -e /dev/dxg ]]; then
    echo "error: /dev/dxg is missing. Run this inside the WSL2 Ubuntu host." >&2
    exit 1
fi

if [[ ! -f /usr/lib/wsl/lib/libdxcore.so || ! -f /usr/lib/wsl/lib/libd3d12.so ]]; then
    echo "error: WSL D3D12 libraries are missing under /usr/lib/wsl/lib." >&2
    exit 1
fi

if ! command -v lsb_release >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y lsb-release
fi

codename="$(lsb_release -sc)"
if [[ "$codename" != "noble" ]]; then
    echo "warning: tested for Ubuntu 24.04 noble; current codename is ${codename}." >&2
fi

sources_file="/etc/apt/sources.list.d/ubuntu.sources"
if [[ -f "$sources_file" ]] && ! grep -q '^Types: .*deb-src' "$sources_file"; then
    echo "enabling deb-src in ${sources_file}"
    sudo cp "$sources_file" "${sources_file}.bak.$(date +%Y%m%d%H%M%S)"
    sudo sed -i -E 's/^Types: deb$/Types: deb deb-src/' "$sources_file"
fi

sudo apt-get update
sudo apt-get install -y build-essential devscripts dpkg-dev equivs vulkan-tools
sudo apt-get build-dep -y mesa

mkdir -p "$workdir"
cd "$workdir"

if [[ ! -d mesa-* ]]; then
    apt-get source mesa
fi

mesa_dir="$(find "$workdir" -maxdepth 1 -type d -name 'mesa-*' | sort | tail -n 1)"
if [[ -z "$mesa_dir" ]]; then
    echo "error: mesa source directory was not created." >&2
    exit 1
fi

cd "$mesa_dir"

if ! grep -q 'microsoft-experimental' debian/rules; then
    sed -i '/^VULKAN_DRIVERS[[:space:]]*=/a VULKAN_DRIVERS += microsoft-experimental' debian/rules
fi

install_file="debian/mesa-vulkan-drivers.install"
for line in \
    'usr/bin/spirv2dxil' \
    'usr/lib/*/libspirv_to_dxil.a' \
    'usr/lib/*/libspirv_to_dxil.so'
do
    grep -qxF "$line" "$install_file" || echo "$line" >> "$install_file"
done

if ! dpkg-parsechangelog -SVersion | grep -q 'wsl'; then
    DEBEMAIL="${DEBEMAIL:-local@wsl}" \
    DEBFULLNAME="${DEBFULLNAME:-local}" \
    dch --local wsl 'Enable WSL2 microsoft-experimental Vulkan driver'
fi

DEB_BUILD_OPTIONS="${DEB_BUILD_OPTIONS:-nocheck}" debuild -uc -us -b

deb="$(find "$workdir" -maxdepth 1 -type f -name 'mesa-vulkan-drivers_*wsl*_amd64.deb' | sort | tail -n 1)"
if [[ -z "$deb" ]]; then
    deb="$(find "$workdir" -maxdepth 1 -type f -name 'mesa-vulkan-drivers_*_amd64.deb' | sort | tail -n 1)"
fi

if [[ -z "$deb" ]]; then
    echo "error: rebuilt mesa-vulkan-drivers package not found in ${workdir}." >&2
    exit 1
fi

echo "built: $deb"

if [[ "$install_after_build" -eq 1 ]]; then
    sudo apt-get install -y "$deb"
    repair_dzn_icd_json
    echo "installed rebuilt mesa-vulkan-drivers."
    echo "checking Vulkan devices..."
    vulkaninfo --summary
else
    echo "install with:"
    echo "  sudo apt-get install -y '$deb'"
fi
