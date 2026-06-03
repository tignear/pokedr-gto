# Devcontainer GPU Setup

The default devcontainer does not hard-code GPU device passthrough. Docker fails
to start when `--device` points at a path that does not exist on the Docker host,
so GPU flags must match the host runtime.

## AMD on Native Linux

Use this only when the Docker host has `/dev/kfd` and `/dev/dri`:

```jsonc
"runArgs": [
  "--device=/dev/kfd",
  "--device=/dev/dri",
  "--group-add=video",
  "--group-add=render",
  "--security-opt=seccomp=unconfined"
]
```

## AMD on WSL2

ROCm on WSL2 uses `/dev/dxg`, not `/dev/kfd` and `/dev/dri`. A WSL2 ROCm
container normally needs the DXCore/HSA libraries mounted from the WSL host:

```jsonc
"runArgs": [
  "--device=/dev/dxg"
],
"mounts": [
  {
    "source": "/usr/lib/wsl/lib/libdxcore.so",
    "target": "/usr/lib/libdxcore.so",
    "type": "bind"
  },
  {
    "source": "/opt/rocm/lib/libhsa-runtime64.so.1",
    "target": "/opt/rocm/lib/libhsa-runtime64.so.1",
    "type": "bind"
  }
]
```

Only add these after confirming the source paths exist on the Docker host.

## Vulkan on WSL2

WSL2 does not expose the AMD GPU as a normal Linux DRM render device. Native
Linux Vulkan drivers such as RADV expect `/dev/dri/renderD*`, so the practical
WSL2 Vulkan route is Mesa's experimental Microsoft dzn driver:

```text
wgpu/Vulkan -> Mesa dzn -> libd3d12/libdxcore -> /dev/dxg -> Windows driver
```

The more reliable route is to enable dzn in the WSL Ubuntu host's
`mesa-vulkan-drivers` package, then pass those host libraries into the
devcontainer. A helper script is provided for the WSL host:

```bash
scripts/build-wsl-mesa-dzn.sh --install
```

Run it from the WSL Ubuntu host, not inside the devcontainer. It:

- enables `deb-src` for Ubuntu package sources when needed;
- installs Mesa build dependencies;
- rebuilds Ubuntu's Mesa source package with
  `VULKAN_DRIVERS += microsoft-experimental`;
- adds `spirv2dxil` and `libspirv_to_dxil` to the
  `mesa-vulkan-drivers` package;
- optionally installs only the rebuilt `mesa-vulkan-drivers` package.

After the host package is installed, verify on the WSL host:

```bash
vulkaninfo --summary
```

The useful signal is a non-CPU physical device. If the host still only reports
`PHYSICAL_DEVICE_TYPE_CPU`, the devcontainer will not get a GPU through Vulkan.

If the WSL host reports a real GPU, rebuild the devcontainer and verify:

```bash
vulkaninfo --summary
cargo run --release -p pokedr-cli -- gpu-info
```

## NVIDIA

`--gpus all` is for NVIDIA or CDI-capable GPU runtimes. It is not the generic
AMD ROCm path.

```jsonc
"runArgs": [
  "--gpus",
  "all"
]
```

## Verification

After rebuilding the container:

```bash
cargo run --release -p pokedr-cli -- gpu-info
```

The expected result is an adapter name containing the target GPU, such as
`RX 9070 XT`. If it prints `no GPU adapter visible to wgpu`, the container still
does not expose a backend that `wgpu` can use.
