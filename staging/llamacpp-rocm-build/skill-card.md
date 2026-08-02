# Skill Card

## Description
Update and compile llama.cpp from source with ROCm/HIP support for AMD RDNA GPUs (RX 7900 XTX etc.). Detects the source checkout state, pulls the latest upstream, rebuilds with inherited cmake options (HIP backend, gfx1100 target, ROCWMMA off on RDNA3), and verifies the binaries see the GPU and generate output.

## Owner
AMD

## License
MIT
