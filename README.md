# ERODNet hardware and firmware

Synthesizable SystemVerilog for the `e_rodnet` radar object-detection accelerator, plus the bare-metal C driver (AXI4-Lite CSR block), a target wrapper and a test.

- `rtl/core/`: the accelerator (FFT, beamforming, ConvFormer blocks, normalisation, activation, pooling, detection, CSR and AXI interfaces, lookup tables as `.mem`)
- `rtl/dma/`: the DMA controller and wrapper that feed it (`filelist_e_rodnet.f` lists the compile order; its paths are relative to the original project layout)

- `drivers/e_rodnet.c`, `include/e_rodnet.h`, `include/e_rodnet_regs.h`: driver API and register map
- `drivers/e_rodnet_target.c`, `include/e_rodnet_target.h`: target/platform wrapper
- `tests/test_erodnet.c`: driver test
- `E_RODNET_DRIVER.md`: driver documentation

The model side lives in [ERODNet](https://github.com/Bhadrinathanvt/ERODNet).

## Dependencies (not included)

The sources include `platform.h`, `log.h`, `utils.h`, `uart.h` and `plic_driver.h` from the Shakti SDK. Build inside a Shakti SDK project; those files are GPL-licensed IIT Madras code and are not redistributed here.
