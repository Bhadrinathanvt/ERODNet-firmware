# ERODNet firmware

Bare-metal C driver for the `e_rodnet` radar object-detection IP (AXI4-Lite CSR block), with a target wrapper and a test.

- `drivers/e_rodnet.c`, `include/e_rodnet.h`, `include/e_rodnet_regs.h`: driver API and register map
- `drivers/e_rodnet_target.c`, `include/e_rodnet_target.h`: target/platform wrapper
- `tests/test_erodnet.c`: driver test
- `E_RODNET_DRIVER.md`: driver documentation

The model side lives in [ERODNet](https://github.com/Bhadrinathanvt/ERODNet).

## Dependencies (not included)

The sources include `platform.h`, `log.h`, `utils.h`, `uart.h` and `plic_driver.h` from the Shakti SDK. Build inside a Shakti SDK project; those files are GPL-licensed IIT Madras code and are not redistributed here.
