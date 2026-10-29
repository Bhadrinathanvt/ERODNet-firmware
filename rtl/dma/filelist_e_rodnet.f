#==========================================================
# e_rodnet DMA Controller File List
# Target: Yamuna (Shakti family) — AXI4 Full, 32-bit data, 32-bit addr
# Updated: 2024 - Added peripheral flow control support
#==========================================================

# Package
rtl/dma_e_rodnet_pkg.sv

# Interfaces
rtl/axi4_intf.sv
rtl/dma_streaming_intf.sv

# Token Assembly/Disassembly
rtl/token_assembler.sv
rtl/token_disassembler.sv

# Core Modules
rtl/dma_e_rodnet_csr.sv
rtl/dma_e_rodnet_engine.sv
rtl/dma_e_rodnet_top.sv

# Integration Wrapper
rtl/e_rodnet_dma_wrapper.sv

# Peripheral Request Manager
rtl/dma_periph_request.sv

# Testbench
tb/dma_e_rodnet_tb.sv
tb/dma_e_rodnet_periph_tb.sv