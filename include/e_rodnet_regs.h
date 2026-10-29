/*
 * e_rodnet_regs.h
 *
 * AXI4-Lite CSR register map for the e_rodnet radar object-detection IP
 * on the Yamuna Shakti SoC.
 *
 * Source of truth: IP/e_rodnet/rtl/wrapper/e_rodnet_axi_top.sv
 * (case blocks at lines ~404-514). All offsets are byte offsets from the
 * IP base address.
 *
 * Bus: AXI4-Lite, 64-bit internal data word.  The Yamuna slow-fabric
 * bridge exposes the slave as 32-bit, so every 64-bit CSR read splits
 * across `off+0` (low 32 bits) and `off+4` (high 32 bits). The status
 * bits in the wrapper live in the *upper* 32 bits of the 64-bit read,
 * so the driver must issue 32-bit accesses at `off+4` to see them
 * (e.g. E_RODNET_STATUS_RD = 0x504, not 0x500).
 *
 * BRESP/RRESP: 0b00 = OKAY, 0b10 = SLVERR.
 * Writes to undecoded offsets are silently ignored; undecoded reads return
 * 0xDEADBEEF on both 32-bit halves.
 *
 * Base address: slow-fabric slot at 0x0005_0000 (override at build time
 * with -DE_RODNET_BASE=0x....).
 */

#ifndef E_RODNET_REGS_H
#define E_RODNET_REGS_H

#include <stdint.h>

#ifndef E_RODNET_BASE
#define E_RODNET_BASE            0x00050000u
#endif

/* ------------------------------------------------------------------ */
/* CSR byte offsets (matches CSR_* localparams in e_rodnet_axi_top)    */
/* ------------------------------------------------------------------ */

/* Global Control (0x0000 - 0x003F)
 * NOTE: 0x000 writes only capture bit[0] into reg_enable. Reads of 0x000
 * show {soft_reset, 3'b0, enable} packed at bits [4:0]. To ACTIVATE
 * soft-reset you must write bit[0] of 0x004 — writing (1<<4) at 0x000
 * is silently ignored by the RTL write decoder. */
#define E_RODNET_GLOBAL_CTRL     0x0000u   /* W:[0] enable;  R:[0]en [4]srst  */
#define E_RODNET_SOFT_RESET      0x0004u   /* W:[0] soft_reset latch (0 to clear) */
#define E_RODNET_FRAME_CONFIG    0x0008u   /* [3:0] history_depth, [31:16] feature_h */
#define E_RODNET_FRAME_W         0x000Cu   /* [15:0] feature_w                */

/* Temporal Fusion (0x0040 - 0x007F) */
#define E_RODNET_TEMPORAL_MODE   0x0040u   /* [1:0] mode (0=None 1=Fusion 2=Agg 3=Both 4=ConvLSTM 5=TDC*) */
#define E_RODNET_TEMPORAL_BLEND  0x0048u   /* [DATA_WIDTH-1:0] signed blend  */

/* ConvFormer (0x0080 - 0x00BF) */
#define E_RODNET_CONV_CONFIG     0x0080u   /* [15:0] channels, [31:16] kernel */
#define E_RODNET_CONV_ACTIVATION 0x0084u   /* [2:0] act_sel (NEW 2026-04)     */

/* SSF / GFF placeholders (not currently decoded by software) */
#define E_RODNET_SSF_CONFIG      0x00C0u
#define E_RODNET_GFF_CONFIG      0x0100u

/* Detection Head (0x0140 - 0x017F) */
#define E_RODNET_DET_CONF_THRESH 0x0140u   /* [DATA_WIDTH-1:0] signed         */
#define E_RODNET_DET_IOU_THRESH  0x0148u   /* [DATA_WIDTH-1:0] signed         */
#define E_RODNET_DET_TOP_K       0x0150u   /* [7:0] top-k                     */

/* NMS / Top-K (0x0180 - 0x01BF)
 * NOTE: in the current RTL, 0x180 is a write-only alias that stores back
 * into reg_iou_threshold (same register 0x148 writes). 0x188 is the NMS
 * on/off switch — its [0] bit is reg_nms_enable. The old header treated
 * 0x188 as an IoU threshold; that was wrong. */
#define E_RODNET_NMS_IOU_ALIAS   0x0180u   /* [DATA_WIDTH-1:0] signed; aliases 0x148 */
#define E_RODNET_NMS_ENABLE      0x0188u   /* [0] nms_enable                  */

/* Sensor DMA Descriptors (0x0200 - 0x02FF) */
#define E_RODNET_DMA_SRC_ADDR    0x0200u
#define E_RODNET_DMA_DST_ADDR    0x0208u
#define E_RODNET_DMA_FRAME_SIZE  0x0210u
#define E_RODNET_DMA_FRAME_COUNT 0x0218u   /* [15:0]                          */
#define E_RODNET_DMA_CONTROL     0x0220u   /* [0] start, [1] abort, [2] loop  */

/* Weight Loading (0x0300 - 0x033F) — CSR AUTO-INC MODE                 */
#define E_RODNET_WEIGHT_ADDR     0x0300u   /* [31:0] start weight offset      */
#define E_RODNET_WEIGHT_COUNT    0x0304u   /* [31:0] #weights in streamed run */
#define E_RODNET_WEIGHT_DATA     0x0308u   /* [DATA_WIDTH-1:0] signed value   */
#define E_RODNET_WEIGHT_CTRL     0x030Cu   /* [0] write, [1] auto_inc,
                                            [7:4] layer_sel                  */

/* Weight Loading (0x0340 - 0x037F) — AXI MASTER DMA MODE               */
#define E_RODNET_WT_DMA_SRC      0x0340u   /* DDR source physical address     */
#define E_RODNET_WT_DMA_COUNT    0x0344u   /* #weights to pull                */
#define E_RODNET_WT_DMA_DST      0x0348u   /* destination offset in weight BR */
#define E_RODNET_WT_DMA_CTRL     0x034Cu   /* [0] start, [3:2] layer_sel      */

/* Weight Loader Status (0x0380, RO). Bits live in LOW 32-bits:
 *     [11:0]  reserved (0)
 *     [12]    busy
 *     [13]    done
 *     [14]    error
 *     [30:15] count_remain (16-bit)
 *     [31]    reserved (0)
 * W1C clear: write 1 to WT_STATUS[0] to clear done/error latches.       */
#define E_RODNET_WT_STATUS       0x0380u

/* Performance Counters (0x0400 - 0x04FF, RO) */
#define E_RODNET_PERF_CYCLES     0x0400u
#define E_RODNET_PERF_FRAMES     0x0408u
#define E_RODNET_PERF_THRUPUT    0x0410u

/* Status (0x0500) and Interrupts (0x0508..) */
#define E_RODNET_STATUS_LO       0x0500u   /* low 32 of 64-bit word (all 0) */
#define E_RODNET_STATUS          0x0504u   /* high 32: [0] done, [1] busy,
                                            [2] dma_error                  */
#define E_RODNET_IRQ_ENABLE      0x0508u   /* [31:0] irq enable mask          */
#define E_RODNET_IRQ_STATUS      0x0510u   /* [31:0] pending; W1C via write   */

/* ------------------------------------------------------------------ */
/* Bitfield masks                                                      */
/* ------------------------------------------------------------------ */

#define E_RODNET_GCTRL_ENABLE        (1u << 0)
/* Read-only bit in GLOBAL_CTRL reflecting the latched reg_soft_reset.
 * Do NOT write (1<<4) at 0x000 expecting a reset — write 0x004[0] instead. */
#define E_RODNET_GCTRL_SOFT_RESET_RD (1u << 4)
#define E_RODNET_SRST_ASSERT         (1u << 0)   /* write to 0x004 to latch */

#define E_RODNET_DMA_CTRL_START      (1u << 0)
#define E_RODNET_DMA_CTRL_ABORT      (1u << 1)
#define E_RODNET_DMA_CTRL_LOOP       (1u << 2)

/* Reads of E_RODNET_STATUS (0x0504): */
#define E_RODNET_STATUS_DONE         (1u << 0)
#define E_RODNET_STATUS_BUSY         (1u << 1)
#define E_RODNET_STATUS_DMA_ERROR    (1u << 2)

/* IRQ bit positions (reg_irq_status & reg_irq_enable).
 *     bit0 = frame_done   -> irq_frame_done pin
 *     bit1 = detection    -> irq_detection  pin
 *     bit2 = error        -> irq_error      pin
 * NOTE: the wrapper has an internal irq_weight_done_int but does NOT wire
 * it into reg_irq_status or any top-level output; bit[3] is therefore
 * inert for both enable and status. Weight DMA completion must be
 * polled via WT_STATUS.DONE (0x380) — see e_rodnet_wait_weight_dma(). */
#define E_RODNET_IRQ_FRAME_DONE      (1u << 0)
#define E_RODNET_IRQ_DETECTION       (1u << 1)
#define E_RODNET_IRQ_ERROR           (1u << 2)

/* WEIGHT_CTRL bit positions */
#define E_RODNET_WEIGHT_CTRL_WRITE       (1u << 0)
#define E_RODNET_WEIGHT_CTRL_AUTO_INC    (1u << 1)
#define E_RODNET_WEIGHT_CTRL_LAYER_SHIFT 4
#define E_RODNET_WEIGHT_CTRL_LAYER_MASK  (0xFu << 4)

/* WT_DMA_CTRL bit positions */
#define E_RODNET_WT_DMA_CTRL_START        (1u << 0)
#define E_RODNET_WT_DMA_CTRL_LAYER_SHIFT  2
#define E_RODNET_WT_DMA_CTRL_LAYER_MASK   (0xFu << 2)

/* WT_STATUS bits (read from 0x380, 32-bit low word) */
#define E_RODNET_WT_STATUS_BUSY            (1u << 12)
#define E_RODNET_WT_STATUS_DONE            (1u << 13)
#define E_RODNET_WT_STATUS_ERROR           (1u << 14)
#define E_RODNET_WT_STATUS_CLEAR           (1u << 0)   /* W-only clear     */
#define E_RODNET_WT_STATUS_REMAIN(v)       (((v) >> 15) & 0xFFFFu)

/* Temporal mode encoding */
#define E_RODNET_TMODE_NONE          0u
#define E_RODNET_TMODE_FUSION        1u
#define E_RODNET_TMODE_AGG           2u
#define E_RODNET_TMODE_BOTH          3u
#define E_RODNET_TMODE_CONVLSTM      4u
#define E_RODNET_TMODE_TDC           5u

/* Activation select (reg_conv_activation) */
#define E_RODNET_ACT_RELU            0u
#define E_RODNET_ACT_LEAKY_RELU      1u
#define E_RODNET_ACT_RELU6           2u
#define E_RODNET_ACT_SIGMOID         3u
#define E_RODNET_ACT_GELU            4u
#define E_RODNET_ACT_STAR_RELU       5u

/* ------------------------------------------------------------------ */
/* Detection output packed format (m_axi_out beat, 64-bit)             */
/*   { pad[5:0], class_id[1:0], conf[7:0], w_idx[15:0], h_idx[15:0] }  */
/* ------------------------------------------------------------------ */

#define E_RODNET_DET_H_IDX(v)        ((uint16_t)((v) & 0xFFFFu))
#define E_RODNET_DET_W_IDX(v)        ((uint16_t)(((v) >> 16) & 0xFFFFu))
#define E_RODNET_DET_CONF(v)         ((uint8_t) (((v) >> 32) & 0xFFu))
#define E_RODNET_DET_CLASS(v)        ((uint8_t) (((v) >> 40) & 0x3u))

typedef struct __attribute__((packed)) {
    uint16_t h_idx;
    uint16_t w_idx;
    uint8_t  conf;
    uint8_t  class_id;
    uint16_t _pad;
} e_rodnet_det_t;

/* ------------------------------------------------------------------ */
/* MMIO accessors (inline, no libc dependency)                         */
/* ------------------------------------------------------------------ */

static inline void e_rodnet_w32(uintptr_t base, uint32_t off, uint32_t val)
{
    *(volatile uint32_t *)(base + off) = val;
}

static inline uint32_t e_rodnet_r32(uintptr_t base, uint32_t off)
{
    return *(volatile uint32_t *)(base + off);
}

/* RISC-V fence to force MMIO ordering. */
static inline void e_rodnet_mmio_fence(void)
{
    __asm__ volatile ("fence iorw, iorw" ::: "memory");
}

#endif /* E_RODNET_REGS_H */
