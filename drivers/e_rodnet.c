/***************************************************************************
 * Project        : Yamuna Shakti SoC - e_rodnet radar detection driver
 * File           : e_rodnet.c
 * Brief          : Bare-metal C driver for the e_rodnet CNN accelerator IP
 *                  (AXI4-Lite CSR + AXI4 master DMA) on the Yamuna C-Class
 *                  SoC. Mirrors the mcp2515 / awr1843aop driver style
 *                  (add_ons/drivers/*.c) and uses the standard Shakti PLIC
 *                  driver for IRQ dispatch.
 *
 * Dropped in by create_shakti_project.sh via `cp -r add_ons/* $PROJECT/`.
 ***************************************************************************/

#include <stdint.h>
#include "e_rodnet.h"
#include "e_rodnet_regs.h"
#include "platform.h"
#include "plic_driver.h"
#include "log.h"

/* Single active handle so shared ISR callbacks can look up state
 * without needing closure context from plic_driver's isr_table. */
static e_rodnet_handle_t *s_active_handle;

/* ------------------------------------------------------------------------- */
/* rdcycle timebase (RV32 - 64-bit count read as two 32-bit halves)          */
/* ------------------------------------------------------------------------- */
static inline uint64_t rv_rdcycle64(void)
{
    uint32_t lo, hi, hi2;
    do {
        __asm__ volatile ("rdcycleh %0" : "=r"(hi));
        __asm__ volatile ("rdcycle  %0" : "=r"(lo));
        __asm__ volatile ("rdcycleh %0" : "=r"(hi2));
    } while (hi != hi2);
    return ((uint64_t)hi << 32) | lo;
}

/* ------------------------------------------------------------------------- */
/* Per-source ISR trampolines installed into isr_table[]                     */
/* ------------------------------------------------------------------------- */
static void e_rodnet_isr_frame(uint32_t src)
{
    (void)src;
    if (s_active_handle) e_rodnet_isr(s_active_handle);
}
static void e_rodnet_isr_detect(uint32_t src)
{
    (void)src;
    if (s_active_handle) e_rodnet_isr(s_active_handle);
}
static void e_rodnet_isr_error(uint32_t src)
{
    (void)src;
    if (s_active_handle) e_rodnet_isr(s_active_handle);
}

/* ------------------------------------------------------------------------- */
/* Lifecycle                                                                  */
/* ------------------------------------------------------------------------- */

e_rodnet_status_t e_rodnet_init(e_rodnet_handle_t *h, uintptr_t base)
{
    if (!h) return E_RODNET_EINVAL;

    h->base  = base ? base : (uintptr_t)E_RODNET_BASE;
    h->flags = 0;

    /* Default PLIC source ids - chosen after the SSPI slots (35) defined
     * in platform.h. Override these before e_rodnet_irq_attach() if the
     * SoC stitch-up uses different numbers. */
    h->plic_irq_frame  = PLIC_INTERRUPT_36;
    h->plic_irq_detect = PLIC_INTERRUPT_37;
    h->plic_irq_error  = PLIC_INTERRUPT_38;

    /* Soft reset: assert via 0x004[0], deassert, then drop GLOBAL_CTRL.
     * The write decoder at 0x000 only captures bit[0] into reg_enable; the
     * soft-reset latch lives behind 0x004, NOT (1<<4) at 0x000. */
    e_rodnet_w32(h->base, E_RODNET_SOFT_RESET, E_RODNET_SRST_ASSERT);
    e_rodnet_mmio_fence();
    e_rodnet_w32(h->base, E_RODNET_SOFT_RESET, 0u);
    e_rodnet_w32(h->base, E_RODNET_GLOBAL_CTRL, 0);
    e_rodnet_w32(h->base, E_RODNET_IRQ_ENABLE,  0);
    e_rodnet_w32(h->base, E_RODNET_IRQ_STATUS,  0xFFFFFFFFu); /* W1C all */
    e_rodnet_mmio_fence();

    s_active_handle = h;
    log_info("e_rodnet: init ok (base=0x%08x)\n", (uint32_t)h->base);
    return E_RODNET_OK;
}

void e_rodnet_deinit(e_rodnet_handle_t *h)
{
    if (!h) return;
    e_rodnet_irq_detach(h);
    e_rodnet_w32(h->base, E_RODNET_GLOBAL_CTRL, 0);
    e_rodnet_w32(h->base, E_RODNET_IRQ_ENABLE,  0);
    e_rodnet_mmio_fence();
    if (s_active_handle == h) s_active_handle = 0;
}

/* ------------------------------------------------------------------------- */
/* Configuration                                                              */
/* ------------------------------------------------------------------------- */

e_rodnet_status_t e_rodnet_configure(e_rodnet_handle_t *h,
                                     const e_rodnet_config_t *cfg)
{
    if (!h || !cfg) return E_RODNET_EINVAL;
    if (e_rodnet_r32(h->base, E_RODNET_STATUS) & E_RODNET_STATUS_BUSY)
        return E_RODNET_EBUSY;
    if (cfg->temporal_mode > E_RODNET_TMODE_TDC) return E_RODNET_EINVAL;
    if (cfg->top_k == 0) return E_RODNET_EINVAL;

    uint32_t frame_cfg = ((uint32_t)cfg->feature_h << 16) |
                         ((uint32_t)cfg->history_depth & 0xFu);
    e_rodnet_w32(h->base, E_RODNET_FRAME_CONFIG, frame_cfg);
    e_rodnet_w32(h->base, E_RODNET_FRAME_W,      (uint32_t)cfg->feature_w);
    e_rodnet_w32(h->base, E_RODNET_TEMPORAL_MODE,  (uint32_t)cfg->temporal_mode);
    e_rodnet_w32(h->base, E_RODNET_TEMPORAL_BLEND, (uint32_t)(uint8_t)cfg->temporal_blend);

    uint32_t conv_cfg = ((uint32_t)cfg->conv_kernel << 16) |
                        (uint32_t)cfg->conv_channels;
    e_rodnet_w32(h->base, E_RODNET_CONV_CONFIG, conv_cfg);
    e_rodnet_w32(h->base, E_RODNET_CONV_ACTIVATION,
                 (uint32_t)(cfg->conv_activation & 0x7u));

    e_rodnet_w32(h->base, E_RODNET_DET_CONF_THRESH, (uint32_t)(uint8_t)cfg->conf_threshold);
    e_rodnet_w32(h->base, E_RODNET_DET_IOU_THRESH,  (uint32_t)(uint8_t)cfg->iou_threshold);
    e_rodnet_w32(h->base, E_RODNET_DET_TOP_K,       (uint32_t)cfg->top_k);
    /* 0x188 is reg_nms_enable in this RTL, not an IoU threshold. Writing
     * the signed threshold here (as older drivers did) corrupted the
     * enable bit. The IoU value already landed at 0x148. */
    e_rodnet_w32(h->base, E_RODNET_NMS_ENABLE, cfg->nms_enable ? 1u : 0u);
    e_rodnet_mmio_fence();
    return E_RODNET_OK;
}

e_rodnet_status_t e_rodnet_dma_setup(e_rodnet_handle_t *h,
                                     const e_rodnet_dma_desc_t *d)
{
    if (!h || !d) return E_RODNET_EINVAL;
    if (d->frame_size == 0 || d->frame_count == 0) return E_RODNET_EINVAL;
    if (e_rodnet_r32(h->base, E_RODNET_STATUS) & E_RODNET_STATUS_BUSY)
        return E_RODNET_EBUSY;

    e_rodnet_w32(h->base, E_RODNET_DMA_SRC_ADDR,    d->src_phys);
    e_rodnet_w32(h->base, E_RODNET_DMA_DST_ADDR,    d->dst_phys);
    e_rodnet_w32(h->base, E_RODNET_DMA_FRAME_SIZE,  d->frame_size);
    e_rodnet_w32(h->base, E_RODNET_DMA_FRAME_COUNT, (uint32_t)d->frame_count);
    e_rodnet_w32(h->base, E_RODNET_DMA_CONTROL,
                 d->loop ? E_RODNET_DMA_CTRL_LOOP : 0);
    e_rodnet_mmio_fence();
    return E_RODNET_OK;
}

/* ------------------------------------------------------------------------- */
/* Weight loading                                                             */
/*                                                                            */
/* Three paths into the layer-indexed weight BRAMs:                           */
/*   1. Per-word CSR write  : e_rodnet_weight_write                           */
/*   2. CSR auto-increment  : e_rodnet_weights_load_csr (fast CPU push)       */
/*   3. AXI master DMA pull : e_rodnet_weights_load_dma (hardware pulls DDR)  */
/*                                                                            */
/* The shared WT_STATUS register at 0x380 reports busy/done/error for both    */
/* CSR-stream and DMA paths; done/error latches are W1C via WT_STATUS[0].     */
/* ------------------------------------------------------------------------- */

static inline uint32_t e_rodnet_weight_ctrl_word(uint8_t layer_id, uint32_t extra)
{
    return extra |
           (((uint32_t)layer_id & 0xFu) << E_RODNET_WEIGHT_CTRL_LAYER_SHIFT);
}

e_rodnet_status_t e_rodnet_weight_write(e_rodnet_handle_t *h,
                                        uint8_t layer_id,
                                        uint32_t offset,
                                        int8_t value)
{
    if (!h) return E_RODNET_EINVAL;
    e_rodnet_w32(h->base, E_RODNET_WEIGHT_ADDR, offset);
    e_rodnet_w32(h->base, E_RODNET_WEIGHT_DATA, (uint32_t)(uint8_t)value);
    e_rodnet_mmio_fence();
    /* WRITE pulse | layer_sel (auto_inc=0 -> single-word write) */
    e_rodnet_w32(h->base, E_RODNET_WEIGHT_CTRL,
                 e_rodnet_weight_ctrl_word(layer_id,
                                           E_RODNET_WEIGHT_CTRL_WRITE));
    e_rodnet_mmio_fence();
    e_rodnet_w32(h->base, E_RODNET_WEIGHT_CTRL,
                 e_rodnet_weight_ctrl_word(layer_id, 0));
    return E_RODNET_OK;
}

e_rodnet_status_t e_rodnet_weights_load_csr(e_rodnet_handle_t *h,
                                            uint8_t layer_id,
                                            uint32_t dst_offset,
                                            const int8_t *values,
                                            uint32_t count)
{
    if (!h || !values || count == 0) return E_RODNET_EINVAL;
    if (e_rodnet_r32(h->base, E_RODNET_WT_STATUS) & E_RODNET_WT_STATUS_BUSY)
        return E_RODNET_EBUSY;

    /* W1C any stale done/error latches from the prior run. */
    e_rodnet_w32(h->base, E_RODNET_WT_STATUS, E_RODNET_WT_STATUS_CLEAR);

    e_rodnet_w32(h->base, E_RODNET_WEIGHT_ADDR,  dst_offset);
    e_rodnet_w32(h->base, E_RODNET_WEIGHT_COUNT, count);
    /* Arm the auto-incrementing streaming path. */
    e_rodnet_w32(h->base, E_RODNET_WEIGHT_CTRL,
                 e_rodnet_weight_ctrl_word(layer_id,
                                           E_RODNET_WEIGHT_CTRL_WRITE |
                                           E_RODNET_WEIGHT_CTRL_AUTO_INC));
    e_rodnet_mmio_fence();

    for (uint32_t i = 0; i < count; ++i) {
        e_rodnet_w32(h->base, E_RODNET_WEIGHT_DATA,
                     (uint32_t)(uint8_t)values[i]);
    }
    e_rodnet_mmio_fence();

    /* Drop WRITE/AUTO_INC but keep layer_sel stable. */
    e_rodnet_w32(h->base, E_RODNET_WEIGHT_CTRL,
                 e_rodnet_weight_ctrl_word(layer_id, 0));
    e_rodnet_mmio_fence();

    /* Spin on WT_STATUS - bounded at ~1M iters in case count_remain never
     * drains (e.g. RTL wedged). 1M CSR reads is < 100 ms at 50 MHz. */
    for (uint32_t spin = 0; spin < 1000000u; ++spin) {
        uint32_t st = e_rodnet_r32(h->base, E_RODNET_WT_STATUS);
        if (st & E_RODNET_WT_STATUS_ERROR) {
            e_rodnet_w32(h->base, E_RODNET_WT_STATUS, E_RODNET_WT_STATUS_CLEAR);
            return E_RODNET_EWEIGHT;
        }
        if (st & E_RODNET_WT_STATUS_DONE) {
            e_rodnet_w32(h->base, E_RODNET_WT_STATUS, E_RODNET_WT_STATUS_CLEAR);
            return E_RODNET_OK;
        }
    }
    return E_RODNET_ETIMEOUT;
}

e_rodnet_status_t e_rodnet_weights_load_dma(e_rodnet_handle_t *h,
                                            const e_rodnet_weight_dma_t *d)
{
    if (!h || !d || d->count == 0) return E_RODNET_EINVAL;
    if (e_rodnet_r32(h->base, E_RODNET_WT_STATUS) & E_RODNET_WT_STATUS_BUSY)
        return E_RODNET_EBUSY;

    /* Clear any stale latches before kicking a new run so the wait path
     * observes this transaction's done/error, not the previous one's. */
    e_rodnet_w32(h->base, E_RODNET_WT_STATUS, E_RODNET_WT_STATUS_CLEAR);

    e_rodnet_w32(h->base, E_RODNET_WT_DMA_SRC,   d->src_phys);
    e_rodnet_w32(h->base, E_RODNET_WT_DMA_COUNT, d->count);
    e_rodnet_w32(h->base, E_RODNET_WT_DMA_DST,   d->dst_offset);
    e_rodnet_mmio_fence();

    uint32_t ctrl = E_RODNET_WT_DMA_CTRL_START |
                    (((uint32_t)d->layer_id & 0xFu)
                        << E_RODNET_WT_DMA_CTRL_LAYER_SHIFT);
    e_rodnet_w32(h->base, E_RODNET_WT_DMA_CTRL, ctrl);
    e_rodnet_mmio_fence();
    return E_RODNET_OK;
}

e_rodnet_status_t e_rodnet_wait_weight_dma(e_rodnet_handle_t *h,
                                           uint64_t cycles)
{
    /* Weight-loader has no wired interrupt in this wrapper — poll-only.
     * The internal irq_weight_done_int is generated but never routed to
     * reg_irq_status or any output pin. */
    if (!h) return E_RODNET_EINVAL;
    uint64_t deadline = cycles ? (rv_rdcycle64() + cycles) : 0;
    for (;;) {
        uint32_t st = e_rodnet_r32(h->base, E_RODNET_WT_STATUS);
        if (st & E_RODNET_WT_STATUS_ERROR) {
            e_rodnet_w32(h->base, E_RODNET_WT_STATUS, E_RODNET_WT_STATUS_CLEAR);
            return E_RODNET_EWEIGHT;
        }
        if (st & E_RODNET_WT_STATUS_DONE) {
            e_rodnet_w32(h->base, E_RODNET_WT_STATUS, E_RODNET_WT_STATUS_CLEAR);
            return E_RODNET_OK;
        }
        if (cycles && rv_rdcycle64() >= deadline) return E_RODNET_ETIMEOUT;
        __asm__ volatile ("nop");
    }
}

/* ------------------------------------------------------------------------- */
/* Start / abort / wait                                                       */
/* ------------------------------------------------------------------------- */

e_rodnet_status_t e_rodnet_start(e_rodnet_handle_t *h)
{
    if (!h) return E_RODNET_EINVAL;
    e_rodnet_w32(h->base, E_RODNET_IRQ_STATUS, 0xFFFFFFFFu);
    /* IRQ bit 3 (weight_done) is unwired in this RTL — don't enable it. */
    e_rodnet_w32(h->base, E_RODNET_IRQ_ENABLE,
                 E_RODNET_IRQ_FRAME_DONE |
                 E_RODNET_IRQ_DETECTION  |
                 E_RODNET_IRQ_ERROR);
    e_rodnet_mmio_fence();
    e_rodnet_w32(h->base, E_RODNET_GLOBAL_CTRL, E_RODNET_GCTRL_ENABLE);
    e_rodnet_mmio_fence();
    uint32_t dma_ctrl = e_rodnet_r32(h->base, E_RODNET_DMA_CONTROL);
    e_rodnet_w32(h->base, E_RODNET_DMA_CONTROL,
                 dma_ctrl | E_RODNET_DMA_CTRL_START);
    e_rodnet_mmio_fence();
    return E_RODNET_OK;
}

void e_rodnet_abort(e_rodnet_handle_t *h)
{
    if (!h) return;
    uint32_t dma_ctrl = e_rodnet_r32(h->base, E_RODNET_DMA_CONTROL);
    e_rodnet_w32(h->base, E_RODNET_DMA_CONTROL,
                 dma_ctrl | E_RODNET_DMA_CTRL_ABORT);
    e_rodnet_mmio_fence();
}

e_rodnet_status_t e_rodnet_wait_done(e_rodnet_handle_t *h, uint64_t cycles)
{
    if (!h) return E_RODNET_EINVAL;
    uint64_t deadline = cycles ? (rv_rdcycle64() + cycles) : 0;
    for (;;) {
        if (h->flags & E_RODNET_IRQ_FRAME_DONE) {
            h->flags &= ~E_RODNET_IRQ_FRAME_DONE;
            return E_RODNET_OK;
        }
        if (e_rodnet_r32(h->base, E_RODNET_STATUS) & E_RODNET_STATUS_DONE)
            return E_RODNET_OK;
        if (cycles && rv_rdcycle64() >= deadline) return E_RODNET_ETIMEOUT;
        __asm__ volatile ("nop");
    }
}

/* ------------------------------------------------------------------------- */
/* Interrupts - dispatched through the standard Shakti PLIC driver           */
/* ------------------------------------------------------------------------- */

e_rodnet_status_t e_rodnet_irq_attach(e_rodnet_handle_t *h)
{
    if (!h) return E_RODNET_EINVAL;

    isr_table[h->plic_irq_frame]  = e_rodnet_isr_frame;
    isr_table[h->plic_irq_detect] = e_rodnet_isr_detect;
    isr_table[h->plic_irq_error]  = e_rodnet_isr_error;

    configure_interrupt(h->plic_irq_frame);
    configure_interrupt(h->plic_irq_detect);
    configure_interrupt(h->plic_irq_error);

    set_interrupt_threshold(0);
    set_interrupt_priority(PLIC_PRIORITY_4, h->plic_irq_frame);
    set_interrupt_priority(PLIC_PRIORITY_3, h->plic_irq_detect);
    set_interrupt_priority(PLIC_PRIORITY_5, h->plic_irq_error);

    interrupt_enable(h->plic_irq_frame);
    interrupt_enable(h->plic_irq_detect);
    interrupt_enable(h->plic_irq_error);
    return E_RODNET_OK;
}

void e_rodnet_irq_detach(e_rodnet_handle_t *h)
{
    if (!h) return;
    interrupt_disable(h->plic_irq_frame);
    interrupt_disable(h->plic_irq_detect);
    interrupt_disable(h->plic_irq_error);
    isr_table[h->plic_irq_frame]  = 0;
    isr_table[h->plic_irq_detect] = 0;
    isr_table[h->plic_irq_error]  = 0;
}

void e_rodnet_isr(e_rodnet_handle_t *h)
{
    if (!h) return;
    uint32_t pending = e_rodnet_r32(h->base, E_RODNET_IRQ_STATUS);
    if (pending == 0) return;
    if (pending & E_RODNET_IRQ_FRAME_DONE)  h->flags |= E_RODNET_IRQ_FRAME_DONE;
    if (pending & E_RODNET_IRQ_DETECTION)   h->flags |= E_RODNET_IRQ_DETECTION;
    if (pending & E_RODNET_IRQ_ERROR)       h->flags |= E_RODNET_IRQ_ERROR;
    e_rodnet_w32(h->base, E_RODNET_IRQ_STATUS, pending);
    e_rodnet_mmio_fence();
}

/* ------------------------------------------------------------------------- */
/* Detection readout                                                          */
/* ------------------------------------------------------------------------- */

size_t e_rodnet_read_detections(const void *dst_phys,
                                e_rodnet_det_t *out,
                                size_t max)
{
    if (!dst_phys || !out || !max) return 0;
    const volatile uint64_t *beats = (const volatile uint64_t *)dst_phys;
    size_t n = 0;
    for (size_t i = 0; i < max; ++i) {
        uint64_t b = beats[i];
        if (b == 0) break;
        out[n].h_idx    = E_RODNET_DET_H_IDX(b);
        out[n].w_idx    = E_RODNET_DET_W_IDX(b);
        out[n].conf     = E_RODNET_DET_CONF(b);
        out[n].class_id = E_RODNET_DET_CLASS(b);
        out[n]._pad     = 0;
        ++n;
    }
    return n;
}

/* ------------------------------------------------------------------------- */
/* Performance counters                                                       */
/* ------------------------------------------------------------------------- */

void e_rodnet_perf_read(e_rodnet_handle_t *h, e_rodnet_perf_t *p)
{
    if (!h || !p) return;
    p->cycles     = e_rodnet_r32(h->base, E_RODNET_PERF_CYCLES);
    p->frames     = e_rodnet_r32(h->base, E_RODNET_PERF_FRAMES);
    p->throughput = e_rodnet_r32(h->base, E_RODNET_PERF_THRUPUT);
}
