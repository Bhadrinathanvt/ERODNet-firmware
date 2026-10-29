/*
 * e_rodnet.h
 *
 * Bare-metal driver API for the e_rodnet radar object-detection IP on the
 * Yamuna Shakti SoC. Mirrors the shakti-sdk BSP driver style (prefixed
 * init/deinit, no hidden malloc, all state caller-owned).
 *
 * Updated 2026-04-23 for the new axi_top CSR map (weight auto-inc +
 * weight DMA + activation select + 64-bit status at 0x504).
 */

#ifndef E_RODNET_H
#define E_RODNET_H

#include <stdint.h>
#include <stddef.h>
#include "e_rodnet_regs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    E_RODNET_OK              =  0,
    E_RODNET_EBUSY           = -1,
    E_RODNET_ETIMEOUT        = -2,
    E_RODNET_EINVAL          = -3,
    E_RODNET_EDMA            = -4,
    E_RODNET_ENOIRQ          = -5,
    E_RODNET_EWEIGHT         = -6,
} e_rodnet_status_t;

typedef struct {
    uint8_t  temporal_mode;     /* E_RODNET_TMODE_* (wrapper truncates to [1:0]) */
    int8_t   temporal_blend;    /* signed Q0.DATA, used when mode = FUSION */
    uint16_t conv_channels;     /* 0..MAX_C                              */
    uint16_t conv_kernel;       /* kernel size (area packed by RTL)      */
    uint8_t  conv_activation;   /* E_RODNET_ACT_*                        */
    int8_t   conf_threshold;    /* signed Q0.DATA detection confidence   */
    int8_t   iou_threshold;     /* signed Q0.DATA NMS IoU (reg 0x148)    */
    uint8_t  top_k;             /* 1..TOP_K                              */
    uint8_t  nms_enable;        /* 0/1 — reg_nms_enable at 0x188         */
    uint16_t feature_h;
    uint16_t feature_w;
    uint8_t  history_depth;     /* 1..HISTORY_DEPTH                      */
} e_rodnet_config_t;

typedef struct {
    uint32_t src_phys;          /* DDR3 source, aligned to frame_size    */
    uint32_t dst_phys;          /* DDR3 dest for detection bursts        */
    uint32_t frame_size;        /* bytes per frame                       */
    uint16_t frame_count;       /* frames in the scatter-gather list     */
    uint8_t  loop;              /* 1 = cycle back to src after last frame */
} e_rodnet_dma_desc_t;

/* Weight-loader DMA descriptor: one invocation per layer. */
typedef struct {
    uint32_t src_phys;          /* DDR source of packed int8 weights     */
    uint32_t count;             /* number of int8 weights to load        */
    uint32_t dst_offset;        /* offset within the target layer's BRAM */
    uint8_t  layer_id;          /* 0..15 layer_sel                        */
} e_rodnet_weight_dma_t;

typedef struct {
    uintptr_t base;             /* MMIO base (defaults to E_RODNET_BASE) */
    uint32_t  plic_irq_frame;   /* PLIC source id for irq_frame_done     */
    uint32_t  plic_irq_detect;  /* PLIC source id for irq_detection       */
    uint32_t  plic_irq_error;   /* PLIC source id for irq_error           */
    volatile uint32_t flags;    /* bit0=frame_done, bit1=detect,
                                   bit2=err                              */
} e_rodnet_handle_t;

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

e_rodnet_status_t e_rodnet_init(e_rodnet_handle_t *h, uintptr_t base);
void              e_rodnet_deinit(e_rodnet_handle_t *h);

/* ------------------------------------------------------------------ */
/* Configuration                                                       */
/* ------------------------------------------------------------------ */

e_rodnet_status_t e_rodnet_configure(e_rodnet_handle_t *h,
                                     const e_rodnet_config_t *cfg);

e_rodnet_status_t e_rodnet_dma_setup(e_rodnet_handle_t *h,
                                     const e_rodnet_dma_desc_t *d);

/* ------------------------------------------------------------------ */
/* Weight loading                                                      */
/* ------------------------------------------------------------------ */

/* Per-weight CSR path (slow; for patches/fine tuning). Uses the new
 * WEIGHT_CTRL.write + layer_sel encoding at 0x30C. */
e_rodnet_status_t e_rodnet_weight_write(e_rodnet_handle_t *h,
                                        uint8_t layer_id,
                                        uint32_t offset,
                                        int8_t value);

/* Bulk weight streaming via the CSR auto-increment path. `values`
 * points to `count` signed int8 weights which are streamed into
 * (layer_id, dst_offset..dst_offset+count-1). Blocks until the
 * hardware reports WT_STATUS.done or WT_STATUS.error. */
e_rodnet_status_t e_rodnet_weights_load_csr(e_rodnet_handle_t *h,
                                            uint8_t layer_id,
                                            uint32_t dst_offset,
                                            const int8_t *values,
                                            uint32_t count);

/* AXI4-master DMA weight load from DDR. The accelerator pulls `count`
 * int8 weights from `src_phys` and writes them into the selected
 * layer's on-chip BRAM starting at `dst_offset`. Non-blocking; pair
 * with e_rodnet_wait_weight_dma().                                */
e_rodnet_status_t e_rodnet_weights_load_dma(e_rodnet_handle_t *h,
                                            const e_rodnet_weight_dma_t *d);

/* Poll / IRQ wait on weight loader completion.  Returns OK on done,
 * EWEIGHT on error bit, ETIMEOUT if `cycles` elapses.                */
e_rodnet_status_t e_rodnet_wait_weight_dma(e_rodnet_handle_t *h,
                                           uint64_t cycles);

/* ------------------------------------------------------------------ */
/* Start / stop / wait                                                 */
/* ------------------------------------------------------------------ */

e_rodnet_status_t e_rodnet_start(e_rodnet_handle_t *h);
void              e_rodnet_abort(e_rodnet_handle_t *h);
e_rodnet_status_t e_rodnet_wait_done(e_rodnet_handle_t *h, uint64_t cycles);

/* ------------------------------------------------------------------ */
/* Interrupts                                                          */
/* ------------------------------------------------------------------ */

e_rodnet_status_t e_rodnet_irq_attach(e_rodnet_handle_t *h);
void              e_rodnet_irq_detach(e_rodnet_handle_t *h);
void              e_rodnet_isr(e_rodnet_handle_t *h);

/* ------------------------------------------------------------------ */
/* Output extraction                                                   */
/* ------------------------------------------------------------------ */

size_t e_rodnet_read_detections(const void *dst_phys,
                                e_rodnet_det_t *out,
                                size_t max);

/* ------------------------------------------------------------------ */
/* Performance counters                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t cycles;
    uint32_t frames;
    uint32_t throughput;
} e_rodnet_perf_t;

void e_rodnet_perf_read(e_rodnet_handle_t *h, e_rodnet_perf_t *p);

#ifdef __cplusplus
}
#endif

#endif /* E_RODNET_H */
