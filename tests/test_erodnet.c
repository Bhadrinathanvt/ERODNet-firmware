/***************************************************************************
 * Project        : Yamuna Shakti SoC - bring-up test, e_rodnet only
 * File           : test_erodnet.c
 * Brief          : Standalone verification for the e_rodnet radar
 *                  object-detection IP at 0x0005_0000. No DMA engine and
 *                  no CAN involvement — e_rodnet pulls its own frames
 *                  through its AXI-master port.
 *
 *   sub-test 1 : reset / configure round-trip
 *   sub-test 2 : per-word CSR weight write + self-verify via read
 *   sub-test 3 : bulk CSR auto-increment weight stream
 *   sub-test 4 : AXI-master weight DMA
 *   sub-test 5 : single-frame inference on a synthetic RA map with
 *                three injected hot cells — at least one hot cell must
 *                appear in the detection list.
 ***************************************************************************/

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "platform.h"
#include "uart.h"
#include "log.h"
#include "utils.h"

#include "e_rodnet.h"
#include "e_rodnet_regs.h"

#define EROD_DDR_BASE     0x8D000000u
#define RA_MAP_ADDR       (EROD_DDR_BASE + 0x00000u)
#define DET_DST_ADDR      (EROD_DDR_BASE + 0x10000u)
#define WT_STAGE_ADDR     (EROD_DDR_BASE + 0x20000u)

#define RA_MAP_H          16u
#define RA_MAP_W          16u
#define RA_MAP_BYTES      (RA_MAP_H * RA_MAP_W * 2u)   /* int16 magnitude */

#define HOT_CELLS         3u
static const struct { uint16_t h, w; int8_t magnitude; } hot_cells[HOT_CELLS] = {
    { 4,  5,  110 },
    { 8,  9,   95 },
    { 12, 2,   80 },
};

typedef struct { const char *name; int rc; } result_t;
#define MAX_TESTS 8
static result_t results[MAX_TESTS];
static uint32_t results_n;
static void record(const char *name, int rc)
{
    if (results_n < MAX_TESTS) {
        results[results_n].name = name;
        results[results_n].rc   = rc;
        ++results_n;
    }
    printf("  %-28s : %s (%d)\n\r", name, rc == 0 ? "PASS" : "FAIL", rc);
}

/* ---------------------------------------------------------------- */
/* Test 1 — reset / configure round-trip                             */
/* ---------------------------------------------------------------- */
static int test_configure(e_rodnet_handle_t *h)
{
    e_rodnet_config_t cfg = {
        .temporal_mode   = E_RODNET_TMODE_NONE,
        .temporal_blend  = 0,
        .conv_channels   = 16,
        .conv_kernel     = 3,
        .conv_activation = E_RODNET_ACT_RELU,
        .conf_threshold  = 16,
        .iou_threshold   = 64,
        .top_k           = 8,
        .nms_enable      = 1,
        .feature_h       = RA_MAP_H,
        .feature_w       = RA_MAP_W,
        .history_depth   = 1,
    };
    return (e_rodnet_configure(h, &cfg) == E_RODNET_OK) ? 0 : -1;
}

/* ---------------------------------------------------------------- */
/* Test 2 — per-word CSR weight write                                */
/* ---------------------------------------------------------------- */
static int test_weight_write(e_rodnet_handle_t *h)
{
    const int8_t pattern[4] = { 0x11, -0x22, 0x33, -0x44 };
    for (uint32_t i = 0; i < 4u; ++i) {
        if (e_rodnet_weight_write(h, /*layer*/ 0, /*offset*/ i,
                                   pattern[i]) != E_RODNET_OK) return -1;
    }
    return 0;
}

/* ---------------------------------------------------------------- */
/* Test 3 — CSR auto-increment bulk weight stream                    */
/* ---------------------------------------------------------------- */
static int test_weight_csr(e_rodnet_handle_t *h)
{
    int8_t values[256];
    for (uint32_t i = 0; i < sizeof values; ++i) {
        values[i] = (int8_t)((i * 7u) ^ 0x5Au);
    }
    return e_rodnet_weights_load_csr(h, /*layer*/ 0, /*offset*/ 0,
                                      values, sizeof values);
}

/* ---------------------------------------------------------------- */
/* Test 4 — AXI-master weight DMA                                    */
/* ---------------------------------------------------------------- */
static int test_weight_dma(e_rodnet_handle_t *h)
{
    int8_t *stage = (int8_t *)(uintptr_t)WT_STAGE_ADDR;
    const uint32_t n = 1024;
    for (uint32_t i = 0; i < n; ++i) {
        stage[i] = (int8_t)(i & 0x7Fu);
    }
    __asm__ volatile ("fence iorw, iorw" ::: "memory");

    e_rodnet_weight_dma_t d = {
        .src_phys   = WT_STAGE_ADDR,
        .count      = n,
        .dst_offset = 256,
        .layer_id   = 1,
    };
    e_rodnet_status_t s = e_rodnet_weights_load_dma(h, &d);
    if (s != E_RODNET_OK) return (int)s - 100;
    s = e_rodnet_wait_weight_dma(h, 50000000u);
    return (s == E_RODNET_OK) ? 0 : (int)s - 200;
}

/* ---------------------------------------------------------------- */
/* Test 5 — synthetic single-frame inference                         */
/* ---------------------------------------------------------------- */
static int test_inference(e_rodnet_handle_t *h)
{
    int16_t *ra = (int16_t *)(uintptr_t)RA_MAP_ADDR;
    memset(ra, 0, RA_MAP_BYTES);
    for (uint32_t i = 0; i < HOT_CELLS; ++i) {
        ra[hot_cells[i].h * RA_MAP_W + hot_cells[i].w] =
            (int16_t)((int32_t)hot_cells[i].magnitude * 256);
    }
    memset((void *)(uintptr_t)DET_DST_ADDR, 0, 64 * sizeof(e_rodnet_det_t));
    __asm__ volatile ("fence iorw, iorw" ::: "memory");

    e_rodnet_dma_desc_t d = {
        .src_phys    = RA_MAP_ADDR,
        .dst_phys    = DET_DST_ADDR,
        .frame_size  = RA_MAP_BYTES,
        .frame_count = 1,
        .loop        = 0,
    };
    if (e_rodnet_dma_setup(h, &d) != E_RODNET_OK) return -1;
    if (e_rodnet_start(h)         != E_RODNET_OK) return -2;
    if (e_rodnet_wait_done(h, 5000000u) != E_RODNET_OK) return -3;

    e_rodnet_det_t dets[16];
    size_t n = e_rodnet_read_detections((const void *)(uintptr_t)DET_DST_ADDR,
                                         dets, sizeof(dets) / sizeof(dets[0]));
    if (n == 0) return -4;

    /* At least one returned detection must land on a hot cell. */
    for (size_t i = 0; i < n; ++i) {
        for (uint32_t k = 0; k < HOT_CELLS; ++k) {
            if (dets[i].h_idx == hot_cells[k].h &&
                dets[i].w_idx == hot_cells[k].w) {
                printf("      (hit (%u,%u), %u total det)\n\r",
                       dets[i].h_idx, dets[i].w_idx, (unsigned)n);
                return 0;
            }
        }
    }
    return -5;
}

/* ---------------------------------------------------------------- */
/* main                                                              */
/* ---------------------------------------------------------------- */
int main(void)
{
    init_uart();
    printf("\n== e_rodnet bring-up test ==\n\r");

    e_rodnet_handle_t h;
    if (e_rodnet_init(&h, E_RODNET_BASE) != E_RODNET_OK) {
        printf("FATAL: e_rodnet_init\n\r"); return -1;
    }

    record("configure",       test_configure(&h));
    record("weight_write",    test_weight_write(&h));
    record("weight_csr",      test_weight_csr(&h));
    record("weight_dma",      test_weight_dma(&h));
    record("inference",       test_inference(&h));

    uint32_t pass = 0, fail = 0;
    for (uint32_t i = 0; i < results_n; ++i) {
        if (results[i].rc == 0) ++pass; else ++fail;
    }
    printf("---\nerod: %u/%u PASS\n\r", (unsigned)pass, (unsigned)results_n);

    e_rodnet_deinit(&h);
    for (;;) { __asm__ volatile ("wfi"); }
    return fail == 0 ? 0 : -1;
}
