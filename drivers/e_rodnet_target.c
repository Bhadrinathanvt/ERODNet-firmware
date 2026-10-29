/***************************************************************************
 * Project        : Yamuna Shakti SoC - e_rodnet target projection
 * File           : e_rodnet_target.c
 * Brief          : Implementation of e_rodnet_target.h — RA-index -> (x,y,z)
 *                  plus optional peak-Doppler-bin velocity lookup.
 ***************************************************************************/
#include "e_rodnet_target.h"

#include <math.h>
#include <stddef.h>

const char *e_rodnet_class_name(e_rodnet_class_t c)
{
    switch (c) {
    case ERODNET_CLASS_PEDESTRIAN: return "pedestrian";
    case ERODNET_CLASS_CYCLIST:    return "cyclist";
    case ERODNET_CLASS_CAR:        return "car";
    default:                       return "unknown";
    }
}

/* Angle mapping: the 4-point angle FFT in the RA processor uniformly
 * samples the azimuth FOV. Bin w=0 -> -FOV/2, bin W-1 -> +FOV/2.        */
static inline float bin_to_theta(uint16_t w_idx, uint16_t W, float fov)
{
    if (W <= 1) return 0.f;
    float u = ((float)w_idx + 0.5f) / (float)W;    /* 0..1 centre-of-bin */
    return (u - 0.5f) * fov;                        /* -fov/2 .. +fov/2   */
}

/* Find the peak-magnitude Doppler bin for this range row and translate
 * it into a signed radial velocity. Bin layout is standard FFT:
 *   bin < doppler_bins/2 -> approaching (positive velocity here),
 *   bin >= doppler_bins/2 -> receding (wrap-around negative). */
static float doppler_peak_mps(const e_rodnet_geom_t *g, uint16_t h_idx)
{
    if (!g->rd_map || g->doppler_bins == 0) return 0.f;
    const int16_t *row = g->rd_map + (size_t)h_idx * g->doppler_bins;
    uint16_t peak_bin = 0;
    int16_t  peak_mag = row[0];
    for (uint16_t d = 1; d < g->doppler_bins; ++d) {
        if (row[d] > peak_mag) { peak_mag = row[d]; peak_bin = d; }
    }
    int32_t signed_bin = (int32_t)peak_bin;
    if (signed_bin >= (int32_t)g->doppler_bins / 2)
        signed_bin -= (int32_t)g->doppler_bins;
    return (float)signed_bin * g->velocity_res_mps;
}

void e_rodnet_project_det(const e_rodnet_det_t  *in,
                          const e_rodnet_geom_t *g,
                          e_rodnet_target_t     *out)
{
    float range = ((float)in->h_idx + 0.5f) * g->range_res_m;
    float theta = bin_to_theta(in->w_idx, g->angle_bins, g->azimuth_fov_rad);

    out->cls   = (in->class_id <= ERODNET_CLASS_CAR)
               ? (e_rodnet_class_t)in->class_id
               : ERODNET_CLASS_UNKNOWN;
    out->x_m   = range * sinf(theta);     /* lateral */
    out->y_m   = range * cosf(theta);     /* downrange */
    out->z_m   = 0.f;                      /* no elevation axis in current RTL */
    out->conf  = (float)((int8_t)in->conf) / 127.f;
    out->v_mps = doppler_peak_mps(g, in->h_idx);
}

size_t e_rodnet_project_dets(const e_rodnet_det_t  *in, size_t n_in,
                             const e_rodnet_geom_t *g,
                             e_rodnet_target_t     *out, size_t max_out)
{
    size_t n = (n_in < max_out) ? n_in : max_out;
    for (size_t i = 0; i < n; ++i) e_rodnet_project_det(&in[i], g, &out[i]);
    return n;
}
