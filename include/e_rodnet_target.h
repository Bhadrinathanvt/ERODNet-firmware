/***************************************************************************
 * Project        : Yamuna Shakti SoC - e_rodnet target projection
 * File           : e_rodnet_target.h
 * Brief          : Turn raw e_rodnet detections (class + RA-map indices)
 *                  into application-level targets: class name + Cartesian
 *                  (x, y, z) in metres + radial velocity in m/s.
 *
 *                  z is always 0 — the current RA processor does azimuth-
 *                  only beamforming (angle_fft_4pt). Plumb in an elevation
 *                  FFT and add a z-index if/when that hardware exists.
 *
 *                  Velocity is not in the e_rodnet beat. Callers either:
 *                    (a) pass a parallel Range-Doppler magnitude map and
 *                        let this module pick the peak-Doppler bin, or
 *                    (b) ignore it and run their own tracker.
 ***************************************************************************/
#ifndef E_RODNET_TARGET_H
#define E_RODNET_TARGET_H

#include <stdint.h>
#include "e_rodnet.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ERODNET_CLASS_PEDESTRIAN = 0,   /* CRUW train_cruw.py */
    ERODNET_CLASS_CYCLIST    = 1,
    ERODNET_CLASS_CAR        = 2,
    ERODNET_CLASS_UNKNOWN    = 3,   /* class_id field is 2-bit; 3 = reserved */
} e_rodnet_class_t;

typedef struct {
    e_rodnet_class_t cls;
    float            x_m;          /* lateral (+ right), metres            */
    float            y_m;          /* downrange forward, metres            */
    float            z_m;          /* always 0 with azimuth-only RA        */
    float            v_mps;        /* radial velocity, m/s (+ approaching) */
    float            conf;         /* 0..1                                  */
} e_rodnet_target_t;

/* Geometry of the RA map fed into e_rodnet. Fill these from the chirp
 * configuration used to synthesize the RA map (not from e_rodnet — the
 * IP has no idea what physical units the bins map to). */
typedef struct {
    uint16_t range_bins;           /* == cfg.feature_h                     */
    uint16_t angle_bins;           /* == cfg.feature_w                     */
    float    range_res_m;          /* metres per range bin                  */
    float    azimuth_fov_rad;      /* full FOV, centred on boresight        */
    /* Optional Doppler lookup. Set rd_map = NULL to skip — v_mps is 0.  */
    const int16_t *rd_map;         /* magnitude, layout [range][doppler]    */
    uint16_t doppler_bins;
    float    velocity_res_mps;     /* metres-per-second per Doppler bin     */
} e_rodnet_geom_t;

const char *e_rodnet_class_name(e_rodnet_class_t c);

/* Project one raw detection into a target. Always returns OK; a missing
 * rd_map just yields v_mps = 0. */
void e_rodnet_project_det(const e_rodnet_det_t   *in,
                          const e_rodnet_geom_t  *g,
                          e_rodnet_target_t      *out);

/* Project a list. `n_in` = source detections, returns the number written. */
size_t e_rodnet_project_dets(const e_rodnet_det_t   *in,
                             size_t                  n_in,
                             const e_rodnet_geom_t  *g,
                             e_rodnet_target_t      *out,
                             size_t                  max_out);

#ifdef __cplusplus
}
#endif

#endif
