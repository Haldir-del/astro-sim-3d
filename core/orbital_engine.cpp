// orbital_engine.cpp
// Native computational core for AstroSim 3D.
//
// Force model (Cartesian inertial frame, primary at origin):
//   a = -mu*r/|r|^3                          two-body point mass
//     + a_J2                                 oblateness (C22,S22,... ignored)
//     + a_drag                               rotating-atmosphere drag
//
// Integrator: embedded Runge-Kutta-Fehlberg 4(5) with adaptive stepsize,
// coefficients taken verbatim from:
//
//   [1] Fehlberg, E., "Classical Fifth-, Sixth-, Seventh-, and Eighth-Order
//       Runge-Kutta Formulas with Stepsize Control", NASA TR R-287,
//       George C. Marshall Space Flight Center, Huntsville, AL, Oct. 1968.
//       https://ntrs.nasa.gov/citations/19680027281
//
// Atmosphere: U.S. Standard Atmosphere 1976 (NOAA/NASA/USAF):
//
//   [2] U.S. Standard Atmosphere, 1976, NOAA/NASA/USAF, Washington D.C., 1976.
//       https://ntrs.nasa.gov/citations/19770009539
//       Exact barometric layer equations up to 86 km geometric altitude;
//       above that, log-linear interpolation of the Part 4 main tables
//       (density vs geometric altitude, 5 km resolution).
//
// J2 secular effects used for validation in tests follow:
//
//   [3] NASA/GSFC, "GDC Orbit Primer" (nodal regression / apsidal precession).
//       https://science.nasa.gov/wp-content/uploads/2023/05/GDC_OrbitPrimer.pdf
//
// Physical constants follow:
//
//   [4] NGA.STND.0036_1.0.0_WGS84, "Department of Defense World Geodetic
//       System 1984", NGA, 2014 (a, 1/f, GM, omega_earth).
//
// State: s = [x, y, z, vx, vy, vz]  (m, m/s)
//
// Terminal events detected every accepted step:
//   impact : |r| <= body_radius   -> status 1 (crossing refined by bisection)
//   escape : |r| >= escape_radius -> status 2 (disabled when <= 0)
//   buffer full                 -> status 3
//
// All array arguments are caller-allocated; the engine never allocates.
//
// Build (Windows / MinGW-w64):
//   g++ -shared -O2 -std=c++17 -o orbital_engine.dll core/orbital_engine.cpp
//       -static-libgcc -static-libstdc++

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace {

constexpr int kDim = 6; // state vector dimension

// ----------------------------------------------------------------------
// Parameter block layout (params[12], see also app/engine.py)
// ----------------------------------------------------------------------
constexpr int P_MU = 0;            // gravitational parameter (m^3/s^2)
constexpr int P_BODY_RADIUS = 1;   // impact sphere radius (m); <=0 disables
constexpr int P_ESCAPE_RADIUS = 2; // escape sphere radius (m); <=0 disables
constexpr int P_J2 = 3;            // zonal harmonic (-); 0 disables
constexpr int P_REQ = 4;           // equatorial radius for J2 scaling (m)
constexpr int P_CD = 5;            // drag coefficient (-)
constexpr int P_AREA = 6;          // frontal area (m^2)
constexpr int P_MASS = 7;          // spacecraft mass (kg)
constexpr int P_OMEGA_EARTH = 8;   // atmosphere angular rate (rad/s)
constexpr int P_LEN = 12;          // total block length (rest reserved)

// Mean sea-level reference radius used by the atmosphere model (USSA76 is
// tabulated against altitude above mean sea level; we approximate MSL with
// the mean Earth radius). Only affects the density lookup, not dynamics.
constexpr double kMslRadius = 6371000.0;

// ----------------------------------------------------------------------
// U.S. Standard Atmosphere 1976 [2]
// ----------------------------------------------------------------------
constexpr double kG0 = 9.80665;        // standard gravity (m/s^2), USSA76
constexpr double kRspec = 287.05287;   // R*/M = 8.31432/0.0289644 (J/kg/K)
constexpr double kUssaRe = 6356766.0;  // Earth radius for geopotential (m)

// Barometric layers below 86 km geometric (geopotential bases).
constexpr int kNumLayers = 7;
constexpr double kLayerH[kNumLayers] = {
    0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0};
constexpr double kLayerT[kNumLayers] = {
    288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65};
constexpr double kLayerL[kNumLayers] = {
    -6.5e-3, 0.0, 1.0e-3, 2.8e-3, 0.0, -2.8e-3, -2.0e-3};
constexpr double kLayerP[kNumLayers] = {
    101325.0, 22632.06, 5474.889, 868.0187, 110.9063, 66.93887, 3.9564200};
constexpr double kTopGeopotential = 84852.0; // top of lowest 7 layers (m)

// Density anchors above 86 km geometric, from USSA76 Part 4 main tables [2].
struct DensityAnchor {
    double z;   // geometric altitude (m)
    double rho; // density (kg/m^3)
};
constexpr DensityAnchor kHighAltDensity[] = {
    {86.0e3, 6.9579e-6},  {90.0e3, 3.4400e-6},  {95.0e3, 1.3873e-6},
    {100.0e3, 5.6044e-7}, {110.0e3, 9.6734e-8}, {120.0e3, 2.2199e-8},
    {130.0e3, 8.1494e-9}, {150.0e3, 2.0752e-9}, {180.0e3, 5.1940e-10},
    {200.0e3, 2.5407e-10},{250.0e3, 6.0706e-11},{300.0e3, 1.9159e-11},
    {350.0e3, 7.0011e-12},{400.0e3, 2.8028e-12},{450.0e3, 1.1843e-12},
    {500.0e3, 5.2148e-13},{550.0e3, 2.3832e-13},{600.0e3, 1.1367e-13},
    {650.0e3, 5.7114e-14},{700.0e3, 3.0698e-14},{750.0e3, 1.7900e-14},
    {800.0e3, 1.1358e-14},{850.0e3, 7.8223e-15},{900.0e3, 5.7587e-15},
    {950.0e3, 4.4525e-15},{1000.0e3, 3.5611e-15}};
constexpr int kNumAnchors =
    static_cast<int>(sizeof(kHighAltDensity) / sizeof(kHighAltDensity[0]));

double ussa76_density(double radius_m) {
    const double z = radius_m - kMslRadius; // geometric altitude (m)
    if (z < 0.0) return kHighAltDensity[0].rho; // clamp below sea level

    if (z <= 86000.0) {
        // Geopotential altitude, USSA76 eq. (17): H = Re*z/(Re+z)
        const double H = kUssaRe * z / (kUssaRe + z);

        int i = kNumLayers - 1;
        while (i > 0 && H < kLayerH[i]) --i;

        const double Tb = kLayerT[i];
        const double Lb = kLayerL[i];
        const double pb = kLayerP[i];
        const double dH = H - kLayerH[i];

        double T, p;
        if (Lb != 0.0) {
            T = Tb + Lb * dH;
            p = pb * std::pow(T / Tb, -kG0 / (Lb * kRspec));
        } else {
            T = Tb;
            p = pb * std::exp(-kG0 * dH / (kRspec * Tb));
        }
        return p / (kRspec * T);
    }

    // Log-linear interpolation between tabulated densities.
    if (z >= kHighAltDensity[kNumAnchors - 1].z) {
        const double rho_top = kHighAltDensity[kNumAnchors - 1].rho;
        return rho_top; // negligible above 1000 km
    }
    int j = 0;
    while (j < kNumAnchors - 1 && kHighAltDensity[j + 1].z < z) ++j;
    const double z0 = kHighAltDensity[j].z, z1 = kHighAltDensity[j + 1].z;
    const double r0 = kHighAltDensity[j].rho, r1 = kHighAltDensity[j + 1].rho;
    const double w = (z - z0) / (z1 - z0);
    return std::exp(std::log(r0) + w * (std::log(r1) - std::log(r0)));
}

// ----------------------------------------------------------------------
// Dynamics
// ----------------------------------------------------------------------
inline void compute_accel(const double* s, const double* params, double* a) {
    const double x = s[0], y = s[1], z = s[2];
    const double vx = s[3], vy = s[4], vz = s[5];

    const double r2 = x * x + y * y + z * z;
    const double r = std::sqrt(r2);
    const double inv_r3 = 1.0 / (r2 * r);
    const double mu = params[P_MU];

    // Two-body point-mass gravity.
    a[0] = -mu * x * inv_r3;
    a[1] = -mu * y * inv_r3;
    a[2] = -mu * z * inv_r3;

    // J2 oblateness perturbation (axisymmetric zonal term).
    const double j2 = params[P_J2];
    if (j2 != 0.0 && r > params[P_REQ]) {
        const double req2 = params[P_REQ] * params[P_REQ];
        const double factor = 1.5 * j2 * mu * req2 / (r2 * r2 * r);
        const double zr2 = 5.0 * z * z / r2;
        a[0] += factor * x * (zr2 - 1.0);
        a[1] += factor * y * (zr2 - 1.0);
        a[2] += factor * z * (zr2 - 3.0);
    }

    // Atmospheric drag with co-rotating atmosphere.
    const double cd = params[P_CD];
    const double area = params[P_AREA];
    const double mass = params[P_MASS];
    if (cd > 0.0 && area > 0.0 && mass > 0.0) {
        const double w = params[P_OMEGA_EARTH];
        // v_rel = v - omega x r  (omega along +Z)
        const double vrx = vx + w * y;
        const double vry = vy - w * x;
        const double vrz = vz;
        const double vr = std::sqrt(vrx * vrx + vry * vry + vrz * vrz);
        if (vr > 0.0) {
            const double rho = ussa76_density(r);
            if (rho > 0.0) {
                const double k = 0.5 * cd * area * rho * vr / mass;
                a[0] -= k * vrx;
                a[1] -= k * vry;
                a[2] -= k * vrz;
            }
        }
    }
}

inline void compute_deriv(const double* s, const double* params, double* d) {
    d[0] = s[3];
    d[1] = s[4];
    d[2] = s[5];
    compute_accel(s, params, d + 3);
}

// Magnitude of the non-two-body acceleration (J2 + drag), recorded per
// sample for telemetry.
inline double perturbation_mag(const double* s, const double* params) {
    double a_full[3], a_two[3], p[P_LEN];

    compute_accel(s, params, a_full);

    for (int i = 0; i < P_LEN; ++i) p[i] = params[i];
    p[P_J2] = 0.0;
    p[P_CD] = 0.0;
    p[P_AREA] = 0.0;
    compute_accel(s, p, a_two);

    const double dx = a_full[0] - a_two[0];
    const double dy = a_full[1] - a_two[1];
    const double dz = a_full[2] - a_two[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// ----------------------------------------------------------------------
// RKF45 stage coefficients, NASA TR R-287 [1]
// ----------------------------------------------------------------------
constexpr double kC2 = 1.0 / 4.0;
constexpr double kC3 = 3.0 / 8.0;
constexpr double kC4 = 12.0 / 13.0;
constexpr double kC5 = 1.0;
constexpr double kC6 = 1.0 / 2.0;

constexpr double kA21 = 1.0 / 4.0;
constexpr double kA31 = 3.0 / 32.0, kA32 = 9.0 / 32.0;
constexpr double kA41 = 1932.0 / 2197.0, kA42 = -7200.0 / 2197.0,
                 kA43 = 7296.0 / 2197.0;
constexpr double kA51 = 439.0 / 216.0, kA52 = -8.0, kA53 = 3680.0 / 513.0,
                 kA54 = -845.0 / 4104.0;
constexpr double kA61 = -8.0 / 27.0, kA62 = 2.0, kA63 = -3544.0 / 2565.0,
                 kA64 = 1859.0 / 4104.0, kA65 = -11.0 / 40.0;

// 4th-order solution weights (accepted propagation).
constexpr double kB4[6] = {25.0 / 216.0, 0.0, 1408.0 / 2565.0,
                           2197.0 / 4104.0, -1.0 / 5.0, 0.0};
// 5th-order solution weights (error estimate / dense output).
constexpr double kB5[6] = {16.0 / 135.0, 0.0, 6656.0 / 12825.0,
                           28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 11.0};
// Difference weights e = B5 - B4 (leading local error estimate).
constexpr double kE[6] = {1.0 / 360.0, 0.0, -128.0 / 4275.0, -2197.0 / 75240.0,
                          1.0 / 50.0, 2.0 / 55.0};

// One RKF45 attempt: computes the six stages and both solutions.
void rkf45_step(const double* s, double h, const double* params,
                double* y4, double* y5, double* err_vec) {
    double k1[kDim], k2[kDim], k3[kDim], k4[kDim], k5[kDim], k6[kDim], tmp[kDim];

    compute_deriv(s, params, k1);
    for (int i = 0; i < kDim; ++i) tmp[i] = s[i] + h * kA21 * k1[i];
    compute_deriv(tmp, params, k2);
    for (int i = 0; i < kDim; ++i)
        tmp[i] = s[i] + h * (kA31 * k1[i] + kA32 * k2[i]);
    compute_deriv(tmp, params, k3);
    for (int i = 0; i < kDim; ++i)
        tmp[i] = s[i] + h * (kA41 * k1[i] + kA42 * k2[i] + kA43 * k3[i]);
    compute_deriv(tmp, params, k4);
    for (int i = 0; i < kDim; ++i)
        tmp[i] = s[i] + h * (kA51 * k1[i] + kA52 * k2[i] + kA53 * k3[i] +
                             kA54 * k4[i]);
    compute_deriv(tmp, params, k5);
    for (int i = 0; i < kDim; ++i)
        tmp[i] = s[i] + h * (kA61 * k1[i] + kA62 * k2[i] + kA63 * k3[i] +
                             kA64 * k4[i] + kA65 * k5[i]);
    compute_deriv(tmp, params, k6);

    for (int i = 0; i < kDim; ++i) {
        y4[i] = s[i] + h * (kB4[0] * k1[i] + kB4[1] * k2[i] + kB4[2] * k3[i] +
                            kB4[3] * k4[i] + kB4[4] * k5[i] + kB4[5] * k6[i]);
        y5[i] = s[i] + h * (kB5[0] * k1[i] + kB5[1] * k2[i] + kB5[2] * k3[i] +
                            kB5[3] * k4[i] + kB5[4] * k5[i] + kB5[5] * k6[i]);
        err_vec[i] = h * (kE[0] * k1[i] + kE[1] * k2[i] + kE[2] * k3[i] +
                          kE[3] * k4[i] + kE[4] * k5[i] + kE[5] * k6[i]);
    }
}

inline double radius_of(const double* s) {
    return std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
}

inline double speed_of(const double* s) {
    return std::sqrt(s[3] * s[3] + s[4] * s[4] + s[5] * s[5]);
}

// Relative error norm: position scaled by current radius, velocity by a
// characteristic speed (max of actual speed and local circular speed) so the
// norm stays meaningful for near-radial or slow trajectories.
inline double scaled_error(const double* err_vec, const double* s,
                           const double* params) {
    const double r = std::max(radius_of(s), 1.0);
    const double v_char =
        std::max(speed_of(s), std::sqrt(params[P_MU] / r));
    double err2 = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double dp = err_vec[i] / r;
        const double dv = err_vec[3 + i] / v_char;
        err2 += dp * dp + dv * dv;
    }
    return std::sqrt(err2 / 3.0);
}

// Refine an impact crossing inside [0, h] starting from s0 by bisection on
// |r(t)| - body_radius, using single 5th-order steps. Returns the state at
// the refined crossing and writes the elapsed time into t_out.
void refine_impact(const double* s0, double h, const double* params,
                   double body_radius, double* s_hit, double* t_out) {
    double lo = 0.0, hi = h;
    double cur[kDim];
    for (int i = 0; i < kDim; ++i) cur[i] = s0[i];
    for (int iter = 0; iter < 40; ++iter) {
        const double mid = 0.5 * (lo + hi);
        double y4[kDim], y5[kDim], ev[kDim];
        rkf45_step(s0, mid, params, y4, y5, ev);
        for (int i = 0; i < kDim; ++i) cur[i] = y5[i];
        if (radius_of(cur) <= body_radius) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    const double tf = 0.5 * (lo + hi);
    double y4[kDim], y5[kDim], ev[kDim];
    rkf45_step(s0, tf, params, y4, y5, ev);
    for (int i = 0; i < kDim; ++i) s_hit[i] = y5[i];
    *t_out = tf;
}

} // namespace

extern "C" {

// Propagates the trajectory and records time/state samples.
//
// Parameters
//   state0     initial state, 6 doubles [x,y,z,vx,vy,vz]
//   duration   total propagation time (s)
//   dt_max     maximum integration step (s)
//   rel_tol    target relative accuracy per step (e.g. 1e-10)
//   params     physical parameter block, see P_* layout above (length >= 12)
//   max_steps  capacity of the output buffers
//   out_t      [max_steps] simulated time of each sample (s)
//   out_state  [max_steps * 6] state history, row-major
//   out_acc    [max_steps] magnitude of J2+drag acceleration per sample (may
//              be NULL to skip)
//   out_status receives the terminal status code
//
// Return value: number of samples written (>= 1), or -1 on invalid arguments.
//
// Status codes: 0 completed, 1 impact, 2 escape, 3 output buffer exhausted.
int simulate_trajectory(const double* state0,
                        double duration,
                        double dt_max,
                        double rel_tol,
                        const double* params,
                        int max_steps,
                        double* out_t,
                        double* out_state,
                        double* out_acc,
                        int* out_status) {
    if (!state0 || !params || !out_t || !out_state || !out_status) return -1;
    if (max_steps < 1 || !(duration > 0.0) || !(dt_max > 0.0) ||
        !(rel_tol > 0.0)) {
        return -1;
    }
    if (!(params[P_MU] > 0.0)) return -1;

    const double h_min = dt_max * 1e-9;

    double s[kDim];
    for (int i = 0; i < kDim; ++i) s[i] = state0[i];

    int n = 0;
    double t = 0.0;
    out_t[n] = t;
    for (int i = 0; i < kDim; ++i) out_state[n * kDim + i] = s[i];
    if (out_acc) out_acc[n] = perturbation_mag(s, params);
    ++n;

    *out_status = 0;
    double h = dt_max;

    while (t < duration) {
        if (h > duration - t) h = duration - t;

        double y4[kDim], y5[kDim], ev[kDim], err = 0.0;
        bool accepted = false;

        while (!accepted) {
            rkf45_step(s, h, params, y4, y5, ev);
            err = scaled_error(ev, s, params);
            if (err <= rel_tol || h <= h_min) {
                accepted = true;
            } else {
                // Reject: shrink with the standard controller exponent 1/5.
                double factor = 0.9 * std::pow(rel_tol / err, 0.2);
                if (factor < 0.1) factor = 0.1;
                h *= factor;
                if (h < h_min) h = h_min;
            }
        }

        const double r_before = radius_of(s);
        const double r_after = radius_of(y5);
        const double body_radius = params[P_BODY_RADIUS];

        bool hit = false;
        if (body_radius > 0.0 && r_after <= body_radius &&
            r_before > body_radius) {
            // Bisect the crossing inside the accepted step for a clean
            // impact point.
            double t_hit = h;
            refine_impact(s, h, params, body_radius, y5, &t_hit);
            hit = true;
            h = t_hit;
        }

        for (int i = 0; i < kDim; ++i) s[i] = y5[i];
        t += h;

        if (n >= max_steps) {
            *out_status = 3;
            return n;
        }
        out_t[n] = t;
        for (int i = 0; i < kDim; ++i) out_state[n * kDim + i] = s[i];
        if (out_acc) out_acc[n] = perturbation_mag(s, params);
        ++n;

        if (hit) {
            *out_status = 1;
            return n;
        }
        if (params[P_ESCAPE_RADIUS] > 0.0 && r_after >= params[P_ESCAPE_RADIUS]) {
            *out_status = 2;
            return n;
        }

        // Grow the step when the last attempt was comfortably accurate.
        if (!hit && err < rel_tol / 8.0) {
            double factor = 0.9 * std::pow(rel_tol / (err + 1e-300), 0.2);
            if (factor > 2.0) factor = 2.0;
            h *= factor;
            if (h > dt_max) h = dt_max;
        }
    }

    return n;
}

} // extern "C"
