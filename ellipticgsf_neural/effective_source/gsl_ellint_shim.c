/*
 * Minimal stand-in for the two GSL functions kerr-circular.c uses.
 *
 *     gsl_sf_ellint_Kcomp(k, mode)   complete elliptic integral K(k)
 *     gsl_sf_ellint_Ecomp(k, mode)   complete elliptic integral E(k)
 *
 * PREFER REAL GSL. This exists so the effective-source library can be built
 * and the Python binding exercised on a machine without GSL installed. Both
 * functions are computed by the arithmetic-geometric mean, which converges
 * quadratically and agrees with GSL and scipy to about 1e-16 relative over
 * 0 <= k < 1 (verified in tests/test_effective_source_library.py). It is accurate, but it is
 * not the library the published results were produced with, so build against
 * GSL for anything whose numbers you intend to publish.
 *
 * Note the argument convention: GSL takes the MODULUS k, not the parameter
 * m = k^2. scipy.special.ellipk/ellipe take m, hence the squaring in the test.
 *
 * Build alongside the source. kerr-circular.c does `#include
 * <gsl/gsl_sf_ellint.h>`, so the folder holding this file must be on the
 * include path -- the stand-in header beside it is what that resolves to:
 *
 *   gcc -O3 -fPIC -shared -I path/to/ellipticgsf_py -o libeffsource.so \
 *       kerr-circular.c path/to/ellipticgsf_py/gsl_ellint_shim.c -lm
 */

#include <math.h>

/* MSVC's <math.h> defines M_PI only if _USE_MATH_DEFINES was defined BEFORE
 * the include, which makes it a portability trap rather than a constant. The
 * value is spelled out here so this file compiles under cl, gcc and clang
 * with no flags. */
static const double SHIM_PI = 3.14159265358979323846;

/* GSL_PREC_DOUBLE etc. are an enum in GSL; the value is ignored here because
 * the AGM is run to full double precision unconditionally. */

static void agm_k_e(double k, double *K, double *E)
{
  double kp2 = 1.0 - k * k;
  if (kp2 < 0.0) kp2 = 0.0;  /* guard against k marginally above 1 */

  double a = 1.0;
  double b = sqrt(kp2);
  double c = k;

  /* E = K * (1 - sum_{n>=0} 2^{n-1} c_n^2), with c_0 = k. */
  double sum = 0.5 * c * c;
  double pow2 = 1.0;

  for (int i = 0; i < 64; i++) {
    if (fabs(c) < 1e-18 * a) break;
    double an = 0.5 * (a + b);
    double bn = sqrt(a * b);
    c = 0.5 * (a - b);
    a = an;
    b = bn;
    pow2 *= 2.0;
    sum += pow2 * 0.5 * c * c;
  }

  *K = SHIM_PI / (2.0 * a);
  *E = *K * (1.0 - sum);
}

double gsl_sf_ellint_Kcomp(double k, int mode)
{
  (void)mode;
  double K, E;
  agm_k_e(k, &K, &E);
  return K;
}

double gsl_sf_ellint_Ecomp(double k, int mode)
{
  (void)mode;
  double K, E;
  agm_k_e(k, &K, &E);
  return E;
}
