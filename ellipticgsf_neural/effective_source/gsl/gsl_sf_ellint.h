/*
 * Minimal stand-in for GSL's <gsl/gsl_sf_ellint.h>.
 *
 * kerr-circular.c includes that header and uses exactly two functions from
 * it, the complete elliptic integrals K and E. This declares just those,
 * so the source compiles with no GSL installation present; the definitions
 * come from ../gsl_ellint_shim.c.
 *
 * Put this file's PARENT directory on the include path, so that the
 * `#include <gsl/gsl_sf_ellint.h>` in kerr-circular.c resolves here:
 *
 *     cl /I path\to\ellipticgsf_py ...
 *     gcc -I path/to/ellipticgsf_py ...
 *
 * Real GSL takes precedence if it is on the include path first, and should:
 * see the note in gsl_ellint_shim.c.
 */

#ifndef ELLIPTICGSF_SHIM_GSL_SF_ELLINT_H
#define ELLIPTICGSF_SHIM_GSL_SF_ELLINT_H

enum { GSL_PREC_DOUBLE = 0, GSL_PREC_SINGLE = 1, GSL_PREC_APPROX = 2 };

double gsl_sf_ellint_Kcomp(double k, int mode);
double gsl_sf_ellint_Ecomp(double k, int mode);

#endif
