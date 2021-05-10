## we solve here u + u^3 = Asin(x) and then perform an analytic expansion

import Base.cbrt

include("plotting_analytic.jl")

# extend cbrt to complex numbers
function cbrt_cplx(z)
    z = Complex(z)
    real(z) >= 0 ? z^(1//3) : -(-z)^(1//3)
end

# real solution of u + p*u^3 = b, using Cardan formulas
function cardan(b)
    p = 1.0
    q = -b
    R = -(4p^3 + 27q^2)
    v1 = cbrt_cplx((-q+sqrt(-R/27))/2)
    v2 = cbrt_cplx((-q-sqrt(-R/27))/2)
    v1 + v2
end

function u0(x)
    cardan(A*sin(x))
end

#  figure()
#  rx = range(-π, π, length=500)
#  plot(rx, u.(rx))

#  rs = range(-π, π, length=500)
#  is = range(-2, 2, length=500)
#  plot_complex_function(rs, is, z->u(z))
