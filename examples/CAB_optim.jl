using PyPlot
import Base.cbrt

### tool functions for computing the solution of u + u^3 = A*sin(x)

# extend cbrt to complex numbers
function cbrt_cplx(z)
    z = Complex(z)
    real(z) >= 0 ? z^(1//3) : -(-z)^(1//3)
end

#  function cardan(b)
#      # we are in the case where p = 1
#      p = 1.0
#      q = -b
#      # the discriminant is R = -(4p^3 + 27q^2) <= 0 when p = 1
#      R = -(4p^3 + 27q^2)
#      v1 = cbrt_cplx((-q+sqrt(-R/27))/2)
#      v2 = cbrt_cplx((-q-sqrt(-R/27))/2)
#      v1 + v2
#  end

# u0 is the real solution of u + u^3 = A*sin(x) on [0,2π]
A = 10
B = imag.(asin(sqrt(4/27)/A*im))
#  function u0(x)
#      cardan(A*sin(x))
#  end

#  u0(x) = 1 / cbrt_cplx(sin(x + B*im)) + 1 / cbrt_cplx(sin(x - B*im))
#  u0(x) = 1 / (sin(x + B*im))^(1/3) + 1 / (sin(x - B*im))^(1/3)
u0(x) = 1 / (sin(x + B*im))^(1/3) + 1 / (sin(x - B*im))^(1/3)



ftsize = 20

function plot_complex_function(Rs, rs, is, f)
    res = f isa Function ? [f(x + im*y) for x in rs, y in is] : f
    res0 = f isa Function ? [f(im*y) for y in is] : f
    figure()
    rc("font", size=ftsize, serif="Computer Modern")
    rc("text", usetex=true)
    plot(Rs, real.(f.(Rs)), label="real part")
    legend()
    figure()
    rc("font", size=ftsize, serif="Computer Modern")
    rc("text", usetex=true)
    subplot(121)
    ux = imag.(res0)
    plot(is, ux, label="\$ {\\rm Im}(u_0({\\rm i} y)) \$")
    plot([B, B], [minimum(ux), maximum(ux)], "r--", label="\$ \\rm branching\\ points\\ \$")
    plot([-B, -B], [minimum(ux), maximum(ux)], "r--")
    xlabel("\$ y \$", size=ftsize)
    ylabel(" ", size=ftsize)
    legend()
    subplot(122)
    pcolormesh(rs, is, angle.(res)', cmap="hsv")
    # singularity at B
    plot([0], [B], "ro")
    plot([0], [-B], "ro")
    xlabel("\$ x \$", size=ftsize)
    ylabel("\$ y \$", size=ftsize)
    colorbar()
    figure()
    plot(Rs, [abs(f(x+B*im)) for x in Rs])
end

Rs = range(-pi, pi, length=20000)
rs = range(-0.1, 0.1, length=200)
is = range(-0.1, 0.1, length=200)
plot_complex_function(Rs, rs, is, z->u0(z))
#  plot_complex_function(Rs, rs, is, z->Complex(z)^(1/3))
#  plot_complex_function(Rs, rs, is, z->cbrt_cplx(z))

