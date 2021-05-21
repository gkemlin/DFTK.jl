using DFTK
using PyPlot
import Base.cbrt

### tool functions for computing the solution of u + u^3 = A*sin(x)

# extend cbrt to complex numbers
function cbrt_cplx(z)
    z = Complex(z)
    real(z) >= 0 ? z^(1//3) : -(-z)^(1//3)
end

# real solution of u + p*u^3 = b, using Cardan formulas
# https://fr.wikipedia.org/wiki/Méthode_de_Cardan
function cardan(b)
    # we are in the case where p = 1
    p = 1.0
    q = -b
    # the discriminant is R = -(4p^3 + 27q^2) <= 0 when p = 1
    R = -(4p^3 + 27q^2)
    v1 = cbrt_cplx((-q+sqrt(-R/27))/2)
    v2 = cbrt_cplx((-q-sqrt(-R/27))/2)
    v1 + v2
end

# u0 is the real solution of u + u^3 = A*sin(x) on [0,2π]
A = 10
function u0(x)
    cardan(A*sin(x))
end

### setting up the DFTK environment for Fourier Transform

# to kill numerical noise
seuil(x) = abs(x) < 1e-14 ? zero(x) : x

# weighted l2 spaces, that are supposed to lead the convergence of the fourier
# coefficients
w(k, B) = cosh(2 * k * B)

# setting up the basis for FFTs
a = 2π
lattice = a * [[1 0 0.]; [0 0 0]; [0 0 0]]
terms = [Kinetic()] # only used to create the basis, does not matter
model = Model(lattice; terms=terms, n_electrons=1,
              spin_polarization=:spinless)  # use "spinless electrons"
Ecut = 200000 # Fourier coefficients are well converged for this value of Ecut
basis = PlaneWaveBasis(model, Ecut, kgrid=(1, 1, 1))

### plotting Fourier coefficients and slopes

figure(1)
ftsize = 20
rc("font", size=ftsize, serif="Computer Modern")
rc("text", usetex=true)

# computing Fourier coefficients of u0
subplot(122)
u0r = ExternalFromReal(r->u0(r[1]))
u0G = r_to_G(basis, basis.kpoints[1], ComplexF64.(u0r(basis).potential))[:,1]
Gs = [abs(G[1]) for G in G_vectors_cart(basis.kpoints[1])][:]
semilogy(Gs, (seuil.(abs.(u0G))), "+", label="\$ u_{0,k} \$")
xlabel("\$ |k| \$", size=ftsize)

# plot theoretical slope, that is the size of the band of the complex plane
# on which u0 is analytic
# we compute it by solving R = 0 in the cardan formula, that is to say when the
# two complex solutions merge together. Computations give
# sin(z) = +/- sqrt(4/27)/A*im  <=>  z = +/- B*im
# with B = imag.(asin(sqrt(4/27)/A*im))
# /!\ note that asin(sqrt(4/27)/A*im) is imaginary as sqrt(4/27)/A*im is
# ==> u0 is analytic on the band of the complex plane of all z with imaginary
# part in absolute value < B
B = imag.(asin(sqrt(4/27)/A*im))
slope_B = [1 / (1000*sqrt(w(G[1],B))) for G in G_vectors_cart(basis.kpoints[1])]
semilogy(Gs, (seuil.(abs.(slope_B))), "k-", label="\$ {\\rm theoretical\\ slope\\ } 1/\\sqrt{\\cosh(2B|k|)} \$, \$B = $(round(B, digits=4))\$")
legend()

subplot(121)
rs = range(-π, π, length=500)
plot(rs, u0.(rs), label="\$ u_0 \$")
xlabel("\$ x \$", size=ftsize)
legend()

#  # compute slope with linear regression to compare with theoretical slope
#  u0_slope = []
#  G_slope = []
#  for (iG, G) in enumerate(G_vectors_cart(basis.kpoints[1]))
#      # ignore small |k| to compute the asymptotic slope
#      # every |k| which passes the test is counted twice (k and -k), but it does
#      # not matter to computer the slope as |u0_k| = |u0_-k|
#      if abs(u0G[iG]) > 1e-8 && abs(u0G[iG]) < 1e-6
#          append!(u0_slope, abs(u0G[iG]))
#          append!(G_slope, abs(G[1]))
#      end
#  end
#  # perform linear regression
#  _, Bs = -[ones(length(G_slope)) Float64.(G_slope)] \ log.(u0_slope)
#  slope_Bs = [1 / (1000*w(G[1],Bs)) for G in G_vectors_cart(basis.kpoints[1])]
#  plot(Gs, log.(seuil.(abs.(slope_Bs))), "r-", label="slope exp(-B|k|), B = $(Bs)")
#  legend()

# plot u0 in complex plane

function plot_complex_function(rs, is, f)
    res = f isa Function ? [f(x + im*y) for x in rs, y in is] : f
    res0 = f isa Function ? [f(im*y) for y in is] : f
    figure(2)
    rc("font", size=ftsize, serif="Computer Modern")
    rc("text", usetex=true)
    subplot(121)
    ux = imag.(res0)
    plot(is, ux, label="\$ {\\rm Im}(u_0({\\rm i} y)) \$")
    plot([B, B], [minimum(ux), maximum(ux)], "r--", label="\$ \\rm branching\\ point\\ \$")
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
end

rs = range(-0.1, 0.1, length=200)
is = range(-0.1, 0.1, length=200)
plot_complex_function(rs, is, z->u0(z))

