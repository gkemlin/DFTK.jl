using PyPlot
using DFTK
using DoubleFloats
using GenericLinearAlgebra
using ForwardDiff

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

### solution for ε = 0
# u0 is the real solution of u + u^3 = A*sin(x) on [0,2π]
A = 10
u0(x) = cardan(A*sin(x))
du0(x) = ForwardDiff.derivative(y -> real(u0(y)), x)


### solution for ε > 0
ε = 0

a = 2π
lattice = a * [[1 0 0.]; [0 0 0]; [0 0 0]]
C = 1/2
α = 2
n_electrons = 1  # Increase this for fun
f(r) = A * sin(r)
source_term = ExternalFromReal(r -> f(r[1]))

println("---------------------------------")
println("ε = $(ε)")
V(r) = 1
terms = [Kinetic(2*ε),
         ExternalFromReal(r -> V(r[1])),
         PowerNonlinearity(C, α),
        ]
model = Model(lattice; n_electrons=n_electrons, terms=terms,
              spin_polarization=:spinless)  # use "spinless electrons"

Ecut = 1000000
tol = 1e-15
basis = PlaneWaveBasis(model, Ecut, kgrid=(1, 1, 1))
scfres = DFTK.custom_direct_minimization(basis, source_term; tol=tol)
println(scfres.energies)

# check resolution
# Extract the converged density and the obtained wave function:
ρ = real(scfres.ρ)[:, 1, 1, 1]  # converged density, first spin component
ψ = scfres.ψ[1][:, 1];    # first kpoint, all G components, first eigenvector
Hψ = scfres.ham.blocks[1] * ψ
Hψr = G_to_r(basis, basis.kpoints[1], Hψ)[:,1,1]
println("|Hψ-f| = ", norm(Hψr - source_term(basis).potential[:,1,1]))

# basis function
function e(G, z, basis)
    exp(G[1] * z * im) / sqrt(basis.model.unit_cell_volume)
end
# cut function
seuil(x) = abs(x) < 1e-10 ? zero(x) : x


# analytical expension
function u(z)
    uu = zero(ComplexF64)
    for (iG, G) in  enumerate(G_vectors(basis.kpoints[1]))
        uu += seuil(ψ[iG]) * e(G, z, basis)
    end
    uu
end
du(x) = ForwardDiff.derivative(y -> real(u(y)), x)

### solving the EDO in φ(y) = u(iy), with X(y) = [φ(y), φ'(y)]

ymax = 0.1
ylength = 1001
is = range(0, ymax, length=ylength)
δy = is[2]-is[1]
φ_l = zeros(ComplexF64, 2, ylength)
φ_l[:, 1] = [u(0), du(0)im]
function g(y, X)
    @assert size(X) == (2,)
    [X[2], (f(y) - X[1] - X[1]^3) / ε]
end

for i = 1:(ylength-1)
    φ_l[:,i+1] = φ_l[:,i] + δy*g(i*δy*im, φ_l[:,i])
end


### plots

x = a * vec(first.(DFTK.r_vectors(basis)))
ψr = G_to_r(basis, basis.kpoints[1], ψ)[:, 1, 1]

figure()
ftsize = 20
rc("font", size=ftsize, serif="Computer Modern")
rc("text", usetex=true)

subplot(121)
plot(x, u0.(x), label="\$ \\varepsilon = 0 \$")
plot(x, real.(ψr), label="\$ \\varepsilon = $(ε) \$")
xlabel("\$ x \$", size = ftsize)
legend()

subplot(122)
plot(is, imag.(u0.(is .* im)), label="\$ {\\rm Im}(u_0({\\rm i} y)) \$")
plot(is, imag.(u.(is .* im)), label="\$ {\\rm Im}(u_\\varepsilon({\\rm i} y))\\ \\varepsilon = $(ε)\\; {\\rm ae} \$")
plot(is, imag.(φ_l[1,:]), label="\$ {\\rm Im}(\\varphi_\\varepsilon({\\rm i} y))\\ \\varepsilon = $(ε)\\; {\\rm EDO} \$")
#  plot(is, imag.(φ_l[2,:]), label="\$ {\\rm Im}(\\varphi'_\\varepsilon({\\rm i} y))\\ \\varepsilon = $(ε)\\; {\\rm EDO} \$")
plot(is, [1/√3 for i in is], "k--")
B = imag(asin(√(4/27)/A * 1im))
plot([B,B], [0, 0.8], "r--")
legend()

