using DFTK
using PyPlot
using Polynomials
using ForwardDiff
using Printf

#### ε = 0
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

A = 0.5
B = imag(asin(√(4/27)im/A))
function u0(x)
    cardan(A*sin(x))
end

#### ε > 0
ε = 0.1
a = 2π
lattice = a * [[1 0 0.]; [0 0 0]; [0 0 0]]
C = 1/2
α = 2
n_electrons = 1  # Increase this for fun
f(r) = A * sin(r)
source_term = ExternalFromReal(r -> f(r[1]))

# solve for ε
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
basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid=(1, 1, 1))
scfres = DFTK.custom_direct_minimization(basis, source_term; tol=tol)
println(scfres.energies)

# basis function
function e(G, z, basis)
    exp(G[1] * z * im) / sqrt(basis.model.unit_cell_volume)
end
# cut function
seuil(x) = abs(x) < 1e-9 ? zero(x) : x
ψ = scfres.ψ[1][:, 1];    # first kpoint, all G components, first eigenvector
function u(z)
    φ = zero(ComplexF64)
    for (iG, G) in  enumerate(G_vectors(basis, basis.kpoints[1]))
        φ += seuil(ψ[iG]) * e(G, z, basis)
    end
    return φ
end
du(x) = ForwardDiff.derivative(y -> real(u(y)), x)


# EDO
g(X,t,ε) = [X[2]; (X[1]^3 - X[1])/ε]
f(X,t,ε) = [X[2]; (A*sinh(t) + X[1]^3 - X[1])/ε]

T = B+2
Nt = 100001
δt = T/(Nt-1)
Lt = 0:δt:T
t = 0

X0d = [0.; du(0)]
Xd = copy(X0d)
LXd = [X0d[1]]
LdXd = [X0d[2]]

for t in Lt[2:end]
    global Xd, ε
    Xd += δt * f(Xd,t,ε)
    append!(LXd, Xd[1])
    append!(LdXd, Xd[2])
end

figure()
ftsize = 30
rc("font", size=ftsize, serif="Computer Modern")
rc("text", usetex=true)

ψ0 = imag(u0.(Lt .* im))
plot(Lt, ψ0, label="\$ {\\rm Im}(u_0({\\rm i}y)) \$")
plot(Lt, LXd, "-", label="\$ \\psi_\\varepsilon(y)\\ \\psi'_\\varepsilon(0) = u_\\varepsilon'(0) \$ ")
plot(Lt, 0 .* Lt)
title("\$ \\varepsilon = $(ε)\\ \\mu = $(A) \$")
ylim(-0.1,4)
xlim(0,2)
xlabel("\$ y \$")

# computable lower bound
yid1 = findfirst(y->y>=1.5, LXd)
η1 = LXd[yid1] - 1
η1f = @sprintf("%.1f", η1)
y01 = Lt[yid1]
Y01 = [LXd[yid1]; LdXd[yid1]]
Y1 = copy(Y01)
LY1 = [Y01[1]]
LdY1 = [Y01[2]]
for t in Lt[2:end]
    global Y1, ε
    Y1 += δt * g(Y1,t,ε)
    append!(LY1, Y1[1])
    append!(LdY1, Y1[2])
end

w(y, y0, η) = (1 + 2/η + exp((y-y0)/sqrt(ε/2))) / (1 + 2/η - exp((y-y0)/sqrt(ε/2)))

plot(y01 .+ Lt, [w(y, y01, η1) for y in y01 .+ Lt],
     "+-", label="\$ \\xi_\\varepsilon(y)\\ \\eta = $(η1f) \$"; markevery=30, ms=10)
legend()
println("limit  $(√(ε/2)*log(1 + 2/η1)+y01))")

