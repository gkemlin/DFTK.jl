using DFTK
using PyPlot
using Polynomials
using ForwardDiff

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

A = 10
B = imag(asin(√(4/27)im/A))
function u0(x)
    cardan(A*sin(x))
end

#### ε > 0
ε = 0.01308
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
global basis
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
    for (iG, G) in  enumerate(G_vectors(basis.kpoints[1]))
        φ += seuil(ψ[iG]) * e(G, z, basis)
    end
    return φ
end
du(x) = ForwardDiff.derivative(y -> real(u(y)), x)


# EDO
#  g(X,t,ε) = [X[2]; (X[1]^3)/ε]
g(X,t,ε) = [X[2]; (X[1]^3 - X[1])/ε]
#  g(X,t,ε) = [X[2]; (A*sinh(t) - X[1])/ε]
f(X,t,ε) = [X[2]; (A*sinh(t) + X[1]^3 - X[1])/ε]

T = 15
Nt = 100001
δt = T/(Nt-1)
Lt = 0:δt:T
t = 0

X0 = [0.; du(0)]
Y0 = [0.; du(0)]
X = copy(X0)
Y = copy(Y0)
LX = [X0[1]]
LdX = [X0[2]]
LY = [Y0[1]]
LdY = [Y0[2]]

for t in Lt[2:end]
    global X, Y, ε
    X += δt * f(X,t,ε)
    Y += δt * g(Y,t,ε)
    append!(LX, X[1])
    append!(LdX, X[2])
    append!(LY, Y[1])
    append!(LdY, Y[2])
end

figure()
ftsize = 30
rc("font", size=ftsize, serif="Computer Modern")
rc("text", usetex=true)

subplot(121)
plot(Lt, imag(u0.(Lt .* im)), label="\$ {\\rm Im}(u_0({\\rm i}y)) \$")
plot(Lt, LX, "+", label="\$ \\psi_\\varepsilon(y) \$", markevery=50, ms=10)
plot(Lt, LY, "+", label="\$ \\phi_\\varepsilon(y) \$", markevery=50, ms=10)
plot(Lt, [1/√3 for t in Lt], "--", label="\$ 1/\\sqrt{3} \$")
plot(Lt, imag(u.(Lt .* im)), label="\$ {\\rm Im }(u_\\varepsilon({\\rm i}y)) \$")
xlim(0,0.2)
ylim(0,1)
xlabel("\$ y \$")
#  plot([0, B], [1/√3, 1/√3], "--", label="1/√3")
#  plot([B, B], [0, 1/√3], "--", label="t=B")
#  plot(Lt, Y0[2]*Lt)
#  plot(Lt, Y0[2]*Lt + Y0[2].^3 * 6 ./ factorial(5) .* (Lt).^5)
#  plot(Lt, Y0[2]*Lt + Y0[2].^3 * 6 ./ factorial(5) .* (Lt).^5 + 36 .* Y0[2].^5 ./ factorial(9) .* (Lt).^9 )
legend()

subplot(122)
plot(Lt, imag(u0.(Lt .* im)), label="\$ {\\rm Im}(u_0({\\rm i}y)) \$")
plot(Lt, LX, "+", label="\$ \\psi_\\varepsilon(y) \$", markevery=50, ms=10)
plot(Lt, LY, "+", label="\$ \\phi_\\varepsilon(y) \$"; markevery=200, ms=10)
plot(Lt, [1/√3 for t in Lt], "--", label="\$ 1/\\sqrt{3} \$")
plot(Lt, [1 for t in Lt], "--", label="\$ 1 \$")
ylim(-1,10)
#  xlim(0,7)
xlabel("\$ y \$")
legend()

STOP


figure()
# dev0 = (LY ./ (Y0[2]*Lt))
dev0 = LY
pol_fit = fit(Lt, dev0, 10)
plot(Lt, dev0)
plot(Lt, pol_fit.(Lt))


STOP

figure()
#  plot(Lt, LdX, label="u'(t)")
plot(Lt, LdY, label="y'(t)")
legend()
