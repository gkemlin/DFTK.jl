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

T = B+2
Nt = 100001
δt = T/(Nt-1)
Lt = 0:δt:T
t = 0

X0 = [0.; 0.]
X0d = [0.; du(0)]
X = copy(X0)
Xd = copy(X0d)
LX = [X0[1]]
LXd = [X0d[1]]
LdX = [X0[2]]
LdXd = [X0d[2]]

for t in Lt[2:end]
    global X, Xd, ε
    X += δt * f(X,t,ε)
    append!(LX, X[1])
    append!(LdX, X[2])
    Xd += δt * f(Xd,t,ε)
    append!(LXd, Xd[1])
    append!(LdXd, Xd[2])
end

figure()
ftsize = 30
rc("font", size=ftsize, serif="Computer Modern")
rc("text", usetex=true)

figure(2)
#  subplot(121)
ψ0 = imag(u0.(Lt .* im))
plot(Lt, ψ0, label="\$ {\\rm Im}(u_0({\\rm i}y)) \$")
plot(Lt, LXd, "-", label="\$ \\psi_\\varepsilon(y)\\ \\psi'_\\varepsilon(0) = u_\\varepsilon'(0) \$ ")
#  plot(Lt, LdXd, "-", label="\$ \\psi'_\\varepsilon(y)\\ \\psi'_\\varepsilon(0) = u_\\varepsilon'(0) \$ ")
#  plot(Lt, [LdXd[1] for t in Lt], "-", label="\$ u_\\varepsilon'(0) \$ ")
#  plot(Lt, LX, "-", label="\$ \\psi_\\varepsilon(y)\\ \\psi'_\\varepsilon(0) = 0 \$ ")
#  plot(Lt, LdX, "-", label="\$ \\psi'_\\varepsilon(y)\\ \\psi'_\\varepsilon(0) = 0 \$ ")
plot(Lt, 0 .* Lt)
title("\$ \\varepsilon = $(ε)\\ \\mu = $(A) \$")
ylim(-0.1,1)
xlim(0,T)
xlabel("\$ y \$")
legend()
println(prod(ψ0 .≥ LXd))

#  subplot(122)
#  plot(Lt, imag(u0.(Lt .* im)), label="\$ {\\rm Im}(u_0({\\rm i}y)) \$")
#  plot(Lt, LXd, "-", label="\$ \\psi_\\varepsilon(y)\\ \\psi'_\\varepsilon(0) = u_\\varepsilon'(0) \$ ")
#  plot(Lt, LX, "-", label="\$ \\psi_\\varepsilon(y)\\ \\psi'_\\varepsilon(0) = 0 \$ ")
#  plot(Lt, 0 .* Lt)
#  title("\$ \\varepsilon = $(ε)\\ \\mu = $(A) \$")
#  xlabel("\$ y \$")
#  legend()


STOP

# computable lower bound
yid1 = findfirst(y->y>=2, LX)
η1 = LX[yid1] - 1
η1f = @sprintf("%.2f", η1)
y01 = Lt[yid1]
Y01 = [LX[yid1]; LdX[yid1]]
Y1 = copy(Y01)
LY1 = [Y01[1]]
LdY1 = [Y01[2]]
for t in Lt[2:end]
    global Y1, ε
    Y1 += δt * g(Y1,t,ε)
    append!(LY1, Y1[1])
    append!(LdY1, Y1[2])
end

yid2 = findfirst(y->y>=1.5, LX)
η2 = LX[yid2] - 1
η2f = @sprintf("%.2f", η2)
y02 = Lt[yid2]
Y02 = [LX[yid2]; LdX[yid2]]
Y2 = copy(Y02)
LY2 = [Y02[1]]
LdY2 = [Y02[2]]
for t in Lt[2:end]
    global Y2, ε
    Y2 += δt * g(Y2,t,ε)
    append!(LY2, Y2[1])
    append!(LdY2, Y2[2])
end

w(y, y0, η) = (1 + 2/η + exp((y-y0)/sqrt(ε/2))) / (1 + 2/η - exp((y-y0)/sqrt(ε/2)))

#  subplot(121)
#  plot(Lt, imag(u0.(Lt .* im)), label="\$ {\\rm Im}(u_0({\\rm i}y)) \$")
#  plot(Lt, LX, "+", label="\$ \\psi_\\varepsilon(y) \$", markevery=50, ms=10)
#  plot(Lt, LY, "+", label="\$ \\phi_\\varepsilon(y) \$", markevery=50, ms=10)
#  plot(Lt, [1/√3 for t in Lt], "--", label="\$ 1/\\sqrt{3} \$")
#  plot(Lt, imag(u.(Lt .* im)), label="\$ {\\rm Im }(u_\\varepsilon({\\rm i}y)) \$")
#  xlim(0,0.2)
#  ylim(0,1)
#  xlabel("\$ y \$")
#  #  plot([0, B], [1/√3, 1/√3], "--", label="1/√3")
#  #  plot([B, B], [0, 1/√3], "--", label="t=B")
#  #  plot(Lt, Y0[2]*Lt)
#  #  plot(Lt, Y0[2]*Lt + Y0[2].^3 * 6 ./ factorial(5) .* (Lt).^5)
#  #  plot(Lt, Y0[2]*Lt + Y0[2].^3 * 6 ./ factorial(5) .* (Lt).^5 + 36 .* Y0[2].^5 ./ factorial(9) .* (Lt).^9 )
#  legend()

subplot(121)
#  plot(Lt, imag(u0.(Lt .* im)), label="\$ {\\rm Im}(u_0({\\rm i}y)) \$")
plot(Lt, LX, "+-", label="\$ \\psi_\\varepsilon(y) \$ (Euler exp)", markevery=30, ms=10)
plot(y01 .+ Lt, LY1, "x-", label="\$ \\phi_\\varepsilon(y) \$ \$ \\eta = $(η1f)\$ "; markevery=30, ms=10)
plot(y02 .+ Lt, LY2, "x-", label="\$ \\phi_\\varepsilon(y) \$ \$ \\eta = $(η2f)\$ "; markevery=30, ms=10)
plot(y01 .+ Lt, [w(y, y01, η1) for y in y01 .+ Lt],
     "+-", label="\$ \\xi_\\varepsilon(y)\\ \\eta = $(η1f) \$"; markevery=30, ms=10)
plot(y02 .+ Lt, [w(y, y02, η2) for y in y02 .+ Lt],
     "+-", label="\$ \\xi_\\varepsilon(y)\\ \\eta = $(η2f) \$"; markevery=30, ms=10)
#  plot(Lt, [1/√3 for t in Lt], "--", label="\$ 1/\\sqrt{3} \$")
#  plot(Lt, [1 for t in Lt], "--", label="\$ 1 \$")
ylim(-1,10)
xlim(0,√(ε/2)*log(1 + 2/η2)+y02)
xlabel("\$ y \$")
legend(loc="upper left")



subplot(122)
#  plot(Lt, imag(u0.(Lt .* im)), label="\$ {\\rm Im}(u_0({\\rm i}y)) \$")
plot(Lt, LX, "+-", label="\$ \\psi_\\varepsilon(y) \$ (Euler exp)", markevery=10, ms=10)
plot(y01 .+ Lt, LY1, "x-", label="\$ \\phi_\\varepsilon(y) \$ \$ \\eta = $(η1f)\$ "; markevery=30, ms=10)
plot(y02 .+ Lt, LY2, "x-", label="\$ \\phi_\\varepsilon(y) \$ \$ \\eta = $(η2f)\$ "; markevery=30, ms=10)
plot(y01 .+ Lt, [w(y, y01, η1) for y in y01 .+ Lt],
     "+-", label="\$ \\xi_\\varepsilon(y)\\ \\eta = $(η1f) \$"; markevery=30, ms=10)
plot(y02 .+ Lt, [w(y, y02, η2) for y in y02 .+ Lt],
     "+-", label="\$ \\xi_\\varepsilon(y)\\ \\eta = $(η2f) \$"; markevery=30, ms=10)
#  plot(Lt, [1/√3 for t in Lt], "--", label="\$ 1/\\sqrt{3} \$")
#  plot(Lt, [1 for t in Lt], "--", label="\$ 1 \$")
ylim(-1,100)
xlim(0,√(ε/2)*log(1 + 2/η2)+y02)
xlabel("\$ y \$")
legend(loc="upper left")

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
