using PyPlot
using DFTK
using DoubleFloats
using GenericLinearAlgebra

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

## weighted l2 spaces of analytic functions
w(G, A) = exp(2*A*abs(G))

V(r) = 1

a = 2π
lattice = a * [[1 0 0.]; [0 0 0]; [0 0 0]]

C = 1/2
α = 2

n_electrons = 1  # Increase this for fun

f(r) = A * sin(r)
source_term = ExternalFromReal(r -> f(r[1]))

#  ε_list = [1e-16, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1]
ε_list = [0, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1]
#  ε_list = [0]
x = nothing
basis = nothing

# basis function
function e(G, z, basis)
    exp(G[1] * z * im) / sqrt(basis.model.unit_cell_volume)
end

# cut function
seuil(x) = abs(x) < 1e-9 ? zero(x) : x

# L^2 and H^s norms
function norm_L2(basis, u)
    Gs = [abs(G[1]) for G in G_vectors(basis.kpoints[1])][:]
    return sqrt(sum(seuil.(abs.(u)).^2))
end
function norm_Hs(basis, u, s)
    Gs = [abs(G[1]) for G in G_vectors(basis.kpoints[1])][:]
    return sqrt(sum((1 .+ Gs.^2).^s .* seuil.(abs.(u)).^2))
end

# u0 solution for ε = 0
u0r = ExternalFromReal(r->u0(r[1]))
u0G = nothing
save0 = false

cvg_L2norm = []
cvg_H1norm = []
cvg_H2norm = []
L2norm = []
H1norm = []
H11norm = []
H2norm = []

figure(1)
ftsize = 20
rc("font", size=ftsize, serif="Computer Modern")
rc("text", usetex=true)

for ε in ε_list

    println("---------------------------------")
    println("ε = $(ε)")
    terms = [Kinetic(2*ε),
             ExternalFromReal(r -> V(r[1])),
             PowerNonlinearity(C, α),
            ]
    model = Model(lattice; n_electrons=n_electrons, terms=terms,
                  spin_polarization=:spinless)  # use "spinless electrons"

    Ecut = 1000000
    tol = 1e-15
    global basis
    basis = PlaneWaveBasis(model, Ecut, kgrid=(1, 1, 1))
    scfres = DFTK.custom_direct_minimization(basis, source_term; tol=tol)
    println(scfres.energies)

    # ## Internals
    # We use the opportunity to explore some of DFTK internals.
    #
    # Extract the converged density and the obtained wave function:
    ρ = real(scfres.ρ)[:, 1, 1, 1]  # converged density, first spin component
    ψ = scfres.ψ[1][:, 1];    # first kpoint, all G components, first eigenvector
    Hψ = scfres.ham.blocks[1] * ψ
    Hψr = G_to_r(basis, basis.kpoints[1], Hψ)[:,1,1]
    println("|Hψ-f| = ", norm(Hψr - source_term(basis).potential[:,1,1]))

    global save0, u0G
    if !save0
        u0G = r_to_G(basis, basis.kpoints[1], ComplexF64.(u0r(basis).potential))[:,1]
        println("|u0-ψ| = ", norm_L2(basis, u0G-ψ))
        println(norm(u0G-ψ))
        save0 = true
    end

    # norms
    append!(cvg_L2norm, norm_L2(basis, u0G-ψ))
    append!(cvg_H1norm, norm_Hs(basis, u0G-ψ, 1))
    append!(cvg_H2norm, norm_Hs(basis, u0G-ψ, 2))
    append!(L2norm, norm_L2(basis, ψ))
    append!(H1norm, norm_Hs(basis, ψ, 1))
    append!(H2norm, norm_Hs(basis, ψ, 2))
    append!(H11norm, norm_Hs(basis, ψ, 1.1))

    # plots
    global x
    x = a * vec(first.(DFTK.r_vectors(basis)))
    ψr = G_to_r(basis, basis.kpoints[1], ψ)[:, 1, 1]

    if ε >= -1
        figure(1)
        subplot(121)
        plot(x, real.(ψr), label="\$ \\varepsilon = $(ε) \$")
        figure(3)
        subplot(122)
        Gs = [abs(G[1]) for G in G_vectors(basis.kpoints[1])][:]
        semilogy(Gs, (seuil.(abs.(ψ))), "+", label="\$ \\varepsilon = $(ε) \$")
        function u(z)
            φ = zero(ComplexF64)
            for (iG, G) in  enumerate(G_vectors(basis.kpoints[1]))
                φ += seuil(ψ[iG]) * e(G, z, basis)
            end
            return φ
        end
        figure(3)
        subplot(121)
        is = range(-0.1, 0.1, length=200)
        plot(is, imag.(u.(is .* im)), label="\$ {\\rm Im}(u_\\varepsilon({\\rm i} y))\\ \\varepsilon = $(ε) \$")
    end

end

## tests
figure(1)
subplot(121)
plot(x, u0.(x), "--", label="\$ \\varepsilon = 0 \$")
#  plot(x, f.(x), "k--", label="f")
xlabel("\$ x \$", size = ftsize)
legend()

figure(1)
subplot(122)
loglog(ε_list, cvg_L2norm, "x-", label="\$ ||u_0-u_\\varepsilon||_{{\\rm L}^2_\\sharp} \$")
loglog(ε_list, cvg_H1norm, "x-", label="\$ ||u_0-u_\\varepsilon||_{{\\rm H}^1_\\sharp} \$")
loglog(ε_list, cvg_H2norm, "x-", label="\$ ||u_0-u_\\varepsilon||_{{\\rm H}^2_\\sharp} \$")
legend()
xlabel("\$ \\varepsilon \$", size=ftsize)

figure(3)
subplot(122)
Gs = [abs(G[1]) for G in G_vectors(basis.kpoints[1])][:]
semilogy(Gs, (seuil.(abs.(u0G))), "+", label="\$ \\varepsilon = 0 \$")
xlim(-15, 315)
xlabel("\$ |k| \$", size = ftsize)
legend()

figure(3)
subplot(121)
is = range(-0.1, 0.1, length=1000)
ylim(-1,1)
plot(is, imag.(u0.(is .* im)), label="\$ {\\rm Im}(u_0({\\rm i} y)) \$")
plot(is, [1/√3 for i in is], "k--")
plot(is, [-1/√3 for i in is], "k--")
B = imag(asin(√(4/27)/A * 1im))
plot([B,B], [-0.8, 0.8], "r--")
plot([-B,-B], [-0.8, 0.8], "r--")
legend(loc="lower right")

figure(2)
subplot(121)
is = range(-1, 1, length=1000)
plot(is, imag.(u0.(is .* im)), label="\$ {\\rm Im}(u_0({\\rm i} y)) \$")
plot(is, [1/√3 for i in is], label="\$ \\frac{1}{\\sqrt{3}}\$" )
plot(is, [-1/√3 for i in is], label="\$ - \\frac{1}{\\sqrt{3}}\$" )
plot(is, [1 for i in is], label="\$ 1 \$" )
plot(is, [-1 for i in is], label="\$ - 1 \$" )
legend()

figure(2)
subplot(122)
is = range(-0.1, 0.1, length=1000)
plot(is, imag.(u0.(is .* im)), label="\$ {\\rm Im}(u_0({\\rm i} y)) \$")
plot(is, [1/√3 for i in is], "k--")
plot(is, [-1/√3 for i in is], "k--")
B = imag(asin(√(4/27)/A * 1im))
plot([B,B], [-0.8, 0.8], "r--")
plot([-B,-B], [-0.8, 0.8], "r--")
legend()
