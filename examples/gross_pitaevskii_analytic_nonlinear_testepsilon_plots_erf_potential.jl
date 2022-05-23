using PyPlot
using DFTK
using LinearAlgebra
using SpecialFunctions

### tool functions for computing the solution of u + u^3 = A*sin(x)
include("plotting_analytic.jl")

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
A = 100
B = imag(asin(√(4/27)/A * 1im))
function u0(x)
    cardan(A*sin(x))
end

## weighted l2 spaces of analytic functions
w(G, A) = exp(2*A*abs(G))

V(r) = erf(1000*cos(r))

a = 2π
lattice = a * [[1 0 0.]; [0 0 0]; [0 0 0]]

C = 1/2
α = 2

n_electrons = 1  # Increase this for fun

f(r) = A * sin(r)
source_term = ExternalFromReal(r -> f(r[1]))

ε_list = [0.1, 2]
x = nothing
basis = nothing

# basis function
function e(G, z, basis)
    exp(G[1] * z * im) / sqrt(basis.model.unit_cell_volume)
end

# cut function
seuil(x) = x

# L^2 and H^s norms
function norm_L2(basis, u)
    Gs = [abs(G[1]) for G in G_vectors(basis, basis.kpoints[1])][:]
    return sqrt(sum(seuil.(abs.(u)).^2))
end
function norm_Hs(basis, u, s)
    Gs = [abs(G[1]) for G in G_vectors(basis, basis.kpoints[1])][:]
    return sqrt(sum((1 .+ Gs.^2).^s .* seuil.(abs.(u)).^2))
end

# u0 solution for ε = 0
u0r = ExternalFromReal(r->u0(r[1]))
u0G = nothing
save0 = false

figure(1)
ftsize = 30
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

    Ecut = 10000000
    tol = 1e-15
    global basis
    basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid=(1, 1, 1))
    scfres = self_consistent_field(basis; tol=tol)
    println(scfres.energies)

    # ## Internals
    # We use the opportunity to explore some of DFTK internals.
    #
    # Extract the converged density and the obtained wave function:
    ρ = real(scfres.ρ)[:, 1, 1, 1]  # converged density, first spin component
    ψ = scfres.ψ[1][:, 1];    # first kpoint, all G components, first eigenvector
    Hψ = scfres.ham.blocks[1] * ψ
    Hψr = G_to_r(basis, basis.kpoints[1], Hψ)[:,1,1]
    println("|Hψ-f| = ", norm(Hψr - source_term(basis).potential_values[:,1,1]))

    global save0, u0G
    if !save0
        u0G = r_to_G(basis, basis.kpoints[1], ComplexF64.(u0r(basis).potential_values))[:,1]
        println("|u0-ψ| = ", norm_L2(basis, u0G-ψ))
        println(norm(u0G-ψ))
        save0 = true
    end

    # plots
    global x
    x = a * vec(first.(DFTK.r_vectors(basis)))
    ψr = G_to_r(basis, basis.kpoints[1], ψ)[:, 1, 1]

    if ε >= -1
        figure(2)
        Gs = [abs(G[1]) for G in G_vectors(basis, basis.kpoints[1])][:]
        GGs = Gs[2:div(length(Gs)+1,2)]
        nG = length(GGs)
        ψG = [ψ[2*k] for k =1:div(nG,2)]
        GGGs = [GGs[2*k] for k =1:div(nG,2)]
        ψGn = ψG[2:end]
        subplot(121)
        semilogy(GGGs, (seuil.(abs.(ψG))), "+", label="\$ \\varepsilon = $(ε) \$")
        subplot(122)
        plot(GGGs[2:end], log.(abs.( seuil.(ψGn) ./ seuil.(ψG[1:end-1] ))), "+", label="\$ \\varepsilon = $(ε) \$")
        function u(z)
            φ = zero(ComplexF64)
            for (iG, G) in  enumerate(G_vectors(basis, basis.kpoints[1]))
                φ += seuil(ψ[iG]) * e(G, z, basis)
            end
            return φ
        end
        figure(1)
        subplot(121)
        is = range(-0.1, 0.1, length=200)
        plot(x, real.(ψr), label="\$ \\varepsilon = $(ε) \$")
        figure(3)
        semilogy(GGGs, (seuil.(abs.(ψG))), "+", label="\$ \\varepsilon = $(ε) \$")
    end

    #  if ε == 1e-5
    #      figure(6)
    #      rs = range(-0.05, 0.05, length=500)
    #      is = range(-0.05, 0.3, length=500)
    #      plot_complex_function(rs, is, z->u(z))
    #      plot(0, B, "ro")
    #  end

end

figure(3)
xlabel("\$ |k| \$")
legend()

figure(2)
subplot(121)
xlabel("\$ |k| \$")
legend()
subplot(122)
xlabel("\$ |k| \$")
legend()

