using PyPlot
using DFTK
using DoubleFloats
using GenericLinearAlgebra

include("solving_nodelta.jl")

## weighted l2 spaces of analytic functions
w(G, A) = cosh(2*A*G)

V(r) = 1

a = 2π
lattice = a * [[1 0 0.]; [0 0 0]; [0 0 0]]

C = 1
α = 2

n_electrons = 1  # Increase this for fun

A = 10
f(r) = A * sin(r)
source_term = ExternalFromReal(r -> f(r[1]))

ε_list = [0.01, 1]
x = nothing
basis = nothing

# basis function
function e(G, z, basis)
    exp(G[1] * z * im) / sqrt(basis.model.unit_cell_volume)
end

# cut function
seuil(x) = abs(x) < 1e-8 ? zero(x) : x

for ε in ε_list

    println("---------------------------------")
    println("ε = $(ε)")
    terms = [Kinetic(2*ε),
             ExternalFromReal(r -> V(r[1])),
             PowerNonlinearity(C, α),
            ]
    model = Model(lattice; n_electrons=n_electrons, terms=terms,
                  spin_polarization=:spinless)  # use "spinless electrons"

    Ecut = 1000
    tol = 1e-12
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
    println(norm(2*Hψr - source_term(basis).potential[:,1,1]))

    # Transform the wave function to real space and fix the phase:
    ψr = G_to_r(basis, basis.kpoints[1], ψ)[:, 1, 1]

    # plots
    global x
    x = a * vec(first.(DFTK.r_vectors(basis)))

    figure(1)
    plot(x, real.(2*ψr), label="ε = $(ε)")

    figure(2)
    Gs = [abs(G[1]) for G in G_vectors_cart(basis.kpoints[1])][:]
    semilogy(Gs, seuil.(abs.(ψ)), "o", label="ε = $(ε)")

    figure()
    suptitle("analytical expansion ε = $(ε)")
    function u(z)
        φ = zero(ComplexF64)
        for (iG, G) in  enumerate(G_vectors_cart(basis.kpoints[1]))
            φ += 2 * seuil(ψ[iG]) * e(G, z, basis)
        end
        return φ
    end
    rs = range(-π, π, length=500)
    is = range(-2, 2, length=500)
    plot_complex_function(rs, is, z->u(z))
end

include("solving_nodelta.jl")

figure(1)
title("u_ε on [0, 2π]")
plot(x, u.(x), "r--", label="ε = 0")
plot(x, f.(x), "k--", label="f")
legend()

figure(2)
ur = ExternalFromReal(r->u(r[1]))
uG = r_to_G(basis, basis.kpoints[1], ComplexF64.(ur(basis).potential))[:,1]
Gs = [abs(G[1]) for G in G_vectors_cart(basis.kpoints[1])][:]
semilogy(Gs, seuil.(abs.(uG)), "o", label="ε = 0")

# plot different w_G(B)
B_list = 1:0.2:2
for B in B_list
    ref_B = [G[1] == 0 ? zero(G[1]) : 1 / sqrt(w(G[1],B))
             for G in G_vectors_cart(basis.kpoints[1])] # ref slope in Fourier
    semilogy(Gs, seuil.(abs.(ref_B)), "x-", label="1 / (√w_B(k)) B = $(B)")
end
xlabel("k")
legend()

figure()
suptitle("analytical expansion ε = 0")
rs = range(-π, π, length=500)
is = range(-2, 2, length=500)
plot_complex_function(rs, is, z->u(z))

