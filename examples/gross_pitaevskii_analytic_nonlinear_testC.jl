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

C_list = [0, 1/200, 1/20, 1/2, 5, 50]
α = 2

n_electrons = 1  # Increase this for fun

A = 10
f(r) = A * sin(r)
source_term = ExternalFromReal(r -> f(r[1]))

ε = 1
x = nothing
basis = nothing

# basis function
function e(G, z, basis)
    exp(G[1] * z * im) / sqrt(basis.model.unit_cell_volume)
end

# cut function
seuil(x) = abs(x) < 1e-8 ? zero(x) : x

for C in C_list

    println("---------------------------------")
    println("C = $(2*C)")
    terms = [Kinetic(2*ε),
             ExternalFromReal(r -> V(r[1])),
             PowerNonlinearity(C, α),
            ]
    model = Model(lattice; n_electrons=n_electrons, terms=terms,
                  spin_polarization=:spinless)  # use "spinless electrons"

    Ecut = 40000
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
    println("|Hψ-f| = ", norm(Hψr - source_term(basis).potential[:,1,1]))

    # Transform the wave function to real space and fix the phase:
    ψr = G_to_r(basis, basis.kpoints[1], ψ)[:, 1, 1]

    # plots
    global x
    x = a * vec(first.(DFTK.r_vectors(basis)))

    figure(3)
    plot(x, real.(ψr), label="C = $(2*C)")
    legend()

    figure(4)
    Gs = [abs(G[1]) for G in G_vectors(basis.kpoints[1])][:]
    semilogy(Gs, seuil.(abs.(ψ)), "+", label="C = $(2*C)")
    legend()

end

