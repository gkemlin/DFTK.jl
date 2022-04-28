using PyPlot
using DFTK
using LinearAlgebra
using DoubleFloats
using GenericLinearAlgebra
using SpecialFunctions

## solve 1D GP eigenvalue problem

a = 2π
lattice = a * [[1 0 0.]; [0 0 0]; [0 0 0]]

V(r) = erf(1000*cos(r))
C = 50
α = 2

n_electrons = 1  # Increase this for fun


# cut function
#  seuil(x) = abs(x) < 1e-12 ? zero(x) : x
seuil(x) = x

for ε in [0.1, 5]

    println("---------------------------------")
    println("ε = $(ε)")
    terms = [Kinetic(2*ε),
             ExternalFromReal(r -> V(r[1])),
             PowerNonlinearity(C, α),
            ]
    model = Model(Array{Float64}(lattice); n_electrons=n_electrons, terms=terms,
                  spin_polarization=:spinless)  # use "spinless electrons"

    Ecut = 10000000
    tol = 1e-12
    basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid=(1, 1, 1))
    scfres = self_consistent_field(basis; tol=tol, maxiter=200, damping=0.1,
                                   is_converged=DFTK.ScfConvergenceDensity(tol))
    println(scfres.energies)

    # ## Internals
    # We use the opportunity to explore some of DFTK internals.
    #
    # Extract the converged density and the obtained wave function:
    ψ = scfres.ψ[1][:, 1];    # first kpoint, all G components, first eigenvector
    Hψ = scfres.ham.blocks[1] * ψ
    println("|Hψ-λψ| = ", norm(Hψ - scfres.eigenvalues[1][1].*ψ))

    # plots
    x = a * vec(first.(DFTK.r_vectors(basis)))
    ψr = G_to_r(basis, basis.kpoints[1], ψ)[:, 1, 1]

    figure(1)
    ftsize = 30
    rc("font", size=ftsize, serif="Computer Modern")
    rc("text", usetex=true)
    Gs = [abs(G[1]) for G in G_vectors(basis, basis.kpoints[1])][:]
    GGs = Gs[2:div(length(Gs)+1,2)]
    nG = length(GGs)
    ψG = [ψ[2k] for k =1:div(nG,2)]
    GGGs = [GGs[2k] for k =1:div(nG,2)]
    ψGn = ψG[2:end]
    subplot(121)
    semilogy(GGGs, (seuil.(abs.(ψG))), "+", label="\$ \\varepsilon = $(ε) \$")
    legend()
    subplot(122)
    plot(GGGs[2:end], log.(abs.( seuil.(ψGn) ./ seuil.(ψG[1:end-1] ))), "+", label="\$ \\varepsilon = $(ε) \$")
    function u(z)
        φ = zero(ComplexF64)
        for (iG, G) in  enumerate(G_vectors(basis, basis.kpoints[1]))
            φ += seuil(ψ[iG]) * e(G, z, basis)
        end
        return φ
    end
    legend()

    figure(2)
    plot(x, abs2.(ψr), label="\$ \\varepsilon = $(ε) \$")
    legend()
end
