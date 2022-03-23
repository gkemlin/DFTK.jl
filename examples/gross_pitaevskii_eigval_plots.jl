using PyPlot
using DFTK
using LinearAlgebra
using DoubleFloats
using GenericLinearAlgebra

## solve 1D GP eigenvalue problem

a = 2π
lattice = a * [[1 0 0.]; [0 0 0]; [0 0 0]]

V(r) = cos(r)
C = 10
α = 2

n_electrons = 1  # Increase this for fun


# cut function
#  seuil(x) = abs(x) < 1e-12 ? zero(x) : x
seuil(x) = x

for ε in [0.1, 0.5, 1, 2]

    println("---------------------------------")
    println("ε = $(ε)")
    terms = [Kinetic(2*ε),
             ExternalFromReal(r -> V(r[1])),
             PowerNonlinearity(C, α),
            ]
    model = Model(Array{Double64}(lattice); n_electrons=n_electrons, terms=terms,
                  spin_polarization=:spinless)  # use "spinless electrons"

    Ecut = 10000000
    tol = 1e-28
    basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid=(1, 1, 1))
    scfres = self_consistent_field(basis; tol=tol, maxiter=200, damping=0.5)# is_converged=DFTK.ScfConvergenceDensity(tol))
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
    Gs = [G[1] for G in G_vectors(basis, basis.kpoints[1])][:]
    Gs = Gs[1:div(end,2)]
    nG = length(Gs)
    nG2 = div(nG,2) - 1
    ψ2kp1 = [ψ[2k+1] for k in 1:nG2]
    ψ2k = [ψ[2k] for k in 1:nG2]
    ψ2km1 = [ψ[2k-1] for k in 1:nG2]

    title("\$ u_k \$")
    semilogy((seuil.(abs.(ψ[1:nG2]))), "+", label="\$ \\varepsilon = $(ε) \$")
    xlabel("\$ |k| \$")
    xlim(0,20)
    legend()

    #  subplot(2,2,2)
    #  title("\$ \\log \\left( \\frac{|u_{2k+1}|}{|u_{2k}|} \\right) \$", y=0.5, x=1.12)
    #  plot(log.(abs.( seuil.(ψ2kp1) ./ seuil.(ψ2k))), "+", label="\$ \\varepsilon = $(ε) \$")
    #  xlim(0,20)
    #  legend()

    #  subplot(2,2,4)
    #  title("\$ \\log \\left( \\frac{|u_{2k+1}|}{|u_{2k-1}|} \\right) \$", y=0.5, x=1.12)
    #  plot(log.(abs.( seuil.(ψ2kp1) ./ seuil.(ψ2km1))), "+", label="\$ \\varepsilon = $(ε) \$")
    #  xlabel("\$ k \$")
    #  xlim(0,20)
    #  legend()

    figure(2)
    plot(x, abs2.(ψr), label="\$ \\varepsilon = $(ε) \$")
end
