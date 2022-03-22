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

ε = 1

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
scfres = self_consistent_field(basis; tol=tol, maxiter=200)# is_converged=DFTK.ScfConvergenceDensity(tol))
println(scfres.energies)

# ## Internals
# We use the opportunity to explore some of DFTK internals.
#
# Extract the converged density and the obtained wave function:
ψ = scfres.ψ[1][:, 1];    # first kpoint, all G components, first eigenvector
Hψ = scfres.ham.blocks[1] * ψ
println("|Hψ-εψ| = ", norm(Hψ - scfres.eigenvalues[1][1].*ψ))

# plots
x = a * vec(first.(DFTK.r_vectors(basis)))
ψr = G_to_r(basis, basis.kpoints[1], ψ)[:, 1, 1]

figure(1)
ftsize = 30
rc("font", size=ftsize, serif="Computer Modern")
rc("text", usetex=true)
Gs = [abs(G[1]) for G in G_vectors(basis, basis.kpoints[1])][:]
#  GGs = Gs[2:div(length(Gs)+1,2)]
#  nG = length(GGs)
subplot(121)
semilogy(Gs, (seuil.(abs.(ψ))), "+", label="\$ \\varepsilon = $(ε) \$")
xlabel("\$ |k| \$")
xlim(0,20)
subplot(122)
plot(Gs[1:end-1], log.(abs.( seuil.(ψ[2:end]) ./ seuil.(ψ[1:end-1] ))), "+", label="\$ \\varepsilon = $(ε) \$")
xlabel("\$ k \$")
xlim(0,20)

figure(2)
plot(x, abs2.(ψr))
