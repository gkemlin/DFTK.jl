using PyPlot
using DFTK
using DoubleFloats
using GenericLinearAlgebra
using Random

Random.seed!(1234)

Ecut = 5000

Gcartmax = Int(ceil(sqrt(Ecut)*10))
rand_coef = [rand() for G in 1:Gcartmax]

## weighted l2 spaces of analytic functions
w(G, A) = cosh(2*A*G)

B = 1
pot(G) = G == 0 ? zero(G) : rand_coef[Int(ceil(abs(G)))] / (abs(G)*sqrt(w(G,B)))

a = 10
lattice = a * [[1 0 0.]; [0 0 0]; [0 0 0]]

C = 1
α = 2

n_electrons = 1  # Increase this for fun
terms = [Kinetic(),
         ExternalFromFourier(G -> pot(G[1])),
         PowerNonlinearity(C, α),
]
model = Model(Array{Double64}(lattice); n_electrons=n_electrons, terms=terms,
              spin_polarization=:spinless)  # use "spinless electrons"

tol = 1e-32
basis = PlaneWaveBasis(model, Ecut, kgrid=(1, 1, 1))
#  scfres = direct_minimization(basis, tol=tol)
scfres = self_consistent_field(basis, tol=tol, maxiter=200)
scfres.energies

# ## Internals
# We use the opportunity to explore some of DFTK internals.
#
# Extract the converged density and the obtained wave function:
ρ = real(scfres.ρ)[:, 1, 1, 1]  # converged density, first spin component
ψ_fourier = scfres.ψ[1][:, 1];    # first kpoint, all G components, first eigenvector
VG = [pot(G[1]) for G in G_vectors_cart(basis.kpoints[1])] # potential in Fourier

# Transform the wave function to real space and fix the phase:
ψ = G_to_r(basis, basis.kpoints[1], ψ_fourier)[:, 1, 1]
ψ /= (ψ[div(end, 2)] / abs(ψ[div(end, 2)]));

# plots
x = a * vec(first.(DFTK.r_vectors(basis)))

figure()
plot(x, real.(ψ), label="real(ψ)")
plot(x, imag.(ψ), label="imag(ψ)")
plot(x, ρ, label="ρ")
legend()

Gs = [abs(G[1]) for G in G_vectors_cart(basis.kpoints[1])][:]
ref_B = [G[1] == 0 ? zero(G[1]) : 1 / sqrt(w(G[1],B))
         for G in G_vectors_cart(basis.kpoints[1])] # ref slope in Fourier

figure()
title("B = $(B), C = $(C)")
semilogy(Gs, abs.(ψ_fourier), "o", label="ψ_k")
semilogy(Gs, abs.(ref_B), "o", label="1 / (√w_B(k))")
legend()
xlabel("|k|")

# test if the solution belongs to all H_A for A<B
#  for i in 1:length(ψ_fourier)
#      if abs(ψ_fourier[i]) <= 1e-21
#          ψ_fourier[i] = zero(ψ_fourier[i])
#      end
#      if abs(ref_B[i]) <= 1e-21
#          ref_B[i] = zero(ref_B[i])
#      end
#  end

#  function norm_A(u, A)
#      weights = [w(G[1], A) for G in G_vectors(basis.kpoints[1])]
#      sqrt(sum(abs2.(u) .* weights))
#  end

#  Ar = 0.1:0.01:B
#  HAnorm_ψ = []
#  HAnorm_V = []
#  for A in Ar
#      append!(HAnorm_ψ, norm_A(ψ_fourier, A))
#      append!(HAnorm_V, norm_A(VG, A))
#  end
#  figure()
#  title("B = $(B), C = $(C)")
#  plot(Ar, HAnorm_ψ, label="|ψ|_A")
#  plot(Ar, HAnorm_V, label="|V|_A")
#  legend()
#  xlabel("A")

