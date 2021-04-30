using PyPlot
using DFTK
using DoubleFloats
using GenericLinearAlgebra

## weighted l2 spaces of analytic functions
function w(G, A)
    cosh(2*A*G)
end

B = 1
function pot(G)
    if G == 0
        return 0.0
    else
        1. / (abs(G)*sqrt(w(G,B)))
    end
end

#  pot(x) = sin(x)

a = 10
lattice = a .* [[1 0 0.]; [0 0 0]; [0 0 0]];

C = 1.0
α = 2;

n_electrons = 1  # Increase this for fun
terms = [Kinetic(),
         ExternalFromFourier(G -> pot(G[1])),
]
model = Model(Array{Double64}(lattice); n_electrons=n_electrons, terms=terms,
              spin_polarization=:spinless);  # use "spinless electrons"

Ecut = 150
basis = PlaneWaveBasis(model, Ecut, kgrid=(1, 1, 1))
scfres = self_consistent_field(basis, tol=1e-14) # This is a constrained preconditioned LBFGS
scfres.energies

#  # ## Internals
#  # We use the opportunity to explore some of DFTK internals.
#  #
#  # Extract the converged density and the obtained wave function:
#  ρ = real(scfres.ρ)[:, 1, 1, 1]  # converged density, first spin component
#  ψ_fourier = scfres.ψ[1][:, 1];    # first kpoint, all G components, first eigenvector

#  # Transform the wave function to real space and fix the phase:
#  ψ = G_to_r(basis, basis.kpoints[1], ψ_fourier)[:, 1, 1]
#  ψ /= (ψ[div(end, 2)] / abs(ψ[div(end, 2)]));

#  # Check whether ``ψ`` is normalised:
#  x = a * vec(first.(DFTK.r_vectors(basis)))
#  N = length(x)
#  dx = a / N  # real-space grid spacing
#  @assert sum(abs2.(ψ)) * dx ≈ 1.0

#  figure()
#  plot(x, real.(ψ), label="real(ψ)")
#  plot(x, imag.(ψ), label="imag(ψ)")
#  plot(x, ρ, label="ρ")
#  legend()

#  ## test if the solution belongs to all H_A for A<B

#  function norm_A(u, A)
#      weights = [w(G[1], A) for G in G_vectors(basis.kpoints[1])]
#      sqrt(sum(abs2.(u) .* weights))
#  end

#  V = [pot(G[1]) for G in G_vectors(basis.kpoints[1])]
#  println(norm_A(V,B))

#  Ar = 0.1:0.01:B
#  HAnorm_ψ = []
#  HAnorm_V = []
#  for A in Ar
#      append!(HAnorm_ψ, norm_A(ψ_fourier, A))
#      append!(HAnorm_V, norm_A(V, A))
#  end
#  figure()
#  plot(Ar, HAnorm_ψ, label="|ψ|_A")
#  plot(Ar, HAnorm_V, label="|V|_A")
#  plot(Ar, [sqrt(2pi^2/6) for A in Ar], label="|V|_B")
#  legend()
#  xlabel("A")

#  for i in 1:length(V)
#      println((ψ_fourier[i], V[i]))
#  end

