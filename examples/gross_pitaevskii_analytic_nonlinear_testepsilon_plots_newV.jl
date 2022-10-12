using PyPlot
using DFTK
using LinearAlgebra
using SpecialFunctions
using DoubleFloats
using GenericLinearAlgebra

include("./plotting_analytic.jl")

# custom V from computations to have poles
C = 1/2
α = 2

λ0 = α * C / (2π)
β = 1 + 1e-6
γ = λ0 / β
V(x) = γ * cos(x)

function sqrt_cplx(z)
    z = Complex(z)
    Θ = angle(z)
    r = abs(z)
    Θ = mod(Θ + π, 2π) - π
    √r * exp(Θ/2*1im)
end

# u0 is the real solution of Vu + u^3 = λ0u on [0,2π]
function u0(x)
    sqrt_cplx(λ0 - V(x)) / √(α*C)
end

# branching point
B = log(β + √(β^2-1))

figure(1)
ftsize = 30
rc("font", size=ftsize, serif="Computer Modern")
rc("text", usetex=true)
Is = range(-1.2*B, 1.2*B, 100000)
is = range(-2*B, 2*B, 1500)
rs = range(-0.0001, 0.0001, 1500)
fr(z) = real(u0(z))
fi(z) = imag(u0(z))
f(z)  = u0(z)
plot(Is, fr.((1im).*Is))
plot(Is, fi.((1im).*Is))
plot(B, 0, "ro")
plot(-B, 0, "ro")

figure(2)
plot_complex_function(rs, is, f; cmap_color="")
plot(0, B, "ro")
plot(0, -B, "ro")

save0 = true

## solve for ε  > 0
a = 2π
lattice = a * [[1 0 0.]; [0 0 0]; [0 0 0]]

n_electrons = 1  # Increase this for fun

ε_list = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1]
x = nothing
basis = nothing

# basis function
function e(G, z, basis)
    exp(G[1] * z * im) / sqrt(basis.model.unit_cell_volume)
end

# cut function
tol = 1e-15
seuil(x) = abs(x) > tol ? x : 0.0
#  seuil(x) = x

for ε in ε_list

    println("---------------------------------")
    println("ε = $(ε)")
    terms = [Kinetic(2),
             ExternalFromReal(r -> V(r[1])/ε),
             PowerNonlinearity(C/ε, α),
            ]
    model = Model(lattice; n_electrons, terms,
                  spin_polarization=:spinless)  # use "spinless electrons"

    Ecut = 10000000
    global basis
    basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid=(1, 1, 1))
    u0r = ExternalFromReal(r->u0(r[1]))
    u0G = DFTK.r_to_G(basis, basis.kpoints[1], ComplexF64.(u0r(basis).potential_values))[:,1]
    ψ0 = [reshape(u0G, length(u0G), 1)]
    scfres = direct_minimization(basis, ψ0; tol, show_trace=false)
    println(scfres.energies)

    # ## Internals
    # We use the opportunity to explore some of DFTK internals.
    #
    # Extract the converged density and the obtained wave function:
    ρ = real(scfres.ρ)[:, 1, 1, 1]  # converged density, first spin component
    ψ = scfres.ψ[1][:, 1];    # first kpoint, all G components, first eigenvector

    # plots
    global x
    x = a * vec(first.(DFTK.r_vectors(basis)))
    ψr = G_to_r(basis, basis.kpoints[1], ψ)[:, 1, 1]

    figure(3)
    if save0
        plot(x, u0.(x), label="\$ \\varepsilon = 0 \$")
    end
    plot(x, real.(ψr), label="\$ \\varepsilon = $(ε) \$")

    figure(4)
    Gs = [abs(G[1]) for G in G_vectors(basis, basis.kpoints[1])][:]
    nG = length(Gs)
    ψG = [ψ[k] for k=1:nG]
    ψGn = [ψ[k+1] for k=1:(nG-1)]
    #  GGs = Gs[2:div(length(Gs)+1,2)]
    #  nG = length(GGs)
    #  ψG = [ψ[2*k] for k =1:div(nG,2)]
    #  GGGs = [GGs[2*k] for k =1:div(nG,2)]
    #  ψGn = ψG[2:end]
    global save0
    if save0
        # plot fourier
        #  subplot(121)
        semilogy(Gs, (seuil.(abs.(u0G))), "+", label="\$ \\varepsilon = 0 \$")
        #  subplot(122)
        #  u0Gn = [u0G[k+1] for k=1:(nG-1)]
        #  plot(Gs[2:end], log.(abs.( seuil.(u0Gn) ./ seuil.(u0G[1:end-1] ))), "+", label="\$ \\varepsilon = 0 \$")
        save0 = false
    end
    #  subplot(121)
    semilogy(Gs, (seuil.(abs.(ψG))), "+", label="\$ \\varepsilon = $(ε) \$")
    #  subplot(122)
    #  plot(Gs[2:end], log.(abs.( seuil.(ψGn) ./ seuil.(ψG[1:end-1] ))), "+", label="\$ \\varepsilon = $(ε) \$")

    println(λ0)
    println(scfres.eigenvalues[1][1]*ε)
    println(abs(λ0 - scfres.eigenvalues[1][1]*ε))


    #  function u(z)
    #      φ = zero(ComplexF64)
    #      for (iG, G) in  enumerate(G_vectors(basis, basis.kpoints[1]))
    #          φ += seuil(ψ[iG]) * e(G, z, basis)
    #      end
    #      return φ
    #  end
    #  figure(1)
    #  plot(is, real.(u.(is .* im)), label="\$ \\varepsilon = $(ε) \$")
end

figure(1)

figure(2)

figure(3)
legend()

figure(4)
#  subplot(121)
xlabel("\$ |k| \$")
legend()
#  subplot(122)
#  xlabel("\$ |k| \$")
