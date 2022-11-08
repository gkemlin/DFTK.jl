using PyPlot
using DFTK
using LinearAlgebra
using SpecialFunctions

include("./plotting_analytic.jl")

# custom V from computations to have poles
C = 1/2
α = 2

λ0 = α * C / (2π)
β = 1 + 1e-4
γ = λ0 / β
V(x) = γ * cos(x)

function sqrt_cplx(z)
    z = Complex(z)
    Θ = angle(z)
    r = abs(z)
    Θ = mod(Θ + π, 2π) - π
    √r * exp(Θ/2*1im)
end

# u0 is the real solution of Vu + αC * u^3 = λ0u on [0,2π]
function u0(x)
    sqrt_cplx(λ0 - V(x)) / √(α*C)
end

# branching point
B = log(β + √(β^2-1))

figure(1, figsize=(22,12))
ftsize = 30
rc("font", size=ftsize, serif="Computer Modern")
rc("text", usetex=true)
Is = range(-1.2*B, 1.2*B, 10000)
is = range(-2*B, 2*B, 200)
rs = range(-0.01, 0.01, 200)
fr(z) = real(u0(z))
fi(z) = imag(u0(z))
f(z)  = u0(z)
subplot(121)
plot(Is, fr.((1im).*Is), label="\$ {\\rm Re}(u_0({\\rm i} y)) \$")
plot(Is, fi.((1im).*Is), label="\$ {\\rm Im}(u_0({\\rm i} y)) \$")
xlabel("\$ y \$")
plot(B, 0, "ro", label="\$ \\pm B_0 \$")
plot(-B, 0, "ro")
legend()

plot_complex_function(rs, is, f; cmap_color="")
plot(0, B, "ro")
plot(0, -B, "ro")
xlabel("\$ x \$")
ylabel("\$ y \$")
savefig("u01_iy.pdf")


save0 = true

## solve for ε  > 0
a = 2π
lattice = a * [[1 0 0.]; [0 0 0]; [0 0 0]]

n_electrons = 1  # Increase this for fun

ε_list = [1e-7, 1e-5, 1e-3, 1e-1]
x = nothing
basis = nothing

# basis function
function e(G, z, basis)
    exp(G[1] * z * im) / sqrt(basis.model.unit_cell_volume)
end

# cut function
Ecut = 100000000
tol = 1e-14
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
    global save0
    if save0
        # plot fourier
        subplot(121)
        semilogy(Gs, (seuil.(abs.(u0G))), "+", label="\$ \\varepsilon = 0 \$")
        semilogy(Gs, 1e-2 ./ sqrt.(abs.(Gs).^3 .* cosh.(2B .* abs.(Gs))), "--",
                 label="\$ 1/(|k|^{3/2} \\sqrt{\\cosh(2B_0k)}) \$")
        subplot(122)
        u0Gn = [u0G[k+1] for k=1:(nG-1)]
        plot(Gs[2:end], log.(abs.( seuil.(u0Gn) ./ seuil.(u0G[1:end-1] ))), "+", label="\$ \\varepsilon = 0 \$")
        plot(Gs[2:end], [-B for k in Gs[2:end]], "--", label="\$ -B_0 \$")
        plot(Gs[2:end], log.(sqrt.(abs.(Gs[1:end-1]).^3 .* cosh.(2B .* abs.(Gs[1:end-1]))) ./ sqrt.(abs.(Gs[2:end]).^3 .* cosh.(2B .* abs.(Gs[2:end])))), "--",
             label="\$ 1/(|k|^{3/2} \\sqrt{\\cosh(2B_0k)}) \$")
        plot(Gs[2:end], log.(sqrt.(abs.(Gs[2:end]).^3 .* cosh.(2B .* abs.(Gs[2:end]))) ./ sqrt.(abs.(Gs[1:end-1]).^3 .* cosh.(2B .* abs.(Gs[1:end-1])))), "--",
             label="\$ 1/(|k|^{3/2} \\sqrt{\\cosh(2B_0k)}) \$")
        #  plot(Gs[2:end], [B for k in Gs[2:end]], "--", label="\$ +B \$")
        save0 = false
    end
    subplot(121)
    semilogy(Gs, (seuil.(abs.(ψG))), "+", label="\$ \\varepsilon = $(ε) \$")
    subplot(122)
    plot(Gs[2:end], log.(abs.( seuil.(ψGn) ./ seuil.(ψG[1:end-1] ))), "+", label="\$ \\varepsilon = $(ε) \$")

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

figure(3)
legend()
savefig("u_r_$(Ecut)_$(tol).pdf")

figure(4)
subplot(121)
xlabel("\$ |k| \$")
xlim(-50, 1500)
ylim(1e-15, 1)
legend()
subplot(122)
xlabel("\$ |k| \$")
legend()
#  xlim(-50, 1500)
#  ylim(-0.1, 0)
savefig("u_fourier_$(Ecut)_$(tol).pdf")
