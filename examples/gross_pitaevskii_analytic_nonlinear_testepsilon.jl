using PyPlot
using DFTK
using DoubleFloats
using GenericLinearAlgebra

### tool functions for computing the solution of u + u^3 = A*sin(x)

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
A = 10
function u0(x)
    cardan(A*sin(x))
end

## weighted l2 spaces of analytic functions
w(G, A) = exp(2*A*abs(G))

V(r) = 1

a = 2π
lattice = a * [[1 0 0.]; [0 0 0]; [0 0 0]]

C = 1/2
α = 2

n_electrons = 1  # Increase this for fun

f(r) = A * sin(r)
source_term = ExternalFromReal(r -> f(r[1]))

#  ε_list = [1e-16, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1]
ε_list = [1e-8, 1e-7, 1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1]
#  ε_list = [0]
x = nothing
basis = nothing

# basis function
function e(G, z, basis)
    exp(G[1] * z * im) / sqrt(basis.model.unit_cell_volume)
end

# cut function
seuil(x) = abs(x) < 1e-8 ? zero(x) : x

# L^2 and H^s norms
function norm_L2(basis, u)
    Gs = [abs(G[1]) for G in G_vectors(basis.kpoints[1])][:]
    return sqrt(sum(seuil.(abs.(u)).^2))
end
function norm_Hs(basis, u, s)
    Gs = [abs(G[1]) for G in G_vectors(basis.kpoints[1])][:]
    return sqrt(sum((1 .+ Gs.^2).^s .* seuil.(abs.(u)).^2))
end

# u0 solution for ε = 0
u0r = ExternalFromReal(r->u0(r[1]))
u0G = nothing
save0 = false

cvg_L2norm = []
cvg_H1norm = []
cvg_H2norm = []
L2norm = []
H1norm = []
H11norm = []
H2norm = []

figure(1)
ftsize = 20
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

    Ecut = 200000
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

    global save0, u0G
    if !save0
        u0G = r_to_G(basis, basis.kpoints[1], ComplexF64.(u0r(basis).potential))[:,1]
        println("|u0-ψ| = ", norm_L2(basis, u0G-ψ))
        println(norm(u0G-ψ))
        save0 = true
    end

    # norms
    append!(cvg_L2norm, norm_L2(basis, u0G-ψ))
    append!(cvg_H1norm, norm_Hs(basis, u0G-ψ, 1))
    append!(cvg_H2norm, norm_Hs(basis, u0G-ψ, 2))
    append!(L2norm, norm_L2(basis, ψ))
    append!(H1norm, norm_Hs(basis, ψ, 1))
    append!(H2norm, norm_Hs(basis, ψ, 2))
    append!(H11norm, norm_Hs(basis, ψ, 1.1))

    # plots
    global x
    x = a * vec(first.(DFTK.r_vectors(basis)))
    ψr = G_to_r(basis, basis.kpoints[1], ψ)[:, 1, 1]

    if ε >= 0.0001
        figure(1)
        subplot(121)
        plot(x, real.(ψr), label="\$ \\varepsilon = $(ε) \$")
        figure(2)
        Gs = [abs(G[1]) for G in G_vectors(basis.kpoints[1])][:]
        semilogy(Gs, (seuil.(abs.(ψ))), "+", label="\$ \\varepsilon = $(ε) \$")
    end

    #  # compute slope
    #  ψ_slope = []
    #  G_slope = []
    #  for (iG, G) in enumerate(G_vectors(basis.kpoints[1]))
    #      if abs(ψ[iG]) > 1e-8 && abs(ψ[iG]) < 1e-6
    #          append!(ψ_slope, abs(ψ[iG]))
    #          append!(G_slope, abs(G[1]))
    #      end
    #  end
    #  _, Bε = -[ones(length(G_slope)) Float64.(G_slope)] \ log.(ψ_slope)
    #  ref_Bε = [1 / sqrt(1000000*w(G[1],Bε)) for G in G_vectors(basis.kpoints[1])] # ref slope in Fourier
    #  plot(Gs, log.(seuil.(abs.(ref_Bε))), "r-", label="1 / (√w_B(k)) B = $(Bε)")


    #  function u(z)
    #      φ = zero(ComplexF64)
    #      for (iG, G) in  enumerate(G_vectors(basis.kpoints[1]))
    #          φ += ψ[iG] * e(G, z, basis)
    #      end
    #      return φ
    #  end

    #  figure(3)
    #  is = range(-0.5, 0.5, length=10001)
    #  ux = abs.(u.(is .* 1im))
    #  plot(is, ux , label="ε = $(ε)")
    #  plot([Bε, Bε], [maximum(ux), minimum(ux)] , "r-")

    #  figure(4)
    #  is = range(-0.5, 0.5, length=1000)
    #  plot(is, abs.(u.(1 .+ is .* 1im)), label="ε = $(ε)")

    #  figure()
    #  suptitle("analytical expansion ε = $(ε)")
    #  rs = range(-π, π, length=500)
    #  is = range(-2, 2, length=500)
    #  plot_complex_function(rs, is, z->u(z))
end

#  # plot norms
#  figure(3)
#  loglog(ε_list, L2norm, "x-", label="L2 norm")
#  loglog(ε_list, H1norm, "x-", label="H1 norm")
#  loglog(ε_list, H11norm, "x-", label="H1.1 norm")
#  loglog(ε_list, H2norm, "x-", label="H2 norm")
#  legend()
#  xlabel("ε")

## tests
figure(1)
subplot(121)
plot(x, u0.(x), "--", label="\$ \\varepsilon = 0 \$")
#  plot(x, f.(x), "k--", label="f")
xlabel("\$ x \$", size = ftsize)
legend()

figure(1)
subplot(122)
loglog(ε_list, cvg_L2norm, "x-", label="\$ ||u_0-u_\\varepsilon||_{{\\rm L}^2_\\sharp} \$")
loglog(ε_list, cvg_H1norm, "x-", label="\$ ||u_0-u_\\varepsilon||_{{\\rm H}^1_\\sharp} \$")
loglog(ε_list, cvg_H2norm, "x-", label="\$ ||u_0-u_\\varepsilon||_{{\\rm H}^2_\\sharp} \$")
legend()
xlabel("\$ \\varepsilon \$", size=ftsize)

figure(2)
Gs = [abs(G[1]) for G in G_vectors(basis.kpoints[1])][:]
semilogy(Gs, (seuil.(abs.(u0G))), "+", label="\$ \\varepsilon = 0 \$")
xlim(-15, 315)
xlabel("\$ |k| \$", size = ftsize)
legend()

# plot different w_G(B)
#  B_list = 1:0.2:2
#  for B in B_list
#      ref_B = [G[1] == 0 ? zero(G[1]) : 1 / sqrt(w(G[1],B))
#               for G in G_vectors(basis.kpoints[1])] # ref slope in Fourier
#      plot(Gs, log.(seuil.(abs.(ref_B))), "x-", label="1 / (√w_B(k)) B = $(B)")
#  end
#  xlabel("k")
#  legend()

#  figure()
#  suptitle("analytical expansion ε = 0")
#  rs = range(-π, π, length=500)
#  is = range(-2, 2, length=500)
#  plot_complex_function(rs, is, z->u0(z))

#  function uu(z)
#      φ = zero(ComplexF64)
#      for (iG, G) in  enumerate(G_vectors(basis.kpoints[1]))
#          φ += u0G[iG] * e(G, z, basis)
#      end
#      return φ
#  end

#  figure(3)
#  is = range(-0.5, 0.5, length=10001)
#  ux = abs.(u0.(is .* 1im))
#  uux = abs.(uu.(is .* 1im))
#  plot(is, ux , label="ε = 0 cardan")
#  plot(is, uux , label="ε = 0 cardan G")
#  plot([B, B], [maximum(ux), minimum(ux)] , "k-")
#  ylim(top=2, bottom=-0.5)
#  legend()

#  figure(4)
#  is = range(-0.1, 0.1, length=1000)
#  ux = abs.(u0.(1 .+ is .* 1im))
#  plot(is, ux , label="ε = 0")
#  plot([B, B], [maximum(ux), minimum(ux)] , "k-")
#  legend()

