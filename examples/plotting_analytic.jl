## ploting analytic functions

using PyPlot

function plot_complex_function(rs, is, f)
    res = f isa Function ? [f(x + im*y) for x in rs, y in is] : f
    subplot(121)
    contour(rs, is, abs.(res)', levels=10)
    colorbar()
    subplot(122)
    pcolormesh(rs, is, angle.(res)', cmap="hsv")
    # only for A = 1
    #  plot([0], [1], "ro")
    #  plot([0], [-1], "ro")
    colorbar()
end

### test
#  rs = range(-0.5, 1.5, length=300)
#  is = range(-0.2, 0.2, length=300)
#  plot_complex_function(rs, is, z->log(z/(z-1)))

