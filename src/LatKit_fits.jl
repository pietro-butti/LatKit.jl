line(x,p) = p[1]*x + p[2] 
line(x::Vector,p) = [line(el,p) for el in x]
line(x::Vector,p::Vector{uwreal}) = [line(el,p) for el in x]


parabola(x,p) = p[1]*x^2 + p[2]*x + p[3] 
parabola(x::Vector,p) = [parabola(el,p) for el in x]
parabola(x::Vector,p::Vector{uwreal}) = [parabola(el,p) for el in x]



χ²(model, xdata, pars, ydata, yerr) = begin
    ŷ = model(xdata, pars)
    sum((ydata .- ŷ).^2 ./ yerr.^2)
end

function uwfit(xdata,ydata::Vector{uwreal}; ord=1)
    model = ord==1 ? line : parabola

    fit = curve_fit(
        model,
        xdata,value.(ydata),
        1. ./ err.(ydata).^2,
        [0.5,0.5,0.5]
    )

    fitp,csqexp = fit_error(
        (p,d)->χ²(model,xdata,p,d,err.(ydata)),
        fit.param,ydata
    )
 
    return fitp, sum(fit.resid.^2) , csqexp
end
function uwfit(model,xdata,ydata; p0=[.5,.5,.5])
    fit = curve_fit(
        model,
        xdata,value.(ydata),
        1. ./ err.(ydata).^2,
        p0
    )

    fitp,csqexp = fit_error(
        (p,d)->χ²(model,xdata,p,d,err.(ydata)),
        fit.param,ydata
    )
 
    return fitp, sum(fit.resid.^2) , csqexp
end