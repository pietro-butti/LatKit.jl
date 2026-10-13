module LatKit
    using DataFrames, Pipe, StatsBase, ADerrors
    using LsqFit    

    # --- Broadcasting --------------------------
    import Base: length, iterate
    length(x::uwreal) = 1
    iterate(x::uwreal) = (x, nothing)
    iterate(x::uwreal,::Nothing) = nothing

    # --- Missing value for plotting -------------
    import ADerrors: err, value
    ADerrors.err(::Missing) = 0
    ADerrors.value(::Missing) = missing


    include("HiRep/HiRep.jl")
        using .HiRep
        export get_runtime, get_confn
        export get_plaquette, get_flow_data
        export read_spectrum, read_disconnected, uwdisc
    
    include("Wflow.jl")
        using .Wflow
        export tcut, tbounds, uwscale
        export slice_at, interpolate_at, find_Z

    include("LatKit_fits.jl")
        export line, parabola, χ², uwfit

end