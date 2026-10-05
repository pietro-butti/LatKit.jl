"""
The basic input for this module is a `DataFrame` object that will be passed as argument to most of the functions in the modules. Such an object must look like:

```
2004002×5 DataFrame
     Row │ itraj  flowt    t2Eplq      t2Esym       qtop      
         │ Int64  Float64  Float64     Float64      Float64   
─────────┼────────────────────────────────────────────────────
       1 │    20     0.0   0.0         0.0          -0.733137
       2 │    20     0.01  0.00133921  0.000195178  -1.48943
       3 │    20     0.02  0.00493909  0.000762703  -1.71782
    ⋮    │   ⋮       ⋮         ⋮            ⋮           ⋮
 2004000 │ 20020    19.99  2.52216     2.46213      -1.92395
 2004001 │ 20020    20.0   2.52345     2.4634       -1.92398
 2004002 │ 20020    20.01  2.52473     2.46468      -1.92402
```

It can be obtained as the output of the function `IO_*.get_flow_data`.
"""
module Wflow
    using DataFrames, ProgressMeter,ADerrors, Optim, Statistics, StatsBase

    """
        uwflow(df::DataFrame, args...; obs=:t2Esym, time=:flowt)

    Groups data by flow time, converts observation values to `uwreal` objects, and returns
    a new DataFrame with the converted values.

    # Arguments
    - `df::DataFrame`: Input dataframe containing flow time and observation columns
    - `args...`: Additional arguments passed to `uwreal` function
    - `obs::Symbol`: Column name of the observable to convert (default: `:t2Esym`)
    - `time::Symbol`: Column name of the flow time (default: `:flowt`)

    # Returns
    - `DataFrame`: New dataframe with `uwreal` converted observations and corresponding flow times
    """
    function uwflow(df::DataFrame, args...; obs=:t2Esym, time=:flowt)
        aux = DataFrame(flowt=Float64[], measure=[])
        for flw in groupby(df,time)
            t = flw[:,time][1]
            t2E = uwreal(flw[:,obs],args...)
            push!(aux, (t, t2E))
        end

        rename!(aux, :measure => obs)
        rename!(aux, :flowt => time)
        return aux
    end

    """
        tcut(df; obs=:t2Esym)

    Returns a function that performs linear interpolation to find the flow time corresponding 
    to a given observable reference value.

    # Arguments
    - `df::DataFrame`: Dataframe containing observable and flow time columns
    - `obs::Symbol`: Column name of the observable (default: `:t2Esym`)

    # Returns
    - `Function`: A function `sref -> Float64` that takes a reference value and returns interpolated flow time
    """
    tcut(df; obs=:t2Esym, time=:flowt) = sref -> begin
        y2 = df[df[:,obs] .> sref, :][1,obs]
        t2 = df[df[:,obs] .> sref, :][1,time]

        y1 = df[df[:,obs] .< sref, :][end,obs]
        t1 = df[df[:,obs] .< sref, :][end,time]

        m = (y2-y1)/(t2-t1)
        q = y1 - m*t1
        
        return (sref - q)/m
    end

    """
        tbounds(flw_data::DataFrame, sref::Float64; obs=:t2Esym)

    Determines the flow time bounds that bracket the reference value by computing
    the average observable across trajectories and performing interpolation.

    In case the average of `flw_data[:,obs]` does not get to `sref`, the returned 
    flow times are the last two available. A warning message gets printed. 

    # Arguments
    - `flw_data::DataFrame`: Dataframe containing flow data with multiple trajectories
    - `sref::Float64`: Reference observable value
    - `obs::Symbol`: Column name of the observable (default: `:t2Esym`)
    - `time::Symbol`: Column name of the flow time (default: `:flowt`)

    # Returns
    - `Tuple{Float64, Float64}`: Lower and upper flow time bounds (t1, t2)
    """
    function tbounds(flw_data::DataFrame, sref::Float64; obs=:t2Esym, time=:flowt)
        # Compute average values
        avg = combine(groupby(flw_data,time), obs => mean => obs)
        
        t1,t2 = nothing, nothing
        try
            scale = tcut(avg,obs=obs,time=time)(sref)
            idx = searchsortedfirst(avg[:,time], scale)
            t1,t2 = avg[:,time][(idx-1):idx]
        catch BoundsError
            t1,t2 = avg[:,time][end-1:end]
            @warn("Flow does not get to $sref. Extrapolating, might be very far off...")
        end
        
        return t1, t2
    end

    """
    confid(flw_data; offset=true)

    Extracts and optionally adjusts trajectory identifiers for confidence interval calculations.

    # Arguments
    - `flw_data::DataFrame`: Dataframe containing trajectory column `:itraj`
    - `offset::Boolean`: If true, shifts trajectory indices to start from a non-zero value (default: true)

    # Returns
    - `Vector{Int64}`: Trajectory identifiers as Int64 values
    """
    function confid(flw_data; offset=false)
        trajs = unique(flw_data.itraj)

        if offset
            trajs .-= trajs[1] # Subtract the first element
            trajs .+= trajs[2] # Add the second element to start form not zero
        end

        # trajs = Int64.(trajs) .// gcd(trajs)
        return Int64.(trajs)
    end
    
    """
        uwscale(flw_data::DataFrame, sref::Float64, uwargs...; obs=:t2Esym, time=:flowt)

    Computes a scaled observable value at the reference point using linear interpolation 
    between two bracketing flow times with proper uncertainty propagation.

    # Arguments
    - `flw_data::DataFrame`: Dataframe containing flow data
    - `sref::Float64`: Reference observable value
    - `uwargs::Tuple`: Tuple of arguments passed to `uwreal` to compute ref scale
    - `obs::Symbol`: Column name of the observable (default: `:t2Esym`)
    - `time::Symbol`: Column name of the flow time (default: `:flowt`)

    # Returns
    - `uwreal`: The interpolated scale at the reference value with uncertainties
    """
    function uwscale(flw_data::DataFrame, sref::Float64, uwargs...; obs=:t2Esym, time=:flowt)
        (t1,t2) = tbounds(flw_data,sref; obs=obs,time=time)      
        
        E1 = flw_data[flw_data[:,time] .== t1,obs]
        E2 = flw_data[flw_data[:,time] .== t2,obs]

        y1 = uwreal(E1,uwargs...)
        y2 = uwreal(E2,uwargs...)

        m = (y2-y1)/(t2-t1)
        q = y1 - m*t1
        
        return (sref - q)/m
    end


    """
        slice_at(df, t0; time=:flowt)

    Extracts a subset of the dataframe corresponding to the flow time closest to the 
    specified target value t0.

    # Arguments
    - `df::DataFrame`: Input dataframe containing a flow time column
    - `t0::Float64`: Target flow time value
    - `time::Symbol`: Column name of the flow time (default: `:flowt`)

    # Returns
    - `SubDataFrame`: Subset of the input dataframe at the nearest flow time to t0
    """
    function slice_at(df, t0; time=:flowt)
        # Extract unique flow time values
        tflow = unique(df[:, time])
        
        # Find index of flow time closest to target value
        it0 = argmin(abs.(tflow .- t0))
        
        # Return grouped data at the identified flow time
        return groupby(df, time)[it0]
    end



    # function find_Z(q::AbstractArray{Float64}; Z0=0.7)
    #     res = optimize( α->mean((α.*q .- round.(α.*q)).^2), [Z0])
    #     return Optim.minimizer(res)
    # end
function find_Z(q::AbstractArray{Float64}; Z0=1.1)
    res = optimize( 
        Z->mean((Z.*q .- round.(Z.*q)).^2), 
        [1.], [2.], [Z0],
        Fminbox(NelderMead())
    )
    return Optim.minimizer(res) |> only
end


    # function Qtop(df::DataFrame, tcut::Float64; alpha=nothing)
    #     α = isnothing(alpha) ? find_optimal_alpha(df,tcut) : alpha
    #     aux = slice_at(df,tcut)[:,[:itraj,:qtop]]
    #     aux.qtop_round = round.(α .* aux.qtop)
    #     return aux
    # end


    export uwflow, tcut, tbounds, uwscale, confid
    export slice_at, find_Z
    
end