"""
    Wflow

Scale setting and flow-time interpolation on gradient-flow data held in a
long-format `DataFrame`, one row per (configuration, flow time):

```
2004002×5 DataFrame
     Row │ itraj  flowt    t2Eplq      t2Esym       qtop
         │ Int64  Float64  Float64     Float64      Float64
─────────┼────────────────────────────────────────────────────
       1 │    20     0.0   0.0         0.0          -0.733137
       2 │    20     0.01  0.00133921  0.000195178  -1.48943
    ⋮    │   ⋮       ⋮         ⋮            ⋮           ⋮
 2004002 │ 20020    20.01  2.52473     2.46468      -1.92402
```

Column names are not fixed: every function takes the observable (`obs`) and
flow-time (`time` / `t`) columns as keywords. The defaults match the layout
above, as produced by `IO_*.get_flow_data`.

Error analysis uses ADerrors. Functions that build `uwreal`s take `uwargs...`,
forwarded to `uwreal(data, uwargs...)`, i.e. `(id,)` or `(id, replica_lengths)`.
The data at each flow time are passed in replica order: rows are sorted by
`order` (default: whichever of `:irep`, `:itraj` are present) before the
`uwreal` is built, so the replica lengths in `uwargs` line up with the data.
"""
module Wflow
    using DataFrames, ProgressMeter, ADerrors, Optim, Statistics, StatsBase


    #---- internal helpers ---------------------------------------------------#

    # Sort keys that put rows in replica order: `order` if given, otherwise
    # whichever of :irep, :itraj the DataFrame has (old single-replica frames
    # have only :itraj).
    _order(d::AbstractDataFrame, order) =
        isnothing(order) ? intersect([:irep, :itraj], propertynames(d)) : order

    # Column `obs` at flow time `tt` (exact grid value), one entry per
    # configuration, in replica order — ready to be passed to `uwreal`.
    function _column_at(d::AbstractDataFrame, t, tt, obs, order)
        rows = d[d[!, t] .== tt, :]
        o    = _order(d, order)
        isempty(o) || sort!(rows, o)
        return Vector{Float64}(rows[!, obs])
    end


    #---- scale setting ------------------------------------------------------#

    """
        tcut(df, sref; obs=:t2Esym, time=:flowt)

    Flow time at which `obs` crosses `sref`, by linear interpolation between the
    last row below and the first row above `sref`:

        t* = t₁ + (sref - y₁) (t₂ - t₁) / (y₂ - y₁)

    Works on plain numbers: `df` should hold one value per flow time (typically
    the Monte Carlo average, see `tbounds`), sorted by `time`, with `obs`
    monotonically increasing across `sref` (true for t²E and its derivatives).

    Throws a `BoundsError` if no row lies above (or below) `sref`.
    """
    function tcut(df::DF, sref::T; obs=:t2Esym, time=:flowt) where {DF<:AbstractDataFrame, T<:AbstractFloat}
        # first point above sref
        y2 = df[df[:,obs] .> sref, :][1,obs]
        t2 = df[df[:,obs] .> sref, :][1,time]

        # last point below sref
        y1 = df[df[:,obs] .< sref, :][end,obs]
        t1 = df[df[:,obs] .< sref, :][end,time]

        # straight line through (t1,y1), (t2,y2), solved for y = sref
        m = (y2-y1)/(t2-t1)
        q = y1 - m*t1

        return (sref - q)/m
    end


    """
        tbounds(flw_data, sref; obs=:t2Esym, time=:flowt)

    Returns the two grid flow times `(t₁, t₂)` closest to t*, defined by ⟨obs⟩(t*) = sref,
    where ⟨obs⟩ is the average over configurations at each flow time.

    t* is found with `tcut` if the flow crosses `sref`, and otherwise by linear
    extrapolation from the last two flow times. Only central values are used here;
    this selects *which* flow times enter the error analysis in `uwscale`.

    If t* ∉ [t₁, t₂], `uwscale` extrapolates linearly, which is only reliable for
    small Δt = dist(t*, [t₁, t₂]). A warning then reports Δt in two forms:
    - Δt/(t₂-t₁): extrapolation distance in units of the lever arm of the two points;
    - Δt/t*: the more physical measure, since flow observables vary on a scale
    set by t itself (smoothing radius √(8t)).

    Errors if ⟨obs⟩ is already above `sref` at the first flow time, or if ⟨obs⟩ is
    not increasing at the last two flow times when extrapolation is needed.
    """
    function tbounds(flw_data::DF, sref::T; obs=:t2Esym, time=:flowt) where {DF<:AbstractDataFrame, T<:AbstractFloat}
        # Monte Carlo average per flow time, sorted in flow time
        avg = sort(combine(groupby(flw_data, time), obs => mean => obs), time)
        tt, yy = avg[!, time], avg[!, obs]

        length(tt) ≥ 2 || error("Need at least two flow times, got $(length(tt))")
        first(yy) < sref ||
            error("⟨$(obs)⟩ = $(first(yy)) is already above sref = $(sref) at the first flow time $(first(tt))")

        # central crossing point: interpolated if the flow reaches sref, extrapolated otherwise
        ts = if last(yy) > sref
            tcut(avg, sref; obs=obs, time=time)
        else
            dy = yy[end] - yy[end-1]
            dy > 0 ||
                error("⟨$(obs)⟩ not increasing at the last two flow times; cannot extrapolate to sref = $(sref)")
            tt[end] + (sref - yy[end]) * (tt[end] - tt[end-1]) / dy
        end

        # two grid points closest to t* (always adjacent on a sorted grid)
        i, j = sort(partialsortperm(abs.(tt .- ts), 1:2))
        t1, t2 = tt[i], tt[j]

        if !(t1 ≤ ts ≤ t2)
            Δ  = min(abs(ts - t1), abs(ts - t2))
            r3 = x -> round(x; sigdigits=3)
            @warn("t* ≈ $(r3(ts)) lies outside the closest pair ($t1, $t2): extrapolating by " *
                "Δt = $(r3(Δ)), i.e. Δt/(t₂-t₁) = $(r3(Δ/(t2 - t1))), Δt/t* = $(r3(Δ/ts)). " *
                "Linear extrapolation is only reliable for small Δt.")
        end
        return t1, t2
    end


    """
        uwscale(flw_data, sref, uwargs...; obs=:t2Esym, time=:flowt, order=nothing)

    Reference flow time (e.g. t₀/a² for `obs = t²E`, `sref = 0.3`) as a `uwreal`.

    The two flow times bracketing the crossing are chosen from central values
    (`tbounds`); at those, `obs` is promoted to primary observables y₁, y₂ and

        t_ref = (sref - q) / m,    m = (y₂ - y₁)/(t₂ - t₁),  q = y₁ - m t₁

    so the error and the correlation between y₁ and y₂ are propagated exactly.
    """
    function uwscale(flw_data::D, sref::Float64, uwargs...; obs=:t2Esym, time=:flowt, order=nothing) where D<:AbstractDataFrame
        (t1, t2) = tbounds(flw_data, sref; obs=obs, time=time)

        # per-configuration values at the bracketing flow times, in replica order
        y1 = uwreal(_column_at(flw_data, time, t1, obs, order), uwargs...)
        y2 = uwreal(_column_at(flw_data, time, t2, obs, order), uwargs...)

        m = (y2-y1)/(t2-t1)
        q = y1 - m*t1

        return (sref - q)/m
    end


    #---- evaluating observables at a given flow time ------------------------#

    """
        interpolate_at(d, tx, uwargs...; obs=:q, t=:flowt, order=nothing)

    ⟨obs⟩ at flow time `tx` by linear interpolation between the two grid flow
    times bracketing it, `ta ≤ tx < tb`:

        ⟨obs⟩(tx) = y_a + (tx - t_a) (y_b - y_a) / (t_b - t_a)

    with y_a, y_b primary observables built from the configurations at t_a, t_b.

    `tx` can be a `uwreal` (e.g. `r * t0` from `uwscale`): the brackets use its
    central value and the interpolation uses `tx` itself, so its error and its
    correlation with `obs` (same configurations) are propagated.
    """
    function interpolate_at(d::AbstractDataFrame, tx::Union{AbstractFloat,uwreal}, uwargs...;
                            obs=:q, t=:flowt, order=nothing)
        tv    = tx isa uwreal ? value(tx) : tx
        tflow = sort(unique(d[!, t]))
        i     = searchsortedlast(tflow, tv)
        1 ≤ i < length(tflow) ||
            error("flow time $(tv) outside the grid [$(first(tflow)), $(last(tflow))]")
        ta, tb = tflow[i], tflow[i+1]

        ya = uwreal(_column_at(d, t, ta, obs, order), uwargs...)
        yb = uwreal(_column_at(d, t, tb, obs, order), uwargs...)

        return ya + (tx - ta) * (yb - ya) / (tb - ta)
    end


    """
        slice_at(df, t0; time=:flowt)

    The rows of `df` at the grid flow time closest to `t0`, as a `SubDataFrame`
    (a view; no copy). No interpolation: use `interpolate_at` when the
    observable is needed exactly at `t0`, or when `t0` carries an error.
    """
    function slice_at(df, t0::T; time=:flowt) where T<:AbstractFloat
        tflow = unique(df[!, time])
        tnear = tflow[argmin(abs.(tflow .- t0))]   # nearest grid flow time
        return view(df, df[!, time] .== tnear, :)
    end


    #---- topological charge -------------------------------------------------#

    """
        find_Z(q; Z0=1.1)

    Multiplicative factor Z ∈ [1, 2] that brings the per-configuration charges
    `q` closest to integers, by minimising

        f(Z) = mean_k (Z q_k - round(Z q_k))²

    with box-constrained Nelder–Mead started at `Z0`. f is piecewise smooth with
    many local minima, so the result depends on `Z0`; check it against a scan of
    f(Z) if in doubt. Meaningful only at flow times where the charge
    distribution already shows peaks near integers.
    """
    function find_Z(q::AbstractArray{Float64}; Z0=1.1)
        res = optimize(
            Z->mean((Z.*q .- round.(Z.*q)).^2),
            [1.], [2.], [Z0],
            Fminbox(NelderMead())
        )
        return Optim.minimizer(res) |> only
    end


    export tcut, tbounds, uwscale
    export slice_at, interpolate_at, find_Z

end