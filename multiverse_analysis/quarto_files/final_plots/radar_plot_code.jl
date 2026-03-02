task_vertices_old(M; radius=1.0, phase=pi/2) =
    [Point2f(radius*cos(phase + 2π*(k-1)/M), radius*sin(phase + 2π*(k-1)/M)) for k in 1:M]

function embed_specialization_old(df::DataFrame; M = maximum(df.task_id))
    verts = task_vertices_old(M)
    g = groupby(df, [:epoch, :bee_id])

    out = DataFrame(epoch=Int[], bee_id=Int[], x=Float32[], y=Float32[])
    for sub in g
        a = zeros(Float64, M)
        for r in eachrow(sub)
            a[r.task_id] = r.accuracies
        end
        s = sum(a)
        if s == 0
            p = Point2f(0, 0)
        else
            w = a ./ s
            p = Point2f(sum(w[i] * verts[i][1] for i in 1:M),
                        sum(w[i] * verts[i][2] for i in 1:M))
        end
        push!(out, (sub.epoch[1], sub.bee_id[1], p[1], p[2]))
    end
    return out, verts
end

"""
Regular M-gon vertices (tasks) on the unit circle.
`phase=π/2` puts task 1 at the top.
"""
function task_vertices(M::Int; radius::Real=1.0, phase::Real=π/2)
    r = Float32(radius)
    ph = Float32(phase)
    return [Point2f(r*cos(ph + 2f0*Float32(π)*(k-1)/M),
                     r*sin(ph + 2f0*Float32(π)*(k-1)/M)) for k in 1:M]
end

"""
Compute 2D embedding point for one (epoch, bee_id) group.

normalize = :sum      -> w = a / sum(a)
normalize = :softmax  -> w = softmax(a / T)  (sharper specialization for smaller T)

If all accuracies are 0 => returns center (0,0).
"""
function accuracy_point_old(acc::AbstractVector, verts::AbstractVector{<:Point2};
                        normalize::Symbol=:sum, temperature::Real=0.05)
    a = Float64.(acc)
    M = length(a)

    w = if normalize == :sum
        s = sum(a)
        s <= 0 ? nothing : (a ./ s)
    elseif normalize == :softmax
        T = float(temperature)
        m = maximum(a)
        ex = exp.((a .- m) ./ T)
        s = sum(ex)
        s <= 0 ? nothing : (ex ./ s)
    else
        error("normalize must be :sum or :softmax")
    end

    w === nothing && return Point2f(0, 0)

    x = 0.0
    y = 0.0
    @inbounds for i in 1:M
        x += w[i] * verts[i][1]
        y += w[i] * verts[i][2]
    end
    return Point2f(Float32(x), Float32(y))
end

function accuracy_point(acc::AbstractVector, verts::AbstractVector{<:Point2};
                        normalize::Symbol=:sum,
                        temperature::Real=0.05,
                        radial_mode::Symbol=:power,   # :power or :threshold
                        gamma::Real=1.0,              # stronger -> stays centered longer
                        s0::Real=0.2)                 # only for :threshold
    a = Float64.(acc)
    M = length(a)

    # proportions for direction
    w = if normalize == :sum
        s = sum(a)
        s <= 0 ? nothing : (a ./ s)
    elseif normalize == :softmax
        T = float(temperature)
        m = maximum(a)
        ex = exp.((a .- m) ./ T)
        s = sum(ex)
        s <= 0 ? nothing : (ex ./ s)
    else
        error("normalize must be :sum or :softmax")
    end

    w === nothing && return Point2f(0, 0)

    # barycentric direction point (on/inside polygon)
    x = 0.0
    y = 0.0
    @inbounds for i in 1:M
        x += w[i] * verts[i][1]
        y += w[i] * verts[i][2]
    end

    # maximum accuracy level (0..1), to scale according to the strongest signal
    s = maximum(a)

    # radial factor r in [0,1]
    r = if radial_mode == :power
        clamp(s, 0, 1)^gamma
    elseif radial_mode == :threshold
        u = clamp((s - s0) / (1 - s0), 0, 1)
        u^gamma
    else
        error("radial_mode must be :power or :threshold")
    end

    return Point2f(Float32(r * x), Float32(r * y))
end


"""
Embed a long dataframe with columns:
:epoch, :bee_id, :task_id, :accuracies

Returns:
- points: DataFrame(epoch, bee_id, x, y)
- verts:  task vertices (Point2f)
"""
function embed_specialization_old_v2(df::DataFrame;
                              M::Int = maximum(df.task_id),
                              normalize::Symbol = :sum,
                              temperature::Real = 0.05)

    verts = task_vertices(M)

    g = groupby(df, [:epoch, :bee_id])
    out = DataFrame(epoch=Int[], bee_id=Int[], x=Float32[], y=Float32[])

    for sub in g
        a = zeros(Float64, M)
        @inbounds for r in eachrow(sub)
            a[r.task_id] = r.accuracies
        end
        p = accuracy_point(a, verts; normalize=normalize, temperature=temperature, radial_mode=:power, gamma=1.0)
        push!(out, (sub.epoch[1], sub.bee_id[1], p[1], p[2]))
    end

    return out, verts
end

function embed_specialization(df::DataFrame;
                              M::Int = maximum(df.task_id),
                              task_order::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                              normalize::Symbol = :sum,
                              temperature::Real = 0.05)

    # mapping: original task_id -> vertex index (1..M)
    order = task_order === nothing ? collect(1:M) : collect(task_order)
    length(order) == M || error("task_order must have length M=$M")
    sort(order) == collect(1:M) || error("task_order must be a permutation of 1:M")

    task_to_vertex = Dict(order[i] => i for i in 1:M)

    verts = task_vertices(M)  # vertex i corresponds to order[i]

    g = groupby(df, [:epoch, :bee_id])
    out = DataFrame(epoch=Int[], bee_id=Int[], x=Float32[], y=Float32[])

    for sub in g
        a = zeros(Float64, M)  # indexed by vertex index now
        @inbounds for r in eachrow(sub)
            vi = task_to_vertex[r.task_id]
            a[vi] = r.accuracies
        end

        p = accuracy_point(a, verts; normalize=normalize, temperature=temperature)
        push!(out, (sub.epoch[1], sub.bee_id[1], p[1], p[2]))
    end

    return out, verts, order
end


# -----------------------------
# 2) Canvas: spiderweb / radar-like background without axes/ticks
# -----------------------------

"""
Draw a radar/spiderweb canvas on an existing Axis.

Keyword knobs are meant to be the "standardized" part you can tweak easily.
"""
function draw_radar_canvas!(ax::Axis, verts::AbstractVector{<:Point2};
                            rings::Int=5,
                            radius::Real=1.0,
                            pad::Real=0.15,
                            show_vertices::Bool=true,
                            show_labels::Bool=true,
                            label_scale::Real=1.10,
                            bg_color = (:black, 0.25),
                            bg_lw::Real=1.0,
                            ring_lw::Real=0.75,
                            #order::AbstractVector{<:Integer} = collect(1:length(verts)),
                            order::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                            fs::Real=24.0)
    order = order === nothing ? collect(1:length(verts)) : collect(order)

    # clean look
    hidedecorations!(ax)
    hidespines!(ax)
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.aspect = DataAspect()

    lim = radius * (1 + pad)
    xlims!(ax, -lim, lim)
    ylims!(ax, -lim, lim)

    # polygon coordinates
    xs  = Float32[v[1] for v in verts]
    ys  = Float32[v[2] for v in verts]
    xsC = vcat(xs, xs[1])
    ysC = vcat(ys, ys[1])

    # outer polygon
    lines!(ax, xsC, ysC; color=bg_color, linewidth=bg_lw)

    # spokes (all in one call)
    spoke_x = Float32[]
    spoke_y = Float32[]
    for v in verts
        append!(spoke_x, (0f0, Float32(v[1]), NaN32))
        append!(spoke_y, (0f0, Float32(v[2]), NaN32))
    end
    lines!(ax, spoke_x, spoke_y; color=bg_color, linewidth=bg_lw)

    # spiderweb rings (scaled polygons)
    if rings > 0
        ring_x = Float32[]
        ring_y = Float32[]
        for r in 1:rings
            ρ = Float32(r/(rings+1))
            append!(ring_x, ρ .* xsC); push!(ring_x, NaN32)
            append!(ring_y, ρ .* ysC); push!(ring_y, NaN32)
        end
        lines!(ax, ring_x, ring_y; color=bg_color, linewidth=ring_lw)
    end

    # vertices + labels
    if show_vertices
        scatter!(ax, xs, ys; color=bg_color, markersize=10)
    end
    if show_labels
        for (i, v) in enumerate(verts)
            text!(ax, "T$(order[i])",
                position = Point2f(Float32(v[1]*label_scale), Float32(v[2]*label_scale)),
                align = (:center, :center),
                fontsize = fs
            )
        end
    end

    return ax
end

# -----------------------------
# 3) Data layer: dots + (optional) trails, encode bee + time
# -----------------------------

# perceptual-ish brightness
luma(c::Colorant) = 0.2126*red(c) + 0.7152*green(c) + 0.0722*blue(c)

function distinct_palette(n::Int;
                          min_luma=0.15,  # avoid near-black
                          max_luma=0.90)  # avoid near-white
    # oversample then filter
    cand = distinguishable_colors(max(5n, 50), [RGB(1,1,1), RGB(0,0,0)])
    good = [c for c in cand if (min_luma ≤ luma(c) ≤ max_luma)]
    length(good) ≥ n || error("Not enough colors after filtering; loosen luma bounds.")
    return good[1:n]
end

dist2(a::RGB{Float32}, b::RGB{Float32}) = (a.r-b.r)^2 + (a.g-b.g)^2 + (a.b-b.b)^2

"""
pretty_palette(n):
- first 7 colors are EXACTLY Makie.wong_colors()
- remaining colors are generated to match Wong-ish aesthetics (no near-white/near-black,
  moderate saturation/lightness), and chosen to be far from the already-chosen colors.

Returns Vector{RGB{Float32}} length n.
"""
function pretty_palette(n::Int;
                        min_luma=0.20, max_luma=0.82,
                        sat_range=(0.40, 0.78),
                        light_range=(0.38, 0.66),
                        oversample::Int = max(12n, 240))

    # --- 1) start with Wong (exact) ---
    wong_rgba = Makie.wong_colors()               # Vector{RGBAf}/RGBA{Float32}, length 7
    wong_rgb  = RGB{Float32}.(wong_rgba)          # drop alpha, keep exact RGB
    chosen = RGB{Float32}[]
    append!(chosen, wong_rgb[1:min(n, length(wong_rgb))])

    n ≤ length(wong_rgb) && return chosen

    # --- 2) candidate pool in HSL (Wong-ish constraints) ---
    cand = RGB{Float32}[]
    hs = range(0f0, 360f0; length=oversample+1)[1:end-1]
    Ss = Float32.(range(sat_range[1], sat_range[2]; length=3))
    Ls = Float32.(range(light_range[1], light_range[2]; length=3))

    for h in hs, s in Ss, l in Ls
        c = RGB{Float32}(HSL(h, s, l))
        y = luma(c)
        if min_luma ≤ y ≤ max_luma
            push!(cand, c)
        end
    end
    isempty(cand) && error("Candidate pool empty; loosen constraints.")

    # Remove candidates too close to existing Wong colors (avoid near-duplicates)
    function too_close_to_any(c, cols; thr=0.015f0)
        for x in cols
            if dist2(c, x) < thr
                return true
            end
        end
        return false
    end
    cand = [c for c in cand if !too_close_to_any(c, chosen)]

    # --- 3) greedy max-min distance selection against all chosen ---
    while length(chosen) < n
        best_i = 0
        best_score = -1f0
        @inbounds for (i, c) in pairs(cand)
            dmin = Inf32
            for ch in chosen
                d = dist2(c, ch)
                d < dmin && (dmin = d)
            end
            if dmin > best_score
                best_score = dmin
                best_i = i
            end
        end
        best_i == 0 && error("Not enough colors; increase oversample or loosen constraints.")
        push!(chosen, cand[best_i])
        deleteat!(cand, best_i)
    end

    return chosen
end

function rank_bees(points::DataFrame; mode::Symbol=:final_max)
    bees = sort(unique(points.bee_id))

    r = sqrt.(points.x.^2 .+ points.y.^2)

    if mode == :final_max
        final_epoch = maximum(points.epoch)
        mask = points.epoch .== final_epoch
        score = Dict{Int, Float32}()
        for b in bees
            mb = mask .& (points.bee_id .== b)
            score[b] = any(mb) ? Float32(maximum(r[mb])) : -Inf32
        end
        return sort(bees; by=b -> score[b], rev=true)

    elseif mode == :overall_max
        score = Dict{Int, Float32}()
        for b in bees
            mb = points.bee_id .== b
            score[b] = Float32(maximum(r[mb]))
        end
        return sort(bees; by=b -> score[b], rev=true)

    else
        error("mode must be :final_max or :overall_max")
    end
end


"""
Plot specialization points.

Encodings:
- color_by = :bee   -> color identifies bees
- time_by  = :alpha -> opacity increases with epoch (recommended for 10k epochs)
           = :size  -> marker size increases with epoch
           = :none

Performance:
- trail_step controls downsampling for trails (e.g. 50 => every 50th epoch)
"""
function plot_specialization!(ax::Axis, points::DataFrame;
                              color_by::Symbol = :bee,
                              time_by::Symbol = :alpha,
                              marker_size::Real = 5.0,
                              marker_size_range::Tuple{Real,Real} = (3.0, 8.0),
                              alpha_range::Tuple{Real,Real} = (0.10, 0.95),

                              show_trails::Bool = true,
                              trail_style::Symbol = :fade,
                              trail_step::Int = 50,
                              trail_lw::Real = 1.0,
                              trail_alpha::Real = 0.35,
                              bee_order::Union{Nothing,AbstractVector{<:Integer}} = nothing)

    # sort once for consistent trails
    sort!(points, [:bee_id, :epoch])

    #bees = sort(unique(points.bee_id))
    bees = rank_bees(points; mode=:final_max)  # best first
    if bee_order !== nothing
        bees = bee_order
    end

    # normalize epoch -> t in [0,1]
    emin, emax = extrema(points.epoch)
    denom = max(emax - emin, 1)
    t = Float32.((points.epoch .- emin) ./ denom)

    # choose base color per point
    base_color = if color_by == :bee
        points.bee_id
    elseif color_by == :epoch
        points.epoch
    else
        :black
    end

    # build per-point markersize / alpha if needed
    ms = fill(Float32(marker_size), nrow(points))
    if time_by == :size
        lo, hi = marker_size_range
        ms = Float32.(lo .+ (hi - lo) .* t)
    end

    #pal = Makie.wong_colors()
    pal = pretty_palette(length(bees); min_luma=0.20, max_luma=0.85, sat_range=(0.45, 0.85), light_range=(0.35, 0.70))
    #avoid = [RGB(1,1,1), RGB(0,0,0)]
    #pal = distinguishable_colors(length(bees), avoid)
    #pal = distinct_palette(length(bees); min_luma=0.15, max_luma=0.90)
    bee_to_idx = Dict(b => i for (i,b) in enumerate(bees))

    # Best for "bee + time" with many epochs: color by bee, alpha by time
    if time_by == :alpha && color_by == :bee
        # convert to explicit RGBA so alpha varies per point


        lo, hi = alpha_range

        cols = Vector{RGBAf}(undef, nrow(points))
        @inbounds for i in 1:nrow(points)
            b = points.bee_id[i]
            #c = pal[mod1(bee_to_idx[b], length(pal))]
            c = pal[bee_to_idx[b]]

            a = Float32(lo + (hi - lo) * t[i])
            cols[i] = RGBAf(c.r, c.g, c.b, a)
        end

        scatter!(ax, points.x, points.y; color=cols, markersize=ms)

    else
        # simpler mappings (AoG-like)
        scatter!(ax, points.x, points.y; color=base_color, markersize=ms)
    end

    if show_trails && trail_style != :none
        for b in bees
            mask = points.bee_id .== b
            p = view(points, mask, :)
            t_p = view(t, mask)

            #c = pal[mod1(bee_to_idx[b], length(pal))]
            c = pal[bee_to_idx[b]]

            base_rgb = RGBf(c.r, c.g, c.b)

            if trail_style == :plain
                idx = 1:trail_step:nrow(p)
                lines!(ax, p.x[idx], p.y[idx];
                       color = (base_rgb, trail_alpha),
                       linewidth = trail_lw)

            elseif trail_style == :fade
                fading_trail!(ax, p.x, p.y, t_p;
                              base_rgb = base_rgb,
                              lw = trail_lw,
                              step = trail_step,
                              alpha_range = alpha_range)
            elseif trail_style == :fade_fast
                fading_trail_fast!(ax, p.x, p.y, t_p;
                                  base_rgb = base_rgb,
                                  lw = trail_lw,
                                  alpha_range = alpha_range)
            else
                error("trail_style must be :none, :plain, :fade, or :fade_fast")
            end
        end
    end


    return ax
end

function fading_trail!(ax, x::AbstractVector, y::AbstractVector, t::AbstractVector;
                       base_rgb::RGBf = RGBf(0.5, 0.5, 0.5),
                       lw::Real = 1.0, step::Int = 10,
                       alpha_range::Tuple{Real,Real} = (0.05, 0.8))
    lo, hi = alpha_range
    idx = 1:step:length(x)
    for k in 1:(length(idx)-1)
        i1, i2 = idx[k], idx[k+1]
        a = Float32(lo + (hi-lo) * t[i2])
        col = RGBAf(base_rgb.r, base_rgb.g, base_rgb.b, a)
        lines!(ax, x[i1:i2], y[i1:i2]; color=col, linewidth=lw)
    end
end

function fading_trail_fast!(ax, x::AbstractVector, y::AbstractVector, t::AbstractVector;
                            base_rgb::RGBf = RGBf(0.5, 0.5, 0.5),
                            lw::Real = 1.0,
                            alpha_range::Tuple{Real,Real} = (0.05, 0.8))
    lo, hi = alpha_range
    cols = Vector{RGBAf}(undef, length(x))
    @inbounds for i in eachindex(x)
        a = Float32(lo + (hi - lo) * t[i])
        cols[i] = RGBAf(base_rgb.r, base_rgb.g, base_rgb.b, a)
    end
    lines!(ax, x, y; color=cols, linewidth=lw)
end
