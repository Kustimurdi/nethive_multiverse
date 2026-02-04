function matrices_by_epoch(df::DataFrame, epochs::AbstractVector{<:Integer};
                           valuecol::Symbol=:accuracies, fill=NaN)
    mats = Dict{Int, Matrix{Float64}}()

    # Use the first epoch to fix ordering and dimensions
    m0, bee_ids, task_ids = state_matrix(df, epochs[1]; valuecol=valuecol, fill=fill)
    mats[Int(epochs[1])] = m0

    # Remaining epochs; enforce same ordering by re-unstacking on full grid
    for e in epochs[2:end]
        m, b, t = state_matrix(df, e; valuecol=valuecol, fill=fill)

        # sanity checks (so plots don't silently compare mismatched orderings)
        b == bee_ids || throw(ArgumentError("bee_id ordering differs at epoch=$e"))
        t == task_ids || throw(ArgumentError("task_id ordering differs at epoch=$e"))

        mats[Int(e)] = m
    end

    return mats, bee_ids, task_ids
end

function task_permutation_greedy(mat::AbstractMatrix{<:Real})
    n_bees, n_tasks = size(mat)
    k = min(n_bees, n_tasks)

    used = falses(n_tasks)
    matched = Int[]

    for i in 1:k
        # pick best remaining task for this bee
        bestj, bestv = 0, -Inf
        for j in 1:n_tasks
            if !used[j] && mat[i, j] > bestv
                bestv = mat[i, j]
                bestj = j
            end
        end
        push!(matched, bestj)
        used[bestj] = true
    end

    rest = [j for j in 1:n_tasks if !used[j]]
    vcat(matched, rest)
end

function apply_col_permutation(mats::Dict{Int, Matrix{Float64}}, perm::Vector{Int})
    Dict(e => m[:, perm] for (e, m) in mats)
end

"""
    plot_state_heatmaps_makie(matsA, matsB, epochs; bee_ids=nothing, task_ids=nothing,
                              row_titles=("dfA","dfB"), colormap=:viridis, clim=:auto)

- matsA/matsB: Dict(epoch => Matrix)  (same shapes)
- epochs: e.g. [0, 100, 500, 2000]
- bee_ids/task_ids: optional vectors used as tick labels
- clim=:auto -> shared colorrange from all matrices (ignores NaNs)
Returns the Makie Figure.
"""
function plot_state_heatmaps_makie(matsA::Dict{Int,<:AbstractMatrix},
                                   matsB::Dict{Int,<:AbstractMatrix},
                                   epochs::AbstractVector{<:Integer};
                                   bee_ids=nothing,
                                   task_ids=nothing,
                                   row_titles=("dfA", "dfB"),
                                   colormap=:viridis,
                                   clim=:auto)

    ne = length(epochs)

    # --- shared color limits (ignore NaNs) ---
    if clim === :auto
        vals = Float64[]
        for e_ in epochs
            e = Int(e_)
            append!(vals, vec(matsA[e]))
            append!(vals, vec(matsB[e]))
        end
        vals = filter(!isnan, vals)
        cmin, cmax = minimum(vals), maximum(vals)
    else
        cmin, cmax = clim
    end
    colorrange = (cmin, cmax)

    # --- figure layout ---
    fig = Figure(resolution = (320*ne + 140, 650))
    

    # global axis labels
    fig[3, 1:ne] = Label(fig, "Agents", fontsize=24)
    fig[1:2, 0]  = Label(fig, "Tasks", rotation=pi/2, fontsize=24)

    hm_ref = nothing

    max_e = maximum(epochs)

    for (j, e_) in enumerate(epochs)
        e = Int(e_)

        # row 1
        ax1 = Axis(fig[1, j], title = "epoch $e", aspect = DataAspect()) #$(row_titles[1]) • 
        hm1 = heatmap!(ax1, matsA[e]; colormap=colormap, colorrange=colorrange)
        hm_ref === nothing && (hm_ref = hm1)

        # row 2
        ax2 = Axis(fig[2, j], title = "epoch $(e+max_e)", aspect = DataAspect()) #$(row_titles[2]) • 
        heatmap!(ax2, matsB[e]; colormap=colormap, colorrange=colorrange)
        
        for ax in (ax1, ax2)
            ax.xticksvisible = false
            ax.yticksvisible = false
            ax.xticklabelsvisible = false
            ax.yticklabelsvisible = false
        end


        # ticks: only show left y ticks and bottom x ticks (cleaner)
        #if bee_ids !== nothing
            #yt = (1:length(bee_ids), string.(bee_ids))
            #ax1.yticks = yt
            #ax2.yticks = yt
        #end
        #if task_ids !== nothing
            #xt = (1:length(task_ids), string.(task_ids))
            #ax1.xticks = xt
            #ax2.xticks = xt
        #end

        ## hide redundant decorations
        #if j != 1
            #ax1.yticklabelsvisible = false
            #ax2.yticklabelsvisible = false
        #end
        #ax1.xticklabelsvisible = false  # top row: hide x tick labels
    end
    #rowsize!(fig.layout, 1, Relative(2))
    #rowsize!(fig.layout, 2, Relative(2))

    # one shared colorbar on the right spanning both rows
    Colorbar(fig[1:2, ne+1], hm_ref, label = "Accuracy", labelsize=28)#, size=25)


    return fig
end


function plot_state_heatmaps(matsA::Dict{Int,<:AbstractMatrix},
                             matsB::Dict{Int,<:AbstractMatrix},
                             epochs::AbstractVector{<:Integer};
                             colormap=:viridis,
                             clim=:auto)

    ne = length(epochs)
    labelsize = 32
    ticklabelsize = 28

    # shared color limits (ignore NaNs)
    if clim === :auto
        vals = Float64[]
        for e_ in epochs
            e = Int(e_)
            append!(vals, vec(matsA[e]))
            append!(vals, vec(matsB[e]))
        end
        vals = filter(!isnan, vals)
        cmin, cmax = minimum(vals), maximum(vals)
    else
        cmin, cmax = clim
    end
    colorrange = (cmin, cmax)

    fig = Figure(size = (320*ne + 120, 650))

    fig[3, 1:ne] = Label(fig, L"\text{Agents}", fontsize=labelsize)
    fig[1:2, 0]  = Label(fig, L"\text{Tasks}", rotation=pi/2, fontsize=labelsize)

    hm_ref = nothing
    max_e = maximum(epochs)

    for (j, e_) in enumerate(epochs)
        e = Int(e_)

        ax1 = Axis(fig[1, j], title=L"\text{epoch } %$e", titlesize=labelsize, aspect=DataAspect())
        hm1 = heatmap!(ax1, matsA[e]; colormap=colormap, colorrange=colorrange)
        hm_ref === nothing && (hm_ref = hm1)

        ax2 = Axis(fig[2, j], title=L"\text{epoch } %$(e + max_e)", titlesize=labelsize, aspect=DataAspect())
        heatmap!(ax2, matsB[e]; colormap=colormap, colorrange=colorrange)

        for ax in (ax1, ax2)
            ax.xticksvisible = false
            ax.yticksvisible = false
            ax.xticklabelsvisible = false
            ax.yticklabelsvisible = false
        end
    end

    # ✅ now row 1 and row 2 exist, so this is safe
    #rowsize!(fig.layout, 1, Fixed(300))
    rowsize!(fig.layout, 2, Fixed(300))
    rowgap!(fig.layout, 10)
    colgap!(fig.layout, 10)

    cb = Colorbar(fig[1:2, ne+1], hm_ref; label=L"\text{Accuracy}", width=18, labelsize=labelsize, ticklabelsize=ticklabelsize)
    cb.ticks = ([0.0, 0.2, 0.4, 0.6, 0.8], [L"0.0", L"0.2", L"0.4", L"0.6", L"0.8"]) #std accuracy

    return fig
end

function permute_mats(mats::AbstractDict{<:Integer,<:AbstractMatrix},
                      perm::AbstractVector{<:Integer};
                      axis::Symbol = :cols)

    axis in (:rows, :cols) || throw(ArgumentError("axis must be :rows or :cols"))

    out = Dict{Int, Matrix{Float64}}()
    for (e, M) in mats
        if axis === :rows
            length(perm) == size(M, 1) || throw(DimensionMismatch("perm length != nrows for epoch=$e"))
            out[Int(e)] = Matrix(M[perm, :])
        else
            length(perm) == size(M, 2) || throw(DimensionMismatch("perm length != ncols for epoch=$e"))
            out[Int(e)] = Matrix(M[:, perm])
        end
    end
    return out
end

permute_ids(ids::AbstractVector, perm::AbstractVector{<:Integer}) = ids[perm]
