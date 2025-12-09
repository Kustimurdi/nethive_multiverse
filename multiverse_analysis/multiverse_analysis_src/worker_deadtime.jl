function getting_the_final_states(log::DataFrame)
    
end

function classify_dominant_bee(log::DataFrame)
    bee_ids = unique(vcat(log.bee1_id, log.bee2_id))
    highest_score = 0.0
    score_bee_id = 0
    for id in bee_ids
        
end

function calc_dead_time_fraction()
end

function calc_avg_dead_time_fraction()
end

