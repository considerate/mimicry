module Mimicry

import Plots
import Zygote
import Flux

include("Arena.jl")
include("Agents.jl")
import .Agents as Agents
import .Arena as Arena


function forward(x :: Vector{Float32}, w :: Matrix{Float32})
    return w * x
end

function main()
    pop_size = 500
    bodies = [Agents.Body(0,0,0) for _ in 1:pop_size]
    for body in bodies
        Agents.randomBody!(body)
    end
    while true
        alive = 0
        #plt = Arena.plot_arena()
        for (i, body) in enumerate(bodies)
            #Agents.plot_body!(body)
            Agents.moveForward!(body)
            if Agents.onTrack(body)
                alive += 1
            else
                Agents.randomBody!(body)
            end
            # end
        end
        println(alive)
        #Plots.display(plt)
        # x :: Vector{Float32} = [0.5, 0.3, 0.2]
        # _, back = Zygote.pullback(forward, x, w)
        # _ = back(1.0)
        # println(y)
        # println((dx,dw))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

Base.@ccallable function julia_main()::Cint
    try
        main()
    catch
        Base.invokelatest(Base.display_error, Base.catch_stack())
        return 1
    end
    return 0
end

end # module
