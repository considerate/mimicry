module Agents

include("Arena.jl")
import .Arena as Arena

import Plots
import LibGEOS

const tau=2*pi

mutable struct Body
    x::Float64
    y::Float64
    theta::Float64
end

# mutable struct Agent
#     network::Network
#     memory::Array{Float64}
#     body::Body
#     optimiser # Flux optimisers all have different types
# end

function randomBody!(b::Body)
    theta = rand()*tau
    x = 0.0
    y = 0.0
    while true
        (x,y) = Arena.randomPoint()
        break
        # if Arena.onTrack(x, y)
        #     break
        # end
    end
    b.x = x
    b.y = y
    b.theta = theta
end

function onTrack(b::Body)
    return Arena.onTrack(b.x, b.y)
end

function moveForward!(b::Body)
    speed = 0.05;
    b.x += speed*sin(b.theta)
    b.y += speed*cos(b.theta)
end


function plot_body!(b::Body)
    poly = LibGEOS.Polygon([[
                [b.x + 0.05*sin(b.theta), b.y + 0.05*cos(b.theta)],
                [b.x + 0.015*sin(b.theta + tau/3), b.y + 0.015*cos(b.theta + tau/3)],
                [b.x + 0.015*sin(b.theta - tau/3), b.y + 0.015*cos(b.theta - tau/3)],
                [b.x + 0.05*sin(b.theta), b.y + 0.05*cos(b.theta)],
            ]])
    Plots.plot!(poly,color="blue",linecolor=nothing)
end

end
