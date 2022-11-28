# run with julia -i [filename.jl]
#
# or julia -i if that doesn't work (it will be slower to start)

# ignore these comments, they refer to previous versions of the code.

# test01: coming back from notebook version, I'm now attempting to make a small
# Gaussian neural network, which will eventually become the controller
# for the 'cars'.

# test02: I decided it was kind of dumb to have the input but not the output
# be part of the network. Later, stored state will need to be part of the
# controller, but that should be done at that time. So now I'm refactoring to
# remove the distinction between a network and its parameters, and have the
# inputs specified at the time of calculating the gradient.

# test03: when fitting abs(), it never seems to make its knee really sharp,
# even though in principle it could. So I'll try making the 'beta' of the
# soft relu a parameter.
#
# I also discovered after all that that normal ReLUs are fine when implemented
# correctly. I might just use ReLUs in the end.

# test04: a version that gets rid of all the new stuff from test03 and just uses
# relu's, implemented correctly.

# test05: adding in the track and body stuff, which I developed in a notebook.

using Zygote
using Plots
using LibGEOS, GeoInterface
using Profile
using DataStructures

import Flux

const tau=2*pi

# parameters
# ----------

# disagnostic parameters

profiling = false

# agent parameters

const sensorParams = [
    (d, a*tau)
    for d in [0.1, 0.2, 0.3, 0.4, 0.5]
    for a in [0.25,0.15,0.05,-0.05,-0.15,-0.25]
]

# network parameters

const n_feedback_nodes = 10
const n_hidden_nodes = 20

const learning_rate = 0.001

const mem_decay_times = exp.(range(
    log(10.0),
    stop=log(100.0),
    length=n_feedback_nodes
))

const min_std_deviation = 0.01

# physics parameters

const speed = 0.05;
const turnRate = tau/40;

# setting up the track

function dead_ends_track()
    shape = readgeom("
    GEOMETRYCOLLECTION(
        LINESTRING(-1  0,  5  0),
        LINESTRING( 0 -1,  0  3),
        LINESTRING( 5  2, -1  2),
        LINESTRING( 4  3,  4 -1),
        LINESTRING( 2 -1,  2 0.75),
        LINESTRING( 2 1.25,  2 3)
    )
    ")
    return buffer(shape, 0.2, 1)
end

function gmail_track()
    shape = readgeom("LINESTRING(0 0, 0 2, 1 1.25, 2 2, 2 0, 0 0)")
    return buffer(shape, 0.2, 3)
end

const arenaShape = dead_ends_track()

# population parameters

const pop_size = 500




# non-parameter consts
# --------------------

const n_sensors = length(sensorParams)

const min_logvar = 2*log(min_std_deviation)

const arena = prepareGeom(arenaShape)

# layer sizes.  n1 = input, n2 = hidden, n3 = output

const n1 = n_sensors + n_feedback_nodes;
const n2 = n_hidden_nodes;
const n3 = 1 + n_feedback_nodes;

const xmin = LibGEOS.getXMin(arenaShape)
const ymin = LibGEOS.getYMin(arenaShape)
const xmax = LibGEOS.getXMax(arenaShape)
const ymax = LibGEOS.getYMax(arenaShape)


# arena stuff
# -----------

function plot_arena()
    plt = plot(
        arenaShape,
        xlims=(xmin,xmax),
        ylims=(ymin,ymax),
        axis=([],false),
        grid=false,
        color=nothing,
    )
    # holes in a shape aren't plotted so we fake it by plotting them in white
    for coords in LibGEOS.GeoInterface.coordinates(arenaShape)[2:end]
        inner = LibGEOS.Polygon([coords])
        # plot!(inner,color="white")
        plot!(inner,color=nothing)
    end
    return plt
end

# Agent body stuff
# ----------------

struct AgentBody
    x::Float64
    y::Float64
    theta::Float64
end

function onTrack(b::AgentBody)::Bool
    return contains(arena,LibGEOS.Point(b.x,b.y))
end

function randomBody()::AgentBody
    theta = rand()*tau
    x = 0.0
    y = 0.0
    while true
        x = rand()*(xmax-xmin) + xmin
        y = rand()*(ymax-ymin) + ymin
        if contains(arena,LibGEOS.Point(x,y))
            break
        end
    end
    return AgentBody(x,y,theta)
end

function sensorLines(b::AgentBody)
    function lineFromParams(length,angle)
        x0 = b.x
        y0 = b.y
        x1 = x0 + length*sin(angle + b.theta)
        y1 = y0 + length*cos(angle + b.theta)
        return LibGEOS.LineString(Vector{Float64}[[x0,y0],[x1,y1]])
    end
    return [lineFromParams(length,angle) for (length,angle) in sensorParams]
end

function sensorPoints(b::AgentBody)
    # for a possible future performance optimisation - only check the endpoints
    # are within the arena, not the whole line.
    function pointFromParams(length,angle)
        x0 = b.x
        y0 = b.y
        x1 = x0 + length*sin(angle + b.theta)
        y1 = y0 + length*cos(angle + b.theta)
        return LibGEOS.Point([x1,y1])
    end
    return [pointFromParams(length,angle) for (length,angle) in sensorParams]
end

function sensorValues(b::AgentBody)
    sensors = sensorPoints(b)
    return [!contains(arena,sensor) for sensor in sensors]
end

function moveForward(b::AgentBody)
    return AgentBody(
        b.x + speed*sin(b.theta),
        b.y + speed*cos(b.theta),
        b.theta
    )
end

function turn(b::AgentBody, amount)
    # 'amount' can be any real number, but is limited to (-1,1) by a tanh
    return AgentBody(
        b.x,
        b.y,
        b.theta + tanh(amount)*turnRate
    )
end

function plot_body!(b::AgentBody)
    poly = LibGEOS.Polygon([[
                [b.x + 0.05*sin(b.theta), b.y + 0.05*cos(b.theta)],
                [b.x + 0.015*sin(b.theta + tau/3), b.y + 0.015*cos(b.theta + tau/3)],
                [b.x + 0.015*sin(b.theta - tau/3), b.y + 0.015*cos(b.theta - tau/3)],
                [b.x + 0.05*sin(b.theta), b.y + 0.05*cos(b.theta)],
            ]])
    plot!(poly,color="blue",linecolor=nothing)
end

function plot_sensors!(b::AgentBody)
    sensors = sensorPoints(b)
    plot!(sensors,markersize=2,linecolor=nothing,color="red")
end


# Network stuff
# -------------

mutable struct Network
    layer1_w::Matrix{Float64}   # n1 -> n2  (n2 x n1)
    layer1_b::Vector{Float64}   # n2
    mean_w::Matrix{Float64}     # n2 -> n3  (n3 x n2)
    mean_b::Vector{Float64}     # n3
    logvar_w::Matrix{Float64}   # n2 -> n3  (n3 x n2)
    logvar_b::Vector{Float64}   # n3
end


function softplus(x, beta=5.0)
    # I would call this a soft ReLU, but apparently
    # its name is softplus
    if x*beta < -30.0
        return zero(x)
    elseif x*beta > 30.0
        return x
    else
        return log(exp(x.*beta)+1)/beta
    end
end

function relu(x)
    # an actual ReLU function
    if x < -0.0
        return zero(x)
    else
        return x
    end
end

function meanslogvars(net::Network, inputs::Vector{Float64})
    layer1_activations = relu.(net.layer1_w * inputs + net.layer1_b)
    means = net.mean_w * layer1_activations + net.mean_b
    logvars = net.logvar_w * layer1_activations + net.logvar_b
    return (means, logvars)
end

# for regularisation
function l1(net::Network)
    result = 0
    for layer in [net.layer1_w, net.mean_w, net.logvar_w]
        result += sum(abs.(layer))
    end
    return result
end
function l2(net::Network)
    result = 0
    for layer in [net.layer1_w, net.mean_w, net.logvar_w]
        result += sum(layer.^2)
    end
    return result
end


# note: currently sample and loss both run meanslogvars - the
# activations of the hidden layer are calculated twice, as I don't
# know how to store them - fixing this is a possible optimisation.

function sample(net::Network, inputs::Vector{Float64})
    means, logvars = meanslogvars(net, inputs)
    sigma = exp.(logvars*0.5)
    return randn(n3).*sigma + means
end
function loss(net::Network, inputs::Vector{Float64}, outputs::Vector{Float64})
    # inputs of size n1, outputs of size n3.
    # for the moment only a single data point is supported, since that's how
    # it will be run in the lineage learning algorithm
    means, logvars = meanslogvars(net, inputs)
    logvars = max.(logvars,min_logvar)
    squared_deviations = (outputs-means).^2
    gaussian_loss = 0.5*sum(
        squared_deviations .* exp.(-logvars) + logvars
    )
    return gaussian_loss #+ 0.01*l2(net)
end

# for testing and initialisation
function random_network()
    layer1_w = randn((n2,n1)) * (1.0/n1)
    layer1_b = randn(n2) * 0
    mean_w = randn((n3,n2)) * (1.0/n2)
    mean_b = randn(n3) * 0
    logvar_w = randn((n3,n2)) * (1.0/n2)
    logvar_b = randn(n3) * 0
    return Network(
        layer1_w,
        layer1_b,
        mean_w,
        mean_b,
        logvar_w,
        logvar_b .- 1
    )
end

function network_gradient(
        net::Network,
        inputs::Vector{Float64},
        outputs::Vector{Float64}
    )
    return gradient(
        () -> loss(net, inputs, outputs),
        Params([
            getfield(net,name)
            for name in fieldnames(Network)
        ])
    )
end

# not used any more - see in Agent
function update!(
        net::Network,
        inputs::Vector{Float64},
        outputs::Vector{Float64},
        lrate::Float64
    )
    grads = network_gradient(net, inputs, outputs)
    for name in fieldnames(Network)
        field = getfield(net, name)
        setfield!(net, name, field - lrate*grads[field])
    end
end

# Agent stuff
# -----------

mutable struct Agent
    feedback_nodes::Array{Float64} # n_feedback_nodes
    network::Network
    body::AgentBody
    optimiser # Flux optimisers all have different types
end

function Agent()
    return Agent(
        zeros(n_feedback_nodes),
        random_network(),
        randomBody(),
        Flux.Optimise.Adam(learning_rate)
    )
end

function alive(agent::Agent)
    return onTrack(agent.body)
end

function step(agent::Agent)
    # possible future (micro-)optimisation: this currently updates the network
    # even if the agent hit the edge - that could be avoided
    sensors = sensorValues(agent.body)
    inputs = [sensors; agent.feedback_nodes*1.0]
    outputs = sample(agent.network, inputs)

    output = outputs[1]
    if isnan(output)
        # we take a zero-tolerance approach to NaNs here - if you output one
        # you are immediately teleported outside the arena and die.
        agent.body = AgentBody(-1000.0,-1000.0,0.0)
        output = 0
    end
    feedback = outputs[2:end]
    agent.feedback_nodes = (
        agent.feedback_nodes.*(1.0 .- 1.0./mem_decay_times)
        + feedback.*(1.0./mem_decay_times)
    )
    agent.body = moveForward(turn(agent.body,output))
    return output
end

function update(agent::Agent)
    # possible future (micro-)optimisation: this currently updates the network
    # even if the agent hit the edge - that could be avoided
    sensors = sensorValues(agent.body)
    inputs = [sensors; agent.feedback_nodes*1.0]
    outputs = sample(agent.network, inputs)
    grads = network_gradient(agent.network, inputs, outputs)
    for name in fieldnames(Network)
        Flux.update!(
            agent.optimiser,
            getfield(agent.network, name),
            grads[getfield(agent.network,name)]
        )
    end

    output = outputs[1]
    if isnan(output)
        # we take a zero-tolerance approach to NaNs here - if you output one
        # you are immediately teleported outside the arena and die.
        agent.body = AgentBody(-1000.0,-1000.0,0.0)
        output = 0
    end
    feedback = outputs[2:end]
    agent.feedback_nodes = (
        agent.feedback_nodes.*(1.0 .- 1.0./mem_decay_times)
        + feedback.*(1.0./mem_decay_times)
    )
    agent.body = moveForward(turn(agent.body,output))
    return output
end

function mimic_loss(net::Network, inputs::Vector{Float64},outputs::Vector{Float64}, target::Float64)
    means, logvars = meanslogvars(net, inputs)
    logvars = max.(logvars,min_logvar)
    squared_deviation = ((means[1]-outputs[1]) - target) ^ 2
    gaussian_loss = 0.5*sum(squared_deviation * exp(-logvars[1]) + logvars[1])
    return gaussian_loss #+ 0.01*l2(net)
end

function mimicry_gradient(
        net::Network,
        inputs::Vector{Float64},
        outputs::Vector{Float64},
        target::Float64,
    )
    return gradient(
        () -> mimic_loss(net, inputs, outputs, target),
        Params([
            getfield(net,name)
            for name in fieldnames(Network)
        ])
    )
end

# train an agent to mimic the outputs in a chronological trajectory of some other agent
function mimic(agent::Agent, trajectory::Array{Tuple{AgentBody,Float64}})
    if size(trajectory)[1] == 0
        return
    end
    original_feedback = copy(agent.feedback_nodes)
    (last_body, last_turning) = pop!(trajectory)
    # replay trajectory
    for (body, _) in trajectory
        sensors = sensorValues(body)
        inputs = [sensors; agent.feedback_nodes*1.0]
        outputs = sample(agent.network, inputs)
        feedback = outputs[2:end]
        agent.feedback_nodes = (
            agent.feedback_nodes.*(1.0 .- 1.0./mem_decay_times)
            + feedback.*(1.0./mem_decay_times)
        )
    end
    sensors = sensorValues(last_body)
    inputs = [sensors; agent.feedback_nodes*1.0]
    outputs = sample(agent.network, inputs)
    # train on last step of trajectory
    grads = mimicry_gradient(agent.network, inputs, outputs, last_turning)
    for name in fieldnames(Network)
        Flux.update!(
            agent.optimiser,
            getfield(agent.network, name),
            grads[getfield(agent.network,name)]
        )
    end
    agent.feedback_nodes = original_feedback
end

function pretty_print(a::Agent)
    println("feedback nodes: $(a.feedback_nodes)")
    for name in fieldnames(Network)
        println("$name: $(getfield(a.network,name))")
    end
end

# Population stuff and main loop
# ------------------------------


function main(run_once = false)
    Base.exit_on_sigint(true)
    population::Array{Agent} = [Agent() for _ in 1:pop_size]
    max_len = 20
    history = CircularBuffer{Array{Tuple{Int64, AgentBody, Float64}}}(max_len)
    iteration = 1
    current::Array{Int64} = [i for i in 1:pop_size]
    outputs::Array{Float64} = [0.0 for _ in 1:pop_size]
    while true
        iteration += 1
        for _ in 1:4
            outputs = [0.0 for _ in 1:pop_size]
            current = [i for i in 1:pop_size]
            for (i, agent) in enumerate(population)
                outputs[i] = update(agent)
                # update(agent)
            end
            all_alive = false
            while !all_alive
                all_alive = true
                all_dead = true
                for k in 1:length(population)
                    # future improvement: this is a fairly dumb way to do things
                    if !alive(population[k])
                        all_alive = false
                        neighbour_index = mod1(k+rand([-1,1]),pop_size)
                        neighbour = population[neighbour_index]
                        # neighbour = population[rand(1:pop_size)]
                        if alive(neighbour)
                            population[k] = deepcopy(neighbour)
                            current[k] = neighbour_index
                        end
                    else
                        all_dead = false
                    end
                end
                if all_dead
                    for agent in population
                        agent.body = randomBody()
                    end
                    all_dead = false
                    all_alive = true
                end
            end
        end
        positions = [agent.body for agent in population]
        pushfirst!(history, collect(zip(current, positions, outputs)))
        trajectory:: Array{Tuple{AgentBody, Float64}} = []
        for agent in population
            trajectory = []
            index = rand(1:pop_size)
            for t in 1:length(history)
                parent, body, output = history[t][index]
                pushfirst!(trajectory, (body, output))
                index = parent
            end
            if size(trajectory)[1] > 0
                mimic(agent, trajectory[1:length(history)÷2])
            end
        end
        GC.gc()
        plt = plot_arena()
        for agent in population
            plot_body!(agent.body)
        end
        plot_sensors!(population[1].body)
        display(plt)
        println(population[1].feedback_nodes)
        if run_once
            break
        end
    end
end

if profiling
    main(true)
    @profile main()
else
    main()
end
