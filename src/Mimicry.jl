module Mimicry

import UnicodePlots
import Zygote
import Flux
import PolygonOps
import StaticArrays
import TERMIOS
using Printf

const tau=2*pi
const Polygon = Vector{StaticArrays.SVector{2, Float64}}
const Point = Tuple{Float64, Float64}
const Polar = Tuple{Float64, Float64}
const Bounds = Tuple{Point, Point}
const Arena = Tuple{Polygon, Polygon, Bounds}

function cleanup()
    print("\033[?25h") # show cursor
end

struct Network
    layer1_w::Matrix{Float64}        # n1 -> n2  (n2 x n1)
    layer1_b::Vector{Float64}        # n2
    mean_w::Matrix{Float64}          # n2 -> n3  (n3 x n2)
    mean_b::Vector{Float64}          # n3
    logvar_w::Matrix{Float64}        # n2 -> n3  (n3 x n2)
    logvar_b::Vector{Float64}        # n3
end

struct Sizes
    n1 :: Int64
    n2 :: Int64
    n3 :: Int64
    n_sensors :: Int64
    n_feedback_nodes :: Int64
end

const AgentParams = Tuple{Vector{Polar}, Sizes}

struct Body
    x::Float64
    y::Float64
    theta::Float64
end

# A Car decides:
# - what angle to turn using a gaussian
mutable struct Car
    feedback_nodes::Vector{Float64} # n_feedback_nodes
    network::Network
    body::Body
    optimiser :: Flux.Adam
end

# A Predator decides:
# - x movement speed using a gaussian (tanh)
# - y movement speed using a gaussian (tanh)
# - whether to attack or not (sigmoid)
mutable struct Predator
    feedback_nodes::Vector{Float64}
    network::Network
    hunger::Int64 # negative hunger implies death
    attacking::Int64
    health :: Int64
    body::Body
    optimiser :: Flux.Adam
end

# A Predator decides:
# - x movement speed using a gaussian (tanh)
# - y movement speed using a gaussian (tanh)
# - whether to attack/eat food or not (sigmoid)
mutable struct Prey
    feedback_nodes::Vector{Float64}
    network::Network
    hunger::Int64
    attacking::Int64
    health :: Int64
    body::Body
    optimiser :: Flux.Adam
end

mutable struct Food
    hunger :: Int64
    health :: Int64
    body :: Body
end

function randomnetwork(s)
    layer1_w = randn((s.n2,s.n1)) * sqrt(2.0/s.n1)
    layer1_b = zeros(s.n2)
    mean_w = randn((s.n3,s.n2)) * sqrt(2.0/s.n2)
    mean_b = zeros(s.n3)
    logvar_w = randn((s.n3,s.n2)) * sqrt(2.0/s.n2)
    logvar_b = zeros(s.n3)
    return Network(
        layer1_w,
        layer1_b,
        mean_w,
        mean_b,
        logvar_w,
        logvar_b
    )
end


function Car(learning_rate :: Float64, params :: AgentParams, arena :: Arena)
    (_, sizes) = params
    return Car(
        randn(sizes.n_feedback_nodes) * (1.0/sizes.n_feedback_nodes),
        randomnetwork(sizes),
        randomBody(arena),
        Flux.Optimise.Adam(learning_rate)
    )
end

function gaussloss(means::Vector{Float64},logvars::Vector{Float64},outputs::Vector{Float64}) ::Float64
    squared_deviations = (outputs-means).^2
    gaussian_loss = 0.5*sum(squared_deviations .* exp.(-logvars) + logvars)/length(outputs)
    return gaussian_loss #+ 0.01*l2(net)
end

function relu(x)
    return max(x, 0)
end

function network(net :: Network, inputs :: Vector{Float64}) ::Tuple{Vector{Float64},Vector{Float64}}
    min_std_deviation = 0.01
    min_logvar = 2*log(min_std_deviation)
    layer1_activations = relu.(net.layer1_w * inputs + net.layer1_b)
    means = net.mean_w * layer1_activations + net.mean_b
    logvars = max.(net.logvar_w * layer1_activations + net.logvar_b, min_logvar)
    return (means, logvars)
end

function randompoint(bounds::Bounds) :: Point
    ((xmin,xmax), (ymin,ymax)) = bounds
    x = rand()*(xmax-xmin) + xmin
    y = rand()*(ymax-ymin) + ymin
    return (x,y)
end

function randomBody(arena :: Arena) :: Body
    (_, _, bounds) = arena
    theta = rand()*tau
    while true
        p = randompoint(bounds)
        if ontrack(p, arena)
            (x,y) = p
            return Body(x, y, theta)
        end
    end
end

function turn(b :: Body, amount)
    # 'amount' can be any real number, but is limited to (-1,1) by a tanh
    turnRate = tau/40;
    return Body(
        b.x,
        b.y,
        b.theta + tanh(amount)*turnRate,
    )
end


function moveForward(b::Body)
    speed = 0.05;

    return Body(
        b.x + speed*sin(b.theta),
        b.y + speed*cos(b.theta),
        b.theta,
    )
end

function createArena() :: Arena
    # TODO: compute this
    polygon :: Polygon = [
      (-1.2, 2.0),
      (-1.0, 2.2),
      (-0.2, 2.2),
      (-0.2, 3.0),
      ( 0.0, 3.2),
      ( 0.2, 3.0),
      ( 0.2, 2.2),
      ( 1.8, 2.2),
      ( 1.8, 3.0),
      ( 2.0, 3.2),
      ( 2.2, 3.0),
      ( 2.2, 2.2),
      ( 3.8, 2.2),
      ( 3.8, 3.0),
      ( 4.0, 3.2),
      ( 4.2, 3.0),
      ( 4.2, 2.2),
      ( 5.0, 2.2),
      ( 5.2, 2.0),
      ( 5.0, 1.8),
      ( 4.2, 1.8),
      ( 4.2, 0.2),
      ( 5.0, 0.2),
      ( 5.2, 0.0),
      ( 5.0,-0.2),
      ( 4.2,-0.2),
      ( 4.2,-1.0),
      ( 4.0,-1.2),
      ( 3.8,-1.0),
      ( 3.8,-0.2),
      ( 2.2,-0.2),
      ( 2.2,-1.0),
      ( 2.0,-1.2),
      ( 1.8,-1.0),
      ( 1.8,-0.2),
      ( 0.2,-0.2),
      ( 0.2,-1.0),
      ( 0.0,-1.2),
      (-0.2,-1.0),
      (-0.2,-0.2),
      (-1.0,-0.2),
      (-1.2,-0.0),
      (-1.0, 0.2),
      (-0.2, 0.2),
      (-0.2, 1.8),
      (-1.0, 1.8),
      (-1.2, 2.0),
    ]
    inner :: Polygon = [
      (0.2, 1.8),
      (1.8, 1.8),
      (1.8, 1.4),
      (2.0, 1.2),
      (2.2, 1.4),
      (2.2, 1.8),
      (3.8, 1.8),
      (3.8, 0.2),
      (2.2, 0.2),
      (2.2, 0.6),
      (2.0, 0.8),
      (1.8, 0.6),
      (1.8, 0.2),
      (0.2, 0.2),
      (0.2, 1.8),
    ]
    xmin = min(minimum(p -> p[1], polygon), minimum(p -> p[1], inner))
    xmax = max(maximum(p -> p[1], polygon), maximum(p -> p[1], inner))
    ymin = min(minimum(p -> p[2], polygon), minimum(p -> p[2], inner))
    ymax = max(maximum(p -> p[2], polygon), maximum(p -> p[2], inner))
    return (polygon, inner, ((xmin, xmax), (ymin, ymax)))
end

function disable_echo()
    backup_termios = TERMIOS.termios()
    TERMIOS.tcgetattr(stdin, backup_termios)
    termios = TERMIOS.termios()
    TERMIOS.tcgetattr(stdin, termios)
     # Disable echo
    termios.c_lflag &= ~TERMIOS.ECHO
    TERMIOS.tcsetattr(stdin, TERMIOS.TCSANOW, termios)
    return (termios, backup_termios)
end

function draw_scene(arena :: Arena, bodies :: Vector{Body})
    width = 121
    height = 44
    (outer, inner, _) = arena
    canvas = UnicodePlots.BrailleCanvas(height, width, origin_y=-1.2, origin_x=-1.2, height=4.4, width=6.4)
    for (before, after) in zip(outer[1:end-1], outer[2:end])
        UnicodePlots.lines!(canvas, before[1], before[2], after[1], after[2]; color=:cyan)
    end
    for (before, after) in zip(inner[1:end-1], inner[2:end])
        UnicodePlots.lines!(canvas, before[1], before[2], after[1], after[2]; color=:cyan)
    end
    plt = UnicodePlots.Plot(canvas)
    UnicodePlots.scatterplot!(plt, [b.x for b in bodies], [b.y for b in bodies], width=80, height=height)
    return (plt, (width, height))
end

function ontrack(p, arena :: Arena)
    (outer, inner, _) = arena
    return PolygonOps.inpolygon(p, outer) == 1 && PolygonOps.inpolygon(p, inner) == 0
end

function sample(means::Vector{Float64},logvars::Vector{Float64})::Vector{Float64}
    sigma = exp.(logvars*0.5)
    return randn(length(means)).*sigma + means
end

function train(net::Network, inputs::Vector{Float64})::Tuple{Tuple{Float64, Vector{Float64},Vector{Float64}, Vector{Float64}}, Any}
    (means, logvars), dforward = Zygote.pullback(network, net, inputs)
    outputs = sample(means,logvars)
    loss, dloss = Zygote.pullback(gaussloss, means, logvars, outputs)
    (dmean, dlogvars, _) = dloss(1.0)
    (grads, _) = dforward((dmean, dlogvars))
    g2 = Zygote._project(net, grads)
    return ((loss, means, logvars, outputs), g2)
end


function updatecar(agent::Car, params :: AgentParams, arena :: Arena)
    # possible future (micro-)optimisation: this currently updates the network
    # even if the agent hit the edge - that could be avoided
    sensors = sensorValues(agent.body, params, arena)
    inputs = [sensors; agent.feedback_nodes*1.0]
    (loss, _, _, outputs), grads = train(agent.network, inputs)
    for name in fieldnames(Network)
        param = getfield(agent.network, name)
        grad = grads[name] # grads[getfield(agent.network,name)]
        Flux.update!(agent.optimiser, param, grad)
    end

    output = outputs[1]
    if isnan(output)
        # we take a zero-tolerance approach to NaNs here - if you output one
        # you are immediately teleported outside the arena and die.
        agent.body = Body(-1000.0,-1000.0,0.0)
        output = 0
    end
    feedback = outputs[2:end]
    mem_decay_times = exp.(range(
        log(10.0),
        stop=log(100.0),
        length=params[2].n_feedback_nodes
    ))
    agent.feedback_nodes = (
        agent.feedback_nodes.*(1.0 .- 1.0./mem_decay_times)
        + feedback.*(1.0./mem_decay_times)
    )
    agent.body = moveForward(turn(agent.body,output))
    return (outputs, loss)
end

function update(bodies, arena :: Arena)
    for (i, body) in enumerate(bodies)
        nextBody = moveForward(body)
        alive = ontrack((nextBody.x, nextBody.y), arena)
        if !alive
            nextBody = randomBody(arena)
        end
        bodies[i] = nextBody
    end
end

function replicatenetwork(source :: Network, target :: Network)
    target.logvar_w[:,:] .= source.logvar_w
    target.logvar_b[:] .= source.logvar_b
    target.mean_w[:,:] .= source.mean_w
    target.mean_b[:] .= source.mean_b
    target.layer1_w[:,:] .= source.layer1_w
    target.layer1_b[:] .= source.layer1_b
end

function replicatecar(source :: Car, target :: Car, arena :: Arena, sizes :: Sizes)
    target.optimiser.eta = source.optimiser.eta
    target.optimiser.beta = source.optimiser.beta
    target.optimiser.state = IdDict()

    if rand() < 0.01
        network = randomnetwork(sizes)
        feedback = zeros(size(target.feedback_nodes))
        target.body = randomBody(arena)
    else
        network = source.network
        feedback = source.feedback_nodes
        target.body = source.body
    end
    replicatenetwork(network, target.network)

    target.feedback_nodes[:] .= feedback
end


function sensorPoints(b::Body, sensorParams :: Vector{Polar}) :: Vector{Point}
    # for a possible future performance optimisation - only check the endpoints
    # are within the arena, not the whole line.
    function pointFromParams(length,angle)
        x0 = b.x
        y0 = b.y
        x1 = x0 + length*sin(angle + b.theta)
        y1 = y0 + length*cos(angle + b.theta)
        return (x1,y1)
    end
    return [pointFromParams(length,angle) for (length,angle) in sensorParams]
end

function sensorValues(b::Body, params :: AgentParams, arena :: Arena) :: Vector{Float64}
    (sensorParams, _) = params
    points = sensorPoints(b, sensorParams)
    return [!ontrack(p, arena) for p in points]
end

function agentparams() :: AgentParams
    sensorParams :: Vector{Polar} = [
        (d, a*tau)
        for d in [0.1, 0.2, 0.3, 0.4, 0.5]
        for a in [0.25,0.15,0.05,-0.05,-0.15,-0.25]
    ]
    n_feedback_nodes = 10
    n_hidden_nodes = 20
    n_sensors = length(sensorParams)
    n1 = n_sensors + n_feedback_nodes;
    n2 = n_hidden_nodes;
    n3 = 1 + n_feedback_nodes;
    sizes = Sizes(n1, n2, n3, n_sensors, n_feedback_nodes)
    return (sensorParams, sizes)
end

function PredatorPrey(learning_rate, params, bounds, f)
    (_, sizes) = params
    (x,y) = randompoint(bounds)
    theta = rand()*tau
    network = randomnetwork(sizes)
    feedback =  randn(sizes.n_feedback_nodes) * (1.0/sizes.n_feedback_nodes)
    body = Body(x,y,theta)
    optimiser = Flux.Optimise.Adam(learning_rate)
    hunger = rand(80:100)
    health = 100
    attacking = 0
    return f(
        feedback,
        network,
        hunger,
        attacking,
        health,
        body,
        optimiser
    )
end

function Predator(learning_rate, params, bounds) :: Predator
    return PredatorPrey(learning_rate, params, bounds, Predator)
end

function Prey(learning_rate, params, bounds) :: Prey
    return PredatorPrey(learning_rate, params, bounds, Prey)
end

function Food(bounds) :: Food
    hunger = rand(20:60)
    health = 100
    (x,y) = randompoint(bounds)
    theta = rand()*tau
    body = Body(x,y,theta)
    return Food(hunger, health, body)
end

function draw_animals(predators :: Vector{Predator}, prey :: Vector{Prey}, food :: Vector{Food}, bounds :: Bounds)
    ((xmin, xmax), (ymin, ymax)) = bounds
    width=121
    height=44
    canvas = UnicodePlots.BrailleCanvas(height, width, origin_y=ymin, origin_x=xmin, height=ymax - ymin, width=xmax - xmin)
    plt = UnicodePlots.Plot(canvas)
    UnicodePlots.scatterplot!(plt, [p.body.x for p in predators], [p.body.y for p in predators], width=width, height=height, color=:red)
    UnicodePlots.scatterplot!(plt, [p.body.x for p in prey], [p.body.y for p in prey], width=width, height=height, color=:green)
    UnicodePlots.scatterplot!(plt, [p.body.x for p in food], [p.body.y for p in food], width=width, height=height, color=:blue)
    return (plt, (width, height))
end

function replicatepredatorprey(source, target, params :: AgentParams, bounds :: Bounds)
    (_, sizes) = params
    target.optimiser.eta = source.optimiser.eta
    target.optimiser.beta = source.optimiser.beta
    target.optimiser.state = IdDict()

    if rand() < 0.01
        network = randomnetwork(sizes)
        feedback = zeros(size(target.feedback_nodes))
        (x,y) = randompoint(bounds)
        theta = rand()*tau
        target.body = Body(x,y,theta)
    else
        network = source.network
        feedback = source.feedback_nodes
        target.body = source.body
    end
    replicatenetwork(network, target.network)

    target.feedback_nodes[:] .= feedback
    target.hunger = rand(80:100)
    target.health = 100
    target.attacking = 0
end

function animals()
    bounds :: Bounds = ((0, 10), (0, 10))
    params = agentparams()
    (n_predators, n_prey, n_food) = (20, 100, 100)
    predators = [Predator(1e-4, params, bounds) for _ in 1:n_predators]
    prey = [Prey(1e-4, params, bounds) for _ in 1:n_prey]
    food = [Food(bounds) for _ in 1:n_food]
    last_print = 0
    while true
        for (k, predator) in enumerate(predators)
            predator.hunger -= 1
            if predator.hunger <= 0 || predator.health <= 0
                neighbour_index = mod1(k+rand([-1,1]),n_predators)
                neighbour = predators[neighbour_index]
                while neighbour.hunger <= 0 || neighbour.health <= 0
                    k = neighbour
                    neighbour_index = mod1(k+rand([-1,1]),n_predators)
                    neighbour = predators[neighbour_index]
                end
                replicatepredatorprey(neighbour, predator, params, bounds)
            end
        end
        for (k, p) in enumerate(prey)
            p.hunger -= 1
            if p.hunger <= 0 || p.health <= 0
                neighbour_index = mod1(k+rand([-1,1]), n_prey)
                neighbour = prey[neighbour_index]
                while neighbour.hunger <= 0 || neighbour.health <= 0
                    k = neighbour
                    neighbour_index = mod1(k+rand([-1,1]), n_predators)
                    neighbour = prey[neighbour_index]
                end
                replicatepredatorprey(neighbour, p, params, bounds)
            end
        end
        current = time_ns()
        if current - last_print > 0.05e9
            (plt, (_, height)) = draw_animals(predators, prey, food, bounds)
            println(UnicodePlots.show(plt))
            print("\033[s") # save cursor
            print(string("\033[",height+2,"A")) # move up height+2 lines
            last_print = current
        end
    end
end

function cars()
    arena = createArena()
    params = agentparams()
    pop_size = 500
    agents = [Car(1e-4, params, arena) for _ in 1:pop_size]
    prev = time_ns()
    last_print = 0
    tpf = 0.001
    while true
        for (k, agent) in enumerate(agents)
            updatecar(agent, params, arena)
            alive = ontrack((agent.body.x, agent.body.y), arena)
            if !alive
                neighbour_index = mod1(k+rand([-1,1]),length(agents))
                neighbour = agents[neighbour_index]
                while !ontrack((neighbour.body.x, neighbour.body.y), arena)
                    k = neighbour
                    neighbour_index = mod1(k+rand([-1,1]),length(agents))
                    neighbour = agents[neighbour_index]
                end
                (_, sizes) = params
                replicatecar(neighbour, agent, arena, sizes)
            end
        end
        current = time_ns()
        if current - last_print > 0.05e9
            (plt, (_, height)) = draw_scene(arena, [agent.body for agent in agents])
            println(UnicodePlots.show(plt))
            @printf "%8.1ffps                   \n" (1/(tpf/1.0e9))
            print("\033[s") # save cursor
            print(string("\033[",height+3,"A")) # move up height+3 lines
            last_print = current
        end
        diff = current - prev
        seconds = diff/1.0e9
        alpha = 1 - exp(-0.001*seconds)
        tpf = tpf * alpha + (1 - alpha) * diff
        prev = current
    end
end

function main()
    atexit(cleanup)
    (_, backup_termios) = disable_echo()
    print("\033[?25l") # hide cursor
    Base.exit_on_sigint(false)
    try
        animals()
    catch e
        if isa(e, Core.InterruptException)
            println("\033[uexiting")
        else
            throw(e)
        end
    finally
        TERMIOS.tcsetattr(stdin, TERMIOS.TCSANOW, backup_termios)
        cleanup()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
