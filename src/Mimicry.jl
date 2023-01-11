module Mimicry

import UnicodePlots
import Zygote
import PolygonOps
import StaticArrays
import TERMIOS
import Lux
import Optimisers
using Printf
import NamedTupleTools
import Random

const tau=2*pi
const Polygon = Vector{StaticArrays.SVector{2, Float64}}
const Point = Tuple{Float64, Float64}
const Polar = Tuple{Float64, Float64}
const Bounds = Tuple{Point, Point}
const Arena = Tuple{Polygon, Polygon, Bounds}

function cleanup()
    print("\033[?25h") # show cursor
end

struct Network <: Lux.AbstractExplicitLayer
    dense :: Lux.Dense
    means :: Lux.Dense
    logvars :: Lux.Dense
end

function Lux.initialparameters(rng::Random.AbstractRNG, network :: Network)
    layers = NamedTupleTools.ntfromstruct(network)
    return Lux.initialparameters(rng, layers)
end

function Lux.initialstates(rng::Random.AbstractRNG, network :: Network)
    layers = NamedTupleTools.ntfromstruct(network)
    return Lux.initialstates(rng, layers)
end

@inline function (network::Network)(x :: AbstractArray, ps :: NamedTuple, st::NamedTuple)
    y, st_dense = network.dense(x, ps.dense, st.dense)
    mu, st_mean = network.means(y, ps.means, st.means)
    sigma, st_logvar = network.logvars(y, ps.logvars, st.logvars)
    return (means=mu, logvars=sigma), (dense=st_dense, mean=st_mean, logvar=st_logvar)
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
    parameters::NamedTuple
    state::NamedTuple
    body::Body
    optimiser_state :: NamedTuple
end

# A Predator decides:
# - x movement speed using a gaussian (tanh)
# - y movement speed using a gaussian (tanh)
# - whether to attack or not (sigmoid)
mutable struct Predator
    animal :: Int64 # TODO: replace with a proper type
    feedback_nodes::Vector{Float64}
    network::Network
    parameters::NamedTuple
    state::NamedTuple
    hunger::Int64 # negative hunger implies death
    attacking::Int64
    health :: Int64
    body::Body
    optimiser_state :: NamedTuple
end

# A Predator decides:
# - x movement speed using a gaussian (tanh)
# - y movement speed using a gaussian (tanh)
# - whether to attack/eat food or not (sigmoid)
mutable struct Prey
    animal :: Int64
    feedback_nodes::Vector{Float64}
    network::Network
    parameters::NamedTuple
    state::NamedTuple
    hunger::Int64
    attacking::Int64
    health :: Int64
    body::Body
    optimiser_state :: NamedTuple
end

mutable struct Food
    hunger :: Int64
    health :: Int64
    body :: Body
end

function randomnetwork(rng, s)
    min_std_deviation = 0.01
    min_logvar = 2*log(min_std_deviation)
    network = Network(Lux.Dense(s.n1, s.n2, Lux.relu),
                      Lux.Dense(s.n2, s.n3),
                      Lux.Dense(s.n2, s.n3, x -> max.(x, min_logvar)),
                     )
    ps, st = Lux.setup(rng, network)
    return network, ps, st
end


function Car(rng :: Random.AbstractRNG, learning_rate :: Float64, params :: AgentParams, arena :: Arena)
    (_, sizes) = params
    network, ps, st = randomnetwork(rng, sizes)
    st_opt = Optimisers.setup(Optimisers.ADAM(learning_rate), ps)
    feedback_nodes = Random.randn(sizes.n_feedback_nodes) * (1.0/sizes.n_feedback_nodes)
    return Car(
        feedback_nodes,
        network,
        ps,
        st,
        randomBody(arena),
        st_opt,
    )
end

function gaussloss(means::Vector{Float64},logvars::Vector{Float64},outputs::Vector{Float64}) ::Float64
    squared_deviations = (outputs-means).^2
    gaussian_loss = 0.5*sum(squared_deviations .* exp.(-logvars) + logvars)/length(outputs)
    return gaussian_loss #+ 0.01*l2(net)
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

function train(net::Network, inputs::Vector{Float64}, ps :: NamedTuple, st :: NamedTuple )::Tuple{Tuple{Float64, Vector{Float64},Vector{Float64}, Vector{Float64}}, Any, Any}
    (probs, st1), dforward = Zygote.pullback(p -> Lux.apply(net, inputs, p, st), ps)
    sampled = sample(probs.means, probs.logvars)
    loss, dloss = Zygote.pullback(p -> gaussloss(p.means, p.logvars, sampled), probs)
    dprob = dloss(1.0)[1]
    grads1 = dforward((dprob, nothing))[1]
    return ((loss, probs.means, probs.logvars, sampled), grads1, st1)
end


function updatecar(agent::Car, params :: AgentParams, arena :: Arena)
    # possible future (micro-)optimisation: this currently updates the network
    # even if the agent hit the edge - that could be avoided
    sensors = sensorValues(agent.body, params, arena)
    inputs = [sensors; agent.feedback_nodes*1.0]
    ((loss, _, _, outputs), grads, _) = train(agent.network, inputs, agent.parameters, agent.state)
    Optimisers.update(agent.optimiser_state, agent.parameters, grads)

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

function replicatecar(rng, source :: Car, target :: Car, arena :: Arena)
    if Random.rand(rng) < 0.01
        ps, st = Lux.setup(rng, target.network)
        target.parameters = ps
        target.state = st
        target.feedback_nodes[:] .=  zeros(size(target.feedback_nodes))
        target.body = randomBody(arena)
    else
        target.network = source.network
        target.feedback_nodes[:] .= source.feedback_nodes
        target.parameters = deepcopy(source.parameters)
        target.state = deepcopy(source.state)
        target.optimiser_state = deepcopy(source.optimiser_state)
        target.body = source.body
    end

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

function agentparams(n_outputs) :: AgentParams
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
    n3 = n_outputs + n_feedback_nodes;
    sizes = Sizes(n1, n2, n3, n_sensors, n_feedback_nodes)
    return (sensorParams, sizes)
end

function PredatorPrey(rng, learning_rate, params, bounds, animal, f)
    (_, sizes) = params
    network, ps, st = randomnetwork(rng, sizes)
    st_opt = Optimisers.setup(Optimisers.ADAM(learning_rate), ps)
    feedback_nodes = Random.randn(sizes.n_feedback_nodes) * (1.0/sizes.n_feedback_nodes)
    (x,y) = randompoint(bounds)
    theta = rand()*tau
    body = Body(x,y,theta)
    hunger = rand(80:100)
    health = 100
    attacking = 0
    return f(
        animal,
        feedback_nodes,
        network,
        ps,
        st,
        hunger,
        attacking,
        health,
        body,
        st_opt,
    )
end

function Predator(rng, learning_rate, params, bounds) :: Predator
    return PredatorPrey(rng, learning_rate, params, bounds, 1, Predator)
end

function Prey(rng, learning_rate, params, bounds) :: Prey
    return PredatorPrey(rng, learning_rate, params, bounds, 2, Prey)
end

function Food(bounds) :: Food
    hunger = rand(60:120)
    health = 40
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

function replicatepredatorprey(rng, source, target, bounds :: Bounds)
    if Random.rand(rng) < 0.01
        ps, st = Lux.setup(rng, target.network)
        target.parameters = ps
        target.state = st
        target.feedback_nodes[:] .=  zeros(size(target.feedback_nodes))
        (x,y) = randompoint(bounds)
        theta = rand()*tau
        target.body = Body(x,y,theta)
    else
        target.network = source.network
        target.feedback_nodes[:] .= source.feedback_nodes
        target.parameters = deepcopy(source.parameters)
        target.state = deepcopy(source.state)
        target.optimiser_state = deepcopy(source.optimiser_state)
        target.body = source.body
    end

    target.hunger = Random.rand(rng, 80:100)
    target.health = 100
    target.attacking = 0
end

function sigmoid(x)
    return 1 / (1 + exp(-x))
end

function sqdist(a,b)
    return (a.x - b.x)^2 + (a.y - b.y)^2
end

function inview(a, b, c, field)
    ac_x = c.x - a.x
    ac_y = c.y - a.y
    ab_x = b.x - a.x
    ab_y = b.y - a.y
    ac_l = sqrt(sqdist(a,c))
    ab_l = sqrt(sqdist(a,b))
    return (ac_x*ab_x + ac_y*ab_y)/(ac_l*ab_l) >= field
end

function outside(p, bounds)
    ((xmin, xmax), (ymin, ymax)) = bounds
    if p[1] < xmin || p[1] > xmax
        return true
    end
    if p[2] < ymin || p[2] > ymax
        return true
    end
    return false
end

function animalsensors(b::Body, params :: AgentParams, bounds :: Bounds, predators :: Vector{Predator}, prey :: Vector{Prey}, food :: Vector{Food}) :: Vector{Float64}
    (sensorParams, _) = params
    points = sensorPoints(b, sensorParams)
    function dist(agent,p)
        dx = agent.body.x - p[1]
        dy = agent.body.y - p[2]
        return sqrt(dx^2 + dy^2)
    end
    radius = 0.05
    function classify(p)
        if outside(p, bounds)
            return 4
        end
        if minimum(map(x -> dist(x,p), predators)) < radius
            return 1
        end
        if minimum(map(x -> dist(x,p), prey)) < radius
            return 2
        end
        if minimum(map(x -> dist(x,p), food)) < radius
            return 3
        end
        return 0
    end
    return [classify(p) for p in points]
end

function clampbounds(body, bounds)
    ((xmin, xmax), (ymin, ymax)) = bounds
    x = max(xmin, min(xmax, body.x))
    y = max(ymin, min(ymax, body.y))
    return Body(x,y,body.theta)
end

function updateanimal(agent, params :: AgentParams, bounds :: Bounds, predators, prey, food)
    sensors = animalsensors(agent.body, params, bounds, predators, prey, food)
    inputs = [sensors; agent.feedback_nodes*1.0]
    (loss, _, _, outputs), grads, _ = train(agent.network, inputs, agent.parameters, agent.state)
    Optimisers.update(agent.optimiser_state, agent.parameters, grads)

    vfwd = tanh(outputs[1])
    vside = tanh(outputs[2])
    speed = 0.1
    attack = sigmoid(outputs[3])
    feedback = outputs[4:end]
    (_, sizes) = params
    mem_decay_times = exp.(range(
        log(10.0),
        stop=log(100.0),
        length=sizes.n_feedback_nodes
    ))
    agent.feedback_nodes = (
        agent.feedback_nodes.*(1.0 .- 1.0./mem_decay_times)
        + feedback.*(1.0./mem_decay_times)
    )
    body = agent.body
    attackradius = 0.5
    attackfield = cos(0.1*tau)
    radiussq = attackradius * attackradius
    x = body.x + cos(body.theta) * vfwd * speed
    y = body.y + sin(body.theta) * vside * speed
    theta = atan(vside, vfwd)
    agent.body = clampbounds(Body(x, y, theta), bounds)
    fwd = Body(body.x + cos(body.theta) * attackradius,
               body.y + sin(body.theta) * attackradius,
               theta,
              )
    agent.hunger -= 2
    if attack >= 0.5
        agent.hunger -= 2
        if agent.animal == 1
            for p in prey
                if sqdist(p.body, body) < radiussq && inview(body, p.body, fwd, attackfield)
                    p.health -= 40
                    if p.health <= 0
                        agent.hunger += 200
                    end
                end
            end
            for f in food
                if sqdist(f.body, body) < radiussq && inview(body, f.body, fwd, attackfield)
                    f.health -= 20
                end
            end
        elseif agent.animal == 2
            for f in food
                if sqdist(f.body, body) < radiussq && inview(body, f.body, fwd, attackfield)
                    f.health -= 40
                    if f.health <= 0
                        agent.hunger += f.hunger
                    end
                end
            end
            for p in predators
                if sqdist(p.body, body) < radiussq && inview(body, p.body, fwd, attackfield)
                    p.health -= 10
                end
            end
        end
    end
    return (outputs, loss)
end

function animals()
    bounds :: Bounds = ((0, 10), (0, 10))
    params = agentparams(3)
    (n_predators, n_prey, n_food) = (50, 100, 50)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    predators = [Predator(rng, 1e-4, params, bounds) for _ in 1:n_predators]
    prey = [Prey(rng, 1e-4, params, bounds) for _ in 1:n_prey]
    food = [Food(bounds) for _ in 1:n_food]
    last_print = 0
    while true
        Threads.@threads for k in 1:length(predators)
            predator = predators[k]
            updateanimal(predator, params, bounds, predators, prey, food)
        end
        alive = [p.hunger > 0 && p.health > 0 for p in predators]
        Threads.@threads for k in 1:length(predators)
            predator = predators[k]
            if !alive[k]
                neighbour = mod1(k+rand([-1,1]),n_predators)
                retries = 0
                target = predators[neighbour]
                while !alive[neighbour]
                    k = neighbour
                    neighbour = mod1(k+rand([-1,1]),n_predators)
                    target = predators[neighbour]
                    retries += 1
                    if retries > 20
                        target = Predator(rng, 1e-4, params, bounds)
                        break
                    end
                end
                @assert alive[neighbour] || retries > 20
                replicatepredatorprey(rng, target, predator, bounds)
            end
        end
        Threads.@threads for k in 1:length(prey)
            p = prey[k]
            updateanimal(p, params, bounds, predators, prey, food)
        end
        alive = [p.hunger > 0 && p.health > 0 for p in prey]
        Threads.@threads for k in 1:length(prey)
            if !alive[k]
                neighbour = mod1(k+Random.rand(rng, [-1,1]), n_prey)
                target = prey[neighbour]
                retries = 0
                while !alive[neighbour]
                    k = neighbour
                    neighbour = mod1(k+rand([-1,1]), n_prey)
                    target = prey[neighbour]
                    retries += 1
                    if retries > 20
                        target = Prey(rng, 1e-4, params, bounds)
                        break
                    end
                end
                @assert alive[neighbour] || retries > 20
                replicatepredatorprey(rng, target, prey[k], bounds)
            end
        end
        for (k, f) in enumerate(food)
            if f.health <= 0
                food[k] = Food(bounds)
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
    params = agentparams(1)
    pop_size = 500
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    agents = [Car(rng, 1e-4, params, arena) for _ in 1:pop_size]
    prev = time_ns()
    last_print = 0
    tpf = 0.001
    while true
        Threads.@threads for k in 1:length(agents)
            agent = agents[k]
            updatecar(agent, params, arena)
        end

        alive = [ontrack((agent.body.x, agent.body.y), arena) for agent in agents]

        Threads.@threads for i in 1:length(agents)
            if !alive[i]
                k = i
                neighbour = mod1(k+rand([-1,1]), length(agents))
                while !alive[neighbour]
                    k = neighbour
                    neighbour = mod1(k+rand([-1,1]), length(agents))
                end
                @assert alive[neighbour]
                replicatecar(rng, agents[neighbour], agents[i], arena)
            end
        end

        current = time_ns()
        if current - last_print > 0.02e9
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
    game = ARGS[1]
    try
        if game == "cars"
            cars()
        elseif game == "animals"
            animals()
        else
            cars()
        end
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
