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
using TimerOutputs
import LinearAlgebra
import ColorSchemes

const tau=2*pi
const Polygon = Vector{StaticArrays.SVector{2, Float64}}
const Point = Tuple{Float64, Float64}
const Polar = Tuple{Float64, Float64}
const Bounds = Tuple{Point, Point}
const Arena = Tuple{Polygon, Polygon, Bounds}

const Prob = Tuple{Vector{Float64}, Vector{Float64}}
const Sampled = Float64
const Parent = Int64

to = TimerOutput()
TimerOutputs.disable_timer!(to)

function cleanup()
    print("\033[?25h") # show cursor
end

struct Network <: Lux.AbstractExplicitLayer
    dense :: Lux.Dense
    means :: Lux.Dense
    logvars :: Lux.Dense
end

struct GaussLoss
    size :: Int64
end
struct KLLoss
    size :: Int64
    l :: Int64
end

function Lux.initialparameters(rng::Random.AbstractRNG, network :: Network)
    layers = NamedTupleTools.ntfromstruct(network)
    return Lux.initialparameters(rng, layers)
end

function Lux.initialstates(rng::Random.AbstractRNG, network :: Network)
    layers = NamedTupleTools.ntfromstruct(network)
    return Lux.initialstates(rng, layers)
end

function initialgradients(layer :: Lux.Dense{use_bias}, ps :: NamedTuple) where {use_bias}
    if use_bias
        return (inputs=zeros(Float64,layer.in_dims), weight=zero(ps.weight), bias=zero(ps.bias))
    else
        return (inputs=zeros(Float64,layer.in_dims), weight=zero(ps.weight))
    end
end

function initialgradients(network :: Network, ps :: NamedTuple)
    means_grads=initialgradients(network.means, ps.means)
    return (dense=initialgradients(network.dense, ps.dense),
            means=means_grads,
            logvars=initialgradients(network.logvars, ps.logvars),
            y=zero(means_grads.inputs)
            )
end
function initialgradients(l :: GaussLoss, _ :: NamedTuple)
    return (means=zeros(Float64, l.size), logvars=zeros(Float64, l.size))
end
function initialgradients(l :: KLLoss, _ :: NamedTuple)
    return (means=zeros(Float64, l.l),
            logvars=zeros(Float64, l.l),
            target_means=zeros(Float64, l.l),
            target_logvars=zeros(Float64, l.l),
            memory=zeros(Float64, l.size)
           )
end

# TODO: There's an optimization to be had with aliasing between memories of different layers. For now, we'll over-allocate a bit for the sake of simplicity.
function initialmemory(layer :: Lux.Dense, _ :: NamedTuple)
    return (x=zeros(Float64,layer.in_dims),y=zeros(Float64,layer.out_dims))
end

function initialmemory(network :: Network, ps :: NamedTuple)
    return (dense=initialmemory(network.dense, ps.dense),
            means=initialmemory(network.means, ps.means),
            logvars=initialmemory(network.logvars, ps.logvars)
            )
end

function initialmemory(l :: GaussLoss, _ :: NamedTuple)
    return (inv_vars=zeros(Float64, l.size), logvars=zeros(Float64, l.size), diffs=zeros(Float64, l.size), losses=zeros(Float64, l.size))
end

function initialmemory(l :: KLLoss, _ :: NamedTuple)
    return (losses=zeros(Float64, l.size),
            diffs=zeros(Float64, l.size),
            mean_diffs_squared=zeros(Float64, l.size),
            var_diffs=zeros(Float64, l.size),
            inv_vars=zeros(Float64, l.size)
           )
end

# A -> (B, dB -> dA)
# (A × X -> B × X, dB × X × G -> dA × G)


@inline function (network::Network)(x :: AbstractArray, ps :: NamedTuple, st::NamedTuple)
    y, st_dense = network.dense(x, ps.dense, st.dense)
    mu, st_mean = network.means(y, ps.means, st.means)
    sigma, st_logvar = network.logvars(y, ps.logvars, st.logvars)
    return (means=mu, logvars=sigma), (dense=st_dense, mean=st_mean, logvar=st_logvar)
end

function logvar_activation(x)
    min_std_deviation = 0.01
    min_logvar = 2*log(min_std_deviation)
    return max.(x, min_logvar)
end

function activation_grad(::typeof(Lux.identity), y, dy)
    return dy
end
function activation_grad(::typeof(Lux.relu), y, dy)
    for i in 1:length(y)
        if y[i] <= 0
            dy[i] = 0
        end
    end
    return dy
end
function activation_grad(::typeof(logvar_activation), y, dy)
    min_std_deviation = 0.01
    min_logvar = 2*log(min_std_deviation)
    for i in 1:length(y)
        if y[i] <= min_logvar
            dy[i] = 0
        end
    end
    return dy
end

@inline function dense_forward(_ :: Lux.Dense, x :: AbstractArray, ps :: NamedTuple, st :: NamedTuple, memory :: NamedTuple)
    #(y, st) = layer(x, ps, st)
    memory.x .= x
    LinearAlgebra.mul!(memory.y, ps.weight, x)
    memory.y .+= ps.bias
    return memory.y, nothing
end

@inline function dense_back(layer :: Lux.Dense, dy :: AbstractArray, ps :: NamedTuple, grads :: NamedTuple, memory :: NamedTuple)
    dy = activation_grad(layer.activation, memory.y, dy)
    grads.weight .= dy .* memory.x'
    # LinearAlgebra.kron!(grads.weight, dy, memory.x)
    grads.bias .= dy
    LinearAlgebra.mul!(grads.inputs, transpose(ps.weight), dy)
end

@inline function network_forward(network :: Network, x :: AbstractArray, ps :: NamedTuple, st :: NamedTuple, memory :: NamedTuple)
    (y, st_dense) = dense_forward(network.dense, x, ps.dense, st.dense, memory.dense)
    (mu, st_mean) = dense_forward(network.means, y, ps.means, st.means, memory.means)
    (sigma, st_logvar) = dense_forward(network.logvars, y, ps.logvars, st.logvars, memory.logvars)
    return (means=mu, logvars=sigma), (dense=st_dense, means=st_mean, logvars=st_logvar)
end

@inline function network_back(network :: Network, dout :: NamedTuple, ps :: NamedTuple, grads :: NamedTuple, memory :: NamedTuple)
    dense_back(network.logvars, dout.logvars, ps.logvars, grads.logvars, memory.logvars)
    dense_back(network.means, dout.means, ps.means, grads.logvars, memory.means)
    grads.y .= grads.logvars.inputs
    grads.y .+= grads.means.inputs
    dense_back(network.dense, grads.y, ps.dense, grads.dense, memory.dense)
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
    age :: Int64
    lineage :: Int64
    feedback_nodes::Vector{Float64} # n_feedback_nodes
    model::@NamedTuple{network::Network, loss::GaussLoss, mimic_loss::KLLoss}
    parameters::NamedTuple
    state::NamedTuple
    body::Body
    optimiser_state :: NamedTuple
    gradients :: NamedTuple
    memory :: NamedTuple
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
    energy::Int64 # negative energy implies death
    age :: Float64
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
    energy::Int64
    age :: Float64
    attacking::Int64
    health :: Int64
    body::Body
    optimiser_state :: NamedTuple
end

mutable struct Food
    energy :: Int64
    health :: Int64
    body :: Body
end


function randomnetwork(rng, s)
    network = Network(Lux.Dense(s.n1, s.n2, Lux.relu),
                      Lux.Dense(s.n2, s.n3),
                      Lux.Dense(s.n2, s.n3, logvar_activation),
                     )
    ps, st = Lux.setup(rng, network)
    return network, ps, st
end


function walk_nested(f, nt :: NamedTuple)
    for x in nt
        walk_nested(f, x)
    end
end
function walk_nested(f, x)
    f(x)
end

function zero_grads(grads :: NamedTuple)
    walk_nested(x -> x .= 0, grads)
end

function Car(rng :: Random.AbstractRNG, learning_rate :: Float64, params :: AgentParams, arena :: Arena)
    (_, sizes) = params
    network, ps, st = randomnetwork(rng, sizes)
    st_opt = Optimisers.setup(Optimisers.Descent(learning_rate), ps)
    feedback_nodes = Random.randn(sizes.n_feedback_nodes) * (1.0/sizes.n_feedback_nodes)
    loss = GaussLoss(sizes.n3)
    mimic_loss = KLLoss(1, sizes.n3)
    nop = NamedTuple()
    grads = (network=initialgradients(network, ps), loss=initialgradients(loss, nop), mimic_loss=initialgradients(mimic_loss,nop))
    memory = (network=initialmemory(network, ps), loss=initialmemory(loss, nop), mimic_loss=initialmemory(mimic_loss,nop))
    model = (network=network,loss=loss,mimic_loss=mimic_loss)
    age = 0
    lineage = 0
    return Car(
        age,
        lineage,
        feedback_nodes,
        model,
        ps,
        st,
        randomBody(arena),
        st_opt,
        grads,
        memory,
    )
end

@inline function gaussloss_forward(_ :: GaussLoss, means :: Vector{Float64}, logvars :: Vector{Float64}, outputs::Vector{Float64}, memory :: NamedTuple)
    memory.diffs .= (means.-outputs)
    memory.inv_vars .= exp.(.-logvars)
    memory.logvars .= logvars
    memory.losses .= memory.diffs.^2
    memory.losses .*= memory.inv_vars
    memory.losses .+= memory.logvars
    return 0.5*Lux.mean(memory.losses)
end

# f(a, b) = a * b + b
# f :: R^n × R^m -> R^k
# a :: R^n
# b :: R^m
# D' f (a, b) (h) = D'(*)(a,b)(h) + D'(snd)(a,b)(h) = (b*h,h*a) + (0, h) = (b*h, h*a + h)
# h :: R^k

@inline function gaussloss_back(_ :: GaussLoss, dloss :: Float64, grads :: NamedTuple, memory :: NamedTuple)
    grads.means .= (2 * dloss) .* memory.diffs
    grads.logvars .= (memory.diffs.^2 .* memory.inv_vars .* memory.logvars .* (-dloss)) .+ dloss
end

function divergence_forward(l :: KLLoss, predicted :: Prob, target :: Prob, memory :: NamedTuple) :: Float64
    # 0.5 (exp(target_logvars - logvars) + ((target_means - means)^2)/exp(logvars) + (target_logvars - logvars) - 1)
    (means,logvars) = predicted
    (target_means, target_logvars) = target
    memory.diffs .= means[1:l.size]
    memory.diffs .-= target_means[1:l.size]
    memory.mean_diffs_squared .= memory.diffs
    memory.mean_diffs_squared .*= memory.mean_diffs_squared
    memory.losses .= memory.mean_diffs_squared
    memory.inv_vars .= .-logvars[1:l.size]
    memory.inv_vars .= exp.(memory.inv_vars)
    memory.losses .*= memory.inv_vars
    memory.var_diffs .= target_logvars[1:l.size]
    memory.var_diffs .-= logvars[1:l.size]
    memory.var_diffs .= exp.(memory.var_diffs[1:l.size])
    memory.losses .+= memory.var_diffs
    memory.losses .-= 1
    return 0.5*sum(memory.losses)
end

# D'(0.5 * (exp(target_logvars - logvars) + ((target_means - means)^2)*exp(-logvars) + (target_logvars - logvars) - 1))(...)(h)
# D'(0.5 * ...)(...)(h)
# D'(*)(k,b)(h) = h*k
# D'(k*_)(x)(h) = D'(_)(x)(k*h)
# D'(+)(a,b)(h) = (h,h)
# D'(exp(target_logvars - logvars) + ((target_means - means)^2)/exp(logvars) + (target_logvars - logvars) - 1)(...)(h) =
#  + D'(exp(target_logvars - logvars))(...)(h)
#  + D'(((target_means - means)^2)/exp(logvars))(..)(h)
#  + D'(target_logvars - logvars)(...)(h)
# D'(target_logvars - logvars)(...)(h) = D'(target_logvars + (-1)*logvars)(...)(h) = (:target_logvars=h, :logvars=-h)
# D'(exp(target_logvars - logvars))(...)(h) = exp(target_logvars - logvars)*D'(target_logvars - logvars)(...)(h)
# D'(f ∘ g)(x)(h) = D'(f)(x)(D'(g)(f(x))(h))
#
# exp(target_logvars - logvars) + ((target_means - means)^2)/exp(logvars)
# y = exp(x)
# D'(exp)(x)(h) = exp(x)*h
# ∂L/∂x = sum_{i} ∂y_i/∂x * ∂L/∂y_i
# ∂L/∂x = exp(x) * ∂L/y
# L(y)
#
# D'(exp(target_logvars - logvars))(h)
#
# D'(((target_means - means)^2)*exp(-logvars))(..)(h) =
#    (target_means - means)^2 * D'(exp(-logvars))(...)(h)
#  + D'((target_means - means)^2)(...)(h) * exp(-logvars)
# D'(exp(-logvars))(...)(h) = exp(-logvars) * D(-logvars)(...)(h) = exp(-logvars) * (:logvars=-h)
#
# D'((target_means - means)^2)(...)(h) * exp(-logvars)
# = (:target_means=2*(target_means - means)*(1)*h, :means=2*(target_means -means)*(-1)*h) * exp(-logvars)

function divergence_back(l :: KLLoss, dloss :: Float64,  gradients :: NamedTuple, memory :: NamedTuple)
    dlosses = gradients.memory
    dlosses .= dloss
    dlosses .*= 0.5
    gradients.target_logvars[1:l.size] .= dlosses
    gradients.logvars[1:l.size] .= .-dlosses
    gradients.target_logvars[1:l.size] .+= memory.var_diffs .* gradients.memory # exp(target_logvars - logvars) * h
    gradients.logvars[1:l.size] .-= memory.var_diffs .* gradients.memory # - exp(target_logvars - logvars) * h
    gradients.logvars[1:l.size] .-= memory.mean_diffs_squared .* memory.inv_vars .* gradients.memory  # - (target_means - means)^2 * exp(-logvars) * h
    gradients.target_means[1:l.size] .= 2.0 .* memory.diffs .* memory.inv_vars .* gradients.memory
    gradients.means[1:l.size] .= -2.0 .* memory.diffs .* memory.inv_vars .* gradients.memory
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


function moveForward(rng :: Random.AbstractRNG, b::Body)
    speed = 0.05;

    return Body(
        b.x + speed*sin(b.theta), # + Random.randn(rng) * 0.001,
        b.y + speed*cos(b.theta), # + Random.randn(rng) * 0.001,
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

@timeit to "draw scene" function draw_scene(arena :: Arena, bodies :: Vector{Body}, ages :: Vector{Int64})
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
    colors=get(ColorSchemes.hawaii,ages,(0.0,500.0))
    for i in eachindex(bodies)
        b = bodies[i]
        c = colors[i]
        col = (round(Int,c.r *255), round(Int,c.g * 255), round(Int,c.b * 255))
        UnicodePlots.points!(canvas, b.x, b.y, color=col)
    end
    plt = UnicodePlots.Plot(canvas)
    # UnicodePlots.scatterplot!(plt, [b.x for b in bodies], [b.y for b in bodies], width=80, height=height, color=colors)
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

@timeit to "train" function train(net::Network, l :: GaussLoss, inputs::Vector{Float64}, ps :: NamedTuple, st :: NamedTuple,  grads::NamedTuple, memory :: NamedTuple)::Tuple{Tuple{Float64, Vector{Float64},Vector{Float64}, Vector{Float64}}, Any, Any}
    (probs, st1) = network_forward(net,inputs,ps,st, memory.network)
    #(probs, st1), dforward = @timeit to "pullback network" Zygote.pullback(p -> Lux.apply(net, inputs, p, st), ps)
    sampled = sample(probs.means, probs.logvars)
    loss = gaussloss_forward(l, probs.means, probs.logvars, sampled, memory.loss)
    # loss, dloss = @timeit to "pullback loss" Zygote.pullback(p -> gaussloss(p.means, p.logvars, sampled), probs)
    gaussloss_back(l, 1.0, grads.loss, memory.loss)
    network_back(net, grads.loss, ps, grads.network, memory.network)
    return ((loss, probs.means, probs.logvars, sampled), grads.network, st1)
end

# @timeit to "train" function train(net::Network, inputs::Vector{Float64}, ps :: NamedTuple, st :: NamedTuple )::Tuple{Tuple{Float64, Vector{Float64},Vector{Float64}, Vector{Float64}}, Any, Any}
#     function model(p)
#         (probs, st1) = Lux.apply(net, inputs, p, st)
#         sampled = Zygote.dropgrad(sample(probs.means, probs.logvars))
#         loss = gaussloss(probs.means, probs.logvars, sampled)
#         return (loss, probs, sampled, st1)
#     end
#
#     results, dmodel = @timeit to "pullback network" Zygote.pullback(model, ps)
#     (loss, probs, sampled, st1) = results
#     grads1 = @timeit to "run network pullback" dmodel((1, nothing, nothing, nothing))[1]
#     return ((loss, probs.means, probs.logvars, sampled), grads1, st1)
# end

function update_feedback(feedback_nodes, feedback)
    mem_decay_times = exp.(range(
        log(10.0),
        stop=log(100.0),
        length=length(feedback_nodes)
    ))
    feedback_nodes.*(1.0 .- 1.0./mem_decay_times) + feedback.*(1.0./mem_decay_times)
end


@timeit to "update car" function updatecar(rng, agent::Car, params :: AgentParams, arena :: Arena)
    # possible future (micro-)optimisation: this currently updates the network
    # even if the agent hit the edge - that could be avoided
    sensors = sensorValues(agent.body, params, arena)
    inputs = [sensors; agent.feedback_nodes]
    ((loss, means, logvars, outputs), grads, _) = train(agent.model.network, agent.model.loss, inputs, agent.parameters, agent.state, agent.gradients, agent.memory)
    @timeit to "optimise" Optimisers.update!(agent.optimiser_state, agent.parameters, grads)
    # @timeit to "zero_grad" zero_grads(agent.gradients)

    output = outputs[1]
    if isnan(output)
        # we take a zero-tolerance approach to NaNs here - if you output one
        # you are immediately teleported outside the arena and die.
        agent.body = Body(-1000.0,-1000.0,0.0)
        output = 0
    end
    feedback = outputs[2:end]
    agent.feedback_nodes .= update_feedback(agent.feedback_nodes, feedback)
    agent.body = moveForward(rng, turn(agent.body,output))
    return (loss, means, logvars, outputs)
end

@timeit to "replicate car" function replicatecar(rng, source :: Car, target :: Car, arena :: Arena)
    if Random.rand(rng) < 0.0
        ps, st = Lux.setup(rng, target.model.network)
        replicateparams(ps, target.parameters)
        replicateparams(st, target.state)
        target.lineage = 0
        target.feedback_nodes .=  zeros(size(target.feedback_nodes))
        target.body = randomBody(arena)
    else
        target.model = source.model
        target.feedback_nodes .= source.feedback_nodes
        replicateparams(source.parameters, target.parameters)
        replicateparams(source.state, target.state)
        replicateparams(source.optimiser_state, target.optimiser_state)
        target.lineage = source.lineage
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
    energy = rand(400:500)
    health = 100
    attacking = 0
    age = 0
    return f(
        animal,
        feedback_nodes,
        network,
        ps,
        st,
        energy,
        age,
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
    energy = rand(60:120)
    health = 40
    (x,y) = randompoint(bounds)
    theta = rand()*tau
    body = Body(x,y,theta)
    return Food(energy, health, body)
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

function replicateparams(source :: NamedTuple, target :: NamedTuple)
    function replicatekey(key)
        if !haskey(target, key)
            return deepcopy(getfield(source, key))
        end
        a = getfield(source, key)
        b = getfield(target, key)
        if isa(a,NamedTuple) && isa(b, NamedTuple)
            return replicateparams(a, b)
        elseif isa(a, Array) && isa(b, Array)
            getfield(target, key) .= getfield(source, key)
        else
            return deepcopy(a)
        end
    end
    return NamedTuple{keys(source)}(map(replicatekey, keys(source)))
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
        replicateparams(source.parameters, target.parameters)
        replicateparams(source.state, target.state)
        replicateparams(source.optimiser_state, target.optimiser_state)
        target.body = source.body
    end

    target.energy = source.energy
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
    agent.feedback_nodes .= update_feedback(agent.feedback_nodes, feedback)
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
    agent.energy -= 2
    if attack >= 0.5
        agent.energy -= 2
        if agent.animal == 1
            for p in prey
                if sqdist(p.body, body) < radiussq && inview(body, p.body, fwd, attackfield)
                    p.health -= 40
                    if p.health <= 0
                        agent.energy += 200
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
                        agent.energy += f.energy
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
    history :: Vector{Vector{Tuple{Prob,Sampled,Body,Parent}}} = []
    while true
        Threads.@threads for k in 1:length(predators)
            predator = predators[k]
            updateanimal(predator, params, bounds, predators, prey, food)
        end
        alive = [p.energy > 0 && p.health > 0 for p in predators]
        Threads.@threads for i in 1:length(predators)
            predator = predators[i]
            if alive[i]
                predator.age += 1
            else
                k = i
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
        alive = [p.energy > 0 && p.health > 0 for p in prey]
        Threads.@threads for i in 1:length(prey)
            if alive[i]
                prey[i].age += 1
            else
                k = 1
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
                replicatepredatorprey(rng, target, prey[i], bounds)
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



@timeit to "train_mimic" function train_mimic(agent, inputs :: Vector{Float64}, targets :: Prob)
    (probs, st1) = network_forward(agent.model.network,inputs, agent.parameters, agent.state, agent.memory.network)
    # agent.state = st1
    loss = divergence_forward(agent.model.mimic_loss, (probs.means, probs.logvars), targets, agent.memory.mimic_loss)
    divergence_back(agent.model.mimic_loss, 1.0, agent.gradients.mimic_loss, agent.memory.mimic_loss)
    network_back(agent.model.network, agent.gradients.mimic_loss, agent.parameters, agent.gradients.network, agent.memory.network)
    return (loss, probs)
end

function mimic(agent::Car, params :: AgentParams, arena :: Arena, trajectory::Array{Tuple{Prob,Body}})
    mid = length(trajectory) ÷ 2
    if mid == 0
        return
    end
    original_feedback = copy(agent.feedback_nodes)
    warmup = trajectory[1:mid]
    training = trajectory[mid+1:end]
    # replay trajectory to warm up feedback_nodes
    for (_, body) in warmup
        sensors = sensorValues(body, params, arena)
        inputs = [sensors; agent.feedback_nodes*1.0]
        prob, _ = network_forward(agent.model.network, inputs, agent.parameters, agent.state, agent.memory.network)
        outputs = sample(prob.means, prob.logvars)
        feedback = outputs[2:end]
        agent.feedback_nodes .= update_feedback(agent.feedback_nodes, feedback)
    end
    for (prob, body) in training
        sensors = sensorValues(body, params, arena)
        inputs = [sensors; agent.feedback_nodes*1.0]
        # train on last step of trajectory
        (_,probs) = train_mimic(agent, inputs, prob)
        outputs = sample(probs.means, probs.logvars)
        feedback = outputs[2:end]
        agent.feedback_nodes .= update_feedback(agent.feedback_nodes, feedback)
    end
    # revert feedback_nodes for the agent
    agent.feedback_nodes .= original_feedback
end

function cars()
    Base.start_reading(stdin)
    started = time_ns()
    arena = createArena()
    params = agentparams(1)
    pop_size = 500
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    agents = [Car(rng, 3e-4, params, arena) for _ in 1:pop_size]
    history :: Vector{Vector{Tuple{Prob,Sampled,Body,Parent}}} = []
    prev = time_ns()
    last_print = 0
    tpf = 0.001
    results = Vector{Tuple{Prob,Sampled}}(undef,(length(agents),))
    parents = [i for i in 1:length(agents)]
    frame = 0
    realtime = false
    target_fps = 30
    while true
        Threads.@threads for k in 1:length(agents)
            agent = agents[k]
            agent.age += 1
            agent.lineage += 1
            (_, means, logvars, sampled) = updatecar(rng, agent, params, arena)
            results[k] = (means, logvars), sampled[1]
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
                agents[i].age = 0
                parents[i] = neighbour
                replicatecar(rng, agents[neighbour], agents[i], arena)
            end
        end
        function agent_step(k)
            prob, sampled = results[k]
            body = agents[k].body
            parent = parents[k]
            return (prob, sampled, body, parent)
        end
        @timeit to "step history" step :: Vector{Tuple{Prob, Sampled, Body, Parent}} = [agent_step(k) for k in 1:length(agents) ]
        pushfirst!(history, step)
        if length(history) > 100
            pop!(history)
        end
        mimic_probability = 0.01
        tasks = []
        for k in 1:length(agents)
            if rand() < mimic_probability
                t = Threads.@spawn begin
                    agent = agents[k]
                    trajectory :: Array{Tuple{Prob, Body}} = []
                    index = argmax([agent.age for agent in agents])
                    #index = rand(1:pop_size)
                    while index == k
                        index = rand(1:pop_size)
                    end
                    for t in 1:length(history)
                        prob, _, body, parent = history[t][index]
                        pushfirst!(trajectory, (prob, body))
                        index = parent
                    end
                    mimic(agent, params, arena, trajectory)
                end
                push!(tasks, t)
            end
        end
        for task in tasks
            Threads.wait(task)
        end

        current = time_ns()
        if current - last_print > 0.05e9
            ages = [agent.age for agent in agents]
            (plt, (_, _)) = draw_scene(arena, [agent.body for agent in agents], ages)
            if to.enabled
                io = PipeBuffer()
                show(IOContext(io), to)
                profiling = string(read(io, String), "\n")
            else
                profiling = ""
            end

            chart = Base.string(plt, color=true)
            mean_age = sum(ages) / length(agents)
            max_age = maximum(ages)
            longest_lineage = maximum([agent.lineage for agent in agents])
            elapsed = current - started
            full_fps =1/(elapsed/(frame*1e9))
            is_realtime = realtime ? "true" : "false"
            summary = @sprintf "%8.1ffps mean: %7.1ffps age: %5.2f max age: %d longest lineage: %d frame: %d realtime %s" (1/(tpf/1.0e9)) full_fps mean_age max_age longest_lineage frame is_realtime
            hist = Base.string(UnicodePlots.histogram(ages, nbins=10, closed=:left, xscale=:log10))
            output = string(chart, "\n", profiling, summary, "\n", hist, "\n")
            lines = countlines(IOBuffer(output))
            print("\033[J") # clear to end of screen
            print(output)
            print("\033[s") # save cursor
            print(string("\033[",lines,"A"))
            last_print = current
        end
        diff = current - prev
        target_step = prev + 1/target_fps * 1e9
        bb = bytesavailable(stdin)
        if bb > 0
            data = read(stdin, bb)
            if data[1] == UInt(32)
                realtime = !realtime
            end
        end
        if realtime && current < target_step
            sleep((target_step - current)/1e9)
        end
        seconds = diff/1.0e9
        alpha = 1 - exp(-0.001*seconds)
        tpf = tpf * alpha + (1 - alpha) * diff
        prev = current
        frame += 1
    end
end

function grads()
    arena = createArena()
    params = agentparams(1)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    car = Car(rng, 1e-4, params, arena)
    sensors = sensorValues(car.body, params, arena)
    inputs = [sensors; car.feedback_nodes*1.0]
    code = Base.code_typed() do
        Zygote._pullback(Zygote.Context(), p -> Lux.apply(car.network, inputs, p, car.state), car.parameters)
    end
    println(code)
    # (probs, st1), dforward = @timeit to "pullback network" Zygote.pullback(p -> Lux.apply(net, inputs, p, st), ps)
    # sampled = @timeit to "sample from distribution" sample(probs.means, probs.logvars)
    # loss, dloss = @timeit to "pullback loss" Zygote.pullback(p -> gaussloss(p.means, p.logvars, sampled), probs)
end

const F_GETFL = Cint(3)
const F_SETFL = Cint(4)
const O_NONBLOCK = Cint(0o00004000)

function main()
    atexit(cleanup)
    s :: RawFD = RawFD(Base.Core.Integer(0))
    flags = ccall(:fcntl, Cint, (RawFD, Cint, Cint...), s, F_GETFL)
    flags2=flags | O_NONBLOCK
    println()
    (_, backup_termios) = disable_echo()
    ccall(:fcntl, Cint, (RawFD, Cint, Cint...), s, F_SETFL, flags2)
    print("\033[?25l") # hide cursor
    Base.exit_on_sigint(false)
    game = ARGS[1]
    if length(ARGS) > 1
        if ARGS[2] == "profile"
            TimerOutputs.enable_timer!(to)
        end
    end
    try
        if game == "cars"
            cars()
        elseif game == "animals"
            animals()
        elseif game == "grads"
            grads()
        else
            cars()
        end
    catch e
        if isa(e, TaskFailedException)
            e = e.task.exception
        end
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
