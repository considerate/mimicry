module Mimicry

import UnicodePlots
import Zygote
import Flux
import PolygonOps
import StaticArrays
import TERMIOS
# import Plots

include("Arena.jl")
include("Agents.jl")
import .Agents as Agents
import .Arena as Arena


function forward(x :: Vector{Float32}, w :: Matrix{Float32})
    return w * x
end

function cleanup()
    print("\033[?25h") # show cursor
end


function main()
    # xs = [-1, 5,  0, 0, 5, -1, 4,  4,  2,    2,    2, 2, -1]
    # ys = [ 0, 0, -1, 3, 2,  2, 3, -1, -1, 0.75, 1.25, 3,  0]
    # TODO: compute this
    points = [
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
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    inner_points = [
      (0.2, 1.8),
      (1.8, 1.8),
      (1.8, 1.2),
      (2.0, 1.0),
      (2.2, 1.2),
      (2.2, 1.8),
      (3.8, 1.8),
      (3.8, 0.2),
      (2.2, 0.2),
      (2.2, 0.8),
      (2.0, 1.0),
      (1.8, 0.8),
      (1.8, 0.2),
      (0.2, 0.2),
      (0.2, 1.8),
    ]
    polygon = StaticArrays.SVector.(xs,ys)
    inner = StaticArrays.SVector.([p[1] for p in inner_points],[p[2] for p in inner_points],)
    atexit(cleanup)
    backup_termios = TERMIOS.termios()
    TERMIOS.tcgetattr(stdin, backup_termios)
    termios = TERMIOS.termios()
    TERMIOS.tcgetattr(stdin, termios)
     # Disable echo
    termios.c_lflag &= ~TERMIOS.ECHO
    TERMIOS.tcsetattr(stdin, TERMIOS.TCSANOW, termios)
    pop_size = 500
    bodies = [Agents.Body(0,0,0) for _ in 1:pop_size]
    for body in bodies
        Agents.randomBody!(body)
    end
    print("\033[?25l") # hide cursor
    tpf = 0.001
    Base.exit_on_sigint(false)
    running = true
    try
        prev = time_ns()
        while running
            alive = 0
            # plt_arena = Arena.plot_arena()
            # Plots.display(plt_arena)
            for body in bodies
                #Agents.plot_body!(body)
                Agents.moveForward!(body)
            end
            positions = [(b.x, b.y) for b in bodies]
            inside = [PolygonOps.inpolygon(p, polygon) for p in positions]
            inside_inner = [PolygonOps.inpolygon(p, inner) for p in positions]
            for (body, alive, crashed) in zip(bodies, inside, inside_inner)
                if alive != 1 || crashed != 0
                    while PolygonOps.inpolygon((body.x,body.y), polygon) != 1 || PolygonOps.inpolygon((body.x, body.y), inner) != 0
                        body.x = rand()*6 - 1
                        body.y = rand()*4
                        body.theta = rand()*2*pi
                    end
                end
            end
            width = 121
            height = 44
            canvas = UnicodePlots.BrailleCanvas(height, width, origin_y=-1.2, origin_x=-1.2, height=4.4, width=6.4)
            for (before, after) in zip(polygon[1:end-1], polygon[2:end])
                UnicodePlots.lines!(canvas, before[1], before[2], after[1], after[2]; color=:cyan)
            end
            for (before, after) in zip(inner[1:end-1], inner[2:end])
                UnicodePlots.lines!(canvas, before[1], before[2], after[1], after[2]; color=:cyan)
            end
            plt = UnicodePlots.Plot(canvas)
            UnicodePlots.scatterplot!(plt, [b.x for b in bodies], [b.y for b in bodies], width=80, height=height)
            println(UnicodePlots.show(plt))
            current = time_ns()
            alpha = 0.9
            tpf = tpf * alpha + (1 - alpha) * (current - prev)
            println(1/(tpf/1.0e9))
            print(string("\033[s\033[",height+3,"A")) # move up height+3 lines
            prev = current
            # sleep(0.1)
            #Plots.display(plt)
            # x :: Vector{Float32} = [0.5, 0.3, 0.2]
            # _, back = Zygote.pullback(forward, x, w)
            # _ = back(1.0)
            # println(y)
            # println((dx,dw))
        end
    catch e
        if isa(e, Core.InterruptException)
            running = false
            TERMIOS.tcsetattr(stdin, TERMIOS.TCSANOW, backup_termios)
            sleep(0.1)
            println("\033[uexiting")
        else
            throw(e)
        end
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
