module Arena
using LibGEOS, GeoInterface
using Plots

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

# TODO: Avoid creating singletons on module import
const arenaShape = dead_ends_track()
const arena = prepareGeom(arenaShape)

const xmin = LibGEOS.getXMin(arenaShape)
const ymin = LibGEOS.getYMin(arenaShape)
const xmax = LibGEOS.getXMax(arenaShape)
const ymax = LibGEOS.getYMax(arenaShape)

function randomPoint() :: Tuple{Float64, Float64}
    x = rand()*(xmax-xmin) + xmin
    y = rand()*(ymax-ymin) + ymin
    return (x,y)
end

function onTrack(x :: Float64, y :: Float64):: Bool
    return contains(arena, LibGEOS.Point(x,y))
end

# println(LibGEOS.GeoInterface.coordinates(arenaShape))
#[[[-1.0, 1.8], [-1.2, 2.0], [-1.0, 2.2], [-0.2, 2.2], [-0.2, 3.0],
#  [1.2246467991473533e-17, 3.2], [0.2, 3.0], [0.2, 2.2], [1.8, 2.2], [1.8, 3.0],
#  [2.0, 3.2], [2.2, 3.0], [2.2, 2.2], [3.8, 2.2], [3.8, 3.0], [4.0, 3.2], [4.2, 3.0],
#  [4.2, 2.2], [5.0, 2.2], [5.2, 2.0], [5.0, 1.8], [4.2, 1.8], [4.2, 0.20000000000000004],
#  [5.0, 0.20000000000000004], [5.2, 0.0], [5.0, -0.20000000000000004], [4.2, -0.20000000000000004],
#  [4.2, -1.0], [4.0, -1.2], [3.8, -1.0], [3.8, -0.20000000000000004], [2.2, -0.20000000000000004],
#  [2.2, -1.0], [2.0, -1.2], [1.8, -1.0], [1.8, -0.20000000000000004], [0.2, -0.20000000000000004],
#  [0.2, -1.0], [1.2246467991473533e-17, -1.2], [-0.2, -1.0], [-0.2, -0.20000000000000004],
#  [-1.0, -0.20000000000000004], [-1.2, 2.4492935982947065e-17], [-1.0, 0.20000000000000004],
#  [-0.2, 0.20000000000000004], [-0.2, 1.8], [-1.0, 1.8]],
# [[0.2, 0.20000000000000004], [1.8, 0.20000000000000004],
#  [1.8, 0.75], [2.0, 0.95], [2.2, 0.75], [2.2, 0.20000000000000004], [3.8, 0.20000000000000004],
#  [3.8, 1.8], [2.2, 1.8], [2.2, 1.25], [2.0, 1.05], [1.8, 1.25], [1.8, 1.8], [0.2, 1.8], [0.2, 0.20000000000000004]
# ]
# ]


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

end
