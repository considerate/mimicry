module Mimicry

import Plots

greet() = print("Hello World!")

Base.@ccallable function julia_main()::Cint
    try
        main()
    catch
        Base.invokelatest(Base.display_error, Base.catch_stack())
        return 1
    end
    return 0
end

function main()
    greet
    for file in ARGS
        if !isfile(file)
            error("could not find file $file")
        end
        println(file)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
