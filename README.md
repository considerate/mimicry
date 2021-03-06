# mimicry

The mimicry system is intended to avoid modelling global goals
explicitly. Instead, each part of the system, referred to as a "cell"
below, has a set of selfish goals in the hope that these local selfish
goals will cause emergent behavior that emulates overarching goals and
the ability to solve complex tasks.

## Evolution and Self-Reinforcement

The most basic selfish goal is optimizing for survival and longevity of
the cell itself. However, instead of modelling this goal explicitly each
cell is trained replicate its own previous behavior. Given a
sufficiently dangerous environment and evolutionary pressures, behavior
that is more likely to lead to survival has higher likelihood to be
retained since only surviving cells may reproduce.

## From Evolution to Social Learning

To increase the speed at which knowledge is shared in the system each
cell is trained not only to replicate their own behavior but also the
behavior other cells it determines to be adept at solving the task it is
presented with. This shifts the system from being purely evolutionary
with knowledge being spread due to sexual reproduction to a memetic one
where ideas can be spread more rapidly with transmission of knowledge
between different lineages of cells through the means of mimicry.

An initial implementation will provide each cell with the longevity of a
collection of neighboring cells. The cell may also inspect the behavior
of the neighboring cells and at each time-step the cell randomly samples
a neighbor weighted by longevity, whose behavior it will train itself to
mimic. If the sampled neighbor died during the time-step the cell
instead trains itself to avoid the behavior exhibited by the neighbor.

The system can be further generalized where the cell isn't provided the
longevity of the neighboring cells by the system but where each cell
instead learns to estimate the fitness of others. This fitness is then
used in place of the system-provided longevity score when selecting
whose behavior to mimic.

## Competitive Environments and Group Dynamics

When placed in a competitive environment with other cells where the cells
compete for the acquirement of resources or where interference competition can
utilised to ensure longevity of its own lineage at the expense of other cells,
the cell may also use ability to inspect the behavior of its neighbors to
better determine its own behavior.

Since each neighbor also has the ability to inspect the behavior of the
cell it is not always optimal to exhibit optimal behavior at each
time-step. It is of interest to study whether the behavior of the cells
will converge to strategies that not only greedily solve the task but to
conservative strategies where the cell is less susceptible to
competition from its neighbors.

A cell may also be constructed from a collection of sub-cells that
collectively determine the behavior of the cell.

## Shared Genetics and General-Purpose Modules

The genetic make-up of the cell, the parameters that determine its behavior,
can be shared between the cells in such a way that training the parameters to
optimize for one cell also implicitly updates the parameters for all other
cells the parameters are shared with. Another form of competition may arise
from this sharing of parameters. By making this module of shared parameters
better the cell also increases the fitness of its competition resulting in a
kind of self-competition where it's not always beneficial to maximize its own
ability.

Moreover, if the shared module is just a subset of the make-up of the cells
then this module has to adapt to solving the multiple goals of the differing
cells. Whether or not this leads to more general-purpose modules or whether the
module becomes useless due to excessive competition of the optimization of the
parameters could be interesting to look into.

Conversely, if the shared module constitutes the entirety of the parameters of
the cell, the cells that share this module may be forced to cooperate if the
downsides of competing outweigh the benefits of following a strategy that leads
to cooperation with the cells with shared parameters.

<!--
-   Emergent goals from selfish behavior
-   Mimimicing self-behavior
-   Dangerous environment
-   Mimimicing others' behavior
-   Estimating longevity (fitness) of self and others
-   Complex goals
-   Interactions between the agents causing death
-   Mimimicing other's behavior if good circumstances, avoiding death
    based on others' experiences
-   Shared genetics
-->
