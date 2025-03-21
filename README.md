# Fitness Visualization

This repository contains some experimental code
that can be used to visualize the fitness landscape
of a population in the context of quantum
circuit synthesis. Although this is the setup most
relevant to our work at the moment, the overall
approach can easily be adapted to any population of
discretely encoded values with one continuous score
assigned to each element in the population.

In our case, the population consists of quantum circuits
that are represented as lists of quantum gates, where
each circuit has a fitness value attached to it.
We want to investigate how
changes to the chromosomes (e.g. inserting or changing
a gate) affect the fitness values of the population.
If small changes to the chromosome cause small changes
to the fitness, the bigger the utility of a local search
strategy.

Our approach consists of the following steps:

1. Generate a population of quantum circuits according
   to a specific schema. It is up to the user to decide
   how the circuits are created and which gates they may
   consist of.
2. Evaluate each circuit in the population according
   to a specified fitness function.
3. Map the genotype representations of the population
   to points in a 2d euclidian space using umap and
   editdistance.
4. Visualize the fitness landscape using matplotlib.
   The x and y coordinates therein correspond to the umap
   embeddings, while the z coordinate corresponds to the
   fitness score of a circuit.

Ideally, we would like our fitness function to share its
global optimum with a behavior-oriented fitness function
like the Jensen-Shannon Distance, but have few local optima.
The corresponding landscape would allow for efficient
search using local hill-climbing.

## Important Notice

In version 0.3.0, the ga4qc specification requires
specific versions for dependencies like numpy. In
future releases, these requirements will likely be
softened. If you run into dependency conflicts when
installing the requirements, consider installing all
other requirements first. Then install ga4qc via

```bash
pip install --no-deps ga4qc==0.3.0
```
