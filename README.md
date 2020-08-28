# TVB multiscale
This module currently provides a solution to interface between TVB and NEST spiking networks for multiscale co-simulations.

On the long term, it will be extended with the purpose to support integration with other spiking network simulators, by offering a generic way to interface between them and the TVB simulator.
 

# Project structure
At the top-level, we have the following folders:
## docker
Set of files used for building and distributing the module.

## docs
Here is were you can find some documentation about the module. In several forms: pdf or a jupyter notebook with documented steps. 

## examples
Set of scripts and jupyter notebooks that act as demos on how to use the API with different use-cases.

## tests
Unit and integration tests for the module.

## tvb_multiscale
This holds the whole codebase. In the below diagram, you can see a representation of the dependencies between the sub-folders:

                core
                /  \
               /    \
        tvb-nest    tvb-elephant

At the point when the module will support other spiking network simulators, a specific folder for it will be added here, at the same level as tvb_nest and tvb_elephant.

Description of sub-folders:

### core
Contains the base code that is considered generic/abstract enough to interface between a spiking network simulator and TVB (inside spiking_models and interfaces).

Here, we also keep I/O related code (read/write from/to H5 format and plots).

### tvb_nest
Code for interfacing with NEST - depends on core and extends the classes defined there in order to specialize them for NEST (inside nest_models and interfaces).

### tvb_elephant
Code that interfaces with Elephant and implements a wrapper around it that generates a co-simulator compatible stimulus.
