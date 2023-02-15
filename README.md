# TVB multiscale
This package currently provides a solution to interface between TVB and NEST, ANNarchy or NETPYNE (NEURON) spiking networks for multiscale co-simulations.

It tries to offer a generic way to interface between them and the TVB simulator.
 

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
This holds the main codebase.

Description of sub-folders:

### core
Contains the base code that is considered generic/abstract enough to interface between a spiking network simulator and TVB (inside spiking_models and interfaces).

Here, we also keep I/O related code (read/write from/to H5 format and plots) and some data_analysis related classes.

### tvb_nest
Code for interfacing with NEST - depends on core and extends the classes defined there in order to specialize them for NEST (inside nest_models and interfaces).

### tvb_annarchy
Code for interfacing with ANNarchy - depends on core and extends the classes defined there in order to specialize them for ANNarchy (inside annarchy_models and interfaces).

### tvb_netpyne
Code for interfacing with NETPYNE - depends on core and extends the classes defined there in order to specialize them for NETPYNE (inside netpyne_models and interfaces).

### tvb_elephant
Code that interfaces with Elephant for spike train generation and analysis functionality (used also for transformations of interfaces between TVB and spiking simulators), as well as for generating spiking train stimuli for TVB.

### tvb_pyspike
Core that interfaces with pyspike package for spike train synchronization analysis.

## Acknowledgments
This  research  has  received  funding  from  the  European  Union’s  Horizon  2020  Framework  Programme  for  Research  and  Innovation  under  the  Specific  Grant  Agreement  Nos.  785907  (Human  Brain Project SGA2),  945539  (Human  Brain Project SGA3), ICEI 800858, VirtualBrainCloud 826421 and ERC 683049.
