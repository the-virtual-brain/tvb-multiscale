from random import randint, uniform
import netpyne
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.utils import source

from netpyne import specs, sim
from netpyne.sim import *

class NetpyneInstance(object):

    spikeGenerators = []
    
    def __init__(self, dt):

        self.dt = dt

        self.spikingPopulationLabels = []

        self.netParams = specs.NetParams()

        # using VecStim model from NEURON for artificial cells serving as stimuli
        self.netParams.cellParams['art_NetStim'] = {'cellModel': 'DynamicNetStim'}

        ## Synaptic mechanism parameters
        #TODO: de-hardcode syn params
        self.netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': 0.8, 'tau2': 5.3, 'e': 0}  # NMDA
        self.netParams.synMechParams['inh'] = {'mod': 'Exp2Syn', 'tau1': 0.6, 'tau2': 8.5, 'e': -75}  # GABA

        # Simulation options
        simConfig = specs.SimConfig()

        simConfig.dt = dt
        # simConfig.verbose = True

        simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
        
        simConfig.recordStep = 0.1
        simConfig.savePickle = False        # Save params, network and sim output to pickle file
        simConfig.saveJson = False

        self.simConfig = simConfig

    def registerCellModel(self, cellModel):
        cellParams = netpyne.specs.Dict()

        cellParams.secs.soma.geom = cellModel.geom.toDict()
        #  {'diam': 18.8, 'L': 18.8, 'Ra': 123.0}
        mechName, mech = cellModel.getMech()
        cellParams.secs.soma.mechs[mechName] = mech
        

        self.netParams.cellParams[cellModel.name] = cellParams

    @property
    def minDelay(self):
        return self.dt

    @property
    def time(self):
        return h.t
    
    def createAndPrepareNetwork(self, simulationLength): # TODO: bad name?

        self.simConfig.recordCellsSpikes = self.spikingPopulationLabels # to exclude stimuli-cells
        self.simConfig.duration = simulationLength

        sim.initialize(self.netParams, None) # simConfig to be provided later
        sim.net.createPops()
        sim.net.createCells()

        # choose N random cells from each population to plot traces for
        n = 5
        rnd = np.random.RandomState(0)
        def includeFor(pop):
            popSize = len(sim.net.pops[pop].cellGids)
            chosen = (pop, rnd.choice(popSize, size=min(n, popSize), replace=False).tolist())
            return chosen
        include = list(map(includeFor, self.spikingPopulationLabels))

        self.simConfig.analysis['plotTraces'] = {'include': include, 'saveFig': True}
        self.simConfig.analysis['plotRaster'] = {'include': self.spikingPopulationLabels, 'saveFig': True}

        sim.setSimCfg(self.simConfig)
        sim.setNetParams(self.netParams)

        sim.net.connectCells()
        sim.net.addStims()
        sim.net.addRxD()
        sim.setupRecording()

        sim.run.prepareSimWithIntervalFunc()

    def connectStimuli(self, sourcePop, targetPop, weight, delay, receptorType):

        sourceCells = self.netParams.popParams[sourcePop]['numCells']
        targetCells = self.netParams.popParams[targetPop]['numCells']

        # connect cells roughly one-to-one ('lamda' for E -> I connections is already taken into account, as it baked into source population size)
        if sourceCells <= targetCells:
            rule = 'divergence'
        else:
            rule = 'convergence'

        connLabel = sourcePop + '->' + targetPop
        self.netParams.connParams[connLabel] = {
            'preConds': {'pop': sourcePop},
            'postConds': {'pop': targetPop},
            rule: 1.0,
            'weight': weight,
            'delay': delay,
            'synMech': receptorType
        }

    def interconnectSpikingPopulations(self, sourcePopulation, targetPopulation, synapticMechanism, weight, delay, probabilityOfConn):

        label = sourcePopulation + "->" + targetPopulation
        self.netParams.connParams[label] = {
            'preConds': {'pop': sourcePopulation},
            'postConds': {'pop': targetPopulation},
            'probability': probabilityOfConn,
            'weight': weight,
            'delay': delay,
            'synMech': synapticMechanism }

    def registerPopulation(self, label, cellModel, size):
        self.spikingPopulationLabels.append(label)
        self.netParams.popParams[label] = {'cellType': cellModel, 'numCells': size}

    def createArtificialCells(self, label, number, params=None):
        print(f"Netpyne:: Creating artif cells for node '{label}' of {number} neurons")
        self.netParams.popParams[label] = {
            'cellType': 'art_NetStim',
            'numCells': number,
        }

    def getSpikes(self, generatedBy=None, startingFrom=None):
        spktimes = np.array(sim.simData['spkt'])
        spkgids = np.array(sim.simData['spkid'])

        if startingFrom is not None:
            inds = np.nonzero(spktimes > startingFrom) # filtered by time # (self.time - timeWind)

            spktimes = spktimes[inds]
            spkgids = spkgids[inds]

        if generatedBy is not None:
            inds = np.isin(spkgids, generatedBy)

            spktimes = spktimes[inds]
            spkgids = spkgids[inds]
        return spktimes, spkgids
    
    def cellGidsForPop(self, popLabel):
        return sim.net.pops[popLabel].cellGids

    def neuronsConnectedWith(self, targetPop):
        gids = []
        for connection in self.netParams.connParams.keys():
            if connection.find(targetPop) >= 0:
                pop = self.netParams.connParams[connection]['postConds']['pop']
                gids.append(self.cellGidsForPop(pop))
        gids = np.array(gids).flatten()
        return gids

    def run(self, length):

        allNeuronsSpikes = {}
        allNeurons = []
        for device in self.spikeGenerators:
            allNeurons.extend(device.own_neurons)
            allNeuronsSpikes.update(device.spikesPerNeuron)

            device.spikesPerNeuron = {} # clear to prepare for next interval run

        intervalEnd = h.t + length
        for gid in allNeurons:
            # currently used .mod implementation allows no more then 3 spikes during the interval.
            # if for given cell they are less than 3, use -1 for the rest. If they are more, the rest will be lost. But for the reasonable spiking rates, this latter case is highly unlikely. 
            spikes = allNeuronsSpikes.get(gid, [])
            spks = [-1] * 3
            for i, spike in enumerate(spikes[:3]):
                spks[i] = spike
            sim.net.cells[gid].hPointp.set_next_spikes(intervalEnd, spks[0], spks[1], spks[2])

        def func(simTime):
            pass
        if self.time + length <= sim.cfg.duration:
            sim.run.runForInterval(length, func)

    def finalize(self):
        sim.run.postRun()
        sim.gatherData()
        sim.analyze()

from neuron import h
class NetpyneProxyDevice(object):

    def __len__(self):
        return 1 # TODO: at least add some explanatory comment

    netpyne_instance = None

    def __init__(self, netpyne_instance):
        self.netpyne_instance = netpyne_instance
