import numpy as np

from netpyne import specs, sim
from netpyne.sim import *
from netpyne import __version__ as __netpyne_version__


class NetpyneModule(object):

    spikeGenerators = []

    __netpyne_version__ = __netpyne_version__

    def __init__(self):
        self.spikeGeneratorPops = []
        self.autoCreatedPops = []
        self._compileOrLoadMod()

    def _compileOrLoadMod(self):
        # Make sure that all required mod-files are compiled (is there a better way to check?)
        try:
            h.DynamicVecStim()
        except:
            import sys, os
            currDir = os.getcwd()

            python_path = sys.executable.split("python")[0]
            tvb_multiscale_path = os.path.abspath(__file__).split("tvb_multiscale")[0]
            # before compiling, need to cd to where those specific mod files live, to avoid erasing any other dll's that might contain other previously compiled model
            os.chdir(f'{tvb_multiscale_path}/tvb_multiscale/tvb_netpyne/netpyne/mod')
            if not os.path.exists('x86_64'):
                print("NetPyNE couldn't find necessary mod-files. Trying to compile..")
                os.system(f'{python_path}nrnivmodl .')
            else:
                print(f"NetPyNE will load mod-files from {os.getcwd()}.")
            import neuron
            neuron.load_mechanisms('.')

            os.chdir(currDir)

    def importModel(self, netParams, simConfig, dt, config):

        simConfig.dt = dt
        simConfig.duration = config.simulation_length

        simConfig.simLabel = 'spiking'
        simConfig.saveFolder = config.out._out_base # TODO: better use some public method

        self.netParams = netParams
        self.simConfig = simConfig

        # using DynamicVecStim model for artificial cells serving as stimuli
        self.netParams.cellParams['art_NetStim'] = {'cellModel': 'DynamicVecStim'}

    @property
    def dt(self):
        return self.simConfig.dt

    @property
    def minDelay(self):
        return self.dt

    @property
    def time(self):
        return h.t
    
    def instantiateNetwork(self):

        sim.initialize(self.netParams, None) # simConfig to be provided later
        sim.net.createPops()
        sim.net.createCells()

        if len(self.autoCreatedPops):
            # choose N random cells from each population to plot traces for
            n = 1
            rnd = np.random.RandomState(0)
            def includeFor(pop):
                popSize = len(sim.net.pops[pop].cellGids)
                chosen = (pop, rnd.choice(popSize, size=min(n, popSize), replace=False).tolist())
                return chosen
            include = list(map(includeFor, self.autoCreatedPops))
            self.simConfig.analysis['plotTraces'] = {'include': include, 'saveFig': True}
            self.simConfig.analysis['plotRaster'] = {'saveFig': True, 'include': self.autoCreatedPops, 'popRates': 'minimal'}

            if self.simConfig.recordCellsSpikes == -1:
                allPopsButSpikeGenerators = [pop for pop in self.netParams.popParams.keys() if pop not in self.spikeGeneratorPops]
                self.simConfig.recordCellsSpikes = allPopsButSpikeGenerators

        sim.setSimCfg(self.simConfig)
        sim.setNetParams(self.netParams)
        sim.net.params.synMechParams.preprocessStringFunctions()
        sim.net.params.cellParams.preprocessStringFunctions()

        sim.net.connectCells()
        sim.net.addStims()
        sim.net.addRxD()
        sim.setupRecording()

        sim.run.prepareSimWithIntervalFunc()

        # interval run is used internally to support communication with TVB. However, users may also need to have feedbacks
        # at some interval, distinct from those used internally. User-defined interval and intervalFunc are read here, and the logic is handled in `run()`
        if hasattr(self.simConfig, 'interval'):
            self.interval = self.simConfig.interval
            self.intervalFunc = self.simConfig.intervalFunc
            self.nextIntervalFuncCall = self.interval
        else:
            self.nextIntervalFuncCall = None


    def connectStimuli(self, sourcePop, targetPop, weight, delay, receptorType, prob=None):
        # TODO: randomize weight and delay, if values do not already contain sting func
        # (e.g. use random_normal_weight() and random_uniform_delay() from netpyne_templates)
        sourceCells = self.netParams.popParams[sourcePop]['numCells']
        targetCells = self.netParams.popParams[targetPop]['numCells']

        if prob:
            rule = 'probability'
            val = prob

        # connect cells roughly one-to-one ('lamda' for E -> I connections is already taken into account, as it baked into source population size)
        elif sourceCells <= targetCells:
            rule = 'divergence'
            val = 1.0
        else:
            rule = 'convergence'
            val = 1.0

        connLabel = sourcePop + '->' + targetPop
        self.netParams.connParams[connLabel] = {
            'preConds': {'pop': sourcePop},
            'postConds': {'pop': targetPop},
            rule: val,
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
        self.autoCreatedPops.append(label)
        self.netParams.popParams[label] = {'cellType': cellModel, 'numCells': size}

    def createArtificialCells(self, label, number, params=None):
        print(f"Netpyne:: Creating artif cells for node '{label}' of {number} neurons")
        self.spikeGeneratorPops.append(label)
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

        self.stimulate(length)

        # handling (potentially) two distinct intervals (see comment in `createAndPrepareNetwork()`)
        tvbIterationEnd = self.time + length
        def _(simTime): pass
        if self.nextIntervalFuncCall:
            while (self.nextIntervalFuncCall < min(tvbIterationEnd, sim.cfg.duration)):
                sim.run.runForInterval(self.nextIntervalFuncCall - self.time, _)
                self.intervalFunc(self.time)
                self.nextIntervalFuncCall = self.time + self.interval
        if tvbIterationEnd > self.time:
            if self.time < sim.cfg.duration:
                sim.run.runForInterval(tvbIterationEnd - self.time, _)

                if self.nextIntervalFuncCall:
                    # add correction to avoid accumulation of arithmetic error due to that h.t advances slightly more than the requested interval
                    correction = self.time - tvbIterationEnd
                    self.nextIntervalFuncCall += correction

    def stimulate(self, length):
        allNeuronsSpikes = {}
        allNeurons = []
        for device in self.spikeGenerators:
            allNeurons.extend(device.own_neurons)
            allNeuronsSpikes.update(device.spikesPerNeuron)

            device.spikesPerNeuron = {} # clear to prepare for next interval run

        intervalEnd = h.t + length
        for gid in allNeurons:
            spikes = allNeuronsSpikes.get(gid, [])
            spikes = h.Vector(spikes)
            sim.net.cells[gid].hPointp.play(spikes, intervalEnd)

    def finalize(self):
        if self.time < sim.cfg.duration:
            stopTime = None
        else:
            stopTime = sim.cfg.duration
        sim.run.postRun(stopTime)
        sim.gatherData()
        sim.analyze()
