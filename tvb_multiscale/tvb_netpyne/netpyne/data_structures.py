class NetpyneCellGeometry(object):

    def __init__(self, diam, length, axialResistance) :
        self.diam = diam
        self.length = length
        self.axialR = axialResistance

    def toDict(self):
        return {'diam': self.diam, 'L': self.length, 'Ra': self.axialR}

class NetpyneMechanism(object):

    def __init__(self, name, gNaBar, gKBar, gLeak, eLeak):
        self.name = name
        self.gNaBar = gNaBar
        self.gKBar = gKBar
        self.gLeak = gLeak
        self.eLeak = eLeak

    def toDict(self):
        return {'gnabar': self.gNaBar, 'gkbar': self.gKBar, 'gl': self.gLeak, 'el': self.eLeak}

class NetpyneCellModel(object):

    def __init__(self, name, geom, mech):
        self.name = name
        self.geom = geom
        self.mech = mech

    def geometry(self):
        return self.geom.toDict()

    def getMech(self):
        return self.mech.name, self.mech.toDict()




        