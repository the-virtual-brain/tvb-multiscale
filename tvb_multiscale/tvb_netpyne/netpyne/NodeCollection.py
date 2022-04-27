class NodeCollection(object):
    
    def __init__(self, brain_region, pop_label, size):
        # TODO: these two properties seem to be excessive..
        self.brain_region = brain_region
        self.pop_label = pop_label
        self.size = size

    def __len__(self):
        return self.size

