
import numpy as np

class Recorder:
    def __init__(self, cell_size, track_id):
        self.mask = np.zeros((cell_size,cell_size), dtype = np.bool)
        self.track_id = track_id
        self.counter = 0
        self.hit = True
    
