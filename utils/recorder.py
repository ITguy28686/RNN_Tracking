
import numpy as np

class Recorder:
    def __init__(self, sell_size, track_id):
        self.mask = np.zeros((sell_size,sell_size), dtype = np.bool)
        self.track_id = track_id
        self.counter = 0
    
