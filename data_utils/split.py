from typing import List

class Split(object):
    def __init__(self, samples: List[tuple]):
        self.samples = samples

    def _len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        '''
        labels: [{"image": image file, "label": label}, ... ]
        '''
        return self.samples[index]