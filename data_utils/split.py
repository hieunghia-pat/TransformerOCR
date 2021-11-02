from typing import List
import cv2 as cv
import torch
import numpy as np

class Split(object):
    def __init__(self, samples: List[tuple]):
        self.labels = samples

    def _len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        '''
        labels: [{"image": image file, "tokens": ["B", "a", "o", " ", "g", "ồ", "m"], "gt": "Bao gồm"}, ... ]
        '''

        label = self.labels[index]
        img_path, tokens, gt = label["image"], label["tokens"], label["gt"]
        img = cv.imread(img_path)

        img = img / 255.

        # resize the image 
        img_h, img_w, _ = img.shape 
        w, h = self.size 
        if w == -1: # keep h, scale w according to h
            scale = img_w / img_h 
            w = round(scale * h)

        if h == -1: # keep w, scale h according to w
            scale = img_w / img_h 
            h = round(w / scale)

        img = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)
            
        # Channels-first
        img = np.transpose(img, (2, 0, 1))
        # As pytorch tensor
        img = torch.from_numpy(img).float()

        tokens = np.zeros(self.max_len, dtype=int)
        for i, w in enumerate('<sos>' + gt.split() + "<eos>"):
            tokens[i] = self.vocab["word_to_ix"][w]
        tokens = torch.from_numpy(tokens)
        
        return img, tokens, gt