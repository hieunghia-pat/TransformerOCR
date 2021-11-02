from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import cv2 as cv
import numpy as np
import os
import json
import pickle
from data_utils.split import Split
from data_utils.utils import make_std_mask

class OCRDataset(Dataset):
    def __init__(self, dir, img_size, vocab):
        super(OCRDataset, self).__init__()
        self.size = img_size
        self.labels =  []
        self.vocab = vocab

        chars = list(set(chars))
        words = list(set(words))
        self.vocab["vocab_char"] = chars
        self.vocab["vocab_word"] = words

        self.vocab["ix_to_char"] = {i: c for i, c in enumerate(chars, 4)}
        self.vocab["ix_to_char"][0] = "<pad>"
        self.vocab["ix_to_char"][1] = "<sos>"
        self.vocab["ix_to_char"][2] = "<eos>"
        self.vocab["ix_to_char"][3] = "<unk>"
        self.vocab["char_to_ix"] = {c: i for i, c in self.vocab["ix_to_char"].items()}

        self.vocab["ix_to_word"] = {i: w for i, w in enumerate(words, 4)}
        self.vocab["ix_to_word"][0] = "<pad>"
        self.vocab["ix_to_word"][1] = "<sos>"
        self.vocab["ix_to_word"][2] = "<eos>"
        self.vocab["ix_to_word"][3] = "<unk>"
        self.vocab["word_to_ix"] = {w: i for i, w in self.vocab["ix_to_word"].items()}

        self.vocab["max_char"] = self.max_char
        self.vocab["max_word"] = self.max_word

        pickle.dump(self.vocab, open("vocab.pkl", "rb+"))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        '''
        labels: [{"image": image file, "label": label}, ... ]
        '''

        label = self.labels[index]
        img_path, label = label["image"], label["label"]
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

        word_label = np.zeros(self.max_len, dtype=int)
        for i, w in enumerate('<sos>' + label.split() + "<eos>"):
            word_label[i] = self.vocab["word_to_ix"][w]
        word_label = torch.from_numpy(word_label)

        char_label = np.zeros(len(label) + 2, dtype=int)
        for i, c in enumerate('<sos>' + list(label) + "<eos>"):
            char_label[i] = self.vocab["char_to_ix"][c]
        char_label = torch.from_numpy(char_label)
        
        return img, char_label, word_label

    def split_dataset(self) -> Tuple[Split]:
        trainsize = len(self.labels) * 0.8
        train_idx = np.random.randint(low=0, high=len(self.labels), size=(trainsize, )).tolist()
        val_idx = [i for i in range(0, len(self.labels)) if i not in train_idx]

        train_split = [self[idx] for idx in train_idx]
        val_split = [self[idx] for idx in val_idx]

        return train_split, val_split

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, imgs, char_y, word_y, trg, pad=0):
        self.imgs = imgs.cuda()
        self.src_mask = torch.from_numpy(np.ones([imgs.size(0), 1, 32], dtype=np.bool)).cuda()
        if trg is not None:
            self.char_y = char_y.cuda()
            self.word_y = word_y.cuda()
            self.char_y_mask = make_std_mask(self.char_y, pad)
            self.word_y_mask = make_std_mask(self.word_y, pad)
            self.char_ntokens = (self.char_y != pad).sum()
            self.word_ntokens = (self.word_y != pad).sum()

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, name):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.name = name

    def forward(self, x):
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name is self.name:
                b = x.size(0)
                c = x.size(1)
                return x.view(b, c, -1).permute(0, 2, 1)
        
        return None

if __name__=='__main__':
    listdataset = OCRDataset('your-lines')
    dataloader = torch.utils.data.DataLoader(listdataset, batch_size=2, shuffle=False, num_workers=0)
    for epoch in range(1):
        for batch_i, (imgs, labels_y, labels) in enumerate(dataloader):
            continue