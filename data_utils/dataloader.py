from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import cv2 as cv
import numpy as np
import os
import json
from split import Split
from utils import collate_fn, make_std_mask
from vocab import Vocab

class OCRDataset(Dataset):
    def __init__(self, dir, image_size, out_level, vocab=None):
        super(OCRDataset, self).__init__()
        self.size = image_size
        self.out_level = out_level
        self.labels =  []
        self.vocab = vocab if vocab is not None else Vocab(dir, out_level)
        self.get_groundtruth(dir)

    def __len__(self):
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
        if self.out_level == "word":
            for i, w in enumerate(['<sos>'] + gt.split() + ["<eos>"]):
                tokens[i] = self.vocab.stoi[w]
        else:
            for i, w in enumerate(['<sos>'] + list(gt) + ["<eos>"]):
                tokens[i] = self.vocab.stoi[w]
        tokens = torch.from_numpy(tokens)
        
        return img, tokens, gt

    def get_groundtruth(self, img_dir):
        self.max_len = 0
        self.labels = []

        for folder in os.listdir(img_dir):
            labels = json.load(open(os.path.join(img_dir, folder, "label.json")))
            for img_file, label in labels.items():
                label = label.strip()
                if self.vocab.out_level == "character":
                    self.labels.append({"image": os.path.join(img_dir, folder, img_file), "tokens": list(label), "gt": label})
                    if self.max_len < len(list(label)):
                        self.max_len = len(list(label))
                else:
                    label = label.split()
                    self.labels.append({"image": os.path.join(img_dir, folder, img_file), "tokens": label.split(), "gt": label})
                    if self.max_len < len(label.split()):
                        self.max_len = len(label.split())

    def split_dataset(self) -> Tuple[Split]:
        trainsize = len(self.labels) * 0.8
        train_idx = np.random.sample(low=0, high=len(self.labels), size=(trainsize, )).tolist()
        val_idx = [i for i in range(0, len(self.labels)) if i not in train_idx]

        train_split = [self[idx] for idx in train_idx]
        val_split = [self[idx] for idx in val_idx]

        return train_split, val_split

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, imgs, tokens, trg, pad=0):
        self.imgs = imgs.cuda()
        self.src_mask = torch.from_numpy(np.ones([imgs.size(0), 1, 32], dtype=np.bool)).cuda()
        if trg is not None:
            self.tokens = tokens.cuda()
            self.tokens_mask = make_std_mask(self.tokens, pad)
            self.ntokens = (self.tokens != pad).sum()

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
    vocab = Vocab("/home/nguyennghiauit/Projects/synthesis_handwritting_creator/UIT_HWDB_word", out_level="character")
    listdataset = OCRDataset("/home/nguyennghiauit/Projects/synthesis_handwritting_creator/UIT_HWDB_word/train_data", 
        image_size=(128, -1), out_level="char", vocab=vocab)
    dataloader = torch.utils.data.DataLoader(listdataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
    for epoch in range(1):
        for batch_i, (imgs, tokens, gts) in enumerate(dataloader):
            print(imgs.shape)
            print(tokens)
            print(gts)
            print("+"*10)