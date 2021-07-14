import torch
from data_loader import ListDataset
from torch.autograd import Variable
from data_loader import subsequent_mask
from data_loader import char2token, vocab, token2char
from model import make_model
from metric_utils.metric_utils import MetricUtils
import cv2 as cv
import os
from pathlib import Path
import numpy as np
from predict import validation

batch_size = 1
val_dataloader = torch.utils.data.DataLoader(ListDataset("/path/to/the/dataset", (128, 64)), batch_size=batch_size, shuffle=False, num_workers=0)
model = make_model(len(char2token)).cuda()

checkpoints = Path("checkpoints")
checkpoints = list(checkpoints.glob("*.pth"))
for checkpoint in checkpoints:
    print(f"Checkpoint: {str(checkpoint)}")
    saved = torch.load(str(checkpoint))
    model.load_state_dict(saved["model"])
    model.cuda()
    print("Calculating ...")
    cer, wer = validation(val_dataloader, model, visual=False)
    print(f"CER: {cer} - WER: {wer}")
    print("========")