import torch
from data_loader import ListDataset
from torch.autograd import Variable
from data_loader import subsequent_mask, Batch
from data_loader import char2token, vocab, token2char
from model import make_model
from metric_utils.metric_utils import MetricUtils
import cv2 as cv
import os
import numpy as np
import fastwer
# uncomment these lines if this code is run on google colab
# from google.colab.patches import cv2_imshow
# import numpy as np

src_mask = torch.from_numpy(np.ones([1, 1, 32], dtype=np.bool)).cuda()
metric = MetricUtils(["PAD"] + list(vocab))

def greedy_decode(src, model, max_len=20, start_symbol=1):
    model.eval()
    global src_mask
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).long().cuda()
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .long().cuda()))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).long().cuda().fill_(next_word)], dim=1)
        if token2char[next_word.item()] == '>':
            break
    ret = ys.cpu().numpy()[0]
    out = [token2char[i] for i in ret]
    out = "".join(out[1:-1])
    return out

def validation(val_dataloader, model, visual=False):
    total_cer, total_wer = 0, 0
    for images, labels, _ in val_dataloader:
        pred = greedy_decode(images.to("cuda"), model)
        text = metric.encode(labels[0])
        cer = fastwer.score_sent(pred, text, char_level=True)
        wer = fastwer.score_sent(pred, text)
        if visual:
            image = images[0].cpu().numpy()
            image = image.transpose(1, 2, 0) * 255
            # os.system("clear")
            print("Original content: ", metric.encode(labels))
            print("Predicted content: ", pred)
            image = cv.resize(image.astype(np.uint8), (128, 64), interpolation=cv.INTER_AREA)
            cv.imshow("Image", image)
            cv.waitKey()
            # uncomment this line if this code is run on google colab
            # cv2_imshow(image)
            print("==============")
        total_cer += cer 
        total_wer += wer
    val_size = len(val_dataloader)
    return total_cer / val_size, total_wer / val_size
