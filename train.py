import torch
from torch.utils.data import DataLoader
import time
from model.transformer import make_model
import os
from loss_utils.LabelSmoothingLoss import LabelSmoothing, SimpleLossCompute
from data_utils.dataloader import Batch, OCRDataset
from tqdm import tqdm
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_epoch(dataloader, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, (imgs, char_labels, word_labels) in enumerate(dataloader):
        batch = Batch(imgs, char_labels, word_labels)
        out = model(batch.imgs, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

def train():
    batch_size = 1

    train_dataloader = DataLoader(OCRDataset(os.path.join(args["image_dir"], "train_data"), image_size=(64, -1)))
    test_dataloader = DataLoader(OCRDataset(os.path.join(args["image_dir"], "test_data"), image_size=(64, -1)))
    
    model = make_model(len(char2token))
    # model.load_state_dict(torch.load('your-pretrain-model-path'))
    model.cuda()
    criterion = LabelSmoothing(size=len(char2token), padding_idx=0, smoothing=0.1)
    criterion.cuda()
    model_opt = NoamOpt(model.tgt_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in tqdm(range(1, 10000)):
        model.train()
        run_epoch(train_dataloader, model, 
              SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        test_loss = run_epoch(test_dataloader, model, 
              SimpleLossCompute(model.generator, criterion, None))
        print("test_loss", test_loss)
        torch.save(model.state_dict(), 'checkpoint/%08d_%f.pth'%(epoch, test_loss))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str,
                            help="Directory to image folder")
    parser.add_argument("--out-level", type=str, 
                            help="predict words or characters")
    parser.add_argument("--checkpoint-path", tyoe="str",
                            help="path to checkpoint folder")
    parser.add_argument("--lr", type=float,
                            help="learning rate")
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--num-layers", type=int, help="number of encoder-decoder layers per encoder-decoder module", default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--dff", type=int, help="feed forward dimension", defautl=1024)
    parser.add_argument("--head", type=int, help="number of heads", default=8)
    parser.add_argument("--beam-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--start-from", type=str, help="resume model from previous checkpoint", default=None)

    args = parser.parse_args()
    args = vars(args)

    train()