import pickle
import torch
from torch.utils.data import DataLoader
from metric_utils.metrics import Metrics
from data_utils.utils import collate_fn
from data_utils.vocab import Vocab
from data_utils.split import Split
from model.transformer import make_model
import os
from loss_utils.LabelSmoothingLoss import LabelSmoothing, SimpleLossCompute, NoamOpt
from data_utils.dataloader import Batch, OCRDataset
from tqdm import tqdm
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def validate_epoch(dataloader, epoch, model, loss_compute, metric):
    total_loss = 0
    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (imgs, char_labels, word_labels) in enumerate(dataloader, 1):
            batch = Batch(imgs, char_labels, word_labels)
            out = model(batch.imgs, batch.trg, batch.src_mask, batch.trg_mask)
            loss = loss_compute(out, batch.trg_y, batch.ntokens)
            total_loss += loss

            pbar.set_postfix(loss=total_loss / it)
            pbar.update()
            
    return total_loss / len(dataloader)

def train_epoch(dataloader, epoch, model, loss_compute):
    total_loss = 0
    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (imgs, char_labels, word_labels) in enumerate(dataloader, 1):
            batch = Batch(imgs, char_labels, word_labels)
            out = model(batch.imgs, batch.trg, batch.src_mask, batch.trg_mask)
            loss = loss_compute(out, batch.trg_y, batch.ntokens)
            total_loss += loss

            pbar.set_postfix(loss=total_loss / it)
            pbar.update()
            
    return total_loss / len(dataloader)

def train():
    if not os.path.isfile(f"vocab_{args['out_level']}.pkl"):
        vocab = Vocab(args["image_dir"], args["out_level"])
    else:
        vocab = pickle.load(open(f"vocab_{args['out_level'].pkl}", "rb"))

    dataset = OCRDataset(os.path.join(args["image_dir"], "train_data"), image_size=(-1, 64))
    
    model = make_model(len(vocab.stoi), N=args["num_layers"], d_model=args["d_model"], d_ff=args["dff"], 
                            h=args["head"], dropout=args["dropout"])
    if args["start_from"] is not None:
        model.load_state_dict(torch.load(args["start_from"]))
    model.cuda()
    criterion = LabelSmoothing(size=len(vocab.stoi), padding_idx=0, smoothing=args["smoothing"])
    criterion.cuda()
    model_opt = NoamOpt(model.tgt_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=args["lr"], betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(1, args["max_epoch"]+1):
        train_data, val_data = dataset.split_dataset()
        train_dataset = Split(train_data)
        val_dataset = Split(val_data)
        train_dataloader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=True, collate_fn=collate_fn)
        model.train()
        train_loss = train_epoch(train_dataloader, epoch, model, 
            SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        val_loss, scores = validate_epoch(val_dataloader, epoch, model, 
            SimpleLossCompute(model.generator, criterion, None))
        print(f"Training loss: {train_loss}")
        print(f"Validation loss: {val_loss}")
        print(f"Validation score: {scores['cer']} CER - {scores['wer']} WER")
        print("+"*13)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str,
                            help="Directory to image folder")
    parser.add_argument("--out-level", type=str, 
                            help="predict words or characters")
    parser.add_argument("--checkpoint-path", tyoe="str",
                            help="path to checkpoint folder")
    parser.add_argument("--lr", type=float,
                            help="learning rate", default=5e-5)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--smoothing", type=float, default=0.3)
    parser.add_argument("--num-layers", type=int, help="number of encoder-decoder layers per encoder-decoder module", default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--dff", type=int, help="feed forward dimension", defautl=1024)
    parser.add_argument("--head", type=int, help="number of heads", default=8)
    parser.add_argument("--beam-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--start-from", type=str, help="resume model from previous checkpoint", default=None)
    parser.add_argument("--max-epoch", type=int, default=100)

    args = parser.parse_args()
    args = vars(args)

    train()