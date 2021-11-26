import pickle
import torch
from torch.nn import DataParallel
from torch.utils.data.dataloader import DataLoader
from metric_utils.metrics import Metrics
from metric_utils.tracker import Tracker
from data_utils.vocab import Vocab
from model.transformer import make_model
import os
from loss_utils.LabelSmoothingLoss import LabelSmoothing, SimpleLossCompute, NoamOpt
from data_utils.dataloader import Batch, OCRDataset, collate_fn
from tqdm import tqdm

import config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_epoch(loaders, train, prefix, epoch, fold, stage, model, loss_compute, metric, tracker):
    if train:
        model.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        model.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}

    for fold_idx in range(fold, len(loaders)):
        loader = loaders[fold_idx]
        try:
            dataset = loader.dataset.dataset
        except:
            dataset = loader.dataset
        pbar = tqdm(loader, desc='Epoch {} - {} - Fold {}'.format(epoch+1, prefix, loaders.index(loader)+1), unit='it', ncols=0)
        loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
        cer_tracker = tracker.track('{}_cer'.format(prefix), tracker_class(**tracker_params))
        wer_tracker = tracker.track('{}_wer'.format(prefix), tracker_class(**tracker_params))
        
        for imgs, tokens, shifted_tokens in pbar:
            batch = Batch(imgs, tokens, shifted_tokens, dataset.vocab.padding_idx)

            fmt = '{:.4f}'.format
            if train:
                logprobs = model(batch.imgs, batch.tokens, batch.src_mask, batch.tokens_mask)
                loss = loss_compute(logprobs, batch.shifted_right_tokens, batch.ntokens)
                loss_tracker.append(loss.item())
                pbar.set_postfix(loss=fmt(loss_tracker.mean.value))
            else:
                outs = model.get_predictions(batch.imgs, batch.src_mask, dataset.vocab, dataset.max_len)
                scores = metric.get_scores(dataset.vocab.decode_sentence(outs.cpu()), dataset.vocab.decode_sentence(tokens.cpu()))
                wer_tracker.append(scores["wer"])
                cer_tracker.append(scores["cer"])
                pbar.set_postfix(cer=fmt(cer_tracker.mean.value), wer=fmt(wer_tracker.mean.value))
            
            pbar.update()

        if config.debug and train:
            torch.save({
                "stage": stage,
                "epoch": epoch,
                "fold": loaders.index(loader)+1,
                "state_dict": model.state_dict(),
                "model_opt": loss_compute.opt,
                "loss": loss_tracker.mean.value
            }, os.path.join(config.tmp_checkpoint_path, "last_model.pth"))

        if not train:
            return {
                "cer": cer_tracker.mean.value,
                "wer": wer_tracker.mean.value
            }
        else:
            return loss_tracker.mean.value

def train():
    if not os.path.isfile(os.path.join(config.checkpoint_path, f"vocab_{config.out_level}.pkl")):
        vocab = Vocab(config.image_dir, config.out_level)
        pickle.dump(vocab, open(os.path.join(config.checkpoint_path, f"vocab_{config.out_level}.pkl"), "wb"))
    else:
        vocab = pickle.load(open(os.path.join(config.checkpoint_path, f"vocab_{config.out_level}.pkl"), "rb"))

    train_dataset = OCRDataset(dir=os.path.join(config.image_dir, "train_data"), image_size=config.image_size, out_level=config.out_level, vocab=vocab)
    test_dataset = OCRDataset(dir=os.path.join(config.image_dir, "test_data"), image_size=config.image_size, out_level=config.out_level, vocab=vocab)
    metric = Metrics(vocab)
    tracker = Tracker()
    
    model = make_model(len(vocab.stoi), N=config.num_layers, d_model=config.d_model, d_ff=config.dff, 
                            h=config.heads, dropout=config.dropout)
    # model = DataParallel(model)

    model.cuda()
    criterion = LabelSmoothing(size=len(vocab.stoi), padding_idx=vocab.padding_idx, smoothing=config.smoothing)
    criterion.cuda()
    model_opt = NoamOpt(model.tgt_embed[0].d_model, 1, config.warmup,
            torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9))

    if config.start_from is not None:
        saved_info = torch.load(config.start_from)
        model.load_state_dict(saved_info["state_dict"])
        from_stage = saved_info["stage"]
        from_epoch = saved_info["epoch"]
        from_fold = saved_info["fold"]
        loss = saved_info["loss"]
        model.load_state_dict(saved_info["state_dict"])
        model_opt = saved_info["model_opt"]
    else:
        from_stage = 0
        from_epoch = 0
        from_fold = 0
        loss = None

    if os.path.isfile(os.path.join(config.checkpoint_path, f"folds_{config.out_level}.pkl")):
        folds = pickle.load(open(os.path.join(config.checkpoint_path, f"folds_{config.out_level}.pkl"), "rb"))
    else:
        folds = train_dataset.get_folds()
        pickle.dump(folds, open(os.path.join(config.checkpoint_path, f"folds_{config.out_level}.pkl"), "wb"))

    test_dataloder = DataLoader(test_dataset, 
                                batch_size=config.batch_size, 
                                shuffle=True, 
                                collate_fn=collate_fn)

                                
    for stage in range(from_stage, len(folds)):
        best_scores = {
                "cer": 0,
                "wer": 0
        }
        
        loss = float("inf")
        for epoch in range(from_epoch, config.max_epoch):
            tmp_loss = run_epoch(folds[:-1], True, "Training", epoch, from_fold, stage, model, 
                SimpleLossCompute(model.generator, criterion, model_opt), metric, tracker)
            loss = tmp_loss if tmp_loss is not None else loss

            if loss <= 1.:
                val_scores = run_epoch([folds[-1]], False, "Validation", epoch, 0, stage, model, 
                    SimpleLossCompute(model.generator, criterion, None), metric, tracker)

                if best_scores["cer"] < val_scores["cer"]:
                    best_scores = val_scores
                    torch.save({
                        "vocab": vocab,
                        "state_dict": model.state_dict(),
                        "model_opt": model_opt,
                        "val_scores": val_scores,
                    }, os.path.join(config.checkpoint_path, f"best_model_stage_{stage+1}.pth"))

                torch.save({
                    "vocab": vocab,
                    "state_dict": model.state_dict(),
                    "model_opt": model_opt,
                    "val_scores": val_scores,
                }, os.path.join(config.checkpoint_path, f"last_model_stage_{stage+1}.pth"))

            print("*"*13)
            from_fold = 0 # start a new epoch

        test_scores = run_epoch([test_dataloder], False, "Evaluation", epoch, 0, stage, model, 
                SimpleLossCompute(model.generator, criterion, None), metric, tracker)

        print(f"Stage {stage+1} completed. Scores on test set: CER = {test_scores['cer']} - WER = {test_scores['wer']}.")
        print("="*23)

        # swapping folds
        for idx in range(len(folds)):
            tmp_fold = folds[idx]
            folds[idx] = folds[idx - 1]
            folds[idx-1] = tmp_fold

        # saving for the new swapped folds
        pickle.dump(folds, open(os.path.join(config.checkpoint_path, f"folds_{config.out_level}.pkl"), "wb"))

if __name__=='__main__':

    train()