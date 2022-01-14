import pickle
import torch
from torch.utils.data.dataloader import DataLoader
from metric_utils.metrics import Metrics
from metric_utils.tracker import Tracker
from data_utils.vocab import Vocab
from model.transformer import make_model
import os
from loss_utils.LabelSmoothingLoss import LabelSmoothing, SimpleLossCompute
from loss_utils.WrappedOptim import WrappedOptim
from data_utils.dataloader import Batch, OCRDataset, collate_fn
from tqdm import tqdm

import config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.is_available():
    device = "cuda"
else: 
    device = "cpu"

def run_epoch(loaders, train, prefix, epoch, fold, model, loss_compute, metric, tracker):
    if train:
        model.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        model.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}

    if config.start_from and fold > 0:
        saved_info = torch.load(config.start_from, map_location=device)
        loss_tracker = saved_info["loss_tracker"]
    else:
        loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))

    if not train:
        cer_tracker = tracker.track('{}_cer'.format(prefix), tracker_class(**tracker_params))
        wer_tracker = tracker.track('{}_wer'.format(prefix), tracker_class(**tracker_params))

    for fold_idx in range(fold, len(loaders)):
        loader = loaders[fold_idx]
        try:
            dataset = loader.dataset.dataset
        except:
            dataset = loader.dataset
        pbar = tqdm(loader, desc='Epoch {} - {} - Fold {}'.format(epoch+1, prefix, fold_idx+1), unit='it', ncols=0)
        
        for imgs, tokens, shifted_tokens in pbar:
            batch = Batch(imgs, tokens, shifted_tokens, dataset.vocab.padding_idx, device=device)

            fmt = '{:.4f}'.format
            if train:
                logprobs = model(batch.imgs, batch.tokens, batch.src_mask, batch.tokens_mask)
                loss = loss_compute(logprobs, batch.shifted_right_tokens, batch.ntokens)
                loss_tracker.append(loss.item())
                pbar.set_postfix(loss=fmt(loss.item()))
            else:
                outs = model.get_predictions(batch.imgs, batch.src_mask, dataset.vocab, dataset.max_len)
                scores = metric.get_scores(dataset.vocab.decode_sentence(outs.to("cpu")), dataset.vocab.decode_sentence(tokens.to("cpu")))
                wer_tracker.append(scores["wer"])
                cer_tracker.append(scores["cer"])
                pbar.set_postfix(cer=fmt(scores["cer"]), wer=fmt(scores["wer"]))
            
            pbar.update()

        if train:
            torch.save({
                "epoch": epoch,
                "fold": fold_idx,
                "state_dict": model.state_dict(),
                "loss_tracker": loss_tracker,
            }, os.path.join(config.checkpoint_path, "last_model.pth"))

            # torch.save({
            #     "state_dict": loss_compute.opt.optimizer.state_dict()
            # }, config.saved_optimizer)

    if not train:
        return {
            "cer": cer_tracker.mean.value,
            "wer": wer_tracker.mean.value
        }
    
    return loss_tracker.mean.value

def train():
    if not os.path.isfile(os.path.join(config.checkpoint_path, f"vocab_{config.out_level}.pkl")):
        vocab = Vocab(config.image_dirs, config.out_level)
        pickle.dump(vocab, open(os.path.join(config.checkpoint_path, f"vocab_{config.out_level}.pkl"), "wb"))
    else:
        vocab = pickle.load(open(os.path.join(config.checkpoint_path, f"vocab_{config.out_level}.pkl"), "rb"))

    train_dataset = OCRDataset(config.train_image_dirs, image_size=config.image_size, out_level=config.out_level, vocab=vocab)
    test_dataset = OCRDataset(config.test_image_dirs, image_size=config.image_size, out_level=config.out_level, vocab=vocab)
    metric = Metrics(vocab)
    tracker = Tracker()
    
    model = make_model(len(vocab.stoi), N=config.num_layers, d_model=config.d_model, d_ff=config.dff, 
                            h=config.heads, dropout=config.dropout) 

    model.to(device)
    criterion = LabelSmoothing(size=len(vocab.stoi), padding_idx=vocab.padding_idx, smoothing=config.smoothing)
    criterion.to(device)

    # model_opt = WrappedOptim(config.d_model, config.factor, config.warmup, 
    #                             torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9))
    model_opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    # if os.path.isfile(config.saved_optimizer):
    #     model_opt.optimizer.load_state_dict(torch.load(config.optimizer)["state_dict"])

    if config.start_from:
        saved_info = torch.load(config.start_from, map_location=device)
        model.load_state_dict(saved_info["state_dict"])
        from_epoch = saved_info["epoch"]
        from_fold = saved_info["fold"] + 1
        model.load_state_dict(saved_info["state_dict"])
    else:
        from_epoch = 0
        from_fold = 0

    if os.path.isfile(os.path.join(config.checkpoint_path, f"folds_{config.out_level}.pkl")):
        folds = pickle.load(open(os.path.join(config.checkpoint_path, f"folds_{config.out_level}.pkl"), "rb"))
    else:
        folds = train_dataset.get_folds(k=10)
        pickle.dump(folds, open(os.path.join(config.checkpoint_path, f"folds_{config.out_level}.pkl"), "wb"))

    test_dataloader = DataLoader(test_dataset, 
                                batch_size=config.batch_test, 
                                shuffle=True, 
                                collate_fn=collate_fn)

    if os.path.isfile(os.path.join(config.checkpoint_path, "best_model.pth")):
        best_info = torch.load(os.path.join(config.checkpoint_path, "best_model.pth"), map_location=device)
        best_scores = best_info["scores"]
    else:
        best_scores = {
            "cer": float("inf"),
            "wer": float("inf")
        }

    for epoch in range(from_epoch, config.max_epoch):
        loss = run_epoch(folds, True, "Training", epoch, from_fold, model, 
            SimpleLossCompute(model.generator, criterion, model_opt), metric, tracker)

        if loss:
            print(f"Training loss: {loss}")
        else:
            loss = saved_info["loss_tracker"].mean.value

        if loss <= 1.:
            test_scores = run_epoch([test_dataloader], False, "Evaluation", epoch, 0, model, 
                    SimpleLossCompute(model.generator, criterion, None), metric, tracker)

            if best_scores["cer"] > test_scores["cer"]:
                best_scores = test_scores
                torch.save({
                    "vocab": vocab,
                    "state_dict": model.state_dict(),
                    "model_opt": model_opt,
                    "scores": test_scores,
                }, os.path.join(config.checkpoint_path, "best_model.pth"))

            print(f"CER on the test set: {test_scores['cer']} - WER on the test set: {test_scores['wer']}")

        print("*"*13)
        from_fold = 0

    print(f"Training completed. Best scores on test set: CER = {best_scores['cer']} - WER = {best_scores['wer']}.")

if __name__=='__main__':

    train()