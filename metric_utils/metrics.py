import numpy as np
import fastwer

class Metrics(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def get_error(self, label, true_label, mode):
        error = 0
        if mode == "character":
            error += fastwer.score_sent(label,true_label, char_level=True)
        else: 
            error += fastwer.score_sent(label, true_label)

        return error

    def get_scores(self, predicted, gts):
        batch_size = predicted.shape[0]
        cer = 0
        wer = 0
        for batch_idx in range(batch_size):
            predicted_label = " ".join(predicted[batch_idx])
            gt_label = " ".join(gts[batch_idx])
            cer += self.get_error(predicted_label, gt_label, mode="character")
            wer += self.get_error(predicted_label, gt_label, mode="word")

        return {
            "cer": cer / batch_size,
            "wer": wer / batch_size
        }