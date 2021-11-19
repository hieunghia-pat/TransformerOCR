import fastwer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

    def get_scores(self, predicteds, gts):
        cer = 0
        wer = 0
        prec = 0
        rec = 0
        f1 = 0
        acc = 0
        batch_size = len(gts)
        for predicted, gt in zip(predicteds, gts):
            acc += accuracy_score(gt, predicted)
            prec += precision_score(gt, predicted, average="macro", zero_division=0)
            rec += recall_score(gt, predicted, average="macro", zero_division=0)
            f1 += f1_score(gt, predicted, average="macro", zero_division=0)
            if len(predicted) == 0:
                cer += len(gt)
                wer += len(gt.split())
                continue
            cer += self.get_error(predicted, gt, mode="character")
            wer += self.get_error(predicted, gt, mode="word")

        return {
            "cer": cer / batch_size,
            "wer": wer / batch_size,
            "accuracy": acc / batch_size,
            "precision": prec / batch_size,
            "recall": rec / batch_size,
            "f1": f1 / batch_size
        }