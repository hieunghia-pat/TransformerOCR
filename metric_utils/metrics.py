import numpy as np
import torch
import fastwer

class Metrics(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def greedy_decode(self, features):
        features = features
        features = features.argmax(dim=-1)

        return features

    def beamsearch_decode(self, logouts):
        pass 

    def encode(self, x):
        return

    def er(self, labels, true_labels, mode):
        labels = labels.cpu()
        true_labels = true_labels.cpu()
        batch_size = labels.size()[0]
        batch_ed = 0
        for batch in range(batch_size):
            decoded_label = self.greedy_decode(labels[batch])
            encoded_label = self.encode(decoded_label)
            encoded_true_label = self.encode(true_labels[batch])
            error = 0
            if mode == "character":
                error += fastwer.score_sent(encoded_label, encoded_true_label, char_level=True)
            else: 
                error += fastwer.score_sent(encoded_label, encoded_true_label)
            batch_ed += error

        return batch_ed / batch_size

    def get_CER(self, validation_data, model):
        cer = 0 # character error
        len_data = len(validation_data)
        for data_point in validation_data:
            images, texts = data_point["image"], data_point["text"]
            with torch.no_grad():
                predicted_texts = model(images, texts)
            cer += self.er(predicted_texts, texts, mode="character")
        
        return cer / len_data

    def get_WER(self, validation_data, model):
        cer = 0 # character error
        len_data = len(validation_data)
        for data_point in validation_data:
            images, texts = data_point["image"], data_point["text"]
            with torch.no_grad():
                predicted_texts = model(images, texts)
            cer += self.er(predicted_texts, texts, mode="word")
        
        return cer / len_data