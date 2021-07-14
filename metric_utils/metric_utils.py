import numpy as np
import torch
import fastwer

class MetricUtils:
    def __init__(self, vocab):
        self.vocab = vocab

    def greedy_decode(self, features):
        features = features
        features = features.argmax(dim=-1)

        return features

    def beamsearch_decode(self, features):
        pass 

    def encode(self, x):
        x = list(x.numpy().flatten())
        '''
        while self.vocab.index("<s>") in x:
            x.remove(self.vocab.index("<s>"))
        while self.vocab.index("<e>") in x:
            x.remove(self.vocab.index("<e>"))
        while self.vocab.index("<p>") in x:
            x.remove(self.vocab.index("<p>"))
        '''
        while self.vocab.index("<") in x:
            x.remove(self.vocab.index("<"))
        while self.vocab.index(">") in x:
            x.remove(self.vocab.index(">"))
        while self.vocab.index("PAD") in x:
            x.remove(self.vocab.index("PAD"))
        text = ""
        for c in x:
            c = int(c)
            text += self.vocab[c]
        return text

    def features_to_string(self, features, decoder="greedy"):
        if decoder == "greedy":
            text = self.encode(self.greedy_decode(features))
        else: 
            text = self.encode(self.beamsearch_decode(features))

        return text

    def EditDistance(self, a, b):
        m, n = len(a), len(b)
        query_table = np.zeros(shape=(m+1, n+1))
        query_table[0, :] = np.arange(0, n+1)
        query_table[:, 0] = np.arange(0, m+1)
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                if a[i-1] == b[j-1]:
                    query_table[i, j] = query_table[i-1, j-1]
                else: 
                    query_table[i, j] = 1 + min(query_table[i-1, j], query_table[i, j-1], query_table[i-1, j-1])

        return query_table[m, n]

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