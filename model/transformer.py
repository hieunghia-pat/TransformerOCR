import torch
import torch.nn as nn
import copy
from torchvision import models
from data_utils.utils import subsequent_mask
from model.utils import clones, SublayerConnection, PositionwiseFeedForward, FeatureExtractor
from model.embedding import Embeddings, PositionalEncoding
from model.encoder import Encoder
from model.decoder import Decoder
from model.attention import MultiHeadedAttention
from model.generator import Generator

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def get_predictions(self, images, src_mask, vocab, max_len):
        encoded_features = self.encode(images, src_mask)
        batch_size = images.shape[0]
        
        ys = torch.ones(size=(batch_size, 1)).fill_(vocab.sos_token).long().cuda()
        for it in range(max_len):
            tgt_mask = subsequent_mask(ys.shape[-1]).long().cuda()
            outs = self.decode(encoded_features, src_mask, ys, tgt_mask)
            outs = self.generator(outs[:, -1])
            outs = outs.argmax(dim=-1)
            ys = torch.cat(ys, outs, dim=1)
        
        return ys

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def make_model(vocab_size, N=4, 
               d_model=256, d_ff=1024, d_feature=1024, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    resnet = getattr(models, 'resnet101')(pretrained=False)

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(FeatureExtractor(d_model, resnet, 'layer3', d_feature), c(position)),
        nn.Sequential(Embeddings(d_model, vocab_size), c(position)),
        Generator(d_model, vocab_size))
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
            
    return model