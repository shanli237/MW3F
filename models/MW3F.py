import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
class Transformer(nn.Module):
    def __init__(self, embed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        """
        Transformer model with encoder and decoder.

        Parameters:
        embed_dim (int): Dimension of the embeddings.
        nhead (int): Number of attention heads.
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
        """
        super().__init__()

        self.encoder = TransformerEncoder(TransformerEncoderLayer(embed_dim, nhead, dim_feedforward, dropout),
                                          num_encoder_layers)
        self.decoder = TransformerDecoder(TransformerDecoderLayer(embed_dim, nhead, dim_feedforward, dropout),
                                          num_decoder_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize model parameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed):
        """
        Forward pass of the Transformer model.

        Parameters:
        src (Tensor): Input source sequence.
        query_embed (Tensor): Query embeddings.
        pos_embed (Tensor): Positional embeddings.

        Returns:
        Tensor: Output of the decoder.
        """
        bs = src.shape[0]
        memory = self.encoder(src, pos_embed)
        query_embed = query_embed.repeat(bs, 1, 1)#label embedding
        tgt = torch.zeros_like(query_embed) 
        output = self.decoder(tgt, memory, pos_embed, query_embed)

        return output

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        """
        Transformer encoder composed of multiple encoder layers.

        Parameters:
        encoder_layer (nn.Module): Encoder layer.
        num_layers (int): Number of encoder layers.
        """
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, pos):
        """
        Forward pass of the Transformer encoder.

        Parameters:
        src (Tensor): Input source sequence.
        pos (Tensor): Positional embeddings.

        Returns:
        Tensor: Encoder output.
        """
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        return output

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        """
        Transformer decoder composed of multiple decoder layers.

        Parameters:
        decoder_layer (nn.Module): Decoder layer.
        num_layers (int): Number of decoder layers.
        """
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos, query_pos):
        """
        Forward pass of the Transformer decoder.

        Parameters:
        tgt (Tensor): Target sequence.
        memory (Tensor): Encoder memory.
        pos (Tensor): Positional embeddings.
        query_pos (Tensor): Query positional embeddings.

        Returns:
        Tensor: Decoder output.
        """
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, pos, query_pos)

        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, nhead, dim_feedforward, dropout):
        """
        Transformer encoder layer.

        Parameters:
        embed_dim (int): Dimension of the embeddings.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, src, pos):
        """
        Forward pass of the Transformer encoder layer.

        Parameters:
        src (Tensor): Input source sequence.
        pos (Tensor): Positional embeddings.

        Returns:
        Tensor: Encoder layer output.
        """
        src2 = self.self_attn(query=src + pos, key=src + pos, value=src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, nhead, dim_feedforward, dropout):
        """
        Transformer decoder layer.

        Parameters:
        embed_dim (int): Dimension of the embeddings.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, tgt, memory, pos, query_pos):
        """
        Forward pass of the Transformer decoder layer.

        Parameters:
        tgt (Tensor): Target sequence.
        memory (Tensor): Encoder memory.
        pos (Tensor): Positional embeddings.
        query_pos (Tensor): Query embeddings.

        Returns:
        Tensor: Decoder layer output.
        """
        tgt2 = self.self_attn(query=tgt + query_pos, key=tgt + query_pos, value=tgt)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt + query_pos, key=memory + pos, value=memory)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt



class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Basic block for residual learning.

        Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        """
        super(BasicBlock, self).__init__()
        self._norm_layer = torch.nn.BatchNorm1d
        self.stride = 1
        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, 3, self.stride, padding=1),
            self._norm_layer(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv1d(out_channels, out_channels, 3, padding=1),
            self._norm_layer(out_channels),
            torch.nn.ReLU(),
        )

        if in_channels != out_channels:
            self.res_layer = torch.nn.Conv1d(in_channels, out_channels, 1, self.stride)
        else:
            self.res_layer = None

    def forward(self, x):
        """
        Forward pass of the BasicBlock.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after residual learning.
        """
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x) + residual


class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(ConvBlock1d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,dilation=dilation, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,kernel_size=kernel_size,dilation=dilation, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        self.last_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.last_relu(out + res)

class LocalProfiling(nn.Module):
    """ Local Profiling module in ARES """
    def __init__(self):
        super(LocalProfiling, self).__init__()
        
        self.net = nn.Sequential(
            ConvBlock1d(in_channels=1, out_channels=32, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=32, out_channels=64, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=64, out_channels=128, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=128, out_channels=256, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        x = self.net(x)
        x= x.permute(0, 2, 1)
        return x




class MW3F(nn.Module):
    def __init__(self, num_classes=100):
        """
        Transformer Model with CNN and DFNet (TMWF).

        Parameters:
        num_classes (int): Number of output classes.
        num_tab (int): Number of tabs.
        """
        super(MW3F, self).__init__()
        embed_dim = 256
        nhead = 8
        dim_feedforward = 256 * 4
        num_encoder_layers = 2
        num_decoder_layers = 2
        max_len = 37
        dropout = 0.1

        self.cnn_layer1 = LocalProfiling()
        self.cnn_layer2 = LocalProfiling()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.trm = Transformer(embed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.pos_embed_d = nn.Embedding(max_len, embed_dim)
        self.pos_embed_t = nn.Embedding(max_len, embed_dim)
        self.query_embed = nn.Embedding(num_classes, embed_dim).weight
        self.fc = GroupWiseLinear(num_classes,embed_dim,bias=True)
        self.token_type_embeddings = nn.Embedding(2, embed_dim)
        self.token_type_embeddings.apply(self.init_weights)

    def init_weights(self,module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input):
        """
        Forward pass of the TMWF model.

        Parameters:
        input (Tensor): Input tensor.

        Returns:
        Tensor: Logits after passing through the model.
        """
        input1,input2 = input 
        
        x1 = self.cnn_layer1(input1)#in:torch.Size([4, 1, 1000]) --> out:torch.Size([4, 21, 256])
        x2 = self.cnn_layer2(input2)
        p1 = self.pos_embed_d(torch.arange(self.pos_embed_d.weight.shape[0], device=x1.device))
        p2 = self.pos_embed_t(torch.arange(self.pos_embed_t.weight.shape[0], device=x2.device))
        self.pos_embed = torch.cat((p1,p2),dim=0)
        x1 = x1 + self.token_type_embeddings(torch.zeros(x1.shape[0],x1.shape[1], dtype=torch.long, device=x1.device))
        x2 = x2 + self.token_type_embeddings(torch.ones(x2.shape[0],x2.shape[1], dtype=torch.long, device=x2.device))
        x = torch.cat((x1,x2),dim=1)
        feat = self.proj(x)#in:torch.Size([4, 21, 256]) --> out:torch.Size([4, 21, 256])
        o = self.trm(feat, self.query_embed.unsqueeze(0), self.pos_embed.unsqueeze(0))#in:torch.Size([4, 21, 256]) --> out:torch.Size([4, 100, 256])
        logits = self.fc(o)
        return logits