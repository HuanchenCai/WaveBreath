"""Round-13 parameterised waveform heads (task-adapted; not author-code reproductions).
All map (batch, time, features) -> (batch, time, 2) belt A/B at the input frame rate."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoid(length, width, device):
    t = torch.arange(length, device=device).float()[:, None]
    f = torch.exp(torch.arange(0, width, 2, device=device).float() * (-math.log(10000) / width))
    pe = torch.zeros(length, width, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(t * f), torch.cos(t * f)
    return pe


class TCN(nn.Module):
    """Residual dilated conv stack; dilation doubles per block."""
    def __init__(self, inputs, width=32, blocks=4, kernel=5, dropout=.1):
        super().__init__()
        self.input = nn.Conv1d(inputs, width, 1)
        self.blocks = nn.ModuleList()
        for b in range(blocks):
            d = 2 ** b
            self.blocks.append(nn.Sequential(nn.Conv1d(width, width, kernel, padding=d * (kernel - 1) // 2, dilation=d),
                                             nn.GELU(), nn.Dropout(dropout), nn.Conv1d(width, width, 1)))
        self.output = nn.Conv1d(width, 2, 1)

    def forward(self, x):
        h = self.input(x.transpose(1, 2))
        for blk in self.blocks:
            h = F.gelu(h + blk(h))
        return self.output(h).transpose(1, 2)


class BiGRU(nn.Module):
    def __init__(self, inputs, hidden=128, layers=2, dropout=.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Dropout(dropout))
        self.rnn = nn.GRU(hidden, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0)
        self.output = nn.Linear(2 * hidden, 2)

    def forward(self, x):
        return self.output(self.rnn(self.proj(x))[0])


class ConvLSTM(nn.Module):
    """1-D convolutions followed by a bidirectional LSTM (Mitra et al. 2024 style)."""
    def __init__(self, inputs, channels=128, conv_layers=2, hidden=128, layers=2, dropout=.1):
        super().__init__()
        convs, c = [], inputs
        for _ in range(conv_layers):
            convs += [nn.Conv1d(c, channels, 5, padding=2), nn.GELU(), nn.Dropout(dropout)]
            c = channels
        self.conv = nn.Sequential(*convs)
        self.rnn = nn.LSTM(channels, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0)
        self.output = nn.Linear(2 * hidden, 2)

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.output(self.rnn(h)[0])


class Transformer(nn.Module):
    def __init__(self, inputs, width=128, layers=2, heads=4, dropout=.1):
        super().__init__()
        self.proj = nn.Linear(inputs, width)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(width, heads, 2 * width, dropout, 'gelu',
                                                                    batch_first=True, norm_first=True),
                                         layers, enable_nested_tensor=False)
        self.output = nn.Linear(width, 2)

    def forward(self, x):
        z = self.proj(x)
        return self.output(self.enc(z + sinusoid(z.shape[1], z.shape[2], z.device)))


class UNet(nn.Module):
    """1-D U-Net over the feature time axis; lengths padded to a multiple of 2**depth."""
    def __init__(self, inputs, base=32, depth=3, dropout=.1):
        super().__init__()
        self.depth = depth
        self.input = nn.Conv1d(inputs, base, 1)
        self.down, self.up, self.dec = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = base
        chans = []
        for _ in range(depth):
            self.down.append(self._block(c, c * 2, dropout)); chans.append(c); c *= 2
        self.mid = self._block(c, c, dropout)
        for skip in reversed(chans):
            self.up.append(nn.ConvTranspose1d(c, skip, 2, stride=2))
            self.dec.append(self._block(2 * skip, skip, dropout)); c = skip
        self.enc0 = self._block(base, base, dropout)
        self.output = nn.Conv1d(base, 2, 1)

    @staticmethod
    def _block(i, o, dropout):
        return nn.Sequential(nn.Conv1d(i, o, 5, padding=2), nn.GroupNorm(4, o), nn.GELU(), nn.Dropout(dropout),
                             nn.Conv1d(o, o, 3, padding=1), nn.GELU())

    def forward(self, x):
        n = x.shape[1]; m = 2 ** self.depth; pad = (-n) % m
        h = self.input(F.pad(x, (0, 0, 0, pad)).transpose(1, 2))
        h = self.enc0(h); skips = []
        for blk in self.down:
            skips.append(h); h = blk(F.max_pool1d(h, 2))
        h = self.mid(h)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            h = dec(torch.cat([up(h), s], dim=1))
        return self.output(h).transpose(1, 2)[:, :n]


FAMILIES = {'tcn': TCN, 'bigru': BiGRU, 'convlstm': ConvLSTM, 'transformer': Transformer, 'unet': UNet}


def build(family, inputs, arch):
    return FAMILIES[family](inputs, **arch)
