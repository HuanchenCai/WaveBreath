"""Small, task-adapted waveform model families; not author-code reproductions."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def positions(length, width, device):
    time = torch.arange(length, device=device).float()[:, None]
    freq = torch.exp(torch.arange(0, width, 2, device=device).float() * (-math.log(10000) / width))
    result = torch.zeros(length, width, device=device)
    result[:, 0::2], result[:, 1::2] = torch.sin(time * freq), torch.cos(time * freq)
    return result


class ConvBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.body = nn.Sequential(nn.Conv1d(inputs, outputs, 5, padding=2), nn.GroupNorm(4, outputs),
                                  nn.GELU(), nn.Conv1d(outputs, outputs, 3, padding=1), nn.GELU())
        self.skip = nn.Conv1d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, x):
        return self.body(x) + self.skip(x)


class WindowAttention(nn.Module):
    def __init__(self, width, window=8, shift=0):
        super().__init__()
        self.window, self.shift = window, shift
        self.norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, 4, dropout=.1, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(),
                                nn.Dropout(.1), nn.Linear(width * 2, width))

    def forward(self, h):
        x = h.transpose(1, 2)
        b, length, width = x.shape
        left = self.shift
        right = (-(length + left)) % self.window
        padded = F.pad(self.norm(x), (0, 0, left, right))
        valid = F.pad(torch.ones(b, length, device=x.device), (left, right))
        blocks = padded.reshape(-1, self.window, width)
        key_mask = ~valid.reshape(-1, self.window).bool()
        z, _ = self.attn(blocks, blocks, blocks, key_padding_mask=key_mask, need_weights=False)
        z = z.reshape(b, -1, width)[:, left:left + length]
        x = x + z
        return (x + self.ff(x)).transpose(1, 2)


class TemporalTransformer(nn.Module):
    def __init__(self, inputs, width=64):
        super().__init__()
        self.input = nn.Linear(inputs, width)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            width, 4, width * 2, dropout=.1, activation='gelu', batch_first=True, norm_first=True),
            num_layers=2, enable_nested_tensor=False)
        self.output = nn.Linear(width, 2)

    def forward(self, x):
        z = self.input(x)
        return self.output(self.encoder(z + positions(z.shape[1], z.shape[2], z.device)))


class SkipFusion(nn.Module):
    """Bidirectional recurrent fusion of skip and decoder features (NAB-inspired)."""
    def __init__(self, inputs, width):
        super().__init__()
        self.project = nn.Conv1d(inputs, width, 1)
        self.rnn = nn.LSTM(width, width // 2, batch_first=True, bidirectional=True)
        self.gate = nn.Conv1d(width, width, 1)

    def forward(self, x):
        z = self.project(x)
        r, _ = self.rnn(z.transpose(1, 2))
        return z + torch.sigmoid(self.gate(z)) * r.transpose(1, 2)


class WaveUNet(nn.Module):
    def __init__(self, inputs, attention=False, nested=False, width=32):
        super().__init__()
        self.stem = nn.Conv1d(inputs, width, 1)
        self.e0, self.e1, self.e2 = ConvBlock(width, width), ConvBlock(width, width * 2), ConvBlock(width * 2, width * 4)
        self.bridge = ConvBlock(width * 4, width * 4)
        self.attention = nn.Sequential(WindowAttention(width * 4), WindowAttention(width * 4, shift=4)) if attention else nn.Identity()
        fusion = SkipFusion if nested else ConvBlock
        self.d2 = fusion(width * 8, width * 2)
        self.d1 = fusion(width * 4, width)
        self.d0 = fusion(width * 2, width)
        self.output = nn.Conv1d(width, 2, 1)

    def forward(self, x):
        e0 = self.e0(self.stem(x.transpose(1, 2)))
        e1 = self.e1(F.avg_pool1d(e0, 2))
        e2 = self.e2(F.avg_pool1d(e1, 2))
        z = self.attention(self.bridge(F.avg_pool1d(e2, 2)))
        for decoder, skip in ((self.d2, e2), (self.d1, e1), (self.d0, e0)):
            z = decoder(torch.cat((F.interpolate(z, size=skip.shape[-1], mode='linear', align_corners=False), skip), dim=1))
        return self.output(z).transpose(1, 2)


class SmallEncoder(nn.Module):
    def __init__(self, inputs, latent=16):
        super().__init__()
        self.network = nn.Sequential(nn.Conv1d(inputs, 32, 1), nn.GELU(),
                                     nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
                                     nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.GELU(),
                                     nn.Conv1d(64, latent, 3, padding=1))

    def forward(self, x):
        return self.network(x.transpose(1, 2))


class SmallDecoder(nn.Module):
    def __init__(self, outputs, latent=16):
        super().__init__()
        self.input = nn.Conv1d(latent, 64, 3, padding=1)
        self.middle = nn.Conv1d(64, 32, 5, padding=2)
        self.output = nn.Conv1d(32, outputs, 5, padding=2)

    def forward(self, z):
        z = F.gelu(self.input(z))
        z = F.gelu(self.middle(F.interpolate(z, scale_factor=2, mode='linear', align_corners=False)))
        return self.output(F.interpolate(z, scale_factor=2, mode='linear', align_corners=False)).transpose(1, 2)


def waveform_loss(pred, target, mask):
    weights = mask[:, :, None]
    mse = ((pred - target).square() * weights).sum() / (2 * mask.sum().clamp_min(1))
    pair = (mask[:, 1:] * mask[:, :-1])[:, :, None]
    deriv = (((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])).square() * pair).sum() / (2 * pair.sum().clamp_min(1))
    return mse + .2 * deriv


class AlignedLatents(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.voice = SmallEncoder(inputs)
        self.private_voice = SmallEncoder(inputs)
        self.belt = SmallEncoder(2, latent=4)
        self.belt_decoder = SmallDecoder(2, latent=4)
        self.voice_decoder = SmallDecoder(inputs, latent=32)
        self.bridge = nn.Sequential(nn.Conv1d(16, 32, 3, padding=1), nn.GELU(), nn.Conv1d(32, 4, 3, padding=1))
        self.reverse_bridge = nn.Sequential(nn.Conv1d(4, 32, 3, padding=1), nn.GELU(), nn.Conv1d(32, 16, 3, padding=1))

    def forward(self, x):
        return self.belt_decoder(self.bridge(self.voice(x)))

    def loss(self, x, y, mask):
        voice, belt, private = self.voice(x), self.belt(y), self.private_voice(x)
        mapped = self.bridge(voice)
        pred, oracle = self.belt_decoder(mapped), self.belt_decoder(belt)
        weight = F.avg_pool1d(mask[:, None], 4)
        alignment = ((mapped - belt).square() * weight).sum() / (weight.sum().clamp_min(1) * 4)
        reverse_error = ((self.reverse_bridge(belt) - voice.detach()).square() * weight).sum() / (weight.sum().clamp_min(1) * 16)
        voice_rec = self.voice_decoder(torch.cat((voice, private), dim=1))
        voice_error = ((voice_rec - x).square() * mask[:, :, None]).sum() / (mask.sum().clamp_min(1) * x.shape[-1])
        # Contrastive pairing is auxiliary; waveform losses prevent a trivial collapsed code.
        a = F.normalize((mapped * weight).sum(dim=2), dim=1)
        b = F.normalize((belt * weight).sum(dim=2), dim=1)
        logits = a @ b.T / .15
        labels = torch.arange(len(x), device=x.device)
        contrast = .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        return waveform_loss(pred, y, mask) + .5 * waveform_loss(oracle, y, mask) + .1 * alignment + .05 * reverse_error + .025 * voice_error + .01 * contrast


class ConditionalField(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv1d(8, 64, 1)
        self.time = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 64))
        self.blocks = nn.ModuleList([nn.Sequential(nn.GroupNorm(4, 64), nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=d, dilation=d), nn.GELU(), nn.Conv1d(64, 64, 1)) for d in (1, 2, 4)])
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.output = nn.Conv1d(64, 4, 1)

    def forward(self, z, condition, time):
        freq = torch.exp(torch.arange(8, device=z.device).float() * (-math.log(1000) / 8))
        phase = time[:, None] * freq[None] * 100
        h = self.input(torch.cat((z, condition), dim=1)) + self.time(torch.cat((phase.sin(), phase.cos()), dim=1))[:, :, None]
        for block in self.blocks:
            h = h + block(h)
        q = h.transpose(1, 2)
        attended, _ = self.attn(q, q, q, need_weights=False)
        return self.output(h + attended.transpose(1, 2))


class LatentGenerator(nn.Module):
    """Conditional latent flow or DDIM, with fixed-seed ensemble evaluation."""
    def __init__(self, inputs, kind):
        super().__init__()
        self.ae = AlignedLatents(inputs)
        self.field = ConditionalField()
        self.kind = kind
        self.register_buffer('latent_mean', torch.zeros(1, 4, 1))
        self.register_buffer('latent_std', torch.ones(1, 4, 1))
        angles = torch.linspace(0, 1, 101)
        alpha = torch.cos((angles + .008) / 1.008 * math.pi / 2).square()
        self.register_buffer('alpha', (alpha / alpha[0]).clamp_min(1e-4))

    def freeze_ae(self):
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad_(False)

    def loss(self, x, y, mask):
        self.ae.eval()
        with torch.no_grad():
            target = (self.ae.belt(y) - self.latent_mean) / self.latent_std
            condition = (self.ae.bridge(self.ae.voice(x)) - self.latent_mean) / self.latent_std
        noise = torch.randn_like(target)
        if self.kind == 'flow':
            t = torch.rand(len(x), device=x.device)
            mixed = (1 - t[:, None, None]) * noise + t[:, None, None] * target
            output = self.field(mixed, condition, t)
            objective = target - noise
            estimate = mixed + (1 - t[:, None, None]) * output
        else:
            index = torch.randint(1, 101, (len(x),), device=x.device)
            t = index.float() / 100
            alpha = self.alpha[index, None, None]
            mixed = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
            output = self.field(mixed, condition, t)
            objective = noise
            estimate = ((mixed - (1 - alpha).sqrt() * output) / alpha.sqrt()).clamp(-5, 5)
        weight = F.avg_pool1d(mask[:, None], 4)
        latent_loss = ((output - objective).square() * weight).sum() / (4 * weight.sum().clamp_min(1))
        pred = self.ae.belt_decoder(estimate * self.latent_std + self.latent_mean)
        return latent_loss + .05 * waveform_loss(pred, y, mask)

    @torch.no_grad()
    def forward(self, x):
        self.ae.eval()
        condition = (self.ae.bridge(self.ae.voice(x)) - self.latent_mean) / self.latent_std
        predictions = []
        for seed in (31, 137, 503, 1009):
            generator = torch.Generator(device=x.device).manual_seed(seed)
            # Same seed distribution for every window, invariant to evaluation batching.
            z = torch.randn((1, 4, condition.shape[-1]), device=x.device, generator=generator).expand(len(x), -1, -1).clone()
            if self.kind == 'flow':
                for step in range(10):
                    time = torch.full((len(x),), step / 10, device=x.device)
                    z = z + self.field(z, condition, time) / 10
            else:
                for step in range(100, 0, -10):
                    time = torch.full((len(x),), step / 100, device=x.device)
                    epsilon = self.field(z, condition, time)
                    a, previous = self.alpha[step], self.alpha[max(0, step - 10)]
                    clean = ((z - (1 - a).sqrt() * epsilon) / a.sqrt()).clamp(-5, 5)
                    z = previous.sqrt() * clean + (1 - previous).sqrt() * epsilon
            predictions.append(self.ae.belt_decoder(z * self.latent_std + self.latent_mean))
        return torch.stack(predictions).mean(dim=0)


def make_model(name, inputs):
    if name == 'transformer':
        return TemporalTransformer(inputs)
    if name == 'wave_unet':
        return WaveUNet(inputs)
    if name == 'swin_unet':
        return WaveUNet(inputs, attention=True)
    if name == 'nab_inspired':
        return WaveUNet(inputs, attention=True, nested=True)
    if name == 'aligned_latent':
        return AlignedLatents(inputs)
    if name in ('flow', 'diffusion'):
        return LatentGenerator(inputs, name)
    if name == 'tcn':
        from whisper_feature_tcn_cv import FeatureTCN
        return FeatureTCN(inputs)
    raise ValueError(name)
