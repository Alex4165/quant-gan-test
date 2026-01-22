import numpy as np
import torch
import torch.nn as nn

import tqdm


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hidden, kernel_size, dilation, rank=None, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_hidden, kernel_size, dilation=dilation, padding='same')
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(n_hidden, n_outputs, kernel_size, dilation=dilation, padding='same')
        self.relu2 = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

        self.net = nn.Sequential(self.dropout, self.conv1, self.relu1, self.dropout, self.conv2, self.relu2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        # Adding quadratic terms
        self.rank = rank
        if rank is not None:
            self.U, self.V, self.W = nn.Linear(n_inputs, rank, bias=False), nn.Linear(n_inputs, rank, bias=False), nn.Linear(rank, n_outputs)

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """x: (batch_size, n_inputs, seq_len)"""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        if self.rank is not None:
            quad = self.U(x.transpose(1, 2)) * self.V(x.transpose(1, 2))
            return out + res + self.W(quad).transpose(1, 2)
        return out + res


class TCN(nn.Module):
    def __init__(self, hidden_size, dilation, kernel, depth, copies, dropout=0.0):
        super().__init__()
        layers = [TemporalBlock(n_inputs=1, n_outputs=hidden_size, n_hidden=hidden_size, kernel_size=1, dilation=1, rank=None)]
        for _ in range(copies):
            for i in range(depth):
                layers.append(TemporalBlock(hidden_size, hidden_size, hidden_size, kernel, dilation**i, rank=None, dropout=dropout))
        self.conv = nn.Conv1d(hidden_size, 1, 1)
        self.net = nn.Sequential(*layers)
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """x: (batch_size, seq_len, 1)"""
        x = x.transpose(1, 2)
        out = self.net(x)
        out = self.conv(out)
        out = out[:, :, -1]
        return out


if __name__ == "__main__":
    epochs = 10
    half_batch_size = 20
    lr = 0.01

    hidden_size = 80
    depth = 2
    dilation = 2
    kernel = 2
    copies = 1
    dropout = 0.5

    seq_len = 1 + 2 * copies * (sum([(kernel-1)*dilation**i for i in range(depth)]))
    print("receptive field length", seq_len)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using", device)
    raw_data = torch.tensor(np.load('real_data.npy'), dtype=torch.float32).to(device)
    raw_data /= max(abs(raw_data))  # normalize to [-1, 1]
    mu, sigma = float(torch.mean(raw_data)), float(torch.std(raw_data))

    rnn = TCN(hidden_size, dilation, kernel, depth, copies, dropout).to(device)
    opti = torch.optim.Adam(rnn.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print("Num params:", sum(p.numel() for p in rnn.parameters() if p.requires_grad))

    data = raw_data
    shuffled_indices = np.arange(len(raw_data) - seq_len - ((len(raw_data)-seq_len) % half_batch_size), dtype=int)
    np.random.shuffle(shuffled_indices)
    shuffled_indices = shuffled_indices.reshape(-1, half_batch_size)

    t = tqdm.tqdm(range(epochs))
    for _ in t:
        l = 0
        for idx in shuffled_indices:
            # Real data
            real_seq = torch.stack([data[i:i + seq_len] for i in idx]).unsqueeze(-1).to(device)
            real_labels = torch.ones((half_batch_size, 1), device=device)

            # # Real 'fake' data (shouldn't converge)
            # idx = torch.randint(0, len(data) - seq_len, (half_batch_size,))
            # fake_seq = torch.stack([data[i:i+seq_len] for i in idx]).unsqueeze(-1).to(device)
            # fake_labels = torch.zeros((half_batch_size, 1), device=device)

            # # Uniform fake data
            # fake_seq = torch.randn((half_batch_size, seq_len, 1), device=device)
            # fake_labels = torch.zeros((half_batch_size, 1), device=device)

            # Black-Scholes i.e. Normal fake data
            fake_seq = torch.normal(mu, sigma, size=(half_batch_size, seq_len, 1), device=device)
            fake_labels = torch.zeros((half_batch_size, 1), device=device)

            # Combine
            inputs = torch.cat([real_seq, fake_seq], dim=0)
            labels = torch.cat([real_labels, fake_labels], dim=0)

            # Forward
            outputs = rnn(inputs)
            loss = criterion(outputs, labels)

            # Backprop
            opti.zero_grad()
            loss.backward()
            opti.step()

            l += loss.item()

        t.set_description(f"Loss: {l / len(shuffled_indices):.4f}")













