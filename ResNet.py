import numpy as np
import torch
import torch.nn as nn

import tqdm


class SkipBlock(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, dropout=0.0, activate_last=True):
        super(SkipBlock, self).__init__()
        self.lin1 = nn.Linear(n_inputs, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_outputs)
        self.downsample = None if n_inputs == n_outputs else nn.Linear(n_inputs, n_outputs)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.activate_last = activate_last

    def forward(self, x):
        # x: (batch_size, n_inputs)
        out = self.lin1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.lin2(out)
        if self.downsample:
            x = self.downsample(x)
        if self.activate_last:
            return self.relu(out) + x
        return out + x


class ResNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_blocks, dropout=0.0, output_size=1):
        super(ResNet, self).__init__()
        layers = [SkipBlock(input_size, hidden_size, hidden_size, dropout=dropout)]
        layers += [SkipBlock(hidden_size, hidden_size, hidden_size, dropout=dropout) for _ in range(num_blocks - 2)]
        layers += [SkipBlock(hidden_size, hidden_size, output_size, activate_last=False)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)  # (batch_size, output_size)


if __name__ == "__main__":
    epochs = 10
    half_batch_size = 15
    lr = 0.005
    num_test_samples = 3

    hidden_size = 80
    num_blocks = 2
    seq_len = 50
    dropout = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using", device)
    data = torch.tensor(np.load('real_data.npy'), dtype=torch.float32).to(device)
    data = (data - torch.mean(data)) / torch.std(data)  # normalize to mean 0, std 1
    mu, sigma = float(torch.mean(data)), float(torch.std(data))

    rnn = ResNet(seq_len, hidden_size, num_blocks, dropout).to(device)
    opti = torch.optim.Adam(rnn.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print("Num params:", sum(p.numel() for p in rnn.parameters() if p.requires_grad))

    len_train_data = len(data) - (num_test_samples + 1) * seq_len
    len_train_data -= len_train_data % half_batch_size  # so we can neatly split up into batches
    shuffled_indices = np.arange(len_train_data, dtype=int)
    np.random.shuffle(shuffled_indices)
    shuffled_indices = shuffled_indices.reshape(-1, half_batch_size)

    t = tqdm.tqdm(range(epochs))
    for _ in t:
        l = 0
        for idx in shuffled_indices:
            # Real data
            real_seq = torch.stack([data[i:i + seq_len] for i in idx]).to(device)
            real_labels = torch.ones((half_batch_size, 1), device=device)

            # # Real 'fake' data (shouldn't converge)
            # idx = torch.randint(0, len(data) - seq_len, (half_batch_size,))
            # fake_seq = torch.stack([data[i:i+seq_len] for i in idx]).to(device)
            # fake_labels = torch.zeros((half_batch_size, 1), device=device)

            # # Uniform fake data
            # fake_seq = torch.randn((half_batch_size, seq_len), device=device)
            # fake_labels = torch.zeros((half_batch_size, 1), device=device)

            # Black-Scholes i.e. Normal fake data
            fake_seq = torch.normal(mu, sigma, size=(half_batch_size, seq_len), device=device)
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

        rnn.eval()
        with torch.no_grad():
            test_data = data[len_train_data:]  # NB the market may have changed significantly in this period
            test_data = torch.stack([test_data[i:i + seq_len] for i in
                                     range(0, len(test_data)-(len(test_data) % seq_len), seq_len)]).to(device)
            if _ == 0:
                print("Testing on", test_data.numel(), "data points")
            ones = 1/(1+np.exp(-rnn(test_data).numpy()))
            fake_seq = torch.normal(mu, sigma, size=(num_test_samples, seq_len), device=device)
            zeros = 1/(1+np.exp(-rnn(fake_seq).numpy()))
        rnn.train()

        t.set_description(f"Train loss: {l / len(shuffled_indices):.4f}, "
                          f"Val conf real: {np.mean(ones):.4f}, fake: {1-np.mean(zeros):.4f}")
