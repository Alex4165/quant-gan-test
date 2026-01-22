import numpy as np
import torch
import torch.nn as nn

import tqdm


class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        hn, _ = self.lstm(x)
        # hn: (batch_size, seq_len, hidden_size)
        hn = hn[:, -1, :]  # output = hn of last layer
        out = self.fc(hn)
        return out  # (batch_size, 1)


if __name__ == "__main__":
    epochs = 30
    seq_len = 20
    half_batch_size = 50
    lr = 0.005
    dropout = 0.5

    hidden_size = seq_len*2
    num_layers = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw_data = torch.tensor(np.load('real_data.npy'), dtype=torch.float32).to(device)
    raw_data -= torch.mean(raw_data)  # normalize to mean 0
    raw_data /= torch.std(raw_data)  # normalize to std 1
    mu, sigma = float(torch.mean(raw_data)), float(torch.std(raw_data))

    rnn = LSTM(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    opti = torch.optim.Adam(rnn.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print("Num params:", sum(p.numel() for p in rnn.parameters() if p.requires_grad))

    len_train_data = len(raw_data) - 2 * seq_len - ((len(raw_data) - 2 * seq_len) % half_batch_size)
    data = raw_data
    shuffled_indices = np.arange(len_train_data, dtype=int)
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

        t.set_description(f"Loss: {l/len(shuffled_indices):.4f}")

    # Test
    test_seq_real = raw_data[len_train_data+seq_len:].unsqueeze(-1).unsqueeze(0).to(device)
    test_seq_fake = torch.normal(mu, sigma, size=test_seq_real.size(), device=device)
    print("testing on", len(test_seq_real), "data points")
    with torch.no_grad():
        test_outputs = rnn(test_seq_real)
        print("1 =", 1/(1+np.exp(-test_outputs.squeeze().cpu().numpy().mean())))

        test_outputs_fake = rnn(test_seq_fake)
        print("0 =", 1/(1+np.exp(-test_outputs_fake.squeeze().cpu().numpy().mean())))













