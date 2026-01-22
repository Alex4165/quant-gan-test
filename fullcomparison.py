from ResNet import ResNet
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

EPOCHS = 100
TEST_SAMPLES = 5
QUARTER_BATCH_SIZE = 10
FAKE_FACTOR = 5  # max 5
LR = 0.01

HIDDEN_SIZE = 256
NUM_BLOCKS = 5
SEQ_LEN = 50
DROPOUT = 0.0

# --- Real data loading ---
real_data = torch.tensor(np.load('real_data.npy'), dtype=torch.float32)
len_real_train_data = len(real_data) - (TEST_SAMPLES + 1) * SEQ_LEN
len_real_train_data -= len_real_train_data % QUARTER_BATCH_SIZE  # so we can neatly split up into batches

shuffled_indices = np.arange(len_real_train_data, dtype=int)
np.random.shuffle(shuffled_indices)
shuffled_indices = shuffled_indices.reshape(-1, QUARTER_BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = ResNet(input_size=SEQ_LEN, hidden_size=HIDDEN_SIZE, num_blocks=NUM_BLOCKS, dropout=DROPOUT, output_size=4).to(device)
opti = torch.optim.Adam(net.parameters(), lr=LR)
weights = torch.tensor([FAKE_FACTOR, 1, 1, 1], device=device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)

print(f"(real data) training size {len_real_train_data} test size "
      f"{(len(real_data) - len_real_train_data) - (len(real_data) - len_real_train_data) % SEQ_LEN}")

# --- Fake data loading ---
gbm, garch, gan = torch.tensor(np.load('gbm_data.npy'), dtype=torch.float32), \
                  torch.tensor(np.load('garch_data.npy'), dtype=torch.float32), \
                  torch.tensor(np.load('gan_data.npy').transpose(), dtype=torch.float32)

sig = torch.nn.Softmax(dim=1)

t = tqdm(range(EPOCHS), desc="Training Progress")
for epoch in t:
    net.train()
    aggr_loss = np.zeros(4)
    for idx in shuffled_indices:
        # Real data
        inputs = torch.stack([real_data[i:i + SEQ_LEN] for i in idx]).to(device)
        labels = torch.zeros((QUARTER_BATCH_SIZE, 4), device=device)
        labels[:, 0] = 1

        # Fake data
        for idx_data_set in range(FAKE_FACTOR):
            for k, _data_set in enumerate([gbm, garch, gan]):
                random_indices = np.random.randint(0, len(_data_set[idx_data_set]) - SEQ_LEN, size=QUARTER_BATCH_SIZE)
                fake_seq = torch.stack([_data_set[idx_data_set][i:i + SEQ_LEN] for i in random_indices]).to(device)
                fake_labels = torch.zeros((QUARTER_BATCH_SIZE, 4), device=device)
                fake_labels[:, k+1] = 1

                # Combine
                inputs = torch.cat([inputs, fake_seq], dim=0)
                labels = torch.cat([labels, fake_labels], dim=0)

        # Forward
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        outputs = sig(outputs)
        score = np.array([torch.mean(outputs[:QUARTER_BATCH_SIZE, 0]).item(),
                         torch.mean(outputs[QUARTER_BATCH_SIZE:(FAKE_FACTOR+1)*QUARTER_BATCH_SIZE, 1]).item(),
                         torch.mean(outputs[(FAKE_FACTOR+1)*QUARTER_BATCH_SIZE:(2*FAKE_FACTOR+1)*QUARTER_BATCH_SIZE, 2]).item(),
                         torch.mean(outputs[(2*FAKE_FACTOR+1)*QUARTER_BATCH_SIZE:, 3]).item()])
        aggr_loss += score

        # Backprop
        opti.zero_grad()
        loss.backward()
        opti.step()

    net.eval()
    with torch.no_grad():
        test_data = real_data[len_real_train_data:]  # NB the market may have changed significantly in this period
        test_data = torch.stack([test_data[i:i + SEQ_LEN] for i in
                                 range(0, len(test_data) - (len(test_data) % SEQ_LEN), SEQ_LEN)]).to(device)
        ones = sig(net(test_data)).cpu().numpy()[:, 0]
    net.train()

    t.set_description(f"Confidence (real gbm garch gan): {aggr_loss / len(shuffled_indices)}, "
                      f"Validation confidence real: {np.mean(ones):.4f}")



