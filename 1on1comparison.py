import datetime
import time

import winsound

from ResNet import ResNet
from LSTM import LSTM

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

chord = [523, 659, 784]

start_time = time.time()

EPOCHS = 100
TEST_SAMPLES = 4
HALF_BATCH_SIZE = 150
FAKE_FACTOR = 5  # max 5
LR = 0.001

SEQ_LEN = 60
HIDDEN_SIZE = 5
NUM_BLOCKS = 1  # at least 2 for ResNet
DROPOUT = 0.
REQUIRE_UNSQUEEZE = True  # True for LSTM, GRU; False for ResNet, TCN
RUNS_PER_COMPARISON = 10

print("hyperparams:")
print("epochs", EPOCHS)
print("test samples", TEST_SAMPLES)
print("half batch size", HALF_BATCH_SIZE)
print("fake factor", FAKE_FACTOR)
print("learning rate", LR)
print("sequence length", SEQ_LEN)
print("hidden size", HIDDEN_SIZE)
print("num blocks", NUM_BLOCKS)
print("dropout", DROPOUT)
print("require unsqueeze", REQUIRE_UNSQUEEZE)
print("runs per comparison", RUNS_PER_COMPARISON)

save_path = 'img/'

# --- Real data loading ---
real_data = torch.tensor(np.load('real_data.npy'), dtype=torch.float32)
len_real_train_data = len(real_data) - (TEST_SAMPLES + 1) * SEQ_LEN
len_real_train_data -= len_real_train_data % HALF_BATCH_SIZE  # so we can neatly split up into batches
mu, sigma = torch.mean(real_data), torch.std(real_data)
real_data = (real_data - mu) / sigma  # normalize to mean 0, std 1

shuffled_indices = np.arange(len_real_train_data, dtype=int)
np.random.shuffle(shuffled_indices)
shuffled_indices = shuffled_indices.reshape(-1, HALF_BATCH_SIZE)

print(f"(real data) training size {len_real_train_data + SEQ_LEN} test size "
      f"{len(real_data) - len_real_train_data - SEQ_LEN}")


def comparison(fake_data, title=""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fake_data = (fake_data - mu) / sigma  # normalize the fake data to match real

    # # --- Preliminary check ---
    # plt.hist(real_data, bins=50, density=True, alpha=0.3, color='r', label='Historical')
    # plt.hist(fake_data[0], bins=50, density=True, alpha=0.3, color='b', label='Fake Data 1')
    # plt.hist(fake_data[1], bins=50, density=True, alpha=0.3, color='g', label='Fake Data 2')
    # plt.hist(fake_data[2], bins=50, density=True, alpha=0.3, color='y', label='Fake Data 3')
    # plt.title(title + " distribution comparison")
    # plt.legend()
    # plt.show()

    train_losses, val_losses = (np.zeros((RUNS_PER_COMPARISON, EPOCHS)),
                                np.zeros((RUNS_PER_COMPARISON, EPOCHS)))
    confidences = np.zeros((RUNS_PER_COMPARISON, EPOCHS))
    for run in range(RUNS_PER_COMPARISON):
        # net = ResNet(input_size=SEQ_LEN, hidden_size=HIDDEN_SIZE, num_blocks=NUM_BLOCKS, dropout=DROPOUT).to(device)
        net = LSTM(hidden_size=HIDDEN_SIZE, num_layers=NUM_BLOCKS, dropout=DROPOUT).to(device)
        opti = torch.optim.Adam(net.parameters(), lr=LR)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=FAKE_FACTOR * torch.ones((1,)))

        if run == 0:
            print("Num params:", sum(p.numel() for p in net.parameters() if p.requires_grad))

        t = tqdm(range(EPOCHS))
        for epoch in t:
            net.train()
            aggr_loss = 0
            for idx in shuffled_indices:
                # Real data
                inputs = torch.stack([real_data[i:i + SEQ_LEN] for i in idx]).to(device)
                labels = torch.ones((HALF_BATCH_SIZE, 1), device=device)

                # Fake data
                for idx_data_set in range(5):
                    if title == "GAN":
                        idx_data_set += (run * EPOCHS + epoch) * 5  # we generated a lot of random data for GAN
                        idx_data_set = idx_data_set % fake_data.shape[0]  # I thought it was 1000 but it's 100
                    random_indices = np.random.randint(0, len(fake_data[idx_data_set]) - SEQ_LEN, size=HALF_BATCH_SIZE)
                    fake_seq = torch.stack([fake_data[idx_data_set][i:i + SEQ_LEN] for i in random_indices]).to(device)
                    fake_labels = torch.zeros((HALF_BATCH_SIZE, 1), device=device)

                    # Combine
                    inputs = torch.cat([inputs, fake_seq], dim=0)
                    labels = torch.cat([labels, fake_labels], dim=0)

                # Forward
                if REQUIRE_UNSQUEEZE:
                    inputs = inputs.unsqueeze(-1)
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # Backprop
                opti.zero_grad()
                loss.backward()
                opti.step()

                aggr_loss += loss.item()

            net.eval()
            with torch.no_grad():
                # NB the market may have changed significantly in this period
                test_data = real_data[len_real_train_data + SEQ_LEN:]
                test_input = torch.stack([test_data[i:i + SEQ_LEN] for i in
                                          range(0, len(test_data) - SEQ_LEN)]).to(device)
                random_indices = np.random.randint(0, len(fake_data[-1]) - SEQ_LEN, size=test_input.shape[0])
                fake_seq = torch.stack([fake_data[-1][i:i + SEQ_LEN] for i in random_indices]).to(device)

                if REQUIRE_UNSQUEEZE:
                    test_input = test_input.unsqueeze(-1)
                    fake_seq = fake_seq.unsqueeze(-1)

                real_out = net(test_input).numpy()
                fake_out = net(fake_seq).numpy()

                # val_l = FAKE_FACTOR * np.log(1 + np.exp(-real_out))
                val_l = np.log(1 + np.exp(-real_out))
                val_l = np.concatenate((val_l, np.log(1 + np.exp(fake_out))))

            t.set_description(f"Train loss: {aggr_loss / len(shuffled_indices):.4f}, "
                              f"Val loss: {np.mean(val_l):.4f}, "
                              f"Avg conf: {np.mean(1 / (1 + np.exp(-real_out))):.4f}/{np.mean(1 - 1 / (1 + np.exp(-fake_out))):.4f}")

            train_losses[run, epoch] = aggr_loss / len(shuffled_indices)
            val_losses[run, epoch] = np.mean(val_l)
            confidences[run, epoch] = 0.5 * (
                    np.mean(1 / (1 + np.exp(-real_out))) + np.mean(1 - 1 / (1 + np.exp(-fake_out))))

    plt.figure(figsize=(8, 6))
    plt.plot(np.mean(train_losses, axis=0), label='Train Loss')
    plt.plot(np.mean(val_losses, axis=0), label='Val Loss')
    plt.fill_between(range(EPOCHS),
                     np.mean(train_losses, axis=0) - np.std(train_losses, axis=0, ddof=1) / np.sqrt(
                         RUNS_PER_COMPARISON),
                     np.mean(train_losses, axis=0) + np.std(train_losses, axis=0, ddof=1) / np.sqrt(
                         RUNS_PER_COMPARISON),
                     alpha=0.2)
    plt.fill_between(range(EPOCHS),
                     np.mean(val_losses, axis=0) - np.std(val_losses, axis=0, ddof=1) / np.sqrt(RUNS_PER_COMPARISON),
                     np.mean(val_losses, axis=0) + np.std(val_losses, axis=0, ddof=1) / np.sqrt(RUNS_PER_COMPARISON),
                     alpha=0.2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.title(title + " Loss Comparison")
    #           # + f" ({NUM_BLOCKS}x{HIDDEN_SIZE} LSTM)")
    plt.legend()
    plt.savefig(save_path + title.replace(" ", "_") + f"_{SEQ_LEN}_loss_comparison.png", dpi=300)
    plt.show()

    return train_losses, val_losses, confidences


# print("--- Noise ---")
# comparison(torch.randn((5, len(real_data)), dtype=torch.float32)*max(abs(real_data)/3), title="Noise")

# print(np.load("gan_data.npy").shape)

print("\n--- GBM Comparison ---")
gbm_trl, gbm_vl, gbm_cf = comparison(torch.tensor(np.load('gbm_data.npy'), dtype=torch.float32), title="GBM")

print("\n--- GARCH(1,1) Comparison ---")
gar_trl, gar_vl, gar_cf = comparison(torch.tensor(np.load('garch_data.npy'), dtype=torch.float32), title="GARCH(1,1)")

print("\n--- GAN Comparison ---")
gan_trl, gan_vl, gan_cf = comparison(torch.tensor(np.load('gan_data.npy').transpose(), dtype=torch.float32),
                                     title="GAN")

print("\n--- Overall Loss Comparison ---")
print("GAN final train loss:", gan_trl.mean(axis=0)[-1], "val loss:", gan_vl.mean(axis=0)[-1])
print("GBM final train loss:", gbm_trl.mean(axis=0)[-1], "val loss:", gbm_vl.mean(axis=0)[-1])
print("GARCH final train loss:", gar_trl.mean(axis=0)[-1], "val loss:", gar_vl.mean(axis=0)[-1])

plt.figure(figsize=(8, 6))
plt.plot(gan_trl.mean(axis=0), label='GAN Train Loss', color='b')
plt.plot(gbm_trl.mean(axis=0), label='GBM Train Loss', color='r')
plt.plot(gar_trl.mean(axis=0), label='GARCH Train Loss', color='g')
plt.plot(gan_vl.mean(axis=0), label='GAN Val Loss', color='b', linestyle='--')
plt.plot(gbm_vl.mean(axis=0), label='GBM Val Loss', color='r', linestyle='--')
plt.plot(gar_vl.mean(axis=0), label='GARCH Val Loss', color='g', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.title("Overall Loss Comparison")
plt.legend()
plt.savefig(save_path + f"{SEQ_LEN}_overall_loss_comparison.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(gan_trl.mean(axis=0), label='GAN Train Loss', color='b')
plt.fill_between(range(EPOCHS),
                 gan_trl.mean(axis=0) - gan_trl.std(axis=0, ddof=1) / np.sqrt(RUNS_PER_COMPARISON),
                 gan_trl.mean(axis=0) + gan_trl.std(axis=0, ddof=1) / np.sqrt(RUNS_PER_COMPARISON),
                 alpha=0.2, color='b')
plt.plot(gbm_trl.mean(axis=0), label='GBM Train Loss', color='r')
plt.fill_between(range(EPOCHS),
                 gbm_trl.mean(axis=0) - gbm_trl.std(axis=0, ddof=1) / np.sqrt(RUNS_PER_COMPARISON),
                 gbm_trl.mean(axis=0) + gbm_trl.std(axis=0, ddof=1) / np.sqrt(RUNS_PER_COMPARISON),
                 alpha=0.2, color='r')
plt.plot(gar_trl.mean(axis=0), label='GARCH Train Loss', color='g')
plt.fill_between(range(EPOCHS),
                 gar_trl.mean(axis=0) - gar_trl.std(axis=0, ddof=1) / np.sqrt(RUNS_PER_COMPARISON),
                 gar_trl.mean(axis=0) + gar_trl.std(axis=0, ddof=1) / np.sqrt(RUNS_PER_COMPARISON),
                 alpha=0.2, color='g')
plt.plot(gan_vl.mean(axis=0), label='GAN Val Loss', color='b', linestyle='--')
plt.fill_between(range(EPOCHS),
                 gan_vl.mean(axis=0) - gan_vl.std(axis=0, ddof=1) / np.sqrt(RUNS_PER_COMPARISON),
                 gan_vl.mean(axis=0) + gan_vl.std(axis=0, ddof=1) / np.sqrt(RUNS_PER_COMPARISON),
                 alpha=0.2, color='b')
plt.plot(gbm_vl.mean(axis=0), label='GBM Val Loss', color='r', linestyle='--')
plt.fill_between(range(EPOCHS),
                 gbm_vl.mean(axis=0) - gbm_vl.std(axis=0, ddof=1) / np.sqrt(RUNS_PER_COMPARISON),
                 gbm_vl.mean(axis=0) + gbm_vl.std(axis=0, ddof=1) / np.sqrt(RUNS_PER_COMPARISON),
                 alpha=0.2, color='r')
plt.plot(gar_vl.mean(axis=0), label='GARCH Val Loss', color='g', linestyle='--')
plt.fill_between(range(EPOCHS),
                 gar_vl.mean(axis=0) - gar_vl.std(axis=0, ddof=1) / np.sqrt(RUNS_PER_COMPARISON),
                 gar_vl.mean(axis=0) + gar_vl.std(axis=0, ddof=1) / np.sqrt(RUNS_PER_COMPARISON),
                 alpha=0.2, color='g')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.title("Overall Loss Comparison")
plt.legend()
plt.savefig(save_path + f"{SEQ_LEN}_overall_loss_comparison_with_bands.png", dpi=300)
plt.show()

ax1 = plt.gca()
ax2 = ax1.twinx()

colors = {"GAN": "b", "GBM": "r", "GARCH": "g"}

# Loss (left axis) with uncertainty bands
for label, trl, c in [("GAN", gan_trl, colors["GAN"]),
                      ("GBM", gbm_trl, colors["GBM"]),
                      ("GARCH", gar_trl, colors["GARCH"])]:
    mean = trl.mean(axis=0)
    err = trl.std(axis=0, ddof=1) / np.sqrt(RUNS_PER_COMPARISON)
    ax1.plot(mean, label=f"{label} train loss", color=c)
    ax1.fill_between(range(EPOCHS), mean - err, mean + err, color=c, alpha=0.2)

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss", color="k")
ax1.tick_params(axis="y", labelcolor="k")

# Accuracy (right axis) with uncertainty bands
for label, cf, c in [("GAN", gan_cf, colors["GAN"]),
                     ("GBM", gbm_cf, colors["GBM"]),
                     ("GARCH", gar_cf, colors["GARCH"])]:
    mean = cf.mean(axis=0)
    err = cf.std(axis=0, ddof=1) / np.sqrt(RUNS_PER_COMPARISON)
    ax2.plot(mean, label=f"{label} val. acc.", color=c, linestyle="--")
    ax2.fill_between(range(EPOCHS), mean - err, mean + err, color=c, alpha=0.2)

ax2.set_ylabel("Accuracy", color="k")
ax2.tick_params(axis="y", labelcolor="k")
ax2.set_ylim(0, 1)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2)

plt.tight_layout()
plt.savefig(save_path + f"{SEQ_LEN}_loss_accuracy_comparison.png", dpi=300)
plt.show()

print("Took", datetime.timedelta(seconds=time.time() - start_time))

for _ in range(3):
    for f in chord:
        winsound.Beep(f, 60)
