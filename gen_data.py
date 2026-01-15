import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

url = "https://stooq.com/q/d/l/?s=^spx&i=d"
sp500 = pd.read_csv(url)

sp500["Date"] = pd.to_datetime(sp500["Date"])

mask = (sp500["Date"] >= "2009-05-01") & (sp500["Date"] <= "2018-12-31")
sp500 = sp500.loc[mask].reset_index(drop=True)
sp500log = np.log(sp500['Close'] / sp500['Close'].shift(1))[1:].values
sp500log_mean = np.mean(sp500log)
sp500log_stationary = sp500log - sp500log_mean

np.save('real_data.npy', sp500log)

# GARCH(1,1) model
model = arch_model(sp500log_stationary*100, mean='Zero', vol='GARCH', p=1, q=1)
res = model.fit(disp='off')

# Quick test
sim = model.simulate(res.params, nobs=len(sp500log_stationary))
simulated_returns = sim['data'].values / 100 + sp500log_mean
plt.hist(simulated_returns, bins=50, density=True, alpha=0.6, color='g', label='GARCH(1,1)')
plt.hist(sp500log, bins=50, density=True, alpha=0.6, color='b', label='Historical')
plt.legend()
plt.show()

plt.plot(np.cumsum(simulated_returns), label='garch')
plt.plot(np.cumsum(sp500log), label='real')
plt.legend()
plt.show()

garch_data = [simulated_returns]
for i in range(4):  # Generate 4 more samples
    sim = model.simulate(res.params, nobs=len(sp500log_stationary))
    simulated_returns = sim['data'].values / 100 + sp500log_mean
    garch_data.append(simulated_returns)

np.save('garch_data.npy', np.array(garch_data))


# GBM model
def generate_gbm_paths(mu, sigma, N, n_paths, dt=1):
    steps = int(N/dt)
    s = np.zeros((n_paths, steps))
    z = np.random.normal(size=(n_paths, steps-1))
    m = (mu - 0.5*sigma**2)*dt
    sig = sigma*np.sqrt(dt)
    for i in range(1, steps):
        s[:, i] = s[:, i-1] + m + sig*z[:, i-1]
    return np.exp(s)


mu_daily = sp500log_mean
sigma = np.std(sp500log)
mu = mu_daily + 0.5 * sigma ** 2

# Quick test
gbm_data = generate_gbm_paths(mu, sigma, N=len(sp500log), n_paths=1).flatten()
gbm_log = np.diff(np.log(gbm_data))  # this is effectively just 'm + sig*z' from the function above
print(f"mean diff: {np.mean(gbm_log)-np.mean(sp500log):.2e}, std diff: {np.std(gbm_log) - np.std(sp500log):.2e}")
plt.hist(gbm_log, bins=50, density=True, alpha=0.6, color='r', label='GBM')
plt.hist(sp500log, bins=50, density=True, alpha=0.6, color='b', label='Historical')
plt.legend()
plt.show()

plt.plot(gbm_log.cumsum(), label='gbm')
plt.plot(sp500log.cumsum(), label='real')
plt.legend()
plt.show()

d = generate_gbm_paths(mu, sigma, N=len(sp500log), n_paths=5)
d = np.diff(np.log(d), axis=1)
np.save('gbm_data.npy', d)

