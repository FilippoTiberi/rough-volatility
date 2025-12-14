import numpy as np
import matplotlib.pyplot as plt

from fbm.cholesky_fbm import simulate_fbm_cholesky
from fbm.hurst_estimation import hurst_variance_scaling

T = 1.0
H = 0.2
t, paths = simulate_fbm_cholesky(n_steps=512, hurst=H, T=T, n_paths=5, random_state=0)

plt.figure(figsize=(10, 4))
for path in paths:
    plt.plot(t, path, alpha=0.7)
plt.title(f"fBm paths (H = {H})")
plt.xlabel("t")
plt.ylabel("B^H_t")
plt.show()

est = hurst_variance_scaling(paths[0])
print("Estimated Hurst:", est)
