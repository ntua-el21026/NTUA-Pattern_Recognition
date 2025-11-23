import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots

# -----------------------------
# 0. Ensure 'plots/' directory exists
# -----------------------------
# If 'plots' does not exist, it is created. If it exists, nothing happens.
os.makedirs("plots", exist_ok=True)

# -----------------------------
# 1. Define x1 and x2 ranges
# -----------------------------

# x1 from 0 to 6 (covers our samples 1, 2, 4 and the exponential tail)
x1 = np.linspace(0, 6, 200)

# x2 ranges:
#   - Before EM: [0, 3]
#   - After EM : [0, 5]
x2_before = np.linspace(0, 3, 150)
x2_after  = np.linspace(0, 5, 150)

# Create 2D grids (mesh) of x1 and x2 values
X1_before, X2_before = np.meshgrid(x1, x2_before)
X1_after,  X2_after  = np.meshgrid(x1, x2_after)

# --------------------------------------
# 2. Define p(x1, x2) before and after EM
# --------------------------------------
# We only care about the *shape* of the distribution (unnormalized):
#   Before EM:  p(x1) ∝ exp(-2 x1),   p(x2) = const on [0, 3]
#   After EM :  p(x1) ∝ exp(-3/7 x1), p(x2) = const on [0, 5]

# Before EM
p1_before = np.exp(-2 * X1_before)     # exponential in x1
p2_before = np.ones_like(X2_before)    # uniform in x2
p_before  = p1_before * p2_before      # joint p(x1,x2) up to a constant

# After EM
p1_after = np.exp(-3 * X1_after / 7.0) # slower-decaying exponential
p2_after = np.ones_like(X2_after)      # uniform on [0,5]
p_after  = p1_after * p2_after         # joint p(x1,x2) up to a constant

# --------------------------------------
# 3. Plot the two surfaces side by side
# --------------------------------------

fig = plt.figure(figsize=(10, 4))

# Before EM
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X1_before, X2_before, p_before)
ax1.set_title('p(x1,x2) at θ(0) of EM')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('density')

# After EM
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X1_after, X2_after, p_after)
ax2.set_title('p(x1,x2) at θ(1) of EM')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('density')

plt.tight_layout()

# --------------------------------------
# 4. Save figure inside 'plots/' directory
# --------------------------------------
output_path = "plots/EM_Plots.png"
fig.savefig(output_path)
