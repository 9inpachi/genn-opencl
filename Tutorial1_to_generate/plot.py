import matplotlib.pyplot as plt
import numpy as np

state = np.loadtxt("spikes.csv", delimiter=",")

# Create plot
figure, axes = plt.subplots(7, sharex=True)

# Plot voltages
for i, t in enumerate(["RS", "FS", "CH", "IB", "TC", "RZ", "LTS"]):
    axes[i].set_title(t)
    axes[i].set_ylabel("V [mV]")
    axes[i].plot(state[:,0], state[:,1 + i])

axes[-1].set_xlabel("Time [ms]")

# Show plot
plt.show()
