import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("predictions.csv", delimiter=",", skiprows=1)
y_true = data[:, 0]
y_pred = data[:, 1]

stage_names = ['Wake', 'NREM', 'REM']
time_hours = np.arange(len(y_true)) * 30 / 3600

plt.figure(figsize=(14, 5))
plt.step(time_hours, y_true, where='post', label='True', linewidth=2, color='black')
plt.step(time_hours, y_pred, where='post', label='Predicted', linestyle='--', linewidth=2, color='red')

plt.yticks([0, 1, 2], stage_names)
plt.gca().invert_yaxis()
plt.xlabel('Time (hours)')
plt.ylabel('Sleep Stage')
plt.title('Sleep Stage Prediction (Hypnogram)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("hypnogram.png", dpi=150)
plt.show()