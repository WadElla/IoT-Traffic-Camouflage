import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['IoT-AD', 'IoT Sentinel', 'MUD']
accuracy_real = [99, 99, 99]  # Accuracy on real dataset
accuracy_obfuscated = [25, 25, 25]  # Accuracy after adaptive obfuscation

x = np.arange(len(datasets))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Bars for real dataset accuracy
bars1 = ax.bar(x - width/2, accuracy_real, width, label='Real Dataset', color='lightblue', edgecolor='blue', hatch='//')
# Bars for obfuscated dataset accuracy
bars2 = ax.bar(x + width/2, accuracy_obfuscated, width, label='Obfuscated Dataset', color='lightsalmon', edgecolor='red', hatch='..')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Datasets')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy of Models on Real and Obfuscated Datasets')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()

# Add value labels on top of the bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)

fig.tight_layout()

plt.show()
