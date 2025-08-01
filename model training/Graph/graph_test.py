import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['Dataset 1', 'Dataset 2', 'Dataset 3']
accuracy_real = [99, 99, 99]  # Accuracy on real dataset
accuracy_obfuscated = [35.67, 30.00, 12.83]  # 

x = np.arange(len(datasets))  # the label locations
width = 0.36  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 8), dpi=200)  # High DPI for on-screen rendering

# Bars for real dataset accuracy
bars1 = ax.bar(x - width/2, accuracy_real, width, label='Real Dataset', color='lightblue', edgecolor='blue', hatch='//', antialiased=True)
# Bars for obfuscated dataset accuracy
bars2 = ax.bar(x + width/2, accuracy_obfuscated, width, label='Obfuscated Dataset', color='lightsalmon', edgecolor='red', hatch='..', antialiased=True)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Datasets', fontsize=24)
ax.set_ylabel('Accuracy (%)', fontsize=24)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=22)
ax.set_yticklabels(ax.get_yticks(), fontsize=22)

# Place the legend above the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.114), ncol=2, frameon=True, fontsize=22)

fig.tight_layout()

# Save the figure with a high DPI and in a vector format for the best quality
fig.savefig('high_quality_plot.png', dpi=300)  # High DPI for bitmap format
fig.savefig('high_quality_plot.pdf')  # Vector format
fig.savefig('high_quality_plot.svg')  # Vector format

plt.show()
