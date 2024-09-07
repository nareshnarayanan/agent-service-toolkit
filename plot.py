import matplotlib.pyplot as plt
import numpy as np

# Data for the graph
labels = ['Pew Research', 'NY Times']
kamala_harris = [46, 49]
donald_trump = [45, 46]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, kamala_harris, width, label='Kamala Harris', color='lightblue')
rects2 = ax.bar(x + width/2, donald_trump, width, label='Donald Trump', color='salmon')

# Adding some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percentage (%)')
ax.set_title('Polling Matchup: Kamala Harris vs Donald Trump')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Function to add labels on top of the bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()