"""
Given a number of input streams, how large should the L1 queue be?
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# Seaborn version >= 0.11.0

sns.set_theme(style="whitegrid")
# Set the palette to the "pastel" default palette:
# sns.set_palette("pastel")

# https://seaborn.pydata.org/tutorial/color_palettes.html
sns.set_palette("Paired")
# sns.set_palette("Set2")
# sns.set_palette("husl", 6)

from probablistic_priority_queue import get_k_distribution

# plt.style.use('ggplot') 

fontsize_tick_label = 15
fontsize_tick = 13
fontsize_text = 13

default_colors = []
for i, color in enumerate(plt.rcParams['axes.prop_cycle']):
    default_colors.append(color["color"])
    print(color["color"], type(color["color"]))

topK= 100
L1_length_list = []
input_stream_min = 2
inupt_stream_max = 32

L1_length_list.append(topK) # 1 input stream

for i in range(input_stream_min, inupt_stream_max + 1):
    print(i)
    N_L1_queue = i
    p_distribution, p_accumulated_distribution = get_k_distribution(
        N_in_stream=N_L1_queue, topK=topK)
    for k in range(1, topK + 1):
        if (p_accumulated_distribution[k]) ** i > 0.99: # all the results are in topK
            L1_length_list.append(k)
            break

print(L1_length_list)
print("Ensures 99 percent all queue get the exact results as full-length")

for l in [4, 8, 16, 32]:
    print(l, L1_length_list[l - 1])

x_tick_labels = np.arange(1, inupt_stream_max + 1)

fig, ax0 = plt.subplots(1, 1, figsize=(8, 1.3))
ax0.plot(x_tick_labels, L1_length_list, color=default_colors[3])
# sns.barplot(x=x_tick_labels, y=p_distribution, ax=ax)

ax0.tick_params(top=False, bottom=True, left=True, right=False)

#ax0.grid(False)
ax0.set_xlabel('Number of L1 queues', fontsize=fontsize_tick)
ax0.set_ylabel('Min length\nper L1 queue', fontsize=fontsize_tick)

ax0.tick_params(axis='x', labelsize=fontsize_tick)
ax0.tick_params(axis='y', labelsize=fontsize_tick)

# x, y, dx, dy
ax0.arrow(32, 100, 0, -90, width=0.1, shape="full", length_includes_head=True, head_length=10, head_width=1, color='black')
ax0.text(30, 50, "AHPQ can save 10X hardware resources\ngiven there are many queues", fontsize=fontsize_text, style='italic', horizontalalignment='right')
ax0.set(ylim=[0, 100], xlim=[1, 33])
# set y tick 0, 25, 50, 75, 100
ax0.set_yticks(np.arange(0, 101, 25))

# remove grid
ax0.grid(False)

for out_dtype in ['png', 'pdf']:
	plt.savefig(f'./img/queue_size_vs_input_stream.{out_dtype}', transparent=False, dpi=200, bbox_inches="tight")
# plt.show()