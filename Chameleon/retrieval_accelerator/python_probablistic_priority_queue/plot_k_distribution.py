"""
From a single L1 queue's perspective: given topK number, 
    what is the probability that k (0~topK) element is landed in the queue?
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

default_colors = []
for i, color in enumerate(plt.rcParams['axes.prop_cycle']):
    default_colors.append(color["color"])
    print(color["color"], type(color["color"]))


N_L1_queue = 16
topK= 100

p_distribution, p_accumulated_distribution = get_k_distribution(
	N_in_stream=N_L1_queue, topK=topK)

x_tick_labels = np.arange(1, topK + 1)
# x_tick_labels = ['k={}'.format(i) for i in x_tick_labels]

fontsize_tick = 15
fontsize_text = 15

fig, ax0 = plt.subplots(1, 1, figsize=(10, 1.5))
ax1 = ax0.twinx()
ax0.bar(x_tick_labels, p_distribution, width=1.0, color=default_colors[0])
ax1.plot(x_tick_labels, p_accumulated_distribution, color=default_colors[1])
# sns.barplot(x=x_tick_labels, y=p_distribution, ax=ax)

#ax0.legend([bar, plot] , [" RCM Model","Large RCM Model"], loc='upper right')
# ax0.legend([bar] , ["Probability that k out of the topK results are in the same queue"], loc='upper right')
# ax1.legend([plt] , [" Accumulated probability"], loc='upper right')

ax0.grid(False)
ax1.grid(False)

ax0.set_ylabel('p(k) in bars', fontsize=fontsize_tick)
# ax0.text(-2, -0.03, 'k', fontsize=fontsize_tick, color='#333333')
# ax0.set_xlabel('k', fontsize=10)
ax1.set_ylabel('P(k) in curve', fontsize=fontsize_tick)
# ax1.set_ylabel('CDF', fontsize=fontsize_tick)
# r'$\sum_{i=0}^{k} p(i)$'

ax1.arrow(20, 0.8, 0, 0.18, width=0.1, shape="full", 
    length_includes_head=True, head_length=0.05, head_width=1, color='black')
ax1.arrow(20, 0.3, 0, -0.3, width=0.1, shape="full", 
    length_includes_head=True, head_length=0.05, head_width=1, color='black')
ax1.text(18 - 1, 0.37, "Given many L1 priority queues, "
    "it is unlikely that \nmore than 20 of the "
    "results appear in the same L1 queue", fontsize=fontsize_text, style='italic')

ax0.tick_params(axis='x', labelsize=fontsize_tick)
ax0.tick_params(axis='y', labelsize=fontsize_tick)
ax1.tick_params(axis='y', labelsize=fontsize_tick)
ax0.set_xlim(0, 100)
ax1.set_ylim(0, 1.05)

# show x tick every 20 elements
ax0.tick_params(top=False, bottom=True, left=True, right=False)#, labelleft=True, labelsize=tick_font)


print("N input streams: {} topK: {}".format(N_L1_queue, topK))

for out_dtype in ['png', 'pdf']:
	plt.savefig(f'./img/k_distribution.{out_dtype}', transparent=False, dpi=200, bbox_inches="tight")
# plt.show()