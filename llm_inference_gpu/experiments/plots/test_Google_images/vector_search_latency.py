"""
Latency result format (in dict): [dbname][index][architecture][k][nprobe][batch_size] = latency_array (ms)
    dbname example: 'SIFT1000M'
    index example: 'IVF32768,PQ16'
    nprobe example: 32
    batch_size example: 64
    architecture example: '16CPU-1GPU' or '32CPU', here the number means the number of CPU cores and GPU cards
    latency_array example: np.array([1.5, 3.4, ...]) (ms)

    
Throughput result format (in dict): [dbname][index][architecture][k][nprobe] = throughput_array (QPS)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns 

import os
import pickle

# Seaborn version >= 0.11.0

sns.set_theme(style="whitegrid")
# Set the palette to the "pastel" default palette:
# sns.set_palette("pastel")

# plt.style.use('seaborn-pastel') 

def get_default_colors():

  default_colors = []
  for i, color in enumerate(plt.rcParams['axes.prop_cycle']):
      default_colors.append(color["color"])
      print(color["color"], type(color["color"]))

  return default_colors

default_colors = get_default_colors()
print("colors length:", len(default_colors))
# print(default_colors[0], type(default_colors[0]))

### Note: For violin graph, a single violin's data must be in the same column
###   e.g., given 3 violin plots, each with 100 points, the shape of the array
###   should be (100, 3), where the first column is for the first violin and so forth
# fake up some data


# Wenqi: flatten the table to a table. It's a dictionary with the key as schema.
#   The value of each key is an array.
# label category data
# xxx   xxx      xxx
# yyy   yyy      yyy

k = 100
nprobe = 32
batch_sizes = [1, 4, 16, 64]
dbname_list = ['SIFT1000M', 'Deep1000M', 'RALM-S1000M', 'RALM-L1000M']

# python color palette: https://matplotlib.org/stable/gallery/color/named_colors.html

colors = {
          # Deep, SIFT, SYN-512
          '8CPU': default_colors[0],
          '8CPU-1GPU': default_colors[1],
          '1FPGA-8CPU': default_colors[2],
          '1FPGA-1GPU': default_colors[3],

          # SYN-1024
          '16CPU': default_colors[4],
          '16CPU-1GPU': default_colors[5],
          '2FPGA-8CPU': default_colors[6],
          '2FPGA-1GPU': default_colors[7],

          # unused
          '32CPU': default_colors[8],

          'box_plot': 'k',
          'median_bubble': 'white',
          }

def get_architectures_and_index_keys(dbname):
    if dbname == 'SIFT1000M' or dbname == 'Deep1000M':
        index_key = 'IVF32768,PQ16'
        architectures = ['8CPU']
    elif dbname == 'RALM-S1000M':
        index_key = 'IVF32768,PQ32'
        architectures = ['8CPU']
    # elif dbname == 'RALM-S2000M':
    #     index_key = 'IVF32768,PQ32'
    #     architectures = ['16CPU', '16CPU-1GPU', '2FPGA-8CPU', '2FPGA-1GPU']
    elif dbname == 'RALM-L1000M':
        index_key = 'IVF32768,PQ64'
        architectures = ['16CPU']
    # elif dbname == 'RALM-L2000M':
    #     index_key = 'IVF32768,PQ64'
    #     architectures = ['16CPU', '16CPU-1GPU', '4FPGA-8CPU', '4FPGA-1GPU']
    return architectures, index_key


def load_obj(dirc, name):
    with open(os.path.join(dirc, name + '.pkl'), 'rb') as f:
        return pickle.load(f)


def plot_seaborn(dbname):

    architectures, index_key = get_architectures_and_index_keys(dbname)
    assert len(architectures) == 1
    arch = architectures[0]

    d = {}
    d['label'] = [] # batch size
    d['data'] = []
    d['category'] = [] # architecture

    latency_dict_baseline = load_obj('../performance_results_archive/', 'vector_search_latency_CPU_GPU_dict')

    for batch_size in batch_sizes:
        latency_array = latency_dict_baseline[dbname][index_key][arch][k][nprobe][batch_size]
        # print(np.average(latency_array))
        for latency in latency_array:
            d['data'].append(latency)
            # d['data'].append(np.average(latency_array))
            d['label'].append(batch_size)
            d['category'].append(arch)
            
    df = pd.DataFrame(data=d)
    print(df)
    # print(df.columns)

    
    plt.figure(figsize=(6, 3))
    # API: https://seaborn.pydata.org/generated/seaborn.violinplot.html
    # inner{“box”, “quartile”, “point”, “stick”, None}, optional
    ax = sns.violinplot(data=df, scale='area', inner='box', x="label", y="data")

    x_tick_labels = ["{}".format(i) for i in batch_sizes]
    ax.set_xticklabels(x_tick_labels)
    # ax.set_yticklabels(y_tick_labels)
    # plt.yticks(rotation=0)
    # # ax.ticklabel_format(axis='both', style='sci')
    # # ax.set_yscale("log")

    ax.tick_params(length=0, top=False, bottom=True, left=True, right=False, 
        labelleft=True, labelbottom=True, labelsize=12)
    ax.set_xlabel('batch size', fontsize=16, labelpad=10)
    ax.set_ylabel('latency (ms)', fontsize=16, labelpad=10)
    # find the max value in the data, set as maximum y limit
    ax.set_ylim(0, 1.1 * max(df['data']))
    ax.set_title(f'{dbname}, {arch}', fontsize=16)
    # plt.text(2, len(y_tick_labels) + 2, "Linear Heatmap", fontsize=16)

    for out_dtype in ['png', 'pdf']:
        plt.savefig(f'./images/vector_search_latency_{dbname}.{out_dtype}', transparent=False, dpi=200, bbox_inches="tight")




if __name__ == "__main__":
    for dbname in dbname_list:
        #plot_seaborn(dbname)
        plot_seaborn(dbname)
