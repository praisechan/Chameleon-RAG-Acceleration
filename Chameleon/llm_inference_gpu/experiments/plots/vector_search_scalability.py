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
def get_default_colors():

  default_colors = []
  for i, color in enumerate(plt.rcParams['axes.prop_cycle']):
      default_colors.append(color["color"])
      print(color["color"], type(color["color"]))

  return default_colors

default_colors = get_default_colors()

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

# sns.set_theme(style="whitegrid")
# Set the palette to the "pastel" default palette:
# sns.set_palette("pastel")

plt.style.use('seaborn-pastel') 

k = 100
nprobe = 32
# batch_sizes = [1, 4, 16, 64]
batch_sizes = [1, 64]
n_nodes_list = [1, 2, 4, 8, 16, 32, 64, 128]
n_queries = 100
# dbname_list = ['SIFT1000M', 'Deep1000M', 'RALM-S1000M', 'RALM-L1000M']
dbname_list = ['RALM-S1000M']


def get_architectures_and_index_keys(dbname):
    if dbname == 'SIFT1000M' or dbname == 'Deep1000M':
        index_key = 'IVF32768,PQ16'
        architectures = ['8CPU', '8CPU-1GPU', '1FPGA-8CPU', '1FPGA-1GPU']
    elif dbname == 'RALM-S1000M':
        index_key = 'IVF32768,PQ32'
        architectures = ['8CPU', '8CPU-1GPU', '1FPGA-8CPU', '1FPGA-1GPU']
    # elif dbname == 'RALM-S2000M':
    #     index_key = 'IVF32768,PQ32'
    #     architectures = ['16CPU', '16CPU-1GPU', '2FPGA-8CPU', '2FPGA-1GPU']
    elif dbname == 'RALM-L1000M':
        index_key = 'IVF32768,PQ64'
        architectures = ['16CPU', '16CPU-1GPU', '2FPGA-8CPU', '2FPGA-1GPU']
    # elif dbname == 'RALM-L2000M':
    #     index_key = 'IVF32768,PQ64'
    #     architectures = ['16CPU', '16CPU-1GPU', '4FPGA-8CPU', '4FPGA-1GPU']
    return architectures, index_key


def load_obj(dirc, name):
    with open(os.path.join(dirc, name + '.pkl'), 'rb') as f:
        return pickle.load(f)


def plot_tail(dbname):

    def get_tail_latency(tail_perc=0.99):
        architectures, index_key = get_architectures_and_index_keys(dbname)
        latency_dict_FPGA = load_obj('./performance_results_archive/', 'vector_search_latency_FPGA_dict')

        tail_latency_dict = {}
        for batch_size in batch_sizes: 
            tail_latency_dict[batch_size] = []

        for arch in ['1FPGA-1GPU']:
            for batch_size in batch_sizes:
                latency_array = latency_dict_FPGA[dbname][index_key][arch][k][nprobe][batch_size]
                print(latency_array)
                for n_nodes in n_nodes_list:
                    simulated_latency = []
                    for i in range(n_queries):
                        # subsample n_queries elements from latency_array
                        np.random.seed(i)
                        latency_array_subsample = np.random.choice(latency_array, n_nodes, replace=True)
                        tail_latency = np.max(latency_array_subsample)
                        simulated_latency.append(tail_latency)
                
                    data_bytes = batch_size * ((1 + nprobe) * 512 * 4 + nprobe * 8 + 100 * (4 + 8))
                    transfer_time = n_nodes * data_bytes / (100 * 1e9 / 8) * 1e-3
                    network_latency = np.ceil(np.log2(n_nodes)) * 10 * 1e-3 + transfer_time

                    tail = np.quantile(simulated_latency, tail_perc) + network_latency
                    print("Min: ", np.min(simulated_latency))
                    print("Max: ", np.max(simulated_latency))
                    print("Network latency: ", network_latency)
                    print("{}th Tail latency: ".format(int(tail_perc * 100)), tail)
                    tail_latency_dict[batch_size].append(tail)
        return tail_latency_dict

    tail_latency_dict_99 = get_tail_latency(tail_perc=0.99)
    tail_latency_dict_50 = get_tail_latency(tail_perc=0.50)

    label_font = 11
    markersize = 8
    tick_font = 9
    legend_font = 10

    fig, ax = plt.subplots(figsize=(6, 1.8))

    x_labels = [str(n_nodes) for n_nodes in n_nodes_list]
 
    ax.plot(x_labels, tail_latency_dict_50[1], linewidth=2, marker='X', markersize=markersize, label='median, b=1, incr={:.1f}%'.format(100 * (tail_latency_dict_50[1][-1] - tail_latency_dict_50[1][0]) / tail_latency_dict_50[1][0]))
    ax.plot(x_labels, tail_latency_dict_50[64], linewidth=2, marker='o', markersize=markersize, label='median, b=64, incr={:.1f}%'.format(100 * (tail_latency_dict_50[64][-1] - tail_latency_dict_50[64][0]) / tail_latency_dict_50[64][0]))

    ax.plot(x_labels, tail_latency_dict_99[1], linewidth=2, marker='v', markersize=markersize, label='99th, b=1, incr={:.1f}%'.format(100 * (tail_latency_dict_99[1][-1] - tail_latency_dict_99[1][0]) / tail_latency_dict_99[1][0]))
    ax.plot(x_labels, tail_latency_dict_99[64], linewidth=2, marker='^', markersize=markersize, label='99th, b=64, incr={:.1f}%'.format(100 * (tail_latency_dict_99[64][-1] - tail_latency_dict_99[64][0]) / tail_latency_dict_99[64][0]))

    ax.text(0, 20, "Dataset: SYN-512", fontsize=legend_font)
    ax.annotate("Close median and 99th latency (b = 1)", xy=(4, tail_latency_dict_99[1][4]), xytext=(2.8, 20), arrowprops={"arrowstyle": '-|>', 'color': '#1f1f1f', 'linewidth': 2}, fontsize=legend_font)
    # ax.annotate("{:.1f}".format(y_Dec_S[0]), xy=(0, y_Dec_S[0]), xytext=(0.3, y_Dec_S[0] * 3), arrowprops={"arrowstyle": '-|>', 'color': '#1f1f1f', 'linewidth': 2}, fontsize=label_font)
    # ax.annotate("{:.1f}".format(y_EncDec_L[-1]), xy=(len(x_labels_EncDec) -1, y_EncDec_L[-1]), xytext=(len(x_labels_EncDec) - 1.8, y_EncDec_L[-1] / 20), arrowprops={"arrowstyle": '-|>', 'color': '#1f1f1f', 'linewidth': 2}, fontsize=label_font)

    # ax.legend([plot_EncDec_S, plot_EncDec_L] , ["EncDec-S","EncDec-L"], loc='upper left')
    # plt.legend([plot_EncDec_S] , ["EncDec_S"], loc='upper left', fontsize=14)
    ax.legend(fontsize=legend_font, ncol=2, loc=(0.02, 0.28), frameon=False)

    ax.set_ylabel('Latency (ms)', fontsize=label_font)
    ax.set_xlabel('Number of FPGA-based Disaggregated Memory Nodes', fontsize=label_font)
    ax.set_ylim([0, 120])
    # ax.set_yscale('log')

    plt.rcParams.update({'figure.autolayout': True})
    fig.tight_layout()
    for out_dtype in ['png', 'pdf']:
        plt.savefig(f'./images/scalability_{dbname}.{out_dtype}', transparent=False, dpi=200, bbox_inches="tight")

    # # plt.show()


if __name__ == "__main__":
    for dbname in dbname_list:
        plot_tail(dbname)
        # plot_tail(dbname, tail_perc=0.99)

    print("==== Summary ====")
    # print("Median speedup: CPU-GPU/CPU: {:.2f}~{:.2f}\tCPU-FPGA/CPU: {:.2f}~{:.2f}\tGPU-FPGA/CPU: {:.2f}~{:.2f}\tGPU-FPGA/CPU-FPGA: {:.2f}~{:.2f}".format(
    #     min_speedup_median_cpu_gpu_to_cpu, max_speedup_median_cpu_gpu_to_cpu, 
    #     min_speedup_median_cpu_fpga_to_cpu, max_speedup_median_cpu_fpga_to_cpu, 
    #     min_speedup_median_gpu_fpga_to_cpu, max_speedup_median_gpu_fpga_to_cpu, 
    #     min_speedup_median_gpu_fpga_to_cpu_fpga, max_speedup_median_gpu_fpga_to_cpu_fpga))
    # print("95th percentile speedup: CPU-GPU/CPU: {:.2f}~{:.2f}\tCPU-FPGA/CPU: {:.2f}~{:.2f}\tGPU-FPGA/CPU: {:.2f}~{:.2f}\tGPU-FPGA/CPU-FPGA: {:.2f}~{:.2f}".format(
    #     min_speedup_p95_cpu_gpu_to_cpu, max_speedup_p95_cpu_gpu_to_cpu,
    #     min_speedup_p95_cpu_fpga_to_cpu, max_speedup_p95_cpu_fpga_to_cpu,
    #     min_speedup_p95_gpu_fpga_to_cpu, max_speedup_p95_gpu_fpga_to_cpu,
    #     min_speedup_p95_gpu_fpga_to_cpu_fpga, max_speedup_p95_gpu_fpga_to_cpu_fpga))
        