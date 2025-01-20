"""
The RALM performance is stored in a hierachical dictionary: 
    d_perf = d[model_name][architecture][retrieval_interval][batch_size]
        (we assume [model_name][index][k][nprobe] are tied to a certain model)
        d_perf['latency_ms'] = [latency_array_1 (batch 1), latency_array_2 (batch 2), ...],
            np array, shape = (batch_size, seq_len)
        d_perf['throughput_tokens_per_sec'] = [throughput_1, throughput_2, ...],
            np array, shape = (batch_size,)
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

plt.style.use('ggplot')

def load_obj(dirc, name):
    with open(os.path.join(dirc, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

e2e_latency_dict_baseline = load_obj('../performance_results_archive/', 'RALM_latency_throughput_CPU_GPU')
search_latency_dict_baseline = load_obj('../performance_results_archive/', 'vector_search_latency_CPU_GPU_dict')

def plot_latency(models=['Dec-S', 'EncDec-S'], batch_size=1):
    x_labels = []
    ys_e2e = []
    ys_search = []
    ys_generation = []

    for model_name in models:

        k = 100
        nprobe = 32

        if model_name == 'Dec-S': 
            CPU_architecture = '8CPU'
            retrieval_intervals = [1]
            dbname = 'RALM-S1000M'
            index_key = 'IVF32768,PQ32'
        elif model_name == 'EncDec-S':
            CPU_architecture = '8CPU'
            retrieval_intervals = [8, 64, 512]
            dbname = 'RALM-S1000M'
            index_key = 'IVF32768,PQ32'
        elif model_name == 'Dec-L':
            CPU_architecture = '16CPU'
            retrieval_intervals = [1]
            dbname = 'RALM-L1000M'
            index_key = 'IVF32768,PQ64'
        elif  model_name == 'EncDec-L':
            CPU_architecture = '16CPU'
            retrieval_intervals = [8, 64, 512]
            dbname = 'RALM-L1000M'
            index_key = 'IVF32768,PQ64'

        seq_length = 512

        for retrieval_interval in retrieval_intervals:
            x_label = f'{model_name}\ninterval={retrieval_interval}' 
            y_e2e_latency = np.sum(e2e_latency_dict_baseline[model_name][CPU_architecture][retrieval_interval][batch_size]['latency_ms'], axis=1) / 1000
            y_e2e_latency = np.average(y_e2e_latency)
            per_search_latency = np.average(search_latency_dict_baseline[dbname][index_key][CPU_architecture][k][nprobe][batch_size]) / 1000
            y_search_latency = seq_length / retrieval_interval * per_search_latency
            y_generation_latency = y_e2e_latency - y_search_latency
            
            x_labels.append(x_label)
            ys_e2e.append(y_e2e_latency)
            ys_search.append(y_search_latency)
            ys_generation.append(y_generation_latency)

    x = np.arange(len(x_labels))  # the label locations

    ys_e2e = np.array(ys_e2e)
    ys_search = np.array(ys_search)
    ys_generation = np.array(ys_generation)
    print(ys_e2e)
    print(ys_search)
    print(ys_generation)


    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(6, 3))
    # rects1  = ax.bar(x - width/2, y_e2e_latency_means, width)#, label='Men')

    rects1_lower  = ax.bar(x - width/2, ys_search, width)#, label='bar 1 lower')
    rects1_upper  = ax.bar(x - width/2, ys_generation, width, bottom=ys_search)#, label='bar 1 higher')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Per-sequence Latency (sec)')
    ax.text(0.5, 0.8, f'{dbname}, batch size={batch_size}', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
    # ax.set_title(f'{dbname}, batch size={batch_size}')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend([rects1_lower, rects1_upper], ["search latency", "generation latency"], loc="lower right", ncol=2, \
        facecolor='white', framealpha=1, frameon=False, bbox_to_anchor=(0, 1, 1, 0.2))

    def autolabel(rects, align='center'):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            text = '{:.1f}'.format(height)

            if align == 'center':
                text_x = 0
            elif align == 'left':
                text_x = -len(text)
            elif align == 'right':
                text_x = len(text)
            ax.annotate(text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(text_x, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=0)


    # autolabel(rects1, align='center')

    plt.rcParams.update({'figure.autolayout': True})


    plt.savefig(f'./images/ralm_latency_per_sequence_{dbname}_batch_{batch_size}.png', transparent=False, dpi=200, bbox_inches="tight")
    # plt.show()

if __name__ == "__main__":

    batch_size = 1
    models = ['Dec-S', 'EncDec-S']#, 'Dec-L', 'EncDec-L']
    for batch_size in [1, 64]:
        plot_latency(models=models, batch_size=batch_size)
    models = ['Dec-L', 'EncDec-L']
    for batch_size in [1, 8]:
        plot_latency(models=models, batch_size=batch_size)