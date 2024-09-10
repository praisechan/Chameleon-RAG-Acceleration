# Plots

The plotting scripts for approximate K-selection queue distributions are in another repository (`PQMem`).

## Generating all plots used in the paper

Vector search latency: 
```
python vector_search_latency.py
```

Vector search scalability estimation:
```
python vector_search_scalability.py
```

RALM latency over steps:
```
python ralm_latency_over_steps_plot.py
```

RALM throughput with batching: 
```
python ralm_throughput.py
```

Calculate the optimal accelerator ratio between the two types of accelerators,
	
```
python accelerator_ratio.py 
```

## Unused plotss:

The total latency (in bar) of a sequence on FPGA-GPU / CPU-GPU:
```
python unused_ralm_latency_per_seq.py
```

The latency oversteps of RALM, including the latency distribution as a side plot:
```
python unused_ralm_latency_over_steps_plot_old.py
```

The latency distribution of RALM.
```
unused_ralm_latency_violin_plot.py
```

Calculate the optimal accelerator ratio between the two types of accelerators,
	this is based on the assumption of T_inference = T_retrieval, without considering the bubbles in between, so it's wrong:
```
python unused_accelerator_ratio.py 
```