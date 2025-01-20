"""
Given a single FPGA of N channels, each channel have M PEs, colleck topK vectors.

Suppose we use a 2-level priority queue (each PE attaches 2 queues), 
    calculate the queue size of the first level L, 
    that barely sacrifices recall but reduces resource consumption significantly.
"""
import numpy as np
from scipy.special import comb

def get_k_distribution(N_in_stream, topK):
    """
    For a single L1 priority queue, the probability that there are 
        k of the topK result in the queue.
    """
    P_in_queue = 1 / N_in_stream
    P_not_in_queue = 1 - 1 / N_in_stream
    
    # For each L1 queue, 1 == the sum of the distribution that the queue contain k top vectors
    p_distribution = np.zeros(topK)
    p_accumulated_distribution = np.zeros(topK)

    for k in range(topK):
        # probability that k of the topK vectors are in a single priority queue
        p_distribution[k] = comb(N=topK, k=k) * (P_in_queue ** k) * (P_not_in_queue ** (topK - k))
        p_accumulated_distribution[k] = np.sum(p_distribution)

    return p_distribution, p_accumulated_distribution

if __name__ == '__main__':
    N = 4 # 4 channels for Enzian/U250
    PQ_bytes = 64
    M = 512 / 8 / PQ_bytes # for 16-byte=128 bit codes and 512-bit AXI width, M = 4 vec per cycle
    topK = 100 # while k is reserved for k top vectors in a queue
    N_PE = N * M 
    N_L1_queue = N_PE * 2
    print("topK: {}\tL1 queue num: {}".format(topK, N_L1_queue))
    print("Average topK vectors in a single queue: {:.2f}".format(topK / N_L1_queue))

    p_distribution, p_accumulated_distribution = get_k_distribution(
        N_in_stream=N_L1_queue, topK=topK)

    L1_length = None
    for k in range(topK):
        if p_accumulated_distribution[k] > 0.9999:
            L1_length = k
            print("queue size = {} ensures 99.99 percent ({}) hit rage".format(
                k, p_accumulated_distribution[k]))
            break

    print("\n\nConclusion: Given")
    print("{} channels".format(N))
    print("{}-byte PQ codes".format(PQ_bytes))
    print("{} L1 priority queues".format(N_L1_queue))
    print("topK = {}".format(topK))
    print("Approximate L1 queue size k = {}: only 0.01 percent of cases there's any topK"
        " result will be filtered out".format(L1_length))

    print("Resource save: ")
    original_total_len = (N_L1_queue + 1) * topK
    print("Originally: {} L1 queue + 1 L2 queue, both of length of {}, total length = {}".format(
        N_L1_queue, topK, original_total_len))
    optimized_total_leng = N_L1_queue * L1_length + 1 * topK
    print("Optimized: {} L1 queue of length {} + 1 L2 queue length of {}, total length = {}".format(
        N_L1_queue, L1_length, topK, optimized_total_leng))
    print("Resource consumption ratio: opt / original = {}".format(optimized_total_leng / original_total_len))