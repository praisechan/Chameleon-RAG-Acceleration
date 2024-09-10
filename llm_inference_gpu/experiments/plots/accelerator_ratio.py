import numpy as np
import pickle
import os

"""
Math of this script can be found in:
https://www.overleaf.com/7828499779mzpgkdgnwndv#d78388
"""
def calculate_throughput(i, b, N_I, N_R, L_I_b, L_R_b, print_debug=True, latency_scales_with_N_R=True):
    """
    Calculate the adjusted inference latency and the generation throughput.

    Parameters:
    i (int): Retrieval interval, number of inference operations between retrievals.
    b (int): Batch size, number of tokens processed per batch.
    N_I (int): Number of inference accelerators.
    N_R (int): Number of retrieval accelerators.
    L_R_b (float): Latency for retrieval per batch. -> we assume the latency decreases proportionally with the number of retrieval accelerators
    L_I_b (float): Initial inference latency per batch. -> we assume the latency remains stable regardless the number of inference accelerators

    Returns:
    float: The end-to-end generation throughput after possible adjustment of inference latency.
    """
    # Calculate retrieval throughput in tokens per second correctly as per the latest formula
    Th_retrieval_tokens = i * b * N_R / L_R_b  # Correct retrieval throughput calculation
    Th_inference_tokens = N_I * b / L_I_b
    if print_debug:
        print(f"Th_retrieval_tokens: {Th_retrieval_tokens:.2f} tokens/sec (the retrieval throughput can support this much of tokens generated at most)")
        print(f"Pure inference throughput: {Th_inference_tokens:.2f} tokens/sec")
        if Th_retrieval_tokens < Th_inference_tokens:
            print("Retrieval is the bottleneck")
        else: 
            print("Inference is the bottleneck")

    if latency_scales_with_N_R: 
        # Assuming retrieval latency decreases proportionally with the number of retrieval accelerators
        Th_generation_assuming_infinite_retrieval_resources = i * b * N_I / (i * L_I_b  + L_R_b / N_R)
    else:
        # assuming retrieval latency remains the same regardless of the number of retrieval accelerators -> thus bubble in GPU is huge
        Th_generation_assuming_infinite_retrieval_resources = i * b * N_I / (i * L_I_b  + L_R_b)

    if Th_generation_assuming_infinite_retrieval_resources > Th_retrieval_tokens:
        if print_debug:
            print("Retrieval is the bottleneck (Th_generation_assuming_infinite_retrieval_resources > Th_retrieval_tokens)")
        Th_generation_final = Th_retrieval_tokens
    else:
        if print_debug:
            print("Inference is the bottleneck (Th_generation_assuming_infinite_retrieval_resources < Th_retrieval_tokens)")
        Th_generation_final = Th_generation_assuming_infinite_retrieval_resources
    
    # # Adjust L'_I(b) if necessary
    # L_prime_I_b = ((1 - N_R) * L_R_b * N_I) / (i * N_R)
    # print("L_prime_I_b" , L_prime_I_b, "L_I_b", L_I_b)
    # L_prime_I_b = max(L_prime_I_b, L_I_b)  # Ensure it does not fall below the original
    # # Calculate the final generation throughput using the adjusted or original L'_I(b)
    # Th_generation_final = i * b  / (i * L_prime_I_b / N_I + L_R_b)
    # if print_debug:
    #     # print(f"Adjusted Inference latency: {L_prime_I_b:.4f} seconds")
    #     print(f"Final generation throughput: {Th_generation_final:.2f} tokens/sec")

    return Th_generation_final
    # return L_prime_I_b, Th_generation_final

def find_optimal_ratio_consider_bubble(i, b, L_I_b, L_R_b, total_units=2, step_size=1, latency_scales_with_N_R=True):
    """
    total_units and step size are integers
    """
    max_throughput = 0
    optimal_N_I = 0
    optimal_N_R = total_units  # Initialize with all units assigned to retrieval

    # Iterate over possible values of N_I from 0 to total_units in steps of step_size
    for N_I in range(1, total_units + 1, step_size):
        N_R = total_units - N_I  # Ensure the sum of N_I and N_R equals total_units
        if N_I == 0 or N_R == 0:
            continue
        # print("N_I: {}, N_R: {}".format(N_I, N_R))
        current_throughput = calculate_throughput(i, b, N_I, N_R, L_I_b, L_R_b, print_debug=False, latency_scales_with_N_R=latency_scales_with_N_R)
        # print(f"N_I: {N_I:.3f}, N_R: {N_R:.3f}, Throughput: {current_throughput:.2f} tokens/second")
        
        if current_throughput > max_throughput:
            max_throughput = current_throughput
            optimal_N_I = N_I
            optimal_N_R = N_R

    return optimal_N_I, optimal_N_R, max_throughput

def find_optimal_ratio_perfect_pipeline(i, b, L_I_b, L_R_b, total_units=1, step_size=0.01):
    optimal_ratio_I_to_R = L_R_b / (i * L_I_b)
    optimal_N_I = total_units / (1 + optimal_ratio_I_to_R)
    optimal_N_R = total_units - optimal_N_I
    return optimal_N_I, optimal_N_R

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

"""
Vector search: 

Latency result format (in dict): [dbname][index][architecture][k][nprobe][batch_size] = latency_array (ms)
    dbname example: 'SIFT1000M'
    index example: 'IVF32768,PQ16'
    nprobe example: 32
    batch_size example: 64
    architecture example: '16CPU-1GPU' or '32CPU', here the number means the number of CPU cores and GPU cards
    latency_array example: np.array([1.5, 3.4, ...]) (ms)

Throughput result format (in dict): [dbname][index][architecture][k][nprobe] = throughput_array (QPS)
"""
def load_obj(dirc, name):
    with open(os.path.join(dirc, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    
    # model_names = ['Dec-S']
    # model_names = ['Dec-S', 'EncDec-S']
    model_names = ['Dec-S', 'EncDec-S', 'Dec-L', 'EncDec-L']

    k = 100
    nprobe = 32
    
    for model_name in model_names:
        if model_name == 'Dec-S': 
            CPU_architecture = '8CPU'
            FPGA_architecture = '1FPGA-1GPU'
            retrieval_intervals = [1]
            batch_sizes = [1, 64]
            dbname = 'RALM-S1000M'
            n_FPGA = 1
        elif model_name == 'EncDec-S':
            CPU_architecture = '8CPU'
            FPGA_architecture = '1FPGA-1GPU'
            retrieval_intervals = [8, 64, 512]
            batch_sizes = [1, 64]
            dbname = 'RALM-S1000M'
            n_FPGA = 1
        elif model_name == 'Dec-L':
            CPU_architecture = '16CPU'
            FPGA_architecture = '2FPGA-1GPU'
            retrieval_intervals = [1]
            batch_sizes = [1, 8]
            dbname = 'RALM-L1000M'
            n_FPGA = 2
        elif  model_name == 'EncDec-L':
            CPU_architecture = '16CPU'
            FPGA_architecture = '2FPGA-1GPU'
            retrieval_intervals = [8, 64, 512]
            batch_sizes = [1, 8]
            dbname = 'RALM-L1000M'
            n_FPGA = 2
                
        dbname_list = ['SIFT1000M', 'Deep1000M', 'RALM-S1000M', 'RALM-L1000M']
        assert dbname in dbname_list, f"dbname should be one of {dbname_list}"

        for batch_size in batch_sizes:

            if batch_size == 1:
                continue    

            # retrieval part
            architectures, index_key = get_architectures_and_index_keys(dbname)
            latency_dict_FPGA = load_obj('./performance_results_archive/', 'vector_search_latency_FPGA_dict')
            
            # inference part
            latency_dict_generation_only = load_obj('./performance_results_archive/', 'RALM_latency_throughput_only_inference')

            for retrieval_interval in retrieval_intervals:

                # both in ms, so / 1000
                if batch_size == 8: # not measured as latency
                    average_retrieval_latency = latency_dict_FPGA[dbname][index_key][FPGA_architecture][k][nprobe][4].mean() * 2 / 1000
                else:
                    average_retrieval_latency = latency_dict_FPGA[dbname][index_key][FPGA_architecture][k][nprobe][batch_size].mean() / 1000
                average_inference_latency = latency_dict_generation_only[model_name][CPU_architecture][retrieval_interval][batch_size]['latency_ms'].mean() / 1000
                # print(f"Retrieval Latency: {average_retrieval_latency:.4f}, Inference Latency: {average_inference_latency:.4f}, Retrieval Interval: {retrieval_interval}, Batch Size: {batch_size}")

                i = retrieval_interval
                b = batch_size
                N_R = 1  # Number of sets of retrieval accelerators
                N_I = 1  # Number of inference accelerators
                L_I_b = average_inference_latency  # Inference latency per batch (seconds)
                L_R_b = average_retrieval_latency

                # Calculate the throughput
                print(f"===== Model: {model_name}, retrieval_interval: {retrieval_interval}, batch_size: {batch_size} =====")
                Th_generation_final = calculate_throughput(i, b, N_I, N_R, L_I_b, L_R_b)
                print(f"RALM end-to-end throughput is {Th_generation_final:.2f} tokens/sec")
                
                def print_ratio_given_n_FPGA(optimal_N_I, optimal_N_R, n_FPGA):
                    optimal_num_FPGA = optimal_N_R * n_FPGA
                    optimal_num_GPU = optimal_N_I
                    if optimal_num_FPGA > optimal_num_GPU:
                        print(f"\tOptimal N_FPGA : N_GPU : {optimal_num_FPGA / optimal_num_GPU: .2f} : 1")
                    else:
                        print(f"\tOptimal N_FPGA : N_GPU : 1 : {optimal_num_GPU / optimal_num_FPGA: .2f}")

                for latency_scales_with_N_R in [True, False]:
                    optimal_N_I, optimal_N_R, max_throughput = find_optimal_ratio_consider_bubble(i, b, L_I_b, L_R_b, total_units=1000, step_size=1, latency_scales_with_N_R=latency_scales_with_N_R)
                    print(f"Retrieval latency scales with N_R: {latency_scales_with_N_R}\tlatency_scales_with_N_R (Consider bubble) Optimal N_I: {optimal_N_I:.3f}, Optimal N_R: {optimal_N_R:.3f}, Maximum Throughput: {max_throughput:.2f} tokens/second")
                    print_ratio_given_n_FPGA(optimal_N_I, optimal_N_R, n_FPGA)
                
                # optimal_N_I, optimal_N_R = find_optimal_ratio_perfect_pipeline(i, b, L_I_b, L_R_b, total_units=2, step_size=0.001)
                # print(f"(Perfect pipelining) Optimal N_I: {optimal_N_I:.3f}, Optimal N_R: {optimal_N_R:.3f}")
                # print_ratio_given_n_FPGA(optimal_N_I, optimal_N_R, n_FPGA)

    # # #### Demo test ####
    # i = 1  # Retrieval interval
    # b = 16  # Batch size
    # N_R = 1  # Number of retrieval accelerators
    # N_I = 1  # Number of inference accelerators
    # N_R = 1.5  # Number of retrieval accelerators
    # N_I = 0.5  # Number of inference accelerators
    # L_I_b = 0.01  # Inference latency per batch (seconds)
    # L_R_b = 0.1  # Retrieval latency per batch (seconds)
    # # L_I_b = 0.01  # Inference latency per batch (seconds)
    # # L_R_b = 0.01  # Retrieval latency per batch (seconds)
    # # Calculate the throughput 
    # result = calculate_throughput(i, b, N_I, N_R, L_I_b, L_R_b)
    # print(result)