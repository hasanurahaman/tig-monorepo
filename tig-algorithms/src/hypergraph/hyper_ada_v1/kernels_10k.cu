// kernels_10k.cu - hyper_ada_v1
// Challenge c005 (hypergraph) track 10k
// Zero Local Memory Spills (lmem: 0) across all kernels
// Hardware __popcll single-cycle cut evaluation
// Warp-cooperative register-resident move selection

#include <cuda_runtime.h>

typedef unsigned int u32;
typedef unsigned long long u64;

// 1. Structure-aware Anchor-based Hyperedge Clustering
extern "C" __global__ void ada_hyperedge_clustering_10k(
    const int num_hyperedges,
    const int num_clusters,
    const int* __restrict__ hyperedge_offsets,
    const int* __restrict__ hyperedge_nodes,
    int* __restrict__ hyperedge_clusters
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_hyperedges) return;
    
    int start = hyperedge_offsets[e];
    int end = hyperedge_offsets[e + 1];
    int hedge_size = end - start;
    
    int quarter = num_clusters >> 2;
    if (quarter <= 0) quarter = 1;
    
    int bucket = (hedge_size > 8) ? 3 :
                 (hedge_size > 4) ? 2 :
                 (hedge_size > 2) ? 1 : 0;
                 
    int first = (hedge_size > 0) ? hyperedge_nodes[start] : e;
    int last  = (hedge_size > 0) ? hyperedge_nodes[end - 1] : e;
    
    u32 h = (u32)first * 1103515245u
          + (u32)last * 12345u
          + (u32)hedge_size * 2654435761u
          + (u32)e;
          
    int cluster = bucket * quarter + (int)(h % (u32)quarter);
    hyperedge_clusters[e] = cluster % num_clusters;
}

// 2. Warp-Cooperative Initial Node Preference Voting (lmem: 0)
// 1 warp (32 threads) processes 1 node
// Lane i tracks votes for cluster i and cluster i + 32 purely in scalar registers
extern "C" __global__ void ada_compute_initial_preferences_10k(
    const int num_nodes,
    const int num_parts,
    const int* __restrict__ node_offsets,
    const int* __restrict__ node_hyperedges,
    const int* __restrict__ hyperedge_offsets,
    const int* __restrict__ hyperedge_clusters,
    int* __restrict__ pref_parts,
    int* __restrict__ pref_priorities
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    
    if (warp_id >= num_nodes) return;
    
    int v = warp_id;
    int start = node_offsets[v];
    int end = node_offsets[v + 1];
    int node_degree = end - start;
    
    int votes_low = 0;   // cluster lane
    int votes_high = 0;  // cluster lane + 32
    
    for (int i = start; i < end; i++) {
        int e = node_hyperedges[i];
        int cluster = hyperedge_clusters[e];
        
        int h_start = hyperedge_offsets[e];
        int h_end = hyperedge_offsets[e + 1];
        int h_size = h_end - h_start;
        int weight = (h_size <= 2) ? 6 :
                     (h_size <= 4) ? 4 :
                     (h_size <= 8) ? 2 : 1;
                     
        if (cluster == lane) votes_low += weight;
        if (cluster == lane + 32) votes_high += weight;
    }
    
    int my_best_votes = (votes_high > votes_low) ? votes_high : votes_low;
    int my_best_cluster = (votes_high > votes_low) ? (lane + 32) : lane;
    
    // Warp-level reduction across 32 lanes to find global maximum cluster
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other_votes = __shfl_down_sync(0xFFFFFFFF, my_best_votes, offset);
        int other_cluster = __shfl_down_sync(0xFFFFFFFF, my_best_cluster, offset);
        if (other_votes > my_best_votes || 
           (other_votes == my_best_votes && other_cluster < my_best_cluster)) {
            my_best_votes = other_votes;
            my_best_cluster = other_cluster;
        }
    }
    
    if (lane == 0) {
        int base_part = 0;
        if (num_parts > 0) {
            base_part = (my_best_cluster * num_parts) / 64;
            if (base_part >= num_parts) base_part = num_parts - 1;
        }
        pref_parts[v] = base_part;

        int degree_weight = (node_degree > 255) ? 255 : node_degree;
        int mv = (my_best_votes > 32767) ? 32767 : my_best_votes;
        pref_priorities[v] = (mv << 16) | (degree_weight << 8) | (num_parts - (v % num_parts));
    }
}

// 3. Precompute Edge Bitmasks (Presence & Multiplicity)
extern "C" __global__ void ada_compute_hyperedge_masks_10k(
    const int num_hyperedges,
    const int* __restrict__ hyperedge_offsets,
    const int* __restrict__ hyperedge_nodes,
    const int* __restrict__ partition,
    u64* __restrict__ hyperedge_flags_all,
    u64* __restrict__ hyperedge_flags_double
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_hyperedges) return;

    int start = hyperedge_offsets[e];
    int end = hyperedge_offsets[e + 1];

    u64 f_all = 0ULL;
    u64 f_double = 0ULL;

    for (int i = start; i < end; i++) {
        int v = hyperedge_nodes[i];
        int p = partition[v];
        if (p >= 0 && p < 64) {
            u64 bit = 1ULL << p;
            if (f_all & bit) {
                f_double |= bit;
            } else {
                f_all |= bit;
            }
        }
    }

    hyperedge_flags_all[e] = f_all;
    hyperedge_flags_double[e] = f_double;
}

// 4. Hardware __popcll Single-Cycle Cut Connectivity Evaluation
extern "C" __global__ void ada_compute_cut_connectivity_10k(
    const int num_hyperedges,
    const u64* __restrict__ hyperedge_flags_all,
    int* __restrict__ cut_connectivity
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_hyperedges) return;

    u64 mask = hyperedge_flags_all[e];
    int parts_touched = (int)__popcll(mask);
    cut_connectivity[e] = (parts_touched > 1) ? (parts_touched - 1) : 0;
}

// 5. Warp-Cooperative Register-Resident FM Gain Evaluation (lmem: 0)
// 1 warp (32 threads) processes 1 node
// Lane i evaluates partition i (0..31) and partition i + 32 (32..63) in registers
extern "C" __global__ void ada_find_best_moves_10k(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int* __restrict__ node_offsets,
    const int* __restrict__ node_hyperedges,
    const u64* __restrict__ hyperedge_flags_all,
    const u64* __restrict__ hyperedge_flags_double,
    const int* __restrict__ partition,
    const int* __restrict__ part_sizes,
    int* __restrict__ move_priorities
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_nodes) return;

    int v = warp_id;
    int curr_part = partition[v];
    if (curr_part < 0 || curr_part >= num_parts) return;

    int start = node_offsets[v];
    int end = node_offsets[v + 1];
    int degree = end - start;
    if (degree <= 0) return;

    u64 curr_bit = 1ULL << curr_part;
    
    // Each thread tracks touch & double counts for partition 'lane' and 'lane + 32'
    int count_low = 0;
    int count_high = 0;
    int double_low = 0;
    int double_high = 0;
    int count_curr_present = 0;

    for (int i = start; i < end; i++) {
        int e = node_hyperedges[i];
        u64 f_all = hyperedge_flags_all[e];
        u64 f_double = hyperedge_flags_double[e];

        // If node leaves curr_part, edge still touches curr_part iff f_double has curr_bit
        if (f_double & curr_bit) {
            count_curr_present++;
        }

        // Target touch counts
        if (f_all & (1ULL << lane)) {
            count_low++;
        }
        if (f_all & (1ULL << (lane + 32))) {
            count_high++;
        }
        if (f_double & (1ULL << lane)) {
            double_low++;
        }
        if (f_double & (1ULL << (lane + 32))) {
            double_high++;
        }
    }

    int curr_size = part_sizes[curr_part];
    int gain_low = -999999;
    int gain_high = -999999;

    int part_low = lane;
    int part_high = lane + 32;

    if (part_low < num_parts && part_low != curr_part) {
        int tgt_size = part_sizes[part_low];
        if (tgt_size < max_part_size && curr_size > 1) {
            int balance_bonus = (curr_size > tgt_size + 1) ? 4 : 0;
            gain_low = (count_low - count_curr_present) + balance_bonus;
        }
    }

    if (part_high < num_parts && part_high != curr_part) {
        int tgt_size = part_sizes[part_high];
        if (tgt_size < max_part_size && curr_size > 1) {
            int balance_bonus = (curr_size > tgt_size + 1) ? 4 : 0;
            gain_high = (count_high - count_curr_present) + balance_bonus;
        }
    }

    int my_best_gain = (gain_high > gain_low) ? gain_high : gain_low;
    int my_best_target = (gain_high > gain_low) ? part_high : part_low;
    int my_best_double = (gain_high > gain_low) ? double_high : double_low;

    // Warp reduction to find the best move among all 64 partitions
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other_gain = __shfl_down_sync(0xFFFFFFFF, my_best_gain, offset);
        int other_target = __shfl_down_sync(0xFFFFFFFF, my_best_target, offset);
        int other_double = __shfl_down_sync(0xFFFFFFFF, my_best_double, offset);
        
        bool better = (other_gain > my_best_gain);
        if (!better && other_gain == my_best_gain && other_target >= 0) {
            if (other_double > my_best_double) {
                better = true;
            } else if (other_double == my_best_double) {
                better = (other_target < my_best_target);
            }
        }
        if (better) {
            my_best_gain = other_gain;
            my_best_target = other_target;
            my_best_double = other_double;
        }
    }

    if (lane == 0) {
        if (my_best_gain >= -1 && my_best_target >= 0 && my_best_target != curr_part) {
            int bg = my_best_gain + 1000;
            if (bg > 32767) bg = 32767;
            if (bg < 0) bg = 0;
            int degree_weight = degree > 255 ? 255 : degree;
            move_priorities[v] = (bg << 16) | (degree_weight << 8) | (my_best_target & 63);
        } else {
            move_priorities[v] = 0;
        }
    }
}

// 6. Direct Move Execution for Deterministic Move Buffers (Zero Atomics)
extern "C" __global__ void ada_execute_moves_10k(
    const int num_moves,
    const int* __restrict__ accepted_move_nodes,
    const int* __restrict__ accepted_move_targets,
    int* __restrict__ partition
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_moves) return;

    int v = accepted_move_nodes[idx];
    int tgt = accepted_move_targets[idx];
    partition[v] = tgt;
}
