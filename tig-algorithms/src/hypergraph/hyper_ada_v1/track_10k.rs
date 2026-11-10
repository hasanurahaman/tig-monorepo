// track_10k.rs - hyper_ada_v1
// Challenge c005 (hypergraph) track 10k
// Zero local memory spills (lmem: 0)
// Deterministic FM gain refinement + Tabu Search + 100% validity guarantees

use anyhow::Result;
use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::{Challenge, Solution};

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> Result<()> {
    let num_nodes = challenge.num_nodes as usize;
    let num_hyperedges = challenge.num_hyperedges as usize;
    let num_parts = challenge.num_parts as usize;
    let max_part_size = challenge.max_part_size as usize;

    let rounds = hyperparameters
        .as_ref()
        .and_then(|p| p.get("rounds").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 5000) as usize)
        .unwrap_or(800);

    let tabu_tenure = 12usize;
    let max_moves_per_round = 512usize;

    // 1. Allocate device buffers
    let mut d_partition = stream.alloc_zeros::<i32>(num_nodes)?;
    let mut d_part_sizes = stream.alloc_zeros::<i32>(num_parts)?;
    let mut d_hyperedge_flags_all = stream.alloc_zeros::<u64>(num_hyperedges)?;
    let mut d_hyperedge_flags_double = stream.alloc_zeros::<u64>(num_hyperedges)?;
    let mut d_hyperedge_clusters = stream.alloc_zeros::<i32>(num_hyperedges)?;
    let mut d_pref_parts = stream.alloc_zeros::<i32>(num_nodes)?;
    let mut d_pref_priorities = stream.alloc_zeros::<i32>(num_nodes)?;
    let mut d_move_priorities = stream.alloc_zeros::<i32>(num_nodes)?;
    let mut d_accepted_nodes = stream.alloc_zeros::<i32>(num_nodes)?;
    let mut d_accepted_targets = stream.alloc_zeros::<i32>(num_nodes)?;

    // 2. Load CUDA kernels
    let cluster_k = module.load_function("ada_hyperedge_clustering_10k")?;
    let prefs_k = module.load_function("ada_compute_initial_preferences_10k")?;
    let compute_masks_k = module.load_function("ada_compute_hyperedge_masks_10k")?;
    let find_moves_k = module.load_function("ada_find_best_moves_10k")?;
    let exec_moves_k = module.load_function("ada_execute_moves_10k")?;

    let block_size = 256u32;
    let hedge_cfg = LaunchConfig {
        grid_dim: (((challenge.num_hyperedges + block_size - 1) / block_size), 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // 32 threads per warp evaluate 1 node with zero local memory spill (8 warps per 256-thread block)
    let warps_per_block = 8u32;
    let warp_node_cfg = LaunchConfig {
        grid_dim: (((challenge.num_nodes + warps_per_block - 1) / warps_per_block), 1, 1),
        block_dim: (warps_per_block * 32, 1, 1),
        shared_mem_bytes: 0,
    };

    // 3. Stage 1: Structure-aware Anchor Hyperedge Clustering
    let num_clusters = 64i32;
    unsafe {
        stream.launch_builder(&cluster_k)
            .arg(&challenge.num_hyperedges)
            .arg(&num_clusters)
            .arg(&challenge.d_hyperedge_offsets)
            .arg(&challenge.d_hyperedge_nodes)
            .arg(&mut d_hyperedge_clusters)
            .launch(hedge_cfg.clone())?;
    }

    // Stage 1b: Warp-cooperative Node Preference Voting (lmem: 0)
    unsafe {
        stream.launch_builder(&prefs_k)
            .arg(&challenge.num_nodes)
            .arg(&challenge.num_parts)
            .arg(&challenge.d_node_offsets)
            .arg(&challenge.d_node_hyperedges)
            .arg(&challenge.d_hyperedge_offsets)
            .arg(&d_hyperedge_clusters)
            .arg(&mut d_pref_parts)
            .arg(&mut d_pref_priorities)
            .launch(warp_node_cfg.clone())?;
    }

    // 4. Host priority sorting & initial capacity-aware placement
    let pref_parts = stream.memcpy_dtov(&d_pref_parts)?;
    let pref_priorities = stream.memcpy_dtov(&d_pref_priorities)?;

    let mut indices: Vec<usize> = (0..num_nodes).collect();
    indices.sort_unstable_by(|&a, &b| {
        pref_priorities[b].cmp(&pref_priorities[a]).then_with(|| a.cmp(&b))
    });

    let mut part_host = vec![0i32; num_nodes];
    let mut part_sizes_host = vec![0i32; num_parts];

    // Seed 1 node per partition first to guarantee zero empty partitions
    for i in 0..num_nodes {
        let node = indices[i];
        let preferred_part = pref_parts[node];
        let start_part = if i < num_parts {
            i as i32
        } else {
            preferred_part
        };

        let mut assigned = false;
        for attempt in 0..num_parts {
            let try_part = ((start_part as usize) + attempt) % num_parts;
            if (part_sizes_host[try_part] as usize) < max_part_size {
                part_host[node] = try_part as i32;
                part_sizes_host[try_part] += 1;
                assigned = true;
                break;
            }
        }
        if !assigned {
            let fallback_part = node % num_parts;
            part_host[node] = fallback_part as i32;
            part_sizes_host[fallback_part] += 1;
        }
    }

    d_partition = stream.memcpy_stod(&part_host)?;
    d_part_sizes = stream.memcpy_stod(&part_sizes_host)?;

    let mut move_keys_host = vec![0i32; num_nodes];
    let mut valid_moves: Vec<(usize, i32)> = Vec::with_capacity(num_nodes);
    let mut accepted_nodes: Vec<i32> = Vec::with_capacity(num_nodes);
    let mut accepted_targets: Vec<i32> = Vec::with_capacity(num_nodes);
    let mut node_tabu_until = vec![0usize; num_nodes];
    let mut tgt_used = vec![0usize; num_parts];

    // 5. Stage 2: Dual-Bitmask Accelerated Deterministic FM Refinement with Tabu Search
    for round in 0..rounds {
        unsafe {
            stream.launch_builder(&compute_masks_k)
                .arg(&challenge.num_hyperedges)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&d_partition)
                .arg(&mut d_hyperedge_flags_all)
                .arg(&mut d_hyperedge_flags_double)
                .launch(hedge_cfg.clone())?;

            stream.launch_builder(&find_moves_k)
                .arg(&challenge.num_nodes)
                .arg(&challenge.num_parts)
                .arg(&challenge.max_part_size)
                .arg(&challenge.d_node_offsets)
                .arg(&challenge.d_node_hyperedges)
                .arg(&d_hyperedge_flags_all)
                .arg(&d_hyperedge_flags_double)
                .arg(&d_partition)
                .arg(&d_part_sizes)
                .arg(&mut d_move_priorities)
                .launch(warp_node_cfg.clone())?;
        }

        stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;

        valid_moves.clear();
        for (node, &key) in move_keys_host.iter().enumerate() {
            if key > 0 {
                let gain = (key >> 16) - 1000;
                if gain >= 0 && (node_tabu_until[node] <= round || gain >= 2) {
                    valid_moves.push((node, key));
                }
            }
        }

        if valid_moves.is_empty() {
            break;
        }

        // Deterministic sort: highest gain first, then degree, then node ID
        valid_moves.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        accepted_nodes.clear();
        accepted_targets.clear();
        tgt_used.fill(0);

        let quota_per_part = if round < 100 { 16usize } else { 8usize };

        for &(node, key) in valid_moves.iter() {
            if accepted_nodes.len() >= max_moves_per_round {
                break;
            }
            let target = (key & 63) as usize;
            let current = part_host[node] as usize;
            if (part_sizes_host[target] as usize) < max_part_size
                && part_sizes_host[current] > 1
                && tgt_used[target] < quota_per_part
            {
                part_host[node] = target as i32;
                part_sizes_host[current] -= 1;
                part_sizes_host[target] += 1;
                tgt_used[target] += 1;
                node_tabu_until[node] = round + tabu_tenure;
                accepted_nodes.push(node as i32);
                accepted_targets.push(target as i32);
            }
        }

        let num_accepted = accepted_nodes.len();
        if num_accepted > 0 {
            stream.memcpy_htod(&accepted_nodes, &mut d_accepted_nodes)?;
            stream.memcpy_htod(&accepted_targets, &mut d_accepted_targets)?;
            stream.memcpy_htod(&part_sizes_host, &mut d_part_sizes)?;

            let moves_cfg = LaunchConfig {
                grid_dim: (((num_accepted as u32 + block_size - 1) / block_size), 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                stream.launch_builder(&exec_moves_k)
                    .arg(&(num_accepted as i32))
                    .arg(&d_accepted_nodes)
                    .arg(&d_accepted_targets)
                    .arg(&mut d_partition)
                    .launch(moves_cfg)?;
            }
        } else {
            break;
        }
    }

    stream.synchronize()?;

    // 6. Host-side Feasibility and Determinism Verification
    enforce_feasibility(&mut part_host, num_parts, max_part_size);

    let final_part: Vec<u32> = part_host
        .into_iter()
        .map(|x| (x.max(0) as u32).min((challenge.num_parts - 1) as u32))
        .collect();

    save_solution(&Solution { partition: final_part })?;
    Ok(())
}

/// Strictly enforces challenge constraints deterministically:
/// 1. Every part has at least 1 node (zero empty partitions).
/// 2. No part exceeds max_part_size nodes.
fn enforce_feasibility(
    partition: &mut [i32],
    num_parts: usize,
    max_part_size: usize,
) {
    let mut part_sizes = vec![0usize; num_parts];
    for &p in partition.iter() {
        let idx = (p as usize).min(num_parts - 1);
        part_sizes[idx] += 1;
    }

    // 1. Fix any empty partitions deterministically
    for p in 0..num_parts {
        if part_sizes[p] == 0 {
            if let Some(donor) = (0..num_parts).filter(|&d| part_sizes[d] > 1).max_by_key(|&d| part_sizes[d]) {
                for v in 0..partition.len() {
                    if partition[v] as usize == donor {
                        partition[v] = p as i32;
                        part_sizes[donor] -= 1;
                        part_sizes[p] += 1;
                        break;
                    }
                }
            }
        }
    }

    // 2. Fix any over-capacity partitions deterministically
    for p in 0..num_parts {
        while part_sizes[p] > max_part_size {
            if let Some(target) = (0..num_parts).filter(|&t| part_sizes[t] < max_part_size).min_by_key(|&t| part_sizes[t]) {
                for v in (0..partition.len()).rev() {
                    if partition[v] as usize == p {
                        partition[v] = target as i32;
                        part_sizes[p] -= 1;
                        part_sizes[target] += 1;
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }
}
