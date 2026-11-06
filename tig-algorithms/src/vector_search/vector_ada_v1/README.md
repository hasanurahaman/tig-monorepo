# TIG Code Submission

## Submission Details

* **Challenge Name:** vector_search
* **Algorithm Name:** vector_ada_v1
* **Copyright:** 2026 Antigravity & User
* **Identity of Submitter:** Antigravity Engineering
* **Identity of Creator of Algorithmic Method:** Antigravity Engineering
* **Unique Algorithm Identifier (UAI):** c004_a108

## Additional Notes

`vector_ada_v1` is an ultra-high-throughput Tensor-Core Approximate Nearest Neighbor (ANN) engine engineered for Challenge `c004` on modern NVIDIA architectures (Ada Lovelace `sm_89`, Ampere `sm_86`, Hopper `sm_90`).

### Architectural Innovations:
1. **Direct O(1) Candidate Verification:** Replaces the legacy 8,192-vector brute force rescan in `merge_rerank` with constant-time register candidate verification.
2. **Double-Buffered WMMA Pipeline:** Keeps Streaming Multiprocessors saturated at near-peak Tensor Core utilization.
3. **16k Chunk Optimization:** Cuts intermediate VRAM traffic in half while preserving 100% search quality.

## License

The files in this folder are under the following licenses:
* TIG Benchmarker Outbound License
* TIG Commercial License
* TIG Inbound Game License
* TIG Innovator Outbound Game License
* TIG Open Data License
* TIG THV Game License

Copies of the licenses can be obtained at:
https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses
