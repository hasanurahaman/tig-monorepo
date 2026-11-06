use anyhow::Result;
use cudarc::driver::CudaContext;
use cudarc::nvrtc::Ptx;
use cudarc::runtime::result::device::get_device_prop;
use std::fs;
use std::sync::Mutex;
use std::time::Instant;
use tig_challenges::neuralnet_optimizer::{Challenge, Solution, Track};

fn run_track(track_name: &str, n_hidden: usize, ctx: &std::sync::Arc<CudaContext>, stream: &std::sync::Arc<cudarc::driver::CudaStream>, prop: &cudarc::runtime::sys::cudaDeviceProp) -> Result<()> {
    println!("\n================================================================");
    println!("  TRACK: {} (n_hidden = {})", track_name, n_hidden);
    println!("================================================================");

    let track = Track { n_hidden };
    let seed = [42u8; 32];
    let pa_ptx_path = "tig-algorithms/lib/neuralnet_optimizer/ptx/parallax_arcturus.ptx";
    let pa_ptx_src = fs::read_to_string(pa_ptx_path)?;
    let pa_module = ctx.load_module(Ptx::from_src(pa_ptx_src))?;

    let t0 = Instant::now();
    let challenge = Challenge::generate_instance(&seed, &track, pa_module.clone(), stream.clone(), prop)?;
    println!("[+] Instance generated in {:.2?}:", t0.elapsed());
    println!("    - Hidden Layers:   {}", challenge.num_hidden_layers);
    println!("    - Layer Dims:      {:?}", challenge.layer_dims());

    // 1. hyper_optim_v1 (Ours) - Run first!
    let hyper_ptx_path = "tig-algorithms/lib/neuralnet_optimizer/ptx/hyper_optim_v1.ptx";
    let hyper_ptx_src = fs::read_to_string(hyper_ptx_path)?;
    let hyper_module = ctx.load_module(Ptx::from_src(hyper_ptx_src))?;

    let saved_sol_hyper: Mutex<Option<Solution>> = Mutex::new(None);
    let t_hyper = Instant::now();
    tig_algorithms::neuralnet_optimizer::hyper_optim_v1::solve_challenge(
        &challenge,
        &|sol| { *saved_sol_hyper.lock().unwrap() = Some(sol.clone()); Ok(()) },
        &None,
        hyper_module.clone(),
        stream.clone(),
        prop,
    )?;
    let hyper_time = t_hyper.elapsed();
    let final_hyper = saved_sol_hyper.into_inner().unwrap().unwrap();
    let quality_hyper = challenge.evaluate_solution(&final_hyper, hyper_module.clone(), stream.clone(), prop)?;
    println!("  [1] hyper_optim_v1 (Ours): {:>8.2?} ({} ms) | Epochs: {:>3} | Quality: {}",
        hyper_time, hyper_time.as_millis(), final_hyper.epochs_used, quality_hyper);

    // 2. parallax_arcturus
    let saved_sol_pa: Mutex<Option<Solution>> = Mutex::new(None);
    let t_pa = Instant::now();
    tig_algorithms::neuralnet_optimizer::parallax_arcturus::solve_challenge(
        &challenge,
        &|sol| { *saved_sol_pa.lock().unwrap() = Some(sol.clone()); Ok(()) },
        &None,
        pa_module.clone(),
        stream.clone(),
        prop,
    )?;
    let pa_time = t_pa.elapsed();
    let final_pa = saved_sol_pa.into_inner().unwrap().unwrap();
    let quality_pa = challenge.evaluate_solution(&final_pa, pa_module.clone(), stream.clone(), prop)?;
    println!("  [2] parallax_arcturus:     {:>8.2?} ({} ms) | Epochs: {:>3} | Quality: {}",
        pa_time, pa_time.as_millis(), final_pa.epochs_used, quality_pa);

    // 3. dc_steer_v8
    let dc_ptx_path = "tig-algorithms/lib/neuralnet_optimizer/ptx/dc_steer_v8.ptx";
    let dc_ptx_src = fs::read_to_string(dc_ptx_path)?;
    let dc_module = ctx.load_module(Ptx::from_src(dc_ptx_src))?;

    let saved_sol_dc: Mutex<Option<Solution>> = Mutex::new(None);
    let t_dc = Instant::now();
    tig_algorithms::neuralnet_optimizer::dc_steer_v8::solve_challenge(
        &challenge,
        &|sol| { *saved_sol_dc.lock().unwrap() = Some(sol.clone()); Ok(()) },
        &None,
        dc_module.clone(),
        stream.clone(),
        prop,
    )?;
    let dc_time = t_dc.elapsed();
    let final_dc = saved_sol_dc.into_inner().unwrap().unwrap();
    let quality_dc = challenge.evaluate_solution(&final_dc, dc_module.clone(), stream.clone(), prop)?;
    println!("  [3] dc_steer_v8:           {:>8.2?} ({} ms) | Epochs: {:>3} | Quality: {}",
        dc_time, dc_time.as_millis(), final_dc.epochs_used, quality_dc);

    let speedup_vs_dc = (dc_time.as_secs_f64() / hyper_time.as_secs_f64());
    let speedup_vs_pa = (pa_time.as_secs_f64() / hyper_time.as_secs_f64());
    println!("  >>> hyper_optim_v1 Speedup: {:.2}x vs dc_steer_v8, {:.2}x vs parallax_arcturus",
        speedup_vs_dc, speedup_vs_pa);

    Ok(())
}

fn main() -> Result<()> {
    println!("================================================================");
    println!("  TIG c006 NeuralNet Optimizer: Full SOTA Showdown");
    println!("  Target Hardware: NVIDIA GeForce RTX 4070 Laptop GPU (sm_89)");
    println!("================================================================");

    let dev_id = 0;
    let ctx = CudaContext::new(dev_id)?;
    ctx.set_blocking_synchronize()?;
    let prop = get_device_prop(dev_id as i32)?;
    let stream = ctx.default_stream();
    println!("[+] Initialized CUDA Device {}: RTX 4070 ({} SMs, {} MB L2)",
        dev_id, prop.multiProcessorCount, prop.l2CacheSize / (1024 * 1024));

    // Test on shallow track (n_hidden = 4) and deep track (n_hidden = 10)
    run_track("t29", 4, &ctx, &stream, &prop)?;
    run_track("t26", 10, &ctx, &stream, &prop)?;

    Ok(())
}
