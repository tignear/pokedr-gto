use pokedr_core::dense_cfr::{
    CfrVariant, DenseCfrConfig, DenseCfrSolver, DenseCfrState,
    gpu::{GpuCfrError, GpuDenseCfrBackend},
};

fn main() {
    let backend = match GpuDenseCfrBackend::new() {
        Ok(backend) => backend,
        Err(GpuCfrError::NoAdapter) => {
            println!("skipping GPU smoke: no GPU adapter visible to wgpu");
            return;
        }
        Err(error) => {
            eprintln!("failed to initialize GPU backend: {error:?}");
            std::process::exit(1);
        }
    };

    let info = backend.adapter_info();
    println!("adapter: {} ({:?})", info.name, info.backend);

    run_one_shot_update(&backend);
    run_resident_updates(&backend);
    println!("GPU smoke passed");
}

fn run_one_shot_update(backend: &GpuDenseCfrBackend) {
    let config = DenseCfrConfig {
        infosets: 4,
        actions: 3,
        variant: CfrVariant::CfrPlus,
    };
    let mut cpu = DenseCfrState::new(config.clone());
    let mut gpu = DenseCfrState::new(config);
    let action_values = [
        1.0, -0.5, 0.25, -1.0, 2.0, 0.0, 0.5, 0.25, -0.75, 3.0, 1.0, -2.0,
    ];
    let reach_weights = [1.0, 0.5, 2.0, 0.25];
    let strategy_weights = [1.0, 1.0, 0.5, 2.0];

    cpu.update_all_infosets(&action_values, &reach_weights, &strategy_weights, 1);
    backend
        .update_all_infosets(
            &mut gpu,
            &action_values,
            &reach_weights,
            &strategy_weights,
            1,
        )
        .unwrap_or_else(|error| fail(&format!("GPU one-shot update failed: {error:?}")));

    assert_close("one-shot regret", cpu.regrets(), gpu.regrets());
    assert_close(
        "one-shot strategy_sum",
        cpu.strategy_sum(),
        gpu.strategy_sum(),
    );
}

fn run_resident_updates(backend: &GpuDenseCfrBackend) {
    let config = DenseCfrConfig {
        infosets: 8,
        actions: 4,
        variant: CfrVariant::Discounted,
    };
    let mut cpu = DenseCfrSolver::new(config.clone());
    let mut gpu = backend.resident_solver(config);

    cpu.run_iterations(5, fill_fixture_iteration_with_state);
    gpu.run_iterations(backend, 5, fill_fixture_iteration)
        .unwrap_or_else(|error| fail(&format!("GPU resident update failed: {error:?}")));

    let downloaded = gpu
        .download(backend)
        .unwrap_or_else(|error| fail(&format!("GPU download failed: {error:?}")));
    assert_close(
        "resident regret",
        cpu.state().regrets(),
        downloaded.regrets(),
    );
    assert_close(
        "resident strategy_sum",
        cpu.state().strategy_sum(),
        downloaded.strategy_sum(),
    );
}

fn fill_fixture_iteration(iteration: usize, batch: &mut pokedr_core::dense_cfr::DenseCfrIteration) {
    for (index, value) in batch.action_values.iter_mut().enumerate() {
        *value = ((index as f32 + iteration as f32) * 0.25).sin();
    }
    batch.reach_weights.fill(1.0);
    batch.strategy_weights.fill(0.75);
}

fn fill_fixture_iteration_with_state(
    iteration: usize,
    _state: &pokedr_core::dense_cfr::DenseCfrState,
    batch: &mut pokedr_core::dense_cfr::DenseCfrIteration,
) {
    fill_fixture_iteration(iteration, batch);
}

fn assert_close(label: &str, expected: &[f32], actual: &[f32]) {
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        if (expected - actual).abs() >= 1e-5 {
            fail(&format!(
                "{label}[{index}] mismatch: expected {expected}, actual {actual}"
            ));
        }
    }
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}
