use pokedr_core::{
    cards::{Board, Card, Rank, Suit},
    dense_cfr::{
        CfrVariant, DenseCfrConfig, DenseCfrIteration, DenseCfrSolver, DenseCfrState,
        gpu::{GpuCfrError, GpuDenseCfrBackend},
    },
    postflop::{ActionSetConfig, Player, PublicState, Street, SubgameTree, SubgameTreeConfig},
    postflop_dense::PostflopDenseLayout,
};

fn main() {
    let mut args = std::env::args();
    let program = args.next().unwrap_or_else(|| "pokedr".to_string());
    match args.next().as_deref() {
        Some("gpu-info") => print_gpu_info(),
        Some("gpu-smoke") => run_gpu_smoke(),
        Some("postflop-smoke") => run_postflop_smoke(),
        Some("rs-poker-smoke") => run_rs_poker_smoke(),
        Some("rs-poker-trace") => run_rs_poker_trace(),
        _ => {
            eprintln!(
                "usage: {program} <gpu-info|gpu-smoke|postflop-smoke|rs-poker-smoke|rs-poker-trace>"
            );
            std::process::exit(2);
        }
    }
}

fn print_gpu_info() {
    let backend = match GpuDenseCfrBackend::new() {
        Ok(backend) => backend,
        Err(GpuCfrError::NoAdapter) => {
            println!("no GPU adapter visible to wgpu");
            return;
        }
        Err(error) => {
            eprintln!("failed to initialize GPU backend: {error:?}");
            std::process::exit(1);
        }
    };
    let info = backend.adapter_info();
    println!("name: {}", info.name);
    println!("backend: {:?}", info.backend);
    println!("device_type: {:?}", info.device_type);
    println!("vendor: 0x{:x}", info.vendor);
    println!("device: 0x{:x}", info.device);
}

fn run_gpu_smoke() {
    println!("initializing GPU backend");
    let backend = match GpuDenseCfrBackend::new() {
        Ok(backend) => backend,
        Err(GpuCfrError::NoAdapter) => {
            eprintln!("no GPU adapter visible to wgpu");
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!("failed to initialize GPU backend: {error:?}");
            std::process::exit(1);
        }
    };
    let info = backend.adapter_info();
    println!("adapter: {} ({:?})", info.name, info.backend);

    println!("running one-shot update");
    run_one_shot_update(&backend);
    println!("one-shot update passed");

    println!("running resident updates");
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
        .unwrap_or_else(|error| {
            eprintln!("GPU one-shot update failed: {error:?}");
            std::process::exit(1);
        });
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
    gpu.run_iterations(&backend, 5, |iteration, batch| {
        println!("dispatch iteration {iteration}");
        fill_fixture_iteration(iteration, batch);
    })
    .unwrap_or_else(|error| {
        eprintln!("GPU update failed: {error:?}");
        std::process::exit(1);
    });

    println!("downloading result");
    let downloaded = gpu.download(&backend).unwrap_or_else(|error| {
        eprintln!("GPU download failed: {error:?}");
        std::process::exit(1);
    });
    assert_close("regret", cpu.state().regrets(), downloaded.regrets());
    assert_close(
        "strategy_sum",
        cpu.state().strategy_sum(),
        downloaded.strategy_sum(),
    );
}

fn run_postflop_smoke() {
    let backend = match GpuDenseCfrBackend::new() {
        Ok(backend) => backend,
        Err(GpuCfrError::NoAdapter) => {
            eprintln!("no GPU adapter visible to wgpu");
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!("failed to initialize GPU backend: {error:?}");
            std::process::exit(1);
        }
    };
    let info = backend.adapter_info();
    println!("adapter: {} ({:?})", info.name, info.backend);

    let tree = smoke_postflop_tree();
    let layout = PostflopDenseLayout::from_tree(&tree);
    let config = layout.dense_config(CfrVariant::CfrPlus);
    println!(
        "public nodes: {}, infosets: {}, max actions: {}",
        tree.nodes().len(),
        layout.infoset_count(),
        layout.max_actions()
    );

    let mut solver =
        backend.resident_solver_with_legal_actions(config.clone(), layout.legal_actions().to_vec());
    solver
        .run_iterations(&backend, 8, |iteration, batch| {
            fill_postflop_fixture_iteration(iteration, &layout, batch);
        })
        .unwrap_or_else(|error| {
            eprintln!("postflop GPU update failed: {error:?}");
            std::process::exit(1);
        });
    let state = solver.download(&backend).unwrap_or_else(|error| {
        eprintln!("postflop GPU download failed: {error:?}");
        std::process::exit(1);
    });
    assert!(
        state.regrets().iter().all(|value| value.is_finite()),
        "regrets must stay finite"
    );
    assert!(
        state.strategy_sum().iter().all(|value| value.is_finite()),
        "strategy sums must stay finite"
    );
    println!("postflop GPU CFR smoke passed");
}

fn run_rs_poker_smoke() {
    let summary = pokedr_agent::run_heads_up_match(smoke_hands(), 7);
    println!("hands: {}", summary.hands);
    println!("hero_net: {:.2}", summary.hero_net);
    println!("villain_net: {:.2}", summary.villain_net);
}

fn run_rs_poker_trace() {
    for trace in pokedr_agent::run_traced_heads_up_match(smoke_hands(), 7) {
        println!(
            "hand {} hero [{}] villain [{}] board [{}] hero_net {:.2} villain_net {:.2}",
            trace.hand_index,
            trace.hero_cards,
            trace.villain_cards,
            trace.board,
            trace.hero_net,
            trace.villain_net
        );
        for action in &trace.actions {
            println!("  {action}");
        }
        for award in &trace.awards {
            println!("  {award}");
        }
    }
}

fn smoke_hands() -> usize {
    std::env::var("POKEDR_HANDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(16)
}

fn smoke_postflop_tree() -> SubgameTree {
    SubgameTree::build(
        PublicState {
            street: Street::Flop,
            board: Board::new(vec![
                Card::new(Rank::Ace, Suit::Spades),
                Card::new(Rank::Seven, Suit::Hearts),
                Card::new(Rank::Two, Suit::Clubs),
            ]),
            pot: 100,
            hero_invested: 50,
            villain_invested: 50,
            effective_stack: 300,
            to_call: 0,
            min_aggressive_amount: 50,
            acting_player: Player::Hero,
            raises_this_street: 0,
            checks_this_street: 0,
        },
        SubgameTreeConfig {
            action_set: ActionSetConfig {
                max_aggressive_actions: 2,
                flop_bet_fractions: vec![0.5],
                turn_bet_fractions: vec![0.5],
                river_bet_fractions: vec![0.5],
                raise_fractions: vec![1.0],
                ..ActionSetConfig::default()
            },
            max_raises_per_street: 1,
            max_depth: 5,
        },
    )
}

fn fill_postflop_fixture_iteration(
    iteration: usize,
    layout: &PostflopDenseLayout,
    batch: &mut DenseCfrIteration,
) {
    batch.action_values.fill(0.0);
    batch.reach_weights.fill(1.0);
    batch.strategy_weights.fill(1.0);
    for infoset in 0..layout.infoset_count() {
        let offset = infoset * layout.max_actions();
        for action in 0..layout.action_count(infoset) {
            batch.action_values[offset + action] =
                ((infoset as f32 * 0.17) + (action as f32 * 0.31) + iteration as f32).sin();
        }
    }
}

fn fill_fixture_iteration(iteration: usize, batch: &mut DenseCfrIteration) {
    for (index, value) in batch.action_values.iter_mut().enumerate() {
        *value = ((index as f32 + iteration as f32) * 0.25).sin();
    }
    batch.reach_weights.fill(1.0);
    batch.strategy_weights.fill(0.75);
}

fn fill_fixture_iteration_with_state(
    iteration: usize,
    _state: &DenseCfrState,
    batch: &mut DenseCfrIteration,
) {
    fill_fixture_iteration(iteration, batch);
}

fn assert_close(label: &str, expected: &[f32], actual: &[f32]) {
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        if (expected - actual).abs() >= 1e-5 {
            eprintln!("{label}[{index}] mismatch: expected {expected}, actual {actual}");
            std::process::exit(1);
        }
    }
}
