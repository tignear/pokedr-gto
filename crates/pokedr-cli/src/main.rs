use pokedr_core::{
    cards::{Board, Card, Rank, Suit},
    dense_cfr::{
        CfrVariant, DenseCfrConfig, DenseCfrIteration, DenseCfrSolver, DenseCfrState,
        gpu::{GpuCfrError, GpuDenseCfrBackend},
    },
    postflop::{ActionSetConfig, Player, PublicState, Street, SubgameTree, SubgameTreeConfig},
    postflop_dense::PostflopDenseLayout,
};
use rusqlite::{Connection, params};
use serde::Deserialize;
use serde_json::{Value, json};

fn main() {
    let mut args = std::env::args();
    let program = args.next().unwrap_or_else(|| "pokedr".to_string());
    match args.next().as_deref() {
        Some("gpu-info") => print_gpu_info(),
        Some("gpu-smoke") => run_gpu_smoke(),
        Some("postflop-smoke") => run_postflop_smoke(),
        Some("solve-flop") => run_solve_flop(parse_flop_command_args(args)),
        Some("solve-flop-metrics") => run_solve_flop_metrics(parse_flop_command_args(args)),
        Some("solve-flop-sweep") => run_solve_flop_sweep(parse_flop_command_args(args)),
        Some("tree-db") => run_tree_db(args),
        Some("rs-poker-smoke") => run_rs_poker_smoke(),
        Some("rs-poker-trace") => run_rs_poker_trace(),
        _ => {
            eprintln!(
                "usage: {program} <gpu-info|gpu-smoke|postflop-smoke|solve-flop|solve-flop-metrics|solve-flop-sweep|tree-db|rs-poker-smoke|rs-poker-trace> ..."
            );
            std::process::exit(2);
        }
    }
}

#[derive(Debug, Clone, Default)]
struct FlopCommandArgs {
    flop: Option<String>,
    config_path: Option<String>,
}

#[derive(Debug, Clone)]
struct TreeDbBuildArgs {
    db_path: String,
    flop: Option<String>,
    config_path: Option<String>,
    combo_limit: usize,
    iterations: Option<usize>,
    max_depth: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct CliConfigFile {
    cfr_iterations: Option<usize>,
    cfr_variant: Option<String>,
    max_depth: Option<usize>,
    max_raises_per_street: Option<u8>,
    max_showdown_runouts: Option<usize>,
    metric_iterations: Option<Vec<usize>>,
    action_set: Option<CliActionSetConfig>,
    dcfr_alpha: Option<f32>,
    dcfr_gamma: Option<f32>,
    dcfr_schedule_alpha_start: Option<f32>,
    dcfr_schedule_alpha_end: Option<f32>,
    dcfr_schedule_gamma_start: Option<f32>,
    dcfr_schedule_gamma_end: Option<f32>,
    dcfr_schedule_horizon: Option<usize>,
    pdcfr_alpha: Option<f32>,
    pdcfr_gamma: Option<f32>,
    pdcfr_eta: Option<f32>,
    pdcfr_eta_start: Option<f32>,
    pdcfr_eta_end: Option<f32>,
    pdcfr_eta_horizon: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct CliActionSetConfig {
    max_aggressive_actions: Option<usize>,
    merge_log_ratio: Option<f32>,
    all_in_threshold: Option<f32>,
    flop_bet_fractions: Option<Vec<f32>>,
    turn_bet_fractions: Option<Vec<f32>>,
    river_bet_fractions: Option<Vec<f32>>,
    raise_fractions: Option<Vec<f32>>,
}

fn print_gpu_info() {
    let (info, supports_shader_float32_atomic) = match GpuDenseCfrBackend::probe_adapter() {
        Ok(probe) => probe,
        Err(GpuCfrError::NoAdapter) => {
            println!("no GPU adapter visible to wgpu");
            return;
        }
        Err(error) => {
            eprintln!("failed to initialize GPU backend: {error:?}");
            std::process::exit(1);
        }
    };
    println!("name: {}", info.name);
    println!("backend: {:?}", info.backend);
    println!("device_type: {:?}", info.device_type);
    println!("vendor: 0x{:x}", info.vendor);
    println!("device: 0x{:x}", info.device);
    println!("shader_float32_atomic: {}", supports_shader_float32_atomic);
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
    let cli_config = load_cli_config(None);
    let summary = pokedr_agent::run_heads_up_match_with_config(
        smoke_hands(),
        7,
        match_config(cli_config.as_ref()),
    );
    println!("hands: {}", summary.hands);
    println!("hero_net: {:.2}", summary.hero_net);
    println!("villain_net: {:.2}", summary.villain_net);
}

fn run_rs_poker_trace() {
    let cli_config = load_cli_config(None);
    for trace in pokedr_agent::run_traced_heads_up_match_with_config(
        smoke_hands(),
        7,
        match_config(cli_config.as_ref()),
    ) {
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

fn run_solve_flop(args: FlopCommandArgs) {
    let cli_config = load_cli_config(args.config_path.as_deref());
    let flop = parse_flop(args.flop.as_deref().unwrap_or("As7h2c")).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let config = fixed_flop_config(cli_config.as_ref());
    println!(
        "solving fixed flop iterations={} variant={:?} depth={} equity_runout_cap={} terminal_runouts=full",
        config.cfr_iterations, config.cfr_variant, config.max_depth, config.max_showdown_runouts
    );
    let summary = pokedr_agent::solve_fixed_flop_once(flop, config);
    println!("board: {}", summary.board);
    println!("iterations: {}", summary.iterations);
    println!("public_decisions: {}", summary.decisions);
    println!("chance_nodes: {}", summary.chance);
    println!("terminals: {}", summary.terminals);
    println!("public_infosets: {}", summary.public_infosets);
    println!("private_infosets: {}", summary.private_infosets);
    println!("max_actions: {}", summary.max_actions);
    println!("elapsed: {:.2}s", summary.elapsed_secs);
}

fn run_solve_flop_metrics(args: FlopCommandArgs) {
    const PIO_STYLE_TARGET_BB100: f32 = 1.0;

    let cli_config = load_cli_config(args.config_path.as_deref());
    let flop = parse_flop(args.flop.as_deref().unwrap_or("As7h2c")).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let config = match_config(cli_config.as_ref());
    let convergence = metric_convergence_settings();
    let iterations = convergence
        .as_ref()
        .map(|settings| metric_convergence_iterations(settings))
        .unwrap_or_else(|| metric_iterations(cli_config.as_ref()));
    let target_bb100 = convergence
        .as_ref()
        .map(|settings| settings.target_bb100)
        .unwrap_or(PIO_STYLE_TARGET_BB100);
    println!(
        "solving fixed flop metrics variant={:?} depth={} equity_runout_cap={} terminal_runouts=full iterations={:?} target_bb100={:.2}",
        config.cfr_variant, config.max_depth, config.max_showdown_runouts, iterations, target_bb100
    );
    let dump_gap_nodes = std::env::var_os("POKEDR_METRIC_LOCAL_GAPS").is_some();
    let mut best_exploitability_bb100 = f32::INFINITY;
    let mut regression_count = 0usize;
    let patience = convergence
        .as_ref()
        .map(|settings| settings.regression_patience)
        .unwrap_or(usize::MAX);
    let gap_node_config = config.clone();
    pokedr_agent::solve_fixed_flop_metrics_with_callback(flop, config, &iterations, |row| {
        print_metric_row(row, target_bb100);
        if dump_gap_nodes {
            print_metric_gap_nodes(flop, gap_node_config.clone(), row);
        }
        if !row.finite
            || row.current_strategy_norm_error > 1.0e-3
            || row.average_strategy_norm_error > 1.0e-3
        {
            eprintln!(
                "stopping metrics: solver produced non-finite or denormalized strategy state"
            );
            return false;
        }
        let Some(exploitability_bb100) = row.root_exploitability.map(|value| value * 100.0) else {
            return true;
        };
        if exploitability_bb100 <= target_bb100 {
            eprintln!(
                "stopping metrics: reached target exploitability {:.3} <= {:.3} bb/100",
                exploitability_bb100, target_bb100
            );
            return false;
        }
        if exploitability_bb100 + 0.25 < best_exploitability_bb100 {
            best_exploitability_bb100 = exploitability_bb100;
            regression_count = 0;
        } else if exploitability_bb100 > best_exploitability_bb100 + 1.0 {
            regression_count += 1;
            if regression_count >= patience {
                eprintln!(
                    "stopping metrics: exploitability regressed for {regression_count} checkpoints; best={best_exploitability_bb100:.3} current={exploitability_bb100:.3} bb/100"
                );
                return false;
            }
        }
        true
    });
}

fn print_metric_gap_nodes(
    flop: [Card; 3],
    config: pokedr_agent::PokedrAgentConfig,
    row: &pokedr_agent::FixedFlopMetricRow,
) {
    let mut nodes = Vec::new();
    if let Some(detail) = &row.local_gap_detail {
        nodes.push(("local_gap_node", detail.node_index));
    }
    if let Some(detail) = &row.recursive_local_gap_detail {
        if !nodes.iter().any(|(_, node)| *node == detail.node_index) {
            nodes.push(("recursive_local_gap_node", detail.node_index));
        }
    }
    for (label, node_index) in nodes {
        match pokedr_agent::dump_fixed_flop_tree_node(flop, config.clone(), node_index) {
            Some(json) => println!("  {label} {json}"),
            None => println!("  {label} missing node={node_index}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PdcfrSweepCandidate {
    alpha: f32,
    gamma: f32,
    eta_start: f32,
    eta_end: f32,
    eta_horizon: usize,
}

#[derive(Debug, Clone)]
struct PdcfrSweepResult {
    candidate: PdcfrSweepCandidate,
    row: pokedr_agent::FixedFlopMetricRow,
}

fn run_solve_flop_sweep(args: FlopCommandArgs) {
    let cli_config = load_cli_config(args.config_path.as_deref());
    let flop = parse_flop(args.flop.as_deref().unwrap_or("As7h2c")).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let base_config = match_config(cli_config.as_ref());
    let iterations = env_usize("POKEDR_SWEEP_ITERATIONS")
        .or_else(|| cli_config.and_then(|config| config.cfr_iterations))
        .unwrap_or(128)
        .max(1);
    let candidates = pdcfr_sweep_candidates();
    eprintln!(
        "sweeping fixed flop pdcfr-plus board={} depth={} equity_runout_cap={} iterations={} candidates={}",
        format_pokedr_cards_for_cli(&flop),
        base_config.max_depth,
        base_config.max_showdown_runouts,
        iterations,
        candidates.len()
    );

    let mut results = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let mut config = base_config.clone();
        config.cfr_iterations = iterations;
        config.cfr_variant = CfrVariant::PdcfrPlus {
            alpha: candidate.alpha,
            gamma: candidate.gamma,
            eta_start: candidate.eta_start,
            eta_end: candidate.eta_end,
            eta_horizon: candidate.eta_horizon,
        };
        eprintln!("running candidate {:?}", config.cfr_variant);
        let rows = pokedr_agent::solve_fixed_flop_metrics(flop, config, &[iterations]);
        let Some(row) = rows.into_iter().last() else {
            continue;
        };
        results.push(PdcfrSweepResult { candidate, row });
    }

    results.sort_by(|left, right| {
        sweep_score(&left.row)
            .partial_cmp(&sweep_score(&right.row))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!(
        "rank,alpha,gamma,eta_start,eta_end,eta_horizon,iterations,elapsed_s,root_exploitability_bb100,root_br_gap,local_br_gap,recursive_root_br_gap,recursive_local_br_gap,root_actions,finite"
    );
    for (rank, result) in results.iter().enumerate() {
        print_sweep_result(rank + 1, result.candidate, &result.row);
    }
}

fn pdcfr_sweep_candidates() -> Vec<PdcfrSweepCandidate> {
    let alphas = env_f32_list("POKEDR_SWEEP_ALPHAS").unwrap_or_else(|| vec![2.5]);
    let gammas = env_f32_list("POKEDR_SWEEP_GAMMAS").unwrap_or_else(|| vec![8.0, 32.0]);
    let eta_schedules = env_eta_schedule_list("POKEDR_SWEEP_ETA_SCHEDULES").unwrap_or_else(|| {
        vec![
            (1.0, 1.0, 128),
            (1.0, 0.0, 64),
            (1.0, 0.0, 128),
            (1.0, 0.0, 256),
        ]
    });
    let mut candidates = Vec::new();
    for alpha in alphas {
        for gamma in &gammas {
            for (eta_start, eta_end, eta_horizon) in &eta_schedules {
                candidates.push(PdcfrSweepCandidate {
                    alpha,
                    gamma: *gamma,
                    eta_start: *eta_start,
                    eta_end: *eta_end,
                    eta_horizon: *eta_horizon,
                });
            }
        }
    }
    candidates
}

fn print_sweep_result(
    rank: usize,
    candidate: PdcfrSweepCandidate,
    row: &pokedr_agent::FixedFlopMetricRow,
) {
    let root_actions = row
        .root_action_probabilities
        .iter()
        .map(|value| format!("{value:.4}"))
        .collect::<Vec<_>>()
        .join("|");
    println!(
        "{rank},{:.3},{:.3},{:.3},{:.3},{},{},{:.2},{},{},{},{},{},{},{}",
        candidate.alpha,
        candidate.gamma,
        candidate.eta_start,
        candidate.eta_end,
        candidate.eta_horizon,
        row.iterations,
        row.elapsed_secs,
        format_optional(row.root_exploitability.map(|value| value * 100.0), 3),
        format_optional(row.root_br_gap, 6),
        format_optional(row.local_br_gap, 6),
        format_optional(row.recursive_root_br_gap, 6),
        format_optional(row.recursive_local_br_gap, 6),
        root_actions,
        row.finite
    );
}

fn sweep_score(row: &pokedr_agent::FixedFlopMetricRow) -> f32 {
    if !row.finite {
        return f32::INFINITY;
    }
    row.root_exploitability.unwrap_or(f32::INFINITY)
}

fn format_optional(value: Option<f32>, decimals: usize) -> String {
    match value {
        Some(value) if value.is_finite() => format!("{value:.decimals$}"),
        _ => "n/a".to_string(),
    }
}

fn print_metric_row(row: &pokedr_agent::FixedFlopMetricRow, target_bb100: f32) {
    let delta = row
        .root_strategy_l1_delta
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "n/a".to_string());
    let root_actions = row
        .root_action_probabilities
        .iter()
        .map(|value| format!("{value:.4}"))
        .collect::<Vec<_>>()
        .join(",");
    let root_exploitability_bb100 = row.root_exploitability.map(|value| value * 100.0);
    let current_root_exploitability_bb100 =
        row.current_root_exploitability.map(|value| value * 100.0);
    let cpu_root_exploitability_bb100 = row.cpu_root_exploitability.map(|value| value * 100.0);
    let cpu_gpu_root_exploitability_delta_bb100 = row
        .cpu_gpu_root_exploitability_delta
        .map(|value| value * 100.0);
    let root_converged = root_exploitability_bb100
        .map(|value| value <= target_bb100)
        .unwrap_or(false);
    println!(
        "board={} iterations={} elapsed={:.2}s root_l1_delta={} root_actions=[{}] root_exploitability={} root_exploitability_bb100={} current_root_exploitability={} current_root_exploitability_bb100={} cpu_root_exploitability={} cpu_root_exploitability_bb100={} cpu_gpu_root_exploitability_delta={} cpu_gpu_root_exploitability_delta_bb100={} pio_style_target_bb100={:.2} pio_style_converged={} hero_root_br_value={} villain_root_br_value={} hero_root_profile_value={} villain_root_profile_value={} root_profile_value_sum={} cpu_hero_root_br_value={} cpu_villain_root_br_value={} cpu_hero_root_profile_value={} cpu_villain_root_profile_value={} cpu_root_profile_value_sum={} current_hero_root_br_value={} current_villain_root_br_value={} current_hero_root_profile_value={} current_villain_root_profile_value={} current_root_profile_value_sum={} root_br_gap={} local_br_gap={} recursive_root_br_gap={} recursive_local_br_gap={} regret_mass={:.3} illegal_mass={:.6} current_norm_err={:.6} avg_norm_err={:.6} finite={}",
        row.board,
        row.iterations,
        row.elapsed_secs,
        delta,
        root_actions,
        row.root_exploitability
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        root_exploitability_bb100
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.current_root_exploitability
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        current_root_exploitability_bb100
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.cpu_root_exploitability
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        cpu_root_exploitability_bb100
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.cpu_gpu_root_exploitability_delta
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        cpu_gpu_root_exploitability_delta_bb100
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "n/a".to_string()),
        target_bb100,
        root_converged,
        row.hero_root_br_value
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.villain_root_br_value
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.hero_root_profile_value
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.villain_root_profile_value
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.hero_root_profile_value
            .zip(row.villain_root_profile_value)
            .map(|(hero, villain)| format!("{:.6}", hero + villain))
            .unwrap_or_else(|| "n/a".to_string()),
        row.cpu_hero_root_br_value
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.cpu_villain_root_br_value
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.cpu_hero_root_profile_value
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.cpu_villain_root_profile_value
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.cpu_hero_root_profile_value
            .zip(row.cpu_villain_root_profile_value)
            .map(|(hero, villain)| format!("{:.6}", hero + villain))
            .unwrap_or_else(|| "n/a".to_string()),
        row.current_hero_root_br_value
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.current_villain_root_br_value
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.current_hero_root_profile_value
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.current_villain_root_profile_value
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.current_hero_root_profile_value
            .zip(row.current_villain_root_profile_value)
            .map(|(hero, villain)| format!("{:.6}", hero + villain))
            .unwrap_or_else(|| "n/a".to_string()),
        row.root_br_gap
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.local_br_gap
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.recursive_root_br_gap
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.recursive_local_br_gap
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.positive_regret_mass,
        row.illegal_strategy_mass,
        row.current_strategy_norm_error,
        row.average_strategy_norm_error,
        row.finite
    );
    if let Some(detail) = &row.local_gap_detail {
        println!("  local_gap_detail {}", format_gap_detail(detail));
    }
    if let Some(detail) = &row.recursive_local_gap_detail {
        println!("  recursive_local_gap_detail {}", format_gap_detail(detail));
    }
}

fn format_gap_detail(detail: &pokedr_agent::LocalGapDetail) -> String {
    format!(
        "gap={:.6} weighted_gap={:.6} reach={:.6} public_infoset={} node={} player={:?} combo_index={} combo=[{}] actions=[{}] avg_strategy=[{}] current_strategy=[{}] regrets=[{}] strategy_sum=[{}] action_values=[{}]",
        detail.gap,
        detail.weighted_gap,
        detail.reach_weight,
        detail.public_infoset,
        detail.node_index,
        detail.player,
        detail.combo_index,
        detail.combo,
        detail.actions.join(","),
        detail
            .average_strategy
            .iter()
            .map(|value| format!("{value:.4}"))
            .collect::<Vec<_>>()
            .join(","),
        detail
            .current_strategy
            .iter()
            .map(|value| format!("{value:.4}"))
            .collect::<Vec<_>>()
            .join(","),
        detail
            .regrets
            .iter()
            .map(|value| format!("{value:.4}"))
            .collect::<Vec<_>>()
            .join(","),
        detail
            .strategy_sum
            .iter()
            .map(|value| format!("{value:.4}"))
            .collect::<Vec<_>>()
            .join(","),
        detail
            .action_values
            .iter()
            .map(|value| format!("{value:.4}"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn run_tree_db(mut args: impl Iterator<Item = String>) {
    match args.next().as_deref() {
        Some("build") => run_tree_db_build(args),
        Some("analyze") => run_tree_db_analyze(args),
        Some("check-line") => run_tree_db_check_line(args),
        Some("top-combos") => run_tree_db_top_combos(args),
        Some("root-br") => run_tree_db_root_br(args),
        Some("node") => run_tree_db_node(args),
        _ => {
            eprintln!(
                "usage: pokedr-cli tree-db <build|analyze|check-line|top-combos|root-br|node>\n  pokedr-cli tree-db build <tree.sqlite> [flop] [--config path.yml]\n  pokedr-cli tree-db analyze <tree.sqlite>\n  pokedr-cli tree-db check-line <tree.sqlite> [--limit n]\n  pokedr-cli tree-db top-combos <tree.sqlite>\n  pokedr-cli tree-db root-br <tree.sqlite> [--limit n]\n  pokedr-cli tree-db node <tree.sqlite> <node_id>"
            );
            std::process::exit(2);
        }
    }
}

fn run_tree_db_build(mut args: impl Iterator<Item = String>) {
    let build_args = parse_tree_db_build_args(&mut args);
    let cli_config = load_cli_config(build_args.config_path.as_deref());
    let flop = parse_flop(build_args.flop.as_deref().unwrap_or("As7h2c")).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let mut config = match_config(cli_config.as_ref());
    if let Some(iterations) = build_args.iterations {
        config.cfr_iterations = iterations.max(1);
    }
    if let Some(max_depth) = build_args.max_depth {
        config.max_depth = max_depth;
    }
    let dump = pokedr_agent::build_fixed_flop_tree_dump_with_combo_limit(
        flop,
        config,
        build_args.combo_limit,
    );
    write_tree_db(&build_args.db_path, &dump);
    println!(
        "wrote {} nodes, {} actions, {} solver nodes, {} combo rows to {}",
        dump.nodes.len(),
        dump.actions.len(),
        dump.solver_nodes.len(),
        dump.solver_combos.len(),
        build_args.db_path
    );
}

fn run_tree_db_top_combos(mut args: impl Iterator<Item = String>) {
    let Some(db_path) = args.next() else {
        eprintln!("usage: pokedr-cli tree-db top-combos <tree.sqlite>");
        std::process::exit(2);
    };
    if let Some(extra) = args.next() {
        eprintln!("unexpected argument: {extra}");
        std::process::exit(2);
    }
    let conn = Connection::open(&db_path).unwrap_or_else(|error| {
        eprintln!("failed to open tree DB {db_path}: {error}");
        std::process::exit(2);
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&db_top_combo_gaps(&conn))
            .expect("combo gap JSON must serialize")
    );
}

fn run_tree_db_root_br(mut args: impl Iterator<Item = String>) {
    let Some(db_path) = args.next() else {
        eprintln!("usage: pokedr-cli tree-db root-br <tree.sqlite> [--limit n]");
        std::process::exit(2);
    };
    let mut limit = 12usize;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--limit" => {
                let value = args.next().unwrap_or_else(|| {
                    eprintln!("--limit requires a number");
                    std::process::exit(2);
                });
                limit = value.parse().unwrap_or_else(|_| {
                    eprintln!("--limit must be a number: {value}");
                    std::process::exit(2);
                });
            }
            extra => {
                eprintln!("unexpected argument: {extra}");
                std::process::exit(2);
            }
        }
    }
    let conn = Connection::open(&db_path).unwrap_or_else(|error| {
        eprintln!("failed to open tree DB {db_path}: {error}");
        std::process::exit(2);
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&db_root_br_report(&conn, limit))
            .expect("root BR JSON must serialize")
    );
}

fn run_tree_db_check_line(mut args: impl Iterator<Item = String>) {
    let Some(db_path) = args.next() else {
        eprintln!("usage: pokedr-cli tree-db check-line <tree.sqlite> [--limit n]");
        std::process::exit(2);
    };
    let mut limit = 12usize;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--limit" => {
                let value = args.next().unwrap_or_else(|| {
                    eprintln!("--limit requires a number");
                    std::process::exit(2);
                });
                limit = value.parse().unwrap_or_else(|_| {
                    eprintln!("--limit must be a number: {value}");
                    std::process::exit(2);
                });
            }
            extra => {
                eprintln!("unexpected argument: {extra}");
                std::process::exit(2);
            }
        }
    }
    let conn = Connection::open(&db_path).unwrap_or_else(|error| {
        eprintln!("failed to open tree DB {db_path}: {error}");
        std::process::exit(2);
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&db_check_line_report(&conn, limit))
            .expect("check line JSON must serialize")
    );
}

fn run_tree_db_node(mut args: impl Iterator<Item = String>) {
    let Some(db_path) = args.next() else {
        eprintln!("usage: pokedr-cli tree-db node <tree.sqlite> <node_id> [--limit n]");
        std::process::exit(2);
    };
    let Some(node_id) = args.next() else {
        eprintln!("usage: pokedr-cli tree-db node <tree.sqlite> <node_id> [--limit n]");
        std::process::exit(2);
    };
    let mut limit = 16usize;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--limit" => {
                let value = args.next().unwrap_or_else(|| {
                    eprintln!("--limit requires a number");
                    std::process::exit(2);
                });
                limit = value.parse().unwrap_or_else(|_| {
                    eprintln!("--limit must be a number: {value}");
                    std::process::exit(2);
                });
            }
            extra => {
                eprintln!("unexpected argument: {extra}");
                std::process::exit(2);
            }
        }
    }
    let node_id = node_id.parse::<i64>().unwrap_or_else(|_| {
        eprintln!("node_id must be an integer: {node_id}");
        std::process::exit(2);
    });
    let conn = Connection::open(&db_path).unwrap_or_else(|error| {
        eprintln!("failed to open tree DB {db_path}: {error}");
        std::process::exit(2);
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&db_node_detail(&conn, node_id, limit))
            .expect("node detail JSON must serialize")
    );
}

fn run_tree_db_analyze(mut args: impl Iterator<Item = String>) {
    let Some(db_path) = args.next() else {
        eprintln!("usage: pokedr-cli tree-db analyze <tree.sqlite>");
        std::process::exit(2);
    };
    if let Some(extra) = args.next() {
        eprintln!("unexpected argument: {extra}");
        std::process::exit(2);
    }
    let conn = Connection::open(&db_path).unwrap_or_else(|error| {
        eprintln!("failed to open tree DB {db_path}: {error}");
        std::process::exit(2);
    });
    let analysis = analyze_tree_db(&conn);
    println!(
        "{}",
        serde_json::to_string_pretty(&analysis).expect("analysis JSON must serialize")
    );
}

fn write_tree_db(path: &str, dump: &pokedr_agent::FixedFlopTreeDump) {
    let _ = std::fs::remove_file(path);
    let mut conn = Connection::open(path).unwrap_or_else(|error| {
        eprintln!("failed to create tree DB {path}: {error}");
        std::process::exit(1);
    });
    conn.execute_batch(
        r#"
        PRAGMA journal_mode = WAL;
        CREATE TABLE nodes (
            node_id INTEGER PRIMARY KEY,
            parent_id INTEGER,
            kind TEXT NOT NULL,
            infoset INTEGER,
            path TEXT NOT NULL,
            street TEXT,
            board TEXT,
            acting_player TEXT,
            pot INTEGER,
            to_call INTEGER,
            hero_invested INTEGER,
            villain_invested INTEGER,
            terminal_kind TEXT
        );
        CREATE TABLE actions (
            node_id INTEGER NOT NULL,
            action_index INTEGER NOT NULL,
            child_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            source TEXT NOT NULL,
            PRIMARY KEY (node_id, action_index)
        );
        CREATE TABLE tree_metrics (
            metric TEXT PRIMARY KEY,
            value REAL
        );
        CREATE TABLE solver_nodes (
            node_id INTEGER PRIMARY KEY,
            infoset INTEGER NOT NULL,
            iterations INTEGER NOT NULL,
            acting_player TEXT NOT NULL,
            action_count INTEGER NOT NULL,
            legal_combo_count INTEGER NOT NULL,
            value_reach_sum REAL NOT NULL,
            avg_strategy_weight_sum REAL NOT NULL,
            current_strategy_weight_sum REAL NOT NULL,
            avg_strategy TEXT NOT NULL,
            current_strategy TEXT NOT NULL,
            avg_action_ev TEXT,
            current_action_ev TEXT,
            avg_policy_ev REAL,
            current_policy_ev REAL,
            avg_gap REAL,
            current_gap REAL,
            avg_recursive_action_ev TEXT,
            current_recursive_action_ev TEXT,
            avg_recursive_policy_ev REAL,
            current_recursive_policy_ev REAL,
            avg_recursive_gap REAL,
            current_recursive_gap REAL
        );
        CREATE TABLE solver_combos (
            node_id INTEGER NOT NULL,
            combo_index INTEGER NOT NULL,
            combo TEXT NOT NULL,
            reach REAL NOT NULL,
            weighted_gap REAL NOT NULL,
            recursive_weighted_gap REAL NOT NULL,
            current_recursive_weighted_gap REAL NOT NULL,
            avg_strategy_weight REAL NOT NULL,
            current_strategy_weight REAL NOT NULL,
            avg_action_values TEXT,
            current_action_values TEXT,
            avg_recursive_action_values TEXT,
            current_recursive_action_values TEXT,
            avg_strategy TEXT NOT NULL,
            current_strategy TEXT NOT NULL,
            regrets TEXT NOT NULL,
            strategy_sum TEXT NOT NULL,
            PRIMARY KEY (node_id, combo_index)
        );
        CREATE TABLE root_br_combos (
            profile TEXT NOT NULL,
            player TEXT NOT NULL,
            combo_index INTEGER NOT NULL,
            combo TEXT NOT NULL,
            root_weight REAL NOT NULL,
            opponent_nonblocking_weight REAL NOT NULL,
            root_value REAL NOT NULL,
            contribution REAL NOT NULL,
            contribution_bb100 REAL NOT NULL,
            PRIMARY KEY (profile, player, combo_index)
        );
        CREATE INDEX nodes_parent_idx ON nodes(parent_id);
        CREATE INDEX solver_nodes_avg_gap_idx ON solver_nodes(avg_gap DESC);
        CREATE INDEX solver_nodes_current_gap_idx ON solver_nodes(current_gap DESC);
        CREATE INDEX root_br_combos_contribution_idx ON root_br_combos(profile, player, contribution DESC);
        "#,
    )
    .unwrap_or_else(|error| {
        eprintln!("failed to initialize tree DB schema: {error}");
        std::process::exit(1);
    });

    let tx = conn.transaction().unwrap();
    if let Some(metrics) = &dump.metrics {
        let mut statement = tx
            .prepare("INSERT INTO tree_metrics VALUES (?1, ?2)")
            .unwrap();
        let mut insert_metric = |metric: &str, value: Option<f32>| {
            statement
                .execute(params![metric, value])
                .expect("tree metric insert must succeed");
        };
        insert_metric("iterations", Some(metrics.iterations as f32));
        insert_metric("root_exploitability", metrics.root_exploitability);
        insert_metric(
            "current_root_exploitability",
            metrics.current_root_exploitability,
        );
        insert_metric(
            "root_exploitability_bb100",
            metrics.root_exploitability.map(|v| v * 100.0),
        );
        insert_metric(
            "current_root_exploitability_bb100",
            metrics.current_root_exploitability.map(|v| v * 100.0),
        );
        insert_metric("hero_root_br_value", metrics.hero_root_br_value);
        insert_metric("villain_root_br_value", metrics.villain_root_br_value);
        insert_metric("hero_root_profile_value", metrics.hero_root_profile_value);
        insert_metric(
            "villain_root_profile_value",
            metrics.villain_root_profile_value,
        );
        insert_metric("hero_root_br_improvement", metrics.hero_root_br_improvement);
        insert_metric(
            "villain_root_br_improvement",
            metrics.villain_root_br_improvement,
        );
        insert_metric(
            "hero_root_br_improvement_bb100",
            metrics.hero_root_br_improvement.map(|v| v * 100.0),
        );
        insert_metric(
            "villain_root_br_improvement_bb100",
            metrics.villain_root_br_improvement.map(|v| v * 100.0),
        );
        insert_metric(
            "current_hero_root_br_value",
            metrics.current_hero_root_br_value,
        );
        insert_metric(
            "current_villain_root_br_value",
            metrics.current_villain_root_br_value,
        );
        insert_metric(
            "current_hero_root_profile_value",
            metrics.current_hero_root_profile_value,
        );
        insert_metric(
            "current_villain_root_profile_value",
            metrics.current_villain_root_profile_value,
        );
        insert_metric(
            "current_hero_root_br_improvement",
            metrics.current_hero_root_br_improvement,
        );
        insert_metric(
            "current_villain_root_br_improvement",
            metrics.current_villain_root_br_improvement,
        );
        insert_metric(
            "current_hero_root_br_improvement_bb100",
            metrics.current_hero_root_br_improvement.map(|v| v * 100.0),
        );
        insert_metric(
            "current_villain_root_br_improvement_bb100",
            metrics
                .current_villain_root_br_improvement
                .map(|v| v * 100.0),
        );
        insert_metric("recursive_root_br_gap", metrics.recursive_root_br_gap);
        insert_metric(
            "current_recursive_root_br_gap",
            metrics.current_recursive_root_br_gap,
        );
    }
    {
        let mut statement = tx
            .prepare(
                "INSERT INTO nodes VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            )
            .unwrap();
        for node in &dump.nodes {
            statement
                .execute(params![
                    db_usize(node.node_id),
                    db_opt_usize(node.parent_id),
                    node.kind,
                    db_opt_usize(node.infoset),
                    node.path,
                    node.street,
                    node.board,
                    node.acting_player,
                    db_opt_u32(node.pot),
                    db_opt_u32(node.to_call),
                    db_opt_u32(node.hero_invested),
                    db_opt_u32(node.villain_invested),
                    node.terminal_kind,
                ])
                .unwrap();
        }
    }
    {
        let mut statement = tx
            .prepare("INSERT INTO actions VALUES (?1, ?2, ?3, ?4, ?5)")
            .unwrap();
        for action in &dump.actions {
            statement
                .execute(params![
                    db_usize(action.node_id),
                    db_usize(action.action_index),
                    db_usize(action.child_id),
                    action.action,
                    action.source,
                ])
                .unwrap();
        }
    }
    {
        let mut statement = tx.prepare(
            "INSERT INTO solver_nodes VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23)",
        ).unwrap();
        for node in &dump.solver_nodes {
            statement
                .execute(params![
                    db_usize(node.node_id),
                    db_usize(node.infoset),
                    db_usize(node.iterations),
                    node.acting_player,
                    db_usize(node.action_count),
                    db_usize(node.legal_combo_count),
                    node.value_reach_sum,
                    node.avg_strategy_weight_sum,
                    node.current_strategy_weight_sum,
                    vec_json(&node.avg_strategy),
                    vec_json(&node.current_strategy),
                    opt_vec_json(&node.avg_action_ev),
                    opt_vec_json(&node.current_action_ev),
                    node.avg_policy_ev,
                    node.current_policy_ev,
                    node.avg_gap,
                    node.current_gap,
                    opt_vec_json(&node.avg_recursive_action_ev),
                    opt_vec_json(&node.current_recursive_action_ev),
                    node.avg_recursive_policy_ev,
                    node.current_recursive_policy_ev,
                    node.avg_recursive_gap,
                    node.current_recursive_gap,
                ])
                .unwrap();
        }
    }
    {
        let mut statement = tx
            .prepare(
                "INSERT INTO solver_combos VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            )
            .unwrap();
        for combo in &dump.solver_combos {
            statement
                .execute(params![
                    db_usize(combo.node_id),
                    db_usize(combo.combo_index),
                    combo.combo,
                    combo.reach,
                    combo.weighted_gap,
                    combo.recursive_weighted_gap,
                    combo.current_recursive_weighted_gap,
                    combo.avg_strategy_weight,
                    combo.current_strategy_weight,
                    opt_vec_json(&combo.avg_action_values),
                    opt_vec_json(&combo.current_action_values),
                    opt_vec_json(&combo.avg_recursive_action_values),
                    opt_vec_json(&combo.current_recursive_action_values),
                    vec_json(&combo.avg_strategy),
                    vec_json(&combo.current_strategy),
                    vec_json(&combo.regrets),
                    vec_json(&combo.strategy_sum),
                ])
                .unwrap();
        }
    }
    {
        let mut statement = tx
            .prepare("INSERT INTO root_br_combos VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)")
            .unwrap();
        for combo in &dump.root_br_combos {
            statement
                .execute(params![
                    combo.profile,
                    combo.player,
                    db_usize(combo.combo_index),
                    combo.combo,
                    combo.root_weight,
                    combo.opponent_nonblocking_weight,
                    combo.root_value,
                    combo.contribution,
                    combo.contribution_bb100,
                ])
                .unwrap();
        }
    }
    tx.commit().unwrap();
}

fn analyze_tree_db(conn: &Connection) -> Value {
    json!({
        "metrics": db_tree_metrics(conn),
        "counts": db_counts(conn),
        "gap_breakdown": db_gap_breakdown(conn),
        "recursive_gap_breakdown": db_recursive_gap_breakdown(conn),
        "top_avg_gaps": db_top_gaps(conn, "avg_gap", "avg_action_ev", "avg_policy_ev"),
        "top_reached_avg_gaps": db_top_reached_gaps(conn, "avg_gap", "avg_action_ev", "avg_policy_ev", "avg_strategy_weight_sum"),
        "top_current_gaps": db_top_gaps(conn, "current_gap", "current_action_ev", "current_policy_ev"),
        "top_reached_current_gaps": db_top_reached_gaps(conn, "current_gap", "current_action_ev", "current_policy_ev", "current_strategy_weight_sum"),
        "top_recursive_avg_gaps": db_top_gaps(conn, "avg_recursive_gap", "avg_recursive_action_ev", "avg_recursive_policy_ev"),
        "top_reached_recursive_avg_gaps": db_top_reached_gaps(conn, "avg_recursive_gap", "avg_recursive_action_ev", "avg_recursive_policy_ev", "avg_strategy_weight_sum"),
        "top_recursive_current_gaps": db_top_gaps(conn, "current_recursive_gap", "current_recursive_action_ev", "current_recursive_policy_ev"),
        "top_reached_recursive_current_gaps": db_top_reached_gaps(conn, "current_recursive_gap", "current_recursive_action_ev", "current_recursive_policy_ev", "current_strategy_weight_sum"),
        "top_combo_gaps": db_top_combo_gaps(conn),
        "root_br_contributors": db_root_br_contributors(conn),
        "root_action_subtrees": db_root_action_subtrees(conn),
    })
}

fn db_root_br_contributors(conn: &Connection) -> Value {
    db_root_br_contributors_with_limit(conn, 40)
}

fn db_root_br_report(conn: &Connection, limit: usize) -> Value {
    json!({
        "metrics": db_tree_metrics(conn),
        "sums": db_root_br_sums(conn),
        "top": db_root_br_contributors_with_limit(conn, limit),
    })
}

fn db_root_br_sums(conn: &Connection) -> Value {
    let mut statement = conn
        .prepare(
            "SELECT profile,
                    player,
                    SUM(contribution),
                    SUM(contribution_bb100)
             FROM root_br_combos
             GROUP BY profile, player
             ORDER BY profile, player",
        )
        .unwrap();
    let rows = statement
        .query_map([], |row| {
            Ok(json!({
                "profile": row.get::<_, String>(0)?,
                "player": row.get::<_, String>(1)?,
                "contribution": row.get::<_, f64>(2)?,
                "contribution_bb100": row.get::<_, f64>(3)?,
            }))
        })
        .unwrap();
    Value::Array(rows.map(Result::unwrap).collect())
}

fn db_root_br_contributors_with_limit(conn: &Connection, limit: usize) -> Value {
    let mut statement = conn
        .prepare(
            "SELECT profile,
                    player,
                    combo_index,
                    combo,
                    root_weight,
                    opponent_nonblocking_weight,
                    root_value,
                    contribution,
                    contribution_bb100
             FROM root_br_combos
             ORDER BY contribution DESC
             LIMIT ?1",
        )
        .unwrap();
    let rows = statement
        .query_map([db_usize(limit)], |row| {
            Ok(json!({
                "profile": row.get::<_, String>(0)?,
                "player": row.get::<_, String>(1)?,
                "combo_index": row.get::<_, i64>(2)?,
                "combo": row.get::<_, String>(3)?,
                "root_weight": row.get::<_, f64>(4)?,
                "opponent_nonblocking_weight": row.get::<_, f64>(5)?,
                "root_value": row.get::<_, f64>(6)?,
                "contribution": row.get::<_, f64>(7)?,
                "contribution_bb100": row.get::<_, f64>(8)?,
            }))
        })
        .unwrap();
    Value::Array(rows.map(Result::unwrap).collect())
}

fn db_tree_metrics(conn: &Connection) -> Value {
    let exists = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'tree_metrics'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if !exists {
        return Value::Null;
    }
    let mut statement = conn
        .prepare("SELECT metric, value FROM tree_metrics ORDER BY metric")
        .unwrap();
    let rows = statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Option<f64>>(1)?))
        })
        .unwrap();
    let mut object = serde_json::Map::new();
    for row in rows {
        let (metric, value) = row.unwrap();
        object.insert(
            metric,
            value.map(|value| json!(value)).unwrap_or(Value::Null),
        );
    }
    Value::Object(object)
}

fn db_counts(conn: &Connection) -> Value {
    let count = |table: &str| -> i64 {
        conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
            row.get(0)
        })
        .unwrap()
    };
    json!({
        "nodes": count("nodes"),
        "actions": count("actions"),
        "solver_nodes": count("solver_nodes"),
        "solver_combos": count("solver_combos"),
    })
}

fn db_gap_breakdown(conn: &Connection) -> Value {
    let mut statement = conn
        .prepare(
            "SELECT n.street,
                    n.acting_player,
                    n.to_call,
                    s.action_count,
                    COUNT(*),
                    AVG(s.avg_gap),
                    MAX(s.avg_gap),
                    AVG(s.current_gap),
                    MAX(s.current_gap)
             FROM solver_nodes s JOIN nodes n ON n.node_id = s.node_id
             WHERE s.avg_gap IS NOT NULL OR s.current_gap IS NOT NULL
             GROUP BY n.street, n.acting_player, n.to_call, s.action_count
             ORDER BY MAX(s.avg_gap) DESC
             LIMIT 40",
        )
        .unwrap();
    let rows = statement
        .query_map([], |row| {
            Ok(json!({
                "street": row.get::<_, Option<String>>(0)?,
                "acting_player": row.get::<_, Option<String>>(1)?,
                "to_call": row.get::<_, Option<i64>>(2)?,
                "action_count": row.get::<_, i64>(3)?,
                "nodes": row.get::<_, i64>(4)?,
                "mean_avg_gap": row.get::<_, Option<f64>>(5)?.unwrap_or(0.0),
                "max_avg_gap": row.get::<_, Option<f64>>(6)?.unwrap_or(0.0),
                "mean_current_gap": row.get::<_, Option<f64>>(7)?.unwrap_or(0.0),
                "max_current_gap": row.get::<_, Option<f64>>(8)?.unwrap_or(0.0),
            }))
        })
        .unwrap();
    Value::Array(rows.map(Result::unwrap).collect())
}

fn db_recursive_gap_breakdown(conn: &Connection) -> Value {
    let mut statement = conn
        .prepare(
            "SELECT n.street,
                    n.acting_player,
                    n.to_call,
                    s.action_count,
                    COUNT(*),
                    AVG(s.avg_recursive_gap),
                    MAX(s.avg_recursive_gap),
                    AVG(s.current_recursive_gap),
                    MAX(s.current_recursive_gap)
             FROM solver_nodes s JOIN nodes n ON n.node_id = s.node_id
             WHERE s.avg_recursive_gap IS NOT NULL OR s.current_recursive_gap IS NOT NULL
             GROUP BY n.street, n.acting_player, n.to_call, s.action_count
             ORDER BY MAX(s.avg_recursive_gap) DESC
             LIMIT 40",
        )
        .unwrap();
    let rows = statement
        .query_map([], |row| {
            Ok(json!({
                "street": row.get::<_, Option<String>>(0)?,
                "acting_player": row.get::<_, Option<String>>(1)?,
                "to_call": row.get::<_, Option<i64>>(2)?,
                "action_count": row.get::<_, i64>(3)?,
                "nodes": row.get::<_, i64>(4)?,
                "mean_avg_recursive_gap": row.get::<_, Option<f64>>(5)?.unwrap_or(0.0),
                "max_avg_recursive_gap": row.get::<_, Option<f64>>(6)?.unwrap_or(0.0),
                "mean_current_recursive_gap": row.get::<_, Option<f64>>(7)?.unwrap_or(0.0),
                "max_current_recursive_gap": row.get::<_, Option<f64>>(8)?.unwrap_or(0.0),
            }))
        })
        .unwrap();
    Value::Array(rows.map(Result::unwrap).collect())
}

fn db_top_gaps(
    conn: &Connection,
    gap_column: &str,
    action_ev_column: &str,
    policy_column: &str,
) -> Value {
    let sql = format!(
        "SELECT n.node_id, n.street, n.acting_player, n.pot, n.to_call, n.path, s.{gap_column}, s.{action_ev_column}, s.{policy_column}, s.avg_strategy, s.current_strategy, s.value_reach_sum, s.avg_strategy_weight_sum, s.current_strategy_weight_sum
         FROM solver_nodes s JOIN nodes n ON n.node_id = s.node_id
         WHERE s.{gap_column} IS NOT NULL
         ORDER BY s.{gap_column} DESC
         LIMIT 20"
    );
    let mut statement = conn.prepare(&sql).unwrap();
    let rows = statement
        .query_map([], |row| {
            Ok(json!({
                "node": row.get::<_, i64>(0)?,
                "street": row.get::<_, Option<String>>(1)?,
                "acting_player": row.get::<_, Option<String>>(2)?,
                "pot": row.get::<_, Option<i64>>(3)?,
                "to_call": row.get::<_, Option<i64>>(4)?,
                "path": json_text_value(row.get::<_, String>(5)?),
                "gap": row.get::<_, f64>(6)?,
                "action_ev": row.get::<_, Option<String>>(7)?.map(json_text_value),
                "policy_ev": row.get::<_, Option<f64>>(8)?,
                "avg_strategy": json_text_value(row.get::<_, String>(9)?),
                "current_strategy": json_text_value(row.get::<_, String>(10)?),
                "value_reach_sum": row.get::<_, f64>(11)?,
                "avg_strategy_weight_sum": row.get::<_, f64>(12)?,
                "current_strategy_weight_sum": row.get::<_, f64>(13)?,
                "actions": db_node_actions(conn, row.get::<_, i64>(0)?),
            }))
        })
        .unwrap();
    Value::Array(rows.map(Result::unwrap).collect())
}

fn db_top_reached_gaps(
    conn: &Connection,
    gap_column: &str,
    action_ev_column: &str,
    policy_column: &str,
    strategy_weight_column: &str,
) -> Value {
    let sql = format!(
        "SELECT n.node_id, n.street, n.acting_player, n.pot, n.to_call, n.path, s.{gap_column}, s.{action_ev_column}, s.{policy_column}, s.avg_strategy, s.current_strategy, s.value_reach_sum, s.avg_strategy_weight_sum, s.current_strategy_weight_sum
         FROM solver_nodes s JOIN nodes n ON n.node_id = s.node_id
         WHERE s.{gap_column} IS NOT NULL AND s.{strategy_weight_column} > 1.0e-6
         ORDER BY s.{gap_column} DESC
         LIMIT 20"
    );
    let mut statement = conn.prepare(&sql).unwrap();
    let rows = statement
        .query_map([], |row| {
            Ok(json!({
                "node": row.get::<_, i64>(0)?,
                "street": row.get::<_, Option<String>>(1)?,
                "acting_player": row.get::<_, Option<String>>(2)?,
                "pot": row.get::<_, Option<i64>>(3)?,
                "to_call": row.get::<_, Option<i64>>(4)?,
                "path": json_text_value(row.get::<_, String>(5)?),
                "gap": row.get::<_, f64>(6)?,
                "action_ev": row.get::<_, Option<String>>(7)?.map(json_text_value),
                "policy_ev": row.get::<_, Option<f64>>(8)?,
                "avg_strategy": json_text_value(row.get::<_, String>(9)?),
                "current_strategy": json_text_value(row.get::<_, String>(10)?),
                "value_reach_sum": row.get::<_, f64>(11)?,
                "avg_strategy_weight_sum": row.get::<_, f64>(12)?,
                "current_strategy_weight_sum": row.get::<_, f64>(13)?,
                "actions": db_node_actions(conn, row.get::<_, i64>(0)?),
            }))
        })
        .unwrap();
    Value::Array(rows.map(Result::unwrap).collect())
}

fn db_top_combo_gaps(conn: &Connection) -> Value {
    let mut statement = conn
        .prepare(
            "SELECT c.node_id,
                    n.street,
                    n.acting_player,
                    n.pot,
                    n.to_call,
                    n.path,
                    c.combo_index,
                    c.combo,
                    c.reach,
                    c.weighted_gap,
                    c.recursive_weighted_gap,
                    c.current_recursive_weighted_gap,
                    c.avg_strategy_weight,
                    c.current_strategy_weight,
                    c.avg_action_values,
                    c.current_action_values,
                    c.avg_recursive_action_values,
                    c.current_recursive_action_values,
                    c.avg_strategy,
                    c.current_strategy,
                    c.regrets
             FROM solver_combos c JOIN nodes n ON n.node_id = c.node_id
             ORDER BY c.weighted_gap DESC
             LIMIT 20",
        )
        .unwrap();
    let rows = statement
        .query_map([], |row| {
            let node_id = row.get::<_, i64>(0)?;
            Ok(json!({
                "node": node_id,
                "street": row.get::<_, Option<String>>(1)?,
                "acting_player": row.get::<_, Option<String>>(2)?,
                "pot": row.get::<_, Option<i64>>(3)?,
                "to_call": row.get::<_, Option<i64>>(4)?,
                "path": json_text_value(row.get::<_, String>(5)?),
                "combo_index": row.get::<_, i64>(6)?,
                "combo": row.get::<_, String>(7)?,
                "reach": row.get::<_, f64>(8)?,
                "weighted_gap": row.get::<_, f64>(9)?,
                "local_gap": local_gap(row.get::<_, f64>(8)?, row.get::<_, f64>(9)?),
                "recursive_weighted_gap": row.get::<_, f64>(10)?,
                "recursive_local_gap": local_gap(row.get::<_, f64>(8)?, row.get::<_, f64>(10)?),
                "current_recursive_weighted_gap": row.get::<_, f64>(11)?,
                "current_recursive_local_gap": local_gap(row.get::<_, f64>(8)?, row.get::<_, f64>(11)?),
                "avg_strategy_weight": row.get::<_, f64>(12)?,
                "current_strategy_weight": row.get::<_, f64>(13)?,
                "avg_action_values": row.get::<_, Option<String>>(14)?.map(json_text_value),
                "current_action_values": row.get::<_, Option<String>>(15)?.map(json_text_value),
                "avg_recursive_action_values": row.get::<_, Option<String>>(16)?.map(json_text_value),
                "current_recursive_action_values": row.get::<_, Option<String>>(17)?.map(json_text_value),
                "avg_strategy": json_text_value(row.get::<_, String>(18)?),
                "current_strategy": json_text_value(row.get::<_, String>(19)?),
                "regrets": json_text_value(row.get::<_, String>(20)?),
                "actions": db_node_actions(conn, node_id),
            }))
        })
        .unwrap();
    Value::Array(rows.map(Result::unwrap).collect())
}

fn db_check_line_report(conn: &Connection, limit: usize) -> Value {
    let root_check_child = conn
        .query_row(
            "SELECT child_id FROM actions WHERE node_id = 0 AND action = 'check' LIMIT 1",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(1);
    json!({
        "root_check_child": root_check_child,
        "flop_villain_node": db_node_detail(conn, root_check_child, limit),
        "flop_villain_aggressive_combos": db_node_aggressive_combos(conn, root_check_child, limit),
        "check_subtree_hero_leaks_avg": db_subtree_top_reached_gaps(
            conn,
            root_check_child,
            "avg_gap",
            "avg_action_ev",
            "avg_policy_ev",
            "avg_strategy_weight_sum",
            limit,
        ),
        "check_subtree_hero_leaks_current": db_subtree_top_reached_gaps(
            conn,
            root_check_child,
            "current_gap",
            "current_action_ev",
            "current_policy_ev",
            "current_strategy_weight_sum",
            limit,
        ),
        "check_subtree_villain_turn_barrels": db_subtree_aggressive_nodes(
            conn,
            root_check_child,
            "Villain",
            "Turn",
            limit,
        ),
    })
}

fn db_node_aggressive_combos(conn: &Connection, node_id: i64, limit: usize) -> Value {
    let actions = db_action_labels(conn, node_id);
    let check_index = actions.iter().position(|action| action == "check");
    let bet_indices: Vec<_> = actions
        .iter()
        .enumerate()
        .filter_map(|(index, action)| action.starts_with("bet:").then_some(index))
        .collect();
    if check_index.is_none() || bet_indices.is_empty() {
        return Value::Array(Vec::new());
    }
    let mut statement = conn
        .prepare(
            "SELECT combo_index,
                    combo,
                    reach,
                    weighted_gap,
                    avg_action_values,
                    current_action_values,
                    avg_strategy,
                    current_strategy,
                    regrets
             FROM solver_combos
             WHERE node_id = ?1",
        )
        .unwrap();
    let mut rows: Vec<Value> = statement
        .query_map([node_id], |row| {
            let avg_action_values_text = row.get::<_, Option<String>>(4)?;
            let current_action_values_text = row.get::<_, Option<String>>(5)?;
            let avg_strategy_text = row.get::<_, String>(6)?;
            let current_strategy_text = row.get::<_, String>(7)?;
            let avg_action_values = avg_action_values_text
                .as_deref()
                .and_then(parse_f32_json_array)
                .unwrap_or_default();
            let current_action_values = current_action_values_text
                .as_deref()
                .and_then(parse_f32_json_array)
                .unwrap_or_default();
            let avg_strategy = parse_f32_json_array(&avg_strategy_text).unwrap_or_default();
            let current_strategy = parse_f32_json_array(&current_strategy_text).unwrap_or_default();
            let avg_bet_probability = sum_indices(&avg_strategy, &bet_indices);
            let current_bet_probability = sum_indices(&current_strategy, &bet_indices);
            let check = check_index.expect("checked above");
            let avg_check_ev = avg_action_values.get(check).copied().unwrap_or(0.0);
            let current_check_ev = current_action_values.get(check).copied().unwrap_or(0.0);
            let (avg_best_bet_index, avg_best_bet_ev) =
                best_index_value(&avg_action_values, &bet_indices).unwrap_or((0, 0.0));
            let (current_best_bet_index, current_best_bet_ev) =
                best_index_value(&current_action_values, &bet_indices).unwrap_or((0, 0.0));
            let avg_bet_edge = avg_best_bet_ev - avg_check_ev;
            let current_bet_edge = current_best_bet_ev - current_check_ev;
            Ok(json!({
                "combo_index": row.get::<_, i64>(0)?,
                "combo": row.get::<_, String>(1)?,
                "reach": row.get::<_, f64>(2)?,
                "weighted_gap": row.get::<_, f64>(3)?,
                "avg_bet_probability": avg_bet_probability,
                "current_bet_probability": current_bet_probability,
                "avg_check_ev": avg_check_ev,
                "avg_best_bet": {
                    "index": avg_best_bet_index,
                    "action": actions.get(avg_best_bet_index).cloned().unwrap_or_default(),
                    "ev": avg_best_bet_ev,
                    "edge_vs_check": avg_bet_edge,
                },
                "current_check_ev": current_check_ev,
                "current_best_bet": {
                    "index": current_best_bet_index,
                    "action": actions.get(current_best_bet_index).cloned().unwrap_or_default(),
                    "ev": current_best_bet_ev,
                    "edge_vs_check": current_bet_edge,
                },
                "avg_action_values": avg_action_values_text.map(json_text_value),
                "current_action_values": current_action_values_text.map(json_text_value),
                "avg_strategy": json_text_value(avg_strategy_text),
                "current_strategy": json_text_value(current_strategy_text),
                "regrets": json_text_value(row.get::<_, String>(8)?),
            }))
        })
        .unwrap()
        .map(Result::unwrap)
        .collect();
    rows.sort_by(|left, right| {
        let left_score = left["current_best_bet"]["edge_vs_check"]
            .as_f64()
            .unwrap_or(0.0)
            * left["current_bet_probability"].as_f64().unwrap_or(0.0);
        let right_score = right["current_best_bet"]["edge_vs_check"]
            .as_f64()
            .unwrap_or(0.0)
            * right["current_bet_probability"].as_f64().unwrap_or(0.0);
        right_score
            .partial_cmp(&left_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows.truncate(limit);
    Value::Array(rows)
}

fn db_node_detail(conn: &Connection, node_id: i64, combo_limit: usize) -> Value {
    let node = conn
        .query_row(
            "SELECT node_id, parent_id, kind, infoset, path, street, board, acting_player, pot, to_call, hero_invested, villain_invested, terminal_kind
             FROM nodes
             WHERE node_id = ?1",
            [node_id],
            |row| {
                Ok(json!({
                    "node": row.get::<_, i64>(0)?,
                    "parent": row.get::<_, Option<i64>>(1)?,
                    "kind": row.get::<_, String>(2)?,
                    "infoset": row.get::<_, Option<i64>>(3)?,
                    "path": json_text_value(row.get::<_, String>(4)?),
                    "street": row.get::<_, Option<String>>(5)?,
                    "board": row.get::<_, Option<String>>(6)?,
                    "acting_player": row.get::<_, Option<String>>(7)?,
                    "pot": row.get::<_, Option<i64>>(8)?,
                    "to_call": row.get::<_, Option<i64>>(9)?,
                    "hero_invested": row.get::<_, Option<i64>>(10)?,
                    "villain_invested": row.get::<_, Option<i64>>(11)?,
                    "terminal_kind": row.get::<_, Option<String>>(12)?,
                }))
            },
        )
        .unwrap_or_else(|error| {
            eprintln!("failed to find node {node_id}: {error}");
            std::process::exit(2);
        });

    json!({
        "node": node,
        "actions": db_node_actions(conn, node_id),
        "solver": db_solver_node(conn, node_id),
        "top_combos_by_local_gap": db_node_combos(conn, node_id, "local", combo_limit),
        "top_combos_by_weighted_gap": db_node_combos(conn, node_id, "weighted", combo_limit),
        "top_combos_by_recursive_local_gap": db_node_combos(conn, node_id, "recursive_local", combo_limit),
        "top_combos_by_recursive_weighted_gap": db_node_combos(conn, node_id, "recursive_weighted", combo_limit),
        "top_combos_by_current_recursive_local_gap": db_node_combos(conn, node_id, "current_recursive_local", combo_limit),
        "top_combos_by_current_recursive_weighted_gap": db_node_combos(conn, node_id, "current_recursive_weighted", combo_limit),
    })
}

fn db_solver_node(conn: &Connection, node_id: i64) -> Value {
    conn.query_row(
        "SELECT infoset,
                iterations,
                acting_player,
                action_count,
                legal_combo_count,
                value_reach_sum,
                avg_strategy_weight_sum,
                current_strategy_weight_sum,
                avg_strategy,
                current_strategy,
                avg_action_ev,
                current_action_ev,
                avg_policy_ev,
                current_policy_ev,
                avg_gap,
                current_gap,
                avg_recursive_action_ev,
                current_recursive_action_ev,
                avg_recursive_policy_ev,
                current_recursive_policy_ev,
                avg_recursive_gap,
                current_recursive_gap
         FROM solver_nodes
         WHERE node_id = ?1",
        [node_id],
        |row| {
            Ok(json!({
                "infoset": row.get::<_, i64>(0)?,
                "iterations": row.get::<_, i64>(1)?,
                "acting_player": row.get::<_, String>(2)?,
                "action_count": row.get::<_, i64>(3)?,
                "legal_combo_count": row.get::<_, i64>(4)?,
                "value_reach_sum": row.get::<_, f64>(5)?,
                "avg_strategy_weight_sum": row.get::<_, f64>(6)?,
                "current_strategy_weight_sum": row.get::<_, f64>(7)?,
                "avg_strategy": json_text_value(row.get::<_, String>(8)?),
                "current_strategy": json_text_value(row.get::<_, String>(9)?),
                "avg_action_ev": row.get::<_, Option<String>>(10)?.map(json_text_value),
                "current_action_ev": row.get::<_, Option<String>>(11)?.map(json_text_value),
                "avg_policy_ev": row.get::<_, Option<f64>>(12)?,
                "current_policy_ev": row.get::<_, Option<f64>>(13)?,
                "avg_gap": row.get::<_, Option<f64>>(14)?,
                "current_gap": row.get::<_, Option<f64>>(15)?,
                "avg_recursive_action_ev": row.get::<_, Option<String>>(16)?.map(json_text_value),
                "current_recursive_action_ev": row.get::<_, Option<String>>(17)?.map(json_text_value),
                "avg_recursive_policy_ev": row.get::<_, Option<f64>>(18)?,
                "current_recursive_policy_ev": row.get::<_, Option<f64>>(19)?,
                "avg_recursive_gap": row.get::<_, Option<f64>>(20)?,
                "current_recursive_gap": row.get::<_, Option<f64>>(21)?,
            }))
        },
    )
    .unwrap_or(Value::Null)
}

fn db_node_combos(conn: &Connection, node_id: i64, order: &str, limit: usize) -> Value {
    let order_sql = match order {
        "local" => "CASE WHEN reach > 0.0 THEN weighted_gap / reach ELSE 0.0 END DESC",
        "weighted" => "weighted_gap DESC",
        "recursive_local" => {
            "CASE WHEN reach > 0.0 THEN recursive_weighted_gap / reach ELSE 0.0 END DESC"
        }
        "recursive_weighted" => "recursive_weighted_gap DESC",
        "current_recursive_local" => {
            "CASE WHEN reach > 0.0 THEN current_recursive_weighted_gap / reach ELSE 0.0 END DESC"
        }
        "current_recursive_weighted" => "current_recursive_weighted_gap DESC",
        _ => unreachable!("unknown combo order"),
    };
    let sql = format!(
        "SELECT combo_index,
                combo,
                reach,
                weighted_gap,
                recursive_weighted_gap,
                current_recursive_weighted_gap,
                avg_strategy_weight,
                current_strategy_weight,
                avg_action_values,
                current_action_values,
                avg_recursive_action_values,
                current_recursive_action_values,
                avg_strategy,
                current_strategy,
                regrets,
                strategy_sum
         FROM solver_combos
         WHERE node_id = ?1
         ORDER BY {order_sql}
         LIMIT ?2"
    );
    let mut statement = conn.prepare(&sql).unwrap();
    let rows = statement
        .query_map(params![node_id, db_usize(limit)], |row| {
            let reach = row.get::<_, f64>(2)?;
            let weighted_gap = row.get::<_, f64>(3)?;
            let recursive_weighted_gap = row.get::<_, f64>(4)?;
            let current_recursive_weighted_gap = row.get::<_, f64>(5)?;
            let avg_strategy_weight = row.get::<_, f64>(6)?;
            let current_strategy_weight = row.get::<_, f64>(7)?;
            let avg_action_values = row.get::<_, Option<String>>(8)?;
            let current_action_values = row.get::<_, Option<String>>(9)?;
            let avg_recursive_action_values = row.get::<_, Option<String>>(10)?;
            let current_recursive_action_values = row.get::<_, Option<String>>(11)?;
            let avg_strategy = row.get::<_, String>(12)?;
            let current_strategy = row.get::<_, String>(13)?;
            let regrets = row.get::<_, String>(14)?;
            let strategy_sum = row.get::<_, String>(15)?;
            let avg_action_values_json = avg_action_values.clone().map(json_text_value);
            let current_action_values_json = current_action_values.clone().map(json_text_value);
            let avg_recursive_action_values_json =
                avg_recursive_action_values.clone().map(json_text_value);
            let current_recursive_action_values_json =
                current_recursive_action_values.clone().map(json_text_value);
            let avg_strategy_json = json_text_value(avg_strategy.clone());
            let current_strategy_json = json_text_value(current_strategy.clone());
            let regrets_json = json_text_value(regrets.clone());
            let strategy_sum_json = json_text_value(strategy_sum.clone());
            Ok(json!({
                "combo_index": row.get::<_, i64>(0)?,
                "combo": row.get::<_, String>(1)?,
                "reach": reach,
                "weighted_gap": weighted_gap,
                "local_gap": local_gap(reach, weighted_gap),
                "recursive_weighted_gap": recursive_weighted_gap,
                "recursive_local_gap": local_gap(reach, recursive_weighted_gap),
                "current_recursive_weighted_gap": current_recursive_weighted_gap,
                "current_recursive_local_gap": local_gap(reach, current_recursive_weighted_gap),
                "avg_strategy_weight": avg_strategy_weight,
                "current_strategy_weight": current_strategy_weight,
                "avg_best": avg_action_values.as_deref().and_then(best_action_summary),
                "current_best": current_action_values.as_deref().and_then(best_action_summary),
                "avg_recursive_best": avg_recursive_action_values.as_deref().and_then(best_action_summary),
                "current_recursive_best": current_recursive_action_values.as_deref().and_then(best_action_summary),
                "avg_action_values": avg_action_values_json,
                "current_action_values": current_action_values_json,
                "avg_recursive_action_values": avg_recursive_action_values_json,
                "current_recursive_action_values": current_recursive_action_values_json,
                "avg_strategy": avg_strategy_json,
                "current_strategy": current_strategy_json,
                "regrets": regrets_json,
                "regret_best": best_action_summary(&regrets),
                "strategy_sum": strategy_sum_json,
                "strategy_sum_total": json_f32_sum(&strategy_sum),
            }))
        })
        .unwrap();
    Value::Array(rows.map(Result::unwrap).collect())
}

fn local_gap(reach: f64, weighted_gap: f64) -> Option<f64> {
    (reach > 0.0).then_some(weighted_gap / reach)
}

fn best_action_summary(json_text: &str) -> Option<Value> {
    let values = parse_f32_json_array(json_text)?;
    let (index, value) = values
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })?;
    Some(json!({ "index": index, "value": value }))
}

fn json_f32_sum(json_text: &str) -> f64 {
    parse_f32_json_array(json_text)
        .map(|values| values.into_iter().map(f64::from).sum())
        .unwrap_or(0.0)
}

fn parse_f32_json_array(json_text: &str) -> Option<Vec<f32>> {
    serde_json::from_str::<Vec<f32>>(json_text).ok()
}

fn db_node_actions(conn: &Connection, node_id: i64) -> Value {
    let mut statement = conn
        .prepare("SELECT action_index, action, child_id, source FROM actions WHERE node_id = ?1 ORDER BY action_index")
        .unwrap();
    let rows = statement
        .query_map([node_id], |row| {
            Ok(json!({
                "index": row.get::<_, i64>(0)?,
                "action": row.get::<_, String>(1)?,
                "child": row.get::<_, i64>(2)?,
                "source": row.get::<_, String>(3)?,
            }))
        })
        .unwrap();
    Value::Array(rows.map(Result::unwrap).collect())
}

fn db_root_action_subtrees(conn: &Connection) -> Value {
    let mut statement = conn
        .prepare("SELECT action_index, action, child_id FROM actions WHERE node_id = 0 ORDER BY action_index")
        .unwrap();
    let rows = statement
        .query_map([], |row| {
            let action_index: i64 = row.get(0)?;
            let action: String = row.get(1)?;
            let child_id: i64 = row.get(2)?;
            Ok(json!({
                "action_index": action_index,
                "action": action,
                "child": child_id,
                "solver_summary": db_subtree_solver_summary(conn, child_id),
                "top_avg_gaps": db_subtree_top_gaps(conn, child_id, "avg_gap", "avg_action_ev", "avg_policy_ev"),
                "top_current_gaps": db_subtree_top_gaps(conn, child_id, "current_gap", "current_action_ev", "current_policy_ev"),
                "top_recursive_avg_gaps": db_subtree_top_gaps(conn, child_id, "avg_recursive_gap", "avg_recursive_action_ev", "avg_recursive_policy_ev"),
                "top_recursive_current_gaps": db_subtree_top_gaps(conn, child_id, "current_recursive_gap", "current_recursive_action_ev", "current_recursive_policy_ev"),
            }))
        })
        .unwrap();
    Value::Array(rows.map(Result::unwrap).collect())
}

fn db_subtree_solver_summary(conn: &Connection, root: i64) -> Value {
    conn.query_row(
        "WITH RECURSIVE subtree(node_id) AS (
            SELECT ?1
            UNION ALL
            SELECT n.node_id FROM nodes n JOIN subtree s ON n.parent_id = s.node_id
         )
         SELECT COUNT(solver_nodes.node_id),
                AVG(avg_gap),
                AVG(current_gap),
                MAX(avg_gap),
                MAX(current_gap),
                AVG(avg_recursive_gap),
                AVG(current_recursive_gap),
                MAX(avg_recursive_gap),
                MAX(current_recursive_gap)
         FROM subtree LEFT JOIN solver_nodes ON solver_nodes.node_id = subtree.node_id",
        [root],
        |row| {
            Ok(json!({
                "solver_nodes": row.get::<_, i64>(0)?,
                "mean_avg_gap": row.get::<_, Option<f64>>(1)?.unwrap_or(0.0),
                "mean_current_gap": row.get::<_, Option<f64>>(2)?.unwrap_or(0.0),
                "max_avg_gap": row.get::<_, Option<f64>>(3)?.unwrap_or(0.0),
                "max_current_gap": row.get::<_, Option<f64>>(4)?.unwrap_or(0.0),
                "mean_avg_recursive_gap": row.get::<_, Option<f64>>(5)?.unwrap_or(0.0),
                "mean_current_recursive_gap": row.get::<_, Option<f64>>(6)?.unwrap_or(0.0),
                "max_avg_recursive_gap": row.get::<_, Option<f64>>(7)?.unwrap_or(0.0),
                "max_current_recursive_gap": row.get::<_, Option<f64>>(8)?.unwrap_or(0.0),
            }))
        },
    )
    .unwrap()
}

fn db_subtree_top_gaps(
    conn: &Connection,
    root: i64,
    gap_column: &str,
    action_ev_column: &str,
    policy_column: &str,
) -> Value {
    let sql = format!(
        "WITH RECURSIVE subtree(node_id) AS (
            SELECT ?1
            UNION ALL
            SELECT n.node_id FROM nodes n JOIN subtree s ON n.parent_id = s.node_id
         )
         SELECT n.node_id, n.street, n.acting_player, n.pot, n.to_call, n.path, s.{gap_column}, s.{action_ev_column}, s.{policy_column}, s.avg_strategy, s.current_strategy
         FROM subtree JOIN solver_nodes s ON s.node_id = subtree.node_id
                      JOIN nodes n ON n.node_id = subtree.node_id
         WHERE s.{gap_column} IS NOT NULL
         ORDER BY s.{gap_column} DESC
         LIMIT 5"
    );
    let mut statement = conn.prepare(&sql).unwrap();
    let rows = statement
        .query_map([root], |row| {
            let node_id = row.get::<_, i64>(0)?;
            Ok(json!({
                "node": node_id,
                "street": row.get::<_, Option<String>>(1)?,
                "acting_player": row.get::<_, Option<String>>(2)?,
                "pot": row.get::<_, Option<i64>>(3)?,
                "to_call": row.get::<_, Option<i64>>(4)?,
                "path": json_text_value(row.get::<_, String>(5)?),
                "gap": row.get::<_, f64>(6)?,
                "action_ev": row.get::<_, Option<String>>(7)?.map(json_text_value),
                "policy_ev": row.get::<_, Option<f64>>(8)?,
                "avg_strategy": json_text_value(row.get::<_, String>(9)?),
                "current_strategy": json_text_value(row.get::<_, String>(10)?),
                "actions": db_node_actions(conn, node_id),
            }))
        })
        .unwrap();
    Value::Array(rows.map(Result::unwrap).collect())
}

fn db_subtree_top_reached_gaps(
    conn: &Connection,
    root: i64,
    gap_column: &str,
    action_ev_column: &str,
    policy_column: &str,
    strategy_weight_column: &str,
    limit: usize,
) -> Value {
    let sql = format!(
        "WITH RECURSIVE subtree(node_id) AS (
            SELECT ?1
            UNION ALL
            SELECT n.node_id FROM nodes n JOIN subtree s ON n.parent_id = s.node_id
         )
         SELECT n.node_id, n.street, n.acting_player, n.pot, n.to_call, n.path, s.{gap_column}, s.{action_ev_column}, s.{policy_column}, s.avg_strategy, s.current_strategy, s.value_reach_sum, s.avg_strategy_weight_sum, s.current_strategy_weight_sum
         FROM subtree JOIN solver_nodes s ON s.node_id = subtree.node_id
                      JOIN nodes n ON n.node_id = subtree.node_id
         WHERE s.{gap_column} IS NOT NULL AND s.{strategy_weight_column} > 1.0e-6
         ORDER BY s.{gap_column} DESC
         LIMIT ?2"
    );
    let mut statement = conn.prepare(&sql).unwrap();
    let rows = statement
        .query_map(params![root, db_usize(limit)], |row| {
            let node_id = row.get::<_, i64>(0)?;
            Ok(json!({
                "node": node_id,
                "street": row.get::<_, Option<String>>(1)?,
                "acting_player": row.get::<_, Option<String>>(2)?,
                "pot": row.get::<_, Option<i64>>(3)?,
                "to_call": row.get::<_, Option<i64>>(4)?,
                "path": json_text_value(row.get::<_, String>(5)?),
                "gap": row.get::<_, f64>(6)?,
                "action_ev": row.get::<_, Option<String>>(7)?.map(json_text_value),
                "policy_ev": row.get::<_, Option<f64>>(8)?,
                "avg_strategy": json_text_value(row.get::<_, String>(9)?),
                "current_strategy": json_text_value(row.get::<_, String>(10)?),
                "value_reach_sum": row.get::<_, f64>(11)?,
                "avg_strategy_weight_sum": row.get::<_, f64>(12)?,
                "current_strategy_weight_sum": row.get::<_, f64>(13)?,
                "actions": db_node_actions(conn, node_id),
            }))
        })
        .unwrap();
    Value::Array(rows.map(Result::unwrap).collect())
}

fn db_subtree_aggressive_nodes(
    conn: &Connection,
    root: i64,
    acting_player: &str,
    street: &str,
    limit: usize,
) -> Value {
    let mut statement = conn
        .prepare(
            "WITH RECURSIVE subtree(node_id) AS (
                SELECT ?1
                UNION ALL
                SELECT n.node_id FROM nodes n JOIN subtree s ON n.parent_id = s.node_id
             )
             SELECT n.node_id,
                    n.pot,
                    n.to_call,
                    n.path,
                    s.avg_action_ev,
                    s.current_action_ev,
                    s.avg_policy_ev,
                    s.current_policy_ev,
                    s.avg_strategy,
                    s.current_strategy,
                    s.avg_strategy_weight_sum,
                    s.current_strategy_weight_sum,
                    s.avg_gap,
                    s.current_gap
             FROM subtree JOIN solver_nodes s ON s.node_id = subtree.node_id
                          JOIN nodes n ON n.node_id = subtree.node_id
             WHERE n.acting_player = ?2
               AND n.street = ?3
               AND n.to_call = 0
               AND s.current_strategy_weight_sum > 1.0e-6",
        )
        .unwrap();
    let mut rows: Vec<Value> = statement
        .query_map(params![root, acting_player, street], |row| {
            let node_id = row.get::<_, i64>(0)?;
            let actions = db_action_labels(conn, node_id);
            let bet_indices: Vec<_> = actions
                .iter()
                .enumerate()
                .filter_map(|(index, action)| action.starts_with("bet:").then_some(index))
                .collect();
            let avg_strategy_text = row.get::<_, String>(8)?;
            let current_strategy_text = row.get::<_, String>(9)?;
            let avg_strategy = parse_f32_json_array(&avg_strategy_text).unwrap_or_default();
            let current_strategy = parse_f32_json_array(&current_strategy_text).unwrap_or_default();
            let avg_bet_probability = sum_indices(&avg_strategy, &bet_indices);
            let current_bet_probability = sum_indices(&current_strategy, &bet_indices);
            Ok(json!({
                "node": node_id,
                "pot": row.get::<_, Option<i64>>(1)?,
                "to_call": row.get::<_, Option<i64>>(2)?,
                "path": json_text_value(row.get::<_, String>(3)?),
                "avg_action_ev": row.get::<_, Option<String>>(4)?.map(json_text_value),
                "current_action_ev": row.get::<_, Option<String>>(5)?.map(json_text_value),
                "avg_policy_ev": row.get::<_, Option<f64>>(6)?,
                "current_policy_ev": row.get::<_, Option<f64>>(7)?,
                "avg_strategy": json_text_value(avg_strategy_text),
                "current_strategy": json_text_value(current_strategy_text),
                "avg_bet_probability": avg_bet_probability,
                "current_bet_probability": current_bet_probability,
                "avg_strategy_weight_sum": row.get::<_, f64>(10)?,
                "current_strategy_weight_sum": row.get::<_, f64>(11)?,
                "avg_gap": row.get::<_, Option<f64>>(12)?,
                "current_gap": row.get::<_, Option<f64>>(13)?,
                "actions": db_node_actions(conn, node_id),
            }))
        })
        .unwrap()
        .map(Result::unwrap)
        .collect();
    rows.sort_by(|left, right| {
        let left_score = left["current_bet_probability"].as_f64().unwrap_or(0.0)
            * left["current_strategy_weight_sum"].as_f64().unwrap_or(0.0);
        let right_score = right["current_bet_probability"].as_f64().unwrap_or(0.0)
            * right["current_strategy_weight_sum"].as_f64().unwrap_or(0.0);
        right_score
            .partial_cmp(&left_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows.truncate(limit);
    Value::Array(rows)
}

fn db_action_labels(conn: &Connection, node_id: i64) -> Vec<String> {
    let mut statement = conn
        .prepare("SELECT action FROM actions WHERE node_id = ?1 ORDER BY action_index")
        .unwrap();
    statement
        .query_map([node_id], |row| row.get::<_, String>(0))
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

fn sum_indices(values: &[f32], indices: &[usize]) -> f64 {
    indices
        .iter()
        .filter_map(|index| values.get(*index))
        .map(|value| f64::from(*value))
        .sum()
}

fn best_index_value(values: &[f32], indices: &[usize]) -> Option<(usize, f32)> {
    indices
        .iter()
        .filter_map(|index| values.get(*index).copied().map(|value| (*index, value)))
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn vec_json(values: &[f32]) -> String {
    serde_json::to_string(values).expect("float array must serialize")
}

fn opt_vec_json(values: &Option<Vec<f32>>) -> Option<String> {
    values.as_ref().map(|values| vec_json(values))
}

fn json_text_value(text: String) -> Value {
    serde_json::from_str(&text).unwrap_or(Value::String(text))
}

fn db_usize(value: usize) -> i64 {
    value as i64
}

fn db_opt_usize(value: Option<usize>) -> Option<i64> {
    value.map(db_usize)
}

fn db_opt_u32(value: Option<u32>) -> Option<i64> {
    value.map(i64::from)
}

fn smoke_hands() -> usize {
    std::env::var("POKEDR_HANDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(16)
}

fn parse_tree_db_build_args(args: &mut impl Iterator<Item = String>) -> TreeDbBuildArgs {
    let Some(db_path) = args.next() else {
        eprintln!(
            "usage: pokedr-cli tree-db build <tree.sqlite> [flop] [--config path.yml] [--combo-limit n|--full-combos]"
        );
        std::process::exit(2);
    };
    let mut parsed = TreeDbBuildArgs {
        db_path,
        flop: None,
        config_path: None,
        combo_limit: 32,
        iterations: None,
        max_depth: None,
    };
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => {
                parsed.config_path = Some(args.next().unwrap_or_else(|| {
                    eprintln!("--config requires a path");
                    std::process::exit(2);
                }));
            }
            "--combo-limit" => {
                let value = args.next().unwrap_or_else(|| {
                    eprintln!("--combo-limit requires a number");
                    std::process::exit(2);
                });
                parsed.combo_limit = value.parse().unwrap_or_else(|_| {
                    eprintln!("--combo-limit must be a number: {value}");
                    std::process::exit(2);
                });
            }
            "--full-combos" => {
                parsed.combo_limit = usize::MAX;
            }
            "--iterations" => {
                let value = args.next().unwrap_or_else(|| {
                    eprintln!("--iterations requires a number");
                    std::process::exit(2);
                });
                parsed.iterations = Some(value.parse().unwrap_or_else(|_| {
                    eprintln!("--iterations must be a number: {value}");
                    std::process::exit(2);
                }));
            }
            "--max-depth" => {
                let value = args.next().unwrap_or_else(|| {
                    eprintln!("--max-depth requires a number");
                    std::process::exit(2);
                });
                parsed.max_depth = Some(value.parse().unwrap_or_else(|_| {
                    eprintln!("--max-depth must be a number: {value}");
                    std::process::exit(2);
                }));
            }
            value if value.starts_with("--") => {
                eprintln!("unknown tree-db build option: {value}");
                std::process::exit(2);
            }
            value => {
                if parsed.flop.replace(value.to_string()).is_some() {
                    eprintln!("multiple flop arguments are not supported");
                    std::process::exit(2);
                }
            }
        }
    }
    parsed
}

fn parse_flop_command_args(args: impl Iterator<Item = String>) -> FlopCommandArgs {
    let mut parsed = FlopCommandArgs::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" | "-c" => {
                let Some(path) = args.next() else {
                    eprintln!("{arg} requires a YAML path");
                    std::process::exit(2);
                };
                parsed.config_path = Some(path);
            }
            _ if arg.starts_with("--config=") => {
                parsed.config_path = Some(arg["--config=".len()..].to_string());
            }
            _ if parsed.flop.is_none() => parsed.flop = Some(arg),
            _ => {
                eprintln!("unexpected argument: {arg}");
                std::process::exit(2);
            }
        }
    }
    parsed
}

fn load_cli_config(path: Option<&str>) -> Option<CliConfigFile> {
    let env_path = std::env::var("POKEDR_CONFIG").ok();
    let path = path.map(str::to_string).or(env_path)?;
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        eprintln!("failed to read config {path}: {error}");
        std::process::exit(2);
    });
    Some(serde_yaml::from_str(&text).unwrap_or_else(|error| {
        eprintln!("failed to parse config {path}: {error}");
        std::process::exit(2);
    }))
}

fn metric_iterations(config: Option<&CliConfigFile>) -> Vec<usize> {
    std::env::var("POKEDR_METRIC_ITERATIONS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|value| *value > 0)
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .or_else(|| {
            config
                .and_then(|config| config.metric_iterations.clone())
                .map(|values| {
                    values
                        .into_iter()
                        .filter(|value| *value > 0)
                        .collect::<Vec<_>>()
                })
                .filter(|values| !values.is_empty())
        })
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32])
}

#[derive(Debug, Clone, Copy)]
struct MetricConvergenceSettings {
    interval: usize,
    max_iterations: usize,
    target_bb100: f32,
    regression_patience: usize,
}

fn metric_convergence_settings() -> Option<MetricConvergenceSettings> {
    let requested = std::env::var_os("POKEDR_METRIC_UNTIL_CONVERGED").is_some()
        || std::env::var_os("POKEDR_METRIC_INTERVAL").is_some()
        || std::env::var_os("POKEDR_METRIC_MAX_ITERATIONS").is_some()
        || std::env::var_os("POKEDR_METRIC_TARGET_BB100").is_some();
    requested.then(|| {
        let interval = env_usize("POKEDR_METRIC_INTERVAL").unwrap_or(64).max(1);
        let max_iterations = env_usize("POKEDR_METRIC_MAX_ITERATIONS")
            .unwrap_or(4096)
            .max(interval);
        MetricConvergenceSettings {
            interval,
            max_iterations,
            target_bb100: env_f32("POKEDR_METRIC_TARGET_BB100").unwrap_or(1.0),
            regression_patience: env_usize("POKEDR_METRIC_REGRESSION_PATIENCE").unwrap_or(3),
        }
    })
}

fn metric_convergence_iterations(settings: &MetricConvergenceSettings) -> Vec<usize> {
    let mut iterations = Vec::new();
    let mut next = settings.interval;
    while next < settings.max_iterations {
        iterations.push(next);
        next = next.saturating_add(settings.interval);
    }
    if iterations.last().copied() != Some(settings.max_iterations) {
        iterations.push(settings.max_iterations);
    }
    iterations
}

fn fixed_flop_config(cli_config: Option<&CliConfigFile>) -> pokedr_agent::PokedrAgentConfig {
    let mut config = match_config(cli_config);
    if env_usize("POKEDR_CFR_ITERATIONS").is_none()
        && cli_config
            .and_then(|config| config.cfr_iterations)
            .is_none()
    {
        config.cfr_iterations = 1;
    }
    config
}

fn match_config(cli_config: Option<&CliConfigFile>) -> pokedr_agent::PokedrAgentConfig {
    let mut config = pokedr_agent::PokedrAgentConfig::default();
    if let Some(file) = cli_config {
        apply_cli_config(&mut config, file);
    }
    config.cfr_iterations = env_usize("POKEDR_CFR_ITERATIONS").unwrap_or(config.cfr_iterations);
    config.cfr_variant = env_cfr_variant("POKEDR_CFR_VARIANT").unwrap_or(config.cfr_variant);
    config.max_depth = env_usize("POKEDR_MAX_DEPTH").unwrap_or(config.max_depth);
    if let Some(value) = env_usize("POKEDR_MAX_RAISES_PER_STREET") {
        config.max_raises_per_street = value.min(u8::MAX as usize) as u8;
    }
    config.max_showdown_runouts =
        env_usize("POKEDR_MAX_SHOWDOWN_RUNOUTS").unwrap_or(config.max_showdown_runouts);
    config
}

fn apply_cli_config(config: &mut pokedr_agent::PokedrAgentConfig, file: &CliConfigFile) {
    if let Some(value) = file.cfr_iterations {
        config.cfr_iterations = value;
    }
    if let Some(value) = file
        .cfr_variant
        .as_deref()
        .map(|value| parse_cfr_variant(value, file))
    {
        config.cfr_variant = value;
    }
    if let Some(value) = file.max_depth {
        config.max_depth = value;
    }
    if let Some(value) = file.max_raises_per_street {
        config.max_raises_per_street = value;
    }
    if let Some(value) = file.max_showdown_runouts {
        config.max_showdown_runouts = value;
    }
    if let Some(action_set) = &file.action_set {
        apply_action_set_config(&mut config.action_set, action_set);
    }
}

fn apply_action_set_config(config: &mut ActionSetConfig, file: &CliActionSetConfig) {
    if let Some(value) = file.max_aggressive_actions {
        config.max_aggressive_actions = value;
    }
    if let Some(value) = file.merge_log_ratio {
        config.merge_log_ratio = value;
    }
    if let Some(value) = file.all_in_threshold {
        config.all_in_threshold = value;
    }
    if let Some(value) = &file.flop_bet_fractions {
        config.flop_bet_fractions = clean_fractions("action_set.flop_bet_fractions", value);
    }
    if let Some(value) = &file.turn_bet_fractions {
        config.turn_bet_fractions = clean_fractions("action_set.turn_bet_fractions", value);
    }
    if let Some(value) = &file.river_bet_fractions {
        config.river_bet_fractions = clean_fractions("action_set.river_bet_fractions", value);
    }
    if let Some(value) = &file.raise_fractions {
        config.raise_fractions = clean_fractions("action_set.raise_fractions", value);
    }
}

fn clean_fractions(name: &str, values: &[f32]) -> Vec<f32> {
    let cleaned = values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    if cleaned.is_empty() {
        eprintln!("{name} must contain at least one positive finite value");
        std::process::exit(2);
    }
    cleaned
}

fn env_cfr_variant(name: &str) -> Option<CfrVariant> {
    let value = std::env::var(name).ok()?;
    Some(parse_env_cfr_variant(name, &value))
}

fn parse_env_cfr_variant(name: &str, value: &str) -> CfrVariant {
    match normalize_variant(value).as_str() {
        "cfr" | "cfr-plus" | "cfrplus" | "plus" => CfrVariant::CfrPlus,
        "discounted" | "dcfr" => CfrVariant::Discounted,
        "dcfr-plus" | "dcfrplus" | "dcfr+" => CfrVariant::DcfrPlus {
            alpha: env_f32("POKEDR_DCFR_ALPHA")
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_DCFR_PLUS_ALPHA),
            gamma: env_f32("POKEDR_DCFR_GAMMA")
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_DCFR_PLUS_GAMMA),
        },
        "dcfr-schedule" | "dcfrschedule" | "hs-dcfr" | "hsdcfr" => CfrVariant::DcfrSchedule {
            alpha_start: env_f32("POKEDR_DCFR_SCHEDULE_ALPHA_START")
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_DCFR_SCHEDULE_ALPHA_START),
            alpha_end: env_f32("POKEDR_DCFR_SCHEDULE_ALPHA_END")
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_DCFR_SCHEDULE_ALPHA_END),
            gamma_start: env_f32("POKEDR_DCFR_SCHEDULE_GAMMA_START")
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_DCFR_SCHEDULE_GAMMA_START),
            gamma_end: env_f32("POKEDR_DCFR_SCHEDULE_GAMMA_END")
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_DCFR_SCHEDULE_GAMMA_END),
            horizon: env_usize("POKEDR_DCFR_SCHEDULE_HORIZON")
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_DCFR_SCHEDULE_HORIZON),
        },
        "pdcfr-plus" | "pdcfrplus" | "pdcfr+" | "pdcfr" => CfrVariant::PdcfrPlus {
            alpha: env_f32("POKEDR_PDCFR_ALPHA")
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_PDCFR_PLUS_ALPHA),
            gamma: env_f32("POKEDR_PDCFR_GAMMA")
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_PDCFR_PLUS_GAMMA),
            eta_start: env_f32("POKEDR_PDCFR_ETA_START")
                .or_else(|| env_f32("POKEDR_PDCFR_ETA"))
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_PDCFR_PLUS_ETA_START),
            eta_end: env_f32("POKEDR_PDCFR_ETA_END")
                .or_else(|| env_f32("POKEDR_PDCFR_ETA"))
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_PDCFR_PLUS_ETA),
            eta_horizon: env_usize("POKEDR_PDCFR_ETA_HORIZON")
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_PDCFR_PLUS_ETA_HORIZON),
        },
        other => {
            eprintln!(
                "unknown {name}={other}; expected cfr-plus, discounted, dcfr-plus, dcfr-schedule, or pdcfr-plus"
            );
            std::process::exit(2);
        }
    }
}

fn parse_cfr_variant(value: &str, config: &CliConfigFile) -> CfrVariant {
    match normalize_variant(value).as_str() {
        "cfr" | "cfr-plus" | "cfrplus" | "plus" => CfrVariant::CfrPlus,
        "discounted" | "dcfr" => CfrVariant::Discounted,
        "dcfr-plus" | "dcfrplus" | "dcfr+" => CfrVariant::DcfrPlus {
            alpha: config
                .dcfr_alpha
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_DCFR_PLUS_ALPHA),
            gamma: config
                .dcfr_gamma
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_DCFR_PLUS_GAMMA),
        },
        "dcfr-schedule" | "dcfrschedule" | "hs-dcfr" | "hsdcfr" => CfrVariant::DcfrSchedule {
            alpha_start: config
                .dcfr_schedule_alpha_start
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_DCFR_SCHEDULE_ALPHA_START),
            alpha_end: config
                .dcfr_schedule_alpha_end
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_DCFR_SCHEDULE_ALPHA_END),
            gamma_start: config
                .dcfr_schedule_gamma_start
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_DCFR_SCHEDULE_GAMMA_START),
            gamma_end: config
                .dcfr_schedule_gamma_end
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_DCFR_SCHEDULE_GAMMA_END),
            horizon: config
                .dcfr_schedule_horizon
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_DCFR_SCHEDULE_HORIZON),
        },
        "pdcfr-plus" | "pdcfrplus" | "pdcfr+" | "pdcfr" => CfrVariant::PdcfrPlus {
            alpha: config
                .pdcfr_alpha
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_PDCFR_PLUS_ALPHA),
            gamma: config
                .pdcfr_gamma
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_PDCFR_PLUS_GAMMA),
            eta_start: config
                .pdcfr_eta_start
                .or(config.pdcfr_eta)
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_PDCFR_PLUS_ETA_START),
            eta_end: config
                .pdcfr_eta_end
                .or(config.pdcfr_eta)
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_PDCFR_PLUS_ETA),
            eta_horizon: config
                .pdcfr_eta_horizon
                .unwrap_or(pokedr_core::dense_cfr::DEFAULT_PDCFR_PLUS_ETA_HORIZON),
        },
        other => {
            eprintln!(
                "unknown cfr_variant={other}; expected cfr-plus, discounted, dcfr-plus, dcfr-schedule, or pdcfr-plus"
            );
            std::process::exit(2);
        }
    }
}

fn normalize_variant(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('_', "-")
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

fn env_f32_list(name: &str) -> Option<Vec<f32>> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<f32>().ok())
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
}

fn env_eta_schedule_list(name: &str) -> Option<Vec<(f32, f32, usize)>> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| parse_eta_schedule(part.trim()))
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
}

fn parse_eta_schedule(input: &str) -> Option<(f32, f32, usize)> {
    let mut parts = input.split(':');
    let start = parts.next()?.parse::<f32>().ok()?;
    let end = parts.next()?.parse::<f32>().ok()?;
    let horizon = parts.next()?.parse::<usize>().ok()?;
    if parts.next().is_some() || !start.is_finite() || !end.is_finite() || horizon == 0 {
        return None;
    }
    Some((start, end, horizon))
}

fn format_pokedr_cards_for_cli(cards: &[Card]) -> String {
    cards
        .iter()
        .map(|card| format_pokedr_card_for_cli(*card))
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_pokedr_card_for_cli(card: Card) -> String {
    let rank = match card.rank() {
        Rank::Two => "2",
        Rank::Three => "3",
        Rank::Four => "4",
        Rank::Five => "5",
        Rank::Six => "6",
        Rank::Seven => "7",
        Rank::Eight => "8",
        Rank::Nine => "9",
        Rank::Ten => "T",
        Rank::Jack => "J",
        Rank::Queen => "Q",
        Rank::King => "K",
        Rank::Ace => "A",
    };
    let suit = match card.suit() {
        Suit::Clubs => "c",
        Suit::Diamonds => "d",
        Suit::Hearts => "h",
        Suit::Spades => "s",
    };
    format!("{rank}{suit}")
}

fn parse_flop(input: &str) -> Result<[Card; 3], String> {
    let normalized = input
        .split(|character: char| character == ',' || character.is_whitespace())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    let tokens = if normalized.len() == 3 {
        normalized
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>()
    } else {
        split_compact_cards(input)?
    };
    if tokens.len() != 3 {
        return Err("flop must contain exactly 3 cards, for example As7h2c".to_string());
    }
    let cards = [
        parse_card(&tokens[0])?,
        parse_card(&tokens[1])?,
        parse_card(&tokens[2])?,
    ];
    let mask = cards
        .iter()
        .fold(0u64, |mask, card| mask | card.deck_mask());
    if mask.count_ones() != 3 {
        return Err("flop contains duplicate cards".to_string());
    }
    Ok(cards)
}

fn split_compact_cards(input: &str) -> Result<Vec<String>, String> {
    let chars = input
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<Vec<_>>();
    let mut cards = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        if index + 1 >= chars.len() {
            return Err("card must include rank and suit".to_string());
        }
        let rank_len = if chars[index] == '1' && chars.get(index + 1) == Some(&'0') {
            2
        } else {
            1
        };
        if index + rank_len >= chars.len() {
            return Err("card must include rank and suit".to_string());
        }
        let token = chars[index..=index + rank_len].iter().collect();
        cards.push(token);
        index += rank_len + 1;
    }
    Ok(cards)
}

fn parse_card(input: &str) -> Result<Card, String> {
    let lower = input.trim().to_ascii_lowercase();
    if lower.len() < 2 {
        return Err(format!("invalid card: {input}"));
    }
    let (rank_part, suit_part) = lower.split_at(lower.len() - 1);
    let rank = match rank_part {
        "2" => Rank::Two,
        "3" => Rank::Three,
        "4" => Rank::Four,
        "5" => Rank::Five,
        "6" => Rank::Six,
        "7" => Rank::Seven,
        "8" => Rank::Eight,
        "9" => Rank::Nine,
        "t" | "10" => Rank::Ten,
        "j" => Rank::Jack,
        "q" => Rank::Queen,
        "k" => Rank::King,
        "a" => Rank::Ace,
        _ => return Err(format!("invalid rank in card: {input}")),
    };
    let suit = match suit_part {
        "c" => Suit::Clubs,
        "d" => Suit::Diamonds,
        "h" => Suit::Hearts,
        "s" => Suit::Spades,
        _ => return Err(format!("invalid suit in card: {input}")),
    };
    Ok(Card::new(rank, suit))
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
                max_aggressive_actions: 4,
                flop_bet_fractions: vec![0.5, 1.0, 1.5],
                turn_bet_fractions: vec![0.5, 1.0, 1.5],
                river_bet_fractions: vec![0.5, 1.0, 1.5],
                raise_fractions: vec![0.5, 1.0],
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
