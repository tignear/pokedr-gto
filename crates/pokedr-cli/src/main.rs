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
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct CliConfigFile {
    cfr_iterations: Option<usize>,
    cfr_variant: Option<String>,
    max_depth: Option<usize>,
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
    let root_converged = root_exploitability_bb100
        .map(|value| value <= target_bb100)
        .unwrap_or(false);
    println!(
        "board={} iterations={} elapsed={:.2}s root_l1_delta={} root_actions=[{}] root_exploitability={} root_exploitability_bb100={} pio_style_target_bb100={:.2} pio_style_converged={} hero_root_br_value={} villain_root_br_value={} root_br_gap={} local_br_gap={} recursive_root_br_gap={} recursive_local_br_gap={} regret_mass={:.3} illegal_mass={:.6} current_norm_err={:.6} avg_norm_err={:.6} finite={}",
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
        target_bb100,
        root_converged,
        row.hero_root_br_value
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.villain_root_br_value
            .map(|value| format!("{value:.6}"))
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
        _ => {
            eprintln!(
                "usage: pokedr-cli tree-db <build|analyze>\n  pokedr-cli tree-db build <tree.sqlite> [flop] [--config path.yml]\n  pokedr-cli tree-db analyze <tree.sqlite>"
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
    let config = match_config(cli_config.as_ref());
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
        CREATE TABLE solver_nodes (
            node_id INTEGER PRIMARY KEY,
            infoset INTEGER NOT NULL,
            iterations INTEGER NOT NULL,
            acting_player TEXT NOT NULL,
            action_count INTEGER NOT NULL,
            legal_combo_count INTEGER NOT NULL,
            avg_strategy TEXT NOT NULL,
            current_strategy TEXT NOT NULL,
            avg_action_ev TEXT,
            current_action_ev TEXT,
            avg_policy_ev REAL,
            current_policy_ev REAL,
            avg_gap REAL,
            current_gap REAL
        );
        CREATE TABLE solver_combos (
            node_id INTEGER NOT NULL,
            combo_index INTEGER NOT NULL,
            combo TEXT NOT NULL,
            reach REAL NOT NULL,
            weighted_gap REAL NOT NULL,
            avg_action_values TEXT,
            current_action_values TEXT,
            avg_strategy TEXT NOT NULL,
            current_strategy TEXT NOT NULL,
            regrets TEXT NOT NULL,
            strategy_sum TEXT NOT NULL,
            PRIMARY KEY (node_id, combo_index)
        );
        CREATE INDEX nodes_parent_idx ON nodes(parent_id);
        CREATE INDEX solver_nodes_avg_gap_idx ON solver_nodes(avg_gap DESC);
        CREATE INDEX solver_nodes_current_gap_idx ON solver_nodes(current_gap DESC);
        "#,
    )
    .unwrap_or_else(|error| {
        eprintln!("failed to initialize tree DB schema: {error}");
        std::process::exit(1);
    });

    let tx = conn.transaction().unwrap();
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
            "INSERT INTO solver_nodes VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
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
                    vec_json(&node.avg_strategy),
                    vec_json(&node.current_strategy),
                    opt_vec_json(&node.avg_action_ev),
                    opt_vec_json(&node.current_action_ev),
                    node.avg_policy_ev,
                    node.current_policy_ev,
                    node.avg_gap,
                    node.current_gap,
                ])
                .unwrap();
        }
    }
    {
        let mut statement = tx
            .prepare(
                "INSERT INTO solver_combos VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
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
                    opt_vec_json(&combo.avg_action_values),
                    opt_vec_json(&combo.current_action_values),
                    vec_json(&combo.avg_strategy),
                    vec_json(&combo.current_strategy),
                    vec_json(&combo.regrets),
                    vec_json(&combo.strategy_sum),
                ])
                .unwrap();
        }
    }
    tx.commit().unwrap();
}

fn analyze_tree_db(conn: &Connection) -> Value {
    json!({
        "counts": db_counts(conn),
        "top_avg_gaps": db_top_gaps(conn, "avg_gap", "avg_action_ev", "avg_policy_ev"),
        "top_current_gaps": db_top_gaps(conn, "current_gap", "current_action_ev", "current_policy_ev"),
        "root_action_subtrees": db_root_action_subtrees(conn),
    })
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

fn db_top_gaps(
    conn: &Connection,
    gap_column: &str,
    action_ev_column: &str,
    policy_column: &str,
) -> Value {
    let sql = format!(
        "SELECT n.node_id, n.street, n.acting_player, n.pot, n.to_call, n.path, s.{gap_column}, s.{action_ev_column}, s.{policy_column}, s.avg_strategy, s.current_strategy
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
                MAX(current_gap)
         FROM subtree LEFT JOIN solver_nodes ON solver_nodes.node_id = subtree.node_id",
        [root],
        |row| {
            Ok(json!({
                "solver_nodes": row.get::<_, i64>(0)?,
                "mean_avg_gap": row.get::<_, Option<f64>>(1)?.unwrap_or(0.0),
                "mean_current_gap": row.get::<_, Option<f64>>(2)?.unwrap_or(0.0),
                "max_avg_gap": row.get::<_, Option<f64>>(3)?.unwrap_or(0.0),
                "max_current_gap": row.get::<_, Option<f64>>(4)?.unwrap_or(0.0),
            }))
        },
    )
    .unwrap()
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
