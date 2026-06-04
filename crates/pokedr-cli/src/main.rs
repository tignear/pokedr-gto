use pokedr_core::{
    cards::{Board, Card, Rank, Suit},
    dense_cfr::{
        CfrVariant, DenseCfrConfig, DenseCfrIteration, DenseCfrSolver, DenseCfrState,
        gpu::{GpuCfrError, GpuDenseCfrBackend},
    },
    postflop::{ActionSetConfig, Player, PublicState, Street, SubgameTree, SubgameTreeConfig},
    postflop_dense::PostflopDenseLayout,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;

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
        Some("dump-flop-tree") => run_dump_flop_tree(parse_flop_command_args(args)),
        Some("analyze-tree-dump") => run_analyze_tree_dump(args),
        Some("rs-poker-smoke") => run_rs_poker_smoke(),
        Some("rs-poker-trace") => run_rs_poker_trace(),
        _ => {
            eprintln!(
                "usage: {program} <gpu-info|gpu-smoke|postflop-smoke|solve-flop|solve-flop-metrics|solve-flop-sweep|dump-flop-tree|analyze-tree-dump|rs-poker-smoke|rs-poker-trace> [flop] [--config path.yml]"
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

fn run_dump_flop_tree(args: FlopCommandArgs) {
    let cli_config = load_cli_config(args.config_path.as_deref());
    let flop = parse_flop(args.flop.as_deref().unwrap_or("As7h2c")).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let config = match_config(cli_config.as_ref());
    if let Some(node) = env_usize("POKEDR_DUMP_NODE") {
        let Some(line) = pokedr_agent::dump_fixed_flop_tree_node(flop, config, node) else {
            eprintln!("POKEDR_DUMP_NODE={node} is outside dumped tree");
            std::process::exit(2);
        };
        println!("{line}");
    } else {
        let lines = pokedr_agent::dump_fixed_flop_tree(flop, config);
        for line in lines {
            println!("{line}");
        }
    }
}

fn run_analyze_tree_dump(mut args: impl Iterator<Item = String>) {
    let Some(path) = args.next() else {
        eprintln!("usage: pokedr-cli analyze-tree-dump <tree.jsonl>");
        std::process::exit(2);
    };
    if let Some(extra) = args.next() {
        eprintln!("unexpected argument: {extra}");
        std::process::exit(2);
    }
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        eprintln!("failed to read tree dump {path}: {error}");
        std::process::exit(2);
    });
    let nodes = parse_tree_dump_jsonl(&path, &text);
    let analysis = analyze_tree_nodes(&nodes);
    println!(
        "{}",
        serde_json::to_string_pretty(&analysis).expect("analysis JSON must serialize")
    );
}

fn parse_tree_dump_jsonl(path: &str, text: &str) -> Vec<Value> {
    let nodes = text
        .lines()
        .enumerate()
        .filter_map(|(line_index, line)| {
            let line = line.trim();
            if line.is_empty() {
                return None;
            }
            Some(serde_json::from_str::<Value>(line).unwrap_or_else(|error| {
                eprintln!(
                    "failed to parse {path}:{} as JSON tree node: {error}",
                    line_index + 1
                );
                std::process::exit(2);
            }))
        })
        .collect::<Vec<_>>();
    if nodes.is_empty() {
        eprintln!("tree dump {path} did not contain any JSON nodes");
        std::process::exit(2);
    }
    nodes
}

fn analyze_tree_nodes(nodes: &[Value]) -> Value {
    let mut by_index = BTreeMap::new();
    let mut parents = BTreeMap::new();
    for node in nodes {
        let index = json_usize_field(node, "index");
        by_index.insert(index, node);
        parents.insert(index, json_optional_usize_field(node, "parent"));
    }

    let root = by_index.get(&0).copied().unwrap_or_else(|| {
        eprintln!("tree dump must contain root node index 0");
        std::process::exit(2);
    });
    let total = summarize_tree_subset(nodes.iter());
    let root_actions = root
        .get("actions")
        .and_then(Value::as_array)
        .map(|actions| {
            actions
                .iter()
                .map(|action| {
                    let child = json_usize_field(action, "child");
                    let subtree_nodes = nodes
                        .iter()
                        .filter(|node| node_belongs_to_subtree(node, child, &parents))
                        .collect::<Vec<_>>();
                    json!({
                        "action_index": json_usize_field(action, "index"),
                        "action": json_string_field(action, "action"),
                        "source": json_string_field(action, "source"),
                        "child": child,
                        "summary": summarize_tree_subset(subtree_nodes.iter().copied()),
                        "solver_summary": summarize_solver_subset(subtree_nodes.iter().copied()),
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    json!({
        "root": root,
        "total": total,
        "solver_summary": summarize_solver_subset(nodes.iter()),
        "root_action_subtrees": root_actions,
    })
}

fn summarize_tree_subset<'a>(nodes: impl Iterator<Item = &'a Value>) -> Value {
    let mut node_count = 0usize;
    let mut decisions = 0usize;
    let mut chances = 0usize;
    let mut terminals = 0usize;
    let mut fold_terminals = 0usize;
    let mut showdown_terminals = 0usize;
    let mut max_depth = 0usize;
    let mut max_branching = 0usize;
    let mut depth_counts = BTreeMap::<usize, usize>::new();
    let mut street_decisions = BTreeMap::<String, usize>::new();
    let mut player_decisions = BTreeMap::<String, usize>::new();
    let mut action_counts = BTreeMap::<String, usize>::new();
    let mut action_source_counts = BTreeMap::<String, usize>::new();
    let mut terminal_pot_counts = BTreeMap::<u64, usize>::new();
    let mut decision_pot_counts = BTreeMap::<u64, usize>::new();
    let mut to_call_counts = BTreeMap::<u64, usize>::new();

    for node in nodes {
        node_count += 1;
        let depth = node
            .get("path")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        max_depth = max_depth.max(depth);
        *depth_counts.entry(depth).or_default() += 1;
        let branching = node
            .get("children")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        max_branching = max_branching.max(branching);

        match node
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
        {
            "decision" => {
                decisions += 1;
                if let Some(state) = node.get("state") {
                    increment_string_field(&mut street_decisions, state, "street");
                    increment_string_field(&mut player_decisions, state, "acting_player");
                    increment_u64_field(&mut decision_pot_counts, state, "pot");
                    increment_u64_field(&mut to_call_counts, state, "to_call");
                }
                if let Some(actions) = node.get("actions").and_then(Value::as_array) {
                    for action in actions {
                        increment_string_field(&mut action_counts, action, "action");
                        increment_string_field(&mut action_source_counts, action, "source");
                    }
                }
            }
            "chance" => chances += 1,
            "terminal" => {
                terminals += 1;
                match node
                    .get("terminal_kind")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                {
                    "Fold" => fold_terminals += 1,
                    "Showdown" => showdown_terminals += 1,
                    _ => {}
                }
                increment_u64_field(&mut terminal_pot_counts, node, "pot");
            }
            _ => {}
        }
    }

    json!({
        "nodes": node_count,
        "decisions": decisions,
        "chances": chances,
        "terminals": terminals,
        "fold_terminals": fold_terminals,
        "showdown_terminals": showdown_terminals,
        "max_depth": max_depth,
        "max_branching": max_branching,
        "depth_counts": depth_counts,
        "street_decisions": street_decisions,
        "player_decisions": player_decisions,
        "action_counts": action_counts,
        "action_source_counts": action_source_counts,
        "decision_pot_counts": decision_pot_counts,
        "to_call_counts": to_call_counts,
        "terminal_pot_counts": terminal_pot_counts,
    })
}

fn summarize_solver_subset<'a>(nodes: impl Iterator<Item = &'a Value>) -> Value {
    let mut solver_nodes = 0usize;
    let mut avg_gap_sum = 0.0f64;
    let mut current_gap_sum = 0.0f64;
    let mut max_avg_gap = 0.0f64;
    let mut max_current_gap = 0.0f64;
    let mut top_avg_gaps = Vec::<Value>::new();
    let mut top_current_gaps = Vec::<Value>::new();

    for node in nodes {
        let Some(solver) = node.get("solver") else {
            continue;
        };
        let avg_gap = solver_gap(solver, "avg_action_ev", "avg_policy_ev");
        let current_gap = solver_gap(solver, "current_action_ev", "current_policy_ev");
        if avg_gap.is_none() && current_gap.is_none() {
            continue;
        }
        solver_nodes += 1;
        if let Some(gap) = avg_gap {
            avg_gap_sum += gap;
            max_avg_gap = max_avg_gap.max(gap);
            push_top_solver_gap(&mut top_avg_gaps, node, solver, gap, "avg_action_ev");
        }
        if let Some(gap) = current_gap {
            current_gap_sum += gap;
            max_current_gap = max_current_gap.max(gap);
            push_top_solver_gap(
                &mut top_current_gaps,
                node,
                solver,
                gap,
                "current_action_ev",
            );
        }
    }

    json!({
        "solver_nodes": solver_nodes,
        "mean_avg_gap": if solver_nodes > 0 { avg_gap_sum / solver_nodes as f64 } else { 0.0 },
        "mean_current_gap": if solver_nodes > 0 { current_gap_sum / solver_nodes as f64 } else { 0.0 },
        "max_avg_gap": max_avg_gap,
        "max_current_gap": max_current_gap,
        "top_avg_gaps": top_avg_gaps,
        "top_current_gaps": top_current_gaps,
    })
}

fn solver_gap(solver: &Value, action_ev_key: &str, policy_ev_key: &str) -> Option<f64> {
    let action_values = solver.get(action_ev_key)?.as_array()?;
    let policy_ev = solver.get(policy_ev_key)?.as_f64()?;
    let best = action_values
        .iter()
        .filter_map(Value::as_f64)
        .fold(f64::NEG_INFINITY, f64::max);
    best.is_finite().then_some((best - policy_ev).max(0.0))
}

fn push_top_solver_gap(
    top: &mut Vec<Value>,
    node: &Value,
    solver: &Value,
    gap: f64,
    action_ev_key: &str,
) {
    let action_values = solver
        .get(action_ev_key)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let best_action = action_values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.as_f64().map(|value| (index, value)))
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index);
    let path = node
        .get("path")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    top.push(json!({
        "gap": gap,
        "node": json_usize_field(node, "index"),
        "infoset": node.get("infoset").cloned().unwrap_or(Value::Null),
        "best_action": best_action,
        "action_values": action_values,
        "policy_ev": solver.get(if action_ev_key == "avg_action_ev" { "avg_policy_ev" } else { "current_policy_ev" }).cloned().unwrap_or(Value::Null),
        "avg_strategy": solver.get("avg_strategy").cloned().unwrap_or(Value::Null),
        "current_strategy": solver.get("current_strategy").cloned().unwrap_or(Value::Null),
        "state": node.get("state").cloned().unwrap_or(Value::Null),
        "path": path,
    }));
    top.sort_by(|left, right| {
        let left_gap = left.get("gap").and_then(Value::as_f64).unwrap_or(0.0);
        let right_gap = right.get("gap").and_then(Value::as_f64).unwrap_or(0.0);
        right_gap
            .partial_cmp(&left_gap)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    top.truncate(10);
}

fn node_belongs_to_subtree(
    node: &Value,
    subtree_root: usize,
    parents: &BTreeMap<usize, Option<usize>>,
) -> bool {
    let mut current = Some(json_usize_field(node, "index"));
    while let Some(index) = current {
        if index == subtree_root {
            return true;
        }
        current = parents.get(&index).copied().flatten();
    }
    false
}

fn increment_string_field(counts: &mut BTreeMap<String, usize>, value: &Value, key: &str) {
    if let Some(text) = value.get(key).and_then(Value::as_str) {
        *counts.entry(text.to_string()).or_default() += 1;
    }
}

fn increment_u64_field(counts: &mut BTreeMap<u64, usize>, value: &Value, key: &str) {
    if let Some(number) = value.get(key).and_then(Value::as_u64) {
        *counts.entry(number).or_default() += 1;
    }
}

fn json_usize_field(value: &Value, key: &str) -> usize {
    value.get(key).and_then(Value::as_u64).unwrap_or_else(|| {
        eprintln!("tree node JSON missing numeric field `{key}`: {value}");
        std::process::exit(2);
    }) as usize
}

fn json_optional_usize_field(value: &Value, key: &str) -> Option<usize> {
    match value.get(key) {
        Some(Value::Null) | None => None,
        Some(value) => Some(value.as_u64().unwrap_or_else(|| {
            eprintln!("tree node JSON field `{key}` must be a number or null: {value}");
            std::process::exit(2);
        }) as usize),
    }
}

fn json_string_field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_else(|| {
            eprintln!("tree node JSON missing string field `{key}`: {value}");
            std::process::exit(2);
        })
        .to_string()
}

fn smoke_hands() -> usize {
    std::env::var("POKEDR_HANDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(16)
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
