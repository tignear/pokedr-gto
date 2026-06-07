use clap::{Parser, Subcommand};
use pokedr_agent::{FlopTreeRequest, build_flop_tree};
use pokedr_core::{
    ActionKind, Board, CfrPlusState, CfrStorageConfig, ChanceExpansion, Player,
    PreparedTerminalCfvSmoke, PublicNodeKind, RangeSpec, RealCfrConfig, RealCfrSolver,
    RealCfrVariant, Street, TreeTemplate, analyze_cfr_storage_scenarios,
    analyze_public_state_duplicates, build_action_slot_layout, dry_run_cfr_plus_iteration,
    plan_cfr_work, terminal_cfv_parallel_smoke,
};
use std::str::FromStr;
use std::time::Instant;

#[derive(Debug, Parser)]
#[command(name = "pokedr-cli")]
#[command(about = "Pokedr solver tooling")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    #[command(about = "Inspect a schematic flop public tree and its exact board-expanded size")]
    BuildTree {
        flop: String,
        #[arg(long, default_value_t = 650)]
        pot: u32,
        #[arg(long, default_value_t = 9700)]
        effective_stack: u32,
        #[arg(long, default_value = "full")]
        oop_range: String,
        #[arg(long, default_value = "full")]
        ip_range: String,
        #[arg(long, default_value = "oop")]
        first_player: String,
        #[arg(long, default_value_t = 20)]
        print_nodes: usize,
        #[arg(long)]
        enumerate_chance: bool,
        #[arg(long, default_value_t = 256)]
        chunk_mib: u32,
    },
    #[command(about = "Build the fixed flop CFR+ layout and optionally allocate solver state")]
    SolveFlop {
        flop: String,
        #[arg(long, default_value_t = 650)]
        pot: u32,
        #[arg(long, default_value_t = 9700)]
        effective_stack: u32,
        #[arg(long, default_value = "full")]
        oop_range: String,
        #[arg(long, default_value = "full")]
        ip_range: String,
        #[arg(long, default_value = "oop")]
        first_player: String,
        #[arg(long)]
        enumerate_chance: bool,
        #[arg(long, default_value_t = 1)]
        iterations: u32,
        #[arg(long, default_value_t = 256)]
        chunk_mib: u32,
        #[arg(long)]
        allocate_state: bool,
        #[arg(long)]
        dry_run_iteration: bool,
        #[arg(long, default_value_t = 0)]
        update_slots: usize,
        #[arg(long)]
        update_chunk: Option<u128>,
        #[arg(long, help = "Run only the CFR state update benchmark")]
        run_state_iteration: bool,
        #[arg(
            long,
            help = "Run a cost benchmark for terminal CFV smoke plus CFR state update; this is not a solver iteration"
        )]
        run_cost_benchmark: bool,
        #[arg(long, default_value_t = 1)]
        state_threads: usize,
        #[arg(long)]
        terminal_cfv_smoke: bool,
        #[arg(long)]
        terminal_cfv_tree_pass: bool,
        #[arg(long)]
        run_real_cfr: bool,
        #[arg(long)]
        run_real_cfr_three_phase: bool,
        #[arg(long, default_value_t = 1)]
        real_cfr_log_interval: u32,
        #[arg(long, default_value_t = 0)]
        real_cfr_exploitability_interval: u32,
        #[arg(long)]
        real_cfr_target_exploitability_bb100: Option<f32>,
        #[arg(long, default_value = "cfr-plus")]
        real_cfr_variant: String,
        #[arg(long, default_value_t = 1.5)]
        dcfr_alpha: f32,
        #[arg(long, default_value_t = 0.0)]
        dcfr_beta: f32,
        #[arg(long, default_value_t = 2.0)]
        dcfr_gamma: f32,
        #[arg(long)]
        run_terminal_board_phase: bool,
        #[arg(long)]
        run_terminal_board_phase_board_major: bool,
        #[arg(long)]
        terminal_board_locality: bool,
        #[arg(long)]
        terminal_eval_breakdown: bool,
        #[arg(long)]
        terminal_cfv_calls: Option<usize>,
        #[arg(long, default_value_t = 0)]
        terminal_cfv_threads: usize,
    },
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        Command::BuildTree {
            flop,
            pot,
            effective_stack,
            oop_range,
            ip_range,
            first_player,
            print_nodes,
            enumerate_chance,
            chunk_mib,
        } => {
            let first_player = parse_player(&first_player)?;
            let request = FlopTreeRequest {
                board: Board::from_str(&flop)?,
                pot,
                effective_stack,
                oop_range: RangeSpec::from_str(&oop_range)?,
                ip_range: RangeSpec::from_str(&ip_range)?,
                first_player,
                action_abstraction: pokedr_core::ActionAbstraction::conservative_default(),
            };
            let tree = if enumerate_chance {
                let template = TreeTemplate {
                    action_abstraction: request.action_abstraction.clone(),
                    chance_expansion: ChanceExpansion::Enumerate,
                };
                let spot = pokedr_core::Spot {
                    board: request.board.clone(),
                    pot: request.pot,
                    effective_stack: request.effective_stack,
                    oop_range: request.oop_range.clone(),
                    ip_range: request.ip_range.clone(),
                    first_player: request.first_player,
                };
                pokedr_core::TreeBuilder::new(template)
                    .map_err(|error| format!("{error:?}"))?
                    .build(spot)
                    .map_err(|error| format!("{error:?}"))?
            } else {
                build_flop_tree(request.clone()).map_err(|error| format!("{error:?}"))?
            };
            let stats = tree.stats();
            let estimate = estimate_tree_work(
                &tree,
                request.oop_range.combos().len(),
                request.ip_range.combos().len(),
            );
            println!("board={}", tree.spot.board);
            println!(
                "spot pot={:.2}bb effective_stack={:.2}bb first_player={:?} oop_combos={} ip_combos={}",
                tree.spot.pot as f32 / 100.0,
                tree.spot.effective_stack as f32 / 100.0,
                tree.spot.first_player,
                tree.spot.oop_range_combos,
                tree.spot.ip_range_combos,
            );
            println!(
                "tree nodes={} decisions={} chances={} terminals={} max_depth={}",
                stats.nodes, stats.decisions, stats.chances, stats.terminals, stats.max_depth
            );
            println!(
                "estimate private_infosets={} action_slots={} private_pairs={} terminal_pair_visits={} memory_regret_strategy_f32_mb={:.1}",
                estimate.private_infosets,
                estimate.action_slots,
                estimate.private_pairs,
                estimate.terminal_pair_visits,
                estimate.memory_regret_strategy_f32_mb
            );
            let duplicates = analyze_public_state_duplicates(&tree);
            println!(
                "duplicate_report decisions={} exact_unique={} exact_duplicates={} boardless_unique={} boardless_duplicates={} action_compatible_unique={} action_compatible_duplicates={} history_exact_unique={} history_exact_duplicates={} history_boardless_unique={} history_boardless_duplicates={}",
                duplicates.decision_nodes,
                duplicates.exact_unique,
                duplicates.exact_duplicates,
                duplicates.boardless_unique,
                duplicates.boardless_duplicates,
                duplicates.action_compatible_unique,
                duplicates.action_compatible_duplicates,
                duplicates.history_exact_unique,
                duplicates.history_exact_duplicates,
                duplicates.history_boardless_unique,
                duplicates.history_boardless_duplicates,
            );
            if !enumerate_chance {
                let plan = plan_cfr_work(
                    &tree,
                    CfrStorageConfig {
                        chunk_target_bytes: chunk_mib as u128 * 1024 * 1024,
                        ..CfrStorageConfig::default()
                    },
                );
                println!(
                    "cfr_plan chunk_target_mib={} total_action_slots={} storage_gib={:.2} chunks={} max_chunk_mib={:.1}",
                    chunk_mib,
                    plan.total_action_slots,
                    plan.storage_gib(),
                    plan.total_chunks,
                    plan.max_chunk_mib(),
                );
                let layout = build_action_slot_layout(
                    &tree,
                    CfrStorageConfig {
                        chunk_target_bytes: chunk_mib as u128 * 1024 * 1024,
                        ..CfrStorageConfig::default()
                    },
                );
                print_storage_scenarios(&layout);
                println!(
                    "cfr_plan_storage regret_f32_strategy_f32_gib={:.2} regret_f32_strategy_u16_gib={:.2} regret_f32_only_gib={:.2}",
                    storage_gib(plan.total_action_slots, 4, 4),
                    storage_gib(plan.total_action_slots, 4, 2),
                    storage_gib(plan.total_action_slots, 4, 0),
                );
                for street in [Street::Flop, Street::Turn, Street::River] {
                    let street_plan = plan.street[pokedr_core::plan::street_index(street)];
                    println!(
                        "cfr_plan_street street={street:?} decisions={} action_slots={} storage_gib={:.2} storage_f32_u16_gib={:.2} storage_regret_only_gib={:.2} chunks={}",
                        street_plan.decisions,
                        street_plan.action_slots,
                        street_plan.storage_gib(),
                        storage_gib(street_plan.action_slots, 4, 2),
                        storage_gib(street_plan.action_slots, 4, 0),
                        street_plan.chunks,
                    );
                }
                println!(
                    "cfr_plan_terminal folds={} showdowns={} allins={} terminal_cfv_calls={} showdown_board_evals={} allin_board_evals={} private_pair_upper_bound={}",
                    plan.terminals.fold_terminals,
                    plan.terminals.showdown_terminals,
                    plan.terminals.all_in_terminals,
                    plan.terminals.terminal_cfv_calls,
                    plan.terminals.showdown_board_evals,
                    plan.terminals.all_in_board_evals,
                    plan.terminals.terminal_private_pair_upper_bound,
                );
            }
            for node in tree.nodes.iter().take(print_nodes) {
                print!(
                    "node id={} street={:?} player={:?} pot={:.2}bb kind=",
                    node.id,
                    node.state.street,
                    node.state.player,
                    node.state.pot as f32 / 100.0
                );
                match &node.kind {
                    PublicNodeKind::Decision { player, actions } => {
                        let labels = actions.iter().map(format_action).collect::<Vec<_>>();
                        println!(
                            "decision acting={player:?} actions=[{}] children={:?}",
                            labels.join(","),
                            node.children
                        );
                    }
                    PublicNodeKind::Chance(chance) => {
                        println!(
                            "chance next={:?} cards={} children={:?}",
                            chance.next_street,
                            chance.cards.len(),
                            node.children
                        );
                    }
                    PublicNodeKind::Terminal { reason } => {
                        println!("terminal reason={reason:?}");
                    }
                }
            }
        }
        Command::SolveFlop {
            flop,
            pot,
            effective_stack,
            oop_range,
            ip_range,
            first_player,
            enumerate_chance,
            iterations,
            chunk_mib,
            allocate_state,
            dry_run_iteration,
            update_slots,
            update_chunk,
            run_state_iteration,
            run_cost_benchmark,
            state_threads,
            terminal_cfv_smoke,
            terminal_cfv_tree_pass,
            run_real_cfr,
            run_real_cfr_three_phase,
            real_cfr_log_interval,
            real_cfr_exploitability_interval,
            real_cfr_target_exploitability_bb100,
            real_cfr_variant,
            dcfr_alpha,
            dcfr_beta,
            dcfr_gamma,
            run_terminal_board_phase,
            run_terminal_board_phase_board_major,
            terminal_board_locality,
            terminal_eval_breakdown,
            terminal_cfv_calls,
            terminal_cfv_threads,
        } => {
            let request = flop_tree_request(
                &flop,
                pot,
                effective_stack,
                &oop_range,
                &ip_range,
                &first_player,
            )?;
            let tree = if enumerate_chance {
                let template = TreeTemplate {
                    action_abstraction: pokedr_core::ActionAbstraction::conservative_default(),
                    chance_expansion: ChanceExpansion::Enumerate,
                };
                let spot = pokedr_core::Spot {
                    board: request.board.clone(),
                    pot: request.pot,
                    effective_stack: request.effective_stack,
                    oop_range: request.oop_range.clone(),
                    ip_range: request.ip_range.clone(),
                    first_player: request.first_player,
                };
                pokedr_core::TreeBuilder::new(template)
                    .map_err(|error| format!("{error:?}"))?
                    .build(spot)
                    .map_err(|error| format!("{error:?}"))?
            } else {
                build_flop_tree(request).map_err(|error| format!("{error:?}"))?
            };
            let config = CfrStorageConfig {
                chunk_target_bytes: chunk_mib as u128 * 1024 * 1024,
                ..CfrStorageConfig::default()
            };
            let real_cfr_variant =
                parse_real_cfr_variant(&real_cfr_variant, dcfr_alpha, dcfr_beta, dcfr_gamma)?;
            let plan = plan_cfr_work(&tree, config);
            let layout = build_action_slot_layout(&tree, config);
            println!(
                "solving flop={} variant={} iterations={iterations}",
                tree.spot.board,
                format_real_cfr_variant(real_cfr_variant),
            );
            println!(
                "layout records={} action_slots={} storage_gib={:.2} flop_slots={} turn_slots={} river_slots={}",
                layout.records.len(),
                layout.total_action_slots,
                layout.storage_gib(),
                layout.flop_slots(),
                layout.turn_slots(),
                layout.river_slots(),
            );
            println!(
                "plan chunks={} max_chunk_mib={:.1} terminal_cfv_calls={} private_pair_upper_bound={}",
                plan.total_chunks,
                plan.max_chunk_mib(),
                plan.terminals.terminal_cfv_calls,
                plan.terminals.terminal_private_pair_upper_bound,
            );
            print_storage_scenarios(&layout);
            if terminal_cfv_smoke {
                let calls = terminal_cfv_calls.unwrap_or_else(|| {
                    usize::try_from(plan.terminals.terminal_cfv_calls).unwrap_or(usize::MAX)
                });
                let started = Instant::now();
                let smoke =
                    terminal_cfv_parallel_smoke(&tree.spot.board, calls, terminal_cfv_threads)?;
                println!(
                    "terminal_cfv_smoke boards={} calls={} threads={} prepare_ms={:.3} eval_ms={:.3} total_ms={:.3} calls_per_sec={:.1} checksum={:.6}",
                    smoke.board_count,
                    smoke.calls,
                    smoke.threads,
                    smoke.prepare_elapsed_ms,
                    smoke.eval_elapsed_ms,
                    started.elapsed().as_secs_f64() * 1000.0,
                    smoke.calls_per_second,
                    smoke.checksum,
                );
            }
            if terminal_cfv_tree_pass {
                let started = Instant::now();
                let pass = pokedr_core::terminal_cfv_tree_pass(
                    &tree,
                    &RangeSpec::from_str(&oop_range)?,
                    &RangeSpec::from_str(&ip_range)?,
                    terminal_cfv_threads,
                )?;
                println!(
                    "terminal_cfv_tree_pass terminals={} board_evals={} threads={} prepare_ms={:.3} eval_ms={:.3} total_ms={:.3} checksum={:.6}",
                    pass.terminals,
                    pass.board_evals,
                    pass.threads,
                    pass.prepare_elapsed_ms,
                    pass.eval_elapsed_ms,
                    started.elapsed().as_secs_f64() * 1000.0,
                    pass.checksum,
                );
            }
            if run_real_cfr {
                let started = Instant::now();
                let mut solver = RealCfrSolver::new(
                    tree.clone(),
                    RangeSpec::from_str(&oop_range)?,
                    RangeSpec::from_str(&ip_range)?,
                )?;
                let summary = solver.run_with_progress(
                    RealCfrConfig {
                        iterations,
                        variant: real_cfr_variant,
                    },
                    |progress| {
                        if real_cfr_log_interval > 0
                            && (progress.iteration == 1
                                || progress.iteration == iterations
                                || progress.iteration % real_cfr_log_interval == 0)
                        {
                            println!(
                                "real_cfr_progress iteration={} terminal_evals={} iteration_ms={:.3} root_oop_value={:.6} root_ip_value={:.6} zero_sum_delta={:.6}",
                                progress.iteration,
                                progress.terminal_evals,
                                progress.elapsed_ms,
                                progress.root_oop_value,
                                progress.root_ip_value,
                                progress.root_oop_value + progress.root_ip_value,
                            );
                        }
                    },
                )?;
                println!(
                    "real_cfr iterations={} decision_nodes={} action_slots={} terminal_evals={} elapsed_ms={:.3} root_oop_value={:.6} root_ip_value={:.6} zero_sum_delta={:.6}",
                    summary.iterations,
                    summary.decision_nodes,
                    summary.action_slots,
                    summary.terminal_evals,
                    started.elapsed().as_secs_f64() * 1000.0,
                    summary.root_oop_value,
                    summary.root_ip_value,
                    summary.root_oop_value + summary.root_ip_value,
                );
            }
            if run_real_cfr_three_phase {
                let started = Instant::now();
                let mut solver = RealCfrSolver::new(
                    tree.clone(),
                    RangeSpec::from_str(&oop_range)?,
                    RangeSpec::from_str(&ip_range)?,
                )?;
                let mut completed = 0u32;
                let mut summary = None;
                let mut total_reach_ms = 0.0;
                let mut total_terminal_ms = 0.0;
                let mut total_backup_ms = 0.0;
                while completed < iterations {
                    let remaining = iterations - completed;
                    let chunk = if real_cfr_exploitability_interval > 0 {
                        remaining.min(real_cfr_exploitability_interval)
                    } else {
                        remaining
                    };
                    let chunk_start = completed;
                    let chunk_summary = solver.run_three_phase(
                        RealCfrConfig {
                            iterations: chunk,
                            variant: real_cfr_variant,
                        },
                        state_threads,
                        |progress| {
                            let global_iteration = chunk_start + progress.iteration;
                            if real_cfr_log_interval > 0
                                && (global_iteration == 1
                                    || global_iteration == iterations
                                    || global_iteration % real_cfr_log_interval == 0)
                            {
                                println!(
                                    "real_cfr_three_phase_progress iteration={} terminal_evals={} reach_ms={:.3} terminal_ms={:.3} backup_ms={:.3} root_oop_value={:.6} root_ip_value={:.6} zero_sum_delta={:.6}",
                                    global_iteration,
                                    progress.terminal_evals,
                                    progress.reach_ms,
                                    progress.terminal_ms,
                                    progress.backup_ms,
                                    progress.root_oop_value,
                                    progress.root_ip_value,
                                    progress.root_oop_value + progress.root_ip_value,
                                );
                            }
                        },
                    )?;
                    total_reach_ms += chunk_summary.reach_ms;
                    total_terminal_ms += chunk_summary.terminal_ms;
                    total_backup_ms += chunk_summary.backup_ms;
                    completed += chunk;
                    summary = Some(chunk_summary);
                    if real_cfr_exploitability_interval > 0 {
                        let exploitability = solver.exploitability(state_threads)?;
                        println!(
                            "real_cfr_exploitability iteration={} profile_oop={:.6} profile_ip={:.6} oop_br={:.6} ip_br={:.6} oop_gain={:.6} ip_gain={:.6} nash_conv_chips={:.6} exploitability_chips={:.6} exploitability_bb_per_100={:.6}",
                            completed,
                            exploitability.profile_oop_value,
                            exploitability.profile_ip_value,
                            exploitability.oop_best_response_value,
                            exploitability.ip_best_response_value,
                            exploitability.oop_gain,
                            exploitability.ip_gain,
                            exploitability.nash_conv_chips,
                            exploitability.exploitability_chips,
                            exploitability.exploitability_bb_per_100,
                        );
                        if real_cfr_target_exploitability_bb100.is_some_and(|target| {
                            exploitability.exploitability_bb_per_100 <= target
                        }) {
                            break;
                        }
                    }
                }
                let summary = summary.expect("at least one iteration must run");
                println!(
                    "real_cfr_three_phase iterations={} states={} decision_nodes={} action_slots={} terminal_evals={} elapsed_ms={:.3} reach_ms={:.3} terminal_ms={:.3} backup_ms={:.3} root_oop_value={:.6} root_ip_value={:.6} zero_sum_delta={:.6}",
                    completed,
                    summary.states,
                    summary.decision_nodes,
                    summary.action_slots,
                    summary.terminal_evals,
                    started.elapsed().as_secs_f64() * 1000.0,
                    total_reach_ms,
                    total_terminal_ms,
                    total_backup_ms,
                    summary.root_oop_value,
                    summary.root_ip_value,
                    summary.root_oop_value + summary.root_ip_value,
                );
            }
            if run_terminal_board_phase {
                let started = Instant::now();
                let solver = RealCfrSolver::new(
                    tree.clone(),
                    RangeSpec::from_str(&oop_range)?,
                    RangeSpec::from_str(&ip_range)?,
                )?;
                let summary = solver.run_terminal_board_phase(state_threads)?;
                println!(
                    "terminal_board_phase threads={} terminal_evals={} elapsed_ms={:.3} total_ms={:.3} checksum={:.6}",
                    state_threads,
                    summary.terminal_evals,
                    summary.elapsed_ms,
                    started.elapsed().as_secs_f64() * 1000.0,
                    summary.checksum,
                );
            }
            if run_terminal_board_phase_board_major {
                let started = Instant::now();
                let solver = RealCfrSolver::new(
                    tree.clone(),
                    RangeSpec::from_str(&oop_range)?,
                    RangeSpec::from_str(&ip_range)?,
                )?;
                let summary = solver.run_terminal_board_phase_board_major(state_threads)?;
                println!(
                    "terminal_board_phase_board_major threads={} terminal_evals={} elapsed_ms={:.3} total_ms={:.3} checksum={:.6}",
                    state_threads,
                    summary.terminal_evals,
                    summary.elapsed_ms,
                    started.elapsed().as_secs_f64() * 1000.0,
                    summary.checksum,
                );
            }
            if terminal_board_locality {
                let solver = RealCfrSolver::new(
                    tree.clone(),
                    RangeSpec::from_str(&oop_range)?,
                    RangeSpec::from_str(&ip_range)?,
                )?;
                let locality = solver.terminal_board_locality()?;
                println!(
                    "terminal_board_locality tasks={} unique_boards={} current_order_runs={} average_run_len={:.3} max_run_len={} min_tasks_per_board={} max_tasks_per_board={} average_tasks_per_board={:.3} board_major_task_mib={:.3}",
                    locality.tasks,
                    locality.unique_boards,
                    locality.current_order_runs,
                    locality.average_run_len,
                    locality.max_run_len,
                    locality.min_tasks_per_board,
                    locality.max_tasks_per_board,
                    locality.average_tasks_per_board,
                    locality.board_major_task_bytes as f64 / (1024.0 * 1024.0),
                );
            }
            if terminal_eval_breakdown {
                let solver = RealCfrSolver::new(
                    tree.clone(),
                    RangeSpec::from_str(&oop_range)?,
                    RangeSpec::from_str(&ip_range)?,
                )?;
                let breakdown = solver.terminal_eval_breakdown()?;
                println!(
                    "terminal_eval_breakdown fold_terminals={} showdown_terminals={} all_in_terminals={} river_showdown_evals={} flop_all_in_runout_evals={} turn_all_in_runout_evals={} river_all_in_evals={} total_evals={}",
                    breakdown.fold_terminals,
                    breakdown.showdown_terminals,
                    breakdown.all_in_terminals,
                    breakdown.river_showdown_evals,
                    breakdown.flop_all_in_runout_evals,
                    breakdown.turn_all_in_runout_evals,
                    breakdown.river_all_in_evals,
                    breakdown.river_showdown_evals
                        + breakdown.flop_all_in_runout_evals
                        + breakdown.turn_all_in_runout_evals
                        + breakdown.river_all_in_evals,
                );
            }
            if !allocate_state {
                println!("dry_run=true state_allocated=false");
                if dry_run_iteration {
                    let dry_run = dry_run_cfr_plus_iteration(&layout, 1);
                    println!(
                        "iteration_dry_run iteration={} records={} infosets={} action_slots={} regret_reads={} regret_writes={} strategy_sum_writes={} strategy_sum_delta={:.3} checksum={:.6}",
                        dry_run.iteration,
                        dry_run.records_visited,
                        dry_run.infosets_visited,
                        dry_run.action_slots_visited,
                        dry_run.regret_reads,
                        dry_run.regret_writes,
                        dry_run.strategy_sum_writes,
                        dry_run.strategy_sum_delta,
                        dry_run.checksum,
                    );
                }
                println!("pass --allocate-state to allocate regret and strategy_sum vectors");
                return Ok(());
            }
            let mut state = CfrPlusState::allocate(layout).map_err(|error| format!("{error:?}"))?;
            println!(
                "dry_run=false state_allocated=true regret_len={} strategy_sum_len={} storage_gib={:.2}",
                state.regret.len(),
                state.strategy_sum.len(),
                state.storage_gib(),
            );
            if update_slots > 0 {
                let started = Instant::now();
                let summary = state.update_prefix_slots(update_slots);
                println!(
                    "prefix_update requested_slots={} updated_slots={} elapsed_ms={:.3} strategy_sum_delta={:.6} regret_checksum={:.6} strategy_sum_checksum={:.6}",
                    summary.requested_slots,
                    summary.updated_slots,
                    started.elapsed().as_secs_f64() * 1000.0,
                    summary.strategy_sum_delta,
                    summary.regret_checksum,
                    summary.strategy_sum_checksum,
                );
            }
            if let Some(chunk_index) = update_chunk {
                let chunk_bytes = chunk_mib as u128 * 1024 * 1024;
                let Some(chunk) = state.layout.slot_chunk(chunk_index, chunk_bytes) else {
                    return Err(format!(
                        "chunk index {chunk_index} is outside the action slot layout"
                    ));
                };
                let requested = chunk.end - chunk.start;
                let started = Instant::now();
                let summary = state.update_slot_chunk(chunk, requested);
                println!(
                    "chunk_update chunk={} start={} end={} requested_slots={} updated_slots={} elapsed_ms={:.3} strategy_sum_delta={:.6} regret_checksum={:.6} strategy_sum_checksum={:.6}",
                    chunk.index,
                    chunk.start,
                    chunk.end,
                    summary.requested_slots,
                    summary.updated_slots,
                    started.elapsed().as_secs_f64() * 1000.0,
                    summary.strategy_sum_delta,
                    summary.regret_checksum,
                    summary.strategy_sum_checksum,
                );
            }
            if run_state_iteration {
                let chunk_bytes = chunk_mib as u128 * 1024 * 1024;
                let started = Instant::now();
                let summary =
                    state.apply_regret_matching_iteration_parallel(chunk_bytes, state_threads);
                println!(
                    "state_iteration chunks={} threads={} updated_slots={} elapsed_ms={:.3} strategy_sum_delta={:.6} regret_checksum={:.6} strategy_sum_checksum={:.6}",
                    summary.chunks,
                    state_threads,
                    summary.updated_slots,
                    started.elapsed().as_secs_f64() * 1000.0,
                    summary.strategy_sum_delta,
                    summary.regret_checksum,
                    summary.strategy_sum_checksum,
                );
            }
            if run_cost_benchmark {
                let chunk_bytes = chunk_mib as u128 * 1024 * 1024;
                let terminal_calls = terminal_cfv_calls.unwrap_or_else(|| {
                    usize::try_from(plan.terminals.terminal_cfv_calls).unwrap_or(usize::MAX)
                });
                let prepare_started = Instant::now();
                let prepared_terminal = PreparedTerminalCfvSmoke::new(&tree.spot.board)?;
                let terminal_prepare_ms = prepare_started.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "cost_benchmark_prepare terminal_boards={} terminal_prepare_ms={:.3}",
                    prepared_terminal.board_count(),
                    terminal_prepare_ms,
                );
                let total_started = Instant::now();
                for iteration in 1..=iterations {
                    let iteration_started = Instant::now();
                    let terminal = prepared_terminal.run(terminal_calls, terminal_cfv_threads)?;
                    let state_started = Instant::now();
                    let summary =
                        state.apply_regret_matching_iteration_parallel(chunk_bytes, state_threads);
                    let state_elapsed_ms = state_started.elapsed().as_secs_f64() * 1000.0;
                    println!(
                        "cost_benchmark iteration={} terminal_cfv_smoke_calls={} terminal_cfv_smoke_threads={} terminal_eval_ms={:.3} state_threads={} state_ms={:.3} total_ms={:.3} updated_slots={} checksum={:.6}",
                        iteration,
                        terminal.calls,
                        terminal.threads,
                        terminal.eval_elapsed_ms,
                        state_threads,
                        state_elapsed_ms,
                        iteration_started.elapsed().as_secs_f64() * 1000.0,
                        summary.updated_slots,
                        terminal.checksum + summary.strategy_sum_checksum,
                    );
                }
                println!(
                    "cost_benchmark_total iterations={} elapsed_ms={:.3}",
                    iterations,
                    total_started.elapsed().as_secs_f64() * 1000.0,
                );
            }
        }
    }
    Ok(())
}

fn flop_tree_request(
    flop: &str,
    pot: u32,
    effective_stack: u32,
    oop_range: &str,
    ip_range: &str,
    first_player: &str,
) -> Result<FlopTreeRequest, String> {
    Ok(FlopTreeRequest {
        board: Board::from_str(flop)?,
        pot,
        effective_stack,
        oop_range: RangeSpec::from_str(oop_range)?,
        ip_range: RangeSpec::from_str(ip_range)?,
        first_player: parse_player(first_player)?,
        action_abstraction: pokedr_core::ActionAbstraction::conservative_default(),
    })
}

struct WorkEstimate {
    private_infosets: u128,
    action_slots: u128,
    private_pairs: u128,
    terminal_pair_visits: u128,
    memory_regret_strategy_f32_mb: f64,
}

fn estimate_tree_work(
    tree: &pokedr_core::PublicTree,
    oop_combos: usize,
    ip_combos: usize,
) -> WorkEstimate {
    let mut private_infosets = 0u128;
    let mut action_slots = 0u128;
    let mut terminals = 0u128;
    for node in &tree.nodes {
        match &node.kind {
            PublicNodeKind::Decision { actions, .. } => {
                let combos = match node.state.player {
                    Player::Oop => oop_combos,
                    Player::Ip => ip_combos,
                } as u128;
                private_infosets += combos;
                action_slots += combos * actions.len() as u128;
            }
            PublicNodeKind::Terminal { .. } => terminals += 1,
            PublicNodeKind::Chance(_) => {}
        }
    }
    let private_pairs = oop_combos as u128 * ip_combos as u128;
    let terminal_pair_visits = terminals * private_pairs;
    let memory_regret_strategy_f32_mb = action_slots as f64 * 2.0 * 4.0 / (1024.0 * 1024.0);
    WorkEstimate {
        private_infosets,
        action_slots,
        private_pairs,
        terminal_pair_visits,
        memory_regret_strategy_f32_mb,
    }
}

fn storage_gib(action_slots: u128, regret_bytes: u128, strategy_sum_bytes: u128) -> f64 {
    action_slots as f64 * (regret_bytes + strategy_sum_bytes) as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn print_storage_scenarios(layout: &pokedr_core::ActionSlotLayout) {
    let scenarios = analyze_cfr_storage_scenarios(layout);
    println!(
        "storage_scenarios total_slots={} river_slots={} f32_f32_gib={:.2} f32_u16_gib={:.2} regret_only_gib={:.2}",
        scenarios.total_slots,
        scenarios.river_slots,
        scenarios.regret_f32_strategy_f32_gib,
        scenarios.regret_f32_strategy_u16_gib,
        scenarios.regret_f32_only_gib,
    );
    println!(
        "storage_scenarios_unordered_river river_ordered_slots={} river_unordered_slots={} f32_f32_gib={:.2} f32_u16_gib={:.2} regret_only_gib={:.2}",
        scenarios.river_ordered_board_slots,
        scenarios.river_unordered_board_slots,
        scenarios.river_unordered_regret_f32_strategy_f32_gib,
        scenarios.river_unordered_regret_f32_strategy_u16_gib,
        scenarios.river_unordered_regret_f32_only_gib,
    );
}

fn parse_player(value: &str) -> Result<Player, String> {
    match value.to_ascii_lowercase().as_str() {
        "oop" => Ok(Player::Oop),
        "ip" => Ok(Player::Ip),
        _ => Err(format!("invalid player {value:?}; expected oop or ip")),
    }
}

fn parse_real_cfr_variant(
    value: &str,
    alpha: f32,
    beta: f32,
    gamma: f32,
) -> Result<RealCfrVariant, String> {
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(format!(
            "invalid dcfr alpha {alpha}; expected finite value >= 0"
        ));
    }
    if !beta.is_finite() || beta < 0.0 {
        return Err(format!(
            "invalid dcfr beta {beta}; expected finite value >= 0"
        ));
    }
    if !gamma.is_finite() || gamma < 0.0 {
        return Err(format!(
            "invalid dcfr gamma {gamma}; expected finite value >= 0"
        ));
    }
    match value.to_ascii_lowercase().as_str() {
        "cfr-plus" | "cfr+" => Ok(RealCfrVariant::CfrPlus),
        "dcfr" => Ok(RealCfrVariant::Dcfr { alpha, beta, gamma }),
        "dcfr-plus" | "dcfr+" => Ok(RealCfrVariant::DcfrPlus { alpha, gamma }),
        _ => Err(format!(
            "invalid real CFR variant {value:?}; expected cfr-plus, dcfr, or dcfr-plus"
        )),
    }
}

fn format_real_cfr_variant(variant: RealCfrVariant) -> String {
    match variant {
        RealCfrVariant::CfrPlus => "cfr-plus".to_string(),
        RealCfrVariant::Dcfr { alpha, beta, gamma } => {
            format!("dcfr(alpha={alpha},beta={beta},gamma={gamma})")
        }
        RealCfrVariant::DcfrPlus { alpha, gamma } => {
            format!("dcfr-plus(alpha={alpha},gamma={gamma})")
        }
    }
}

fn format_action(action: &ActionKind) -> String {
    match action {
        ActionKind::Check => "check".to_string(),
        ActionKind::Bet { amount } => format!("bet:{:.2}bb", *amount as f32 / 100.0),
        ActionKind::Call { amount } => format!("call:{:.2}bb", *amount as f32 / 100.0),
        ActionKind::Fold => "fold".to_string(),
        ActionKind::Raise { to } => format!("raise_to:{:.2}bb", *to as f32 / 100.0),
        ActionKind::AllIn { to } => format!("allin_to:{:.2}bb", *to as f32 / 100.0),
    }
}
