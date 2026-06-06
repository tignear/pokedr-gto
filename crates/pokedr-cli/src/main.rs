use clap::{Parser, Subcommand};
use pokedr_agent::{FlopTreeRequest, build_flop_tree};
use pokedr_core::{
    ActionKind, Board, CfrPlusState, CfrStorageConfig, ChanceExpansion, Player, PublicNodeKind,
    RangeSpec, Street, TreeTemplate, build_action_slot_layout, dry_run_cfr_plus_iteration,
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
        #[arg(long)]
        terminal_cfv_smoke: bool,
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
            iterations,
            chunk_mib,
            allocate_state,
            dry_run_iteration,
            update_slots,
            update_chunk,
            terminal_cfv_smoke,
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
            let tree = build_flop_tree(request).map_err(|error| format!("{error:?}"))?;
            let config = CfrStorageConfig {
                chunk_target_bytes: chunk_mib as u128 * 1024 * 1024,
                ..CfrStorageConfig::default()
            };
            let plan = plan_cfr_work(&tree, config);
            let layout = build_action_slot_layout(&tree, config);
            println!(
                "solving flop={} variant=CfrPlus iterations={iterations}",
                tree.spot.board
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

fn parse_player(value: &str) -> Result<Player, String> {
    match value.to_ascii_lowercase().as_str() {
        "oop" => Ok(Player::Oop),
        "ip" => Ok(Player::Ip),
        _ => Err(format!("invalid player {value:?}; expected oop or ip")),
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
