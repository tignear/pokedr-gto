use clap::{Parser, Subcommand};
use pokedr_agent::{FlopTreeRequest, build_flop_tree};
use pokedr_core::{
    ActionAbstraction, ActionKind, Board, CfrStorageConfig, ChanceExpansion, NodeLocalCfrSolver,
    Player, PublicNodeKind, PublicTree, RangeSpec, RealCfrAverageStrategy, RealCfrConfig,
    RealCfrVariant, Spot, Street, TreeBuilder, TreeTemplate, fixed_flop_future_board_isomorphism,
    full_deck_future_board_isomorphism_survey, plan_cfr_work,
};
use std::str::FromStr;
use std::time::Instant;

#[derive(Debug, Parser)]
#[command(name = "pokedr-cli")]
#[command(about = "Pokedr postflop solver tooling")]
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
        #[arg(long, default_value = "conservative")]
        tree_preset: String,
        #[arg(long, default_value_t = 20)]
        print_nodes: usize,
        #[arg(long)]
        enumerate_chance: bool,
        #[arg(long, default_value_t = 256)]
        chunk_mib: u32,
    },
    #[command(about = "Solve a fixed flop with the node-local full-range postflop CFR solver")]
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
        #[arg(long, default_value = "conservative")]
        tree_preset: String,
        #[arg(long)]
        enumerate_chance: bool,
        #[arg(long, default_value_t = 1)]
        iterations: u32,
        #[arg(long, default_value_t = 1)]
        threads: usize,
        #[arg(long, default_value_t = 1)]
        log_interval: u32,
        #[arg(long, default_value_t = 0)]
        exploitability_interval: u32,
        #[arg(long)]
        target_exploitability_bb100: Option<f32>,
        #[arg(long, default_value = "dcfr-plus")]
        variant: String,
        #[arg(long, default_value = "reach-weighted")]
        average_strategy: String,
        #[arg(long, default_value_t = 1.5)]
        dcfr_alpha: f32,
        #[arg(long, default_value_t = 0.0)]
        dcfr_beta: f32,
        #[arg(long, default_value_t = 2.0)]
        dcfr_gamma: f32,
    },
    #[command(about = "Inspect exact future-board suit isomorphism for a fixed flop and ranges")]
    BoardIsomorphism {
        flop: String,
        #[arg(long, default_value = "full")]
        oop_range: String,
        #[arg(long, default_value = "full")]
        ip_range: String,
        #[arg(long, default_value_t = 8)]
        print_turns: usize,
        #[arg(
            long,
            help = "Survey every unordered flop instead of only the supplied flop"
        )]
        survey_all_flops: bool,
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
            tree_preset,
            print_nodes,
            enumerate_chance,
            chunk_mib,
        } => {
            let request = flop_tree_request(
                &flop,
                pot,
                effective_stack,
                &oop_range,
                &ip_range,
                &first_player,
                &tree_preset,
            )?;
            let tree = build_tree(request.clone(), enumerate_chance)?;
            print_tree_report(&tree, &request, chunk_mib, print_nodes);
        }
        Command::SolveFlop {
            flop,
            pot,
            effective_stack,
            oop_range,
            ip_range,
            first_player,
            tree_preset,
            enumerate_chance,
            iterations,
            threads,
            log_interval,
            exploitability_interval,
            target_exploitability_bb100,
            variant,
            average_strategy,
            dcfr_alpha,
            dcfr_beta,
            dcfr_gamma,
        } => {
            let request = flop_tree_request(
                &flop,
                pot,
                effective_stack,
                &oop_range,
                &ip_range,
                &first_player,
                &tree_preset,
            )?;
            let tree = build_tree(request.clone(), enumerate_chance)?;
            let variant = parse_cfr_variant(&variant, dcfr_alpha, dcfr_beta, dcfr_gamma)?;
            let average_strategy = parse_average_strategy(&average_strategy)?;
            solve_flop(
                tree,
                request.oop_range,
                request.ip_range,
                iterations,
                threads,
                log_interval,
                exploitability_interval,
                target_exploitability_bb100,
                RealCfrConfig {
                    iterations,
                    variant,
                    average_strategy,
                },
            )?;
        }
        Command::BoardIsomorphism {
            flop,
            oop_range,
            ip_range,
            print_turns,
            survey_all_flops,
        } => {
            let oop_range = RangeSpec::from_str(&oop_range)?;
            let ip_range = RangeSpec::from_str(&ip_range)?;
            if survey_all_flops {
                let survey = full_deck_future_board_isomorphism_survey(&oop_range, &ip_range)?;
                println!("flops={}", survey.flops);
                println!(
                    "ordered_turn_river concrete_events_per_flop={} min_representative_events={} max_representative_events={} average_representative_events={:.3} average_eliminated_events={:.3} average_eliminated_fraction={:.6}",
                    survey.ordered_turn_river_concrete_events_per_flop,
                    survey.min_representative_events,
                    survey.max_representative_events,
                    survey.average_representative_events,
                    survey.average_eliminated_events,
                    survey.average_eliminated_fraction
                );
                return Ok(());
            }
            let flop = Board::from_str(&flop)?;
            let report = fixed_flop_future_board_isomorphism(&flop, &oop_range, &ip_range)?;
            println!("board={}", report.flop);
            println!(
                "valid_public_range_suit_permutations={}",
                report.valid_permutations
            );
            println!(
                "turn concrete_events={} classes={} eliminated={} multiplicity_sum={}",
                report.turn.concrete_events,
                report.turn.classes.len(),
                report
                    .turn
                    .concrete_events
                    .saturating_sub(report.turn.classes.len()),
                report
                    .turn
                    .classes
                    .iter()
                    .map(|class| class.multiplicity)
                    .sum::<usize>()
            );
            println!(
                "ordered_turn_river concrete_events={} representative_events={} eliminated={}",
                report.ordered_turn_river_concrete_events,
                report.ordered_turn_river_representative_events,
                report
                    .ordered_turn_river_concrete_events
                    .saturating_sub(report.ordered_turn_river_representative_events)
            );
            for (index, turn_class) in report.turn.classes.iter().take(print_turns).enumerate() {
                let turn = turn_class
                    .representative
                    .first()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "-".to_string());
                let river = &report.representative_turn_river_classes[index];
                println!(
                    "turn_class index={} card={} multiplicity={} river_concrete_events={} river_classes={} river_eliminated={}",
                    index,
                    turn,
                    turn_class.multiplicity,
                    river.concrete_events,
                    river.classes.len(),
                    river.concrete_events.saturating_sub(river.classes.len())
                );
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn solve_flop(
    tree: PublicTree,
    oop_range: RangeSpec,
    ip_range: RangeSpec,
    iterations: u32,
    threads: usize,
    log_interval: u32,
    exploitability_interval: u32,
    target_exploitability_bb100: Option<f32>,
    config: RealCfrConfig,
) -> Result<(), String> {
    println!(
        "solving flop={} variant={} average_strategy={} iterations={} threads={}",
        tree.spot.board,
        format_cfr_variant(config.variant),
        format_average_strategy(config.average_strategy),
        iterations,
        threads,
    );
    let started = Instant::now();
    let mut solver = NodeLocalCfrSolver::new(tree, oop_range, ip_range)?;
    let interval = if exploitability_interval > 0 {
        exploitability_interval
    } else if target_exploitability_bb100.is_some() {
        log_interval.max(16)
    } else {
        0
    };
    let mut completed = 0u32;
    let mut summary = solver.summary();
    while completed < iterations {
        let remaining = iterations - completed;
        let chunk = if interval > 0 {
            remaining.min(interval)
        } else {
            remaining
        };
        let chunk_start = completed;
        summary = solver.run_with_progress(
            RealCfrConfig {
                iterations: chunk,
                ..config
            },
            |progress| {
                let global_iteration = chunk_start + progress.iteration;
                if log_interval > 0
                    && (global_iteration == 1
                        || global_iteration == iterations
                        || global_iteration % log_interval == 0)
                {
                    println!(
                        "node_cfr_progress iteration={} terminal_evals={} iteration_ms={:.3} oop_pass_value={:.6} ip_pass_value={:.6}",
                        global_iteration,
                        progress.terminal_evals,
                        progress.elapsed_ms,
                        progress.oop_update_pass_value,
                        progress.ip_update_pass_value,
                    );
                }
            },
        )?;
        completed = summary.iterations;
        if interval > 0 {
            let exploitability = solver.exploitability(threads)?;
            println!(
                "node_cfr_exploitability iteration={} profile_oop={:.6} profile_ip={:.6} zero_sum_delta={:.6} oop_br={:.6} ip_br={:.6} oop_gain={:.6} ip_gain={:.6} nash_conv_chips={:.6} exploitability_chips={:.6} exploitability_bb_per_100={:.6}",
                completed,
                exploitability.profile_oop_value,
                exploitability.profile_ip_value,
                exploitability.profile_oop_value + exploitability.profile_ip_value,
                exploitability.oop_best_response_value,
                exploitability.ip_best_response_value,
                exploitability.oop_gain,
                exploitability.ip_gain,
                exploitability.nash_conv_chips,
                exploitability.exploitability_chips,
                exploitability.exploitability_bb_per_100,
            );
            if target_exploitability_bb100
                .is_some_and(|target| exploitability.exploitability_bb_per_100 <= target)
            {
                break;
            }
        }
    }
    println!(
        "node_cfr iterations={} states={} decision_states={} action_slots={} terminal_evals={} elapsed_ms={:.3} oop_pass_value={:.6} ip_pass_value={:.6}",
        summary.iterations,
        summary.states,
        summary.decision_states,
        summary.action_slots,
        summary.terminal_evals,
        summary.elapsed_ms,
        summary.oop_update_pass_value,
        summary.ip_update_pass_value,
    );
    println!(
        "node_cfr_state_allocated=true storage_gib={:.3} total_elapsed_ms={:.3}",
        summary.storage_gib,
        started.elapsed().as_secs_f64() * 1000.0,
    );
    Ok(())
}

fn build_tree(request: FlopTreeRequest, enumerate_chance: bool) -> Result<PublicTree, String> {
    if enumerate_chance {
        let template = TreeTemplate {
            action_abstraction: request.action_abstraction.clone(),
            chance_expansion: ChanceExpansion::Enumerate,
        };
        let spot = Spot {
            board: request.board,
            pot: request.pot,
            effective_stack: request.effective_stack,
            oop_range: request.oop_range,
            ip_range: request.ip_range,
            first_player: request.first_player,
        };
        TreeBuilder::new(template)
            .map_err(|error| format!("{error:?}"))?
            .build(spot)
            .map_err(|error| format!("{error:?}"))
    } else {
        build_flop_tree(request).map_err(|error| format!("{error:?}"))
    }
}

fn print_tree_report(
    tree: &PublicTree,
    request: &FlopTreeRequest,
    chunk_mib: u32,
    print_nodes: usize,
) {
    let stats = tree.stats();
    let estimate = estimate_tree_work(
        tree,
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
        "tree_by_street flop_decisions={} turn_decisions={} river_decisions={}",
        stats.decisions_by_street[0], stats.decisions_by_street[1], stats.decisions_by_street[2],
    );
    println!(
        "estimate private_infosets={} action_slots={} private_pairs={} terminal_pair_visits={} memory_regret_strategy_f32_mb={:.1}",
        estimate.private_infosets,
        estimate.action_slots,
        estimate.private_pairs,
        estimate.terminal_pair_visits,
        estimate.memory_regret_strategy_f32_mb
    );
    let plan = plan_cfr_work(
        tree,
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
    for street in [Street::Flop, Street::Turn, Street::River] {
        let street_plan = plan.street[pokedr_core::plan::street_index(street)];
        println!(
            "cfr_plan_street street={street:?} decisions={} action_slots={} storage_gib={:.2} chunks={}",
            street_plan.decisions,
            street_plan.action_slots,
            street_plan.storage_gib(),
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

fn flop_tree_request(
    flop: &str,
    pot: u32,
    effective_stack: u32,
    oop_range: &str,
    ip_range: &str,
    first_player: &str,
    tree_preset: &str,
) -> Result<FlopTreeRequest, String> {
    Ok(FlopTreeRequest {
        board: Board::from_str(flop)?,
        pot,
        effective_stack,
        oop_range: RangeSpec::from_str(oop_range)?,
        ip_range: RangeSpec::from_str(ip_range)?,
        first_player: parse_player(first_player)?,
        action_abstraction: parse_tree_preset(tree_preset)?,
    })
}

fn parse_tree_preset(value: &str) -> Result<ActionAbstraction, String> {
    match value {
        "conservative" => Ok(ActionAbstraction::conservative_default()),
        "postflop-basic" | "postflop-solver-basic" => {
            Ok(ActionAbstraction::postflop_solver_basic())
        }
        other => Err(format!(
            "unknown tree preset {other:?}; expected conservative or postflop-basic"
        )),
    }
}

#[derive(Debug, Clone, Copy)]
struct WorkEstimate {
    private_infosets: u128,
    action_slots: u128,
    private_pairs: u128,
    terminal_pair_visits: u128,
    memory_regret_strategy_f32_mb: f64,
}

fn estimate_tree_work(tree: &PublicTree, oop_combos: usize, ip_combos: usize) -> WorkEstimate {
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

fn parse_player(value: &str) -> Result<Player, String> {
    match value.to_ascii_lowercase().as_str() {
        "oop" => Ok(Player::Oop),
        "ip" => Ok(Player::Ip),
        _ => Err(format!("invalid player {value:?}; expected oop or ip")),
    }
}

fn parse_cfr_variant(
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
            "invalid CFR variant {value:?}; expected cfr-plus, dcfr, or dcfr-plus"
        )),
    }
}

fn format_cfr_variant(variant: RealCfrVariant) -> String {
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

fn parse_average_strategy(value: &str) -> Result<RealCfrAverageStrategy, String> {
    match value.to_ascii_lowercase().as_str() {
        "reach-weighted" | "reach" | "standard" => Ok(RealCfrAverageStrategy::ReachWeighted),
        "local" | "local-unweighted" | "reference" => Ok(RealCfrAverageStrategy::Local),
        _ => Err(format!(
            "invalid average strategy {value:?}; expected reach-weighted or local"
        )),
    }
}

fn format_average_strategy(value: RealCfrAverageStrategy) -> &'static str {
    match value {
        RealCfrAverageStrategy::ReachWeighted => "reach-weighted",
        RealCfrAverageStrategy::Local => "local",
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
