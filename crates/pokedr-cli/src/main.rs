use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
};
use clap::{Parser, Subcommand};
use pokedr_agent::{FlopTreeRequest, build_flop_tree};
use pokedr_core::{
    ActionAbstraction, ActionKind, Board, CfrStorageConfig, ChanceExpansion, ComboWeight,
    NodeLocalCfrSolver, NodeLocalSolutionNodeKind, NodeLocalSolutionSnapshot, Player,
    PublicNodeKind, PublicTree, RangeSpec, RealCfrAverageStrategy, RealCfrConfig, RealCfrVariant,
    Spot, Street, TreeBuilder, TreeTemplate, fixed_flop_future_board_isomorphism,
    full_deck_future_board_isomorphism_survey, plan_cfr_work,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;
use tower_http::services::{ServeDir, ServeFile};
use tracing::info;

#[derive(Debug, Parser)]
#[command(name = "pokedr-cli")]
#[command(about = "Pokedr postflop solver tooling")]
struct Cli {
    #[arg(
        long,
        global = true,
        help = "Tracing level: error, warn, info, debug, trace"
    )]
    log_level: Option<String>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    #[command(about = "Inspect a schematic flop public tree and its exact board-expanded size")]
    BuildTree {
        #[arg(required_unless_present = "config")]
        flop: Option<String>,
        #[arg(long)]
        config: Vec<PathBuf>,
        #[arg(long)]
        pot: Option<u32>,
        #[arg(long)]
        effective_stack: Option<u32>,
        #[arg(long)]
        oop_range: Option<String>,
        #[arg(long)]
        ip_range: Option<String>,
        #[arg(long)]
        first_player: Option<String>,
        #[arg(long)]
        tree_preset: Option<String>,
        #[arg(long)]
        print_nodes: Option<usize>,
        #[arg(long)]
        enumerate_chance: bool,
        #[arg(long)]
        chunk_mib: Option<u32>,
    },
    #[command(about = "Solve a fixed flop with the node-local full-range postflop CFR solver")]
    SolveFlop {
        #[arg(required_unless_present = "config")]
        flop: Option<String>,
        #[arg(long)]
        config: Vec<PathBuf>,
        #[arg(long)]
        pot: Option<u32>,
        #[arg(long)]
        effective_stack: Option<u32>,
        #[arg(long)]
        oop_range: Option<String>,
        #[arg(long)]
        ip_range: Option<String>,
        #[arg(long)]
        first_player: Option<String>,
        #[arg(long)]
        tree_preset: Option<String>,
        #[arg(long)]
        enumerate_chance: bool,
        #[arg(long)]
        iterations: Option<u32>,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long)]
        log_interval: Option<u32>,
        #[arg(long)]
        exploitability_interval: Option<u32>,
        #[arg(long)]
        target_exploitability_bb100: Option<f32>,
        #[arg(long)]
        variant: Option<String>,
        #[arg(long)]
        average_strategy: Option<String>,
        #[arg(long)]
        dcfr_alpha: Option<f32>,
        #[arg(long)]
        dcfr_beta: Option<f32>,
        #[arg(long)]
        dcfr_gamma: Option<f32>,
    },
    #[command(about = "Solve a fixed flop and serve an interactive solution viewer")]
    Viewer {
        #[arg(required_unless_present = "config")]
        flop: Option<String>,
        #[arg(long)]
        config: Vec<PathBuf>,
        #[arg(long)]
        pot: Option<u32>,
        #[arg(long)]
        effective_stack: Option<u32>,
        #[arg(long)]
        oop_range: Option<String>,
        #[arg(long)]
        ip_range: Option<String>,
        #[arg(long)]
        first_player: Option<String>,
        #[arg(long)]
        tree_preset: Option<String>,
        #[arg(long)]
        enumerate_chance: bool,
        #[arg(long)]
        iterations: Option<u32>,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long)]
        log_interval: Option<u32>,
        #[arg(long)]
        exploitability_interval: Option<u32>,
        #[arg(long)]
        target_exploitability_bb100: Option<f32>,
        #[arg(long)]
        variant: Option<String>,
        #[arg(long)]
        average_strategy: Option<String>,
        #[arg(long)]
        dcfr_alpha: Option<f32>,
        #[arg(long)]
        dcfr_beta: Option<f32>,
        #[arg(long)]
        dcfr_gamma: Option<f32>,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 5174)]
        port: u16,
        #[arg(long, default_value = "viewer/dist")]
        assets: PathBuf,
    },
    #[command(about = "Inspect exact future-board suit isomorphism for a fixed flop and ranges")]
    BoardIsomorphism {
        #[arg(required_unless_present = "config")]
        flop: Option<String>,
        #[arg(long)]
        config: Vec<PathBuf>,
        #[arg(long)]
        oop_range: Option<String>,
        #[arg(long)]
        ip_range: Option<String>,
        #[arg(long)]
        print_turns: Option<usize>,
        #[arg(
            long,
            help = "Survey every unordered flop instead of only the supplied flop"
        )]
        survey_all_flops: bool,
    },
}

fn main() -> Result<(), String> {
    let Cli { log_level, command } = Cli::parse();
    match command {
        Command::BuildTree {
            flop,
            config,
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
            let config = load_config(&config)?;
            init_logging(log_level.as_deref(), &config)?;
            let spot = resolve_spot_options(
                &config,
                SpotCliOverrides {
                    flop,
                    pot,
                    effective_stack,
                    oop_range,
                    ip_range,
                    first_player,
                },
            )?;
            let tree_options = resolve_tree_options(&config, tree_preset, enumerate_chance);
            let request = flop_tree_request(&spot, &tree_options.tree_preset)?;
            log_config(config.source.as_deref(), &spot, &tree_options);
            let tree = build_tree(request.clone(), tree_options.enumerate_chance)?;
            print_tree_report(
                &tree,
                &request,
                chunk_mib
                    .or(config.output.as_ref().and_then(|output| output.chunk_mib))
                    .unwrap_or(256),
                print_nodes
                    .or(config.output.as_ref().and_then(|output| output.print_nodes))
                    .unwrap_or(20),
            );
        }
        Command::SolveFlop {
            flop,
            config,
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
            let config = load_config(&config)?;
            init_logging(log_level.as_deref(), &config)?;
            let spot = resolve_spot_options(
                &config,
                SpotCliOverrides {
                    flop,
                    pot,
                    effective_stack,
                    oop_range,
                    ip_range,
                    first_player,
                },
            )?;
            let tree_options = resolve_tree_options(&config, tree_preset, enumerate_chance);
            let solver_options = resolve_solver_options(
                &config,
                SolverCliOverrides {
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
                },
            )?;
            let request = flop_tree_request(&spot, &tree_options.tree_preset)?;
            log_config(config.source.as_deref(), &spot, &tree_options);
            log_solver_config(&solver_options);
            let tree = build_tree(request.clone(), tree_options.enumerate_chance)?;
            log_tree_summary(&tree, &request);
            solve_flop(
                tree,
                request.oop_range,
                request.ip_range,
                solver_options.iterations,
                solver_options.threads,
                solver_options.log_interval,
                solver_options.exploitability_interval,
                solver_options.target_exploitability_bb100,
                RealCfrConfig {
                    iterations: solver_options.iterations,
                    variant: solver_options.variant,
                    average_strategy: solver_options.average_strategy,
                },
            )?;
        }
        Command::Viewer {
            flop,
            config,
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
            host,
            port,
            assets,
        } => {
            let config = load_config(&config)?;
            init_logging(log_level.as_deref(), &config)?;
            let spot = resolve_spot_options(
                &config,
                SpotCliOverrides {
                    flop,
                    pot,
                    effective_stack,
                    oop_range,
                    ip_range,
                    first_player,
                },
            )?;
            let tree_options = resolve_tree_options(&config, tree_preset, enumerate_chance);
            let solver_options = resolve_solver_options(
                &config,
                SolverCliOverrides {
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
                },
            )?;
            let request = flop_tree_request(&spot, &tree_options.tree_preset)?;
            log_config(config.source.as_deref(), &spot, &tree_options);
            log_solver_config(&solver_options);
            let tree = build_tree(request.clone(), tree_options.enumerate_chance)?;
            log_tree_summary(&tree, &request);
            let solution = solve_for_viewer(tree, request, &solver_options)?;
            serve_viewer(solution, host, port, assets)?;
        }
        Command::BoardIsomorphism {
            flop,
            config,
            oop_range,
            ip_range,
            print_turns,
            survey_all_flops,
        } => {
            let config = load_config(&config)?;
            init_logging(log_level.as_deref(), &config)?;
            let spot = resolve_spot_options(
                &config,
                SpotCliOverrides {
                    flop,
                    pot: None,
                    effective_stack: None,
                    oop_range,
                    ip_range,
                    first_player: None,
                },
            )?;
            let oop_range = RangeSpec::from_str(&spot.oop_range)?;
            let ip_range = RangeSpec::from_str(&spot.ip_range)?;
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
            let flop = Board::from_str(&spot.flop)?;
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
            let print_turns = print_turns
                .or(config.output.as_ref().and_then(|output| output.print_turns))
                .unwrap_or(8);
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

#[derive(Debug, Default, Deserialize)]
struct CliConfig {
    #[serde(skip)]
    source: Option<String>,
    #[serde(default)]
    spot: Option<SpotConfig>,
    #[serde(default)]
    tree: Option<TreeConfig>,
    #[serde(default)]
    solver: Option<SolverConfig>,
    #[serde(default)]
    output: Option<OutputConfig>,
    #[serde(default)]
    logging: Option<LoggingConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct SpotConfig {
    flop: Option<String>,
    pot: Option<u32>,
    effective_stack: Option<u32>,
    oop_range: Option<String>,
    ip_range: Option<String>,
    first_player: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct TreeConfig {
    preset: Option<String>,
    tree_preset: Option<String>,
    enumerate_chance: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct SolverConfig {
    iterations: Option<u32>,
    threads: Option<usize>,
    log_interval: Option<u32>,
    exploitability_interval: Option<u32>,
    target_exploitability_bb100: Option<f32>,
    variant: Option<String>,
    average_strategy: Option<String>,
    dcfr_alpha: Option<f32>,
    dcfr_beta: Option<f32>,
    dcfr_gamma: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
struct OutputConfig {
    print_nodes: Option<usize>,
    print_turns: Option<usize>,
    chunk_mib: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
struct LoggingConfig {
    level: Option<String>,
}

impl CliConfig {
    fn merge(&mut self, other: CliConfig) {
        merge_option_struct(&mut self.spot, other.spot, SpotConfig::merge);
        merge_option_struct(&mut self.tree, other.tree, TreeConfig::merge);
        merge_option_struct(&mut self.solver, other.solver, SolverConfig::merge);
        merge_option_struct(&mut self.output, other.output, OutputConfig::merge);
        merge_option_struct(&mut self.logging, other.logging, LoggingConfig::merge);
    }
}

impl SpotConfig {
    fn merge(&mut self, other: Self) {
        merge_option(&mut self.flop, other.flop);
        merge_option(&mut self.pot, other.pot);
        merge_option(&mut self.effective_stack, other.effective_stack);
        merge_option(&mut self.oop_range, other.oop_range);
        merge_option(&mut self.ip_range, other.ip_range);
        merge_option(&mut self.first_player, other.first_player);
    }
}

impl TreeConfig {
    fn merge(&mut self, other: Self) {
        merge_option(&mut self.preset, other.preset);
        merge_option(&mut self.tree_preset, other.tree_preset);
        merge_option(&mut self.enumerate_chance, other.enumerate_chance);
    }
}

impl SolverConfig {
    fn merge(&mut self, other: Self) {
        merge_option(&mut self.iterations, other.iterations);
        merge_option(&mut self.threads, other.threads);
        merge_option(&mut self.log_interval, other.log_interval);
        merge_option(
            &mut self.exploitability_interval,
            other.exploitability_interval,
        );
        merge_option(
            &mut self.target_exploitability_bb100,
            other.target_exploitability_bb100,
        );
        merge_option(&mut self.variant, other.variant);
        merge_option(&mut self.average_strategy, other.average_strategy);
        merge_option(&mut self.dcfr_alpha, other.dcfr_alpha);
        merge_option(&mut self.dcfr_beta, other.dcfr_beta);
        merge_option(&mut self.dcfr_gamma, other.dcfr_gamma);
    }
}

impl OutputConfig {
    fn merge(&mut self, other: Self) {
        merge_option(&mut self.print_nodes, other.print_nodes);
        merge_option(&mut self.print_turns, other.print_turns);
        merge_option(&mut self.chunk_mib, other.chunk_mib);
    }
}

impl LoggingConfig {
    fn merge(&mut self, other: Self) {
        merge_option(&mut self.level, other.level);
    }
}

fn merge_option<T>(base: &mut Option<T>, override_value: Option<T>) {
    if override_value.is_some() {
        *base = override_value;
    }
}

fn merge_option_struct<T>(
    base: &mut Option<T>,
    override_value: Option<T>,
    merge: impl FnOnce(&mut T, T),
) {
    if let Some(override_value) = override_value {
        if let Some(base) = base {
            merge(base, override_value);
        } else {
            *base = Some(override_value);
        }
    }
}

#[derive(Debug)]
struct SpotCliOverrides {
    flop: Option<String>,
    pot: Option<u32>,
    effective_stack: Option<u32>,
    oop_range: Option<String>,
    ip_range: Option<String>,
    first_player: Option<String>,
}

#[derive(Debug)]
struct SolverCliOverrides {
    iterations: Option<u32>,
    threads: Option<usize>,
    log_interval: Option<u32>,
    exploitability_interval: Option<u32>,
    target_exploitability_bb100: Option<f32>,
    variant: Option<String>,
    average_strategy: Option<String>,
    dcfr_alpha: Option<f32>,
    dcfr_beta: Option<f32>,
    dcfr_gamma: Option<f32>,
}

#[derive(Debug)]
struct SpotOptions {
    flop: String,
    pot: u32,
    effective_stack: u32,
    oop_range: String,
    ip_range: String,
    first_player: String,
}

#[derive(Debug)]
struct TreeOptions {
    tree_preset: String,
    enumerate_chance: bool,
}

#[derive(Debug)]
struct SolverOptions {
    iterations: u32,
    threads: usize,
    log_interval: u32,
    exploitability_interval: u32,
    target_exploitability_bb100: Option<f32>,
    variant: RealCfrVariant,
    average_strategy: RealCfrAverageStrategy,
}

fn load_config(paths: &[PathBuf]) -> Result<CliConfig, String> {
    if paths.is_empty() {
        return Ok(CliConfig::default());
    }
    let mut merged = CliConfig::default();
    let mut sources = Vec::with_capacity(paths.len());
    for path in paths {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("failed to read config {}: {error}", path.display()))?;
        let config: CliConfig = toml::from_str(&text)
            .map_err(|error| format!("failed to parse config {}: {error}", path.display()))?;
        sources.push(path.display().to_string());
        merged.merge(config);
    }
    merged.source = Some(sources.join(","));
    Ok(merged)
}

fn init_logging(cli_level: Option<&str>, config: &CliConfig) -> Result<(), String> {
    let level = cli_level
        .or_else(|| {
            config
                .logging
                .as_ref()
                .and_then(|logging| logging.level.as_deref())
        })
        .unwrap_or("info");
    let level = tracing::Level::from_str(level).map_err(|_| {
        format!("invalid log level {level:?}; expected error, warn, info, debug, or trace")
    })?;
    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(false)
        .without_time()
        .compact()
        .try_init()
        .map_err(|error| format!("failed to initialize tracing subscriber: {error}"))
}

fn resolve_spot_options(config: &CliConfig, cli: SpotCliOverrides) -> Result<SpotOptions, String> {
    let spot = config.spot.as_ref();
    let flop = cli
        .flop
        .or_else(|| spot.and_then(|spot| spot.flop.clone()))
        .ok_or_else(|| {
            "missing flop; pass it as an argument or set spot.flop in config".to_string()
        })?;
    Ok(SpotOptions {
        flop,
        pot: cli
            .pot
            .or_else(|| spot.and_then(|spot| spot.pot))
            .unwrap_or(650),
        effective_stack: cli
            .effective_stack
            .or_else(|| spot.and_then(|spot| spot.effective_stack))
            .unwrap_or(9700),
        oop_range: cli
            .oop_range
            .or_else(|| spot.and_then(|spot| spot.oop_range.clone()))
            .unwrap_or_else(|| "full".to_string()),
        ip_range: cli
            .ip_range
            .or_else(|| spot.and_then(|spot| spot.ip_range.clone()))
            .unwrap_or_else(|| "full".to_string()),
        first_player: cli
            .first_player
            .or_else(|| spot.and_then(|spot| spot.first_player.clone()))
            .unwrap_or_else(|| "oop".to_string()),
    })
}

fn resolve_tree_options(
    config: &CliConfig,
    cli_tree_preset: Option<String>,
    cli_enumerate_chance: bool,
) -> TreeOptions {
    let tree = config.tree.as_ref();
    TreeOptions {
        tree_preset: cli_tree_preset
            .or_else(|| tree.and_then(|tree| tree.tree_preset.clone()))
            .or_else(|| tree.and_then(|tree| tree.preset.clone()))
            .unwrap_or_else(|| "conservative".to_string()),
        enumerate_chance: cli_enumerate_chance
            || tree.and_then(|tree| tree.enumerate_chance).unwrap_or(false),
    }
}

fn resolve_solver_options(
    config: &CliConfig,
    cli: SolverCliOverrides,
) -> Result<SolverOptions, String> {
    let solver = config.solver.as_ref();
    let iterations = cli
        .iterations
        .or_else(|| solver.and_then(|solver| solver.iterations))
        .unwrap_or(1);
    let threads = cli
        .threads
        .or_else(|| solver.and_then(|solver| solver.threads))
        .map(resolve_thread_count)
        .unwrap_or_else(default_thread_count);
    let log_interval = cli
        .log_interval
        .or_else(|| solver.and_then(|solver| solver.log_interval))
        .unwrap_or(1);
    let exploitability_interval = cli
        .exploitability_interval
        .or_else(|| solver.and_then(|solver| solver.exploitability_interval))
        .unwrap_or(0);
    let target_exploitability_bb100 = cli
        .target_exploitability_bb100
        .or_else(|| solver.and_then(|solver| solver.target_exploitability_bb100));
    let variant = cli
        .variant
        .or_else(|| solver.and_then(|solver| solver.variant.clone()))
        .unwrap_or_else(|| "dcfr-plus".to_string());
    let average_strategy = cli
        .average_strategy
        .or_else(|| solver.and_then(|solver| solver.average_strategy.clone()))
        .unwrap_or_else(|| "reach-weighted".to_string());
    let dcfr_alpha = cli
        .dcfr_alpha
        .or_else(|| solver.and_then(|solver| solver.dcfr_alpha))
        .unwrap_or(1.5);
    let dcfr_beta = cli
        .dcfr_beta
        .or_else(|| solver.and_then(|solver| solver.dcfr_beta))
        .unwrap_or(0.0);
    let dcfr_gamma = cli
        .dcfr_gamma
        .or_else(|| solver.and_then(|solver| solver.dcfr_gamma))
        .unwrap_or(2.0);
    Ok(SolverOptions {
        iterations,
        threads,
        log_interval,
        exploitability_interval,
        target_exploitability_bb100,
        variant: parse_cfr_variant(&variant, dcfr_alpha, dcfr_beta, dcfr_gamma)?,
        average_strategy: parse_average_strategy(&average_strategy)?,
    })
}

fn resolve_thread_count(threads: usize) -> usize {
    if threads == 0 {
        default_thread_count()
    } else {
        threads
    }
}

fn default_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

fn log_config(source: Option<&str>, spot: &SpotOptions, tree: &TreeOptions) {
    info!(
        config = source.unwrap_or("<cli-only>"),
        flop = %spot.flop,
        pot_bb = spot.pot as f32 / 100.0,
        effective_stack_bb = spot.effective_stack as f32 / 100.0,
        first_player = %spot.first_player,
        oop_range = %spot.oop_range,
        ip_range = %spot.ip_range,
        tree_preset = %tree.tree_preset,
        enumerate_chance = tree.enumerate_chance,
        "solver_input"
    );
}

fn log_solver_config(options: &SolverOptions) {
    info!(
        iterations = options.iterations,
        threads = options.threads,
        log_interval = options.log_interval,
        exploitability_interval = options.exploitability_interval,
        target_exploitability_bb100 = options.target_exploitability_bb100,
        variant = %format_cfr_variant(options.variant),
        average_strategy = format_average_strategy(options.average_strategy),
        "solver_config"
    );
}

fn log_tree_summary(tree: &PublicTree, request: &FlopTreeRequest) {
    let stats = tree.stats();
    let estimate = estimate_tree_work(
        tree,
        request.oop_range.combos().len(),
        request.ip_range.combos().len(),
    );
    info!(
        nodes = stats.nodes,
        decisions = stats.decisions,
        chances = stats.chances,
        terminals = stats.terminals,
        max_depth = stats.max_depth,
        private_infosets = estimate.private_infosets,
        action_slots = estimate.action_slots,
        terminal_pair_visits = estimate.terminal_pair_visits,
        memory_regret_strategy_f32_mb = estimate.memory_regret_strategy_f32_mb,
        "tree_summary"
    );
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
    info!(
        flop = %tree.spot.board,
        variant = %format_cfr_variant(config.variant),
        average_strategy = format_average_strategy(config.average_strategy),
        iterations,
        threads,
        "node_cfr_start"
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
                    info!(
                        iteration = global_iteration,
                        terminal_evals = progress.terminal_evals,
                        iteration_ms = progress.elapsed_ms,
                        oop_pass_value = progress.oop_update_pass_value,
                        ip_pass_value = progress.ip_update_pass_value,
                        "node_cfr_progress"
                    );
                }
            },
        )?;
        completed = summary.iterations;
        if interval > 0 {
            let exploitability = solver.exploitability(threads)?;
            info!(
                iteration = completed,
                profile_oop = exploitability.profile_oop_value,
                profile_ip = exploitability.profile_ip_value,
                zero_sum_delta = exploitability.profile_oop_value + exploitability.profile_ip_value,
                oop_br = exploitability.oop_best_response_value,
                ip_br = exploitability.ip_best_response_value,
                oop_gain = exploitability.oop_gain,
                ip_gain = exploitability.ip_gain,
                nash_conv_chips = exploitability.nash_conv_chips,
                exploitability_chips = exploitability.exploitability_chips,
                exploitability_bb_per_100 = exploitability.exploitability_bb_per_100,
                "node_cfr_exploitability"
            );
            if target_exploitability_bb100
                .is_some_and(|target| exploitability.exploitability_bb_per_100 <= target)
            {
                break;
            }
        }
    }
    info!(
        iterations = summary.iterations,
        states = summary.states,
        decision_states = summary.decision_states,
        action_slots = summary.action_slots,
        terminal_evals = summary.terminal_evals,
        elapsed_ms = summary.elapsed_ms,
        oop_pass_value = summary.oop_update_pass_value,
        ip_pass_value = summary.ip_update_pass_value,
        "node_cfr_summary"
    );
    info!(
        state_allocated = true,
        storage_gib = summary.storage_gib,
        total_elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
        "node_cfr_finish"
    );
    Ok(())
}

fn solve_for_viewer(
    tree: PublicTree,
    request: FlopTreeRequest,
    options: &SolverOptions,
) -> Result<ViewerSolution, String> {
    info!(
        flop = %tree.spot.board,
        variant = %format_cfr_variant(options.variant),
        average_strategy = format_average_strategy(options.average_strategy),
        iterations = options.iterations,
        threads = options.threads,
        "viewer_solve_start"
    );
    let started = Instant::now();
    let mut solver =
        NodeLocalCfrSolver::new(tree, request.oop_range.clone(), request.ip_range.clone())?;
    let summary = solver.run_with_progress(
        RealCfrConfig {
            iterations: options.iterations,
            variant: options.variant,
            average_strategy: options.average_strategy,
        },
        |progress| {
            if options.log_interval > 0
                && (progress.iteration == 1
                    || progress.iteration == options.iterations
                    || progress.iteration % options.log_interval == 0)
            {
                info!(
                    iteration = progress.iteration,
                    terminal_evals = progress.terminal_evals,
                    iteration_ms = progress.elapsed_ms,
                    oop_pass_value = progress.oop_update_pass_value,
                    ip_pass_value = progress.ip_update_pass_value,
                    "viewer_solve_progress"
                );
            }
        },
    )?;
    let exploitability =
        if options.exploitability_interval > 0 || options.target_exploitability_bb100.is_some() {
            Some(solver.exploitability(options.threads)?)
        } else {
            None
        };
    let snapshot = solver.solution_snapshot();
    info!(
        iterations = summary.iterations,
        nodes = snapshot.nodes.len(),
        storage_gib = summary.storage_gib,
        total_elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
        "viewer_solve_finish"
    );
    Ok(viewer_solution_from_snapshot(
        snapshot,
        request,
        summary,
        exploitability.map(|value| value.exploitability_bb_per_100),
        started.elapsed().as_secs_f64() * 1000.0,
    ))
}

#[derive(Debug, Clone, Serialize)]
struct ViewerSolution {
    summary: ViewerSummary,
    oop_combos: Vec<ViewerCombo>,
    ip_combos: Vec<ViewerCombo>,
    nodes: Vec<ViewerNode>,
}

#[derive(Debug, Clone, Serialize)]
struct ViewerSummary {
    board: String,
    pot_bb: f32,
    effective_stack_bb: f32,
    first_player: String,
    iterations: u32,
    solver_elapsed_ms: f64,
    storage_gib: f64,
    exploitability_bb_per_100: Option<f32>,
    nodes: usize,
    decision_states: usize,
    action_slots: usize,
    oop_combos: usize,
    ip_combos: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ViewerCombo {
    index: usize,
    label: String,
    class: String,
    weight: f32,
}

#[derive(Debug, Clone, Serialize)]
struct ViewerNode {
    id: usize,
    public_node: usize,
    board: String,
    street: String,
    pot_bb: f32,
    player: String,
    kind: String,
    children: Vec<usize>,
    actions: Vec<ViewerAction>,
    strategy: Option<ViewerStrategy>,
}

#[derive(Debug, Clone, Serialize)]
struct ViewerNodeListItem {
    id: usize,
    public_node: usize,
    board: String,
    street: String,
    pot_bb: f32,
    player: String,
    kind: String,
    children: Vec<usize>,
    actions: Vec<ViewerAction>,
}

#[derive(Debug, Clone, Serialize)]
struct ViewerAction {
    index: usize,
    label: String,
    child: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct ViewerStrategy {
    player: String,
    combos: usize,
    actions: usize,
    action_major: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
struct ViewerCombos {
    oop: Vec<ViewerCombo>,
    ip: Vec<ViewerCombo>,
}

fn viewer_solution_from_snapshot(
    snapshot: NodeLocalSolutionSnapshot,
    request: FlopTreeRequest,
    summary: pokedr_core::NodeLocalCfrSummary,
    exploitability_bb_per_100: Option<f32>,
    solver_elapsed_ms: f64,
) -> ViewerSolution {
    let oop_combos = viewer_combos(&snapshot.oop_combos);
    let ip_combos = viewer_combos(&snapshot.ip_combos);
    let nodes = snapshot
        .nodes
        .into_iter()
        .map(|node| {
            let actions = node
                .actions
                .iter()
                .enumerate()
                .map(|(index, action)| ViewerAction {
                    index,
                    label: format_action(action),
                    child: node.children.get(index).copied(),
                })
                .collect::<Vec<_>>();
            let kind = match node.kind {
                NodeLocalSolutionNodeKind::Decision => "decision".to_string(),
                NodeLocalSolutionNodeKind::Chance => "chance".to_string(),
                NodeLocalSolutionNodeKind::Terminal { reason } => {
                    format!("terminal:{reason:?}").to_ascii_lowercase()
                }
            };
            ViewerNode {
                id: node.id,
                public_node: node.public_node,
                board: node.board.to_string(),
                street: format!("{:?}", node.street).to_ascii_lowercase(),
                pot_bb: node.pot as f32 / 100.0,
                player: format_player(node.player),
                kind,
                children: node.children,
                actions,
                strategy: node.strategy.map(|strategy| ViewerStrategy {
                    player: format_player(strategy.player),
                    combos: strategy.combos,
                    actions: strategy.actions,
                    action_major: strategy.action_major,
                }),
            }
        })
        .collect::<Vec<_>>();
    ViewerSolution {
        summary: ViewerSummary {
            board: request.board.to_string(),
            pot_bb: request.pot as f32 / 100.0,
            effective_stack_bb: request.effective_stack as f32 / 100.0,
            first_player: format_player(request.first_player),
            iterations: snapshot.iterations,
            solver_elapsed_ms,
            storage_gib: summary.storage_gib,
            exploitability_bb_per_100,
            nodes: nodes.len(),
            decision_states: summary.decision_states,
            action_slots: summary.action_slots,
            oop_combos: oop_combos.len(),
            ip_combos: ip_combos.len(),
        },
        oop_combos,
        ip_combos,
        nodes,
    }
}

fn viewer_combos(combos: &[ComboWeight]) -> Vec<ViewerCombo> {
    combos
        .iter()
        .enumerate()
        .map(|(index, combo)| ViewerCombo {
            index,
            label: format!("{}{}", combo.first, combo.second),
            class: combo_class(combo),
            weight: combo.weight,
        })
        .collect()
}

fn combo_class(combo: &ComboWeight) -> String {
    let (high, low) = if combo.first.rank.index() >= combo.second.rank.index() {
        (combo.first, combo.second)
    } else {
        (combo.second, combo.first)
    };
    let high_rank = high.to_string().chars().next().unwrap_or('?');
    let low_rank = low.to_string().chars().next().unwrap_or('?');
    if high.rank == low.rank {
        format!("{high_rank}{low_rank}")
    } else if high.suit == low.suit {
        format!("{high_rank}{low_rank}s")
    } else {
        format!("{high_rank}{low_rank}o")
    }
}

#[derive(Clone)]
struct ViewerState {
    solution: Arc<ViewerSolution>,
}

fn serve_viewer(
    solution: ViewerSolution,
    host: String,
    port: u16,
    assets: PathBuf,
) -> Result<(), String> {
    let state = ViewerState {
        solution: Arc::new(solution),
    };
    let api = Router::new()
        .route("/solution", get(api_solution))
        .route("/combos", get(api_combos))
        .route("/summary", get(api_summary))
        .route("/nodes", get(api_nodes))
        .route("/node/{id}", get(api_node))
        .with_state(state);
    let app = Router::new().nest("/api", api).fallback_service(
        ServeDir::new(&assets).not_found_service(ServeFile::new(assets.join("index.html"))),
    );
    let address = format!("{host}:{port}");
    info!(url = %format!("http://{address}"), assets = %assets.display(), "viewer_listen");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("failed to create tokio runtime: {error}"))?;
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&address)
            .await
            .map_err(|error| format!("failed to bind viewer server at {address}: {error}"))?;
        axum::serve(listener, app)
            .await
            .map_err(|error| format!("viewer server failed: {error}"))
    })
}

async fn api_summary(State(state): State<ViewerState>) -> Json<ViewerSummary> {
    Json(state.solution.summary.clone())
}

async fn api_solution(State(state): State<ViewerState>) -> Json<ViewerSolution> {
    Json((*state.solution).clone())
}

async fn api_combos(State(state): State<ViewerState>) -> Json<ViewerCombos> {
    Json(ViewerCombos {
        oop: state.solution.oop_combos.clone(),
        ip: state.solution.ip_combos.clone(),
    })
}

async fn api_nodes(State(state): State<ViewerState>) -> Json<Vec<ViewerNodeListItem>> {
    Json(
        state
            .solution
            .nodes
            .iter()
            .map(|node| ViewerNodeListItem {
                id: node.id,
                public_node: node.public_node,
                board: node.board.clone(),
                street: node.street.clone(),
                pot_bb: node.pot_bb,
                player: node.player.clone(),
                kind: node.kind.clone(),
                children: node.children.clone(),
                actions: node.actions.clone(),
            })
            .collect(),
    )
}

async fn api_node(Path(id): Path<usize>, State(state): State<ViewerState>) -> impl IntoResponse {
    if let Some(node) = state.solution.nodes.get(id) {
        Json(node.clone()).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("unknown node {id}") })),
        )
            .into_response()
    }
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

fn flop_tree_request(spot: &SpotOptions, tree_preset: &str) -> Result<FlopTreeRequest, String> {
    Ok(FlopTreeRequest {
        board: Board::from_str(&spot.flop)?,
        pot: spot.pot,
        effective_stack: spot.effective_stack,
        oop_range: RangeSpec::from_str(&spot.oop_range)?,
        ip_range: RangeSpec::from_str(&spot.ip_range)?,
        first_player: parse_player(&spot.first_player)?,
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

fn format_player(player: Player) -> String {
    match player {
        Player::Oop => "oop",
        Player::Ip => "ip",
    }
    .to_string()
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
