use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
};
use clap::{Parser, Subcommand};
use pokedr_agent::{FlopTreeRequest, build_flop_tree};
use pokedr_core::terminal_cfv::PreparedTerminalBoard;
use pokedr_core::{
    ActionAbstraction, ActionKind, Board, Card, CfrStorageConfig, ChanceExpansion, ComboWeight,
    NodeLocalCfrSolver, NodeLocalSolutionNodeKind, NodeLocalSolutionSnapshot, Player,
    PublicNodeKind, PublicTree, RaisePolicy, RangeSpec, RealCfrAverageStrategy, RealCfrConfig,
    RealCfrVariant, Spot, Street, StreetTemplate, TreeBuilder, TreeTemplate,
    fixed_flop_future_board_isomorphism, full_deck_future_board_isomorphism_survey, plan_cfr_work,
};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tower_http::services::{ServeDir, ServeFile};
use tracing::info;

use pokedr_core::tree::{BetSizeSpec, RaiseSizeSpec};

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
        #[arg(
            long,
            help = "Write the solved public tree and strategies to a SQLite database"
        )]
        db: Option<PathBuf>,
        #[arg(
            long,
            default_value = "turn",
            help = "Maximum street whose strategy payload is written to --db: flop, turn, or river"
        )]
        db_max_street: String,
    },
    #[command(about = "Solve a fixed flop and serve an interactive solution viewer")]
    Viewer {
        #[arg(required_unless_present_any = ["config", "db"])]
        flop: Option<String>,
        #[arg(long)]
        config: Vec<PathBuf>,
        #[arg(
            long,
            help = "Serve an existing SQLite solution database instead of solving"
        )]
        db: Option<PathBuf>,
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
            let tree_options = resolve_tree_options(&config, tree_preset, enumerate_chance)?;
            let request = flop_tree_request(&spot, tree_options.action_abstraction.clone())?;
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
            db,
            db_max_street,
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
            let tree_options = resolve_tree_options(&config, tree_preset, enumerate_chance)?;
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
            let request = flop_tree_request(&spot, tree_options.action_abstraction.clone())?;
            log_config(config.source.as_deref(), &spot, &tree_options);
            log_solver_config(&solver_options);
            let tree = build_tree(request.clone(), tree_options.enumerate_chance)?;
            log_tree_summary(&tree, &request);
            let solver = solve_flop(
                tree,
                request.oop_range.clone(),
                request.ip_range.clone(),
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
            if let Some(path) = db {
                let max_street = parse_street(&db_max_street)?;
                export_solution_db(
                    &path,
                    &solver,
                    &request,
                    &solver_options,
                    tree_options.enumerate_chance,
                    max_street,
                )?;
            }
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
            db,
            host,
            port,
            assets,
        } => {
            let config = load_config(&config)?;
            init_logging(log_level.as_deref(), &config)?;
            if let Some(path) = db {
                let viewer = load_viewer_from_db(&path)?;
                serve_viewer(viewer, host, port, assets)?;
                return Ok(());
            }
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
            let tree_options = resolve_tree_options(&config, tree_preset, enumerate_chance)?;
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
            let request = flop_tree_request(&spot, tree_options.action_abstraction.clone())?;
            log_config(config.source.as_deref(), &spot, &tree_options);
            log_solver_config(&solver_options);
            let tree = build_tree(request.clone(), tree_options.enumerate_chance)?;
            log_tree_summary(&tree, &request);
            let viewer = solve_for_viewer(tree, request, &solver_options)?;
            serve_viewer(viewer, host, port, assets)?;
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
    min_bet: Option<u32>,
    flop_first_bets: Option<Vec<String>>,
    flop_donk_bets: Option<Vec<String>>,
    turn_first_bets: Option<Vec<String>>,
    turn_donk_bets: Option<Vec<String>>,
    river_first_bets: Option<Vec<String>>,
    river_donk_bets: Option<Vec<String>>,
    raise_multiplier: Option<f32>,
    raise_sizes: Option<Vec<String>>,
    max_raises_per_street: Option<u8>,
    shove_spr_threshold: Option<f32>,
    shove_commit_fraction: Option<f32>,
    add_all_in_threshold: Option<f32>,
    force_all_in_threshold: Option<f32>,
    merging_threshold: Option<f32>,
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
        merge_option(&mut self.min_bet, other.min_bet);
        merge_option(&mut self.flop_first_bets, other.flop_first_bets);
        merge_option(&mut self.flop_donk_bets, other.flop_donk_bets);
        merge_option(&mut self.turn_first_bets, other.turn_first_bets);
        merge_option(&mut self.turn_donk_bets, other.turn_donk_bets);
        merge_option(&mut self.river_first_bets, other.river_first_bets);
        merge_option(&mut self.river_donk_bets, other.river_donk_bets);
        merge_option(&mut self.raise_multiplier, other.raise_multiplier);
        merge_option(&mut self.raise_sizes, other.raise_sizes);
        merge_option(&mut self.max_raises_per_street, other.max_raises_per_street);
        merge_option(&mut self.shove_spr_threshold, other.shove_spr_threshold);
        merge_option(&mut self.shove_commit_fraction, other.shove_commit_fraction);
        merge_option(&mut self.add_all_in_threshold, other.add_all_in_threshold);
        merge_option(
            &mut self.force_all_in_threshold,
            other.force_all_in_threshold,
        );
        merge_option(&mut self.merging_threshold, other.merging_threshold);
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
    action_abstraction: ActionAbstraction,
}

#[derive(Debug, Clone)]
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
) -> Result<TreeOptions, String> {
    let tree = config.tree.as_ref();
    let tree_preset = cli_tree_preset
        .or_else(|| tree.and_then(|tree| tree.tree_preset.clone()))
        .or_else(|| tree.and_then(|tree| tree.preset.clone()))
        .unwrap_or_else(|| "conservative".to_string());
    let mut action_abstraction = parse_tree_preset(&tree_preset)?;
    if let Some(tree) = tree {
        apply_tree_config_overrides(&mut action_abstraction, tree)?;
    }
    Ok(TreeOptions {
        tree_preset,
        enumerate_chance: cli_enumerate_chance
            || tree.and_then(|tree| tree.enumerate_chance).unwrap_or(false),
        action_abstraction,
    })
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
        min_bet_bb = tree.action_abstraction.min_bet as f32 / 100.0,
        max_raises_per_street = tree.action_abstraction.raise.max_raises_per_street,
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
) -> Result<NodeLocalCfrSolver, String> {
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
    if summary.terminal_calls > 0 {
        info!(
            scratch_allocations = summary.scratch_allocations,
            terminal_calls = summary.terminal_calls,
            terminal_ms = ns_to_ms(summary.terminal_ns),
            fold_calls = summary.fold_calls,
            fold_ms = ns_to_ms(summary.fold_ns),
            showdown_calls = summary.showdown_calls,
            showdown_ms = ns_to_ms(summary.showdown_ns),
            showdown_only_calls = summary.showdown_only_calls,
            showdown_only_ms = ns_to_ms(summary.showdown_only_ns),
            allin_calls = summary.allin_calls,
            allin_ms = ns_to_ms(summary.allin_ns),
            allin_flop_calls = summary.allin_flop_calls,
            allin_flop_ms = ns_to_ms(summary.allin_flop_ns),
            allin_turn_calls = summary.allin_turn_calls,
            allin_turn_ms = ns_to_ms(summary.allin_turn_ns),
            allin_river_calls = summary.allin_river_calls,
            allin_river_ms = ns_to_ms(summary.allin_river_ns),
            "node_cfr_profile"
        );
    }
    info!(
        state_allocated = true,
        storage_gib = summary.storage_gib,
        total_elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
        "node_cfr_finish"
    );
    Ok(solver)
}

fn export_solution_db(
    path: &PathBuf,
    solver: &NodeLocalCfrSolver,
    request: &FlopTreeRequest,
    options: &SolverOptions,
    enumerate_chance: bool,
    max_strategy_street: Street,
) -> Result<(), String> {
    let summary = solver.summary();
    let snapshot = solver.solution_snapshot_until_street(Some(max_strategy_street));
    let mut connection = Connection::open(path).map_err(|error| {
        format!(
            "failed to open solution database {}: {error}",
            path.display()
        )
    })?;
    connection
        .execute_batch(
            r#"
            PRAGMA foreign_keys = OFF;
            DROP TABLE IF EXISTS metadata;
            DROP TABLE IF EXISTS combos;
            DROP TABLE IF EXISTS nodes;
            DROP TABLE IF EXISTS actions;
            DROP TABLE IF EXISTS edges;

            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE combos (
                player TEXT NOT NULL,
                combo_index INTEGER NOT NULL,
                label TEXT NOT NULL,
                class TEXT NOT NULL,
                first_card TEXT NOT NULL,
                second_card TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY (player, combo_index)
            );

            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY,
                public_node INTEGER NOT NULL,
                street TEXT NOT NULL,
                board TEXT NOT NULL,
                pot_bb REAL NOT NULL,
                player TEXT NOT NULL,
                kind TEXT NOT NULL,
                terminal_reason TEXT,
                strategy_player TEXT,
                strategy_combos INTEGER,
                strategy_actions INTEGER,
                strategy_action_major_json TEXT
            );

            CREATE TABLE actions (
                node_id INTEGER NOT NULL,
                action_index INTEGER NOT NULL,
                label TEXT NOT NULL,
                child_id INTEGER,
                PRIMARY KEY (node_id, action_index)
            );

            CREATE TABLE edges (
                parent_id INTEGER NOT NULL,
                edge_index INTEGER NOT NULL,
                child_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                PRIMARY KEY (parent_id, edge_index)
            );

            CREATE INDEX idx_nodes_public_node ON nodes(public_node);
            CREATE INDEX idx_nodes_street ON nodes(street);
            CREATE INDEX idx_nodes_board ON nodes(board);
            CREATE INDEX idx_edges_child ON edges(child_id);
            "#,
        )
        .map_err(|error| format!("failed to initialize solution database: {error}"))?;

    let transaction = connection
        .transaction()
        .map_err(|error| format!("failed to start solution database transaction: {error}"))?;
    insert_metadata(
        &transaction,
        &[
            ("schema_version", "1".to_string()),
            ("flop", request.board.to_string()),
            ("pot", request.pot.to_string()),
            ("effective_stack", request.effective_stack.to_string()),
            ("first_player", format_player(request.first_player)),
            ("oop_range", format_exact_range(&snapshot.oop_combos)),
            ("ip_range", format_exact_range(&snapshot.ip_combos)),
            ("iterations", snapshot.iterations.to_string()),
            ("variant", format_cfr_variant(options.variant)),
            (
                "average_strategy",
                format_average_strategy(options.average_strategy).to_string(),
            ),
            ("threads", options.threads.to_string()),
            ("enumerate_chance", enumerate_chance.to_string()),
            (
                "max_strategy_street",
                format_street(max_strategy_street).to_string(),
            ),
            ("states", summary.states.to_string()),
            ("decision_states", summary.decision_states.to_string()),
            ("action_slots", summary.action_slots.to_string()),
            ("storage_gib", summary.storage_gib.to_string()),
            (
                "action_abstraction_json",
                action_abstraction_json(&request.action_abstraction).to_string(),
            ),
        ],
    )?;
    insert_combos(&transaction, "oop", &snapshot.oop_combos)?;
    insert_combos(&transaction, "ip", &snapshot.ip_combos)?;
    let node_ids = snapshot
        .nodes
        .iter()
        .map(|node| node.id)
        .collect::<HashSet<_>>();
    let nodes_by_id = snapshot
        .nodes
        .iter()
        .map(|node| (node.id, node))
        .collect::<HashMap<_, _>>();
    for node in &snapshot.nodes {
        insert_solution_node(&transaction, node)?;
    }
    for node in &snapshot.nodes {
        insert_solution_edges(&transaction, &node_ids, &nodes_by_id, node)?;
    }
    transaction
        .commit()
        .map_err(|error| format!("failed to commit solution database: {error}"))?;
    info!(
        path = %path.display(),
        nodes = snapshot.nodes.len(),
        max_strategy_street = format_street(max_strategy_street),
        "solution_db_written"
    );
    Ok(())
}

fn insert_metadata(connection: &Connection, values: &[(&str, String)]) -> Result<(), String> {
    for (key, value) in values {
        connection
            .execute(
                "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
                params![key, value],
            )
            .map_err(|error| format!("failed to insert metadata {key}: {error}"))?;
    }
    Ok(())
}

fn insert_combos(
    connection: &Connection,
    player: &str,
    combos: &[ComboWeight],
) -> Result<(), String> {
    for (index, combo) in combos.iter().enumerate() {
        connection
            .execute(
                r#"
                INSERT INTO combos
                    (player, combo_index, label, class, first_card, second_card, weight)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                "#,
                params![
                    player,
                    index as i64,
                    format!("{}{}", combo.first, combo.second),
                    combo_class(combo),
                    combo.first.to_string(),
                    combo.second.to_string(),
                    combo.weight,
                ],
            )
            .map_err(|error| format!("failed to insert {player} combo {index}: {error}"))?;
    }
    Ok(())
}

fn insert_solution_node(
    connection: &Connection,
    node: &pokedr_core::NodeLocalSolutionNode,
) -> Result<(), String> {
    let (kind, terminal_reason) = match node.kind {
        NodeLocalSolutionNodeKind::Decision => ("decision", None),
        NodeLocalSolutionNodeKind::Chance => ("chance", None),
        NodeLocalSolutionNodeKind::Terminal { reason } => {
            ("terminal", Some(format!("{reason:?}").to_ascii_lowercase()))
        }
    };
    let strategy_player = node
        .strategy
        .as_ref()
        .map(|strategy| format_player(strategy.player));
    let strategy_combos = node
        .strategy
        .as_ref()
        .map(|strategy| strategy.combos as i64);
    let strategy_actions = node
        .strategy
        .as_ref()
        .map(|strategy| strategy.actions as i64);
    let strategy_json = node
        .strategy
        .as_ref()
        .map(|strategy| serde_json::to_string(&strategy.action_major))
        .transpose()
        .map_err(|error| format!("failed to serialize strategy for node {}: {error}", node.id))?;
    connection
        .execute(
            r#"
            INSERT INTO nodes
                (id, public_node, street, board, pot_bb, player, kind, terminal_reason,
                 strategy_player, strategy_combos, strategy_actions, strategy_action_major_json)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
            "#,
            params![
                node.id as i64,
                node.public_node as i64,
                format_street(node.street),
                node.board.to_string(),
                node.pot as f32 / 100.0,
                format_player(node.player),
                kind,
                terminal_reason,
                strategy_player,
                strategy_combos,
                strategy_actions,
                strategy_json,
            ],
        )
        .map_err(|error| format!("failed to insert node {}: {error}", node.id))?;
    Ok(())
}

fn insert_solution_edges(
    connection: &Connection,
    node_ids: &HashSet<usize>,
    nodes_by_id: &HashMap<usize, &pokedr_core::NodeLocalSolutionNode>,
    node: &pokedr_core::NodeLocalSolutionNode,
) -> Result<(), String> {
    for (index, action) in node.actions.iter().enumerate() {
        connection
            .execute(
                "INSERT INTO actions (node_id, action_index, label, child_id) VALUES (?1, ?2, ?3, ?4)",
                params![
                    node.id as i64,
                    index as i64,
                    format_action(action),
                    node.children.get(index).copied().map(|child| child as i64)
                ],
            )
            .map_err(|error| format!("failed to insert action {index} for node {}: {error}", node.id))?;
    }
    for (index, child) in node.children.iter().copied().enumerate() {
        if !node_ids.contains(&child) {
            continue;
        }
        let label = node
            .actions
            .get(index)
            .map(format_action)
            .or_else(|| {
                nodes_by_id
                    .get(&child)
                    .map(|child_node| child_node.board.to_string())
            })
            .unwrap_or_else(|| format!("next:{child}"));
        connection
            .execute(
                "INSERT INTO edges (parent_id, edge_index, child_id, label) VALUES (?1, ?2, ?3, ?4)",
                params![node.id as i64, index as i64, child as i64, label],
            )
            .map_err(|error| {
                format!(
                    "failed to insert edge {index} for node {} -> {child}: {error}",
                    node.id
                )
            })?;
    }
    Ok(())
}

fn format_exact_range(combos: &[ComboWeight]) -> String {
    combos
        .iter()
        .map(|combo| {
            let label = format!("{}{}", combo.first, combo.second);
            if (combo.weight - 1.0).abs() < f32::EPSILON {
                label
            } else {
                format!("{label}:{}", combo.weight)
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn action_abstraction_json(action: &ActionAbstraction) -> serde_json::Value {
    serde_json::json!({
        "min_bet": action.min_bet,
        "flop": {
            "first_bet_sizes": action.flop.first_bet_sizes.iter().map(format_bet_size_spec).collect::<Vec<_>>(),
            "donk_bet_sizes": action.flop.donk_bet_sizes.iter().map(format_bet_size_spec).collect::<Vec<_>>(),
        },
        "turn": {
            "first_bet_sizes": action.turn.first_bet_sizes.iter().map(format_bet_size_spec).collect::<Vec<_>>(),
            "donk_bet_sizes": action.turn.donk_bet_sizes.iter().map(format_bet_size_spec).collect::<Vec<_>>(),
        },
        "river": {
            "first_bet_sizes": action.river.first_bet_sizes.iter().map(format_bet_size_spec).collect::<Vec<_>>(),
            "donk_bet_sizes": action.river.donk_bet_sizes.iter().map(format_bet_size_spec).collect::<Vec<_>>(),
        },
        "raise": {
            "raise_multiplier": action.raise.raise_multiplier,
            "raise_sizes": action.raise.raise_sizes.iter().map(format_raise_size_spec).collect::<Vec<_>>(),
            "max_raises_per_street": action.raise.max_raises_per_street,
            "shove_spr_threshold": action.raise.shove_spr_threshold,
            "shove_commit_fraction": action.raise.shove_commit_fraction,
            "add_all_in_threshold": action.raise.add_all_in_threshold,
            "force_all_in_threshold": action.raise.force_all_in_threshold,
            "merging_threshold": action.raise.merging_threshold,
        },
    })
}

fn format_bet_size_spec(value: &BetSizeSpec) -> String {
    match value {
        BetSizeSpec::PotFraction(fraction) => format!("{fraction}p"),
        BetSizeSpec::Geometric {
            streets,
            max_pot_fraction,
        } => format!("geo:{streets}:{max_pot_fraction}"),
        BetSizeSpec::AllIn => "allin".to_string(),
    }
}

fn format_raise_size_spec(value: &RaiseSizeSpec) -> String {
    match value {
        RaiseSizeSpec::PotFraction(fraction) => format!("{fraction}p"),
        RaiseSizeSpec::PreviousBetMultiplier(multiplier) => format!("{multiplier}x"),
        RaiseSizeSpec::Geometric {
            streets,
            max_pot_fraction,
        } => format!("geo:{streets}:{max_pot_fraction}"),
        RaiseSizeSpec::AllIn => "allin".to_string(),
    }
}

fn ns_to_ms(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

fn solve_for_viewer(
    tree: PublicTree,
    request: FlopTreeRequest,
    options: &SolverOptions,
) -> Result<ViewerBundle, String> {
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
    let interval = if options.exploitability_interval > 0 {
        options.exploitability_interval
    } else if options.target_exploitability_bb100.is_some() {
        options.log_interval.max(16)
    } else {
        0
    };
    let mut completed = 0u32;
    let mut summary = solver.summary();
    let mut exploitability = None;
    while completed < options.iterations {
        let remaining = options.iterations - completed;
        let chunk = if interval > 0 {
            remaining.min(interval)
        } else {
            remaining
        };
        let chunk_start = completed;
        summary = solver.run_with_progress(
            RealCfrConfig {
                iterations: chunk,
                variant: options.variant,
                average_strategy: options.average_strategy,
            },
            |progress| {
                let global_iteration = chunk_start + progress.iteration;
                if options.log_interval > 0
                    && (global_iteration == 1
                        || global_iteration == options.iterations
                        || global_iteration % options.log_interval == 0)
                {
                    info!(
                        iteration = global_iteration,
                        terminal_evals = progress.terminal_evals,
                        iteration_ms = progress.elapsed_ms,
                        oop_pass_value = progress.oop_update_pass_value,
                        ip_pass_value = progress.ip_update_pass_value,
                        "viewer_solve_progress"
                    );
                }
            },
        )?;
        completed = summary.iterations;
        if interval > 0 {
            let current = solver.exploitability(options.threads)?;
            info!(
                iteration = completed,
                exploitability_bb_per_100 = current.exploitability_bb_per_100,
                "viewer_solve_exploitability"
            );
            let reached_target = options
                .target_exploitability_bb100
                .is_some_and(|target| current.exploitability_bb_per_100 <= target);
            exploitability = Some(current);
            if reached_target {
                break;
            }
        }
    }
    let snapshot = solver.solution_snapshot();
    info!(
        iterations = summary.iterations,
        nodes = snapshot.nodes.len(),
        storage_gib = summary.storage_gib,
        total_elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
        "viewer_solve_finish"
    );
    let solution = viewer_solution_from_snapshot(
        snapshot,
        request,
        summary,
        exploitability.map(|value| value.exploitability_bb_per_100),
        started.elapsed().as_secs_f64() * 1000.0,
    );
    Ok(ViewerBundle {
        solution,
        solver: Some(solver),
        resolver: None,
    })
}

#[derive(Debug, Clone, Serialize)]
struct ViewerSolution {
    summary: ViewerSummary,
    oop_combos: Vec<ViewerCombo>,
    ip_combos: Vec<ViewerCombo>,
    nodes: Vec<ViewerNode>,
    #[serde(skip)]
    oop_weights: Vec<ComboWeight>,
    #[serde(skip)]
    ip_weights: Vec<ComboWeight>,
}

struct ViewerBundle {
    solution: ViewerSolution,
    solver: Option<NodeLocalCfrSolver>,
    resolver: Option<ViewerDbResolver>,
}

#[derive(Debug, Clone)]
struct ViewerDbResolver {
    db_path: PathBuf,
    request: FlopTreeRequest,
    options: SolverOptions,
    enumerate_chance: bool,
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
    choices: Vec<ViewerBranch>,
    strategy: Option<ViewerStrategy>,
}

#[derive(Debug, Clone, Serialize)]
struct ViewerAction {
    index: usize,
    label: String,
    child: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct ViewerBranch {
    label: String,
    child: usize,
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

#[derive(Debug, Clone, Serialize)]
struct ViewerEquity {
    board: String,
    pot_bb: f32,
    terminal_boards: usize,
    pair_weight: f64,
    oop_equity: f32,
    ip_equity: f32,
    oop_win_weight: f64,
    ip_win_weight: f64,
    tie_weight: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ViewerStrategyEv {
    board: String,
    pot_bb: f32,
    oop_ev_bb: f32,
    ip_ev_bb: f32,
    oop_weight: f32,
    ip_weight: f32,
    terminal_evals: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ViewerActionEv {
    board: String,
    pot_bb: f32,
    player: String,
    combos: usize,
    actions: usize,
    action_major_bb: Vec<f32>,
    terminal_evals: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ViewerReach {
    oop: Vec<f32>,
    ip: Vec<f32>,
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
    let mut nodes = snapshot
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
                choices: Vec::new(),
                strategy: node.strategy.map(|strategy| ViewerStrategy {
                    player: format_player(strategy.player),
                    combos: strategy.combos,
                    actions: strategy.actions,
                    action_major: strategy.action_major,
                }),
            }
        })
        .collect::<Vec<_>>();
    for index in 0..nodes.len() {
        let choices = if nodes[index].kind == "chance" {
            nodes[index]
                .children
                .iter()
                .filter_map(|child| {
                    nodes.get(*child).map(|node| ViewerBranch {
                        label: node.board.clone(),
                        child: *child,
                    })
                })
                .collect::<Vec<_>>()
        } else {
            nodes[index]
                .actions
                .iter()
                .filter_map(|action| {
                    action.child.map(|child| ViewerBranch {
                        label: action.label.clone(),
                        child,
                    })
                })
                .collect::<Vec<_>>()
        };
        nodes[index].choices = choices;
    }
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
        oop_weights: snapshot.oop_combos,
        ip_weights: snapshot.ip_combos,
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

fn load_viewer_from_db(path: &PathBuf) -> Result<ViewerBundle, String> {
    let connection = Connection::open(path).map_err(|error| {
        format!(
            "failed to open solution database {}: {error}",
            path.display()
        )
    })?;
    let metadata = load_db_metadata(&connection)?;
    let oop_weights = load_db_combo_weights(&connection, "oop")?;
    let ip_weights = load_db_combo_weights(&connection, "ip")?;
    let oop_combos = viewer_combos(&oop_weights);
    let ip_combos = viewer_combos(&ip_weights);
    let mut nodes = load_db_nodes(&connection)?;
    attach_db_edges(&connection, &mut nodes)?;
    let board = metadata_required(&metadata, "flop")?.to_string();
    let pot = metadata_u32(&metadata, "pot")?;
    let effective_stack = metadata_u32(&metadata, "effective_stack")?;
    let first_player = metadata_required(&metadata, "first_player")?.to_string();
    let iterations = metadata_u32(&metadata, "iterations")?;
    let storage_gib = metadata_f64(&metadata, "storage_gib").unwrap_or(0.0);
    let request = FlopTreeRequest {
        board: Board::from_str(&board)?,
        pot,
        effective_stack,
        oop_range: RangeSpec::new(oop_weights.clone())?,
        ip_range: RangeSpec::new(ip_weights.clone())?,
        first_player: parse_player(&first_player)?,
        action_abstraction: parse_action_abstraction_json(metadata_required(
            &metadata,
            "action_abstraction_json",
        )?)?,
    };
    let options = SolverOptions {
        iterations,
        threads: metadata_usize(&metadata, "threads").unwrap_or_else(default_thread_count),
        log_interval: 0,
        exploitability_interval: 0,
        target_exploitability_bb100: None,
        variant: parse_formatted_cfr_variant(
            metadata
                .get("variant")
                .map(String::as_str)
                .unwrap_or("dcfr-plus(alpha=1.5,gamma=2)"),
        )?,
        average_strategy: parse_average_strategy(
            metadata
                .get("average_strategy")
                .map(String::as_str)
                .unwrap_or("reach-weighted"),
        )?,
    };
    let solution = ViewerSolution {
        summary: ViewerSummary {
            board,
            pot_bb: pot as f32 / 100.0,
            effective_stack_bb: effective_stack as f32 / 100.0,
            first_player,
            iterations,
            solver_elapsed_ms: 0.0,
            storage_gib,
            exploitability_bb_per_100: None,
            nodes: nodes.len(),
            decision_states: metadata_usize(&metadata, "decision_states").unwrap_or(0),
            action_slots: metadata_usize(&metadata, "action_slots").unwrap_or(0),
            oop_combos: oop_combos.len(),
            ip_combos: ip_combos.len(),
        },
        oop_combos,
        ip_combos,
        nodes,
        oop_weights,
        ip_weights,
    };
    Ok(ViewerBundle {
        solution,
        solver: None,
        resolver: Some(ViewerDbResolver {
            db_path: path.clone(),
            request,
            options,
            enumerate_chance: metadata
                .get("enumerate_chance")
                .is_some_and(|value| value == "true"),
        }),
    })
}

fn load_db_metadata(connection: &Connection) -> Result<HashMap<String, String>, String> {
    let mut statement = connection
        .prepare("SELECT key, value FROM metadata")
        .map_err(|error| format!("failed to read metadata: {error}"))?;
    let rows = statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|error| format!("failed to query metadata: {error}"))?;
    let mut metadata = HashMap::new();
    for row in rows {
        let (key, value) = row.map_err(|error| format!("failed to load metadata row: {error}"))?;
        metadata.insert(key, value);
    }
    Ok(metadata)
}

fn load_db_combo_weights(
    connection: &Connection,
    player: &str,
) -> Result<Vec<ComboWeight>, String> {
    let mut statement = connection
        .prepare(
            r#"
            SELECT first_card, second_card, weight
            FROM combos
            WHERE player = ?1
            ORDER BY combo_index
            "#,
        )
        .map_err(|error| format!("failed to prepare {player} combo query: {error}"))?;
    let rows = statement
        .query_map(params![player], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f32>(2)?,
            ))
        })
        .map_err(|error| format!("failed to query {player} combos: {error}"))?;
    let mut combos = Vec::new();
    for row in rows {
        let (first, second, weight) =
            row.map_err(|error| format!("failed to load {player} combo row: {error}"))?;
        combos.push(ComboWeight {
            first: Card::from_str(&first)?,
            second: Card::from_str(&second)?,
            weight,
        });
    }
    Ok(combos)
}

fn load_db_nodes(connection: &Connection) -> Result<Vec<ViewerNode>, String> {
    let mut statement = connection
        .prepare(
            r#"
            SELECT id, public_node, street, board, pot_bb, player, kind, terminal_reason,
                   strategy_player, strategy_combos, strategy_actions, strategy_action_major_json
            FROM nodes
            ORDER BY id
            "#,
        )
        .map_err(|error| format!("failed to prepare node query: {error}"))?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, f32>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, Option<String>>(7)?,
                row.get::<_, Option<String>>(8)?,
                row.get::<_, Option<i64>>(9)?,
                row.get::<_, Option<i64>>(10)?,
                row.get::<_, Option<String>>(11)?,
            ))
        })
        .map_err(|error| format!("failed to query nodes: {error}"))?;
    let mut nodes = Vec::new();
    for row in rows {
        let (
            id,
            public_node,
            street,
            board,
            pot_bb,
            player,
            kind,
            terminal_reason,
            strategy_player,
            strategy_combos,
            strategy_actions,
            strategy_json,
        ) = row.map_err(|error| format!("failed to load node row: {error}"))?;
        let id = db_usize(id, "nodes.id")?;
        let public_node = db_usize(public_node, "nodes.public_node")?;
        let strategy_combos = strategy_combos
            .map(|value| db_usize(value, "nodes.strategy_combos"))
            .transpose()?;
        let strategy_actions = strategy_actions
            .map(|value| db_usize(value, "nodes.strategy_actions"))
            .transpose()?;
        if id != nodes.len() {
            return Err(format!(
                "solution database node ids must be contiguous; expected {}, got {id}",
                nodes.len()
            ));
        }
        let kind = if kind == "terminal" {
            terminal_reason
                .map(|reason| format!("terminal:{reason}"))
                .unwrap_or_else(|| "terminal".to_string())
        } else {
            kind
        };
        let strategy = match (
            strategy_player,
            strategy_combos,
            strategy_actions,
            strategy_json,
        ) {
            (Some(player), Some(combos), Some(actions), Some(json)) => Some(ViewerStrategy {
                player,
                combos,
                actions,
                action_major: serde_json::from_str(&json)
                    .map_err(|error| format!("failed to parse strategy for node {id}: {error}"))?,
            }),
            _ => None,
        };
        nodes.push(ViewerNode {
            id,
            public_node,
            board,
            street,
            pot_bb,
            player,
            kind,
            children: Vec::new(),
            actions: Vec::new(),
            choices: Vec::new(),
            strategy,
        });
    }
    Ok(nodes)
}

fn attach_db_edges(connection: &Connection, nodes: &mut [ViewerNode]) -> Result<(), String> {
    let mut statement = connection
        .prepare("SELECT parent_id, child_id, label FROM edges ORDER BY parent_id, edge_index")
        .map_err(|error| format!("failed to prepare edge query: {error}"))?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .map_err(|error| format!("failed to query edges: {error}"))?;
    for row in rows {
        let (parent, child, label) =
            row.map_err(|error| format!("failed to load edge row: {error}"))?;
        let parent = db_usize(parent, "edges.parent_id")?;
        let child = db_usize(child, "edges.child_id")?;
        let Some(node) = nodes.get_mut(parent) else {
            return Err(format!("edge references unknown parent node {parent}"));
        };
        node.children.push(child);
        node.choices.push(ViewerBranch { label, child });
    }

    let mut statement = connection
        .prepare("SELECT node_id, action_index, label, child_id FROM actions ORDER BY node_id, action_index")
        .map_err(|error| format!("failed to prepare action query: {error}"))?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<i64>>(3)?,
            ))
        })
        .map_err(|error| format!("failed to query actions: {error}"))?;
    for row in rows {
        let (node_id, index, label, child) =
            row.map_err(|error| format!("failed to load action row: {error}"))?;
        let node_id = db_usize(node_id, "actions.node_id")?;
        let index = db_usize(index, "actions.action_index")?;
        let child = child
            .map(|value| db_usize(value, "actions.child_id"))
            .transpose()?;
        let Some(node) = nodes.get_mut(node_id) else {
            return Err(format!("action references unknown node {node_id}"));
        };
        node.actions.push(ViewerAction {
            index,
            label,
            child,
        });
    }
    Ok(())
}

fn db_usize(value: i64, field: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{field} value {value} is negative or too large"))
}

fn metadata_required<'a>(
    metadata: &'a HashMap<String, String>,
    key: &str,
) -> Result<&'a str, String> {
    metadata
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("solution database missing metadata {key:?}"))
}

fn metadata_u32(metadata: &HashMap<String, String>, key: &str) -> Result<u32, String> {
    metadata_required(metadata, key)?
        .parse::<u32>()
        .map_err(|error| format!("metadata {key:?} is not a u32: {error}"))
}

fn metadata_usize(metadata: &HashMap<String, String>, key: &str) -> Option<usize> {
    metadata.get(key).and_then(|value| value.parse().ok())
}

fn metadata_f64(metadata: &HashMap<String, String>, key: &str) -> Option<f64> {
    metadata.get(key).and_then(|value| value.parse().ok())
}

#[derive(Clone)]
struct ViewerState {
    solution: Arc<ViewerSolution>,
    solver: Arc<Mutex<Option<NodeLocalCfrSolver>>>,
    resolver: Option<ViewerDbResolver>,
    resolved_strategy_cache: Arc<Mutex<HashMap<usize, ViewerStrategy>>>,
    equity_cache: Arc<Mutex<HashMap<usize, ViewerEquity>>>,
    strategy_ev_cache: Arc<Mutex<HashMap<usize, ViewerStrategyEv>>>,
    action_ev_cache: Arc<Mutex<HashMap<usize, ViewerActionEv>>>,
    reach_cache: Arc<Mutex<HashMap<usize, ViewerReach>>>,
}

fn serve_viewer(
    viewer: ViewerBundle,
    host: String,
    port: u16,
    assets: PathBuf,
) -> Result<(), String> {
    let state = ViewerState {
        solution: Arc::new(viewer.solution),
        solver: Arc::new(Mutex::new(viewer.solver)),
        resolver: viewer.resolver,
        resolved_strategy_cache: Arc::new(Mutex::new(HashMap::new())),
        equity_cache: Arc::new(Mutex::new(HashMap::new())),
        strategy_ev_cache: Arc::new(Mutex::new(HashMap::new())),
        action_ev_cache: Arc::new(Mutex::new(HashMap::new())),
        reach_cache: Arc::new(Mutex::new(HashMap::new())),
    };
    let api = Router::new()
        .route("/combos", get(api_combos))
        .route("/equity/{id}", get(api_equity))
        .route("/reach/{id}", get(api_reach))
        .route("/strategy-ev/{id}", get(api_strategy_ev))
        .route("/action-ev/{id}", get(api_action_ev))
        .route("/summary", get(api_summary))
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

async fn api_combos(State(state): State<ViewerState>) -> Json<ViewerCombos> {
    Json(ViewerCombos {
        oop: state.solution.oop_combos.clone(),
        ip: state.solution.ip_combos.clone(),
    })
}

async fn api_equity(Path(id): Path<usize>, State(state): State<ViewerState>) -> impl IntoResponse {
    if let Some(cached) = state
        .equity_cache
        .lock()
        .expect("equity cache poisoned")
        .get(&id)
    {
        return Json(cached.clone()).into_response();
    }
    let Some(node) = state.solution.nodes.get(id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("unknown node {id}") })),
        )
            .into_response();
    };
    let board = match Board::from_str(&node.board) {
        Ok(board) => board,
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": error })),
            )
                .into_response();
        }
    };
    match raw_range_equity(
        &board,
        node.pot_bb,
        &state.solution.oop_weights,
        &state.solution.ip_weights,
    ) {
        Ok(equity) => {
            state
                .equity_cache
                .lock()
                .expect("equity cache poisoned")
                .insert(id, equity.clone());
            Json(equity).into_response()
        }
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": error })),
        )
            .into_response(),
    }
}

async fn api_reach(Path(id): Path<usize>, State(state): State<ViewerState>) -> impl IntoResponse {
    if let Some(cached) = state
        .reach_cache
        .lock()
        .expect("reach cache poisoned")
        .get(&id)
    {
        return Json(cached.clone()).into_response();
    }
    if state.resolver.is_some()
        && state
            .solver
            .lock()
            .expect("viewer solver poisoned")
            .is_none()
    {
        match db_viewer_reach_at_node(&state, id) {
            Ok(value) => {
                state
                    .reach_cache
                    .lock()
                    .expect("reach cache poisoned")
                    .insert(id, value.clone());
                return Json(value).into_response();
            }
            Err(error) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": error })),
                )
                    .into_response();
            }
        }
    }
    let result = ensure_viewer_solver(&state).and_then(|mut solver| {
        solver
            .as_mut()
            .expect("viewer solver must be initialized")
            .display_reach_at_node(id)
    });
    match result {
        Ok((oop, ip)) => {
            let value = ViewerReach { oop, ip };
            state
                .reach_cache
                .lock()
                .expect("reach cache poisoned")
                .insert(id, value.clone());
            Json(value).into_response()
        }
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": error })),
        )
            .into_response(),
    }
}

async fn api_strategy_ev(
    Path(id): Path<usize>,
    State(state): State<ViewerState>,
) -> impl IntoResponse {
    if let Some(cached) = state
        .strategy_ev_cache
        .lock()
        .expect("strategy EV cache poisoned")
        .get(&id)
    {
        return Json(cached.clone()).into_response();
    }
    if state.resolver.is_some()
        && state
            .solver
            .lock()
            .expect("viewer solver poisoned")
            .is_none()
    {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": "strategy EV is not stored in this database" })),
        )
            .into_response();
    }
    let result = ensure_viewer_solver(&state).and_then(|mut solver| {
        solver
            .as_mut()
            .expect("viewer solver must be initialized")
            .strategy_ev_at_node(id)
    });
    match result {
        Ok(ev) => {
            let value = ViewerStrategyEv {
                board: ev.board.to_string(),
                pot_bb: ev.pot as f32 / 100.0,
                oop_ev_bb: ev.oop_value / 100.0,
                ip_ev_bb: ev.ip_value / 100.0,
                oop_weight: ev.oop_weight,
                ip_weight: ev.ip_weight,
                terminal_evals: ev.terminal_evals,
            };
            state
                .strategy_ev_cache
                .lock()
                .expect("strategy EV cache poisoned")
                .insert(id, value.clone());
            Json(value).into_response()
        }
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": error })),
        )
            .into_response(),
    }
}

async fn api_action_ev(
    Path(id): Path<usize>,
    State(state): State<ViewerState>,
) -> impl IntoResponse {
    if let Some(cached) = state
        .action_ev_cache
        .lock()
        .expect("action EV cache poisoned")
        .get(&id)
    {
        return Json(cached.clone()).into_response();
    }
    if state.resolver.is_some()
        && state
            .solver
            .lock()
            .expect("viewer solver poisoned")
            .is_none()
    {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": "action EV is not stored in this database" })),
        )
            .into_response();
    }
    let result = ensure_viewer_solver(&state).and_then(|mut solver| {
        solver
            .as_mut()
            .expect("viewer solver must be initialized")
            .action_ev_at_node(id)
    });
    match result {
        Ok(ev) => {
            let value = ViewerActionEv {
                board: ev.board.to_string(),
                pot_bb: ev.pot as f32 / 100.0,
                player: format_player(ev.player),
                combos: ev.combos,
                actions: ev.actions,
                action_major_bb: ev
                    .action_major
                    .into_iter()
                    .map(|value| value / 100.0)
                    .collect(),
                terminal_evals: ev.terminal_evals,
            };
            state
                .action_ev_cache
                .lock()
                .expect("action EV cache poisoned")
                .insert(id, value.clone());
            Json(value).into_response()
        }
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": error })),
        )
            .into_response(),
    }
}

async fn api_node(Path(id): Path<usize>, State(state): State<ViewerState>) -> impl IntoResponse {
    let Some(source_node) = state.solution.nodes.get(id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("unknown node {id}") })),
        )
            .into_response();
    };
    let mut node = source_node.clone();
    if node.strategy.is_none() && node.kind == "decision" {
        if let Some(strategy) = state
            .resolved_strategy_cache
            .lock()
            .expect("resolved strategy cache poisoned")
            .get(&id)
            .cloned()
        {
            node.strategy = Some(strategy);
        } else {
            match resolve_viewer_node_strategy(&state, id) {
                Ok(Some(strategy)) => {
                    node.strategy = Some(strategy);
                }
                Ok(None) => {}
                Err(error) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({ "error": error })),
                    )
                        .into_response();
                }
            }
        }
    }
    Json(node).into_response()
}

type ViewerSolverGuard<'a> = std::sync::MutexGuard<'a, Option<NodeLocalCfrSolver>>;

fn ensure_viewer_solver(state: &ViewerState) -> Result<ViewerSolverGuard<'_>, String> {
    let mut guard = state.solver.lock().expect("viewer solver poisoned");
    if guard.is_none() {
        let resolver = state
            .resolver
            .as_ref()
            .ok_or_else(|| "viewer solver is not available for this database".to_string())?;
        info!(
            db = %resolver.db_path.display(),
            iterations = resolver.options.iterations,
            "viewer_lazy_solver_start"
        );
        let tree = build_tree(resolver.request.clone(), resolver.enumerate_chance)?;
        let solver = solve_flop(
            tree,
            resolver.request.oop_range.clone(),
            resolver.request.ip_range.clone(),
            resolver.options.iterations,
            resolver.options.threads,
            0,
            0,
            None,
            RealCfrConfig {
                iterations: resolver.options.iterations,
                variant: resolver.options.variant,
                average_strategy: resolver.options.average_strategy,
            },
        )?;
        *guard = Some(solver);
        info!(db = %resolver.db_path.display(), "viewer_lazy_solver_finish");
    }
    Ok(guard)
}

fn resolve_viewer_node_strategy(
    state: &ViewerState,
    id: usize,
) -> Result<Option<ViewerStrategy>, String> {
    let value = if let Some(resolver) = &state.resolver {
        let Some(node) = state.solution.nodes.get(id) else {
            return Ok(None);
        };
        solve_viewer_subtree_root_strategy(resolver, node.public_node)?
    } else {
        let mut solver = ensure_viewer_solver(state)?;
        let snapshot = solver
            .as_mut()
            .expect("viewer solver must be initialized")
            .solution_snapshot_until_street(Some(Street::River));
        let Some(node) = snapshot.nodes.into_iter().find(|node| node.id == id) else {
            return Ok(None);
        };
        let Some(strategy) = node.strategy else {
            return Ok(None);
        };
        Some(ViewerStrategy {
            player: format_player(strategy.player),
            combos: strategy.combos,
            actions: strategy.actions,
            action_major: strategy.action_major,
        })
    };
    let Some(value) = value else {
        return Ok(None);
    };
    state
        .resolved_strategy_cache
        .lock()
        .expect("resolved strategy cache poisoned")
        .insert(id, value.clone());
    Ok(Some(value))
}

fn db_viewer_reach_at_node(state: &ViewerState, id: usize) -> Result<ViewerReach, String> {
    let mut path = Vec::new();
    if !find_viewer_path(&state.solution.nodes, 0, id, &mut path) {
        return Err(format!("node {id} is not reachable from root"));
    }
    let mut oop = state
        .solution
        .oop_weights
        .iter()
        .map(|combo| combo.weight)
        .collect::<Vec<_>>();
    let mut ip = state
        .solution
        .ip_weights
        .iter()
        .map(|combo| combo.weight)
        .collect::<Vec<_>>();
    if let Some(root) = state.solution.nodes.first() {
        let board = Board::from_str(&root.board)?;
        zero_viewer_board_conflicts(&mut oop, &state.solution.oop_weights, &board);
        zero_viewer_board_conflicts(&mut ip, &state.solution.ip_weights, &board);
    }
    for pair in path.windows(2) {
        let parent_id = pair[0];
        let child_id = pair[1];
        let parent = &state.solution.nodes[parent_id];
        let child = &state.solution.nodes[child_id];
        if parent.kind == "chance" {
            let board = Board::from_str(&child.board)?;
            zero_viewer_board_conflicts(&mut oop, &state.solution.oop_weights, &board);
            zero_viewer_board_conflicts(&mut ip, &state.solution.ip_weights, &board);
            continue;
        }
        if parent.kind != "decision" {
            continue;
        }
        let Some(action_index) = parent
            .actions
            .iter()
            .find(|action| action.child == Some(child_id))
            .map(|action| action.index)
        else {
            return Err(format!(
                "node {parent_id} has no action leading to child {child_id}"
            ));
        };
        let strategy = parent.strategy.clone().or_else(|| {
            state
                .resolved_strategy_cache
                .lock()
                .expect("resolved strategy cache poisoned")
                .get(&parent_id)
                .cloned()
        });
        let Some(strategy) = strategy else {
            return Err(format!(
                "node {parent_id} strategy is not available for DB reach propagation"
            ));
        };
        let reach = if strategy.player == "oop" {
            &mut oop
        } else {
            &mut ip
        };
        for (combo_index, value) in reach.iter_mut().enumerate() {
            let frequency = strategy
                .action_major
                .get(action_index * strategy.combos + combo_index)
                .copied()
                .unwrap_or(0.0);
            *value *= frequency;
        }
    }
    Ok(ViewerReach { oop, ip })
}

fn find_viewer_path(
    nodes: &[ViewerNode],
    current: usize,
    target: usize,
    path: &mut Vec<usize>,
) -> bool {
    path.push(current);
    if current == target {
        return true;
    }
    let Some(node) = nodes.get(current) else {
        path.pop();
        return false;
    };
    for child in &node.children {
        if find_viewer_path(nodes, *child, target, path) {
            return true;
        }
    }
    path.pop();
    false
}

fn zero_viewer_board_conflicts(reach: &mut [f32], combos: &[ComboWeight], board: &Board) {
    for (value, combo) in reach.iter_mut().zip(combos) {
        if board.contains(combo.first) || board.contains(combo.second) {
            *value = 0.0;
        }
    }
}

fn solve_viewer_subtree_root_strategy(
    resolver: &ViewerDbResolver,
    root_public_node: usize,
) -> Result<Option<ViewerStrategy>, String> {
    info!(
        db = %resolver.db_path.display(),
        root_public_node,
        iterations = resolver.options.iterations,
        "viewer_lazy_subtree_solver_start"
    );
    let full_tree = build_tree(resolver.request.clone(), resolver.enumerate_chance)?;
    let subtree = public_subtree_from(&full_tree, root_public_node)?;
    let mut solver = NodeLocalCfrSolver::new(
        subtree,
        resolver.request.oop_range.clone(),
        resolver.request.ip_range.clone(),
    )?;
    solver.run_with_progress(
        RealCfrConfig {
            iterations: resolver.options.iterations,
            variant: resolver.options.variant,
            average_strategy: resolver.options.average_strategy,
        },
        |_| {},
    )?;
    let snapshot = solver.solution_snapshot_until_street(Some(Street::River));
    let oop_combos = snapshot.oop_combos.clone();
    let ip_combos = snapshot.ip_combos.clone();
    let strategy = snapshot
        .nodes
        .into_iter()
        .next()
        .and_then(|node| node.strategy);
    info!(
        db = %resolver.db_path.display(),
        root_public_node,
        "viewer_lazy_subtree_solver_finish"
    );
    strategy
        .map(|strategy| expand_subtree_strategy(resolver, strategy, &oop_combos, &ip_combos))
        .transpose()
}

fn expand_subtree_strategy(
    resolver: &ViewerDbResolver,
    strategy: pokedr_core::NodeLocalStrategySnapshot,
    oop_subtree_combos: &[ComboWeight],
    ip_subtree_combos: &[ComboWeight],
) -> Result<ViewerStrategy, String> {
    let (source_combos, target_combos) = match strategy.player {
        Player::Oop => (oop_subtree_combos, resolver.request.oop_range.combos()),
        Player::Ip => (ip_subtree_combos, resolver.request.ip_range.combos()),
    };
    let mut target_by_combo = HashMap::with_capacity(target_combos.len());
    for (index, combo) in target_combos.iter().enumerate() {
        target_by_combo.insert(combo_key(combo), index);
    }
    let mut source_to_target = Vec::with_capacity(source_combos.len());
    for combo in source_combos {
        let target = target_by_combo
            .get(&combo_key(combo))
            .copied()
            .ok_or_else(|| {
                format!(
                    "subtree combo {}{} is missing from viewer range",
                    combo.first, combo.second
                )
            })?;
        source_to_target.push(target);
    }
    let mut action_major = vec![0.0; strategy.actions * target_combos.len()];
    for action in 0..strategy.actions {
        let source_base = action * strategy.combos;
        let target_base = action * target_combos.len();
        for (source_index, target_index) in source_to_target.iter().copied().enumerate() {
            action_major[target_base + target_index] =
                strategy.action_major[source_base + source_index];
        }
    }
    Ok(ViewerStrategy {
        player: format_player(strategy.player),
        combos: target_combos.len(),
        actions: strategy.actions,
        action_major,
    })
}

fn combo_key(combo: &ComboWeight) -> (u8, u8) {
    let first = combo.first.index() as u8;
    let second = combo.second.index() as u8;
    if first <= second {
        (first, second)
    } else {
        (second, first)
    }
}

fn public_subtree_from(tree: &PublicTree, root: usize) -> Result<PublicTree, String> {
    let root_node = tree
        .nodes
        .get(root)
        .ok_or_else(|| format!("public node {root} is out of bounds"))?;
    let mut old_to_new = HashMap::new();
    let mut order = Vec::new();
    collect_public_subtree_order(tree, root, &mut old_to_new, &mut order)?;
    let mut nodes = Vec::with_capacity(order.len());
    for (new_id, old_id) in order.iter().copied().enumerate() {
        let old_node = &tree.nodes[old_id];
        let mut node = old_node.clone();
        node.id = new_id;
        node.children = old_node
            .children
            .iter()
            .map(|child| {
                old_to_new
                    .get(child)
                    .copied()
                    .ok_or_else(|| format!("subtree child {child} was not collected"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        nodes.push(node);
    }
    let mut spot = tree.spot.clone();
    spot.board = root_node.state.board.clone();
    spot.pot = root_node.state.pot;
    spot.first_player = root_node.state.player;
    Ok(PublicTree { spot, nodes })
}

fn collect_public_subtree_order(
    tree: &PublicTree,
    node: usize,
    old_to_new: &mut HashMap<usize, usize>,
    order: &mut Vec<usize>,
) -> Result<(), String> {
    if old_to_new.contains_key(&node) {
        return Ok(());
    }
    let new_id = order.len();
    old_to_new.insert(node, new_id);
    order.push(node);
    let public = tree
        .nodes
        .get(node)
        .ok_or_else(|| format!("public node {node} is out of bounds"))?;
    for child in &public.children {
        collect_public_subtree_order(tree, *child, old_to_new, order)?;
    }
    Ok(())
}

fn raw_range_equity(
    board: &Board,
    pot_bb: f32,
    oop_combos: &[ComboWeight],
    ip_combos: &[ComboWeight],
) -> Result<ViewerEquity, String> {
    let terminal_boards = viewer_terminal_boards(board)?;
    let mut oop_win_weight = 0.0f64;
    let mut ip_win_weight = 0.0f64;
    let mut tie_weight = 0.0f64;

    for terminal_board in &terminal_boards {
        let prepared = PreparedTerminalBoard::new(terminal_board)?;
        for oop in oop_combos {
            let Some(oop_index) = prepared.combo_index(oop.first, oop.second) else {
                continue;
            };
            for ip in ip_combos {
                if combos_collide(oop, ip) {
                    continue;
                }
                let Some(ip_index) = prepared.combo_index(ip.first, ip.second) else {
                    continue;
                };
                let weight = (oop.weight as f64) * (ip.weight as f64);
                if weight == 0.0 {
                    continue;
                }
                let oop_strength = prepared.strength(oop_index);
                let ip_strength = prepared.strength(ip_index);
                if oop_strength > ip_strength {
                    oop_win_weight += weight;
                } else if oop_strength < ip_strength {
                    ip_win_weight += weight;
                } else {
                    tie_weight += weight;
                }
            }
        }
    }

    let pair_weight = oop_win_weight + ip_win_weight + tie_weight;
    let oop_equity = if pair_weight > 0.0 {
        ((oop_win_weight + tie_weight * 0.5) / pair_weight) as f32
    } else {
        0.0
    };
    Ok(ViewerEquity {
        board: board.to_string(),
        pot_bb,
        terminal_boards: terminal_boards.len(),
        pair_weight,
        oop_equity,
        ip_equity: 1.0 - oop_equity,
        oop_win_weight,
        ip_win_weight,
        tie_weight,
    })
}

fn viewer_terminal_boards(board: &Board) -> Result<Vec<Board>, String> {
    match board.cards().len() {
        5 => Ok(vec![board.clone()]),
        4 => {
            let deck = board.remaining_deck();
            let mut boards = Vec::with_capacity(deck.len());
            for river in deck {
                boards.push(board.push(river)?);
            }
            Ok(boards)
        }
        3 => {
            let deck = board.remaining_deck();
            let mut boards = Vec::with_capacity(deck.len() * (deck.len() - 1) / 2);
            for turn in 0..deck.len() {
                for river in turn + 1..deck.len() {
                    boards.push(board.push(deck[turn])?.push(deck[river])?);
                }
            }
            Ok(boards)
        }
        other => Err(format!("equity board has invalid card count {other}")),
    }
}

fn combos_collide(left: &ComboWeight, right: &ComboWeight) -> bool {
    left.first == right.first
        || left.first == right.second
        || left.second == right.first
        || left.second == right.second
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
    spot: &SpotOptions,
    action_abstraction: ActionAbstraction,
) -> Result<FlopTreeRequest, String> {
    Ok(FlopTreeRequest {
        board: Board::from_str(&spot.flop)?,
        pot: spot.pot,
        effective_stack: spot.effective_stack,
        oop_range: RangeSpec::from_str(&spot.oop_range)?,
        ip_range: RangeSpec::from_str(&spot.ip_range)?,
        first_player: parse_player(&spot.first_player)?,
        action_abstraction,
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

fn apply_tree_config_overrides(
    action: &mut ActionAbstraction,
    config: &TreeConfig,
) -> Result<(), String> {
    if let Some(min_bet) = config.min_bet {
        if min_bet == 0 {
            return Err("tree.min_bet must be greater than zero".to_string());
        }
        action.min_bet = min_bet;
    }
    if let Some(values) = &config.flop_first_bets {
        action.flop.first_bet_sizes = parse_bet_size_list("tree.flop_first_bets", values)?;
    }
    if let Some(values) = &config.flop_donk_bets {
        action.flop.donk_bet_sizes = parse_bet_size_list("tree.flop_donk_bets", values)?;
    }
    if let Some(values) = &config.turn_first_bets {
        action.turn.first_bet_sizes = parse_bet_size_list("tree.turn_first_bets", values)?;
    }
    if let Some(values) = &config.turn_donk_bets {
        action.turn.donk_bet_sizes = parse_bet_size_list("tree.turn_donk_bets", values)?;
    }
    if let Some(values) = &config.river_first_bets {
        action.river.first_bet_sizes = parse_bet_size_list("tree.river_first_bets", values)?;
    }
    if let Some(values) = &config.river_donk_bets {
        action.river.donk_bet_sizes = parse_bet_size_list("tree.river_donk_bets", values)?;
    }
    if let Some(raise_multiplier) = config.raise_multiplier {
        if !raise_multiplier.is_finite() || raise_multiplier <= 1.0 {
            return Err("tree.raise_multiplier must be finite and greater than 1.0".to_string());
        }
        action.raise.raise_multiplier = raise_multiplier;
    }
    if let Some(values) = &config.raise_sizes {
        action.raise.raise_sizes = parse_raise_size_list("tree.raise_sizes", values)?;
    }
    if let Some(value) = config.max_raises_per_street {
        action.raise.max_raises_per_street = value;
    }
    if let Some(value) = config.shove_spr_threshold {
        action.raise.shove_spr_threshold =
            validate_non_negative_finite("tree.shove_spr_threshold", value)?;
    }
    if let Some(value) = config.shove_commit_fraction {
        action.raise.shove_commit_fraction =
            validate_non_negative_finite("tree.shove_commit_fraction", value)?;
    }
    if let Some(value) = config.add_all_in_threshold {
        action.raise.add_all_in_threshold =
            validate_non_negative_finite("tree.add_all_in_threshold", value)?;
    }
    if let Some(value) = config.force_all_in_threshold {
        action.raise.force_all_in_threshold =
            validate_non_negative_finite("tree.force_all_in_threshold", value)?;
    }
    if let Some(value) = config.merging_threshold {
        action.raise.merging_threshold =
            validate_non_negative_finite("tree.merging_threshold", value)?;
    }
    Ok(())
}

fn parse_bet_size_list(field: &str, values: &[String]) -> Result<Vec<BetSizeSpec>, String> {
    values
        .iter()
        .map(|value| parse_bet_size(field, value))
        .collect()
}

fn parse_raise_size_list(field: &str, values: &[String]) -> Result<Vec<RaiseSizeSpec>, String> {
    values
        .iter()
        .map(|value| parse_raise_size(field, value))
        .collect()
}

fn parse_bet_size(field: &str, value: &str) -> Result<BetSizeSpec, String> {
    let normalized = value.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "allin" | "all-in" | "jam" | "shove" => Ok(BetSizeSpec::AllIn),
        "geo" | "geometric" => Ok(BetSizeSpec::Geometric {
            streets: 0,
            max_pot_fraction: f32::INFINITY,
        }),
        _ if normalized.starts_with("geo:") => {
            let (streets, max_pot_fraction) = parse_geometric_size(field, &normalized)?;
            Ok(BetSizeSpec::Geometric {
                streets,
                max_pot_fraction,
            })
        }
        _ => parse_pot_fraction(field, value).map(BetSizeSpec::PotFraction),
    }
}

fn parse_raise_size(field: &str, value: &str) -> Result<RaiseSizeSpec, String> {
    let normalized = value.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "allin" | "all-in" | "jam" | "shove" => Ok(RaiseSizeSpec::AllIn),
        "geo" | "geometric" => Ok(RaiseSizeSpec::Geometric {
            streets: 0,
            max_pot_fraction: f32::INFINITY,
        }),
        _ if normalized.starts_with("geo:") => {
            let (streets, max_pot_fraction) = parse_geometric_size(field, &normalized)?;
            Ok(RaiseSizeSpec::Geometric {
                streets,
                max_pot_fraction,
            })
        }
        _ if normalized.ends_with('x') => {
            let multiplier = parse_positive_f32(field, normalized.trim_end_matches('x'))?;
            if multiplier <= 1.0 {
                return Err(format!(
                    "{field} multiplier {value:?} must be greater than 1.0"
                ));
            }
            Ok(RaiseSizeSpec::PreviousBetMultiplier(multiplier))
        }
        _ => parse_pot_fraction(field, value).map(RaiseSizeSpec::PotFraction),
    }
}

fn parse_geometric_size(field: &str, value: &str) -> Result<(u8, f32), String> {
    let mut parts = value.split(':');
    let _geo = parts.next();
    let streets = parts
        .next()
        .ok_or_else(|| format!("{field} geometric size {value:?} missing street count"))?
        .parse::<u8>()
        .map_err(|_| format!("{field} geometric size {value:?} has invalid street count"))?;
    let raw_max_pot_fraction = parts
        .next()
        .ok_or_else(|| format!("{field} geometric size {value:?} missing max pot fraction"))?;
    let max_pot_fraction = if raw_max_pot_fraction == "inf" || raw_max_pot_fraction == "infinity" {
        f32::INFINITY
    } else {
        raw_max_pot_fraction
            .parse::<f32>()
            .map_err(|_| format!("{field} geometric size {value:?} has invalid max pot fraction"))?
    };
    if parts.next().is_some() || max_pot_fraction.is_nan() || max_pot_fraction <= 0.0 {
        return Err(format!("{field} geometric size {value:?} is invalid"));
    }
    Ok((streets, max_pot_fraction))
}

fn parse_action_abstraction_json(value: &str) -> Result<ActionAbstraction, String> {
    let root: serde_json::Value = serde_json::from_str(value)
        .map_err(|error| format!("failed to parse action abstraction metadata: {error}"))?;
    let min_bet = json_u32(&root, "min_bet")?;
    let flop = parse_street_template_json(&root, "flop")?;
    let turn = parse_street_template_json(&root, "turn")?;
    let river = parse_street_template_json(&root, "river")?;
    let raise = root
        .get("raise")
        .ok_or_else(|| "action abstraction missing raise section".to_string())?;
    Ok(ActionAbstraction {
        min_bet,
        flop,
        turn,
        river,
        raise: RaisePolicy {
            raise_multiplier: json_f32(raise, "raise_multiplier")?,
            raise_sizes: json_string_array(raise, "raise_sizes")?
                .iter()
                .map(|value| {
                    parse_raise_size("metadata.action_abstraction_json.raise_sizes", value)
                })
                .collect::<Result<Vec<_>, _>>()?,
            max_raises_per_street: json_u32(raise, "max_raises_per_street")? as u8,
            shove_spr_threshold: json_f32(raise, "shove_spr_threshold")?,
            shove_commit_fraction: json_f32(raise, "shove_commit_fraction")?,
            add_all_in_threshold: json_f32(raise, "add_all_in_threshold")?,
            force_all_in_threshold: json_f32(raise, "force_all_in_threshold")?,
            merging_threshold: json_f32(raise, "merging_threshold")?,
        },
    })
}

fn parse_street_template_json(
    root: &serde_json::Value,
    key: &str,
) -> Result<StreetTemplate, String> {
    let value = root
        .get(key)
        .ok_or_else(|| format!("action abstraction missing {key} section"))?;
    Ok(StreetTemplate {
        first_bet_sizes: json_string_array(value, "first_bet_sizes")?
            .iter()
            .map(|value| parse_bet_size("metadata.action_abstraction_json.first_bet_sizes", value))
            .collect::<Result<Vec<_>, _>>()?,
        donk_bet_sizes: json_string_array(value, "donk_bet_sizes")?
            .iter()
            .map(|value| parse_bet_size("metadata.action_abstraction_json.donk_bet_sizes", value))
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn json_string_array(value: &serde_json::Value, key: &str) -> Result<Vec<String>, String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("action abstraction field {key:?} must be an array"))?
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("action abstraction field {key:?} must contain strings"))
        })
        .collect()
}

fn json_u32(value: &serde_json::Value, key: &str) -> Result<u32, String> {
    let raw = value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("action abstraction field {key:?} must be an unsigned integer"))?;
    u32::try_from(raw).map_err(|_| format!("action abstraction field {key:?} exceeds u32"))
}

fn json_f32(value: &serde_json::Value, key: &str) -> Result<f32, String> {
    let raw = value
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| format!("action abstraction field {key:?} must be a number"))?;
    if !raw.is_finite() {
        return Err(format!("action abstraction field {key:?} must be finite"));
    }
    Ok(raw as f32)
}

fn parse_pot_fraction(field: &str, value: &str) -> Result<f32, String> {
    let normalized = value.trim().to_ascii_lowercase();
    let (number, divisor) = if let Some(percent) = normalized.strip_suffix('%') {
        (percent, 100.0)
    } else if let Some(pot) = normalized.strip_suffix('p') {
        (pot, 1.0)
    } else {
        (normalized.as_str(), 1.0)
    };
    let fraction = parse_positive_f32(field, number)? / divisor;
    if fraction <= 0.0 {
        return Err(format!(
            "{field} pot fraction {value:?} must be greater than zero"
        ));
    }
    Ok(fraction)
}

fn parse_positive_f32(field: &str, value: &str) -> Result<f32, String> {
    let parsed = value
        .trim()
        .parse::<f32>()
        .map_err(|_| format!("{field} value {value:?} is not a number"))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(format!(
            "{field} value {value:?} must be finite and positive"
        ));
    }
    Ok(parsed)
}

fn validate_non_negative_finite(field: &str, value: f32) -> Result<f32, String> {
    if !value.is_finite() || value < 0.0 {
        Err(format!("{field} must be finite and non-negative"))
    } else {
        Ok(value)
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

fn parse_street(value: &str) -> Result<Street, String> {
    match value.to_ascii_lowercase().as_str() {
        "flop" => Ok(Street::Flop),
        "turn" => Ok(Street::Turn),
        "river" => Ok(Street::River),
        _ => Err(format!(
            "invalid street {value:?}; expected flop, turn, or river"
        )),
    }
}

fn format_street(street: Street) -> &'static str {
    match street {
        Street::Flop => "flop",
        Street::Turn => "turn",
        Street::River => "river",
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

fn parse_formatted_cfr_variant(value: &str) -> Result<RealCfrVariant, String> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized == "cfr-plus" || normalized == "cfr+" {
        return Ok(RealCfrVariant::CfrPlus);
    }
    if normalized.starts_with("dcfr-plus(") {
        let alpha = parse_named_f32(&normalized, "alpha").unwrap_or(1.5);
        let gamma = parse_named_f32(&normalized, "gamma").unwrap_or(2.0);
        return parse_cfr_variant("dcfr-plus", alpha, 0.0, gamma);
    }
    if normalized.starts_with("dcfr(") {
        let alpha = parse_named_f32(&normalized, "alpha").unwrap_or(1.5);
        let beta = parse_named_f32(&normalized, "beta").unwrap_or(0.0);
        let gamma = parse_named_f32(&normalized, "gamma").unwrap_or(2.0);
        return parse_cfr_variant("dcfr", alpha, beta, gamma);
    }
    parse_cfr_variant(&normalized, 1.5, 0.0, 2.0)
}

fn parse_named_f32(value: &str, name: &str) -> Option<f32> {
    value
        .trim_end_matches(')')
        .split(['(', ','])
        .find_map(|part| {
            let (key, raw) = part.split_once('=')?;
            if key.trim() == name {
                raw.trim().parse::<f32>().ok()
            } else {
                None
            }
        })
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
