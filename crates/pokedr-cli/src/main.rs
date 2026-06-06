use clap::{Parser, Subcommand};
use pokedr_agent::{FlopTreeRequest, build_flop_tree};
use pokedr_core::{
    ActionKind, Board, ChanceExpansion, Player, PublicNodeKind, RangeSpec, Street, TreeTemplate,
};
use std::str::FromStr;

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
                let schematic = estimate_schematic_work(&tree);
                println!(
                    "schematic_exact flop_decisions={} turn_decisions={} river_decisions={} flop_action_slots={} turn_action_slots={} river_action_slots={} total_action_slots={} terminal_pair_evals_per_iter={} memory_regret_strategy_f32_mb={:.1} memory_regret_f32_strategy_f16_mb={:.1} flop_turn_only_f32_mb={:.1}",
                    schematic.flop_decisions,
                    schematic.turn_decisions,
                    schematic.river_decisions,
                    schematic.flop_action_slots,
                    schematic.turn_action_slots,
                    schematic.river_action_slots,
                    schematic.total_action_slots(),
                    schematic.terminal_pair_evals_per_iter,
                    schematic.memory_regret_strategy_f32_mb(),
                    schematic.memory_regret_f32_strategy_f16_mb(),
                    schematic.flop_turn_only_f32_mb(),
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
    }
    Ok(())
}

struct WorkEstimate {
    private_infosets: u128,
    action_slots: u128,
    private_pairs: u128,
    terminal_pair_visits: u128,
    memory_regret_strategy_f32_mb: f64,
}

#[derive(Default)]
struct SchematicEstimate {
    flop_decisions: u128,
    turn_decisions: u128,
    river_decisions: u128,
    flop_action_slots: u128,
    turn_action_slots: u128,
    river_action_slots: u128,
    terminal_pair_evals_per_iter: u128,
}

impl SchematicEstimate {
    fn total_action_slots(&self) -> u128 {
        self.flop_action_slots + self.turn_action_slots + self.river_action_slots
    }

    fn memory_regret_strategy_f32_mb(&self) -> f64 {
        self.total_action_slots() as f64 * 2.0 * 4.0 / (1024.0 * 1024.0)
    }

    fn memory_regret_f32_strategy_f16_mb(&self) -> f64 {
        self.total_action_slots() as f64 * (4.0 + 2.0) / (1024.0 * 1024.0)
    }

    fn flop_turn_only_f32_mb(&self) -> f64 {
        (self.flop_action_slots + self.turn_action_slots) as f64 * 2.0 * 4.0 / (1024.0 * 1024.0)
    }
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

fn estimate_schematic_work(tree: &pokedr_core::PublicTree) -> SchematicEstimate {
    let mut estimate = SchematicEstimate::default();
    for node in &tree.nodes {
        let PublicNodeKind::Decision { actions, .. } = &node.kind else {
            if matches!(node.kind, PublicNodeKind::Terminal { .. }) {
                let board_count = match node.state.street {
                    Street::Flop => 1u128,
                    Street::Turn => 49u128,
                    Street::River => 49u128 * 48u128,
                };
                let live_combos = match node.state.street {
                    Street::Flop => choose2(49),
                    Street::Turn => choose2(48),
                    Street::River => choose2(47),
                };
                estimate.terminal_pair_evals_per_iter += board_count * live_combos * live_combos;
            }
            continue;
        };
        let board_count = match node.state.street {
            Street::Flop => 1u128,
            Street::Turn => 49u128,
            Street::River => 49u128 * 48u128,
        };
        let live_combos = match node.state.street {
            Street::Flop => choose2(49),
            Street::Turn => choose2(48),
            Street::River => choose2(47),
        };
        let slots = board_count * live_combos * actions.len() as u128;
        match node.state.street {
            Street::Flop => {
                estimate.flop_decisions += 1;
                estimate.flop_action_slots += slots;
            }
            Street::Turn => {
                estimate.turn_decisions += 1;
                estimate.turn_action_slots += slots;
            }
            Street::River => {
                estimate.river_decisions += 1;
                estimate.river_action_slots += slots;
            }
        }
    }
    estimate
}

fn choose2(count: u128) -> u128 {
    count * (count - 1) / 2
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
