use crate::tree::{Player, PublicNodeKind, PublicTree, Street, TerminalReason};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CfrStorageConfig {
    pub chunk_target_bytes: u128,
    pub regret_bytes: u128,
    pub strategy_sum_bytes: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CfrWorkPlan {
    pub street: [StreetWorkPlan; 3],
    pub terminals: TerminalWorkPlan,
    pub total_action_slots: u128,
    pub total_storage_bytes: u128,
    pub total_chunks: u128,
    pub max_chunk_bytes: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StreetWorkPlan {
    pub decisions: u128,
    pub action_slots: u128,
    pub storage_bytes: u128,
    pub chunks: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TerminalWorkPlan {
    pub fold_terminals: u128,
    pub showdown_terminals: u128,
    pub all_in_terminals: u128,
    pub showdown_board_evals: u128,
    pub all_in_board_evals: u128,
    pub terminal_cfv_calls: u128,
    pub terminal_private_pair_upper_bound: u128,
}

impl Default for CfrStorageConfig {
    fn default() -> Self {
        Self {
            chunk_target_bytes: 256 * 1024 * 1024,
            regret_bytes: 4,
            strategy_sum_bytes: 4,
        }
    }
}

impl CfrWorkPlan {
    pub fn storage_gib(&self) -> f64 {
        bytes_to_gib(self.total_storage_bytes)
    }

    pub fn max_chunk_mib(&self) -> f64 {
        bytes_to_mib(self.max_chunk_bytes)
    }
}

impl StreetWorkPlan {
    pub fn storage_gib(&self) -> f64 {
        bytes_to_gib(self.storage_bytes)
    }
}

pub fn plan_cfr_work(tree: &PublicTree, config: CfrStorageConfig) -> CfrWorkPlan {
    let mut street = [StreetWorkPlan::default(); 3];
    let mut terminals = TerminalWorkPlan::default();
    let slot_bytes = config.regret_bytes + config.strategy_sum_bytes;

    for node in &tree.nodes {
        match &node.kind {
            PublicNodeKind::Decision { player, actions } => {
                let street_index = street_index(node.state.street);
                let public_boards = public_board_multiplicity(node.state.street);
                let private_combos = live_private_combos(node.state.street);
                let acting_combos = match player {
                    Player::Oop | Player::Ip => private_combos,
                };
                let action_slots = public_boards * acting_combos * actions.len() as u128;
                street[street_index].decisions += 1;
                street[street_index].action_slots += action_slots;
            }
            PublicNodeKind::Terminal { reason } => match reason {
                TerminalReason::Fold => terminals.fold_terminals += 1,
                TerminalReason::Showdown => {
                    terminals.showdown_terminals += 1;
                    terminals.showdown_board_evals += public_board_multiplicity(node.state.street);
                    terminals.terminal_cfv_calls += public_board_multiplicity(node.state.street);
                    terminals.terminal_private_pair_upper_bound +=
                        public_board_multiplicity(node.state.street)
                            * private_pair_count(node.state.street);
                }
                TerminalReason::AllIn => {
                    terminals.all_in_terminals += 1;
                    let boards = all_in_runout_multiplicity(node.state.street);
                    terminals.all_in_board_evals += boards;
                    terminals.terminal_cfv_calls += boards;
                    terminals.terminal_private_pair_upper_bound +=
                        boards * private_pair_count(Street::River);
                }
            },
            PublicNodeKind::Chance(_) => {}
        }
    }

    let mut total_action_slots = 0u128;
    let mut total_storage_bytes = 0u128;
    let mut total_chunks = 0u128;
    let mut max_chunk_bytes = 0u128;
    for street_plan in &mut street {
        street_plan.storage_bytes = street_plan.action_slots * slot_bytes;
        street_plan.chunks = ceil_div(
            street_plan.storage_bytes,
            config.chunk_target_bytes.max(slot_bytes),
        )
        .max(if street_plan.storage_bytes == 0 { 0 } else { 1 });
        total_action_slots += street_plan.action_slots;
        total_storage_bytes += street_plan.storage_bytes;
        total_chunks += street_plan.chunks;
        max_chunk_bytes = max_chunk_bytes.max(if street_plan.chunks == 0 {
            0
        } else {
            ceil_div(street_plan.storage_bytes, street_plan.chunks)
        });
    }

    CfrWorkPlan {
        street,
        terminals,
        total_action_slots,
        total_storage_bytes,
        total_chunks,
        max_chunk_bytes,
    }
}

pub fn street_index(street: Street) -> usize {
    match street {
        Street::Flop => 0,
        Street::Turn => 1,
        Street::River => 2,
    }
}

pub fn public_board_multiplicity(street: Street) -> u128 {
    match street {
        Street::Flop => 1,
        Street::Turn => 49,
        Street::River => 49 * 48,
    }
}

pub fn live_private_combos(street: Street) -> u128 {
    match street {
        Street::Flop => choose2(49),
        Street::Turn => choose2(48),
        Street::River => choose2(47),
    }
}

pub fn private_pair_count(street: Street) -> u128 {
    let combos = live_private_combos(street);
    combos * combos
}

fn all_in_runout_multiplicity(street: Street) -> u128 {
    match street {
        Street::Flop => 49 * 48,
        Street::Turn => 48,
        Street::River => 1,
    }
}

fn choose2(count: u128) -> u128 {
    count * (count - 1) / 2
}

fn ceil_div(value: u128, divisor: u128) -> u128 {
    if value == 0 {
        0
    } else {
        (value - 1) / divisor + 1
    }
}

fn bytes_to_mib(bytes: u128) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn bytes_to_gib(bytes: u128) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Board, RangeSpec, Spot, TreeBuilder, TreeTemplate};
    use std::str::FromStr;

    #[test]
    fn default_flop_plan_is_dominated_by_river_storage() {
        let tree = TreeBuilder::new(TreeTemplate::conservative_default())
            .unwrap()
            .build(Spot {
                board: Board::from_str("As7h2c").unwrap(),
                pot: 650,
                effective_stack: 9700,
                oop_range: RangeSpec::full_deck_uniform(),
                ip_range: RangeSpec::full_deck_uniform(),
                first_player: Player::Oop,
            })
            .unwrap();
        let plan = plan_cfr_work(&tree, CfrStorageConfig::default());
        assert_eq!(plan.street[street_index(Street::Flop)].decisions, 14);
        assert_eq!(plan.street[street_index(Street::Turn)].decisions, 130);
        assert_eq!(plan.street[street_index(Street::River)].decisions, 665);
        assert!(
            plan.street[street_index(Street::River)].action_slots > plan.total_action_slots / 2
        );
        assert!(plan.terminals.fold_terminals > 0);
        assert!(plan.terminals.showdown_terminals > 0);
        assert!(plan.terminals.all_in_terminals > 0);
        let naive_all_terminal_pairs = tree.stats().terminals as u128
            * public_board_multiplicity(Street::River)
            * private_pair_count(Street::River);
        assert!(plan.terminals.terminal_private_pair_upper_bound < naive_all_terminal_pairs);
    }
}
