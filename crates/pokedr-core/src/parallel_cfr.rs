use crate::cards::Board;
use crate::isomorphism::next_card_isomorphism;
use crate::range::{ComboWeight, RangeSpec};
use crate::tree::{Player, PublicNodeKind, PublicTree};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelCfrStorageReport {
    pub nodes: usize,
    pub decision_nodes: usize,
    pub chance_nodes: usize,
    pub terminal_nodes: usize,
    pub action_slots: usize,
    pub strategy_slots: usize,
    pub regret_slots: usize,
    pub ip_cfvalue_slots: usize,
    pub chance_cfvalue_slots: usize,
    pub scratch_value_slots: usize,
    pub parallel_cut_nodes: usize,
    pub max_parallel_fanout: usize,
    pub concrete_chance_events: usize,
    pub representative_chance_events: usize,
    pub chance_permutation_members: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParallelStatePlan {
    node_id: usize,
    board: Board,
    postorder: usize,
    subtree_start_postorder: usize,
    subtree_end_postorder: usize,
    parallel_cut: bool,
    children: Vec<usize>,
    chance_permutation_codes: Vec<Vec<u8>>,
    concrete_chance_events: usize,
    acting_player: Option<Player>,
    cfvalue_storage_player: Option<Player>,
}

#[derive(Debug, Clone)]
pub struct ParallelCfrSolver {
    oop_combos: Vec<ComboWeight>,
    ip_combos: Vec<ComboWeight>,
    states: Vec<ParallelStatePlan>,
    storage: ParallelCfrStorageReport,
}

impl ParallelCfrSolver {
    pub fn new(
        tree: PublicTree,
        oop_range: RangeSpec,
        ip_range: RangeSpec,
    ) -> Result<Self, String> {
        let oop_combos = oop_range.combos().to_vec();
        let ip_combos = ip_range.combos().to_vec();
        let mut states = Vec::new();
        let mut index_by_key = BTreeMap::new();
        collect_state_from(
            &tree,
            &oop_range,
            &ip_range,
            0,
            &tree.spot.board,
            &mut states,
            &mut index_by_key,
        )?;
        let mut visited = vec![false; states.len()];
        let mut postorder_states = Vec::with_capacity(states.len());
        build_state_postorder(&states, 0, &mut visited, &mut postorder_states)?;
        for (postorder, state_index) in postorder_states.iter().copied().enumerate() {
            states[state_index].postorder = postorder;
        }
        for state_index in 0..states.len() {
            let start = state_subtree_start_postorder(&states, state_index);
            states[state_index].subtree_start_postorder = start;
            states[state_index].subtree_end_postorder = states[state_index].postorder + 1;
        }
        let storage = storage_report(&tree, &oop_combos, &ip_combos, &states);
        Ok(Self {
            oop_combos,
            ip_combos,
            states,
            storage,
        })
    }

    pub fn storage_report(&self) -> ParallelCfrStorageReport {
        self.storage
    }

    pub fn node_count(&self) -> usize {
        self.storage.nodes
    }

    pub fn combo_counts(&self) -> (usize, usize) {
        (self.oop_combos.len(), self.ip_combos.len())
    }

    pub fn parallel_cut_subtree_postorder_ranges(&self) -> Vec<(usize, usize)> {
        self.states
            .iter()
            .filter(|state| state.parallel_cut)
            .map(|state| (state.subtree_start_postorder, state.subtree_end_postorder))
            .collect()
    }
}

fn collect_state_from(
    tree: &PublicTree,
    oop_range: &RangeSpec,
    ip_range: &RangeSpec,
    node_id: usize,
    board: &Board,
    states: &mut Vec<ParallelStatePlan>,
    index_by_key: &mut BTreeMap<(usize, u64), usize>,
) -> Result<usize, String> {
    let key = (node_id, board_key(board));
    if let Some(index) = index_by_key.get(&key) {
        return Ok(*index);
    }
    let node = tree
        .nodes
        .get(node_id)
        .ok_or_else(|| "public tree node id is out of bounds".to_string())?;
    let state_index = states.len();
    index_by_key.insert(key, state_index);
    states.push(ParallelStatePlan {
        node_id,
        board: board.clone(),
        postorder: 0,
        subtree_start_postorder: 0,
        subtree_end_postorder: 0,
        parallel_cut: false,
        children: Vec::new(),
        chance_permutation_codes: Vec::new(),
        concrete_chance_events: 0,
        acting_player: None,
        cfvalue_storage_player: None,
    });
    let mut children = Vec::new();
    let mut chance_permutation_codes = Vec::new();
    let mut concrete_chance_events = 0usize;
    let mut acting_player = None;
    let mut cfvalue_storage_player = None;
    match &node.kind {
        PublicNodeKind::Terminal { .. } => {}
        PublicNodeKind::Chance(_) => {
            cfvalue_storage_player = Some(node.state.player.other());
            let Some(child) = node.children.first().copied() else {
                return Ok(state_index);
            };
            let chance = next_card_isomorphism(board, oop_range, ip_range);
            concrete_chance_events = chance.concrete_events;
            for class in chance.classes {
                let card = *class
                    .representative
                    .first()
                    .ok_or_else(|| "chance class has no representative card".to_string())?;
                children.push(collect_state_from(
                    tree,
                    oop_range,
                    ip_range,
                    child,
                    &board.push(card)?,
                    states,
                    index_by_key,
                )?);
                chance_permutation_codes.push(
                    class
                        .members
                        .iter()
                        .map(|member| member.permutation_to_representative.code())
                        .collect(),
                );
            }
        }
        PublicNodeKind::Decision { player, .. } => {
            acting_player = Some(*player);
            if state_index == 0
                || matches!(node.children.first(), Some(child) if tree.nodes[*child].state.street != node.state.street)
            {
                cfvalue_storage_player = Some(Player::Ip);
            }
            for child in &node.children {
                children.push(collect_state_from(
                    tree,
                    oop_range,
                    ip_range,
                    *child,
                    board,
                    states,
                    index_by_key,
                )?);
            }
        }
    }
    states[state_index].parallel_cut = match &node.kind {
        PublicNodeKind::Chance(_) => board.cards().len() < 5 && children.len() > 1,
        PublicNodeKind::Decision { .. } => board.cards().len() < 5 && children.len() > 1,
        PublicNodeKind::Terminal { .. } => false,
    };
    states[state_index].children = children;
    states[state_index].chance_permutation_codes = chance_permutation_codes;
    states[state_index].concrete_chance_events = concrete_chance_events;
    states[state_index].acting_player = acting_player;
    states[state_index].cfvalue_storage_player = cfvalue_storage_player;
    Ok(state_index)
}

fn build_state_postorder(
    states: &[ParallelStatePlan],
    state_index: usize,
    visited: &mut [bool],
    postorder_states: &mut Vec<usize>,
) -> Result<(), String> {
    if visited[state_index] {
        return Ok(());
    }
    visited[state_index] = true;
    for child in &states[state_index].children {
        build_state_postorder(states, *child, visited, postorder_states)?;
    }
    postorder_states.push(state_index);
    Ok(())
}

fn state_subtree_start_postorder(states: &[ParallelStatePlan], state_index: usize) -> usize {
    states[state_index]
        .children
        .iter()
        .map(|child| state_subtree_start_postorder(states, *child))
        .min()
        .unwrap_or(states[state_index].postorder)
}

fn storage_report(
    tree: &PublicTree,
    oop_combos: &[ComboWeight],
    ip_combos: &[ComboWeight],
    states: &[ParallelStatePlan],
) -> ParallelCfrStorageReport {
    let mut report = ParallelCfrStorageReport {
        nodes: tree.nodes.len(),
        decision_nodes: 0,
        chance_nodes: 0,
        terminal_nodes: 0,
        action_slots: 0,
        strategy_slots: 0,
        regret_slots: 0,
        ip_cfvalue_slots: 0,
        chance_cfvalue_slots: 0,
        scratch_value_slots: 0,
        parallel_cut_nodes: 0,
        max_parallel_fanout: 0,
        concrete_chance_events: 0,
        representative_chance_events: 0,
        chance_permutation_members: 0,
    };
    let mut strategy_offset = 0usize;
    let mut regret_offset = 0usize;
    let mut ip_cfvalue_offset = 0usize;
    let mut chance_cfvalue_offset = 0usize;
    for state in states {
        let node = &tree.nodes[state.node_id];
        report.max_parallel_fanout = report.max_parallel_fanout.max(state.children.len());
        if state.parallel_cut {
            report.parallel_cut_nodes += 1;
        }
        match &node.kind {
            PublicNodeKind::Decision { player, actions } => {
                report.decision_nodes += 1;
                let combos = match player {
                    Player::Oop => oop_combos.len(),
                    Player::Ip => ip_combos.len(),
                };
                let slots = combos * actions.len();
                report.action_slots += slots;
                report.strategy_slots += slots;
                report.regret_slots += slots;
                strategy_offset += slots;
                regret_offset += slots;
                if state.cfvalue_storage_player == Some(Player::Ip) {
                    report.ip_cfvalue_slots += ip_combos.len();
                    ip_cfvalue_offset += ip_combos.len();
                }
            }
            PublicNodeKind::Chance(_) => {
                report.chance_nodes += 1;
                report.concrete_chance_events += state.concrete_chance_events;
                report.representative_chance_events += state.children.len();
                report.chance_permutation_members += state
                    .chance_permutation_codes
                    .iter()
                    .map(Vec::len)
                    .sum::<usize>();
                let combos = match node.state.player {
                    Player::Oop => ip_combos.len(),
                    Player::Ip => oop_combos.len(),
                };
                report.scratch_value_slots += combos;
                report.chance_cfvalue_slots += combos;
                chance_cfvalue_offset += combos;
            }
            PublicNodeKind::Terminal { .. } => {
                report.terminal_nodes += 1;
            }
        }
    }
    debug_assert_eq!(report.strategy_slots, strategy_offset);
    debug_assert_eq!(report.regret_slots, regret_offset);
    debug_assert_eq!(report.ip_cfvalue_slots, ip_cfvalue_offset);
    debug_assert_eq!(report.chance_cfvalue_slots, chance_cfvalue_offset);
    report
}

fn board_key(board: &Board) -> u64 {
    board
        .cards()
        .iter()
        .fold(0u64, |key, card| key | (1u64 << card.index()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::Board;
    use crate::tree::{ActionAbstraction, ChanceExpansion, Spot, TreeBuilder, TreeTemplate};
    use std::str::FromStr;

    #[test]
    fn parallel_cfr_plan_reports_tree_storage_and_parallel_cuts() {
        let board = Board::from_str("As7h2c").unwrap();
        let oop_range = RangeSpec::from_str("AcAd,KcKd").unwrap();
        let ip_range = RangeSpec::from_str("QcQd,JcJd").unwrap();
        let tree = TreeBuilder::new(TreeTemplate {
            action_abstraction: ActionAbstraction::postflop_solver_basic(),
            chance_expansion: ChanceExpansion::Enumerate,
        })
        .unwrap()
        .build(Spot {
            board,
            pot: 200,
            effective_stack: 900,
            oop_range: oop_range.clone(),
            ip_range: ip_range.clone(),
            first_player: Player::Oop,
        })
        .unwrap();
        let solver = ParallelCfrSolver::new(tree, oop_range, ip_range).unwrap();
        let report = solver.storage_report();
        assert!(report.nodes > 0);
        assert!(report.decision_nodes > 0);
        assert!(report.action_slots > 0);
        assert_eq!(report.strategy_slots, report.regret_slots);
        assert!(report.parallel_cut_nodes > 0);
        assert!(report.max_parallel_fanout > 1);
    }
}
