use crate::cards::Board;
use crate::isomorphism::next_card_isomorphism;
use crate::range::{ComboWeight, RangeSpec};
use crate::tree::{Player, PublicNodeKind, PublicTree};
use std::cell::UnsafeCell;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeLocalCfrSummary {
    pub states: usize,
    pub decision_states: usize,
    pub action_slots: usize,
    pub storage_gib: f64,
}

#[derive(Debug)]
pub struct NodeLocalCfrSolver {
    tree: PublicTree,
    oop_range: RangeSpec,
    ip_range: RangeSpec,
    oop_combos: Vec<ComboWeight>,
    ip_combos: Vec<ComboWeight>,
    nodes: Vec<NodeLocalNodeCell>,
    node_by_key: BTreeMap<(usize, u64), usize>,
    action_slots: usize,
    decision_states: usize,
}

#[derive(Debug)]
#[repr(transparent)]
struct NodeLocalNodeCell(UnsafeCell<NodeLocalNode>);

unsafe impl Send for NodeLocalNodeCell {}
unsafe impl Sync for NodeLocalNodeCell {}

impl NodeLocalNodeCell {
    fn new(node: NodeLocalNode) -> Self {
        Self(UnsafeCell::new(node))
    }

    fn get(&self) -> &NodeLocalNode {
        unsafe { &*self.0.get() }
    }

    fn get_mut(&self) -> &mut NodeLocalNode {
        unsafe { &mut *self.0.get() }
    }
}

#[derive(Debug)]
struct NodeLocalNode {
    public_node: usize,
    board: Board,
    kind: NodeLocalKind,
    children: Vec<usize>,
    chance_concrete_events: usize,
    chance_permutation_codes: Vec<Vec<u8>>,
    regrets: Vec<f32>,
    strategy_sum: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeLocalKind {
    Terminal,
    Chance,
    Decision { player: Player, actions: usize },
}

impl NodeLocalCfrSolver {
    pub fn new(
        tree: PublicTree,
        oop_range: RangeSpec,
        ip_range: RangeSpec,
    ) -> Result<Self, String> {
        let oop_combos = oop_range.combos().to_vec();
        let ip_combos = ip_range.combos().to_vec();
        let mut solver = Self {
            tree,
            oop_range,
            ip_range,
            oop_combos,
            ip_combos,
            nodes: Vec::new(),
            node_by_key: BTreeMap::new(),
            action_slots: 0,
            decision_states: 0,
        };
        let board = solver.tree.spot.board.clone();
        solver.collect_node(0, &board)?;
        Ok(solver)
    }

    pub fn summary(&self) -> NodeLocalCfrSummary {
        let mut board_cards = 0usize;
        let mut public_nodes = 0usize;
        for node in &self.nodes {
            let node = node.get();
            board_cards += node.board.cards().len();
            public_nodes = public_nodes.wrapping_add(node.public_node);
        }
        std::hint::black_box((board_cards, public_nodes));
        NodeLocalCfrSummary {
            states: self.nodes.len(),
            decision_states: self.decision_states,
            action_slots: self.action_slots,
            storage_gib: self.storage_gib(),
        }
    }

    pub fn storage_gib(&self) -> f64 {
        self.action_slots as f64 * 2.0 * std::mem::size_of::<f32>() as f64
            / (1024.0 * 1024.0 * 1024.0)
    }

    fn collect_node(&mut self, public_node: usize, board: &Board) -> Result<usize, String> {
        let key = (public_node, ordered_board_key(board));
        if let Some(index) = self.node_by_key.get(&key) {
            return Ok(*index);
        }

        let index = self.nodes.len();
        self.node_by_key.insert(key, index);
        self.nodes.push(NodeLocalNodeCell::new(NodeLocalNode {
            public_node,
            board: board.clone(),
            kind: NodeLocalKind::Terminal,
            children: Vec::new(),
            chance_concrete_events: 0,
            chance_permutation_codes: Vec::new(),
            regrets: Vec::new(),
            strategy_sum: Vec::new(),
        }));

        let public = self
            .tree
            .nodes
            .get(public_node)
            .ok_or_else(|| "public node index is out of bounds".to_string())?
            .clone();

        let mut kind = NodeLocalKind::Terminal;
        let mut children = Vec::new();
        let mut chance_concrete_events = 0usize;
        let mut chance_permutation_codes = Vec::new();
        let mut regrets = Vec::new();
        let mut strategy_sum = Vec::new();

        match public.kind {
            PublicNodeKind::Terminal { .. } => {}
            PublicNodeKind::Chance(_) => {
                kind = NodeLocalKind::Chance;
                let Some(child) = public.children.first().copied() else {
                    return Ok(index);
                };
                let chance = next_card_isomorphism(board, &self.oop_range, &self.ip_range);
                chance_concrete_events = chance.concrete_events;
                for class in chance.classes {
                    let card = *class
                        .representative
                        .first()
                        .ok_or_else(|| "chance class has no representative".to_string())?;
                    children.push(self.collect_node(child, &board.push(card)?)?);
                    chance_permutation_codes.push(
                        class
                            .members
                            .iter()
                            .map(|member| member.permutation_to_representative.code())
                            .collect(),
                    );
                }
            }
            PublicNodeKind::Decision { player, actions } => {
                kind = NodeLocalKind::Decision {
                    player,
                    actions: actions.len(),
                };
                for child in public.children {
                    children.push(self.collect_node(child, board)?);
                }
                if actions.len() > 1 {
                    let combos = match player {
                        Player::Oop => self.oop_combos.len(),
                        Player::Ip => self.ip_combos.len(),
                    };
                    let slots = combos * actions.len();
                    regrets.resize(slots, 0.0);
                    strategy_sum.resize(slots, 0.0);
                    self.action_slots += slots;
                    self.decision_states += 1;
                }
            }
        }

        let node = self.nodes[index].get_mut();
        node.kind = kind;
        node.children = children;
        node.chance_concrete_events = chance_concrete_events;
        node.chance_permutation_codes = chance_permutation_codes;
        node.regrets = regrets;
        node.strategy_sum = strategy_sum;
        Ok(index)
    }
}

fn ordered_board_key(board: &Board) -> u64 {
    board
        .cards()
        .iter()
        .fold(0u64, |key, card| (key << 6) | card.index() as u64)
}
