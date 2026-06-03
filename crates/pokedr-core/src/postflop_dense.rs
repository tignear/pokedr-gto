use crate::{
    dense_cfr::{CfrVariant, DenseCfrConfig},
    postflop::{ActionCandidate, PublicNodeKind, SubgameTree},
};

#[derive(Debug, Clone)]
pub struct PostflopDenseLayout {
    node_to_infoset: Vec<Option<usize>>,
    infoset_nodes: Vec<usize>,
    action_counts: Vec<usize>,
    max_actions: usize,
    legal_actions: Vec<bool>,
    child_by_action: Vec<Option<usize>>,
}

impl PostflopDenseLayout {
    pub fn from_tree(tree: &SubgameTree) -> Self {
        let mut node_to_infoset = vec![None; tree.nodes().len()];
        let mut infoset_nodes = Vec::new();
        let mut action_counts = Vec::new();
        let mut max_actions = 0;

        for (node_index, node) in tree.nodes().iter().enumerate() {
            if let PublicNodeKind::Decision { actions, .. } = &node.kind {
                let infoset = infoset_nodes.len();
                node_to_infoset[node_index] = Some(infoset);
                infoset_nodes.push(node_index);
                action_counts.push(actions.len());
                max_actions = max_actions.max(actions.len());
            }
        }

        assert!(!infoset_nodes.is_empty(), "tree must contain decisions");
        assert!(max_actions > 0, "decision nodes must contain actions");

        let mut legal_actions = vec![false; infoset_nodes.len() * max_actions];
        let mut child_by_action = vec![None; infoset_nodes.len() * max_actions];
        for (infoset, &node_index) in infoset_nodes.iter().enumerate() {
            let node = &tree.nodes()[node_index];
            let action_count = action_counts[infoset];
            assert_eq!(
                node.children.len(),
                action_count,
                "public tree children must match decision actions"
            );
            let offset = infoset * max_actions;
            for action in 0..action_count {
                legal_actions[offset + action] = true;
                child_by_action[offset + action] = Some(node.children[action]);
            }
        }

        Self {
            node_to_infoset,
            infoset_nodes,
            action_counts,
            max_actions,
            legal_actions,
            child_by_action,
        }
    }

    pub fn dense_config(&self, variant: CfrVariant) -> DenseCfrConfig {
        DenseCfrConfig {
            infosets: self.infoset_count(),
            actions: self.max_actions,
            variant,
        }
    }

    pub fn node_infoset(&self, node: usize) -> Option<usize> {
        self.node_to_infoset[node]
    }

    pub fn infoset_node(&self, infoset: usize) -> usize {
        self.infoset_nodes[infoset]
    }

    pub fn infoset_count(&self) -> usize {
        self.infoset_nodes.len()
    }

    pub fn max_actions(&self) -> usize {
        self.max_actions
    }

    pub fn action_count(&self, infoset: usize) -> usize {
        self.action_counts[infoset]
    }

    pub fn legal_actions(&self) -> &[bool] {
        &self.legal_actions
    }

    pub fn child_for_action(&self, infoset: usize, action: usize) -> Option<usize> {
        assert!(infoset < self.infoset_count());
        assert!(action < self.max_actions);
        self.child_by_action[infoset * self.max_actions + action]
    }

    pub fn action<'a>(
        &self,
        tree: &'a SubgameTree,
        infoset: usize,
        action: usize,
    ) -> Option<&'a ActionCandidate> {
        if action >= self.action_count(infoset) {
            return None;
        }
        let node = &tree.nodes()[self.infoset_node(infoset)];
        let PublicNodeKind::Decision { actions, .. } = &node.kind else {
            unreachable!("infoset nodes are decisions");
        };
        Some(&actions[action])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cards::{Board, Card, Rank, Suit},
        dense_cfr::DenseCfrState,
        postflop::{ActionSetConfig, Player, PublicState, Street, SubgameTreeConfig},
    };

    fn flop() -> Board {
        Board::new(vec![
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Seven, Suit::Hearts),
            Card::new(Rank::Two, Suit::Clubs),
        ])
    }

    fn tree() -> SubgameTree {
        SubgameTree::build(
            PublicState {
                street: Street::Flop,
                board: flop(),
                pot: 100,
                effective_stack: 300,
                to_call: 0,
                min_aggressive_amount: 50,
                acting_player: Player::Hero,
                raises_this_street: 0,
                checks_this_street: 0,
            },
            SubgameTreeConfig {
                action_set: ActionSetConfig {
                    max_aggressive_actions: 2,
                    flop_bet_fractions: vec![0.5],
                    turn_bet_fractions: vec![0.5],
                    river_bet_fractions: vec![0.5],
                    raise_fractions: vec![1.0],
                    ..ActionSetConfig::default()
                },
                max_raises_per_street: 1,
                max_depth: 4,
            },
        )
    }

    #[test]
    fn layout_maps_public_decisions_to_dense_infosets() {
        let tree = tree();
        let layout = PostflopDenseLayout::from_tree(&tree);

        assert_eq!(layout.infoset_count(), tree.decision_count());
        assert!(layout.max_actions() >= 2);
        for infoset in 0..layout.infoset_count() {
            let node = &tree.nodes()[layout.infoset_node(infoset)];
            let PublicNodeKind::Decision { actions, .. } = &node.kind else {
                panic!("infoset should point at a decision");
            };
            assert_eq!(layout.action_count(infoset), actions.len());
            assert_eq!(
                layout.node_infoset(layout.infoset_node(infoset)),
                Some(infoset)
            );
            for action in 0..layout.action_count(infoset) {
                assert_eq!(
                    layout.child_for_action(infoset, action),
                    Some(node.children[action])
                );
                assert!(layout.action(&tree, infoset, action).is_some());
            }
        }
    }

    #[test]
    fn dense_state_uses_layout_mask_for_padding_actions() {
        let tree = tree();
        let layout = PostflopDenseLayout::from_tree(&tree);
        let config = layout.dense_config(CfrVariant::CfrPlus);
        let state = DenseCfrState::new_with_legal_actions(config, layout.legal_actions().to_vec());

        for infoset in 0..layout.infoset_count() {
            let mut strategy = vec![0.0; layout.max_actions()];
            state.strategy_for(infoset, &mut strategy);
            let sum: f32 = strategy.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
            for (action, probability) in strategy.iter().enumerate() {
                if action >= layout.action_count(infoset) {
                    assert_eq!(*probability, 0.0);
                }
            }
        }
    }
}
