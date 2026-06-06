use pokedr_core::{
    ActionAbstraction, Board, Player, PublicTree, RangeSpec, Spot, TreeBuildError, TreeBuilder,
    TreeTemplate,
};

#[derive(Debug, Clone)]
pub struct FlopTreeRequest {
    pub board: Board,
    pub pot: u32,
    pub effective_stack: u32,
    pub oop_range: RangeSpec,
    pub ip_range: RangeSpec,
    pub first_player: Player,
    pub action_abstraction: ActionAbstraction,
}

impl FlopTreeRequest {
    pub fn standard_srp(board: Board, oop_range: RangeSpec, ip_range: RangeSpec) -> Self {
        Self {
            board,
            pot: 650,
            effective_stack: 9700,
            oop_range,
            ip_range,
            first_player: Player::Oop,
            action_abstraction: ActionAbstraction::conservative_default(),
        }
    }
}

pub fn build_flop_tree(request: FlopTreeRequest) -> Result<PublicTree, TreeBuildError> {
    let template = TreeTemplate {
        action_abstraction: request.action_abstraction,
        ..TreeTemplate::conservative_default()
    };
    let spot = Spot {
        board: request.board,
        pot: request.pot,
        effective_stack: request.effective_stack,
        oop_range: request.oop_range,
        ip_range: request.ip_range,
        first_player: request.first_player,
    };
    TreeBuilder::new(template)?.build(spot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn builds_standard_srp_tree() {
        let request = FlopTreeRequest::standard_srp(
            Board::from_str("As7h2c").unwrap(),
            RangeSpec::full_deck_uniform(),
            RangeSpec::full_deck_uniform(),
        );
        let tree = build_flop_tree(request).unwrap();
        let stats = tree.stats();
        assert!(stats.decisions > 100);
        assert!(stats.terminals > 100);
    }
}
