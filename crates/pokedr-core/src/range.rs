use crate::cards::{Board, Card};

pub const COMBO_COUNT: usize = 1326;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Combo {
    pub first: Card,
    pub second: Card,
}

#[derive(Debug, Clone)]
pub struct ComboIndexer {
    combos: Vec<Combo>,
    indices: [[Option<u16>; Card::COUNT]; Card::COUNT],
}

#[derive(Debug, Clone)]
pub struct HandRange {
    weights: Vec<f32>,
}

impl Combo {
    pub fn new(first: Card, second: Card) -> Self {
        assert_ne!(first, second, "combo cannot contain duplicate card");
        if first.index() < second.index() {
            Self { first, second }
        } else {
            Self {
                first: second,
                second: first,
            }
        }
    }

    pub fn deck_mask(self) -> u64 {
        self.first.deck_mask() | self.second.deck_mask()
    }

    pub fn collides_with(self, mask: u64) -> bool {
        self.deck_mask() & mask != 0
    }
}

impl Default for ComboIndexer {
    fn default() -> Self {
        Self::new()
    }
}

impl ComboIndexer {
    pub fn new() -> Self {
        let mut combos = Vec::with_capacity(COMBO_COUNT);
        let mut indices = [[None; Card::COUNT]; Card::COUNT];
        for first in 0..Card::COUNT as u8 {
            for second in first + 1..Card::COUNT as u8 {
                let index = combos.len() as u16;
                combos.push(Combo {
                    first: Card::from_index(first),
                    second: Card::from_index(second),
                });
                indices[first as usize][second as usize] = Some(index);
                indices[second as usize][first as usize] = Some(index);
            }
        }
        debug_assert_eq!(combos.len(), COMBO_COUNT);
        Self { combos, indices }
    }

    pub fn combos(&self) -> &[Combo] {
        &self.combos
    }

    pub fn combo(&self, index: usize) -> Combo {
        self.combos[index]
    }

    pub fn index(&self, first: Card, second: Card) -> Option<usize> {
        self.indices[first.index() as usize][second.index() as usize].map(usize::from)
    }

    pub fn board_legal_mask(&self, board: &Board) -> Vec<bool> {
        let board_mask = board.deck_mask();
        self.combos
            .iter()
            .map(|combo| !combo.collides_with(board_mask))
            .collect()
    }
}

impl HandRange {
    pub fn zero() -> Self {
        Self {
            weights: vec![0.0; COMBO_COUNT],
        }
    }

    pub fn uniform() -> Self {
        Self {
            weights: vec![1.0; COMBO_COUNT],
        }
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }

    pub fn apply_board_mask(&mut self, indexer: &ComboIndexer, board: &Board) {
        let board_mask = board.deck_mask();
        for (weight, combo) in self.weights.iter_mut().zip(indexer.combos()) {
            if combo.collides_with(board_mask) {
                *weight = 0.0;
            }
        }
    }

    pub fn normalize(&mut self) -> f32 {
        let sum: f32 = self.weights.iter().sum();
        if sum > 0.0 {
            for weight in &mut self.weights {
                *weight /= sum;
            }
        }
        sum
    }

    pub fn masked_uniform(indexer: &ComboIndexer, board: &Board) -> Self {
        let mut range = Self::uniform();
        range.apply_board_mask(indexer, board);
        range.normalize();
        range
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::{Rank, Suit};

    #[test]
    fn combo_indexer_contains_all_unordered_pairs() {
        let indexer = ComboIndexer::new();
        assert_eq!(indexer.combos().len(), COMBO_COUNT);

        let ac = Card::new(Rank::Ace, Suit::Clubs);
        let ad = Card::new(Rank::Ace, Suit::Diamonds);
        assert_eq!(indexer.index(ac, ad), indexer.index(ad, ac));
        assert!(indexer.index(ac, ac).is_none());
    }

    #[test]
    fn board_mask_removes_blocked_combos() {
        let indexer = ComboIndexer::new();
        let board = Board::new(vec![
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::King, Suit::Hearts),
            Card::new(Rank::Two, Suit::Clubs),
        ]);
        let legal = indexer.board_legal_mask(&board);

        assert_eq!(legal.iter().filter(|is_legal| **is_legal).count(), 1176);
        for (index, combo) in indexer.combos().iter().enumerate() {
            if combo.collides_with(board.deck_mask()) {
                assert!(!legal[index]);
            }
        }
    }

    #[test]
    fn masked_uniform_normalizes_over_legal_combos() {
        let indexer = ComboIndexer::new();
        let board = Board::new(vec![
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::King, Suit::Hearts),
            Card::new(Rank::Two, Suit::Clubs),
        ]);
        let range = HandRange::masked_uniform(&indexer, &board);

        let sum: f32 = range.weights().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert_eq!(range.weights().iter().filter(|w| **w > 0.0).count(), 1176);
    }
}
