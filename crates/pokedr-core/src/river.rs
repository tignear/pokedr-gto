use crate::cards::{Card, deck};
use crate::hand_eval::{HandValue, evaluate_seven};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Combo {
    pub first: Card,
    pub second: Card,
}

#[derive(Debug, Clone)]
pub struct RiverCombo {
    pub combo: Combo,
    pub mask: u64,
    pub value: HandValue,
}

#[derive(Debug, Clone)]
pub struct RiverBlockerReport {
    pub hero: RiverCombo,
    pub blocked_villain_combos: usize,
    pub blocked_top_combos: usize,
    pub total_villain_combos: usize,
    pub top_villain_combos: usize,
}

impl Combo {
    pub fn new(first: Card, second: Card) -> Option<Self> {
        if first == second {
            return None;
        }

        Some(Self { first, second })
    }

    pub fn mask(self) -> u64 {
        self.first.mask() | self.second.mask()
    }

    pub fn label(self) -> String {
        format!("{}{}", card_label(self.first), card_label(self.second))
    }
}

impl RiverCombo {
    pub fn label(&self) -> String {
        self.combo.label()
    }
}

pub fn river_combos(board: [Card; 5]) -> Vec<RiverCombo> {
    let board_mask = board_mask(board);
    let cards = deck();
    let mut combos = Vec::with_capacity(1081);

    for first_index in 0..cards.len() {
        let first = cards[first_index];
        if board_mask & first.mask() != 0 {
            continue;
        }

        for &second in cards.iter().skip(first_index + 1) {
            let Some(combo) = Combo::new(first, second) else {
                continue;
            };
            let mask = combo.mask();
            if board_mask & mask != 0 {
                continue;
            }

            let value = evaluate_seven([
                combo.first,
                combo.second,
                board[0],
                board[1],
                board[2],
                board[3],
                board[4],
            ]);
            combos.push(RiverCombo { combo, mask, value });
        }
    }

    combos.sort_by(|left, right| {
        right
            .value
            .cmp(&left.value)
            .then_with(|| left.combo.first.0.cmp(&right.combo.first.0))
            .then_with(|| left.combo.second.0.cmp(&right.combo.second.0))
    });
    combos
}

pub fn river_blocker_reports(
    board: [Card; 5],
    hero_combos: &[RiverCombo],
    villain_combos: &[RiverCombo],
    top_fraction: f64,
) -> Vec<RiverBlockerReport> {
    let top_count = ((villain_combos.len() as f64) * top_fraction.clamp(0.0, 1.0))
        .ceil()
        .max(1.0) as usize;
    let top_count = top_count.min(villain_combos.len());
    let top_villain = &villain_combos[..top_count];
    let board_mask = board_mask(board);

    hero_combos
        .iter()
        .filter(|hero| hero.mask & board_mask == 0)
        .map(|hero| {
            let blocked_villain_combos = villain_combos
                .iter()
                .filter(|villain| hero.mask & villain.mask != 0)
                .count();
            let blocked_top_combos = top_villain
                .iter()
                .filter(|villain| hero.mask & villain.mask != 0)
                .count();

            RiverBlockerReport {
                hero: hero.clone(),
                blocked_villain_combos,
                blocked_top_combos,
                total_villain_combos: villain_combos.len(),
                top_villain_combos: top_count,
            }
        })
        .collect()
}

pub fn board_mask(board: [Card; 5]) -> u64 {
    board.iter().fold(0_u64, |mask, card| mask | card.mask())
}

fn card_label(card: Card) -> String {
    let rank = match card.rank() {
        14 => 'A',
        13 => 'K',
        12 => 'Q',
        11 => 'J',
        10 => 'T',
        9 => '9',
        8 => '8',
        7 => '7',
        6 => '6',
        5 => '5',
        4 => '4',
        3 => '3',
        2 => '2',
        _ => '?',
    };
    let suit = match card.suit() {
        0 => 'c',
        1 => 'd',
        2 => 'h',
        3 => 's',
        _ => '?',
    };
    format!("{rank}{suit}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(rank: u8, suit: u8) -> Card {
        Card::new(rank, suit)
    }

    #[test]
    fn river_combos_exclude_board_cards() {
        let board = [c(14, 0), c(13, 0), c(12, 0), c(11, 0), c(2, 1)];
        let combos = river_combos(board);

        assert_eq!(combos.len(), 1081);
        assert!(
            combos
                .iter()
                .all(|combo| combo.mask & board_mask(board) == 0)
        );
    }

    #[test]
    fn river_combos_order_stronger_hands_first() {
        let board = [c(14, 0), c(13, 0), c(12, 0), c(11, 0), c(2, 1)];
        let combos = river_combos(board);

        assert!(combos[0].value >= combos[1].value);
        assert_eq!(combos[0].label(), "2cTc");
    }

    #[test]
    fn blocker_report_counts_top_range_blockers() {
        let board = [c(14, 0), c(13, 0), c(12, 0), c(11, 0), c(2, 1)];
        let combos = river_combos(board);
        let ten_spade = combos
            .iter()
            .find(|combo| combo.label() == "2cTc")
            .cloned()
            .expect("combo should be available");
        let reports = river_blocker_reports(board, &[ten_spade], &combos, 0.01);

        assert_eq!(reports.len(), 1);
        assert!(reports[0].blocked_villain_combos > 0);
        assert!(reports[0].blocked_top_combos > 0);
    }
}
