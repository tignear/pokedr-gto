use crate::cards::Card;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HandClass {
    pub high: u8,
    pub low: u8,
    pub suited: bool,
}

impl HandClass {
    pub fn new(high: u8, low: u8, suited: bool) -> Self {
        debug_assert!((2..=14).contains(&high));
        debug_assert!((2..=14).contains(&low));
        debug_assert!(high >= low);
        debug_assert!(high != low || !suited);
        Self { high, low, suited }
    }

    pub fn label(self) -> String {
        let mut label = format!("{}{}", rank_label(self.high), rank_label(self.low));
        if self.high != self.low {
            label.push(if self.suited { 's' } else { 'o' });
        }
        label
    }

    pub fn combos(self) -> Vec<[Card; 2]> {
        let mut combos = Vec::new();

        if self.high == self.low {
            for first_suit in 0..4 {
                for second_suit in (first_suit + 1)..4 {
                    combos.push([
                        Card::new(self.high, first_suit),
                        Card::new(self.low, second_suit),
                    ]);
                }
            }

            return combos;
        }

        for high_suit in 0..4 {
            for low_suit in 0..4 {
                if self.suited == (high_suit == low_suit) {
                    combos.push([
                        Card::new(self.high, high_suit),
                        Card::new(self.low, low_suit),
                    ]);
                }
            }
        }

        combos
    }
}

pub fn all_hand_classes() -> Vec<HandClass> {
    let mut classes = Vec::with_capacity(169);

    for high in (2..=14).rev() {
        for low in (2..=high).rev() {
            if high == low {
                classes.push(HandClass::new(high, low, false));
            } else {
                classes.push(HandClass::new(high, low, true));
                classes.push(HandClass::new(high, low, false));
            }
        }
    }

    classes
}

pub fn rank_label(rank: u8) -> char {
    match rank {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_classes_contains_169_preflop_classes() {
        assert_eq!(all_hand_classes().len(), 169);
    }

    #[test]
    fn combo_counts_match_holdem() {
        assert_eq!(HandClass::new(14, 14, false).combos().len(), 6);
        assert_eq!(HandClass::new(14, 13, true).combos().len(), 4);
        assert_eq!(HandClass::new(14, 13, false).combos().len(), 12);
    }
}
