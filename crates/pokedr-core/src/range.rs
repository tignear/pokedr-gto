use crate::cards::{Board, Card, Rank, Suit};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComboWeight {
    pub first: Card,
    pub second: Card,
    pub weight: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RangeSpec {
    combos: Vec<ComboWeight>,
}

impl RangeSpec {
    pub fn new(combos: Vec<ComboWeight>) -> Result<Self, String> {
        for combo in &combos {
            if combo.first == combo.second {
                return Err(format!("range combo repeats card {}", combo.first));
            }
            if !combo.weight.is_finite() || combo.weight < 0.0 {
                return Err("range weights must be finite and non-negative".to_string());
            }
        }
        Ok(Self { combos })
    }

    pub fn full_deck_uniform() -> Self {
        let deck = Card::deck();
        let mut combos = Vec::with_capacity(1326);
        for i in 0..deck.len() {
            for j in i + 1..deck.len() {
                combos.push(ComboWeight {
                    first: deck[i],
                    second: deck[j],
                    weight: 1.0,
                });
            }
        }
        Self { combos }
    }

    pub fn combos(&self) -> &[ComboWeight] {
        &self.combos
    }

    pub fn without_board_conflicts(&self, board: &Board) -> Self {
        Self {
            combos: self
                .combos
                .iter()
                .copied()
                .filter(|combo| !board.contains(combo.first) && !board.contains(combo.second))
                .collect(),
        }
    }
}

impl FromStr for RangeSpec {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.trim().eq_ignore_ascii_case("full") {
            return Ok(Self::full_deck_uniform());
        }
        let mut combos = Vec::new();
        for token in value
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
        {
            let (cards, weight) = token
                .split_once(':')
                .map_or((token, 1.0), |(cards, weight)| {
                    (cards, weight.parse::<f32>().unwrap_or(f32::NAN))
                });
            if is_exact_combo_token(cards) {
                let first = Card::from_str(&cards[0..2])?;
                let second = Card::from_str(&cards[2..4])?;
                push_combo(&mut combos, first, second, weight);
            } else {
                expand_range_token(cards, weight, &mut combos)?;
            }
        }
        Self::new(combos)
    }
}

fn is_exact_combo_token(token: &str) -> bool {
    token.len() == 4
        && token
            .chars()
            .nth(1)
            .is_some_and(|value| matches!(value, 'c' | 'C' | 'd' | 'D' | 'h' | 'H' | 's' | 'S'))
        && token
            .chars()
            .nth(3)
            .is_some_and(|value| matches!(value, 'c' | 'C' | 'd' | 'D' | 'h' | 'H' | 's' | 'S'))
}

fn push_combo(combos: &mut Vec<ComboWeight>, first: Card, second: Card, weight: f32) {
    let (first, second) = if first.index() <= second.index() {
        (first, second)
    } else {
        (second, first)
    };
    if !combos
        .iter()
        .any(|combo| combo.first == first && combo.second == second)
    {
        combos.push(ComboWeight {
            first,
            second,
            weight,
        });
    }
}

fn expand_range_token(
    token: &str,
    weight: f32,
    combos: &mut Vec<ComboWeight>,
) -> Result<(), String> {
    let token = token.trim();
    if token.len() < 2 || token.len() > 4 {
        return Err(format!("invalid range token {token:?}"));
    }
    let mut chars = token.chars();
    let high = chars
        .next()
        .and_then(parse_rank)
        .ok_or_else(|| format!("invalid range rank in {token:?}"))?;
    let low = chars
        .next()
        .and_then(parse_rank)
        .ok_or_else(|| format!("invalid range rank in {token:?}"))?;
    let suffix = chars.next();
    let plus = chars.next();
    if chars.next().is_some() {
        return Err(format!("invalid range token {token:?}"));
    }
    let (suitedness, has_plus) = match (suffix, plus) {
        (None, None) => (Suitedness::Both, false),
        (Some('+'), None) => (Suitedness::Both, true),
        (Some('s' | 'S'), None) => (Suitedness::Suited, false),
        (Some('o' | 'O'), None) => (Suitedness::Offsuit, false),
        (Some('s' | 'S'), Some('+')) => (Suitedness::Suited, true),
        (Some('o' | 'O'), Some('+')) => (Suitedness::Offsuit, true),
        _ => return Err(format!("invalid range token {token:?}")),
    };
    if high == low {
        if suitedness != Suitedness::Both {
            return Err(format!(
                "pocket pair token cannot use suitedness: {token:?}"
            ));
        }
        for rank in pair_ranks(high, has_plus) {
            expand_pair(rank, weight, combos);
        }
    } else {
        if rank_index(high) < rank_index(low) {
            return Err(format!(
                "range token must list the higher rank first, got {token:?}"
            ));
        }
        for kicker in broadway_kickers(high, low, has_plus) {
            expand_unpaired(high, kicker, suitedness, weight, combos);
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Suitedness {
    Both,
    Suited,
    Offsuit,
}

fn parse_rank(value: char) -> Option<Rank> {
    match value {
        '2' => Some(Rank::Two),
        '3' => Some(Rank::Three),
        '4' => Some(Rank::Four),
        '5' => Some(Rank::Five),
        '6' => Some(Rank::Six),
        '7' => Some(Rank::Seven),
        '8' => Some(Rank::Eight),
        '9' => Some(Rank::Nine),
        'T' | 't' => Some(Rank::Ten),
        'J' | 'j' => Some(Rank::Jack),
        'Q' | 'q' => Some(Rank::Queen),
        'K' | 'k' => Some(Rank::King),
        'A' | 'a' => Some(Rank::Ace),
        _ => None,
    }
}

fn rank_index(rank: Rank) -> usize {
    rank.index()
}

fn rank_from_index(index: usize) -> Rank {
    Rank::all()[index]
}

fn pair_ranks(rank: Rank, has_plus: bool) -> Vec<Rank> {
    if has_plus {
        (rank_index(rank)..=rank_index(Rank::Ace))
            .map(rank_from_index)
            .collect()
    } else {
        vec![rank]
    }
}

fn broadway_kickers(high: Rank, low: Rank, has_plus: bool) -> Vec<Rank> {
    if has_plus {
        (rank_index(low)..rank_index(high))
            .map(rank_from_index)
            .collect()
    } else {
        vec![low]
    }
}

fn expand_pair(rank: Rank, weight: f32, combos: &mut Vec<ComboWeight>) {
    let suits = Suit::all();
    for i in 0..suits.len() {
        for j in i + 1..suits.len() {
            push_combo(
                combos,
                Card {
                    rank,
                    suit: suits[i],
                },
                Card {
                    rank,
                    suit: suits[j],
                },
                weight,
            );
        }
    }
}

fn expand_unpaired(
    high: Rank,
    low: Rank,
    suitedness: Suitedness,
    weight: f32,
    combos: &mut Vec<ComboWeight>,
) {
    let suits = Suit::all();
    for high_suit in suits {
        for low_suit in suits {
            let suited = high_suit == low_suit;
            if suitedness == Suitedness::Suited && !suited {
                continue;
            }
            if suitedness == Suitedness::Offsuit && suited {
                continue;
            }
            push_combo(
                combos,
                Card {
                    rank: high,
                    suit: high_suit,
                },
                Card {
                    rank: low,
                    suit: low_suit,
                },
                weight,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_traditional_pair_plus_range() {
        let range = RangeSpec::from_str("99+").unwrap();
        assert_eq!(range.combos().len(), 6 * 6);
    }

    #[test]
    fn parses_traditional_suited_plus_range() {
        let range = RangeSpec::from_str("A2s+").unwrap();
        assert_eq!(range.combos().len(), 12 * 4);
    }

    #[test]
    fn parses_traditional_offsuit_plus_range() {
        let range = RangeSpec::from_str("ATo+").unwrap();
        assert_eq!(range.combos().len(), 4 * 12);
    }

    #[test]
    fn parses_mixed_exact_and_traditional_ranges_with_weights() {
        let range = RangeSpec::from_str("AhAd,99+:0.5,AKs").unwrap();
        assert_eq!(range.combos().len(), 6 * 6 + 4);
        assert!(
            range
                .combos()
                .iter()
                .any(|combo| combo.first == Card::from_str("9c").unwrap()
                    && combo.second == Card::from_str("9d").unwrap()
                    && combo.weight == 0.5)
        );
    }

    #[test]
    fn earlier_tokens_keep_weight_when_later_tokens_overlap() {
        let range = RangeSpec::from_str("TT+,88+:0.7").unwrap();
        assert_eq!(range.combos().len(), 7 * 6);
        assert!(
            range
                .combos()
                .iter()
                .any(|combo| combo.first == Card::from_str("Tc").unwrap()
                    && combo.second == Card::from_str("Td").unwrap()
                    && combo.weight == 1.0)
        );
        assert!(
            range
                .combos()
                .iter()
                .any(|combo| combo.first == Card::from_str("8c").unwrap()
                    && combo.second == Card::from_str("8d").unwrap()
                    && combo.weight == 0.7)
        );
    }
}
