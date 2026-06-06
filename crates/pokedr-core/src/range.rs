use crate::cards::Card;
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
            if cards.len() != 4 {
                return Err(format!("range combo must be four chars, got {cards:?}"));
            }
            let first = Card::from_str(&cards[0..2])?;
            let second = Card::from_str(&cards[2..4])?;
            combos.push(ComboWeight {
                first,
                second,
                weight,
            });
        }
        Self::new(combos)
    }
}
