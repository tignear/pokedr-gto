#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Card(u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Suit {
    Clubs = 0,
    Diamonds = 1,
    Hearts = 2,
    Spades = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rank {
    Two = 0,
    Three = 1,
    Four = 2,
    Five = 3,
    Six = 4,
    Seven = 5,
    Eight = 6,
    Nine = 7,
    Ten = 8,
    Jack = 9,
    Queen = 10,
    King = 11,
    Ace = 12,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    cards: Vec<Card>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandBits {
    pub rank_mask: u16,
    pub suit_masks: [u16; 4],
    pub rank_counts: [u8; 13],
}

impl Card {
    pub const COUNT: usize = 52;

    pub fn new(rank: Rank, suit: Suit) -> Self {
        Self((suit as u8) * 13 + rank as u8)
    }

    pub fn from_index(index: u8) -> Self {
        assert!(index < Self::COUNT as u8, "card index out of range");
        Self(index)
    }

    pub fn index(self) -> u8 {
        self.0
    }

    pub fn rank(self) -> Rank {
        Rank::from_index(self.0 % 13)
    }

    pub fn suit(self) -> Suit {
        Suit::from_index(self.0 / 13)
    }

    pub fn deck_mask(self) -> u64 {
        1u64 << self.0
    }

    pub fn rank_mask(self) -> u16 {
        self.rank().mask()
    }
}

impl Suit {
    pub fn from_index(index: u8) -> Self {
        match index {
            0 => Self::Clubs,
            1 => Self::Diamonds,
            2 => Self::Hearts,
            3 => Self::Spades,
            _ => panic!("suit index out of range"),
        }
    }
}

impl Rank {
    pub fn from_index(index: u8) -> Self {
        match index {
            0 => Self::Two,
            1 => Self::Three,
            2 => Self::Four,
            3 => Self::Five,
            4 => Self::Six,
            5 => Self::Seven,
            6 => Self::Eight,
            7 => Self::Nine,
            8 => Self::Ten,
            9 => Self::Jack,
            10 => Self::Queen,
            11 => Self::King,
            12 => Self::Ace,
            _ => panic!("rank index out of range"),
        }
    }

    pub fn mask(self) -> u16 {
        match self {
            Self::Ace => (1 << 0) | (1 << 13),
            _ => 1 << (self as u16 + 1),
        }
    }

    pub fn count_index(self) -> usize {
        self as usize
    }
}

impl Board {
    pub fn new(cards: Vec<Card>) -> Self {
        assert!(cards.len() <= 5, "board can contain at most five cards");
        let mut seen = 0u64;
        for card in &cards {
            let bit = card.deck_mask();
            assert!(seen & bit == 0, "duplicate board card");
            seen |= bit;
        }
        Self { cards }
    }

    pub fn cards(&self) -> &[Card] {
        &self.cards
    }

    pub fn deck_mask(&self) -> u64 {
        self.cards
            .iter()
            .fold(0, |mask, card| mask | card.deck_mask())
    }

    pub fn hand_bits(&self) -> HandBits {
        HandBits::from_cards(&self.cards)
    }

    pub fn with_card(&self, card: Card) -> Self {
        assert!(
            self.deck_mask() & card.deck_mask() == 0,
            "board card collision"
        );
        let mut cards = self.cards.clone();
        cards.push(card);
        Self::new(cards)
    }
}

impl HandBits {
    pub fn empty() -> Self {
        Self {
            rank_mask: 0,
            suit_masks: [0; 4],
            rank_counts: [0; 13],
        }
    }

    pub fn from_cards(cards: &[Card]) -> Self {
        let mut bits = Self::empty();
        for card in cards {
            bits.add(*card);
        }
        bits
    }

    pub fn add(&mut self, card: Card) {
        let rank_mask = card.rank_mask();
        self.rank_mask |= rank_mask;
        self.suit_masks[card.suit() as usize] |= rank_mask;
        self.rank_counts[card.rank().count_index()] += 1;
    }

    pub fn has_flush(&self) -> bool {
        self.suit_masks.iter().any(|mask| mask.count_ones() >= 5)
    }

    pub fn straight_high_bit(&self) -> Option<u8> {
        straight_high_bit(self.rank_mask)
    }
}

pub fn straight_high_bit(rank_mask: u16) -> Option<u8> {
    for high in (4..=13).rev() {
        let window = 0b1_1111u16 << (high - 4);
        if rank_mask & window == window {
            return Some(high);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ace_sets_low_and_high_rank_bits() {
        assert_eq!(Rank::Ace.mask(), (1 << 0) | (1 << 13));
    }

    #[test]
    fn wheel_straight_is_detected_with_ace_low_bit() {
        let cards = [
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Two, Suit::Clubs),
            Card::new(Rank::Three, Suit::Diamonds),
            Card::new(Rank::Four, Suit::Hearts),
            Card::new(Rank::Five, Suit::Spades),
        ];
        assert_eq!(HandBits::from_cards(&cards).straight_high_bit(), Some(4));
    }

    #[test]
    fn broadway_straight_prefers_ace_high() {
        let cards = [
            Card::new(Rank::Ten, Suit::Spades),
            Card::new(Rank::Jack, Suit::Clubs),
            Card::new(Rank::Queen, Suit::Diamonds),
            Card::new(Rank::King, Suit::Hearts),
            Card::new(Rank::Ace, Suit::Spades),
        ];
        assert_eq!(HandBits::from_cards(&cards).straight_high_bit(), Some(13));
    }

    #[test]
    fn suit_masks_detect_flush() {
        let cards = [
            Card::new(Rank::Two, Suit::Hearts),
            Card::new(Rank::Four, Suit::Hearts),
            Card::new(Rank::Six, Suit::Hearts),
            Card::new(Rank::Eight, Suit::Hearts),
            Card::new(Rank::Ten, Suit::Hearts),
        ];
        assert!(HandBits::from_cards(&cards).has_flush());
    }
}
