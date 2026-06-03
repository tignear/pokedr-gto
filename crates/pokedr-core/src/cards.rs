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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct HandStrength(u32);

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

impl HandStrength {
    pub fn category(self) -> u8 {
        (self.0 >> 20) as u8
    }

    pub fn raw(self) -> u32 {
        self.0
    }
}

pub fn evaluate(cards: &[Card]) -> HandStrength {
    assert!(
        (5..=7).contains(&cards.len()),
        "hand evaluation requires five to seven cards"
    );
    let bits = HandBits::from_cards(cards);
    evaluate_bits(&bits)
}

fn evaluate_bits(bits: &HandBits) -> HandStrength {
    let flush_mask = bits
        .suit_masks
        .iter()
        .copied()
        .find(|mask| mask.count_ones() >= 5);
    if let Some(mask) = flush_mask {
        if let Some(high) = straight_high_bit(mask) {
            return pack(8, &[high]);
        }
    }

    let groups = rank_groups(&bits.rank_counts);
    if let Some(&(four, _)) = groups.iter().find(|(_, count)| *count == 4) {
        let kicker = highest_excluding(bits.rank_mask, &[four], 1);
        return pack(7, &[four, kicker[0]]);
    }

    let trips: Vec<u8> = groups
        .iter()
        .filter_map(|(rank, count)| (*count == 3).then_some(*rank))
        .collect();
    let pairs: Vec<u8> = groups
        .iter()
        .filter_map(|(rank, count)| (*count == 2).then_some(*rank))
        .collect();
    if !trips.is_empty() && (!pairs.is_empty() || trips.len() >= 2) {
        let trip = trips[0];
        let pair = pairs.first().copied().unwrap_or_else(|| trips[1]);
        return pack(6, &[trip, pair]);
    }

    if let Some(mask) = flush_mask {
        let kickers = highest_from_mask(mask, 5);
        return pack(5, &kickers);
    }

    if let Some(high) = bits.straight_high_bit() {
        return pack(4, &[high]);
    }

    if let Some(&trip) = trips.first() {
        let kickers = highest_excluding(bits.rank_mask, &[trip], 2);
        return pack(3, &[trip, kickers[0], kickers[1]]);
    }

    if pairs.len() >= 2 {
        let kicker = highest_excluding(bits.rank_mask, &[pairs[0], pairs[1]], 1);
        return pack(2, &[pairs[0], pairs[1], kicker[0]]);
    }

    if let Some(&pair) = pairs.first() {
        let kickers = highest_excluding(bits.rank_mask, &[pair], 3);
        return pack(1, &[pair, kickers[0], kickers[1], kickers[2]]);
    }

    pack(0, &highest_from_mask(bits.rank_mask, 5))
}

fn rank_groups(rank_counts: &[u8; 13]) -> Vec<(u8, u8)> {
    let mut groups = Vec::new();
    for rank in (0..13).rev() {
        let count = rank_counts[rank];
        if count > 0 {
            groups.push((rank as u8 + 1, count));
        }
    }
    groups
}

fn highest_from_mask(mask: u16, count: usize) -> Vec<u8> {
    let mut ranks = Vec::with_capacity(count);
    for rank in (1..=13).rev() {
        if mask & (1 << rank) != 0 {
            ranks.push(rank as u8);
            if ranks.len() == count {
                break;
            }
        }
    }
    ranks
}

fn highest_excluding(mask: u16, excluded: &[u8], count: usize) -> Vec<u8> {
    let mut ranks = Vec::with_capacity(count);
    for rank in (1..=13).rev() {
        if excluded.contains(&(rank as u8)) {
            continue;
        }
        if mask & (1 << rank) != 0 {
            ranks.push(rank as u8);
            if ranks.len() == count {
                break;
            }
        }
    }
    ranks
}

fn pack(category: u8, ranks: &[u8]) -> HandStrength {
    let mut value = (category as u32) << 20;
    for (index, rank) in ranks.iter().take(5).enumerate() {
        value |= (*rank as u32) << (16 - index * 4);
    }
    HandStrength(value)
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

    #[test]
    fn evaluator_orders_major_hand_classes() {
        let straight_flush = evaluate(&[
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::King, Suit::Spades),
            Card::new(Rank::Queen, Suit::Spades),
            Card::new(Rank::Jack, Suit::Spades),
            Card::new(Rank::Ten, Suit::Spades),
        ]);
        let quads = evaluate(&[
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Ace, Suit::Hearts),
            Card::new(Rank::Ace, Suit::Diamonds),
            Card::new(Rank::Ace, Suit::Clubs),
            Card::new(Rank::King, Suit::Spades),
        ]);
        let full_house = evaluate(&[
            Card::new(Rank::King, Suit::Spades),
            Card::new(Rank::King, Suit::Hearts),
            Card::new(Rank::King, Suit::Diamonds),
            Card::new(Rank::Queen, Suit::Clubs),
            Card::new(Rank::Queen, Suit::Spades),
        ]);

        assert!(straight_flush > quads);
        assert!(quads > full_house);
    }

    #[test]
    fn evaluator_detects_wheel_straight() {
        let wheel = evaluate(&[
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Two, Suit::Clubs),
            Card::new(Rank::Three, Suit::Diamonds),
            Card::new(Rank::Four, Suit::Hearts),
            Card::new(Rank::Five, Suit::Spades),
        ]);

        assert_eq!(wheel.category(), 4);
    }
}
