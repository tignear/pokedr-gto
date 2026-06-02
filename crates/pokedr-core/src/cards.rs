#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Card(pub u8);

impl Card {
    pub fn new(rank: u8, suit: u8) -> Self {
        debug_assert!((2..=14).contains(&rank));
        debug_assert!(suit < 4);
        Self((rank - 2) * 4 + suit)
    }

    pub fn rank(self) -> u8 {
        self.0 / 4 + 2
    }

    pub fn suit(self) -> u8 {
        self.0 % 4
    }

    pub fn mask(self) -> u64 {
        1_u64 << self.0
    }
}

pub fn deck() -> [Card; 52] {
    let mut cards = [Card(0); 52];
    let mut index = 0;

    for rank in 2..=14 {
        for suit in 0..4 {
            cards[index] = Card::new(rank, suit);
            index += 1;
        }
    }

    cards
}
