use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Suit {
    Clubs,
    Diamonds,
    Hearts,
    Spades,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Rank {
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
    Ace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Card {
    pub rank: Rank,
    pub suit: Suit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    cards: Vec<Card>,
}

impl Suit {
    pub fn all() -> [Self; 4] {
        [Self::Clubs, Self::Diamonds, Self::Hearts, Self::Spades]
    }

    fn parse(value: char) -> Option<Self> {
        match value {
            'c' | 'C' => Some(Self::Clubs),
            'd' | 'D' => Some(Self::Diamonds),
            'h' | 'H' => Some(Self::Hearts),
            's' | 'S' => Some(Self::Spades),
            _ => None,
        }
    }

    fn label(self) -> char {
        match self {
            Self::Clubs => 'c',
            Self::Diamonds => 'd',
            Self::Hearts => 'h',
            Self::Spades => 's',
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::Clubs => 0,
            Self::Diamonds => 1,
            Self::Hearts => 2,
            Self::Spades => 3,
        }
    }
}

impl Rank {
    pub fn all() -> [Self; 13] {
        [
            Self::Two,
            Self::Three,
            Self::Four,
            Self::Five,
            Self::Six,
            Self::Seven,
            Self::Eight,
            Self::Nine,
            Self::Ten,
            Self::Jack,
            Self::Queen,
            Self::King,
            Self::Ace,
        ]
    }

    fn parse(value: char) -> Option<Self> {
        match value {
            '2' => Some(Self::Two),
            '3' => Some(Self::Three),
            '4' => Some(Self::Four),
            '5' => Some(Self::Five),
            '6' => Some(Self::Six),
            '7' => Some(Self::Seven),
            '8' => Some(Self::Eight),
            '9' => Some(Self::Nine),
            'T' | 't' => Some(Self::Ten),
            'J' | 'j' => Some(Self::Jack),
            'Q' | 'q' => Some(Self::Queen),
            'K' | 'k' => Some(Self::King),
            'A' | 'a' => Some(Self::Ace),
            _ => None,
        }
    }

    fn label(self) -> char {
        match self {
            Self::Two => '2',
            Self::Three => '3',
            Self::Four => '4',
            Self::Five => '5',
            Self::Six => '6',
            Self::Seven => '7',
            Self::Eight => '8',
            Self::Nine => '9',
            Self::Ten => 'T',
            Self::Jack => 'J',
            Self::Queen => 'Q',
            Self::King => 'K',
            Self::Ace => 'A',
        }
    }

    pub fn value(self) -> u8 {
        match self {
            Self::Two => 2,
            Self::Three => 3,
            Self::Four => 4,
            Self::Five => 5,
            Self::Six => 6,
            Self::Seven => 7,
            Self::Eight => 8,
            Self::Nine => 9,
            Self::Ten => 10,
            Self::Jack => 11,
            Self::Queen => 12,
            Self::King => 13,
            Self::Ace => 14,
        }
    }

    pub fn index(self) -> usize {
        (self.value() - 2) as usize
    }
}

impl Card {
    pub fn deck() -> Vec<Self> {
        let mut deck = Vec::with_capacity(52);
        for rank in Rank::all() {
            for suit in Suit::all() {
                deck.push(Self { rank, suit });
            }
        }
        deck
    }

    pub fn index(self) -> usize {
        self.rank.index() * 4 + self.suit.index()
    }
}

impl Board {
    pub fn new(cards: Vec<Card>) -> Result<Self, String> {
        if cards.len() > 5 {
            return Err("board cannot contain more than five cards".to_string());
        }
        for i in 0..cards.len() {
            for j in i + 1..cards.len() {
                if cards[i] == cards[j] {
                    return Err(format!("duplicate board card {}", cards[i]));
                }
            }
        }
        Ok(Self { cards })
    }

    pub fn cards(&self) -> &[Card] {
        &self.cards
    }

    pub fn contains(&self, card: Card) -> bool {
        self.cards.contains(&card)
    }

    pub fn push(&self, card: Card) -> Result<Self, String> {
        let mut cards = self.cards.clone();
        cards.push(card);
        Self::new(cards)
    }

    pub fn remaining_deck(&self) -> Vec<Card> {
        Card::deck()
            .into_iter()
            .filter(|card| !self.contains(*card))
            .collect()
    }
}

impl FromStr for Card {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut chars = value.chars();
        let rank = chars
            .next()
            .and_then(Rank::parse)
            .ok_or_else(|| format!("invalid card rank in {value:?}"))?;
        let suit = chars
            .next()
            .and_then(Suit::parse)
            .ok_or_else(|| format!("invalid card suit in {value:?}"))?;
        if chars.next().is_some() {
            return Err(format!("invalid card length for {value:?}"));
        }
        Ok(Self { rank, suit })
    }
}

impl FromStr for Board {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if !value.len().is_multiple_of(2) {
            return Err("board string must have an even number of characters".to_string());
        }
        let cards = value
            .as_bytes()
            .chunks(2)
            .map(|chunk| {
                std::str::from_utf8(chunk)
                    .map_err(|error| error.to_string())
                    .and_then(Card::from_str)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(cards)
    }
}

impl fmt::Display for Card {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(out, "{}{}", self.rank.label(), self.suit.label())
    }
}

impl fmt::Display for Board {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        for card in &self.cards {
            write!(out, "{card}")?;
        }
        Ok(())
    }
}
