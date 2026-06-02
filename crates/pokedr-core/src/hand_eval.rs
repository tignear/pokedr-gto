use crate::cards::Card;

pub type HandValue = u32;

pub fn evaluate_seven(cards: [Card; 7]) -> HandValue {
    let mut best = 0;

    for a in 0..3 {
        for b in (a + 1)..4 {
            for c in (b + 1)..5 {
                for d in (c + 1)..6 {
                    for e in (d + 1)..7 {
                        best = best.max(evaluate_five([
                            cards[a], cards[b], cards[c], cards[d], cards[e],
                        ]));
                    }
                }
            }
        }
    }

    best
}

pub fn evaluate_five(cards: [Card; 5]) -> HandValue {
    let mut rank_counts = [0_u8; 15];
    let mut suit_counts = [0_u8; 4];
    let mut rank_mask = 0_u16;

    for card in cards {
        rank_counts[card.rank() as usize] += 1;
        suit_counts[card.suit() as usize] += 1;
        rank_mask |= rank_bit(card.rank());
    }

    let is_flush = suit_counts.iter().any(|&count| count == 5);
    let straight_high = straight_high(rank_mask);

    if is_flush && straight_high > 0 {
        return encode(8, &[straight_high]);
    }

    let groups = rank_groups(&rank_counts);

    if groups[0].1 == 4 {
        return encode(7, &[groups[0].0, groups[1].0]);
    }

    if groups[0].1 == 3 && groups[1].1 == 2 {
        return encode(6, &[groups[0].0, groups[1].0]);
    }

    if is_flush {
        return encode(5, &ranks_desc(&rank_counts));
    }

    if straight_high > 0 {
        return encode(4, &[straight_high]);
    }

    if groups[0].1 == 3 {
        return encode(3, &[groups[0].0, groups[1].0, groups[2].0]);
    }

    if groups[0].1 == 2 && groups[1].1 == 2 {
        return encode(2, &[groups[0].0, groups[1].0, groups[2].0]);
    }

    if groups[0].1 == 2 {
        return encode(1, &[groups[0].0, groups[1].0, groups[2].0, groups[3].0]);
    }

    encode(0, &ranks_desc(&rank_counts))
}

fn straight_high(rank_mask: u16) -> u8 {
    for high in (6..=14).rev() {
        let pattern = 0b1_1111_u16 << (high - 6);
        if rank_mask & pattern == pattern {
            return high;
        }
    }

    let wheel = rank_bit(14) | rank_bit(5) | rank_bit(4) | rank_bit(3) | rank_bit(2);
    if rank_mask & wheel == wheel {
        return 5;
    }

    0
}

fn rank_bit(rank: u8) -> u16 {
    1_u16 << (rank - 2)
}

fn rank_groups(rank_counts: &[u8; 15]) -> Vec<(u8, u8)> {
    let mut groups = Vec::new();

    for rank in (2..=14).rev() {
        let count = rank_counts[rank as usize];
        if count > 0 {
            groups.push((rank, count));
        }
    }

    groups.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| right.0.cmp(&left.0)));
    groups
}

fn ranks_desc(rank_counts: &[u8; 15]) -> Vec<u8> {
    let mut ranks = Vec::new();

    for rank in (2..=14).rev() {
        for _ in 0..rank_counts[rank as usize] {
            ranks.push(rank);
        }
    }

    ranks
}

fn encode(category: u8, ranks: &[u8]) -> HandValue {
    let mut value = (category as u32) << 20;

    for (index, rank) in ranks.iter().enumerate() {
        value |= (*rank as u32) << (16 - index * 4);
    }

    value
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(rank: u8, suit: u8) -> Card {
        Card::new(rank, suit)
    }

    #[test]
    fn hand_categories_order_correctly() {
        let straight_flush = evaluate_five([c(14, 0), c(13, 0), c(12, 0), c(11, 0), c(10, 0)]);
        let quads = evaluate_five([c(9, 0), c(9, 1), c(9, 2), c(9, 3), c(2, 0)]);
        let full_house = evaluate_five([c(8, 0), c(8, 1), c(8, 2), c(3, 0), c(3, 1)]);
        let flush = evaluate_five([c(14, 1), c(9, 1), c(7, 1), c(5, 1), c(2, 1)]);
        let straight = evaluate_five([c(5, 0), c(4, 1), c(3, 2), c(2, 3), c(14, 0)]);

        assert!(straight_flush > quads);
        assert!(quads > full_house);
        assert!(full_house > flush);
        assert!(flush > straight);
    }
}
