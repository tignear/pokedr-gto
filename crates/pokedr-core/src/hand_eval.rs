use crate::cards::Card;

pub type HandValue = u32;

pub fn evaluate_seven(cards: [Card; 7]) -> HandValue {
    evaluate_cards(&cards)
}

pub fn evaluate_five(cards: [Card; 5]) -> HandValue {
    evaluate_cards(&cards)
}

fn evaluate_cards(cards: &[Card]) -> HandValue {
    let mut rank_counts = [0_u8; 15];
    let mut suit_counts = [0_u8; 4];
    let mut rank_mask = 0_u16;
    let mut suit_rank_masks = [0_u16; 4];

    for &card in cards {
        rank_counts[card.rank() as usize] += 1;
        suit_counts[card.suit() as usize] += 1;
        rank_mask |= rank_bit(card.rank());
        suit_rank_masks[card.suit() as usize] |= rank_bit(card.rank());
    }

    if let Some(flush_mask) = flush_mask(&suit_counts, &suit_rank_masks) {
        let straight_flush_high = straight_high(flush_mask);
        if straight_flush_high > 0 {
            return encode(8, &[straight_flush_high]);
        }
    }

    let straight_high = straight_high(rank_mask);

    if let Some(four_rank) = highest_rank_with_count(&rank_counts, 4) {
        return encode(
            7,
            &[four_rank, top_ranks(rank_mask, 1, rank_bit(four_rank))[0]],
        );
    }

    if let Some(trips_rank) = highest_rank_with_count(&rank_counts, 3) {
        if let Some(full_house_rank) = highest_full_house_pair_rank(&rank_counts, trips_rank) {
            return encode(6, &[trips_rank, full_house_rank]);
        }
    }

    if let Some(flush_mask) = flush_mask(&suit_counts, &suit_rank_masks) {
        return encode(5, &top_ranks(flush_mask, 5, 0));
    }

    if straight_high > 0 {
        return encode(4, &[straight_high]);
    }

    if let Some(trips_rank) = highest_rank_with_count(&rank_counts, 3) {
        let kickers = top_ranks(rank_mask, 2, rank_bit(trips_rank));
        return encode(3, &[trips_rank, kickers[0], kickers[1]]);
    }

    let pairs = ranks_with_count(&rank_counts, 2);
    if pairs.len() >= 2 {
        let kicker = top_ranks(rank_mask, 1, rank_bit(pairs[0]) | rank_bit(pairs[1]))[0];
        return encode(2, &[pairs[0], pairs[1], kicker]);
    }

    if pairs.len() == 1 {
        let kickers = top_ranks(rank_mask, 3, rank_bit(pairs[0]));
        return encode(1, &[pairs[0], kickers[0], kickers[1], kickers[2]]);
    }

    encode(0, &top_ranks(rank_mask, 5, 0))
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

fn flush_mask(suit_counts: &[u8; 4], suit_rank_masks: &[u16; 4]) -> Option<u16> {
    (0..4)
        .filter(|&suit| suit_counts[suit] >= 5)
        .map(|suit| suit_rank_masks[suit])
        .max_by_key(|&mask| top_ranks(mask, 5, 0))
}

fn highest_rank_with_count(rank_counts: &[u8; 15], target: u8) -> Option<u8> {
    (2..=14)
        .rev()
        .find(|&rank| rank_counts[rank as usize] == target)
}

fn highest_full_house_pair_rank(rank_counts: &[u8; 15], trips_rank: u8) -> Option<u8> {
    (2..=14)
        .rev()
        .find(|&rank| rank != trips_rank && rank_counts[rank as usize] >= 2)
}

fn ranks_with_count(rank_counts: &[u8; 15], target: u8) -> Vec<u8> {
    (2..=14)
        .rev()
        .filter(|&rank| rank_counts[rank as usize] == target)
        .collect()
}

fn top_ranks(rank_mask: u16, count: usize, excluded_mask: u16) -> Vec<u8> {
    let mut ranks = Vec::with_capacity(count);
    let available = rank_mask & !excluded_mask;

    for rank in (2..=14).rev() {
        if available & rank_bit(rank) != 0 {
            ranks.push(rank);
            if ranks.len() == count {
                break;
            }
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

    #[test]
    fn seven_card_evaluator_uses_best_five_cards() {
        let value = evaluate_seven([
            c(14, 0),
            c(14, 1),
            c(14, 2),
            c(13, 0),
            c(13, 1),
            c(2, 0),
            c(3, 0),
        ]);

        assert_eq!(value, encode(6, &[14, 13]));
    }

    #[test]
    fn seven_card_evaluator_finds_flush_and_straight_flush() {
        let flush = evaluate_seven([
            c(14, 1),
            c(11, 1),
            c(9, 1),
            c(7, 1),
            c(3, 1),
            c(2, 0),
            c(4, 2),
        ]);
        let straight_flush = evaluate_seven([
            c(9, 2),
            c(8, 2),
            c(7, 2),
            c(6, 2),
            c(5, 2),
            c(14, 0),
            c(14, 1),
        ]);

        assert!(straight_flush > flush);
    }
}
