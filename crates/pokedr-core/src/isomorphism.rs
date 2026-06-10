use crate::cards::{Board, Card, Suit};
use crate::range::RangeSpec;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SuitPermutation {
    map: [Suit; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComboSwap {
    pub left: usize,
    pub right: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChanceClassMember {
    pub concrete: Vec<Card>,
    pub permutation_to_representative: SuitPermutation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChanceClass {
    pub representative: Vec<Card>,
    pub multiplicity: usize,
    pub members: Vec<ChanceClassMember>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NextCardIsomorphism {
    pub public_board: Board,
    pub concrete_events: usize,
    pub classes: Vec<ChanceClass>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FutureBoardIsomorphismReport {
    pub flop: Board,
    pub valid_permutations: usize,
    pub turn: NextCardIsomorphism,
    pub representative_turn_river_classes: Vec<NextCardIsomorphism>,
    pub ordered_turn_river_concrete_events: usize,
    pub ordered_turn_river_representative_events: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FutureBoardIsomorphismSurvey {
    pub flops: usize,
    pub ordered_turn_river_concrete_events_per_flop: usize,
    pub min_representative_events: usize,
    pub max_representative_events: usize,
    pub average_representative_events: f64,
    pub average_eliminated_events: f64,
    pub average_eliminated_fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalBoardClassMember {
    pub concrete_board: Board,
    pub permutation_to_representative: SuitPermutation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalBoardClass {
    pub representative_board: Board,
    pub multiplicity: usize,
    pub members: Vec<TerminalBoardClassMember>,
}

impl SuitPermutation {
    pub fn identity() -> Self {
        Self {
            map: [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades],
        }
    }

    pub fn apply_card(self, card: Card) -> Card {
        Card {
            rank: card.rank,
            suit: self.map[card.suit.index()],
        }
    }

    pub fn inverse(self) -> Self {
        let mut inverse = [Suit::Clubs; 4];
        for (source, target) in self.map.iter().enumerate() {
            inverse[target.index()] = Suit::all()[source];
        }
        Self { map: inverse }
    }

    pub fn code(self) -> u8 {
        self.map
            .iter()
            .fold(0u8, |code, suit| code * 4 + suit.index() as u8)
    }
}

pub fn all_suit_permutations() -> Vec<SuitPermutation> {
    let suits = Suit::all();
    let mut permutations = Vec::with_capacity(24);
    for a in suits {
        for b in suits {
            if b == a {
                continue;
            }
            for c in suits {
                if c == a || c == b {
                    continue;
                }
                for d in suits {
                    if d == a || d == b || d == c {
                        continue;
                    }
                    permutations.push(SuitPermutation { map: [a, b, c, d] });
                }
            }
        }
    }
    permutations
}

pub fn ranges_preserve_all_suit_permutations(oop_range: &RangeSpec, ip_range: &RangeSpec) -> bool {
    suit_permutations_preserving_ranges(oop_range, ip_range).len() == all_suit_permutations().len()
}

pub fn suit_permutations_preserving_ranges(
    oop_range: &RangeSpec,
    ip_range: &RangeSpec,
) -> Vec<SuitPermutation> {
    all_suit_permutations()
        .into_iter()
        .filter(|permutation| preserves_range(*permutation, oop_range))
        .filter(|permutation| preserves_range(*permutation, ip_range))
        .collect()
}

pub fn private_combo_permutation_indices(
    combos: &[crate::range::ComboWeight],
    permutation: SuitPermutation,
) -> Option<Vec<usize>> {
    let mut index_by_pair = HashMap::with_capacity(combos.len());
    for (index, combo) in combos.iter().enumerate() {
        index_by_pair.insert(card_pair_index(combo.first, combo.second), index);
    }
    combos
        .iter()
        .map(|combo| {
            let first = permutation.apply_card(combo.first);
            let second = permutation.apply_card(combo.second);
            index_by_pair.get(&card_pair_index(first, second)).copied()
        })
        .collect()
}

pub fn private_combo_swap_list(
    combos: &[crate::range::ComboWeight],
    permutation: SuitPermutation,
) -> Vec<ComboSwap> {
    let mut index_by_pair = HashMap::with_capacity(combos.len());
    for (index, combo) in combos.iter().enumerate() {
        index_by_pair.insert(card_pair_index(combo.first, combo.second), index);
    }
    let mut swaps = Vec::new();
    for (index, combo) in combos.iter().enumerate() {
        let first = permutation.apply_card(combo.first);
        let second = permutation.apply_card(combo.second);
        let Some(&target) = index_by_pair.get(&card_pair_index(first, second)) else {
            continue;
        };
        if index < target {
            swaps.push(ComboSwap {
                left: index,
                right: target,
            });
        }
    }
    swaps
}

pub fn terminal_board_isomorphism(
    public_board: &Board,
    oop_range: &RangeSpec,
    ip_range: &RangeSpec,
) -> Result<Vec<TerminalBoardClass>, String> {
    let range_permutations = suit_permutations_preserving_ranges(oop_range, ip_range);
    let permutations = valid_permutations_for_public_board(public_board, &range_permutations);
    let deck = public_board.remaining_deck();
    let mut classes = HashMap::<Vec<usize>, TerminalBoardClass>::new();
    match public_board.cards().len() {
        5 => {
            let key = card_indices_sorted(public_board.cards());
            classes.insert(
                key,
                TerminalBoardClass {
                    representative_board: public_board.clone(),
                    multiplicity: 1,
                    members: vec![TerminalBoardClassMember {
                        concrete_board: public_board.clone(),
                        permutation_to_representative: SuitPermutation::identity(),
                    }],
                },
            );
        }
        4 => {
            for river in deck {
                push_terminal_board_class(public_board, &[river], &permutations, &mut classes)?;
            }
        }
        3 => {
            for turn in 0..deck.len() {
                for river in turn + 1..deck.len() {
                    push_terminal_board_class(
                        public_board,
                        &[deck[turn], deck[river]],
                        &permutations,
                        &mut classes,
                    )?;
                }
            }
        }
        other => return Err(format!("terminal board has invalid length {other}")),
    }
    let mut classes = classes.into_values().collect::<Vec<_>>();
    classes.sort_by_key(|class| card_indices_sorted(class.representative_board.cards()));
    Ok(classes)
}

pub fn fixed_flop_future_board_isomorphism(
    flop: &Board,
    oop_range: &RangeSpec,
    ip_range: &RangeSpec,
) -> Result<FutureBoardIsomorphismReport, String> {
    if flop.cards().len() != 3 {
        return Err("future board isomorphism report requires a three-card flop".to_string());
    }
    let range_permutations = suit_permutations_preserving_ranges(oop_range, ip_range);
    fixed_flop_future_board_isomorphism_with_range_permutations(flop, &range_permutations)
}

pub fn full_deck_future_board_isomorphism_survey(
    oop_range: &RangeSpec,
    ip_range: &RangeSpec,
) -> Result<FutureBoardIsomorphismSurvey, String> {
    let range_permutations = suit_permutations_preserving_ranges(oop_range, ip_range);
    let deck = Card::deck();
    let mut flops = 0usize;
    let mut min_representative_events = usize::MAX;
    let mut max_representative_events = 0usize;
    let mut total_representative_events = 0usize;
    for first in 0..deck.len() {
        for second in first + 1..deck.len() {
            for third in second + 1..deck.len() {
                let flop = Board::new(vec![deck[first], deck[second], deck[third]])?;
                let report = fixed_flop_future_board_isomorphism_with_range_permutations(
                    &flop,
                    &range_permutations,
                )?;
                flops += 1;
                min_representative_events =
                    min_representative_events.min(report.ordered_turn_river_representative_events);
                max_representative_events =
                    max_representative_events.max(report.ordered_turn_river_representative_events);
                total_representative_events += report.ordered_turn_river_representative_events;
            }
        }
    }
    let concrete = 49 * 48;
    let average_representative_events = total_representative_events as f64 / flops as f64;
    let average_eliminated_events = concrete as f64 - average_representative_events;
    Ok(FutureBoardIsomorphismSurvey {
        flops,
        ordered_turn_river_concrete_events_per_flop: concrete,
        min_representative_events,
        max_representative_events,
        average_representative_events,
        average_eliminated_events,
        average_eliminated_fraction: average_eliminated_events / concrete as f64,
    })
}

fn fixed_flop_future_board_isomorphism_with_range_permutations(
    flop: &Board,
    range_permutations: &[SuitPermutation],
) -> Result<FutureBoardIsomorphismReport, String> {
    if flop.cards().len() != 3 {
        return Err("future board isomorphism report requires a three-card flop".to_string());
    }
    let valid_permutations = valid_permutations_for_public_board(flop, range_permutations);
    let turn = next_card_isomorphism_with_range_permutations(flop, range_permutations);
    let mut representative_turn_river_classes = Vec::with_capacity(turn.classes.len());
    let mut ordered_turn_river_representative_events = 0usize;
    for turn_class in &turn.classes {
        let turn_card = *turn_class
            .representative
            .first()
            .ok_or_else(|| "turn class has no representative card".to_string())?;
        let turn_board = flop.push(turn_card)?;
        let river = next_card_isomorphism_with_range_permutations(&turn_board, range_permutations);
        ordered_turn_river_representative_events += turn_class.multiplicity * river.classes.len();
        representative_turn_river_classes.push(river);
    }

    Ok(FutureBoardIsomorphismReport {
        flop: flop.clone(),
        valid_permutations: valid_permutations.len(),
        turn,
        representative_turn_river_classes,
        ordered_turn_river_concrete_events: 49 * 48,
        ordered_turn_river_representative_events,
    })
}

pub fn next_card_isomorphism(
    public_board: &Board,
    oop_range: &RangeSpec,
    ip_range: &RangeSpec,
) -> NextCardIsomorphism {
    let range_permutations = suit_permutations_preserving_ranges(oop_range, ip_range);
    next_card_isomorphism_with_range_permutations(public_board, &range_permutations)
}

pub fn next_card_isomorphism_with_permutations(
    public_board: &Board,
    range_permutations: &[SuitPermutation],
) -> NextCardIsomorphism {
    next_card_isomorphism_with_range_permutations(public_board, range_permutations)
}

fn next_card_isomorphism_with_range_permutations(
    public_board: &Board,
    range_permutations: &[SuitPermutation],
) -> NextCardIsomorphism {
    let permutations = valid_permutations_for_public_board(public_board, range_permutations);
    let mut counts = HashMap::<Vec<usize>, usize>::new();
    let mut members = HashMap::<Vec<usize>, Vec<ChanceClassMember>>::new();
    for card in public_board.remaining_deck() {
        let (representative, permutation) =
            canonical_future_cards(public_board, &[card], &permutations);
        let representative = card_indices(&representative);
        *counts.entry(representative.clone()).or_insert(0) += 1;
        members
            .entry(representative)
            .or_default()
            .push(ChanceClassMember {
                concrete: vec![card],
                permutation_to_representative: permutation,
            });
    }
    let mut classes = counts
        .into_iter()
        .map(|(representative, multiplicity)| ChanceClass {
            members: members.remove(&representative).unwrap_or_default(),
            representative: representative
                .into_iter()
                .map(index_to_card)
                .collect::<Vec<_>>(),
            multiplicity,
        })
        .collect::<Vec<_>>();
    classes.sort_by_key(|class| card_indices(&class.representative));
    NextCardIsomorphism {
        public_board: public_board.clone(),
        concrete_events: public_board.remaining_deck().len(),
        classes,
    }
}

fn valid_permutations_for_public_board(
    board: &Board,
    range_permutations: &[SuitPermutation],
) -> Vec<SuitPermutation> {
    range_permutations
        .iter()
        .copied()
        .filter(|permutation| preserves_public_board(*permutation, board))
        .collect()
}

fn canonical_future_cards(
    public_board: &Board,
    future_cards: &[Card],
    permutations: &[SuitPermutation],
) -> (Vec<Card>, SuitPermutation) {
    let mut best = card_indices_sorted(future_cards);
    let mut best_permutation = SuitPermutation::identity();
    for permutation in permutations {
        let permuted = future_cards
            .iter()
            .map(|card| permutation.apply_card(*card))
            .collect::<Vec<_>>();
        if permuted.iter().any(|card| public_board.contains(*card)) {
            continue;
        }
        let indices = card_indices_sorted(&permuted);
        if indices < best {
            best = indices;
            best_permutation = *permutation;
        }
    }
    (
        best.into_iter().map(index_to_card).collect(),
        best_permutation,
    )
}

fn push_terminal_board_class(
    public_board: &Board,
    future_cards: &[Card],
    permutations: &[SuitPermutation],
    classes: &mut HashMap<Vec<usize>, TerminalBoardClass>,
) -> Result<(), String> {
    let (representative_future, permutation) =
        canonical_future_cards(public_board, future_cards, permutations);
    let representative_board = board_with_future(public_board, &representative_future)?;
    let concrete_board = board_with_future(public_board, future_cards)?;
    let key = card_indices_sorted(representative_board.cards());
    let entry = classes.entry(key).or_insert_with(|| TerminalBoardClass {
        representative_board,
        multiplicity: 0,
        members: Vec::new(),
    });
    entry.multiplicity += 1;
    entry.members.push(TerminalBoardClassMember {
        concrete_board,
        permutation_to_representative: permutation,
    });
    Ok(())
}

fn board_with_future(public_board: &Board, future_cards: &[Card]) -> Result<Board, String> {
    let mut cards = public_board.cards().to_vec();
    cards.extend_from_slice(future_cards);
    Board::new(cards)
}

fn preserves_public_board(permutation: SuitPermutation, board: &Board) -> bool {
    let mut before = card_indices(board.cards());
    let mut after = board
        .cards()
        .iter()
        .map(|card| permutation.apply_card(*card).index())
        .collect::<Vec<_>>();
    before.sort_unstable();
    after.sort_unstable();
    before == after
}

fn preserves_range(permutation: SuitPermutation, range: &RangeSpec) -> bool {
    let mut weights = [0.0f32; 1326];
    for combo in range.combos() {
        weights[card_pair_index(combo.first, combo.second)] = combo.weight;
    }
    for combo in range.combos() {
        let first = permutation.apply_card(combo.first);
        let second = permutation.apply_card(combo.second);
        if weights[card_pair_index(first, second)].to_bits() != combo.weight.to_bits() {
            return false;
        }
    }
    true
}

fn card_indices(cards: &[Card]) -> Vec<usize> {
    cards.iter().map(|card| card.index()).collect()
}

fn card_indices_sorted(cards: &[Card]) -> Vec<usize> {
    let mut indices = card_indices(cards);
    indices.sort_unstable();
    indices
}

fn index_to_card(index: usize) -> Card {
    let rank = crate::cards::Rank::all()[index / 4];
    let suit = Suit::all()[index % 4];
    Card { rank, suit }
}

fn card_pair_index(first: Card, second: Card) -> usize {
    let first = first.index();
    let second = second.index();
    let (low, high) = if first < second {
        (first, second)
    } else {
        (second, first)
    };
    low * (103 - low) / 2 + high - low - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn monotone_full_ranges_collapse_empty_suits_on_turn() {
        let flop = Board::from_str("AsKsQs").unwrap();
        let range = RangeSpec::full_deck_uniform();
        let report = fixed_flop_future_board_isomorphism(&flop, &range, &range).unwrap();

        assert_eq!(report.turn.concrete_events, 49);
        assert_eq!(report.turn.classes.len(), 23);
        assert_eq!(
            report
                .turn
                .classes
                .iter()
                .map(|class| class.multiplicity)
                .sum::<usize>(),
            49
        );
        assert_eq!(report.ordered_turn_river_concrete_events, 49 * 48);
        assert!(report.ordered_turn_river_representative_events < 49 * 48);
    }

    #[test]
    fn exact_suit_range_reduces_collapse() {
        let flop = Board::from_str("AsKsQs").unwrap();
        let full = RangeSpec::full_deck_uniform();
        let exact = RangeSpec::from_str("AhAd").unwrap();

        let symmetric = fixed_flop_future_board_isomorphism(&flop, &full, &full).unwrap();
        let asymmetric = fixed_flop_future_board_isomorphism(&flop, &exact, &full).unwrap();

        assert!(asymmetric.valid_permutations < symmetric.valid_permutations);
        assert!(asymmetric.turn.classes.len() > symmetric.turn.classes.len());
        assert_eq!(
            asymmetric
                .turn
                .classes
                .iter()
                .map(|class| class.multiplicity)
                .sum::<usize>(),
            49
        );
    }

    #[test]
    fn rainbow_unpaired_flop_has_no_public_suit_symmetry() {
        let flop = Board::from_str("As7h2c").unwrap();
        let range = RangeSpec::full_deck_uniform();
        let report = fixed_flop_future_board_isomorphism(&flop, &range, &range).unwrap();

        assert_eq!(report.valid_permutations, 1);
        assert_eq!(report.turn.classes.len(), 49);
        assert!(report.ordered_turn_river_representative_events < 49 * 48);
    }

    #[test]
    fn paired_flop_collapses_public_suit_symmetry() {
        let flop = Board::from_str("AsAh7c").unwrap();
        let range = RangeSpec::full_deck_uniform();
        let report = fixed_flop_future_board_isomorphism(&flop, &range, &range).unwrap();

        assert!(report.valid_permutations > 1);
        assert_eq!(report.turn.concrete_events, 49);
        assert!(report.turn.classes.len() < 49);
        assert_eq!(
            report
                .turn
                .classes
                .iter()
                .map(|class| class.multiplicity)
                .sum::<usize>(),
            49
        );
        assert_eq!(report.ordered_turn_river_concrete_events, 49 * 48);
        assert!(report.ordered_turn_river_representative_events < 49 * 48);
    }

    #[test]
    fn suit_symmetric_range_check_rejects_exact_suit_combos() {
        let pair_class = RangeSpec::from_str("AA").unwrap();
        let exact = RangeSpec::from_str("AsAh").unwrap();

        assert!(ranges_preserve_all_suit_permutations(
            &pair_class,
            &pair_class
        ));
        assert!(!ranges_preserve_all_suit_permutations(&exact, &pair_class));
    }

    #[test]
    fn chance_class_records_permutation_to_representative() {
        let flop = Board::from_str("AsKsQs").unwrap();
        let range = RangeSpec::full_deck_uniform();
        let report = fixed_flop_future_board_isomorphism(&flop, &range, &range).unwrap();
        let class = report
            .turn
            .classes
            .iter()
            .find(|class| class.multiplicity > 1)
            .unwrap();

        assert_eq!(class.members.len(), class.multiplicity);
        for member in &class.members {
            let mapped = member
                .concrete
                .iter()
                .map(|card| member.permutation_to_representative.apply_card(*card))
                .collect::<Vec<_>>();
            assert_eq!(mapped, class.representative);
        }
    }

    #[test]
    fn private_combo_swap_list_maps_full_deck_combo_indices() {
        let range = RangeSpec::full_deck_uniform();
        let permutation = SuitPermutation {
            map: [Suit::Diamonds, Suit::Clubs, Suit::Hearts, Suit::Spades],
        };
        let swaps = private_combo_swap_list(range.combos(), permutation);

        assert!(!swaps.is_empty());
        for swap in swaps.iter().take(16) {
            let left = range.combos()[swap.left];
            let right = range.combos()[swap.right];
            assert_eq!(permutation.apply_card(left.first), right.first);
            assert_eq!(permutation.apply_card(left.second), right.second);
        }
    }

    #[test]
    fn terminal_board_isomorphism_preserves_concrete_runout_count() {
        let flop = Board::from_str("AsKsQs").unwrap();
        let range = RangeSpec::full_deck_uniform();
        let classes = terminal_board_isomorphism(&flop, &range, &range).unwrap();

        assert!(classes.len() < 49 * 48 / 2);
        assert_eq!(
            classes
                .iter()
                .map(|class| class.multiplicity)
                .sum::<usize>(),
            49 * 48 / 2
        );
        for class in &classes {
            assert_eq!(class.members.len(), class.multiplicity);
            for member in &class.members {
                let mapped = member
                    .concrete_board
                    .cards()
                    .iter()
                    .map(|card| member.permutation_to_representative.apply_card(*card))
                    .collect::<Vec<_>>();
                assert_eq!(
                    card_indices_sorted(&mapped),
                    card_indices_sorted(class.representative_board.cards())
                );
            }
        }
    }

    #[test]
    fn paired_terminal_board_isomorphism_preserves_concrete_runout_count() {
        let flop = Board::from_str("AsAh7c").unwrap();
        let range = RangeSpec::full_deck_uniform();
        let classes = terminal_board_isomorphism(&flop, &range, &range).unwrap();

        assert!(classes.len() < 49 * 48 / 2);
        assert_eq!(
            classes
                .iter()
                .map(|class| class.multiplicity)
                .sum::<usize>(),
            49 * 48 / 2
        );
        assert!(classes.iter().any(|class| class.multiplicity > 1));
    }
}
