use std::{
    collections::BTreeMap,
    sync::mpsc,
    time::{Duration, Instant},
};

use wgpu::util::DeviceExt;

use crate::cards::{Card, evaluate};

use super::{DenseCfrConfig, DenseCfrIteration, DenseCfrRunStats, DenseCfrState};

const WORKGROUP_SIZE: u32 = 64;
const SHOWDOWN_CARDS: usize = 9;

const CFR_UPDATE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> regrets: array<f32>;
@group(0) @binding(1) var<storage, read_write> strategy_sum: array<f32>;
@group(0) @binding(2) var<storage, read> action_values: array<f32>;
@group(0) @binding(3) var<storage, read> reach_weights: array<f32>;
@group(0) @binding(4) var<storage, read> strategy_weights: array<f32>;
@group(0) @binding(5) var<storage, read> params: array<u32>;
@group(0) @binding(6) var<storage, read> legal_actions: array<u32>;
@group(0) @binding(7) var<storage, read_write> prediction: array<f32>;

fn positive(value: f32) -> f32 {
    return max(value, 0.0);
}

fn strategy_at(offset: u32, action: u32, actions: u32, normalizer: f32) -> f32 {
    if legal_actions[offset + action] == 0u {
        return 0.0;
    }
    if normalizer > 0.0 {
        return positive(effective_regret(offset + action)) / normalizer;
    }
    return 1.0 / f32(actions);
}

fn effective_regret(index: u32) -> f32 {
    let variant = params[2];
    if variant == 3u {
        let eta = bitcast<f32>(params[6]);
        return regrets[index] + eta * prediction[index];
    }
    return regrets[index];
}

fn regret_discount(variant: u32, iteration: u32, alpha: f32) -> f32 {
    if variant == 1u {
        let t = f32(max(iteration, 1u));
        return t / (t + 1.0);
    }
    if variant == 2u || variant == 3u || variant == 4u {
        if iteration <= 1u {
            return 0.0;
        }
        let weighted = pow(f32(iteration - 1u), alpha);
        return weighted / (weighted + 1.5);
    }
    return 1.0;
}

fn average_strategy_discount(variant: u32, iteration: u32, gamma: f32) -> f32 {
    if (variant == 2u || variant == 3u || variant == 4u) && iteration > 1u {
        let t = f32(iteration);
        return pow((t - 1.0) / t, gamma);
    }
    return 1.0;
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let infoset = id.x;
    let infosets = params[0];
    let actions = params[1];
    let variant = params[2];
    let iteration = params[3];
    let dcfr_alpha = bitcast<f32>(params[4]);
    let dcfr_gamma = bitcast<f32>(params[5]);
    if infoset >= infosets {
        return;
    }

    let offset = infoset * actions;
    var normalizer = 0.0;
    var legal_count = 0u;
    for (var action = 0u; action < actions; action = action + 1u) {
        if legal_actions[offset + action] != 0u {
            legal_count = legal_count + 1u;
            normalizer = normalizer + positive(effective_regret(offset + action));
        }
    }

    var node_value = 0.0;
    for (var action = 0u; action < actions; action = action + 1u) {
        let strategy = select(
            1.0 / f32(max(legal_count, 1u)),
            strategy_at(offset, action, actions, normalizer),
            normalizer > 0.0
        );
        if legal_actions[offset + action] == 0u {
            continue;
        }
        node_value = node_value + strategy * action_values[offset + action];
    }

    let discount = regret_discount(variant, iteration, dcfr_alpha);
    let average_discount = average_strategy_discount(variant, iteration, dcfr_gamma);

    for (var action = 0u; action < actions; action = action + 1u) {
        if legal_actions[offset + action] == 0u {
            regrets[offset + action] = 0.0;
            if variant == 3u {
                prediction[offset + action] = 0.0;
            }
            strategy_sum[offset + action] = 0.0;
            continue;
        }
        let strategy = select(
            1.0 / f32(max(legal_count, 1u)),
            strategy_at(offset, action, actions, normalizer),
            normalizer > 0.0
        );
        let regret = (action_values[offset + action] - node_value) * reach_weights[infoset];
        var updated = regrets[offset + action] * discount + regret;
        if variant == 0u || variant == 2u || variant == 3u || variant == 4u {
            updated = max(updated, 0.0);
        }
        regrets[offset + action] = updated;
        if variant == 3u {
            prediction[offset + action] = regret;
        }
        strategy_sum[offset + action] = strategy_sum[offset + action] * average_discount + strategy_weights[infoset] * strategy;
    }
}
"#;

const PUBLIC_TREE_CFR_UPDATE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> regrets: array<f32>;
@group(0) @binding(1) var<storage, read_write> strategy_sum: array<f32>;
@group(0) @binding(2) var<storage, read> action_values: array<f32>;
@group(0) @binding(3) var<storage, read> action_weights: array<f32>;
@group(0) @binding(4) var<storage, read> reach_weights: array<f32>;
@group(0) @binding(5) var<storage, read> strategy_weights: array<f32>;
@group(0) @binding(6) var<storage, read> params: array<u32>;
@group(0) @binding(7) var<storage, read> legal_actions: array<u32>;
@group(0) @binding(8) var<storage, read_write> prediction: array<f32>;

fn positive(value: f32) -> f32 {
    return max(value, 0.0);
}

fn strategy_at(offset: u32, action: u32, actions: u32, normalizer: f32) -> f32 {
    if legal_actions[offset + action] == 0u {
        return 0.0;
    }
    if normalizer > 0.0 {
        return positive(effective_regret(offset + action)) / normalizer;
    }
    return 1.0 / f32(actions);
}

fn effective_regret(index: u32) -> f32 {
    let variant = params[2];
    if variant == 3u {
        let eta = bitcast<f32>(params[6]);
        return regrets[index] + eta * prediction[index];
    }
    return regrets[index];
}

fn action_value(action_offset: u32) -> f32 {
    let weight = action_weights[action_offset];
    if weight > 0.0 {
        return action_values[action_offset] / weight;
    }
    return 0.0;
}

fn regret_discount(variant: u32, iteration: u32, alpha: f32) -> f32 {
    if variant == 1u {
        let t = f32(max(iteration, 1u));
        return t / (t + 1.0);
    }
    if variant == 2u || variant == 3u || variant == 4u {
        if iteration <= 1u {
            return 0.0;
        }
        let weighted = pow(f32(iteration - 1u), alpha);
        return weighted / (weighted + 1.5);
    }
    return 1.0;
}

fn average_strategy_discount(variant: u32, iteration: u32, gamma: f32) -> f32 {
    if (variant == 2u || variant == 3u || variant == 4u) && iteration > 1u {
        let t = f32(iteration);
        return pow((t - 1.0) / t, gamma);
    }
    return 1.0;
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let infoset = id.x;
    let infosets = params[0];
    let actions = params[1];
    let variant = params[2];
    let iteration = params[3];
    let dcfr_alpha = bitcast<f32>(params[4]);
    let dcfr_gamma = bitcast<f32>(params[5]);
    if infoset >= infosets {
        return;
    }

    let offset = infoset * actions;
    var normalizer = 0.0;
    var legal_count = 0u;
    for (var action = 0u; action < actions; action = action + 1u) {
        if legal_actions[offset + action] != 0u {
            legal_count = legal_count + 1u;
            normalizer = normalizer + positive(effective_regret(offset + action));
        }
    }

    var node_value = 0.0;
    for (var action = 0u; action < actions; action = action + 1u) {
        if legal_actions[offset + action] == 0u {
            continue;
        }
        let strategy = select(
            1.0 / f32(max(legal_count, 1u)),
            strategy_at(offset, action, actions, normalizer),
            normalizer > 0.0
        );
        node_value = node_value + strategy * action_value(offset + action);
    }

    let discount = regret_discount(variant, iteration, dcfr_alpha);
    let average_discount = average_strategy_discount(variant, iteration, dcfr_gamma);

    let reach_weight = reach_weights[infoset];
    let raw_strategy_weight = strategy_weights[infoset];
    var strategy_weight = raw_strategy_weight * f32(iteration);
    if variant == 2u || variant == 3u || variant == 4u {
        strategy_weight = raw_strategy_weight;
    }
    for (var action = 0u; action < actions; action = action + 1u) {
        if legal_actions[offset + action] == 0u {
            regrets[offset + action] = 0.0;
            if variant == 3u {
                prediction[offset + action] = 0.0;
            }
            strategy_sum[offset + action] = 0.0;
            continue;
        }
        let strategy = select(
            1.0 / f32(max(legal_count, 1u)),
            strategy_at(offset, action, actions, normalizer),
            normalizer > 0.0
        );
        let regret = (action_value(offset + action) - node_value) * reach_weight;
        var updated = regrets[offset + action] * discount + regret;
        if variant == 0u || variant == 2u || variant == 3u || variant == 4u {
            updated = max(updated, 0.0);
        }
        regrets[offset + action] = updated;
        if variant == 3u {
            prediction[offset + action] = regret;
        }
        strategy_sum[offset + action] = strategy_sum[offset + action] * average_discount + strategy_weight * strategy;
    }
}
"#;

const SHOWDOWN_SHADER: &str = r#"
struct ShowdownTask {
    cards: array<u32, 9>,
};

@group(0) @binding(0) var<storage, read> tasks: array<ShowdownTask>;
@group(0) @binding(1) var<storage, read_write> equities: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

fn rank_value(card: u32) -> u32 {
    return (card % 13u) + 1u;
}

fn suit_value(card: u32) -> u32 {
    return card / 13u;
}

fn rank_bit(card: u32) -> u32 {
    let rank = card % 13u;
    if rank == 12u {
        return (1u << 0u) | (1u << 13u);
    }
    return 1u << (rank + 1u);
}

fn pack(category: u32, a: u32, b: u32, c: u32, d: u32, e: u32) -> u32 {
    return (category << 20u)
        | (a << 16u)
        | (b << 12u)
        | (c << 8u)
        | (d << 4u)
        | e;
}

fn straight_high(rank_mask: u32) -> u32 {
    var high = 13u;
    loop {
        let window = 31u << (high - 4u);
        if (rank_mask & window) == window {
            return high;
        }
        if high == 4u {
            break;
        }
        high = high - 1u;
    }
    return 0u;
}

fn highest_from_mask(mask: u32, skip_a: u32, skip_b: u32, take: u32, slot: u32) -> u32 {
    var seen = 0u;
    var rank = 13u;
    loop {
        if rank != skip_a && rank != skip_b && (mask & (1u << rank)) != 0u {
            if seen == slot {
                return rank;
            }
            seen = seen + 1u;
            if seen >= take {
                return 0u;
            }
        }
        if rank == 1u {
            break;
        }
        rank = rank - 1u;
    }
    return 0u;
}

fn evaluate_task(task: ShowdownTask, hero: bool) -> u32 {
    var rank_mask = 0u;
    var suit_masks = array<u32, 4>(0u, 0u, 0u, 0u);
    var counts = array<u32, 13>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);

    for (var i = 0u; i < 7u; i = i + 1u) {
        var card = task.cards[i + 2u];
        if i == 0u {
            card = select(task.cards[2], task.cards[0], hero);
        }
        if i == 1u {
            card = select(task.cards[3], task.cards[1], hero);
        }
        let bit = rank_bit(card);
        let suit = suit_value(card);
        let rank_index = card % 13u;
        rank_mask = rank_mask | bit;
        suit_masks[suit] = suit_masks[suit] | bit;
        counts[rank_index] = counts[rank_index] + 1u;
    }

    var flush_mask = 0u;
    for (var suit = 0u; suit < 4u; suit = suit + 1u) {
        if countOneBits(suit_masks[suit]) >= 5u {
            flush_mask = suit_masks[suit];
        }
    }
    if flush_mask != 0u {
        let sf = straight_high(flush_mask);
        if sf != 0u {
            return pack(8u, sf, 0u, 0u, 0u, 0u);
        }
    }

    var four = 0u;
    var trip1 = 0u;
    var trip2 = 0u;
    var pair1 = 0u;
    var pair2 = 0u;
    var rank = 13u;
    loop {
        let count = counts[rank - 1u];
        if count == 4u && four == 0u {
            four = rank;
        }
        if count == 3u {
            if trip1 == 0u {
                trip1 = rank;
            } else if trip2 == 0u {
                trip2 = rank;
            }
        }
        if count == 2u {
            if pair1 == 0u {
                pair1 = rank;
            } else if pair2 == 0u {
                pair2 = rank;
            }
        }
        if rank == 1u {
            break;
        }
        rank = rank - 1u;
    }

    if four != 0u {
        let kicker = highest_from_mask(rank_mask, four, 0u, 1u, 0u);
        return pack(7u, four, kicker, 0u, 0u, 0u);
    }
    if trip1 != 0u && (pair1 != 0u || trip2 != 0u) {
        let full_pair = select(pair1, trip2, pair1 == 0u);
        return pack(6u, trip1, full_pair, 0u, 0u, 0u);
    }
    if flush_mask != 0u {
        return pack(
            5u,
            highest_from_mask(flush_mask, 0u, 0u, 5u, 0u),
            highest_from_mask(flush_mask, 0u, 0u, 5u, 1u),
            highest_from_mask(flush_mask, 0u, 0u, 5u, 2u),
            highest_from_mask(flush_mask, 0u, 0u, 5u, 3u),
            highest_from_mask(flush_mask, 0u, 0u, 5u, 4u)
        );
    }
    let straight = straight_high(rank_mask);
    if straight != 0u {
        return pack(4u, straight, 0u, 0u, 0u, 0u);
    }
    if trip1 != 0u {
        return pack(
            3u,
            trip1,
            highest_from_mask(rank_mask, trip1, 0u, 2u, 0u),
            highest_from_mask(rank_mask, trip1, 0u, 2u, 1u),
            0u,
            0u
        );
    }
    if pair1 != 0u && pair2 != 0u {
        return pack(
            2u,
            pair1,
            pair2,
            highest_from_mask(rank_mask, pair1, pair2, 1u, 0u),
            0u,
            0u
        );
    }
    if pair1 != 0u {
        return pack(
            1u,
            pair1,
            highest_from_mask(rank_mask, pair1, 0u, 3u, 0u),
            highest_from_mask(rank_mask, pair1, 0u, 3u, 1u),
            highest_from_mask(rank_mask, pair1, 0u, 3u, 2u),
            0u
        );
    }
    return pack(
        0u,
        highest_from_mask(rank_mask, 0u, 0u, 5u, 0u),
        highest_from_mask(rank_mask, 0u, 0u, 5u, 1u),
        highest_from_mask(rank_mask, 0u, 0u, 5u, 2u),
        highest_from_mask(rank_mask, 0u, 0u, 5u, 3u),
        highest_from_mask(rank_mask, 0u, 0u, 5u, 4u)
    );
}

@compute @workgroup_size(64)
fn showdown(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let count = params[0];
    if index >= count {
        return;
    }
    let task = tasks[index];
    let hero_strength = evaluate_task(task, true);
    let villain_strength = evaluate_task(task, false);
    if hero_strength > villain_strength {
        equities[index] = 1.0;
    } else if hero_strength == villain_strength {
        equities[index] = 0.5;
    } else {
        equities[index] = 0.0;
    }
}
"#;

const SHOWDOWN_MATRIX_SHADER: &str = r#"
struct Combo {
    cards: array<u32, 2>,
};

struct FinalBoard {
    cards: array<u32, 5>,
};

@group(0) @binding(0) var<storage, read> combos: array<Combo>;
@group(0) @binding(1) var<storage, read> boards: array<FinalBoard>;
@group(0) @binding(2) var<storage, read_write> equities: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

fn card_mask(card: u32) -> u32 {
    return 1u << card;
}

fn rank_bit(card: u32) -> u32 {
    let rank = card % 13u;
    if rank == 12u {
        return (1u << 0u) | (1u << 13u);
    }
    return 1u << (rank + 1u);
}

fn suit_value(card: u32) -> u32 {
    return card / 13u;
}

fn pack(category: u32, a: u32, b: u32, c: u32, d: u32, e: u32) -> u32 {
    return (category << 20u) | (a << 16u) | (b << 12u) | (c << 8u) | (d << 4u) | e;
}

fn straight_high(rank_mask: u32) -> u32 {
    var high = 13u;
    loop {
        let window = 31u << (high - 4u);
        if (rank_mask & window) == window {
            return high;
        }
        if high == 4u {
            break;
        }
        high = high - 1u;
    }
    return 0u;
}

fn highest_from_mask(mask: u32, skip_a: u32, skip_b: u32, take: u32, slot: u32) -> u32 {
    var seen = 0u;
    var rank = 13u;
    loop {
        if rank != skip_a && rank != skip_b && (mask & (1u << rank)) != 0u {
            if seen == slot {
                return rank;
            }
            seen = seen + 1u;
            if seen >= take {
                return 0u;
            }
        }
        if rank == 1u {
            break;
        }
        rank = rank - 1u;
    }
    return 0u;
}

fn evaluate_combo_board(private_cards: Combo, board: FinalBoard) -> u32 {
    var rank_mask = 0u;
    var suit_masks = array<u32, 4>(0u, 0u, 0u, 0u);
    var counts = array<u32, 13>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);

    for (var i = 0u; i < 7u; i = i + 1u) {
        var card = private_cards.cards[0];
        if i == 1u {
            card = private_cards.cards[1];
        }
        if i >= 2u {
            card = board.cards[i - 2u];
        }
        rank_mask = rank_mask | rank_bit(card);
        let suit = suit_value(card);
        suit_masks[suit] = suit_masks[suit] | rank_bit(card);
        counts[card % 13u] = counts[card % 13u] + 1u;
    }

    var flush_mask = 0u;
    for (var suit = 0u; suit < 4u; suit = suit + 1u) {
        if countOneBits(suit_masks[suit]) >= 5u {
            flush_mask = suit_masks[suit];
        }
    }
    if flush_mask != 0u {
        let sf = straight_high(flush_mask);
        if sf != 0u {
            return pack(8u, sf, 0u, 0u, 0u, 0u);
        }
    }

    var four = 0u;
    var trip1 = 0u;
    var trip2 = 0u;
    var pair1 = 0u;
    var pair2 = 0u;
    var rank = 13u;
    loop {
        let count = counts[rank - 1u];
        if count == 4u && four == 0u {
            four = rank;
        }
        if count == 3u {
            if trip1 == 0u {
                trip1 = rank;
            } else if trip2 == 0u {
                trip2 = rank;
            }
        }
        if count == 2u {
            if pair1 == 0u {
                pair1 = rank;
            } else if pair2 == 0u {
                pair2 = rank;
            }
        }
        if rank == 1u {
            break;
        }
        rank = rank - 1u;
    }

    if four != 0u {
        return pack(7u, four, highest_from_mask(rank_mask, four, 0u, 1u, 0u), 0u, 0u, 0u);
    }
    if trip1 != 0u && (pair1 != 0u || trip2 != 0u) {
        return pack(6u, trip1, select(pair1, trip2, pair1 == 0u), 0u, 0u, 0u);
    }
    if flush_mask != 0u {
        return pack(
            5u,
            highest_from_mask(flush_mask, 0u, 0u, 5u, 0u),
            highest_from_mask(flush_mask, 0u, 0u, 5u, 1u),
            highest_from_mask(flush_mask, 0u, 0u, 5u, 2u),
            highest_from_mask(flush_mask, 0u, 0u, 5u, 3u),
            highest_from_mask(flush_mask, 0u, 0u, 5u, 4u)
        );
    }
    let straight = straight_high(rank_mask);
    if straight != 0u {
        return pack(4u, straight, 0u, 0u, 0u, 0u);
    }
    if trip1 != 0u {
        return pack(
            3u,
            trip1,
            highest_from_mask(rank_mask, trip1, 0u, 2u, 0u),
            highest_from_mask(rank_mask, trip1, 0u, 2u, 1u),
            0u,
            0u
        );
    }
    if pair1 != 0u && pair2 != 0u {
        return pack(2u, pair1, pair2, highest_from_mask(rank_mask, pair1, pair2, 1u, 0u), 0u, 0u);
    }
    if pair1 != 0u {
        return pack(
            1u,
            pair1,
            highest_from_mask(rank_mask, pair1, 0u, 3u, 0u),
            highest_from_mask(rank_mask, pair1, 0u, 3u, 1u),
            highest_from_mask(rank_mask, pair1, 0u, 3u, 2u),
            0u
        );
    }
    return pack(
        0u,
        highest_from_mask(rank_mask, 0u, 0u, 5u, 0u),
        highest_from_mask(rank_mask, 0u, 0u, 5u, 1u),
        highest_from_mask(rank_mask, 0u, 0u, 5u, 2u),
        highest_from_mask(rank_mask, 0u, 0u, 5u, 3u),
        highest_from_mask(rank_mask, 0u, 0u, 5u, 4u)
    );
}

fn combo_mask(combo: Combo) -> u32 {
    return card_mask(combo.cards[0]) | card_mask(combo.cards[1]);
}

fn board_mask(board: FinalBoard) -> u32 {
    return card_mask(board.cards[0])
        | card_mask(board.cards[1])
        | card_mask(board.cards[2])
        | card_mask(board.cards[3])
        | card_mask(board.cards[4]);
}

@compute @workgroup_size(64)
fn matrix(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let combo_count = params[0];
    let board_count = params[1];
    let pair_start = params[2];
    let output_count = params[3];
    if index >= output_count {
        return;
    }
    let pair = pair_start + index;
    let hero_index = pair / combo_count;
    let villain_index = pair % combo_count;
    let hero = combos[hero_index];
    let villain = combos[villain_index];
    let private_mask = combo_mask(hero) | combo_mask(villain);
    if countOneBits(private_mask) < 4u {
        equities[index] = 0.5;
        return;
    }

    var sum = 0.0;
    var count = 0u;
    for (var board_index = 0u; board_index < board_count; board_index = board_index + 1u) {
        let board = boards[board_index];
        if (board_mask(board) & private_mask) != 0u {
            continue;
        }
        let hero_strength = evaluate_combo_board(hero, board);
        let villain_strength = evaluate_combo_board(villain, board);
        if hero_strength > villain_strength {
            sum = sum + 1.0;
        } else if hero_strength == villain_strength {
            sum = sum + 0.5;
        }
        count = count + 1u;
    }

    if count == 0u {
        equities[index] = 0.5;
    } else {
        equities[index] = sum / f32(count);
    }
}

@compute @workgroup_size(64)
fn strengths(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * 4194240u;
    let combo_count = params[0];
    let board_count = params[1];
    if index >= combo_count * board_count {
        return;
    }
    let board_index = index / combo_count;
    let combo_index = index % combo_count;
    equities[index] = f32(evaluate_combo_board(combos[combo_index], boards[board_index]));
}
"#;

const PUBLIC_TREE_LAYER_REACH_INIT_SHADER: &str = r#"
struct Params {
    combo_count: u32,
    node_count: u32,
    max_actions: u32,
    output_len: u32,
    variant: u32,
    edge_count: u32,
    tile_node_start: u32,
    eta_bits: u32,
};

@group(0) @binding(0) var<storage, read> root_weights: array<f32>;
@group(0) @binding(1) var<storage, read_write> hero_reaches: array<f32>;
@group(0) @binding(2) var<storage, read_write> villain_reaches: array<f32>;
@group(0) @binding(3) var<storage, read_write> combo_live: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn reach_init_tile(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * params.output_len;
    let value_count = params.node_count * params.combo_count;
    if index >= value_count {
        return;
    }
    let local_node = index / params.combo_count;
    let combo = index % params.combo_count;
    hero_reaches[index] = 0.0;
    villain_reaches[index] = 0.0;
    combo_live[index] = 0.0;
    if params.tile_node_start == 0u && local_node == 0u && root_weights[combo] >= 0.0 {
        hero_reaches[index] = 1.0;
        villain_reaches[index] = root_weights[combo];
        combo_live[index] = 1.0;
    }
}
"#;

const PUBLIC_TREE_LAYER_REACH_SHADER: &str = r#"
struct Combo { cards: array<u32, 2>, };
struct Edge {
    parent: u32,
    child: u32,
    action: u32,
    card: u32,
};
struct TreeNode {
    kind: u32,
    acting_player: u32,
    public_infoset: u32,
    first_child: u32,
    child_count: u32,
    terminal_kind: u32,
    showdown_offset: u32,
    _pad0: u32,
    pot: f32,
    hero_invested: f32,
    _pad1: f32,
    _pad2: f32,
};
struct Params {
    combo_count: u32,
    node_count: u32,
    max_actions: u32,
    output_len: u32,
    variant: u32,
    edge_count: u32,
    tile_node_start: u32,
    eta_bits: u32,
};

@group(0) @binding(0) var<storage, read> parent_nodes: array<TreeNode>;
@group(0) @binding(1) var<storage, read> edges: array<Edge>;
@group(0) @binding(2) var<storage, read> combos: array<Combo>;
@group(0) @binding(3) var<storage, read> root_weights: array<f32>;
@group(0) @binding(4) var<storage, read> regrets: array<f32>;
@group(0) @binding(5) var<storage, read> parent_hero_reaches: array<f32>;
@group(0) @binding(6) var<storage, read> parent_villain_reaches: array<f32>;
@group(0) @binding(7) var<storage, read> parent_combo_live: array<f32>;
@group(0) @binding(8) var<storage, read_write> child_hero_reaches: array<f32>;
@group(0) @binding(9) var<storage, read_write> child_villain_reaches: array<f32>;
@group(0) @binding(10) var<storage, read_write> child_combo_live: array<f32>;
@group(0) @binding(11) var<storage, read> prediction: array<f32>;
@group(0) @binding(12) var<uniform> params: Params;

fn combo_has_card(combo: Combo, card: u32) -> bool {
    return combo.cards[0] == card || combo.cards[1] == card;
}

fn effective_regret(index: u32) -> f32 {
    if params.variant == 3u {
        return regrets[index] + bitcast<f32>(params.eta_bits) * prediction[index];
    }
    return regrets[index];
}

fn strategy_probability(node: TreeNode, private_combo: u32, action: u32) -> f32 {
    let private_infoset = node.public_infoset * params.combo_count * 2u
        + node.acting_player * params.combo_count
        + private_combo;
    let offset = private_infoset * params.max_actions;
    var normalizer = 0.0;
    for (var i = 0u; i < node.child_count; i = i + 1u) {
        normalizer = normalizer + max(effective_regret(offset + i), 0.0);
    }
    if normalizer > 0.0 {
        return max(effective_regret(offset + action), 0.0) / normalizer;
    }
    return 1.0 / f32(max(node.child_count, 1u));
}

@compute @workgroup_size(64)
fn reach_edge_tile(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * params.output_len;
    let value_count = params.edge_count * params.combo_count;
    if index >= value_count {
        return;
    }
    let combo = index % params.combo_count;
    let edge_slot = index / params.combo_count;
    let edge = edges[edge_slot];
    let node = parent_nodes[edge.parent];
    let parent_offset = edge.parent * params.combo_count + combo;
    let child_offset = edge.child * params.combo_count + combo;
    let hero_reach = parent_hero_reaches[parent_offset];
    let villain_reach = parent_villain_reaches[parent_offset];
    let live = parent_combo_live[parent_offset];
    if node.kind == 0u {
        let probability = strategy_probability(node, combo, edge.action);
        if node.acting_player == 0u {
            child_hero_reaches[child_offset] = hero_reach * probability;
            child_villain_reaches[child_offset] = villain_reach;
        } else {
            child_hero_reaches[child_offset] = hero_reach;
            child_villain_reaches[child_offset] = villain_reach * probability;
        }
        child_combo_live[child_offset] = live;
    } else if node.kind == 1u {
        if combo_has_card(combos[combo], edge.card) {
            child_hero_reaches[child_offset] = 0.0;
            child_villain_reaches[child_offset] = 0.0;
            child_combo_live[child_offset] = 0.0;
        } else {
            child_hero_reaches[child_offset] = hero_reach;
            child_villain_reaches[child_offset] = villain_reach;
            child_combo_live[child_offset] = live;
        }
    }
}
"#;

const PUBLIC_TREE_TERMINAL_PARTIAL_SHADER: &str = r#"
struct TreeNode {
    kind: u32,
    acting_player: u32,
    public_infoset: u32,
    first_child: u32,
    child_count: u32,
    terminal_kind: u32,
    showdown_offset: u32,
    _pad0: u32,
    pot: f32,
    hero_invested: f32,
    _pad1: f32,
    _pad2: f32,
};
struct Bounds {
    group_start: u32,
    group_end: u32,
    legal: u32,
    _pad0: u32,
};
struct PrefixPair {
    hero: f32,
    villain: f32,
};
struct Params {
    combo_count: u32,
    terminal_count: u32,
    board_count: u32,
    prefix_stride: u32,
    order_stride: u32,
    x_invocations: u32,
    _pad0: u32,
    terminal_start: u32,
};
struct ChunkParams {
    terminal_count: u32,
    x_invocations: u32,
    terminal_start: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> nodes: array<TreeNode>;
@group(0) @binding(1) var<storage, read> terminal_nodes: array<u32>;
@group(0) @binding(2) var<storage, read> combo_order: array<u32>;
@group(0) @binding(3) var<storage, read> combo_bounds: array<Bounds>;
@group(0) @binding(4) var<storage, read> hero_reaches: array<f32>;
@group(0) @binding(5) var<storage, read> villain_reaches: array<f32>;
@group(0) @binding(6) var<storage, read_write> prefix_pairs: array<PrefixPair>;
@group(0) @binding(7) var<uniform> params: Params;
var<immediate> chunk: ChunkParams;

const TERMINAL_PREFIX_CAPACITY: u32 = 2048u;
var<workgroup> scan_pairs: array<PrefixPair, 2048>;

fn terminal_scan_upsweep(stride: u32, local_index: u32) {
    let step = stride * 2u;
    for (var slot = local_index; slot < TERMINAL_PREFIX_CAPACITY / step; slot = slot + 256u) {
        let scan_target = (slot + 1u) * step - 1u;
        let source = scan_target - stride;
        scan_pairs[scan_target] = PrefixPair(
            scan_pairs[scan_target].hero + scan_pairs[source].hero,
            scan_pairs[scan_target].villain + scan_pairs[source].villain,
        );
    }
}

fn terminal_scan_downsweep(stride: u32, local_index: u32) {
    let step = stride * 2u;
    for (var slot = local_index; slot < TERMINAL_PREFIX_CAPACITY / step; slot = slot + 256u) {
        let scan_target = (slot + 1u) * step - 1u;
        let source = scan_target - stride;
        let left = scan_pairs[source];
        scan_pairs[source] = scan_pairs[scan_target];
        scan_pairs[scan_target] = PrefixPair(
            scan_pairs[scan_target].hero + left.hero,
            scan_pairs[scan_target].villain + left.villain,
        );
    }
}

@compute @workgroup_size(256)
fn terminal_partial(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let index = workgroup_id.x + workgroup_id.y * chunk.x_invocations;
    let output_count = chunk.terminal_count * params.board_count;
    let is_live = index < output_count && params.combo_count + 1u <= TERMINAL_PREFIX_CAPACITY;
    let board = select(0u, index % params.board_count, is_live);
    let terminal_slot = select(0u, index / params.board_count, is_live);
    let group_terminal_slot = chunk.terminal_start + terminal_slot;
    let node_index = terminal_nodes[group_terminal_slot];
    let node_offset = node_index * params.combo_count;
    let table_board = group_terminal_slot * params.board_count + board;
    let order_base = table_board * params.order_stride;
    let prefix_base = (terminal_slot * params.board_count + board) * params.prefix_stride;

    for (var position = local_id.x; position < TERMINAL_PREFIX_CAPACITY; position = position + 256u) {
        var pair = PrefixPair(0.0, 0.0);
        if is_live && position < params.combo_count {
            let combo = combo_order[order_base + position];
            if combo != 0xffffffffu {
                let bounds = combo_bounds[table_board * params.combo_count + combo];
                if bounds.legal != 0u {
                    pair = PrefixPair(
                        hero_reaches[node_offset + combo],
                        villain_reaches[node_offset + combo],
                    );
                }
            }
        }
        scan_pairs[position] = pair;
    }
    workgroupBarrier();

    terminal_scan_upsweep(1u, local_id.x);
    workgroupBarrier();
    terminal_scan_upsweep(2u, local_id.x);
    workgroupBarrier();
    terminal_scan_upsweep(4u, local_id.x);
    workgroupBarrier();
    terminal_scan_upsweep(8u, local_id.x);
    workgroupBarrier();
    terminal_scan_upsweep(16u, local_id.x);
    workgroupBarrier();
    terminal_scan_upsweep(32u, local_id.x);
    workgroupBarrier();
    terminal_scan_upsweep(64u, local_id.x);
    workgroupBarrier();
    terminal_scan_upsweep(128u, local_id.x);
    workgroupBarrier();
    terminal_scan_upsweep(256u, local_id.x);
    workgroupBarrier();
    terminal_scan_upsweep(512u, local_id.x);
    workgroupBarrier();
    terminal_scan_upsweep(1024u, local_id.x);
    workgroupBarrier();

    if local_id.x == 0u {
        scan_pairs[TERMINAL_PREFIX_CAPACITY - 1u] = PrefixPair(0.0, 0.0);
    }
    workgroupBarrier();

    terminal_scan_downsweep(1024u, local_id.x);
    workgroupBarrier();
    terminal_scan_downsweep(512u, local_id.x);
    workgroupBarrier();
    terminal_scan_downsweep(256u, local_id.x);
    workgroupBarrier();
    terminal_scan_downsweep(128u, local_id.x);
    workgroupBarrier();
    terminal_scan_downsweep(64u, local_id.x);
    workgroupBarrier();
    terminal_scan_downsweep(32u, local_id.x);
    workgroupBarrier();
    terminal_scan_downsweep(16u, local_id.x);
    workgroupBarrier();
    terminal_scan_downsweep(8u, local_id.x);
    workgroupBarrier();
    terminal_scan_downsweep(4u, local_id.x);
    workgroupBarrier();
    terminal_scan_downsweep(2u, local_id.x);
    workgroupBarrier();
    terminal_scan_downsweep(1u, local_id.x);
    workgroupBarrier();

    for (var position = local_id.x; is_live && position <= params.combo_count; position = position + 256u) {
        prefix_pairs[prefix_base + position] = scan_pairs[position];
    }
}
"#;

const PUBLIC_TREE_TERMINAL_PARTIAL_SERIAL_SHADER: &str = r#"
struct TreeNode {
    kind: u32,
    acting_player: u32,
    public_infoset: u32,
    first_child: u32,
    child_count: u32,
    terminal_kind: u32,
    showdown_offset: u32,
    _pad0: u32,
    pot: f32,
    hero_invested: f32,
    _pad1: f32,
    _pad2: f32,
};
struct Bounds {
    group_start: u32,
    group_end: u32,
    legal: u32,
    _pad0: u32,
};
struct PrefixPair {
    hero: f32,
    villain: f32,
};
struct Params {
    combo_count: u32,
    terminal_count: u32,
    board_count: u32,
    prefix_stride: u32,
    order_stride: u32,
    x_invocations: u32,
    _pad0: u32,
    terminal_start: u32,
};
struct ChunkParams {
    terminal_count: u32,
    x_invocations: u32,
    terminal_start: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> nodes: array<TreeNode>;
@group(0) @binding(1) var<storage, read> terminal_nodes: array<u32>;
@group(0) @binding(2) var<storage, read> combo_order: array<u32>;
@group(0) @binding(3) var<storage, read> combo_bounds: array<Bounds>;
@group(0) @binding(4) var<storage, read> hero_reaches: array<f32>;
@group(0) @binding(5) var<storage, read> villain_reaches: array<f32>;
@group(0) @binding(6) var<storage, read_write> prefix_pairs: array<PrefixPair>;
@group(0) @binding(7) var<uniform> params: Params;
var<immediate> chunk: ChunkParams;

@compute @workgroup_size(64)
fn terminal_partial(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * chunk.x_invocations;
    let output_count = chunk.terminal_count * params.board_count;
    if index >= output_count {
        return;
    }
    let board = index % params.board_count;
    let terminal_slot = index / params.board_count;
    let group_terminal_slot = chunk.terminal_start + terminal_slot;
    let node_index = terminal_nodes[group_terminal_slot];
    let node_offset = node_index * params.combo_count;
    let table_board = group_terminal_slot * params.board_count + board;
    let order_base = table_board * params.order_stride;
    let prefix_base = (terminal_slot * params.board_count + board) * params.prefix_stride;
    var hero_sum = 0.0;
    var villain_sum = 0.0;
    prefix_pairs[prefix_base] = PrefixPair(0.0, 0.0);
    for (var position = 0u; position < params.combo_count; position = position + 1u) {
        let combo = combo_order[order_base + position];
        if combo != 0xffffffffu {
            let bounds = combo_bounds[table_board * params.combo_count + combo];
            if bounds.legal != 0u {
                hero_sum = hero_sum + hero_reaches[node_offset + combo];
                villain_sum = villain_sum + villain_reaches[node_offset + combo];
            }
        }
        prefix_pairs[prefix_base + position + 1u] = PrefixPair(hero_sum, villain_sum);
    }
}
"#;

const PUBLIC_TREE_TERMINAL_REDUCE_SHADER: &str = r#"
struct TreeNode {
    kind: u32,
    acting_player: u32,
    public_infoset: u32,
    first_child: u32,
    child_count: u32,
    terminal_kind: u32,
    showdown_offset: u32,
    board_count: u32,
    pot: f32,
    hero_invested: f32,
    chance_scale: f32,
    showdown_denominator: f32,
};
struct Bounds {
    group_start: u32,
    group_end: u32,
    legal: u32,
    _pad0: u32,
};
struct PrefixPair {
    hero: f32,
    villain: f32,
};
struct Params {
    combo_count: u32,
    terminal_count: u32,
    board_count: u32,
    prefix_stride: u32,
    blocker_stride: u32,
    x_invocations: u32,
    _pad0: u32,
    terminal_start: u32,
};
struct ChunkParams {
    terminal_count: u32,
    x_invocations: u32,
    terminal_start: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> nodes: array<TreeNode>;
@group(0) @binding(1) var<storage, read> terminal_nodes: array<u32>;
@group(0) @binding(2) var<storage, read> combo_bounds: array<Bounds>;
@group(0) @binding(3) var<storage, read> blocker_neighbors: array<u32>;
@group(0) @binding(4) var<storage, read> hero_reaches: array<f32>;
@group(0) @binding(5) var<storage, read> villain_reaches: array<f32>;
@group(0) @binding(6) var<storage, read> prefix_pairs: array<PrefixPair>;
@group(0) @binding(7) var<storage, read_write> hero_values: array<f32>;
@group(0) @binding(8) var<storage, read_write> villain_values: array<f32>;
@group(0) @binding(9) var<uniform> params: Params;
var<immediate> chunk: ChunkParams;

@compute @workgroup_size(64)
fn terminal_reduce(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * chunk.x_invocations;
    let output_count = chunk.terminal_count * params.combo_count;
    if index >= output_count {
        return;
    }
    let combo = index % params.combo_count;
    let terminal_slot = index / params.combo_count;
    let group_terminal_slot = chunk.terminal_start + terminal_slot;
    let node_index = terminal_nodes[group_terminal_slot];
    let node = nodes[node_index];
    let node_offset = node_index * params.combo_count;
    let denom = max(node.showdown_denominator, 1.0);
    var hero_value = 0.0;
    var villain_value = 0.0;
    for (var local_board = 0u; local_board < node.board_count; local_board = local_board + 1u) {
        let table_board = group_terminal_slot * params.board_count + local_board;
        let bounds = combo_bounds[table_board * params.combo_count + combo];
        if bounds.legal == 0u {
            continue;
        }
        let prefix_base = (terminal_slot * params.board_count + local_board) * params.prefix_stride;
        let total_pair = prefix_pairs[prefix_base + params.combo_count];
        let win_pair = prefix_pairs[prefix_base + bounds.group_start];
        let tie_pair = prefix_pairs[prefix_base + bounds.group_end];
        var hero_win = win_pair.villain;
        var hero_tie = tie_pair.villain - win_pair.villain;
        var hero_total = total_pair.villain;
        var villain_win = win_pair.hero;
        var villain_tie = tie_pair.hero - win_pair.hero;
        var villain_total = total_pair.hero;

        let neighbor_base = combo * params.blocker_stride;
        for (var slot = 0u; slot < params.blocker_stride; slot = slot + 1u) {
            let opponent = blocker_neighbors[neighbor_base + slot];
            if opponent == 0xffffffffu {
                continue;
            }
            let opponent_bounds = combo_bounds[table_board * params.combo_count + opponent];
            if opponent_bounds.legal == 0u {
                continue;
            }
            let opponent_hero_reach = hero_reaches[node_offset + opponent];
            let opponent_villain_reach = villain_reaches[node_offset + opponent];
            hero_total = hero_total - opponent_villain_reach;
            villain_total = villain_total - opponent_hero_reach;
            if opponent_bounds.group_end <= bounds.group_start {
                hero_win = hero_win - opponent_villain_reach;
                villain_win = villain_win - opponent_hero_reach;
            } else if opponent_bounds.group_start == bounds.group_start {
                hero_tie = hero_tie - opponent_villain_reach;
                villain_tie = villain_tie - opponent_hero_reach;
            }
        }
        hero_value = hero_value
            + (node.pot * (hero_win + 0.5 * hero_tie) - node.hero_invested * hero_total) / denom;
        let villain_invested = node.pot - node.hero_invested;
        villain_value = villain_value
            + (node.pot * (villain_win + 0.5 * villain_tie) - villain_invested * villain_total) / denom;
    }
    let value_index = node_index * params.combo_count + combo;
    hero_values[value_index] = hero_value * node.chance_scale;
    villain_values[value_index] = villain_value * node.chance_scale;
}
"#;

const PUBLIC_TREE_TERMINAL_CARD_PREFIX_SHADER: &str = r#"
struct Combo { cards: array<u32, 2>, };
struct TreeNode {
    kind: u32,
    acting_player: u32,
    public_infoset: u32,
    first_child: u32,
    child_count: u32,
    terminal_kind: u32,
    showdown_offset: u32,
    board_count: u32,
    pot: f32,
    hero_invested: f32,
    chance_scale: f32,
    showdown_denominator: f32,
};
struct Bounds {
    group_start: u32,
    group_end: u32,
    legal: u32,
    _pad0: u32,
};
struct PrefixPair {
    hero: f32,
    villain: f32,
};
struct Params {
    combo_count: u32,
    terminal_count: u32,
    board_count: u32,
    prefix_stride: u32,
    card_stride: u32,
    x_invocations: u32,
    _pad0: u32,
    terminal_start: u32,
};

@group(0) @binding(0) var<storage, read> nodes: array<TreeNode>;
@group(0) @binding(1) var<storage, read> terminal_nodes: array<u32>;
@group(0) @binding(2) var<storage, read> combo_order: array<u32>;
@group(0) @binding(3) var<storage, read> combo_bounds: array<Bounds>;
@group(0) @binding(4) var<storage, read> combos: array<Combo>;
@group(0) @binding(5) var<storage, read> hero_reaches: array<f32>;
@group(0) @binding(6) var<storage, read> villain_reaches: array<f32>;
@group(0) @binding(7) var<storage, read_write> card_prefix_pairs: array<PrefixPair>;
@group(0) @binding(8) var<uniform> params: Params;

@compute @workgroup_size(64)
fn terminal_card_prefix(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let index = workgroup_id.x + workgroup_id.y * params.x_invocations;
    let output_count = params.terminal_count * params.board_count;
    if index >= output_count {
        return;
    }
    let card = local_id.x;
    if card >= 52u {
        return;
    }
    let board = index % params.board_count;
    let terminal_slot = index / params.board_count;
    let group_terminal_slot = params.terminal_start + terminal_slot;
    let node_index = terminal_nodes[group_terminal_slot];
    let node_offset = node_index * params.combo_count;
    let table_board = group_terminal_slot * params.board_count + board;
    let order_base = table_board * params.combo_count;
    let output_base =
        (terminal_slot * params.board_count * 52u + board * 52u + card) * params.card_stride;
    var hero_sum = 0.0;
    var villain_sum = 0.0;
    card_prefix_pairs[output_base] = PrefixPair(0.0, 0.0);
    for (var position = 0u; position < params.combo_count; position = position + 1u) {
        let combo_index = combo_order[order_base + position];
        if combo_index != 0xffffffffu {
            let bounds = combo_bounds[table_board * params.combo_count + combo_index];
            let combo = combos[combo_index];
            if bounds.legal != 0u {
                if bounds.group_start == position {
                    card_prefix_pairs[output_base + position] =
                        PrefixPair(hero_sum, villain_sum);
                }
                if combo.cards[0] == card || combo.cards[1] == card {
                    hero_sum = hero_sum + hero_reaches[node_offset + combo_index];
                    villain_sum = villain_sum + villain_reaches[node_offset + combo_index];
                }
                if bounds.group_end == position + 1u {
                    card_prefix_pairs[output_base + position + 1u] =
                        PrefixPair(hero_sum, villain_sum);
                }
            }
        }
    }
    card_prefix_pairs[output_base + params.combo_count] = PrefixPair(hero_sum, villain_sum);
}
"#;

const PUBLIC_TREE_TERMINAL_CARD_AGGREGATE_REDUCE_SHADER: &str = r#"
struct Combo { cards: array<u32, 2>, };
struct TreeNode {
    kind: u32,
    acting_player: u32,
    public_infoset: u32,
    first_child: u32,
    child_count: u32,
    terminal_kind: u32,
    showdown_offset: u32,
    board_count: u32,
    pot: f32,
    hero_invested: f32,
    chance_scale: f32,
    showdown_denominator: f32,
};
struct Bounds {
    group_start: u32,
    group_end: u32,
    legal: u32,
    _pad0: u32,
};
struct PrefixPair {
    hero: f32,
    villain: f32,
};
struct Params {
    combo_count: u32,
    terminal_count: u32,
    board_count: u32,
    prefix_stride: u32,
    card_stride: u32,
    x_invocations: u32,
    _pad0: u32,
    terminal_start: u32,
};

@group(0) @binding(0) var<storage, read> nodes: array<TreeNode>;
@group(0) @binding(1) var<storage, read> terminal_nodes: array<u32>;
@group(0) @binding(2) var<storage, read> combo_bounds: array<Bounds>;
@group(0) @binding(3) var<storage, read> combos: array<Combo>;
@group(0) @binding(4) var<storage, read> hero_reaches: array<f32>;
@group(0) @binding(5) var<storage, read> villain_reaches: array<f32>;
@group(0) @binding(6) var<storage, read> prefix_pairs: array<PrefixPair>;
@group(0) @binding(7) var<storage, read> card_prefix_pairs: array<PrefixPair>;
@group(0) @binding(8) var<storage, read_write> hero_values: array<f32>;
@group(0) @binding(9) var<storage, read_write> villain_values: array<f32>;
@group(0) @binding(10) var<uniform> params: Params;

fn card_prefix_base(terminal_slot: u32, local_board: u32, card: u32) -> u32 {
    return (terminal_slot * params.board_count * 52u + local_board * 52u + card) * params.card_stride;
}

@compute @workgroup_size(64)
fn terminal_reduce(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * params.x_invocations;
    let output_count = params.terminal_count * params.combo_count;
    if index >= output_count {
        return;
    }
    let combo = index % params.combo_count;
    let terminal_slot = index / params.combo_count;
    let group_terminal_slot = params.terminal_start + terminal_slot;
    let node_index = terminal_nodes[group_terminal_slot];
    let node = nodes[node_index];
    let node_offset = node_index * params.combo_count;
    let private_combo = combos[combo];
    let denom = max(node.showdown_denominator, 1.0);
    var hero_value = 0.0;
    var villain_value = 0.0;
    for (var local_board = 0u; local_board < node.board_count; local_board = local_board + 1u) {
        let table_board = group_terminal_slot * params.board_count + local_board;
        let bounds = combo_bounds[table_board * params.combo_count + combo];
        if bounds.legal == 0u {
            continue;
        }
        let prefix_base = (terminal_slot * params.board_count + local_board) * params.prefix_stride;
        let total_pair = prefix_pairs[prefix_base + params.combo_count];
        let win_pair = prefix_pairs[prefix_base + bounds.group_start];
        let tie_pair = prefix_pairs[prefix_base + bounds.group_end];

        let card0_base = card_prefix_base(terminal_slot, local_board, private_combo.cards[0]);
        let card1_base = card_prefix_base(terminal_slot, local_board, private_combo.cards[1]);
        let card0_total = card_prefix_pairs[card0_base + params.combo_count];
        let card1_total = card_prefix_pairs[card1_base + params.combo_count];
        let card0_win = card_prefix_pairs[card0_base + bounds.group_start];
        let card1_win = card_prefix_pairs[card1_base + bounds.group_start];
        let card0_tie_start = card_prefix_pairs[card0_base + bounds.group_start];
        let card0_tie_end = card_prefix_pairs[card0_base + bounds.group_end];
        let card1_tie_start = card_prefix_pairs[card1_base + bounds.group_start];
        let card1_tie_end = card_prefix_pairs[card1_base + bounds.group_end];

        let self_hero = hero_reaches[node_offset + combo];
        let self_villain = villain_reaches[node_offset + combo];

        let hero_block_total = card0_total.villain + card1_total.villain - self_villain;
        let hero_block_win = card0_win.villain + card1_win.villain;
        let hero_block_tie =
            (card0_tie_end.villain - card0_tie_start.villain)
            + (card1_tie_end.villain - card1_tie_start.villain)
            - self_villain;
        let villain_block_total = card0_total.hero + card1_total.hero - self_hero;
        let villain_block_win = card0_win.hero + card1_win.hero;
        let villain_block_tie =
            (card0_tie_end.hero - card0_tie_start.hero)
            + (card1_tie_end.hero - card1_tie_start.hero)
            - self_hero;

        let hero_win = win_pair.villain - hero_block_win;
        let hero_tie = tie_pair.villain - win_pair.villain - hero_block_tie;
        let hero_total = total_pair.villain - hero_block_total;
        let villain_win = win_pair.hero - villain_block_win;
        let villain_tie = tie_pair.hero - win_pair.hero - villain_block_tie;
        let villain_total = total_pair.hero - villain_block_total;

        hero_value = hero_value
            + (node.pot * (hero_win + 0.5 * hero_tie) - node.hero_invested * hero_total) / denom;
        let villain_invested = node.pot - node.hero_invested;
        villain_value = villain_value
            + (node.pot * (villain_win + 0.5 * villain_tie) - villain_invested * villain_total) / denom;
    }
    let value_index = node_index * params.combo_count + combo;
    hero_values[value_index] = hero_value * node.chance_scale;
    villain_values[value_index] = villain_value * node.chance_scale;
}
"#;

const PUBLIC_TREE_FOLD_AGGREGATE_SHADER: &str = r#"
struct Combo { cards: array<u32, 2>, };
struct TreeNode {
    kind: u32,
    acting_player: u32,
    public_infoset: u32,
    first_child: u32,
    child_count: u32,
    terminal_kind: u32,
    showdown_offset: u32,
    _pad0: u32,
    pot: f32,
    hero_invested: f32,
    _pad1: f32,
    _pad2: f32,
};
struct Params {
    combo_count: u32,
    terminal_count: u32,
    slots_per_terminal: u32,
    output_len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> terminal_nodes: array<u32>;
@group(0) @binding(1) var<storage, read> combos: array<Combo>;
@group(0) @binding(2) var<storage, read> hero_reaches: array<f32>;
@group(0) @binding(3) var<storage, read> villain_reaches: array<f32>;
@group(0) @binding(4) var<storage, read_write> hero_aggregates: array<f32>;
@group(0) @binding(5) var<storage, read_write> villain_aggregates: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn fold_aggregate(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * params.output_len;
    let output_count = params.terminal_count * params.slots_per_terminal;
    if index >= output_count {
        return;
    }
    let slot = index % params.slots_per_terminal;
    let terminal_slot = index / params.slots_per_terminal;
    let node_index = terminal_nodes[terminal_slot];
    let node_offset = node_index * params.combo_count;
    var hero_sum = 0.0;
    var villain_sum = 0.0;
    if slot == 0u {
        for (var combo = 0u; combo < params.combo_count; combo = combo + 1u) {
            hero_sum = hero_sum + hero_reaches[node_offset + combo];
            villain_sum = villain_sum + villain_reaches[node_offset + combo];
        }
    } else {
        let card = slot - 1u;
        for (var combo = 0u; combo < params.combo_count; combo = combo + 1u) {
            let private_combo = combos[combo];
            if private_combo.cards[0] == card || private_combo.cards[1] == card {
                hero_sum = hero_sum + hero_reaches[node_offset + combo];
                villain_sum = villain_sum + villain_reaches[node_offset + combo];
            }
        }
    }
    hero_aggregates[index] = hero_sum;
    villain_aggregates[index] = villain_sum;
}
"#;

const PUBLIC_TREE_FOLD_VALUE_SHADER: &str = r#"
struct Combo { cards: array<u32, 2>, };
struct TreeNode {
    kind: u32,
    acting_player: u32,
    public_infoset: u32,
    first_child: u32,
    child_count: u32,
    terminal_kind: u32,
    showdown_offset: u32,
    _pad0: u32,
    pot: f32,
    hero_invested: f32,
    _pad1: f32,
    _pad2: f32,
};
struct Params {
    combo_count: u32,
    terminal_count: u32,
    slots_per_terminal: u32,
    output_len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> nodes: array<TreeNode>;
@group(0) @binding(1) var<storage, read> terminal_nodes: array<u32>;
@group(0) @binding(2) var<storage, read> combos: array<Combo>;
@group(0) @binding(3) var<storage, read> hero_reaches: array<f32>;
@group(0) @binding(4) var<storage, read> villain_reaches: array<f32>;
@group(0) @binding(5) var<storage, read> hero_aggregates: array<f32>;
@group(0) @binding(6) var<storage, read> villain_aggregates: array<f32>;
@group(0) @binding(7) var<storage, read_write> hero_values: array<f32>;
@group(0) @binding(8) var<storage, read_write> villain_values: array<f32>;
@group(0) @binding(9) var<storage, read> combo_live: array<f32>;
@group(0) @binding(10) var<uniform> params: Params;

@compute @workgroup_size(64)
fn fold_value(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * params.output_len;
    let output_count = params.terminal_count * params.combo_count;
    if index >= output_count {
        return;
    }
    let combo = index % params.combo_count;
    let terminal_slot = index / params.combo_count;
    let node_index = terminal_nodes[terminal_slot];
    let node = nodes[node_index];
    let private_combo = combos[combo];
    let aggregate_base = terminal_slot * params.slots_per_terminal;
    let node_offset = node_index * params.combo_count;
    let hero_self = hero_reaches[node_offset + combo];
    let villain_self = villain_reaches[node_offset + combo];
    let hero_noncolliding = hero_aggregates[aggregate_base]
        - hero_aggregates[aggregate_base + private_combo.cards[0] + 1u]
        - hero_aggregates[aggregate_base + private_combo.cards[1] + 1u]
        + hero_self;
    let villain_noncolliding = villain_aggregates[aggregate_base]
        - villain_aggregates[aggregate_base + private_combo.cards[0] + 1u]
        - villain_aggregates[aggregate_base + private_combo.cards[1] + 1u]
        + villain_self;

    var hero_payoff = 0.0;
    if node.terminal_kind == 0u {
        hero_payoff = -node.hero_invested;
    } else {
        hero_payoff = node.pot - node.hero_invested;
    }
    let value_index = node_offset + combo;
    let structurally_live = combo_live[value_index] > 0.0;
    hero_values[value_index] = select(0.0, villain_noncolliding * hero_payoff * node._pad1, structurally_live);
    villain_values[value_index] = select(0.0, hero_noncolliding * (-hero_payoff) * node._pad1, structurally_live);
}
"#;

const PUBLIC_TREE_LAYER_BACKUP_INIT_SHADER: &str = r#"
struct TreeNode {
    kind: u32,
    acting_player: u32,
    public_infoset: u32,
    first_child: u32,
    child_count: u32,
    terminal_kind: u32,
    showdown_offset: u32,
    _pad0: u32,
    pot: f32,
    hero_invested: f32,
    _pad1: f32,
    _pad2: f32,
};
struct Params {
    combo_count: u32,
    node_count: u32,
    max_actions: u32,
    output_len: u32,
    value_player: u32,
    child_tile_start: u32,
    child_tile_end: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> parent_nodes: array<TreeNode>;
@group(0) @binding(1) var<storage, read_write> parent_hero_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> parent_villain_values: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn backup_init_tile(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * params.output_len;
    let value_count = params.node_count * params.combo_count;
    if index >= value_count {
        return;
    }
    let node_slot = index / params.combo_count;
    let node = parent_nodes[node_slot];
    if node.kind != 0u && node.kind != 1u {
        return;
    }
    var hero_value = 0.0;
    var villain_value = 0.0;
    if node.kind == 0u && node.acting_player == 0u && params.value_player == 0u {
        hero_value = -1.0e30;
    }
    if node.kind == 0u && node.acting_player == 1u && params.value_player == 1u {
        villain_value = -1.0e30;
    }
    parent_hero_values[index] = hero_value;
    parent_villain_values[index] = villain_value;
}
"#;

const PUBLIC_TREE_LAYER_BACKUP_SHADER: &str = r#"
struct TreeNode {
    kind: u32,
    acting_player: u32,
    public_infoset: u32,
    first_child: u32,
    child_count: u32,
    terminal_kind: u32,
    showdown_offset: u32,
    _pad0: u32,
    pot: f32,
    hero_invested: f32,
    _pad1: f32,
    _pad2: f32,
};
struct Params {
    combo_count: u32,
    node_count: u32,
    max_actions: u32,
    output_len: u32,
    value_player: u32,
    child_tile_start: u32,
    child_tile_end: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> parent_nodes: array<TreeNode>;
@group(0) @binding(1) var<storage, read> parent_children: array<u32>;
@group(0) @binding(2) var<storage, read> child_hero_values: array<f32>;
@group(0) @binding(3) var<storage, read> child_villain_values: array<f32>;
@group(0) @binding(4) var<storage, read> parent_hero_reaches: array<f32>;
@group(0) @binding(5) var<storage, read> parent_villain_reaches: array<f32>;
@group(0) @binding(6) var<storage, read> child_hero_reaches: array<f32>;
@group(0) @binding(7) var<storage, read> child_villain_reaches: array<f32>;
@group(0) @binding(8) var<storage, read_write> parent_hero_values: array<f32>;
@group(0) @binding(9) var<storage, read_write> parent_villain_values: array<f32>;
@group(0) @binding(10) var<uniform> params: Params;

fn strategy_from_reach(parent_offset: u32, child_offset: u32, acting_player: u32) -> f32 {
    if acting_player == 0u {
        let parent = parent_hero_reaches[parent_offset];
        if parent > 0.0 {
            return child_hero_reaches[child_offset] / parent;
        }
    } else {
        let parent = parent_villain_reaches[parent_offset];
        if parent > 0.0 {
            return child_villain_reaches[child_offset] / parent;
        }
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn backup_child_tile(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * params.output_len;
    let value_count = params.node_count * params.combo_count;
    if index >= value_count {
        return;
    }
    let combo = index % params.combo_count;
    let node_slot = index / params.combo_count;
    let node = parent_nodes[node_slot];
    if node.kind != 0u && node.kind != 1u {
        return;
    }

    var hero_value = parent_hero_values[index];
    var villain_value = parent_villain_values[index];
    for (var action = 0u; action < node.child_count; action = action + 1u) {
        let child_slot = parent_children[node.first_child + action];
        if child_slot < params.child_tile_start || child_slot >= params.child_tile_end {
            continue;
        }
        let child_local = child_slot - params.child_tile_start;
        let child_offset = child_local * params.combo_count + combo;
        let hero_child_value = child_hero_values[child_offset];
        let villain_child_value = child_villain_values[child_offset];
        if node.kind == 1u {
            hero_value = hero_value + hero_child_value;
            villain_value = villain_value + villain_child_value;
        } else if node.acting_player == 0u {
            let probability = strategy_from_reach(index, child_offset, 0u);
            if params.value_player == 0u {
                hero_value = max(hero_value, hero_child_value);
            } else {
                hero_value = hero_value + probability * hero_child_value;
            }
            villain_value = villain_value + villain_child_value;
        } else {
            let probability = strategy_from_reach(index, child_offset, 1u);
            hero_value = hero_value + hero_child_value;
            if params.value_player == 1u {
                villain_value = max(villain_value, villain_child_value);
            } else {
                villain_value = villain_value + probability * villain_child_value;
            }
        }
    }
    parent_hero_values[index] = hero_value;
    parent_villain_values[index] = villain_value;
}
"#;

const PUBLIC_TREE_LAYER_OUTPUT_SHADER: &str = r#"
struct Combo { cards: array<u32, 2>, };
struct Edge {
    parent: u32,
    child: u32,
    action: u32,
    card: u32,
};
struct TreeNode {
    kind: u32,
    acting_player: u32,
    public_infoset: u32,
    first_child: u32,
    child_count: u32,
    terminal_kind: u32,
    showdown_offset: u32,
    _pad0: u32,
    pot: f32,
    hero_invested: f32,
    _pad1: f32,
    _pad2: f32,
};
struct Params {
    combo_count: u32,
    node_count: u32,
    max_actions: u32,
    output_len: u32,
    value_player: u32,
    public_infoset_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> nodes: array<TreeNode>;
@group(0) @binding(1) var<storage, read> combos: array<Combo>;
@group(0) @binding(2) var<storage, read> hero_reaches: array<f32>;
@group(0) @binding(3) var<storage, read> villain_reaches: array<f32>;
@group(0) @binding(4) var<storage, read_write> hero_aggregates: array<f32>;
@group(0) @binding(5) var<storage, read_write> villain_aggregates: array<f32>;
@group(0) @binding(6) var<storage, read_write> action_values: array<f32>;
@group(0) @binding(7) var<storage, read> combo_live: array<f32>;
@group(0) @binding(8) var<storage, read> edges: array<Edge>;
@group(0) @binding(9) var<storage, read> child_hero_values: array<f32>;
@group(0) @binding(10) var<storage, read> child_villain_values: array<f32>;
@group(0) @binding(11) var<storage, read_write> action_weights: array<f32>;
@group(0) @binding(12) var<storage, read_write> reach_weights: array<f32>;
@group(0) @binding(13) var<storage, read_write> strategy_weights: array<f32>;
@group(0) @binding(14) var<uniform> params: Params;

@compute @workgroup_size(64)
fn decision_aggregate_tile(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * params.output_len;
    let slots = 53u;
    let value_count = params.node_count * slots;
    if index >= value_count {
        return;
    }
    let node_slot = index / slots;
    let slot = index % slots;
    let node = nodes[node_slot];
    if node.kind != 0u {
        return;
    }
    var hero_sum = 0.0;
    var villain_sum = 0.0;
    if slot == 0u {
        for (var combo = 0u; combo < params.combo_count; combo = combo + 1u) {
            let offset = node_slot * params.combo_count + combo;
            hero_sum = hero_sum + hero_reaches[offset];
            villain_sum = villain_sum + villain_reaches[offset];
        }
    } else {
        let card = slot - 1u;
        for (var combo = 0u; combo < params.combo_count; combo = combo + 1u) {
            let private_combo = combos[combo];
            if private_combo.cards[0] == card || private_combo.cards[1] == card {
                let offset = node_slot * params.combo_count + combo;
                hero_sum = hero_sum + hero_reaches[offset];
                villain_sum = villain_sum + villain_reaches[offset];
            }
        }
    }
    let aggregate_index = node.public_infoset * slots + slot;
    hero_aggregates[aggregate_index] = hero_sum;
    villain_aggregates[aggregate_index] = villain_sum;
}

@compute @workgroup_size(64)
fn denominator_tile(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * params.output_len;
    let value_count = params.node_count * params.combo_count;
    if index >= value_count {
        return;
    }
    let combo = index % params.combo_count;
    let node_slot = index / params.combo_count;
    let node = nodes[node_slot];
    if node.kind != 0u || node.acting_player != params.value_player {
        return;
    }
    if combo_live[index] <= 0.0 {
        let private_infoset = node.public_infoset * params.combo_count * 2u
            + params.value_player * params.combo_count
            + combo;
        reach_weights[private_infoset] = 0.0;
        return;
    }
    let private_combo = combos[combo];
    let aggregate_base = node.public_infoset * 53u;
    var value_weight = 0.0;
    if node.acting_player == 0u {
        let self_reach = villain_reaches[index];
        value_weight = villain_aggregates[aggregate_base]
            - villain_aggregates[aggregate_base + private_combo.cards[0] + 1u]
            - villain_aggregates[aggregate_base + private_combo.cards[1] + 1u]
            + self_reach;
    } else {
        let self_reach = hero_reaches[index];
        value_weight = hero_aggregates[aggregate_base]
            - hero_aggregates[aggregate_base + private_combo.cards[0] + 1u]
            - hero_aggregates[aggregate_base + private_combo.cards[1] + 1u]
            + self_reach;
    }
    let private_infoset = node.public_infoset * params.combo_count * 2u
        + params.value_player * params.combo_count
        + combo;
    reach_weights[private_infoset] = node._pad1 * value_weight;
}

@compute @workgroup_size(64)
fn action_edge_tile(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * params.output_len;
    let edge_count = params.node_count;
    let value_count = edge_count * params.combo_count;
    if index >= value_count {
        return;
    }
    let combo = index % params.combo_count;
    let edge_slot = index / params.combo_count;
    let edge = edges[edge_slot];
    let node = nodes[edge.parent];
    if node.kind != 0u || node.acting_player != params.value_player {
        return;
    }
    let private_infoset = node.public_infoset * params.combo_count * 2u
        + params.value_player * params.combo_count
        + combo;
    let action_index = private_infoset * params.max_actions + edge.action;
    let action_len = params.output_len;
    let child_offset = edge.child * params.combo_count + combo;
    let action_value = select(
        child_villain_values[child_offset],
        child_hero_values[child_offset],
        params.value_player == 0u
    );
    action_values[action_index] = action_value;
    action_weights[action_index] = reach_weights[private_infoset];
    if edge.action == 0u {
        var strategy_weight = node._pad1 * hero_reaches[edge.parent * params.combo_count + combo];
        if node.acting_player == 1u {
            strategy_weight = node._pad1 * villain_reaches[edge.parent * params.combo_count + combo];
        }
        strategy_weights[private_infoset] = strategy_weight;
    }
}
"#;

#[derive(Debug)]
pub enum GpuCfrError {
    NoAdapter,
    RequestDevice(String),
    MapFailed(String),
}

pub struct GpuDenseCfrBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
    adapter_features: wgpu::Features,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    public_tree_cfr_update_pipeline: wgpu::ComputePipeline,
    public_tree_cfr_update_bind_group_layout: wgpu::BindGroupLayout,
    showdown_pipeline: wgpu::ComputePipeline,
    showdown_bind_group_layout: wgpu::BindGroupLayout,
    showdown_matrix_pipeline: wgpu::ComputePipeline,
    showdown_matrix_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_layer_reach_init_pipeline: wgpu::ComputePipeline,
    public_tree_layer_reach_edge_pipeline: wgpu::ComputePipeline,
    public_tree_layer_reach_init_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_layer_reach_edge_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_terminal_partial_pipeline: wgpu::ComputePipeline,
    public_tree_terminal_partial_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_terminal_reduce_pipeline: wgpu::ComputePipeline,
    public_tree_terminal_reduce_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_terminal_card_prefix_pipeline: Option<wgpu::ComputePipeline>,
    public_tree_terminal_card_prefix_bind_group_layout: Option<wgpu::BindGroupLayout>,
    public_tree_terminal_card_aggregate_reduce_pipeline: Option<wgpu::ComputePipeline>,
    public_tree_terminal_card_aggregate_reduce_bind_group_layout: Option<wgpu::BindGroupLayout>,
    public_tree_fold_aggregate_pipeline: wgpu::ComputePipeline,
    public_tree_fold_aggregate_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_fold_value_pipeline: wgpu::ComputePipeline,
    public_tree_fold_value_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_layer_backup_init_pipeline: wgpu::ComputePipeline,
    public_tree_layer_backup_child_pipeline: wgpu::ComputePipeline,
    public_tree_layer_backup_init_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_layer_backup_child_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_layer_decision_aggregate_pipeline: wgpu::ComputePipeline,
    public_tree_layer_denominator_pipeline: wgpu::ComputePipeline,
    public_tree_layer_action_edge_pipeline: wgpu::ComputePipeline,
    public_tree_layer_output_bind_group_layout: wgpu::BindGroupLayout,
}

pub struct GpuDenseCfrState {
    infosets: usize,
    actions: usize,
    variant: super::CfrVariant,
    legal_actions: Vec<u32>,
    legal_actions_buffer: wgpu::Buffer,
    regrets: wgpu::Buffer,
    prediction: wgpu::Buffer,
    strategy_sum: wgpu::Buffer,
}

pub struct GpuResidentDenseCfrSolver {
    config: DenseCfrConfig,
    state: GpuDenseCfrState,
    iterations: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuShowdownTask {
    pub cards: [u32; SHOWDOWN_CARDS],
}

unsafe impl bytemuck::Zeroable for GpuShowdownTask {}
unsafe impl bytemuck::Pod for GpuShowdownTask {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuPrivateCombo {
    pub cards: [u32; 2],
}

unsafe impl bytemuck::Zeroable for GpuPrivateCombo {}
unsafe impl bytemuck::Pod for GpuPrivateCombo {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuFinalBoard {
    pub cards: [u32; 5],
}

unsafe impl bytemuck::Zeroable for GpuFinalBoard {}
unsafe impl bytemuck::Pod for GpuFinalBoard {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuPublicTreeNode {
    pub kind: u32,
    pub acting_player: u32,
    pub public_infoset: u32,
    pub first_child: u32,
    pub child_count: u32,
    pub terminal_kind: u32,
    pub showdown_offset: u32,
    pub _pad0: u32,
    pub pot: f32,
    pub hero_invested: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

unsafe impl bytemuck::Zeroable for GpuPublicTreeNode {}
unsafe impl bytemuck::Pod for GpuPublicTreeNode {}

#[derive(Debug, Clone)]
pub struct GpuShowdownStrengthOrder {
    pub combo_order: Vec<u32>,
    pub combo_bounds: Vec<GpuShowdownComboBounds>,
    pub blocker_neighbors: Vec<u32>,
    pub blocker_neighbor_stride: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuShowdownComboBounds {
    pub group_start: u32,
    pub group_end: u32,
    pub legal: u32,
    pub _pad0: u32,
}

unsafe impl bytemuck::Zeroable for GpuShowdownComboBounds {}
unsafe impl bytemuck::Pod for GpuShowdownComboBounds {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GpuPublicTreeEdge {
    parent: u32,
    child: u32,
    action: u32,
    card: u32,
}

unsafe impl bytemuck::Zeroable for GpuPublicTreeEdge {}
unsafe impl bytemuck::Pod for GpuPublicTreeEdge {}

struct GpuPublicTreeIterationContext {
    nodes_len: usize,
    combos_len: usize,
    actions: usize,
    action_len: usize,
    output_len: usize,
    node_combo_len: usize,
    public_infoset_count: usize,
    layered: GpuPublicTreeLayered,
    combo_buffer: wgpu::Buffer,
    root_weights_buffer: wgpu::Buffer,
    action_values_buffer: wgpu::Buffer,
    action_weights_buffer: wgpu::Buffer,
    reach_weights_buffer: wgpu::Buffer,
    strategy_weights_buffer: wgpu::Buffer,
    layer_tiles: Vec<Vec<GpuPublicTreeLayerTileBuffers>>,
    fold_terminal_nodes: Vec<u32>,
    showdown_terminal_nodes: Vec<u32>,
    terminal_tile_count: usize,
    terminal_board_tile_count: usize,
    terminal_chunk_size: usize,
    terminal_blocker_neighbors_buffer: wgpu::Buffer,
    terminal_blocker_neighbor_stride: usize,
    terminal_prefix_pair_budget: usize,
    terminal_prefix_pairs_buffer: wgpu::Buffer,
    terminal_card_prefix_pair_budget: usize,
    terminal_card_prefix_pairs_buffer: wgpu::Buffer,
    hero_decision_aggregates_buffer: wgpu::Buffer,
    villain_decision_aggregates_buffer: wgpu::Buffer,
}

struct GpuPublicTreeLayerTileBuffers {
    node_start: usize,
    node_end: usize,
    node_buffer: wgpu::Buffer,
    child_buffer: wgpu::Buffer,
    fold_terminal_nodes: Vec<u32>,
    showdown_terminal_groups: Vec<GpuTerminalGroupBufferCache>,
    hero_reaches_buffer: wgpu::Buffer,
    villain_reaches_buffer: wgpu::Buffer,
    combo_live_buffer: wgpu::Buffer,
    hero_values_buffer: wgpu::Buffer,
    villain_values_buffer: wgpu::Buffer,
}

struct GpuPublicTreeOutputBuffers {
    action_values: wgpu::Buffer,
    action_weights: wgpu::Buffer,
    reach_weights: wgpu::Buffer,
    strategy_weights: wgpu::Buffer,
}

#[derive(Debug, Clone)]
struct GpuPublicTreeLayer {
    nodes: Vec<GpuPublicTreeNode>,
    children: Vec<u32>,
    child_cards: Vec<u32>,
}

#[derive(Debug, Clone)]
struct GpuPublicTreeLayerTile {
    node_start: usize,
}

#[derive(Debug, Clone)]
struct GpuPublicTreeLayerEdgeTile {
    parent_layer: usize,
    child_layer: usize,
    parent_tile: GpuPublicTreeLayerTile,
    child_tile: GpuPublicTreeLayerTile,
    edges: Vec<GpuPublicTreeEdge>,
}

#[derive(Debug, Clone)]
struct GpuPublicTreeLayered {
    layers: Vec<GpuPublicTreeLayer>,
    max_layer_nodes: usize,
    node_tile_size: usize,
    max_layer_tiles: usize,
    reach_edge_tiles: Vec<GpuPublicTreeLayerEdgeTile>,
}

struct GpuTerminalGroupCache {
    board_count: usize,
    terminal_nodes: Vec<u32>,
    combo_order: Vec<u32>,
    combo_bounds: Vec<GpuShowdownComboBounds>,
}

struct GpuTerminalGroupBufferCache {
    board_count: usize,
    terminal_nodes_len: usize,
    terminal_nodes_buffer: wgpu::Buffer,
    combo_order_buffer: wgpu::Buffer,
    combo_bounds_buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GpuPublicTreeParams {
    combo_count: u32,
    node_count: u32,
    max_actions: u32,
    output_len: u32,
    pair_start: u32,
    chunk_pairs: u32,
    _pad0: u32,
    _pad1: u32,
}

unsafe impl bytemuck::Zeroable for GpuPublicTreeParams {}
unsafe impl bytemuck::Pod for GpuPublicTreeParams {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GpuTerminalChunkParams {
    terminal_count: u32,
    x_invocations: u32,
    terminal_start: u32,
    _pad0: u32,
}

unsafe impl bytemuck::Zeroable for GpuTerminalChunkParams {}
unsafe impl bytemuck::Pod for GpuTerminalChunkParams {}

pub fn showdown_strength_order(
    combos: &[GpuPrivateCombo],
    final_boards: &[GpuFinalBoard],
) -> GpuShowdownStrengthOrder {
    let (combo_order, combo_bounds) = showdown_strength_order_data(combos, final_boards);
    let (blocker_neighbors, blocker_neighbor_stride) = showdown_blocker_neighbors(combos);

    GpuShowdownStrengthOrder {
        combo_order,
        combo_bounds,
        blocker_neighbors,
        blocker_neighbor_stride,
    }
}

fn showdown_strength_order_data(
    combos: &[GpuPrivateCombo],
    final_boards: &[GpuFinalBoard],
) -> (Vec<u32>, Vec<GpuShowdownComboBounds>) {
    let combo_count = combos.len();
    let mut combo_order = Vec::with_capacity(final_boards.len() * combo_count);
    let mut combo_bounds =
        vec![GpuShowdownComboBounds::default(); final_boards.len() * combo_count];

    for (board_index, board) in final_boards.iter().enumerate() {
        let mut ranked: Vec<_> = combos
            .iter()
            .enumerate()
            .filter_map(|(combo_index, combo)| {
                (!combo_hits_final_board_for_order(*combo, *board)).then_some((
                    evaluate_combo_final_board(*combo, *board),
                    combo_index as u32,
                ))
            })
            .collect();
        ranked.sort_unstable_by_key(|(strength, combo_index)| (*strength, *combo_index));

        let board_order_offset = combo_order.len();
        combo_order.extend(ranked.iter().map(|(_, combo_index)| *combo_index));
        combo_order.resize(board_order_offset + combo_count, u32::MAX);

        let board_combo_offset = board_index * combo_count;
        let mut group_start = 0usize;
        while group_start < ranked.len() {
            let strength = ranked[group_start].0;
            let mut group_end = group_start + 1;
            while group_end < ranked.len() && ranked[group_end].0 == strength {
                group_end += 1;
            }
            for &(_, combo_index) in &ranked[group_start..group_end] {
                let slot = board_combo_offset + combo_index as usize;
                combo_bounds[slot] = GpuShowdownComboBounds {
                    group_start: group_start as u32,
                    group_end: group_end as u32,
                    legal: 1,
                    _pad0: 0,
                };
            }
            group_start = group_end;
        }
    }
    (combo_order, combo_bounds)
}

fn showdown_blocker_neighbors(combos: &[GpuPrivateCombo]) -> (Vec<u32>, usize) {
    let combo_count = combos.len();
    let blocker_neighbor_stride = combos
        .iter()
        .map(|&combo| {
            combos
                .iter()
                .filter(|&&other| combos_collide_for_order(combo, other))
                .count()
        })
        .max()
        .unwrap_or(1)
        .max(1);
    let mut blocker_neighbors = vec![u32::MAX; combo_count * blocker_neighbor_stride];
    for (combo_index, &combo) in combos.iter().enumerate() {
        let mut slot = combo_index * blocker_neighbor_stride;
        for (other_index, &other) in combos.iter().enumerate() {
            if combos_collide_for_order(combo, other) {
                blocker_neighbors[slot] = other_index as u32;
                slot += 1;
            }
        }
    }
    (blocker_neighbors, blocker_neighbor_stride)
}

fn terminal_group_caches(
    nodes: &[GpuPublicTreeNode],
    terminal_nodes: &[u32],
    combos: &[GpuPrivateCombo],
    showdown_boards: &[GpuFinalBoard],
) -> Vec<GpuTerminalGroupCache> {
    const MAX_TERMINAL_GROUP_TABLE_BYTES: usize = 124 * 1024 * 1024;

    let mut groups: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    for &node_index in terminal_nodes {
        let node = nodes[node_index as usize];
        groups
            .entry(node._pad0 as usize)
            .or_default()
            .push(node_index);
    }

    let mut caches = Vec::new();
    for (board_count, terminal_nodes) in groups {
        if board_count == 0 {
            continue;
        }

        let table_bytes_per_terminal = board_count.saturating_mul(combos.len()).saturating_mul(
            std::mem::size_of::<u32>() + std::mem::size_of::<GpuShowdownComboBounds>(),
        );
        let terminals_per_group = (MAX_TERMINAL_GROUP_TABLE_BYTES / table_bytes_per_terminal)
            .max(1)
            .min(terminal_nodes.len());

        for terminal_chunk in terminal_nodes.chunks(terminals_per_group) {
            let mut combo_order =
                Vec::with_capacity(terminal_chunk.len() * board_count * combos.len());
            let mut combo_bounds =
                Vec::with_capacity(terminal_chunk.len() * board_count * combos.len());
            for &node_index in terminal_chunk {
                let node = nodes[node_index as usize];
                let board_base = node.showdown_offset as usize;
                let boards = &showdown_boards[board_base..board_base + board_count];
                let (node_combo_order, node_combo_bounds) =
                    showdown_strength_order_data(combos, boards);
                combo_order.extend(node_combo_order);
                combo_bounds.extend(node_combo_bounds);
            }
            caches.push(GpuTerminalGroupCache {
                board_count,
                terminal_nodes: terminal_chunk.to_vec(),
                combo_order,
                combo_bounds,
            });
        }
    }
    caches
}

fn requested_wgpu_backends() -> Option<wgpu::Backends> {
    let value = std::env::var("POKEDR_WGPU_BACKENDS")
        .ok()
        .or_else(|| std::env::var("WGPU_BACKENDS").ok())?;
    let mut backends = wgpu::Backends::empty();
    for token in value
        .split(|character: char| character == ',' || character == ';' || character == '|')
        .flat_map(str::split_whitespace)
    {
        match token.to_ascii_lowercase().as_str() {
            "dx12" | "d3d12" | "directx12" => backends |= wgpu::Backends::DX12,
            "vulkan" | "vk" => backends |= wgpu::Backends::VULKAN,
            "gl" | "opengl" | "gles" => backends |= wgpu::Backends::GL,
            "metal" => backends |= wgpu::Backends::METAL,
            "webgpu" | "browser_webgpu" | "browser-webgpu" => {
                backends |= wgpu::Backends::BROWSER_WEBGPU;
            }
            "noop" => backends |= wgpu::Backends::NOOP,
            _ => {}
        }
    }
    (!backends.is_empty()).then_some(backends)
}

fn public_tree_terminal_partial_shader_source(
    backend: wgpu::Backend,
) -> (&'static str, &'static str) {
    let force_parallel = std::env::var("POKEDR_GPU_TERMINAL_PARALLEL_PREFIX")
        .ok()
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "parallel"
            )
        })
        .unwrap_or(false);
    let allow_unsafe_dx12_parallel =
        std::env::var_os("POKEDR_GPU_UNSAFE_DX12_PARALLEL_PREFIX").is_some();
    if force_parallel && (!matches!(backend, wgpu::Backend::Dx12) || allow_unsafe_dx12_parallel) {
        return (PUBLIC_TREE_TERMINAL_PARTIAL_SHADER, "parallel-forced");
    }
    if matches!(backend, wgpu::Backend::Dx12) {
        return (PUBLIC_TREE_TERMINAL_PARTIAL_SERIAL_SHADER, "serial");
    }
    (PUBLIC_TREE_TERMINAL_PARTIAL_SHADER, "parallel")
}

fn trace_pipeline_step(step: &str) {
    if std::env::var_os("POKEDR_GPU_PIPELINE_TRACE").is_some() {
        eprintln!("pokedr: gpu pipeline {step}");
    }
}

fn terminal_card_prefix_requested() -> bool {
    std::env::var("POKEDR_GPU_TERMINAL_CARD_PREFIX")
        .ok()
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

async fn request_gpu_adapter() -> Result<wgpu::Adapter, GpuCfrError> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    if let Some(backends) = requested_wgpu_backends() {
        descriptor.backends = backends;
    } else {
        #[cfg(windows)]
        {
            descriptor.backends = wgpu::Backends::DX12;
        }
        #[cfg(not(windows))]
        {
            descriptor.backends = wgpu::Backends::VULKAN;
        }
    }
    descriptor
        .flags
        .insert(wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER);
    if std::env::var_os("POKEDR_GPU_INIT_TRACE").is_some() {
        eprintln!(
            "pokedr: gpu init creating instance backends={:?}",
            descriptor.backends
        );
    }
    let instance = wgpu::Instance::new(descriptor);
    if std::env::var_os("POKEDR_GPU_INIT_TRACE").is_some() {
        eprintln!("pokedr: gpu init requesting high-performance adapter");
    }
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .map_err(|_| GpuCfrError::NoAdapter)?;
    if std::env::var_os("POKEDR_GPU_INIT_TRACE").is_some() {
        let adapter_info = adapter.get_info();
        eprintln!(
            "pokedr: gpu init adapter name={} backend={:?}",
            adapter_info.name, adapter_info.backend
        );
    }
    Ok(adapter)
}

fn combos_collide_for_order(left: GpuPrivateCombo, right: GpuPrivateCombo) -> bool {
    left.cards[0] == right.cards[0]
        || left.cards[0] == right.cards[1]
        || left.cards[1] == right.cards[0]
        || left.cards[1] == right.cards[1]
}

fn combo_hits_final_board_for_order(combo: GpuPrivateCombo, board: GpuFinalBoard) -> bool {
    board.cards.contains(&combo.cards[0]) || board.cards.contains(&combo.cards[1])
}

fn evaluate_combo_final_board(combo: GpuPrivateCombo, board: GpuFinalBoard) -> u32 {
    let cards = [
        Card::from_index(combo.cards[0] as u8),
        Card::from_index(combo.cards[1] as u8),
        Card::from_index(board.cards[0] as u8),
        Card::from_index(board.cards[1] as u8),
        Card::from_index(board.cards[2] as u8),
        Card::from_index(board.cards[3] as u8),
        Card::from_index(board.cards[4] as u8),
    ];
    evaluate(&cards).raw()
}

pub struct GpuRootTerminalValues {
    pub action_values: Vec<f32>,
    pub reach_weights: Vec<f32>,
    pub strategy_weights: Vec<f32>,
    pub root_hero_values: Vec<f32>,
    pub root_villain_values: Vec<f32>,
}

impl GpuDenseCfrBackend {
    pub fn new() -> Result<Self, GpuCfrError> {
        pollster::block_on(Self::new_async())
    }

    pub fn probe_adapter() -> Result<(wgpu::AdapterInfo, bool), GpuCfrError> {
        pollster::block_on(Self::probe_adapter_async())
    }

    pub async fn probe_adapter_async() -> Result<(wgpu::AdapterInfo, bool), GpuCfrError> {
        let adapter = request_gpu_adapter().await?;
        let supports_shader_float32_atomic = adapter
            .features()
            .contains(wgpu::Features::SHADER_FLOAT32_ATOMIC);
        Ok((adapter.get_info(), supports_shader_float32_atomic))
    }

    pub async fn new_async() -> Result<Self, GpuCfrError> {
        let adapter = request_gpu_adapter().await?;
        let adapter_info = adapter.get_info();
        let adapter_features = adapter.features();
        let required_limits = adapter.limits();
        let terminal_immediate_size = std::mem::size_of::<GpuTerminalChunkParams>() as u32;
        if !adapter_features.contains(wgpu::Features::IMMEDIATES)
            || required_limits.max_immediate_size < terminal_immediate_size
        {
            return Err(GpuCfrError::RequestDevice(format!(
                "GPU backend requires {} bytes of immediates for streamed terminal CFR; adapter supports immediates={} max_immediate_size={}",
                terminal_immediate_size,
                adapter_features.contains(wgpu::Features::IMMEDIATES),
                required_limits.max_immediate_size,
            )));
        }
        if std::env::var_os("POKEDR_GPU_INIT_TRACE").is_some() {
            eprintln!("pokedr: gpu init requesting device");
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("pokedr dense CFR device"),
                required_features: wgpu::Features::IMMEDIATES,
                required_limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|error| GpuCfrError::RequestDevice(error.to_string()))?;
        if std::env::var_os("POKEDR_GPU_INIT_TRACE").is_some() {
            eprintln!("pokedr: gpu init creating pipelines");
        }
        trace_pipeline_step("dense_cfr_update:start");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dense CFR update shader"),
            source: wgpu::ShaderSource::Wgsl(CFR_UPDATE_SHADER.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dense CFR bind group layout"),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, false),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(6, true),
                storage_entry(7, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dense CFR pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dense CFR update pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        trace_pipeline_step("dense_cfr_update:done");
        let public_tree_cfr_update_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree CFR update shader"),
                source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_CFR_UPDATE_SHADER.into()),
            });
        let public_tree_cfr_update_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("public tree CFR update bind group layout"),
                entries: &[
                    storage_entry(0, false),
                    storage_entry(1, false),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, true),
                    storage_entry(5, true),
                    storage_entry(6, true),
                    storage_entry(7, true),
                    storage_entry(8, false),
                ],
            });
        let public_tree_cfr_update_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("public tree CFR update pipeline layout"),
                bind_group_layouts: &[Some(&public_tree_cfr_update_bind_group_layout)],
                immediate_size: 0,
            });
        let public_tree_cfr_update_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree CFR update pipeline"),
                layout: Some(&public_tree_cfr_update_pipeline_layout),
                module: &public_tree_cfr_update_shader,
                entry_point: Some("update"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let showdown_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("showdown equity shader"),
            source: wgpu::ShaderSource::Wgsl(SHOWDOWN_SHADER.into()),
        });
        let showdown_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("showdown equity bind group layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    storage_entry(2, true),
                ],
            });
        let showdown_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("showdown equity pipeline layout"),
                bind_group_layouts: &[Some(&showdown_bind_group_layout)],
                immediate_size: 0,
            });
        let showdown_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("showdown equity pipeline"),
            layout: Some(&showdown_pipeline_layout),
            module: &showdown_shader,
            entry_point: Some("showdown"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let showdown_matrix_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("showdown matrix shader"),
            source: wgpu::ShaderSource::Wgsl(SHOWDOWN_MATRIX_SHADER.into()),
        });
        let showdown_matrix_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("showdown matrix bind group layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                    storage_entry(3, true),
                ],
            });
        let showdown_matrix_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("showdown matrix pipeline layout"),
                bind_group_layouts: &[Some(&showdown_matrix_bind_group_layout)],
                immediate_size: 0,
            });
        let showdown_matrix_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("showdown matrix pipeline"),
                layout: Some(&showdown_matrix_pipeline_layout),
                module: &showdown_matrix_shader,
                entry_point: Some("matrix"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let public_tree_layer_reach_init_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree layer reach init shader"),
                source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_LAYER_REACH_INIT_SHADER.into()),
            });
        let public_tree_layer_reach_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree layer reach shader"),
                source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_LAYER_REACH_SHADER.into()),
            });
        let public_tree_layer_reach_init_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("public tree layer reach init bind group layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    storage_entry(2, false),
                    storage_entry(3, false),
                    uniform_entry(4),
                ],
            });
        let public_tree_layer_reach_edge_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("public tree layer reach edge bind group layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, true),
                    storage_entry(5, true),
                    storage_entry(6, true),
                    storage_entry(7, true),
                    storage_entry(8, false),
                    storage_entry(9, false),
                    storage_entry(10, false),
                    storage_entry(11, true),
                    uniform_entry(12),
                ],
            });
        let public_tree_layer_reach_init_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("public tree layer reach init pipeline layout"),
                bind_group_layouts: &[Some(&public_tree_layer_reach_init_bind_group_layout)],
                immediate_size: 0,
            });
        let public_tree_layer_reach_edge_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("public tree layer reach edge pipeline layout"),
                bind_group_layouts: &[Some(&public_tree_layer_reach_edge_bind_group_layout)],
                immediate_size: 0,
            });
        let public_tree_layer_reach_init_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree layer reach init pipeline"),
                layout: Some(&public_tree_layer_reach_init_pipeline_layout),
                module: &public_tree_layer_reach_init_shader,
                entry_point: Some("reach_init_tile"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let public_tree_layer_reach_edge_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree layer reach edge pipeline"),
                layout: Some(&public_tree_layer_reach_edge_pipeline_layout),
                module: &public_tree_layer_reach_shader,
                entry_point: Some("reach_edge_tile"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        trace_pipeline_step("public_tree_terminal_partial:start");
        let (partial_shader_source, partial_shader_mode) =
            public_tree_terminal_partial_shader_source(adapter_info.backend);
        if std::env::var_os("POKEDR_GPU_PIPELINE_TRACE").is_some() {
            eprintln!(
                "pokedr: gpu pipeline public_tree_terminal_partial backend={:?} mode={} forced_parallel_env={}",
                adapter_info.backend,
                partial_shader_mode,
                std::env::var("POKEDR_GPU_TERMINAL_PARALLEL_PREFIX")
                    .ok()
                    .unwrap_or_default(),
            );
        }
        let public_tree_terminal_partial_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree terminal partial shader"),
                source: wgpu::ShaderSource::Wgsl(partial_shader_source.into()),
            });
        let public_tree_terminal_partial_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("public tree terminal partial bind group layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, true),
                    storage_entry(5, true),
                    storage_entry(6, false),
                    uniform_entry(7),
                ],
            });
        let public_tree_terminal_partial_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("public tree terminal partial pipeline layout"),
                bind_group_layouts: &[Some(&public_tree_terminal_partial_bind_group_layout)],
                immediate_size: terminal_immediate_size,
            });
        let public_tree_terminal_partial_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree terminal partial pipeline"),
                layout: Some(&public_tree_terminal_partial_pipeline_layout),
                module: &public_tree_terminal_partial_shader,
                entry_point: Some("terminal_partial"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        trace_pipeline_step("public_tree_terminal_partial:done");
        trace_pipeline_step("public_tree_terminal_reduce:start");
        let public_tree_terminal_reduce_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree terminal reduce shader"),
                source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_TERMINAL_REDUCE_SHADER.into()),
            });
        let public_tree_terminal_reduce_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("public tree terminal reduce bind group layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, true),
                    storage_entry(5, true),
                    storage_entry(6, true),
                    storage_entry(7, false),
                    storage_entry(8, false),
                    uniform_entry(9),
                ],
            });
        let public_tree_terminal_reduce_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("public tree terminal reduce pipeline layout"),
                bind_group_layouts: &[Some(&public_tree_terminal_reduce_bind_group_layout)],
                immediate_size: terminal_immediate_size,
            });
        let public_tree_terminal_reduce_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree terminal reduce pipeline"),
                layout: Some(&public_tree_terminal_reduce_pipeline_layout),
                module: &public_tree_terminal_reduce_shader,
                entry_point: Some("terminal_reduce"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        trace_pipeline_step("public_tree_terminal_reduce:done");
        let (
            public_tree_terminal_card_prefix_pipeline,
            public_tree_terminal_card_prefix_bind_group_layout,
            public_tree_terminal_card_aggregate_reduce_pipeline,
            public_tree_terminal_card_aggregate_reduce_bind_group_layout,
        ) = if terminal_card_prefix_requested() {
            trace_pipeline_step("public_tree_terminal_card_prefix:start");
            let card_prefix_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree terminal card prefix shader"),
                source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_TERMINAL_CARD_PREFIX_SHADER.into()),
            });
            let card_prefix_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("public tree terminal card prefix bind group layout"),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, true),
                        storage_entry(2, true),
                        storage_entry(3, true),
                        storage_entry(4, true),
                        storage_entry(5, true),
                        storage_entry(6, true),
                        storage_entry(7, false),
                        uniform_entry(8),
                    ],
                });
            let card_prefix_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("public tree terminal card prefix pipeline layout"),
                    bind_group_layouts: &[Some(&card_prefix_bind_group_layout)],
                    immediate_size: 0,
                });
            let card_prefix_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("public tree terminal card prefix pipeline"),
                    layout: Some(&card_prefix_pipeline_layout),
                    module: &card_prefix_shader,
                    entry_point: Some("terminal_card_prefix"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
            trace_pipeline_step("public_tree_terminal_card_prefix:done");

            trace_pipeline_step("public_tree_terminal_card_aggregate_reduce:start");
            let card_reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree terminal card aggregate reduce shader"),
                source: wgpu::ShaderSource::Wgsl(
                    PUBLIC_TREE_TERMINAL_CARD_AGGREGATE_REDUCE_SHADER.into(),
                ),
            });
            let card_reduce_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("public tree terminal card aggregate reduce bind group layout"),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, true),
                        storage_entry(2, true),
                        storage_entry(3, true),
                        storage_entry(4, true),
                        storage_entry(5, true),
                        storage_entry(6, true),
                        storage_entry(7, true),
                        storage_entry(8, false),
                        storage_entry(9, false),
                        uniform_entry(10),
                    ],
                });
            let card_reduce_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("public tree terminal card aggregate reduce pipeline layout"),
                    bind_group_layouts: &[Some(&card_reduce_bind_group_layout)],
                    immediate_size: 0,
                });
            let card_reduce_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("public tree terminal card aggregate reduce pipeline"),
                    layout: Some(&card_reduce_pipeline_layout),
                    module: &card_reduce_shader,
                    entry_point: Some("terminal_reduce"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
            trace_pipeline_step("public_tree_terminal_card_aggregate_reduce:done");
            (
                Some(card_prefix_pipeline),
                Some(card_prefix_bind_group_layout),
                Some(card_reduce_pipeline),
                Some(card_reduce_bind_group_layout),
            )
        } else {
            (None, None, None, None)
        };
        let public_tree_fold_aggregate_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree fold aggregate shader"),
                source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_FOLD_AGGREGATE_SHADER.into()),
            });
        let public_tree_fold_aggregate_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("public tree fold aggregate bind group layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, false),
                    storage_entry(5, false),
                    uniform_entry(6),
                ],
            });
        let public_tree_fold_aggregate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("public tree fold aggregate pipeline layout"),
                bind_group_layouts: &[Some(&public_tree_fold_aggregate_bind_group_layout)],
                immediate_size: 0,
            });
        let public_tree_fold_aggregate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree fold aggregate pipeline"),
                layout: Some(&public_tree_fold_aggregate_pipeline_layout),
                module: &public_tree_fold_aggregate_shader,
                entry_point: Some("fold_aggregate"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let public_tree_fold_value_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree fold value shader"),
                source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_FOLD_VALUE_SHADER.into()),
            });
        let public_tree_fold_value_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("public tree fold value bind group layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, true),
                    storage_entry(5, true),
                    storage_entry(6, true),
                    storage_entry(7, false),
                    storage_entry(8, false),
                    storage_entry(9, true),
                    uniform_entry(10),
                ],
            });
        let public_tree_fold_value_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("public tree fold value pipeline layout"),
                bind_group_layouts: &[Some(&public_tree_fold_value_bind_group_layout)],
                immediate_size: 0,
            });
        let public_tree_fold_value_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree fold value pipeline"),
                layout: Some(&public_tree_fold_value_pipeline_layout),
                module: &public_tree_fold_value_shader,
                entry_point: Some("fold_value"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let public_tree_layer_backup_init_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree layer backup init shader"),
                source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_LAYER_BACKUP_INIT_SHADER.into()),
            });
        let public_tree_layer_backup_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree layer backup shader"),
                source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_LAYER_BACKUP_SHADER.into()),
            });
        let public_tree_layer_backup_init_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("public tree layer backup init bind group layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    storage_entry(2, false),
                    uniform_entry(3),
                ],
            });
        let public_tree_layer_backup_child_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("public tree layer backup child bind group layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, true),
                    storage_entry(5, true),
                    storage_entry(6, true),
                    storage_entry(7, true),
                    storage_entry(8, false),
                    storage_entry(9, false),
                    uniform_entry(10),
                ],
            });
        let public_tree_layer_backup_init_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("public tree layer backup init pipeline layout"),
                bind_group_layouts: &[Some(&public_tree_layer_backup_init_bind_group_layout)],
                immediate_size: 0,
            });
        let public_tree_layer_backup_child_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("public tree layer backup child pipeline layout"),
                bind_group_layouts: &[Some(&public_tree_layer_backup_child_bind_group_layout)],
                immediate_size: 0,
            });
        let public_tree_layer_backup_init_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree layer backup init pipeline"),
                layout: Some(&public_tree_layer_backup_init_pipeline_layout),
                module: &public_tree_layer_backup_init_shader,
                entry_point: Some("backup_init_tile"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let public_tree_layer_backup_child_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree layer backup child pipeline"),
                layout: Some(&public_tree_layer_backup_child_pipeline_layout),
                module: &public_tree_layer_backup_shader,
                entry_point: Some("backup_child_tile"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let public_tree_layer_output_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree layer output shader"),
                source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_LAYER_OUTPUT_SHADER.into()),
            });
        let public_tree_layer_output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("public tree layer output bind group layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, false),
                    storage_entry(5, false),
                    storage_entry(6, false),
                    storage_entry(7, true),
                    storage_entry(8, true),
                    storage_entry(9, true),
                    storage_entry(10, true),
                    storage_entry(11, false),
                    storage_entry(12, false),
                    storage_entry(13, false),
                    uniform_entry(14),
                ],
            });
        let public_tree_layer_output_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("public tree layer output pipeline layout"),
                bind_group_layouts: &[Some(&public_tree_layer_output_bind_group_layout)],
                immediate_size: 0,
            });
        let public_tree_layer_decision_aggregate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree layer decision aggregate pipeline"),
                layout: Some(&public_tree_layer_output_pipeline_layout),
                module: &public_tree_layer_output_shader,
                entry_point: Some("decision_aggregate_tile"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let public_tree_layer_denominator_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree layer denominator pipeline"),
                layout: Some(&public_tree_layer_output_pipeline_layout),
                module: &public_tree_layer_output_shader,
                entry_point: Some("denominator_tile"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let public_tree_layer_action_edge_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree layer action edge pipeline"),
                layout: Some(&public_tree_layer_output_pipeline_layout),
                module: &public_tree_layer_output_shader,
                entry_point: Some("action_edge_tile"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Ok(Self {
            device,
            queue,
            adapter_info,
            adapter_features,
            pipeline,
            bind_group_layout,
            public_tree_cfr_update_pipeline,
            public_tree_cfr_update_bind_group_layout,
            showdown_pipeline,
            showdown_bind_group_layout,
            showdown_matrix_pipeline,
            showdown_matrix_bind_group_layout,
            public_tree_layer_reach_init_pipeline,
            public_tree_layer_reach_edge_pipeline,
            public_tree_layer_reach_init_bind_group_layout,
            public_tree_layer_reach_edge_bind_group_layout,
            public_tree_terminal_partial_pipeline,
            public_tree_terminal_partial_bind_group_layout,
            public_tree_terminal_reduce_pipeline,
            public_tree_terminal_reduce_bind_group_layout,
            public_tree_terminal_card_prefix_pipeline,
            public_tree_terminal_card_prefix_bind_group_layout,
            public_tree_terminal_card_aggregate_reduce_pipeline,
            public_tree_terminal_card_aggregate_reduce_bind_group_layout,
            public_tree_fold_aggregate_pipeline,
            public_tree_fold_aggregate_bind_group_layout,
            public_tree_fold_value_pipeline,
            public_tree_fold_value_bind_group_layout,
            public_tree_layer_backup_init_pipeline,
            public_tree_layer_backup_child_pipeline,
            public_tree_layer_backup_init_bind_group_layout,
            public_tree_layer_backup_child_bind_group_layout,
            public_tree_layer_decision_aggregate_pipeline,
            public_tree_layer_denominator_pipeline,
            public_tree_layer_action_edge_pipeline,
            public_tree_layer_output_bind_group_layout,
        })
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    pub fn adapter_features(&self) -> wgpu::Features {
        self.adapter_features
    }

    pub fn supports_shader_float32_atomic(&self) -> bool {
        self.adapter_features
            .contains(wgpu::Features::SHADER_FLOAT32_ATOMIC)
    }

    pub fn wait_idle(&self) -> Result<(), GpuCfrError> {
        self.profile_poll()
    }

    fn gpu_profile_enabled() -> bool {
        std::env::var_os("POKEDR_GPU_PROFILE").is_some()
    }

    fn profile_poll(&self) -> Result<(), GpuCfrError> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map(|_| ())
            .map_err(|error| GpuCfrError::MapFailed(error.to_string()))
    }

    fn finish_profile_phase(
        &self,
        encoder: wgpu::CommandEncoder,
        phase: &str,
        start: Option<Instant>,
    ) -> Result<wgpu::CommandEncoder, GpuCfrError> {
        let Some(start) = start else {
            return Ok(encoder);
        };
        self.queue.submit(Some(encoder.finish()));
        self.profile_poll()?;
        eprintln!(
            "pokedr: gpu profile phase={} elapsed_ms={:.3}",
            phase,
            start.elapsed().as_secs_f64() * 1000.0
        );
        Ok(self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("public tree profiled iteration encoder"),
            }))
    }

    fn submit_final_profile_phase(
        &self,
        encoder: wgpu::CommandEncoder,
        phase: &str,
        start: Option<Instant>,
    ) -> Result<(), GpuCfrError> {
        self.queue.submit(Some(encoder.finish()));
        if let Some(start) = start {
            self.profile_poll()?;
            eprintln!(
                "pokedr: gpu profile phase={} elapsed_ms={:.3}",
                phase,
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(())
    }

    pub fn update_all_infosets(
        &self,
        state: &mut DenseCfrState,
        action_values: &[f32],
        reach_weights: &[f32],
        strategy_weights: &[f32],
        iteration: usize,
    ) -> Result<(), GpuCfrError> {
        assert_eq!(action_values.len(), state.infosets * state.actions);
        assert_eq!(reach_weights.len(), state.infosets);
        assert_eq!(strategy_weights.len(), state.infosets);

        let params = [
            state.infosets as u32,
            state.actions as u32,
            variant_code(state.variant),
            iteration as u32,
            variant_dcfr_alpha(state.variant, iteration).to_bits(),
            variant_dcfr_gamma(state.variant, iteration).to_bits(),
            variant_prediction_eta(state.variant).to_bits(),
        ];
        let regrets = storage_buffer(&self.device, "regrets", &state.regrets);
        let prediction = storage_buffer(&self.device, "prediction", &state.prediction);
        let strategy_sum = storage_buffer(&self.device, "strategy sum", &state.strategy_sum);
        let action_values = readonly_buffer(&self.device, "action values", action_values);
        let reach_weights = readonly_buffer(&self.device, "reach weights", reach_weights);
        let strategy_weights = readonly_buffer(&self.device, "strategy weights", strategy_weights);
        let params = readonly_buffer(&self.device, "params", &params);
        let legal_actions = readonly_buffer(
            &self.device,
            "legal actions",
            &legal_actions_u32(&state.legal_actions),
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dense CFR bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                bind_entry(0, &regrets),
                bind_entry(1, &strategy_sum),
                bind_entry(2, &action_values),
                bind_entry(3, &reach_weights),
                bind_entry(4, &strategy_weights),
                bind_entry(5, &params),
                bind_entry(6, &legal_actions),
                bind_entry(7, &prediction),
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dense CFR update encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dense CFR update pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (state.infosets as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        let regret_readback = readback_buffer(&self.device, state.regrets.len());
        let prediction_readback = state
            .variant
            .uses_prediction()
            .then(|| readback_buffer(&self.device, state.prediction.len()));
        let strategy_readback = readback_buffer(&self.device, state.strategy_sum.len());
        copy_buffer(
            &mut encoder,
            &regrets,
            &regret_readback,
            state.regrets.len(),
        );
        if let Some(prediction_readback) = &prediction_readback {
            copy_buffer(
                &mut encoder,
                &prediction,
                prediction_readback,
                state.prediction.len(),
            );
        }
        copy_buffer(
            &mut encoder,
            &strategy_sum,
            &strategy_readback,
            state.strategy_sum.len(),
        );
        let submission = self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;

        let regrets_len = state.regrets.len();
        let strategy_sum_len = state.strategy_sum.len();
        let updated_regrets = read_f32_buffer(&self.device, &regret_readback, regrets_len)?;
        if let Some(prediction_readback) = &prediction_readback {
            let updated_prediction =
                read_f32_buffer(&self.device, prediction_readback, state.prediction.len())?;
            state.prediction.copy_from_slice(&updated_prediction);
        }
        let updated_strategy_sum =
            read_f32_buffer(&self.device, &strategy_readback, strategy_sum_len)?;
        state.regrets.copy_from_slice(&updated_regrets);
        state.strategy_sum.copy_from_slice(&updated_strategy_sum);
        Ok(())
    }

    pub fn showdown_equities(&self, tasks: &[GpuShowdownTask]) -> Result<Vec<f32>, GpuCfrError> {
        if tasks.is_empty() {
            return Ok(Vec::new());
        }

        let task_buffer = readonly_buffer(&self.device, "showdown tasks", tasks);
        let mut output = vec![0.0f32; tasks.len()];
        let output_buffer = storage_buffer(&self.device, "showdown equities", &output);
        let params = readonly_buffer(&self.device, "showdown params", &[tasks.len() as u32]);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("showdown equity bind group"),
            layout: &self.showdown_bind_group_layout,
            entries: &[
                bind_entry(0, &task_buffer),
                bind_entry(1, &output_buffer),
                bind_entry(2, &params),
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("showdown equity encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("showdown equity pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.showdown_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (tasks.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        let readback = readback_buffer(&self.device, tasks.len());
        copy_buffer(&mut encoder, &output_buffer, &readback, tasks.len());
        let submission = self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;
        output = read_f32_buffer(&self.device, &readback, tasks.len())?;
        Ok(output)
    }

    pub fn showdown_matrix(
        &self,
        combos: &[GpuPrivateCombo],
        final_boards: &[GpuFinalBoard],
    ) -> Result<Vec<f32>, GpuCfrError> {
        self.showdown_matrix_range(combos, final_boards, 0, combos.len() * combos.len())
    }

    fn showdown_matrix_range(
        &self,
        combos: &[GpuPrivateCombo],
        final_boards: &[GpuFinalBoard],
        pair_start: usize,
        output_count: usize,
    ) -> Result<Vec<f32>, GpuCfrError> {
        if combos.is_empty() {
            return Ok(Vec::new());
        }
        assert!(
            !final_boards.is_empty(),
            "showdown matrix needs at least one final board"
        );

        let pair_count = combos.len() * combos.len();
        assert!(pair_start <= pair_count);
        assert!(output_count <= pair_count - pair_start);
        let combo_buffer = readonly_buffer(&self.device, "showdown matrix combos", combos);
        let board_buffer = readonly_buffer(&self.device, "showdown matrix boards", final_boards);
        let output = vec![0.0f32; output_count];
        let output_buffer = storage_buffer(&self.device, "showdown matrix equities", &output);
        let params = readonly_buffer(
            &self.device,
            "showdown matrix params",
            &[
                combos.len() as u32,
                final_boards.len() as u32,
                pair_start as u32,
                output_count as u32,
            ],
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("showdown matrix bind group"),
            layout: &self.showdown_matrix_bind_group_layout,
            entries: &[
                bind_entry(0, &combo_buffer),
                bind_entry(1, &board_buffer),
                bind_entry(2, &output_buffer),
                bind_entry(3, &params),
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("showdown matrix encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("showdown matrix pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.showdown_matrix_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (output_count as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        let readback = readback_buffer(&self.device, output_count);
        copy_buffer(&mut encoder, &output_buffer, &readback, output_count);
        let submission = self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;
        read_f32_buffer(&self.device, &readback, output_count)
    }

    #[allow(clippy::too_many_arguments)]
    fn fill_fold_values(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        node_buffer: &wgpu::Buffer,
        fold_terminal_nodes: &[u32],
        combo_buffer: &wgpu::Buffer,
        hero_reaches_buffer: &wgpu::Buffer,
        villain_reaches_buffer: &wgpu::Buffer,
        combo_live_buffer: &wgpu::Buffer,
        hero_values_buffer: &wgpu::Buffer,
        villain_values_buffer: &wgpu::Buffer,
        combo_count: usize,
    ) -> Result<(), GpuCfrError> {
        if fold_terminal_nodes.is_empty() || combo_count == 0 {
            return Ok(());
        }

        const FOLD_AGGREGATE_SLOTS: usize = 53;
        let fold_terminal_nodes_buffer = readonly_buffer(
            &self.device,
            "public tree fold terminal nodes",
            fold_terminal_nodes,
        );
        let aggregate_len = fold_terminal_nodes.len() * FOLD_AGGREGATE_SLOTS;
        let hero_aggregates_buffer = uninit_storage_buffer(
            &self.device,
            "public tree hero fold aggregates",
            aggregate_len,
            false,
        );
        let villain_aggregates_buffer = uninit_storage_buffer(
            &self.device,
            "public tree villain fold aggregates",
            aggregate_len,
            false,
        );

        let (aggregate_x_groups, aggregate_y_groups, aggregate_x_invocations) =
            dispatch_grid(aggregate_len);
        let aggregate_params = uniform_buffer(
            &self.device,
            "public tree fold aggregate params",
            &[GpuPublicTreeParams {
                combo_count: combo_count as u32,
                node_count: fold_terminal_nodes.len() as u32,
                max_actions: FOLD_AGGREGATE_SLOTS as u32,
                output_len: aggregate_x_invocations,
                pair_start: 0,
                chunk_pairs: 0,
                _pad0: 0,
                _pad1: 0,
            }],
        );
        let aggregate_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("public tree fold aggregate bind group"),
            layout: &self.public_tree_fold_aggregate_bind_group_layout,
            entries: &[
                bind_entry(0, &fold_terminal_nodes_buffer),
                bind_entry(1, combo_buffer),
                bind_entry(2, hero_reaches_buffer),
                bind_entry(3, villain_reaches_buffer),
                bind_entry(4, &hero_aggregates_buffer),
                bind_entry(5, &villain_aggregates_buffer),
                bind_entry(6, &aggregate_params),
            ],
        });
        let value_invocations = fold_terminal_nodes.len() * combo_count;
        let (value_x_groups, value_y_groups, value_x_invocations) =
            dispatch_grid(value_invocations);
        let value_params = uniform_buffer(
            &self.device,
            "public tree fold value params",
            &[GpuPublicTreeParams {
                combo_count: combo_count as u32,
                node_count: fold_terminal_nodes.len() as u32,
                max_actions: FOLD_AGGREGATE_SLOTS as u32,
                output_len: value_x_invocations,
                pair_start: 0,
                chunk_pairs: 0,
                _pad0: 0,
                _pad1: 0,
            }],
        );
        let value_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("public tree fold value bind group"),
            layout: &self.public_tree_fold_value_bind_group_layout,
            entries: &[
                bind_entry(0, node_buffer),
                bind_entry(1, &fold_terminal_nodes_buffer),
                bind_entry(2, combo_buffer),
                bind_entry(3, hero_reaches_buffer),
                bind_entry(4, villain_reaches_buffer),
                bind_entry(5, &hero_aggregates_buffer),
                bind_entry(6, &villain_aggregates_buffer),
                bind_entry(7, hero_values_buffer),
                bind_entry(8, villain_values_buffer),
                bind_entry(9, combo_live_buffer),
                bind_entry(10, &value_params),
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("public tree fold aggregate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.public_tree_fold_aggregate_pipeline);
            pass.set_bind_group(0, &aggregate_bind_group, &[]);
            pass.dispatch_workgroups(aggregate_x_groups, aggregate_y_groups, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("public tree fold value pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.public_tree_fold_value_pipeline);
            pass.set_bind_group(0, &value_bind_group, &[]);
            pass.dispatch_workgroups(value_x_groups, value_y_groups, 1);
        }
        Ok(())
    }

    fn terminal_group_buffer_caches(
        &self,
        terminal_groups: &[GpuTerminalGroupCache],
    ) -> Vec<GpuTerminalGroupBufferCache> {
        terminal_groups
            .iter()
            .map(|group| GpuTerminalGroupBufferCache {
                board_count: group.board_count,
                terminal_nodes_len: group.terminal_nodes.len(),
                terminal_nodes_buffer: readonly_buffer(
                    &self.device,
                    "public tree streamed terminal nodes",
                    &group.terminal_nodes,
                ),
                combo_order_buffer: readonly_buffer(
                    &self.device,
                    "public tree streamed terminal combo strength order",
                    &group.combo_order,
                ),
                combo_bounds_buffer: readonly_buffer(
                    &self.device,
                    "public tree streamed terminal combo strength bounds",
                    &group.combo_bounds,
                ),
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn fill_terminal_values_streaming(
        &self,
        node_buffer: &wgpu::Buffer,
        terminal_groups: &[GpuTerminalGroupBufferCache],
        combo_buffer: &wgpu::Buffer,
        blocker_neighbors_buffer: &wgpu::Buffer,
        hero_reaches_buffer: &wgpu::Buffer,
        villain_reaches_buffer: &wgpu::Buffer,
        hero_values_buffer: &wgpu::Buffer,
        villain_values_buffer: &wgpu::Buffer,
        terminal_prefix_pairs_buffer: &wgpu::Buffer,
        terminal_card_prefix_pairs_buffer: &wgpu::Buffer,
        combo_count: usize,
        blocker_neighbor_stride: usize,
        max_terminal_prefix_pairs: usize,
        max_terminal_card_prefix_pairs: usize,
    ) -> Result<(), GpuCfrError> {
        if terminal_groups.is_empty() || combo_count == 0 {
            return Ok(());
        }
        let use_card_prefix = terminal_card_prefix_requested()
            && self.public_tree_terminal_card_prefix_pipeline.is_some()
            && self
                .public_tree_terminal_card_aggregate_reduce_pipeline
                .is_some();
        let submit_batch = std::env::var("POKEDR_GPU_TERMINAL_SUBMIT_BATCH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(64)
            .max(1);
        let stage_profile = std::env::var_os("POKEDR_GPU_TERMINAL_STAGE_PROFILE").is_some();
        let mut partial_elapsed = Duration::ZERO;
        let mut card_prefix_elapsed = Duration::ZERO;
        let mut reduce_elapsed = Duration::ZERO;
        let mut profiled_chunks = 0usize;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("public tree streamed terminal encoder"),
            });
        let mut pending_chunks = 0usize;
        for group in terminal_groups {
            let partial_params = uniform_buffer(
                &self.device,
                "public tree streamed terminal partial params",
                &[GpuPublicTreeParams {
                    combo_count: combo_count as u32,
                    node_count: group.terminal_nodes_len as u32,
                    max_actions: group.board_count as u32,
                    output_len: (combo_count + 1) as u32,
                    pair_start: combo_count as u32,
                    chunk_pairs: 0,
                    _pad0: 0,
                    _pad1: 0,
                }],
            );
            let partial_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("public tree streamed terminal partial bind group"),
                layout: &self.public_tree_terminal_partial_bind_group_layout,
                entries: &[
                    bind_entry(0, node_buffer),
                    bind_entry(1, &group.terminal_nodes_buffer),
                    bind_entry(2, &group.combo_order_buffer),
                    bind_entry(3, &group.combo_bounds_buffer),
                    bind_entry(4, hero_reaches_buffer),
                    bind_entry(5, villain_reaches_buffer),
                    bind_entry(6, terminal_prefix_pairs_buffer),
                    bind_entry(7, &partial_params),
                ],
            });
            let reduce_params = uniform_buffer(
                &self.device,
                "public tree streamed terminal reduce params",
                &[GpuPublicTreeParams {
                    combo_count: combo_count as u32,
                    node_count: group.terminal_nodes_len as u32,
                    max_actions: group.board_count as u32,
                    output_len: (combo_count + 1) as u32,
                    pair_start: blocker_neighbor_stride as u32,
                    chunk_pairs: 0,
                    _pad0: 0,
                    _pad1: 0,
                }],
            );
            let reduce_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("public tree streamed terminal reduce bind group"),
                layout: &self.public_tree_terminal_reduce_bind_group_layout,
                entries: &[
                    bind_entry(0, node_buffer),
                    bind_entry(1, &group.terminal_nodes_buffer),
                    bind_entry(2, &group.combo_bounds_buffer),
                    bind_entry(3, blocker_neighbors_buffer),
                    bind_entry(4, hero_reaches_buffer),
                    bind_entry(5, villain_reaches_buffer),
                    bind_entry(6, terminal_prefix_pairs_buffer),
                    bind_entry(7, hero_values_buffer),
                    bind_entry(8, villain_values_buffer),
                    bind_entry(9, &reduce_params),
                ],
            });
            let prefix_pairs_per_terminal = group.board_count * (combo_count + 1);
            let card_prefix_pairs_per_terminal = group.board_count * 52 * (combo_count + 1);
            let prefix_chunk_size = max_terminal_prefix_pairs / prefix_pairs_per_terminal;
            let card_prefix_chunk_size = if use_card_prefix {
                max_terminal_card_prefix_pairs / card_prefix_pairs_per_terminal
            } else {
                usize::MAX
            };
            let terminal_chunk_size = prefix_chunk_size
                .min(card_prefix_chunk_size)
                .max(1)
                .min(group.terminal_nodes_len.max(1));
            let chunk_use_card_prefix =
                use_card_prefix && card_prefix_pairs_per_terminal <= max_terminal_card_prefix_pairs;
            for terminal_start in (0..group.terminal_nodes_len).step_by(terminal_chunk_size) {
                let terminal_count =
                    terminal_chunk_size.min(group.terminal_nodes_len - terminal_start);
                let partial_workgroups = (terminal_count * group.board_count) as u32;
                let partial_x_groups = partial_workgroups.min(65_535).max(1);
                let partial_y_groups = partial_workgroups.div_ceil(partial_x_groups);
                let partial_chunk_params = GpuTerminalChunkParams {
                    terminal_count: terminal_count as u32,
                    x_invocations: partial_x_groups,
                    terminal_start: terminal_start as u32,
                    _pad0: 0,
                };

                let reduce_invocations = terminal_count * combo_count;
                let (reduce_x_groups, reduce_y_groups, reduce_x_invocations) =
                    dispatch_grid(reduce_invocations);
                let reduce_chunk_params = GpuTerminalChunkParams {
                    terminal_count: terminal_count as u32,
                    x_invocations: reduce_x_invocations,
                    terminal_start: terminal_start as u32,
                    _pad0: 0,
                };
                let card_prefix_workgroups = (terminal_count * group.board_count) as u32;
                let card_prefix_x_groups = card_prefix_workgroups.min(65_535).max(1);
                let card_prefix_y_groups = card_prefix_workgroups.div_ceil(card_prefix_x_groups);
                let card_prefix_params = chunk_use_card_prefix.then(|| {
                    uniform_buffer(
                        &self.device,
                        "public tree streamed terminal card prefix params",
                        &[GpuPublicTreeParams {
                            combo_count: combo_count as u32,
                            node_count: terminal_count as u32,
                            max_actions: group.board_count as u32,
                            output_len: (combo_count + 1) as u32,
                            pair_start: (combo_count + 1) as u32,
                            chunk_pairs: card_prefix_x_groups,
                            _pad0: 0,
                            _pad1: terminal_start as u32,
                        }],
                    )
                });
                let card_reduce_params = chunk_use_card_prefix.then(|| {
                    uniform_buffer(
                        &self.device,
                        "public tree streamed terminal card aggregate reduce params",
                        &[GpuPublicTreeParams {
                            combo_count: combo_count as u32,
                            node_count: terminal_count as u32,
                            max_actions: group.board_count as u32,
                            output_len: (combo_count + 1) as u32,
                            pair_start: (combo_count + 1) as u32,
                            chunk_pairs: reduce_x_invocations,
                            _pad0: 0,
                            _pad1: terminal_start as u32,
                        }],
                    )
                });
                let card_prefix_bind_group =
                    card_prefix_params.as_ref().map(|card_prefix_params| {
                        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("public tree streamed terminal card prefix bind group"),
                            layout: self
                                .public_tree_terminal_card_prefix_bind_group_layout
                                .as_ref()
                                .expect(
                                    "card prefix layout must exist when card prefix is enabled",
                                ),
                            entries: &[
                                bind_entry(0, node_buffer),
                                bind_entry(1, &group.terminal_nodes_buffer),
                                bind_entry(2, &group.combo_order_buffer),
                                bind_entry(3, &group.combo_bounds_buffer),
                                bind_entry(4, combo_buffer),
                                bind_entry(5, hero_reaches_buffer),
                                bind_entry(6, villain_reaches_buffer),
                                bind_entry(7, terminal_card_prefix_pairs_buffer),
                                bind_entry(8, card_prefix_params),
                            ],
                        })
                    });
                let card_reduce_bind_group = card_reduce_params.as_ref().map(|card_reduce_params| {
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("public tree streamed terminal card aggregate reduce bind group"),
                        layout: self
                            .public_tree_terminal_card_aggregate_reduce_bind_group_layout
                            .as_ref()
                            .expect(
                                "card aggregate reduce layout must exist when card prefix is enabled",
                            ),
                        entries: &[
                            bind_entry(0, node_buffer),
                            bind_entry(1, &group.terminal_nodes_buffer),
                            bind_entry(2, &group.combo_bounds_buffer),
                            bind_entry(3, combo_buffer),
                            bind_entry(4, hero_reaches_buffer),
                            bind_entry(5, villain_reaches_buffer),
                            bind_entry(6, terminal_prefix_pairs_buffer),
                            bind_entry(7, terminal_card_prefix_pairs_buffer),
                            bind_entry(8, hero_values_buffer),
                            bind_entry(9, villain_values_buffer),
                            bind_entry(10, card_reduce_params),
                        ],
                    })
                });
                if stage_profile {
                    let mut partial_encoder =
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("public tree terminal partial profile encoder"),
                            });
                    {
                        let mut pass =
                            partial_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("public tree streamed terminal partial pass"),
                                timestamp_writes: None,
                            });
                        pass.set_pipeline(&self.public_tree_terminal_partial_pipeline);
                        pass.set_bind_group(0, &partial_bind_group, &[]);
                        pass.set_immediates(0, bytemuck::bytes_of(&partial_chunk_params));
                        pass.dispatch_workgroups(partial_x_groups, partial_y_groups, 1);
                    }
                    let start = Instant::now();
                    self.queue.submit(Some(partial_encoder.finish()));
                    self.profile_poll()?;
                    partial_elapsed += start.elapsed();

                    if chunk_use_card_prefix {
                        let mut card_prefix_encoder =
                            self.device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("public tree terminal card prefix profile encoder"),
                                });
                        {
                            let mut pass = card_prefix_encoder.begin_compute_pass(
                                &wgpu::ComputePassDescriptor {
                                    label: Some("public tree streamed terminal card prefix pass"),
                                    timestamp_writes: None,
                                },
                            );
                            pass.set_pipeline(
                                self.public_tree_terminal_card_prefix_pipeline
                                    .as_ref()
                                    .expect("card prefix pipeline must exist when enabled"),
                            );
                            pass.set_bind_group(
                                0,
                                card_prefix_bind_group
                                    .as_ref()
                                    .expect("card prefix bind group must exist when enabled"),
                                &[],
                            );
                            pass.dispatch_workgroups(card_prefix_x_groups, card_prefix_y_groups, 1);
                        }
                        let start = Instant::now();
                        self.queue.submit(Some(card_prefix_encoder.finish()));
                        self.profile_poll()?;
                        card_prefix_elapsed += start.elapsed();
                    }

                    let mut reduce_encoder =
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("public tree terminal reduce profile encoder"),
                            });
                    {
                        let mut pass =
                            reduce_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("public tree streamed terminal reduce pass"),
                                timestamp_writes: None,
                            });
                        if chunk_use_card_prefix {
                            pass.set_pipeline(
                                self.public_tree_terminal_card_aggregate_reduce_pipeline
                                    .as_ref()
                                    .expect(
                                        "card aggregate reduce pipeline must exist when enabled",
                                    ),
                            );
                            pass.set_bind_group(
                                0,
                                card_reduce_bind_group.as_ref().expect(
                                    "card aggregate reduce bind group must exist when enabled",
                                ),
                                &[],
                            );
                        } else {
                            pass.set_pipeline(&self.public_tree_terminal_reduce_pipeline);
                            pass.set_bind_group(0, &reduce_bind_group, &[]);
                            pass.set_immediates(0, bytemuck::bytes_of(&reduce_chunk_params));
                        }
                        pass.dispatch_workgroups(reduce_x_groups, reduce_y_groups, 1);
                    }
                    let start = Instant::now();
                    self.queue.submit(Some(reduce_encoder.finish()));
                    self.profile_poll()?;
                    reduce_elapsed += start.elapsed();
                    profiled_chunks += 1;
                    continue;
                }
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("public tree streamed terminal partial pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.public_tree_terminal_partial_pipeline);
                    pass.set_bind_group(0, &partial_bind_group, &[]);
                    pass.set_immediates(0, bytemuck::bytes_of(&partial_chunk_params));
                    pass.dispatch_workgroups(partial_x_groups, partial_y_groups, 1);
                }

                if chunk_use_card_prefix {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("public tree streamed terminal card prefix pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(
                        self.public_tree_terminal_card_prefix_pipeline
                            .as_ref()
                            .expect("card prefix pipeline must exist when enabled"),
                    );
                    pass.set_bind_group(
                        0,
                        card_prefix_bind_group
                            .as_ref()
                            .expect("card prefix bind group must exist when enabled"),
                        &[],
                    );
                    pass.dispatch_workgroups(card_prefix_x_groups, card_prefix_y_groups, 1);
                }

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("public tree streamed terminal reduce pass"),
                        timestamp_writes: None,
                    });
                    if chunk_use_card_prefix {
                        pass.set_pipeline(
                            self.public_tree_terminal_card_aggregate_reduce_pipeline
                                .as_ref()
                                .expect("card aggregate reduce pipeline must exist when enabled"),
                        );
                        pass.set_bind_group(
                            0,
                            card_reduce_bind_group
                                .as_ref()
                                .expect("card aggregate reduce bind group must exist when enabled"),
                            &[],
                        );
                    } else {
                        pass.set_pipeline(&self.public_tree_terminal_reduce_pipeline);
                        pass.set_bind_group(0, &reduce_bind_group, &[]);
                        pass.set_immediates(0, bytemuck::bytes_of(&reduce_chunk_params));
                    }
                    pass.dispatch_workgroups(reduce_x_groups, reduce_y_groups, 1);
                }
                pending_chunks += 1;
                if pending_chunks >= submit_batch {
                    self.queue.submit(Some(encoder.finish()));
                    encoder = self
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("public tree streamed terminal encoder"),
                        });
                    pending_chunks = 0;
                }
            }
        }
        if pending_chunks > 0 {
            self.queue.submit(Some(encoder.finish()));
        }
        self.profile_poll()?;
        if stage_profile {
            eprintln!(
                "pokedr: gpu profile phase=cfv_terminal_partial elapsed_ms={:.3} chunks={}",
                partial_elapsed.as_secs_f64() * 1000.0,
                profiled_chunks
            );
            eprintln!(
                "pokedr: gpu profile phase=cfv_terminal_card_prefix elapsed_ms={:.3} chunks={}",
                card_prefix_elapsed.as_secs_f64() * 1000.0,
                profiled_chunks
            );
            eprintln!(
                "pokedr: gpu profile phase=cfv_terminal_reduce elapsed_ms={:.3} chunks={}",
                reduce_elapsed.as_secs_f64() * 1000.0,
                profiled_chunks
            );
        }
        Ok(())
    }

    fn public_tree_layer_tile_buffers(
        &self,
        layered: &GpuPublicTreeLayered,
        combos: &[GpuPrivateCombo],
        showdown_boards: &[GpuFinalBoard],
    ) -> Vec<Vec<GpuPublicTreeLayerTileBuffers>> {
        layered
            .layers
            .iter()
            .enumerate()
            .map(|(_, layer)| {
                (0..layer.nodes.len())
                    .step_by(layered.node_tile_size)
                    .map(|node_start| {
                        let node_end = (node_start + layered.node_tile_size).min(layer.nodes.len());
                        let mut tile_nodes = Vec::with_capacity(node_end - node_start);
                        let mut tile_children = Vec::new();
                        let mut tile_child_cards = Vec::new();
                        let mut fold_terminal_nodes = Vec::new();
                        let mut showdown_terminal_nodes = Vec::new();

                        for source_slot in node_start..node_end {
                            let local_slot = (source_slot - node_start) as u32;
                            let source = layer.nodes[source_slot];
                            if source.kind != 0 && source.kind != 1 {
                                if source.terminal_kind == 2 {
                                    showdown_terminal_nodes.push(local_slot);
                                } else {
                                    fold_terminal_nodes.push(local_slot);
                                }
                            }
                            let first_child = tile_children.len() as u32;
                            for action in 0..source.child_count as usize {
                                let source_child_offset = source.first_child as usize + action;
                                tile_children.push(layer.children[source_child_offset]);
                                tile_child_cards.push(layer.child_cards[source_child_offset]);
                            }
                            tile_nodes.push(GpuPublicTreeNode {
                                first_child,
                                ..source
                            });
                        }

                        let value_len = tile_nodes.len() * combos.len();
                        let showdown_terminal_groups =
                            self.terminal_group_buffer_caches(&terminal_group_caches(
                                &tile_nodes,
                                &showdown_terminal_nodes,
                                combos,
                                showdown_boards,
                            ));

                        GpuPublicTreeLayerTileBuffers {
                            node_start,
                            node_end,
                            node_buffer: readonly_buffer(
                                &self.device,
                                "public tree layer tile nodes",
                                &tile_nodes,
                            ),
                            child_buffer: readonly_buffer(
                                &self.device,
                                "public tree layer tile children",
                                &tile_children,
                            ),
                            fold_terminal_nodes,
                            showdown_terminal_groups,
                            hero_reaches_buffer: uninit_storage_buffer(
                                &self.device,
                                "public tree layer tile hero reaches",
                                value_len,
                                false,
                            ),
                            villain_reaches_buffer: uninit_storage_buffer(
                                &self.device,
                                "public tree layer tile villain reaches",
                                value_len,
                                false,
                            ),
                            combo_live_buffer: uninit_storage_buffer(
                                &self.device,
                                "public tree layer tile combo live mask",
                                value_len,
                                false,
                            ),
                            hero_values_buffer: uninit_storage_buffer(
                                &self.device,
                                "public tree layer tile hero values",
                                value_len,
                                true,
                            ),
                            villain_values_buffer: uninit_storage_buffer(
                                &self.device,
                                "public tree layer tile villain values",
                                value_len,
                                true,
                            ),
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn propagate_layer_reaches(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &GpuPublicTreeIterationContext,
        regrets_buffer: &wgpu::Buffer,
        prediction_buffer: &wgpu::Buffer,
        variant: super::CfrVariant,
    ) {
        for layer_tiles in &ctx.layer_tiles {
            for tile in layer_tiles {
                let value_count = (tile.node_end - tile.node_start) * ctx.combos_len;
                if value_count == 0 {
                    continue;
                }
                let (x_groups, y_groups, x_invocations) = dispatch_grid(value_count);
                let params = uniform_buffer(
                    &self.device,
                    "public tree layer reach init params",
                    &[GpuPublicTreeParams {
                        combo_count: ctx.combos_len as u32,
                        node_count: (tile.node_end - tile.node_start) as u32,
                        max_actions: ctx.actions as u32,
                        output_len: x_invocations,
                        pair_start: variant_code(variant),
                        chunk_pairs: 0,
                        _pad0: tile.node_start as u32,
                        _pad1: variant_prediction_eta(variant).to_bits(),
                    }],
                );
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("public tree layer reach init bind group"),
                    layout: &self.public_tree_layer_reach_init_bind_group_layout,
                    entries: &[
                        bind_entry(0, &ctx.root_weights_buffer),
                        bind_entry(1, &tile.hero_reaches_buffer),
                        bind_entry(2, &tile.villain_reaches_buffer),
                        bind_entry(3, &tile.combo_live_buffer),
                        bind_entry(4, &params),
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("public tree layer reach init pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.public_tree_layer_reach_init_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(x_groups, y_groups, 1);
            }
        }

        for edge_tile in &ctx.layered.reach_edge_tiles {
            let parent_tile_index = edge_tile.parent_tile.node_start / ctx.layered.node_tile_size;
            let child_tile_index = edge_tile.child_tile.node_start / ctx.layered.node_tile_size;
            let parent_tile = &ctx.layer_tiles[edge_tile.parent_layer][parent_tile_index];
            let child_tile = &ctx.layer_tiles[edge_tile.child_layer][child_tile_index];
            let edge_buffer = readonly_buffer(
                &self.device,
                "public tree layer reach edges",
                &edge_tile.edges,
            );
            let invocation_count = edge_tile.edges.len() * ctx.combos_len;
            let (x_groups, y_groups, x_invocations) = dispatch_grid(invocation_count);
            let params = uniform_buffer(
                &self.device,
                "public tree layer reach edge params",
                &[GpuPublicTreeParams {
                    combo_count: ctx.combos_len as u32,
                    node_count: (parent_tile.node_end - parent_tile.node_start) as u32,
                    max_actions: ctx.actions as u32,
                    output_len: x_invocations,
                    pair_start: variant_code(variant),
                    chunk_pairs: edge_tile.edges.len() as u32,
                    _pad0: 0,
                    _pad1: variant_prediction_eta(variant).to_bits(),
                }],
            );
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("public tree layer reach edge bind group"),
                layout: &self.public_tree_layer_reach_edge_bind_group_layout,
                entries: &[
                    bind_entry(0, &parent_tile.node_buffer),
                    bind_entry(1, &edge_buffer),
                    bind_entry(2, &ctx.combo_buffer),
                    bind_entry(3, &ctx.root_weights_buffer),
                    bind_entry(4, regrets_buffer),
                    bind_entry(5, &parent_tile.hero_reaches_buffer),
                    bind_entry(6, &parent_tile.villain_reaches_buffer),
                    bind_entry(7, &parent_tile.combo_live_buffer),
                    bind_entry(8, &child_tile.hero_reaches_buffer),
                    bind_entry(9, &child_tile.villain_reaches_buffer),
                    bind_entry(10, &child_tile.combo_live_buffer),
                    bind_entry(11, prediction_buffer),
                    bind_entry(12, &params),
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("public tree layer reach edge pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.public_tree_layer_reach_edge_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(x_groups, y_groups, 1);
        }
    }

    fn backup_layer_values(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &GpuPublicTreeIterationContext,
        br_player: u32,
    ) {
        for parent_layer_index in (0..ctx.layer_tiles.len().saturating_sub(1)).rev() {
            let child_layer_index = parent_layer_index + 1;
            for parent_tile in &ctx.layer_tiles[parent_layer_index] {
                let value_count = (parent_tile.node_end - parent_tile.node_start) * ctx.combos_len;
                if value_count == 0 {
                    continue;
                }
                let (x_groups, y_groups, x_invocations) = dispatch_grid(value_count);
                let init_params = uniform_buffer(
                    &self.device,
                    "public tree layer backup init params",
                    &[GpuPublicTreeParams {
                        combo_count: ctx.combos_len as u32,
                        node_count: (parent_tile.node_end - parent_tile.node_start) as u32,
                        max_actions: ctx.actions as u32,
                        output_len: x_invocations,
                        pair_start: br_player,
                        chunk_pairs: 0,
                        _pad0: 0,
                        _pad1: 0,
                    }],
                );
                let init_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("public tree layer backup init bind group"),
                    layout: &self.public_tree_layer_backup_init_bind_group_layout,
                    entries: &[
                        bind_entry(0, &parent_tile.node_buffer),
                        bind_entry(1, &parent_tile.hero_values_buffer),
                        bind_entry(2, &parent_tile.villain_values_buffer),
                        bind_entry(3, &init_params),
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("public tree layer backup init pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.public_tree_layer_backup_init_pipeline);
                    pass.set_bind_group(0, &init_bind_group, &[]);
                    pass.dispatch_workgroups(x_groups, y_groups, 1);
                }

                for child_tile in &ctx.layer_tiles[child_layer_index] {
                    let params = uniform_buffer(
                        &self.device,
                        "public tree layer backup child params",
                        &[GpuPublicTreeParams {
                            combo_count: ctx.combos_len as u32,
                            node_count: (parent_tile.node_end - parent_tile.node_start) as u32,
                            max_actions: ctx.actions as u32,
                            output_len: x_invocations,
                            pair_start: br_player,
                            chunk_pairs: child_tile.node_start as u32,
                            _pad0: child_tile.node_end as u32,
                            _pad1: 0,
                        }],
                    );
                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("public tree layer backup child bind group"),
                        layout: &self.public_tree_layer_backup_child_bind_group_layout,
                        entries: &[
                            bind_entry(0, &parent_tile.node_buffer),
                            bind_entry(1, &parent_tile.child_buffer),
                            bind_entry(2, &child_tile.hero_values_buffer),
                            bind_entry(3, &child_tile.villain_values_buffer),
                            bind_entry(4, &parent_tile.hero_reaches_buffer),
                            bind_entry(5, &parent_tile.villain_reaches_buffer),
                            bind_entry(6, &child_tile.hero_reaches_buffer),
                            bind_entry(7, &child_tile.villain_reaches_buffer),
                            bind_entry(8, &parent_tile.hero_values_buffer),
                            bind_entry(9, &parent_tile.villain_values_buffer),
                            bind_entry(10, &params),
                        ],
                    });
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("public tree layer backup child pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.public_tree_layer_backup_child_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(x_groups, y_groups, 1);
                }
            }
        }
    }

    fn write_layer_outputs(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &GpuPublicTreeIterationContext,
    ) {
        let empty_edge = GpuPublicTreeEdge {
            parent: 0,
            child: 0,
            action: 0,
            card: u32::MAX,
        };
        let empty_edge_buffer = readonly_buffer(
            &self.device,
            "public tree empty layer output edge",
            &[empty_edge],
        );
        for layer_tiles in &ctx.layer_tiles {
            for tile in layer_tiles {
                let decision_invocations = (tile.node_end - tile.node_start) * 53usize;
                if decision_invocations == 0 {
                    continue;
                }
                let (x_groups, y_groups, x_invocations) = dispatch_grid(decision_invocations);
                let params = uniform_buffer(
                    &self.device,
                    "public tree layer decision aggregate params",
                    &[GpuPublicTreeParams {
                        combo_count: ctx.combos_len as u32,
                        node_count: (tile.node_end - tile.node_start) as u32,
                        max_actions: ctx.actions as u32,
                        output_len: x_invocations,
                        pair_start: 0,
                        chunk_pairs: ctx.public_infoset_count as u32,
                        _pad0: 0,
                        _pad1: 0,
                    }],
                );
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("public tree layer decision aggregate bind group"),
                    layout: &self.public_tree_layer_output_bind_group_layout,
                    entries: &[
                        bind_entry(0, &tile.node_buffer),
                        bind_entry(1, &ctx.combo_buffer),
                        bind_entry(2, &tile.hero_reaches_buffer),
                        bind_entry(3, &tile.villain_reaches_buffer),
                        bind_entry(4, &ctx.hero_decision_aggregates_buffer),
                        bind_entry(5, &ctx.villain_decision_aggregates_buffer),
                        bind_entry(6, &ctx.action_values_buffer),
                        bind_entry(7, &tile.combo_live_buffer),
                        bind_entry(8, &empty_edge_buffer),
                        bind_entry(9, &tile.hero_values_buffer),
                        bind_entry(10, &tile.villain_values_buffer),
                        bind_entry(11, &ctx.action_weights_buffer),
                        bind_entry(12, &ctx.reach_weights_buffer),
                        bind_entry(13, &ctx.strategy_weights_buffer),
                        bind_entry(14, &params),
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("public tree layer decision aggregate pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.public_tree_layer_decision_aggregate_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(x_groups, y_groups, 1);
            }
        }

        for value_player in [0u32, 1u32] {
            for layer_tiles in &ctx.layer_tiles {
                for tile in layer_tiles {
                    let invocations = (tile.node_end - tile.node_start) * ctx.combos_len;
                    if invocations == 0 {
                        continue;
                    }
                    let (x_groups, y_groups, _) = dispatch_grid(invocations);
                    let params = uniform_buffer(
                        &self.device,
                        "public tree layer denominator params",
                        &[GpuPublicTreeParams {
                            combo_count: ctx.combos_len as u32,
                            node_count: (tile.node_end - tile.node_start) as u32,
                            max_actions: ctx.actions as u32,
                            output_len: ctx.action_len as u32,
                            pair_start: value_player,
                            chunk_pairs: ctx.public_infoset_count as u32,
                            _pad0: 0,
                            _pad1: 0,
                        }],
                    );
                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("public tree layer denominator bind group"),
                        layout: &self.public_tree_layer_output_bind_group_layout,
                        entries: &[
                            bind_entry(0, &tile.node_buffer),
                            bind_entry(1, &ctx.combo_buffer),
                            bind_entry(2, &tile.hero_reaches_buffer),
                            bind_entry(3, &tile.villain_reaches_buffer),
                            bind_entry(4, &ctx.hero_decision_aggregates_buffer),
                            bind_entry(5, &ctx.villain_decision_aggregates_buffer),
                            bind_entry(6, &ctx.action_values_buffer),
                            bind_entry(7, &tile.combo_live_buffer),
                            bind_entry(8, &empty_edge_buffer),
                            bind_entry(9, &tile.hero_values_buffer),
                            bind_entry(10, &tile.villain_values_buffer),
                            bind_entry(11, &ctx.action_weights_buffer),
                            bind_entry(12, &ctx.reach_weights_buffer),
                            bind_entry(13, &ctx.strategy_weights_buffer),
                            bind_entry(14, &params),
                        ],
                    });
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("public tree layer denominator pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.public_tree_layer_denominator_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(x_groups, y_groups, 1);
                }
            }
        }

        for edge_tile in &ctx.layered.reach_edge_tiles {
            let parent_tile_index = edge_tile.parent_tile.node_start / ctx.layered.node_tile_size;
            let child_tile_index = edge_tile.child_tile.node_start / ctx.layered.node_tile_size;
            let parent_tile = &ctx.layer_tiles[edge_tile.parent_layer][parent_tile_index];
            let child_tile = &ctx.layer_tiles[edge_tile.child_layer][child_tile_index];
            let edge_buffer = readonly_buffer(
                &self.device,
                "public tree layer action edges",
                &edge_tile.edges,
            );
            let invocations = edge_tile.edges.len() * ctx.combos_len;
            let (x_groups, y_groups, _) = dispatch_grid(invocations);
            for value_player in [0u32, 1u32] {
                let params = uniform_buffer(
                    &self.device,
                    "public tree layer action edge params",
                    &[GpuPublicTreeParams {
                        combo_count: ctx.combos_len as u32,
                        node_count: edge_tile.edges.len() as u32,
                        max_actions: ctx.actions as u32,
                        output_len: ctx.action_len as u32,
                        pair_start: value_player,
                        chunk_pairs: ctx.public_infoset_count as u32,
                        _pad0: 0,
                        _pad1: 0,
                    }],
                );
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("public tree layer action edge bind group"),
                    layout: &self.public_tree_layer_output_bind_group_layout,
                    entries: &[
                        bind_entry(0, &parent_tile.node_buffer),
                        bind_entry(1, &ctx.combo_buffer),
                        bind_entry(2, &parent_tile.hero_reaches_buffer),
                        bind_entry(3, &parent_tile.villain_reaches_buffer),
                        bind_entry(4, &ctx.hero_decision_aggregates_buffer),
                        bind_entry(5, &ctx.villain_decision_aggregates_buffer),
                        bind_entry(6, &ctx.action_values_buffer),
                        bind_entry(7, &parent_tile.combo_live_buffer),
                        bind_entry(8, &edge_buffer),
                        bind_entry(9, &child_tile.hero_values_buffer),
                        bind_entry(10, &child_tile.villain_values_buffer),
                        bind_entry(11, &ctx.action_weights_buffer),
                        bind_entry(12, &ctx.reach_weights_buffer),
                        bind_entry(13, &ctx.strategy_weights_buffer),
                        bind_entry(14, &params),
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("public tree layer action edge pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.public_tree_layer_action_edge_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(x_groups, y_groups, 1);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn public_tree_iteration_context(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        villain_weights: &[f32],
        showdown_boards: &[GpuFinalBoard],
        infosets: usize,
        actions: usize,
    ) -> GpuPublicTreeIterationContext {
        assert!(!nodes.is_empty());
        assert_eq!(combo_legal.len(), combos.len());
        assert_eq!(villain_weights.len(), combos.len());
        assert_eq!(
            infosets,
            nodes_public_infoset_count(nodes) * combos.len() * 2
        );

        let action_len = infosets * actions;
        let output_len = action_len * 2 + infosets * 2;
        let node_combo_len = nodes.len() * combos.len();
        let layered = public_tree_layered(
            nodes,
            children,
            child_cards,
            combos.len(),
            self.device.limits().max_storage_buffer_binding_size,
        );
        if std::env::var_os("POKEDR_GPU_LAYER_TRACE").is_some() {
            eprintln!(
                "pokedr: gpu public tree layers={} max_layer_nodes={} node_tile_size={} max_layer_tiles={} reach_edge_tiles={} max_layer_node_combos={} full_node_combos={}",
                layered.layers.len(),
                layered.max_layer_nodes,
                layered.node_tile_size,
                layered.max_layer_tiles,
                layered.reach_edge_tiles.len(),
                layered.max_layer_nodes * combos.len(),
                node_combo_len,
            );
        }

        let combo_buffer = readonly_buffer(&self.device, "public tree combos", combos);
        let root_weights: Vec<_> = combo_legal
            .iter()
            .zip(villain_weights)
            .map(|(is_legal, weight)| if *is_legal != 0 { *weight } else { -1.0 })
            .collect();
        let root_weights_buffer =
            readonly_buffer(&self.device, "public tree root weights", &root_weights);
        let action_values_buffer = uninit_storage_buffer(
            &self.device,
            "public tree action values output",
            action_len,
            true,
        );
        let action_weights_buffer = uninit_storage_buffer(
            &self.device,
            "public tree action weights output",
            action_len,
            true,
        );
        let reach_weights_buffer = uninit_storage_buffer(
            &self.device,
            "public tree reach weights output",
            infosets,
            true,
        );
        let strategy_weights_buffer = uninit_storage_buffer(
            &self.device,
            "public tree strategy weights output",
            infosets,
            true,
        );
        let fold_terminal_nodes: Vec<_> = nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                (node.kind != 0 && node.kind != 1 && node.terminal_kind != 2)
                    .then_some(index as u32)
            })
            .collect();
        let showdown_terminal_nodes: Vec<_> = nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                (node.kind != 0 && node.kind != 1 && node.terminal_kind == 2)
                    .then_some(index as u32)
            })
            .collect();
        let public_infoset_count = nodes_public_infoset_count(nodes);
        let terminal_tile_count = 1;
        let _terminal_tile_size = combos.len().max(1);
        let max_showdown_boards = showdown_terminal_nodes
            .iter()
            .map(|node_index| nodes[*node_index as usize]._pad0 as usize)
            .max()
            .unwrap_or(1)
            .max(1);
        let terminal_board_tile_count = 1;
        let _terminal_board_tile_size = max_showdown_boards;
        let (blocker_neighbors, blocker_neighbor_stride) = showdown_blocker_neighbors(combos);
        let terminal_blocker_neighbors_buffer = readonly_buffer(
            &self.device,
            "public tree terminal blocker neighbors",
            &blocker_neighbors,
        );
        let default_max_terminal_prefix_pairs = 8_000_000usize;
        let max_terminal_prefix_pairs = std::env::var("POKEDR_GPU_MAX_TERMINAL_PREFIX_PAIRS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(default_max_terminal_prefix_pairs)
            .max(max_showdown_boards * (combos.len() + 1));
        let terminal_chunk_size =
            (max_terminal_prefix_pairs / (max_showdown_boards * (combos.len() + 1))).max(1);
        let terminal_prefix_pairs_buffer = uninit_storage_buffer(
            &self.device,
            "public tree streamed terminal prefix pairs scratch",
            max_terminal_prefix_pairs * 2,
            false,
        );
        let default_max_terminal_card_prefix_pairs = 8_000_000usize;
        let min_terminal_card_prefix_pairs = 52 * (combos.len() + 1);
        let max_terminal_card_prefix_pairs =
            std::env::var("POKEDR_GPU_MAX_TERMINAL_CARD_PREFIX_PAIRS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(default_max_terminal_card_prefix_pairs)
                .max(min_terminal_card_prefix_pairs);
        let terminal_card_prefix_pairs_buffer = uninit_storage_buffer(
            &self.device,
            "public tree streamed terminal card prefix pairs scratch",
            max_terminal_card_prefix_pairs * 2,
            false,
        );
        const DECISION_AGGREGATE_SLOTS: usize = 53;
        let decision_aggregate_len = public_infoset_count * DECISION_AGGREGATE_SLOTS;
        let hero_decision_aggregates_buffer = uninit_storage_buffer(
            &self.device,
            "public tree hero decision aggregates",
            decision_aggregate_len,
            false,
        );
        let villain_decision_aggregates_buffer = uninit_storage_buffer(
            &self.device,
            "public tree villain decision aggregates",
            decision_aggregate_len,
            false,
        );
        let layer_tiles = self.public_tree_layer_tile_buffers(&layered, combos, showdown_boards);

        GpuPublicTreeIterationContext {
            nodes_len: nodes.len(),
            combos_len: combos.len(),
            actions,
            action_len,
            output_len,
            node_combo_len,
            public_infoset_count,
            layered,
            combo_buffer,
            root_weights_buffer,
            action_values_buffer,
            action_weights_buffer,
            reach_weights_buffer,
            strategy_weights_buffer,
            layer_tiles,
            fold_terminal_nodes,
            showdown_terminal_nodes,
            terminal_tile_count,
            terminal_board_tile_count,
            terminal_chunk_size,
            terminal_blocker_neighbors_buffer,
            terminal_blocker_neighbor_stride: blocker_neighbor_stride,
            terminal_prefix_pair_budget: max_terminal_prefix_pairs,
            terminal_prefix_pairs_buffer,
            terminal_card_prefix_pair_budget: max_terminal_card_prefix_pairs,
            terminal_card_prefix_pairs_buffer,
            hero_decision_aggregates_buffer,
            villain_decision_aggregates_buffer,
        }
    }

    fn public_tree_iteration_output_with_context(
        &self,
        ctx: &GpuPublicTreeIterationContext,
        regrets_buffer: &wgpu::Buffer,
        prediction_buffer: &wgpu::Buffer,
        variant: super::CfrVariant,
        br_player: u32,
    ) -> Result<(GpuPublicTreeOutputBuffers, usize, usize), GpuCfrError> {
        if std::env::var_os("POKEDR_SOLVER_PROGRESS_OFF").is_none() {
            eprintln!(
                "pokedr: gpu public tree cfv nodes={} combos={} node_combo_values={} folds={} showdowns={} terminal_tiles={} board_tiles={} terminal_chunk={}",
                ctx.nodes_len,
                ctx.combos_len,
                ctx.node_combo_len,
                ctx.fold_terminal_nodes.len(),
                ctx.showdown_terminal_nodes.len(),
                ctx.terminal_tile_count,
                ctx.terminal_board_tile_count,
                ctx.terminal_chunk_size
            );
        }

        let profile = Self::gpu_profile_enabled();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("public tree iteration encoder"),
            });
        let mut phase_start = profile.then(Instant::now);
        self.propagate_layer_reaches(
            &mut encoder,
            ctx,
            regrets_buffer,
            prediction_buffer,
            variant,
        );
        encoder = self.finish_profile_phase(encoder, "cfv_reach", phase_start)?;
        phase_start = profile.then(Instant::now);

        for layer_tiles in &ctx.layer_tiles {
            for tile in layer_tiles {
                self.fill_fold_values(
                    &mut encoder,
                    &tile.node_buffer,
                    &tile.fold_terminal_nodes,
                    &ctx.combo_buffer,
                    &tile.hero_reaches_buffer,
                    &tile.villain_reaches_buffer,
                    &tile.combo_live_buffer,
                    &tile.hero_values_buffer,
                    &tile.villain_values_buffer,
                    ctx.combos_len,
                )?;
            }
        }
        self.queue.submit(Some(encoder.finish()));
        for layer_tiles in &ctx.layer_tiles {
            for tile in layer_tiles {
                self.fill_terminal_values_streaming(
                    &tile.node_buffer,
                    &tile.showdown_terminal_groups,
                    &ctx.combo_buffer,
                    &ctx.terminal_blocker_neighbors_buffer,
                    &tile.hero_reaches_buffer,
                    &tile.villain_reaches_buffer,
                    &tile.hero_values_buffer,
                    &tile.villain_values_buffer,
                    &ctx.terminal_prefix_pairs_buffer,
                    &ctx.terminal_card_prefix_pairs_buffer,
                    ctx.combos_len,
                    ctx.terminal_blocker_neighbor_stride,
                    ctx.terminal_prefix_pair_budget,
                    ctx.terminal_card_prefix_pair_budget,
                )?;
            }
        }
        if let Some(start) = phase_start {
            eprintln!(
                "pokedr: gpu profile phase=cfv_terminal elapsed_ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("public tree post-terminal iteration encoder"),
            });
        phase_start = profile.then(Instant::now);

        self.backup_layer_values(&mut encoder, ctx, br_player);
        encoder = self.finish_profile_phase(encoder, "cfv_backup", phase_start)?;
        phase_start = profile.then(Instant::now);

        self.write_layer_outputs(&mut encoder, ctx);
        encoder = self.finish_profile_phase(encoder, "cfv_decision_denominator", phase_start)?;
        phase_start = profile.then(Instant::now);
        self.submit_final_profile_phase(encoder, "cfv_action_aggregate", phase_start)?;
        Ok((
            GpuPublicTreeOutputBuffers {
                action_values: ctx.action_values_buffer.clone(),
                action_weights: ctx.action_weights_buffer.clone(),
                reach_weights: ctx.reach_weights_buffer.clone(),
                strategy_weights: ctx.strategy_weights_buffer.clone(),
            },
            ctx.output_len,
            ctx.action_len,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn public_tree_iteration_values(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        villain_weights: &[f32],
        showdown_boards: &[GpuFinalBoard],
        state: &DenseCfrState,
    ) -> Result<GpuRootTerminalValues, GpuCfrError> {
        self.public_tree_values_with_br_player(
            nodes,
            children,
            child_cards,
            combos,
            combo_legal,
            villain_weights,
            showdown_boards,
            state,
            2,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn public_tree_best_response_values(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        villain_weights: &[f32],
        showdown_boards: &[GpuFinalBoard],
        state: &DenseCfrState,
        br_player: u32,
    ) -> Result<GpuRootTerminalValues, GpuCfrError> {
        assert!(br_player < 2, "best-response player must be 0 or 1");
        self.public_tree_values_with_br_player(
            nodes,
            children,
            child_cards,
            combos,
            combo_legal,
            villain_weights,
            showdown_boards,
            state,
            br_player,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn public_tree_values_with_br_player(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        villain_weights: &[f32],
        showdown_boards: &[GpuFinalBoard],
        state: &DenseCfrState,
        br_player: u32,
    ) -> Result<GpuRootTerminalValues, GpuCfrError> {
        let regrets_buffer = readonly_buffer(&self.device, "public tree regrets", &state.regrets);
        let prediction_buffer =
            readonly_buffer(&self.device, "public tree prediction", &state.prediction);
        let profile = Self::gpu_profile_enabled();
        let setup_start = profile.then(Instant::now);
        let context = self.public_tree_iteration_context(
            nodes,
            children,
            child_cards,
            combos,
            combo_legal,
            villain_weights,
            showdown_boards,
            state.infosets,
            state.actions,
        );
        if let Some(start) = setup_start {
            eprintln!(
                "pokedr: gpu profile phase=cfv_setup elapsed_ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        let (output_buffers, _output_len, action_len) = self
            .public_tree_iteration_output_with_context(
                &context,
                &regrets_buffer,
                &prediction_buffer,
                state.variant,
                br_player,
            )?;
        let action_values_readback = readback_buffer(&self.device, action_len);
        let action_weights_readback = readback_buffer(&self.device, action_len);
        let reach_weights_readback = readback_buffer(&self.device, state.infosets);
        let strategy_weights_readback = readback_buffer(&self.device, state.infosets);
        let root_hero_values_readback = readback_buffer(&self.device, combos.len());
        let root_villain_values_readback = readback_buffer(&self.device, combos.len());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("public tree readback encoder"),
            });
        copy_buffer(
            &mut encoder,
            &output_buffers.action_values,
            &action_values_readback,
            action_len,
        );
        copy_buffer(
            &mut encoder,
            &output_buffers.action_weights,
            &action_weights_readback,
            action_len,
        );
        copy_buffer(
            &mut encoder,
            &output_buffers.reach_weights,
            &reach_weights_readback,
            state.infosets,
        );
        copy_buffer(
            &mut encoder,
            &output_buffers.strategy_weights,
            &strategy_weights_readback,
            state.infosets,
        );
        let root_tile = &context.layer_tiles[0][0];
        copy_buffer(
            &mut encoder,
            &root_tile.hero_values_buffer,
            &root_hero_values_readback,
            combos.len(),
        );
        copy_buffer(
            &mut encoder,
            &root_tile.villain_values_buffer,
            &root_villain_values_readback,
            combos.len(),
        );
        let submission = self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;
        let mut action_values = read_f32_buffer(&self.device, &action_values_readback, action_len)?;
        let value_weights = read_f32_buffer(&self.device, &action_weights_readback, action_len)?;
        for (value, weight) in action_values.iter_mut().zip(value_weights) {
            if weight > 0.0 {
                *value /= weight;
            } else {
                *value = 0.0;
            }
        }
        Ok(GpuRootTerminalValues {
            action_values,
            reach_weights: read_f32_buffer(&self.device, &reach_weights_readback, state.infosets)?,
            strategy_weights: read_f32_buffer(
                &self.device,
                &strategy_weights_readback,
                state.infosets,
            )?,
            root_hero_values: read_f32_buffer(
                &self.device,
                &root_hero_values_readback,
                combos.len(),
            )?,
            root_villain_values: read_f32_buffer(
                &self.device,
                &root_villain_values_readback,
                combos.len(),
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn public_tree_update_state(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        villain_weights: &[f32],
        showdown_boards: &[GpuFinalBoard],
        state: &mut GpuDenseCfrState,
        iteration: usize,
    ) -> Result<(), GpuCfrError> {
        let profile = Self::gpu_profile_enabled();
        let setup_start = profile.then(Instant::now);
        let context = self.public_tree_iteration_context(
            nodes,
            children,
            child_cards,
            combos,
            combo_legal,
            villain_weights,
            showdown_boards,
            state.infosets,
            state.actions,
        );
        if let Some(start) = setup_start {
            eprintln!(
                "pokedr: gpu profile phase=cfv_setup elapsed_ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        self.public_tree_update_state_with_context(&context, state, iteration)
    }

    fn public_tree_update_state_with_context(
        &self,
        context: &GpuPublicTreeIterationContext,
        state: &mut GpuDenseCfrState,
        iteration: usize,
    ) -> Result<(), GpuCfrError> {
        let profile = Self::gpu_profile_enabled();
        let cfv_start = profile.then(Instant::now);
        let (output_buffers, _output_len, _action_len) = self
            .public_tree_iteration_output_with_context(
                context,
                &state.regrets,
                &state.prediction,
                state.variant,
                2,
            )?;
        if let Some(start) = cfv_start {
            self.profile_poll()?;
            eprintln!(
                "pokedr: gpu profile iteration={} phase=cfv elapsed_ms={:.3}",
                iteration,
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        let params = readonly_buffer(
            &self.device,
            "public tree CFR update params",
            &[
                state.infosets as u32,
                state.actions as u32,
                variant_code(state.variant),
                iteration as u32,
                variant_dcfr_alpha(state.variant, iteration).to_bits(),
                variant_dcfr_gamma(state.variant, iteration).to_bits(),
                variant_prediction_eta(state.variant).to_bits(),
            ],
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("public tree CFR update bind group"),
            layout: &self.public_tree_cfr_update_bind_group_layout,
            entries: &[
                bind_entry(0, &state.regrets),
                bind_entry(1, &state.strategy_sum),
                bind_entry(2, &output_buffers.action_values),
                bind_entry(3, &output_buffers.action_weights),
                bind_entry(4, &output_buffers.reach_weights),
                bind_entry(5, &output_buffers.strategy_weights),
                bind_entry(6, &params),
                bind_entry(7, &state.legal_actions_buffer),
                bind_entry(8, &state.prediction),
            ],
        });
        let update_start = profile.then(Instant::now);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("public tree CFR update encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("public tree CFR update pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.public_tree_cfr_update_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (state.infosets as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        if let Some(start) = update_start {
            self.profile_poll()?;
            eprintln!(
                "pokedr: gpu profile iteration={} phase=cfr_update elapsed_ms={:.3}",
                iteration,
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn public_tree_run_iterations(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        villain_weights: &[f32],
        showdown_boards: &[GpuFinalBoard],
        state: &mut GpuDenseCfrState,
        iterations: usize,
    ) -> Result<(), GpuCfrError> {
        self.public_tree_run_iterations_from(
            nodes,
            children,
            child_cards,
            combos,
            combo_legal,
            villain_weights,
            showdown_boards,
            state,
            1,
            iterations.max(1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn public_tree_run_iterations_from(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        villain_weights: &[f32],
        showdown_boards: &[GpuFinalBoard],
        state: &mut GpuDenseCfrState,
        first_iteration: usize,
        iterations: usize,
    ) -> Result<(), GpuCfrError> {
        if iterations == 0 {
            return Ok(());
        }
        let profile = Self::gpu_profile_enabled();
        let setup_start = profile.then(Instant::now);
        let context = self.public_tree_iteration_context(
            nodes,
            children,
            child_cards,
            combos,
            combo_legal,
            villain_weights,
            showdown_boards,
            state.infosets,
            state.actions,
        );
        if let Some(start) = setup_start {
            eprintln!(
                "pokedr: gpu profile phase=cfv_setup elapsed_ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        let flush_interval = std::env::var("POKEDR_GPU_ITERATION_FLUSH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(4)
            .max(1);
        for iteration in first_iteration..first_iteration + iterations {
            self.public_tree_update_state_with_context(&context, state, iteration)?;
            if iteration % flush_interval == 0 {
                let flush_start = profile.then(Instant::now);
                self.device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: None,
                    })
                    .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;
                if let Some(start) = flush_start {
                    eprintln!(
                        "pokedr: gpu profile iteration={} phase=flush elapsed_ms={:.3}",
                        iteration,
                        start.elapsed().as_secs_f64() * 1000.0
                    );
                }
            }
        }
        self.wait_idle()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn public_tree_run_iteration_checkpoints(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        villain_weights: &[f32],
        showdown_boards: &[GpuFinalBoard],
        state: &mut GpuDenseCfrState,
        checkpoints: &[usize],
    ) -> Result<Vec<(usize, DenseCfrState, f32)>, GpuCfrError> {
        let mut checkpoints: Vec<_> = checkpoints
            .iter()
            .copied()
            .map(|value| value.max(1))
            .collect();
        checkpoints.sort_unstable();
        checkpoints.dedup();
        let Some(&last_checkpoint) = checkpoints.last() else {
            return Ok(Vec::new());
        };

        let profile = Self::gpu_profile_enabled();
        let setup_start = profile.then(Instant::now);
        let context = self.public_tree_iteration_context(
            nodes,
            children,
            child_cards,
            combos,
            combo_legal,
            villain_weights,
            showdown_boards,
            state.infosets,
            state.actions,
        );
        if let Some(start) = setup_start {
            eprintln!(
                "pokedr: gpu profile phase=cfv_setup elapsed_ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }

        let flush_interval = std::env::var("POKEDR_GPU_ITERATION_FLUSH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(4)
            .max(1);
        let mut checkpoint_index = 0usize;
        let mut states = Vec::with_capacity(checkpoints.len());
        let started = Instant::now();
        for iteration in 1..=last_checkpoint {
            self.public_tree_update_state_with_context(&context, state, iteration)?;
            if iteration % flush_interval == 0 {
                self.device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: None,
                    })
                    .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;
            }
            if checkpoints.get(checkpoint_index) == Some(&iteration) {
                let downloaded = state.download(self)?;
                states.push((iteration, downloaded, started.elapsed().as_secs_f32()));
                checkpoint_index += 1;
            }
        }
        Ok(states)
    }

    pub fn upload_state(&self, state: &DenseCfrState) -> GpuDenseCfrState {
        let legal_actions = legal_actions_u32(&state.legal_actions);
        GpuDenseCfrState {
            infosets: state.infosets,
            actions: state.actions,
            variant: state.variant,
            legal_actions_buffer: readonly_buffer(
                &self.device,
                "resident legal actions",
                &legal_actions,
            ),
            legal_actions,
            regrets: storage_buffer(&self.device, "resident regrets", &state.regrets),
            prediction: storage_buffer(&self.device, "resident prediction", &state.prediction),
            strategy_sum: storage_buffer(
                &self.device,
                "resident strategy sum",
                &state.strategy_sum,
            ),
        }
    }

    pub fn zeroed_state(&self, config: super::DenseCfrConfig) -> GpuDenseCfrState {
        let state = DenseCfrState::new(config);
        self.upload_state(&state)
    }

    pub fn zeroed_state_with_legal_actions(
        &self,
        config: super::DenseCfrConfig,
        legal_actions: Vec<bool>,
    ) -> GpuDenseCfrState {
        let state = DenseCfrState::new_with_legal_actions(config, legal_actions);
        self.upload_state(&state)
    }

    pub fn resident_solver(&self, config: DenseCfrConfig) -> GpuResidentDenseCfrSolver {
        GpuResidentDenseCfrSolver {
            state: self.zeroed_state(config.clone()),
            config,
            iterations: 0,
        }
    }

    pub fn resident_solver_with_legal_actions(
        &self,
        config: DenseCfrConfig,
        legal_actions: Vec<bool>,
    ) -> GpuResidentDenseCfrSolver {
        GpuResidentDenseCfrSolver {
            state: self.zeroed_state_with_legal_actions(config.clone(), legal_actions),
            config,
            iterations: 0,
        }
    }
}

impl GpuDenseCfrState {
    pub fn infosets(&self) -> usize {
        self.infosets
    }

    pub fn actions(&self) -> usize {
        self.actions
    }

    pub fn update_all_infosets(
        &mut self,
        backend: &GpuDenseCfrBackend,
        action_values: &[f32],
        reach_weights: &[f32],
        strategy_weights: &[f32],
        iteration: usize,
    ) -> Result<(), GpuCfrError> {
        assert_eq!(action_values.len(), self.infosets * self.actions);
        assert_eq!(reach_weights.len(), self.infosets);
        assert_eq!(strategy_weights.len(), self.infosets);

        let params = [
            self.infosets as u32,
            self.actions as u32,
            variant_code(self.variant),
            iteration as u32,
            variant_dcfr_alpha(self.variant, iteration).to_bits(),
            variant_dcfr_gamma(self.variant, iteration).to_bits(),
            variant_prediction_eta(self.variant).to_bits(),
        ];
        let action_values =
            readonly_buffer(&backend.device, "resident action values", action_values);
        let reach_weights =
            readonly_buffer(&backend.device, "resident reach weights", reach_weights);
        let strategy_weights = readonly_buffer(
            &backend.device,
            "resident strategy weights",
            strategy_weights,
        );
        let params = readonly_buffer(&backend.device, "resident params", &params);
        let bind_group = backend
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("resident dense CFR bind group"),
                layout: &backend.bind_group_layout,
                entries: &[
                    bind_entry(0, &self.regrets),
                    bind_entry(1, &self.strategy_sum),
                    bind_entry(2, &action_values),
                    bind_entry(3, &reach_weights),
                    bind_entry(4, &strategy_weights),
                    bind_entry(5, &params),
                    bind_entry(6, &self.legal_actions_buffer),
                    bind_entry(7, &self.prediction),
                ],
            });

        let mut encoder = backend
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident dense CFR update encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("resident dense CFR update pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&backend.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (self.infosets as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        let submission = backend.queue.submit(Some(encoder.finish()));
        backend
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;
        Ok(())
    }

    pub fn download(&self, backend: &GpuDenseCfrBackend) -> Result<DenseCfrState, GpuCfrError> {
        let len = self.infosets * self.actions;
        let regret_readback = readback_buffer(&backend.device, len);
        let prediction_readback = readback_buffer(&backend.device, len);
        let strategy_readback = readback_buffer(&backend.device, len);
        let mut encoder = backend
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident dense CFR download encoder"),
            });
        copy_buffer(&mut encoder, &self.regrets, &regret_readback, len);
        copy_buffer(&mut encoder, &self.prediction, &prediction_readback, len);
        copy_buffer(&mut encoder, &self.strategy_sum, &strategy_readback, len);
        let submission = backend.queue.submit(Some(encoder.finish()));
        backend
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;
        let legal_actions: Vec<_> = self.legal_actions.iter().map(|value| *value != 0).collect();
        let legal_action_counts =
            super::legal_action_counts(self.infosets, self.actions, &legal_actions);
        Ok(DenseCfrState {
            infosets: self.infosets,
            actions: self.actions,
            variant: self.variant,
            legal_actions,
            legal_action_counts,
            regrets: read_f32_buffer(&backend.device, &regret_readback, len)?,
            prediction: read_f32_buffer(&backend.device, &prediction_readback, len)?,
            strategy_sum: read_f32_buffer(&backend.device, &strategy_readback, len)?,
        })
    }
}

impl GpuResidentDenseCfrSolver {
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn run_iterations(
        &mut self,
        backend: &GpuDenseCfrBackend,
        count: usize,
        mut fill_iteration: impl FnMut(usize, &mut DenseCfrIteration),
    ) -> Result<DenseCfrRunStats, GpuCfrError> {
        let mut batch = DenseCfrIteration::new(&self.config);
        for _ in 0..count {
            let iteration = self.iterations + 1;
            fill_iteration(iteration, &mut batch);
            batch.validate(&self.config);
            self.state.update_all_infosets(
                backend,
                &batch.action_values,
                &batch.reach_weights,
                &batch.strategy_weights,
                iteration,
            )?;
            self.iterations = iteration;
        }
        Ok(DenseCfrRunStats { iterations: count })
    }

    pub fn download(&self, backend: &GpuDenseCfrBackend) -> Result<DenseCfrState, GpuCfrError> {
        self.state.download(backend)
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn storage_buffer(device: &wgpu::Device, label: &str, data: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

fn uninit_storage_buffer(
    device: &wgpu::Device,
    label: &str,
    len: usize,
    copy_src: bool,
) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE;
    if copy_src {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: byte_len::<f32>(len),
        usage,
        mapped_at_creation: false,
    })
}

fn readonly_buffer<T: bytemuck::NoUninit>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn uniform_buffer<T: bytemuck::NoUninit>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn dispatch_grid(invocations: usize) -> (u32, u32, u32) {
    let groups = (invocations as u32).div_ceil(WORKGROUP_SIZE).max(1);
    let x_groups = groups.min(65_535);
    let y_groups = groups.div_ceil(x_groups);
    let x_invocations = x_groups * WORKGROUP_SIZE;
    (x_groups, y_groups, x_invocations)
}

fn readback_buffer(device: &wgpu::Device, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dense CFR readback"),
        size: byte_len::<f32>(len),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn copy_buffer(
    encoder: &mut wgpu::CommandEncoder,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    len: usize,
) {
    encoder.copy_buffer_to_buffer(src, 0, dst, 0, byte_len::<f32>(len));
}

fn read_f32_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Result<Vec<f32>, GpuCfrError> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;
    receiver
        .recv()
        .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?
        .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;
    let mapped = slice.get_mapped_range();
    let values = bytemuck::cast_slice::<u8, f32>(&mapped)[..len].to_vec();
    drop(mapped);
    buffer.unmap();
    Ok(values)
}

fn byte_len<T>(len: usize) -> u64 {
    (len * std::mem::size_of::<T>()) as u64
}

fn variant_code(variant: super::CfrVariant) -> u32 {
    match variant {
        super::CfrVariant::CfrPlus => 0,
        super::CfrVariant::Discounted => 1,
        super::CfrVariant::DcfrPlus { .. } => 2,
        super::CfrVariant::PdcfrPlus { .. } => 3,
        super::CfrVariant::DcfrSchedule { .. } => 4,
    }
}

fn variant_dcfr_alpha(variant: super::CfrVariant, iteration: usize) -> f32 {
    match variant {
        super::CfrVariant::DcfrPlus { alpha, .. } | super::CfrVariant::PdcfrPlus { alpha, .. } => {
            alpha
        }
        super::CfrVariant::DcfrSchedule {
            alpha_start,
            alpha_end,
            horizon,
            ..
        } => scheduled_value(alpha_start, alpha_end, iteration, horizon),
        _ => super::DEFAULT_DCFR_PLUS_ALPHA,
    }
}

fn variant_dcfr_gamma(variant: super::CfrVariant, iteration: usize) -> f32 {
    match variant {
        super::CfrVariant::DcfrPlus { gamma, .. } | super::CfrVariant::PdcfrPlus { gamma, .. } => {
            gamma
        }
        super::CfrVariant::DcfrSchedule {
            gamma_start,
            gamma_end,
            horizon,
            ..
        } => scheduled_value(gamma_start, gamma_end, iteration, horizon),
        _ => super::DEFAULT_DCFR_PLUS_GAMMA,
    }
}

fn variant_prediction_eta(variant: super::CfrVariant) -> f32 {
    match variant {
        super::CfrVariant::PdcfrPlus { eta, .. } => eta,
        _ => 0.0,
    }
}

fn scheduled_value(start: f32, end: f32, iteration: usize, horizon: usize) -> f32 {
    let horizon = horizon.max(2);
    let progress = (iteration.saturating_sub(1) as f32 / (horizon - 1) as f32).clamp(0.0, 1.0);
    start + (end - start) * progress
}

fn legal_actions_u32(legal_actions: &[bool]) -> Vec<u32> {
    legal_actions
        .iter()
        .map(|is_legal| u32::from(*is_legal))
        .collect()
}

fn nodes_public_infoset_count(nodes: &[GpuPublicTreeNode]) -> usize {
    nodes
        .iter()
        .filter(|node| node.kind == 0)
        .map(|node| node.public_infoset as usize + 1)
        .max()
        .unwrap_or(0)
}

fn public_tree_layered(
    nodes: &[GpuPublicTreeNode],
    children: &[u32],
    child_cards: &[u32],
    combo_count: usize,
    max_storage_buffer_binding_size: u64,
) -> GpuPublicTreeLayered {
    let mut depths = vec![0usize; nodes.len()];
    for (parent, node) in nodes.iter().enumerate() {
        if node.kind != 0 && node.kind != 1 {
            continue;
        }
        let child_depth = depths[parent] + 1;
        for action in 0..node.child_count as usize {
            let child = children[node.first_child as usize + action] as usize;
            depths[child] = depths[child].max(child_depth);
        }
    }

    let max_depth = depths.iter().copied().max().unwrap_or(0);
    let mut layer_globals = vec![Vec::new(); max_depth + 1];
    for (node_index, depth) in depths.iter().copied().enumerate() {
        layer_globals[depth].push(node_index as u32);
    }

    let mut global_to_layer = vec![(0u32, 0u32); nodes.len()];
    for (layer_index, globals) in layer_globals.iter().enumerate() {
        for (slot, node_index) in globals.iter().copied().enumerate() {
            global_to_layer[node_index as usize] = (layer_index as u32, slot as u32);
        }
    }

    let mut layers = Vec::with_capacity(layer_globals.len());
    let mut max_layer_nodes = 0usize;
    for globals in layer_globals {
        max_layer_nodes = max_layer_nodes.max(globals.len());
        let mut layer_nodes = Vec::with_capacity(globals.len());
        let mut layer_children = Vec::new();
        let mut layer_child_cards = Vec::new();
        for global_node in globals.iter().copied() {
            let source = nodes[global_node as usize];
            let first_child = layer_children.len() as u32;
            for action in 0..source.child_count as usize {
                let child_offset = source.first_child as usize + action;
                let global_child = children[child_offset] as usize;
                let (_, child_slot) = global_to_layer[global_child];
                layer_children.push(child_slot);
                layer_child_cards.push(child_cards.get(child_offset).copied().unwrap_or(52));
            }
            layer_nodes.push(GpuPublicTreeNode {
                first_child,
                ..source
            });
        }
        layers.push(GpuPublicTreeLayer {
            nodes: layer_nodes,
            children: layer_children,
            child_cards: layer_child_cards,
        });
    }

    let node_tile_size = layer_node_tile_size(combo_count, max_storage_buffer_binding_size);
    let max_layer_tiles = layers
        .iter()
        .map(|layer| layer.nodes.len().div_ceil(node_tile_size))
        .max()
        .unwrap_or(0);
    let reach_edge_tiles = public_tree_layer_edge_tiles(&layers, node_tile_size);

    GpuPublicTreeLayered {
        node_tile_size,
        max_layer_tiles,
        reach_edge_tiles,
        layers,
        max_layer_nodes,
    }
}

fn layer_node_tile_size(combo_count: usize, max_storage_buffer_binding_size: u64) -> usize {
    let bytes_per_node = combo_count.max(1) as u64 * std::mem::size_of::<f32>() as u64;
    (max_storage_buffer_binding_size / bytes_per_node)
        .max(1)
        .try_into()
        .unwrap_or(usize::MAX)
}

fn public_tree_layer_edge_tiles(
    layers: &[GpuPublicTreeLayer],
    node_tile_size: usize,
) -> Vec<GpuPublicTreeLayerEdgeTile> {
    let mut tiles = Vec::new();
    for parent_layer in 0..layers.len().saturating_sub(1) {
        let child_layer = parent_layer + 1;
        let parent = &layers[parent_layer];
        let child = &layers[child_layer];
        for parent_start in (0..parent.nodes.len()).step_by(node_tile_size) {
            let parent_end = (parent_start + node_tile_size).min(parent.nodes.len());
            for child_start in (0..child.nodes.len()).step_by(node_tile_size) {
                let child_end = (child_start + node_tile_size).min(child.nodes.len());
                let mut edges = Vec::new();
                for parent_slot in parent_start..parent_end {
                    let node = parent.nodes[parent_slot];
                    if node.kind != 0 && node.kind != 1 {
                        continue;
                    }
                    for action in 0..node.child_count as usize {
                        let child_offset = node.first_child as usize + action;
                        let child_slot = parent.children[child_offset] as usize;
                        if !(child_start..child_end).contains(&child_slot) {
                            continue;
                        }
                        edges.push(GpuPublicTreeEdge {
                            parent: (parent_slot - parent_start) as u32,
                            child: (child_slot - child_start) as u32,
                            action: action as u32,
                            card: parent
                                .child_cards
                                .get(child_offset)
                                .copied()
                                .unwrap_or(u32::MAX),
                        });
                    }
                }
                if !edges.is_empty() {
                    tiles.push(GpuPublicTreeLayerEdgeTile {
                        parent_layer,
                        child_layer,
                        parent_tile: GpuPublicTreeLayerTile {
                            node_start: parent_start,
                        },
                        child_tile: GpuPublicTreeLayerTile {
                            node_start: child_start,
                        },
                        edges,
                    });
                }
            }
        }
    }
    tiles
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense_cfr::{CfrVariant, DenseCfrConfig};

    #[test]
    fn showdown_strength_order_is_sorted_and_marks_equal_groups() {
        let combos = [
            GpuPrivateCombo { cards: [12, 25] },
            GpuPrivateCombo { cards: [11, 24] },
            GpuPrivateCombo { cards: [0, 1] },
            GpuPrivateCombo { cards: [2, 3] },
        ];
        let boards = [GpuFinalBoard {
            cards: [4, 5, 6, 7, 8],
        }];

        let order = showdown_strength_order(&combos, &boards);
        assert_eq!(order.combo_order.len(), combos.len());
        assert_eq!(order.combo_bounds.len(), combos.len());
        assert_eq!(order.blocker_neighbor_stride, 1);

        let strengths: Vec<_> = order
            .combo_order
            .iter()
            .filter(|&&combo_index| combo_index != u32::MAX)
            .map(|combo_index| evaluate_combo_final_board(combos[*combo_index as usize], boards[0]))
            .collect();
        assert!(strengths.windows(2).all(|pair| pair[0] <= pair[1]));

        for combo_index in 0..combos.len() {
            let bounds = order.combo_bounds[combo_index];
            let start = bounds.group_start as usize;
            let end = bounds.group_end as usize;
            assert_eq!(bounds.legal, 1);
            assert!(start < end);
            assert!(end <= order.combo_order.len());
            let strength = evaluate_combo_final_board(combos[combo_index], boards[0]);
            assert!(
                order.combo_order[start..end]
                    .iter()
                    .all(
                        |peer| evaluate_combo_final_board(combos[*peer as usize], boards[0])
                            == strength
                    )
            );
        }
    }

    #[test]
    fn board_major_blocker_correction_matches_bruteforce_on_turn_runouts() {
        let public = [12, 18, 27, 35];
        let combos = [
            GpuPrivateCombo { cards: [0, 1] },
            GpuPrivateCombo { cards: [2, 3] },
            GpuPrivateCombo { cards: [4, 5] },
            GpuPrivateCombo { cards: [6, 7] },
            GpuPrivateCombo { cards: [8, 9] },
            GpuPrivateCombo { cards: [10, 11] },
            GpuPrivateCombo { cards: [13, 14] },
            GpuPrivateCombo { cards: [15, 16] },
        ];
        let reaches = [0.17, 0.03, 0.21, 0.11, 0.07, 0.19, 0.13, 0.09];
        let boards = full_final_boards(&public);

        let brute = brute_force_terminal_cfv(&combos, &boards, &reaches, 17.5, 6.0);
        let board_major = board_major_terminal_cfv(&combos, &public, &boards, &reaches, 17.5, 6.0);
        let card_aggregate =
            card_aggregate_terminal_cfv(&combos, &public, &boards, &reaches, 17.5, 6.0);
        assert_close_vec("turn board-major cfv", &brute, &board_major, 1.0e-4);
        assert_close_vec(
            "turn card-aggregate cfv",
            &board_major,
            &card_aggregate,
            1.0e-4,
        );
    }

    #[test]
    fn board_major_blocker_correction_matches_bruteforce_on_flop_runouts() {
        let public = [12, 18, 27];
        let combos = [
            GpuPrivateCombo { cards: [0, 1] },
            GpuPrivateCombo { cards: [2, 3] },
            GpuPrivateCombo { cards: [4, 5] },
            GpuPrivateCombo { cards: [6, 7] },
            GpuPrivateCombo { cards: [8, 9] },
            GpuPrivateCombo { cards: [10, 11] },
            GpuPrivateCombo { cards: [13, 14] },
        ];
        let reaches = [0.23, 0.05, 0.17, 0.31, 0.02, 0.13, 0.09];
        let boards = full_final_boards(&public);

        let brute = brute_force_terminal_cfv(&combos, &boards, &reaches, 22.0, 8.5);
        let board_major = board_major_terminal_cfv(&combos, &public, &boards, &reaches, 22.0, 8.5);
        let card_aggregate =
            card_aggregate_terminal_cfv(&combos, &public, &boards, &reaches, 22.0, 8.5);
        assert_close_vec("flop board-major cfv", &brute, &board_major, 1.0e-4);
        assert_close_vec(
            "flop card-aggregate cfv",
            &board_major,
            &card_aggregate,
            1.0e-4,
        );
    }

    fn full_final_boards(public: &[u32]) -> Vec<GpuFinalBoard> {
        assert!(public.len() <= 5);
        let public_mask = public.iter().fold(0u64, |mask, card| mask | (1u64 << card));
        let missing = 5 - public.len();
        let deck: Vec<_> = (0..Card::COUNT as u32)
            .filter(|card| public_mask & (1u64 << card) == 0)
            .collect();
        let mut boards = Vec::new();
        match missing {
            0 => {
                boards.push(final_board_from_public_runout(public, &[]));
            }
            1 => {
                for &card in &deck {
                    boards.push(final_board_from_public_runout(public, &[card]));
                }
            }
            2 => {
                for first in 0..deck.len() {
                    for second in first + 1..deck.len() {
                        boards.push(final_board_from_public_runout(
                            public,
                            &[deck[first], deck[second]],
                        ));
                    }
                }
            }
            _ => panic!("tests only cover flop or later boards"),
        }
        boards
    }

    fn final_board_from_public_runout(public: &[u32], runout: &[u32]) -> GpuFinalBoard {
        let mut cards = [u32::MAX; 5];
        for (slot, card) in public.iter().chain(runout).enumerate() {
            cards[slot] = *card;
        }
        GpuFinalBoard { cards }
    }

    fn brute_force_terminal_cfv(
        combos: &[GpuPrivateCombo],
        boards: &[GpuFinalBoard],
        reaches: &[f32],
        pot: f32,
        invested: f32,
    ) -> Vec<f32> {
        let mut values = vec![0.0; combos.len()];
        for (hero_index, &hero) in combos.iter().enumerate() {
            for (villain_index, &villain) in combos.iter().enumerate() {
                if combos_collide(hero, villain) {
                    continue;
                }
                let mut equity_sum = 0.0;
                let mut valid_boards = 0usize;
                for &board in boards {
                    if combo_hits_final_board(hero, board) || combo_hits_final_board(villain, board)
                    {
                        continue;
                    }
                    valid_boards += 1;
                    let hero_strength = evaluate_combo_final_board(hero, board);
                    let villain_strength = evaluate_combo_final_board(villain, board);
                    if hero_strength > villain_strength {
                        equity_sum += 1.0;
                    } else if hero_strength == villain_strength {
                        equity_sum += 0.5;
                    }
                }
                if valid_boards == 0 {
                    continue;
                }
                values[hero_index] +=
                    reaches[villain_index] * (pot * equity_sum / valid_boards as f32 - invested);
            }
        }
        values
    }

    fn board_major_terminal_cfv(
        combos: &[GpuPrivateCombo],
        public: &[u32],
        boards: &[GpuFinalBoard],
        reaches: &[f32],
        pot: f32,
        invested: f32,
    ) -> Vec<f32> {
        let denominator = full_runout_pair_denominator(public.len()) as f32;
        let mut values = vec![0.0; combos.len()];
        for &board in boards {
            let mut ordered: Vec<_> = combos
                .iter()
                .enumerate()
                .filter_map(|(combo_index, &combo)| {
                    (!combo_hits_final_board(combo, board)).then(|| {
                        (
                            evaluate_combo_final_board(combo, board),
                            combo_index,
                            reaches[combo_index],
                        )
                    })
                })
                .collect();
            ordered.sort_unstable_by_key(|(strength, combo_index, _)| (*strength, *combo_index));

            let mut prefix = Vec::with_capacity(ordered.len() + 1);
            prefix.push(0.0f32);
            for &(_, _, reach) in &ordered {
                prefix.push(prefix.last().copied().unwrap() + reach);
            }

            for (hero_index, &hero) in combos.iter().enumerate() {
                if combo_hits_final_board(hero, board) {
                    continue;
                }
                let hero_strength = evaluate_combo_final_board(hero, board);
                let group_start =
                    ordered.partition_point(|(strength, _, _)| *strength < hero_strength);
                let group_end =
                    ordered.partition_point(|(strength, _, _)| *strength <= hero_strength);

                let win_raw = prefix[group_start];
                let tie_raw = prefix[group_end] - prefix[group_start];
                let total_raw = prefix.last().copied().unwrap_or(0.0);
                let mut block_win = 0.0;
                let mut block_tie = 0.0;
                let mut block_total = 0.0;
                for (villain_index, &villain) in combos.iter().enumerate() {
                    if !combos_collide(hero, villain) || combo_hits_final_board(villain, board) {
                        continue;
                    }
                    let villain_reach = reaches[villain_index];
                    let villain_strength = evaluate_combo_final_board(villain, board);
                    block_total += villain_reach;
                    if villain_strength < hero_strength {
                        block_win += villain_reach;
                    } else if villain_strength == hero_strength {
                        block_tie += villain_reach;
                    }
                }

                let win = win_raw - block_win;
                let tie = tie_raw - block_tie;
                let total = total_raw - block_total;
                values[hero_index] += (pot * (win + 0.5 * tie) - invested * total) / denominator;
            }
        }
        values
    }

    fn card_aggregate_terminal_cfv(
        combos: &[GpuPrivateCombo],
        public: &[u32],
        boards: &[GpuFinalBoard],
        reaches: &[f32],
        pot: f32,
        invested: f32,
    ) -> Vec<f32> {
        let denominator = full_runout_pair_denominator(public.len()) as f32;
        let mut values = vec![0.0; combos.len()];
        for &board in boards {
            let mut ordered: Vec<_> = combos
                .iter()
                .enumerate()
                .filter_map(|(combo_index, &combo)| {
                    (!combo_hits_final_board(combo, board)).then(|| {
                        (
                            evaluate_combo_final_board(combo, board),
                            combo_index,
                            reaches[combo_index],
                        )
                    })
                })
                .collect();
            ordered.sort_unstable_by_key(|(strength, combo_index, _)| (*strength, *combo_index));

            let mut group_starts = Vec::new();
            let mut group_ends = Vec::new();
            let mut cursor = 0usize;
            while cursor < ordered.len() {
                let strength = ordered[cursor].0;
                let end = ordered.partition_point(|(candidate, _, _)| *candidate <= strength);
                group_starts.push(cursor);
                group_ends.push(end);
                cursor = end;
            }

            let group_count = group_starts.len();
            let mut combo_group = vec![usize::MAX; combos.len()];
            let mut equal_total = vec![0.0f32; group_count];
            let mut equal_card = vec![vec![0.0f32; group_count]; Card::COUNT];

            for group in 0..group_count {
                for &(_, combo_index, reach) in &ordered[group_starts[group]..group_ends[group]] {
                    combo_group[combo_index] = group;
                    equal_total[group] += reach;
                    let combo = combos[combo_index];
                    equal_card[combo.cards[0] as usize][group] += reach;
                    equal_card[combo.cards[1] as usize][group] += reach;
                }
            }

            let mut prefix_total = vec![0.0f32; group_count + 1];
            for group in 0..group_count {
                prefix_total[group + 1] = prefix_total[group] + equal_total[group];
            }

            let mut prefix_card = vec![vec![0.0f32; group_count + 1]; Card::COUNT];
            for card in 0..Card::COUNT {
                for group in 0..group_count {
                    prefix_card[card][group + 1] =
                        prefix_card[card][group] + equal_card[card][group];
                }
            }

            for (hero_index, &hero) in combos.iter().enumerate() {
                if combo_hits_final_board(hero, board) {
                    continue;
                }
                let group = combo_group[hero_index];
                assert_ne!(group, usize::MAX);
                let hero_reach = reaches[hero_index];
                let first_card = hero.cards[0] as usize;
                let second_card = hero.cards[1] as usize;

                let win_raw = prefix_total[group];
                let tie_raw = equal_total[group];
                let total_raw = prefix_total[group_count];

                let block_total = prefix_card[first_card][group_count]
                    + prefix_card[second_card][group_count]
                    - hero_reach;
                let block_win = prefix_card[first_card][group] + prefix_card[second_card][group];
                let block_tie =
                    equal_card[first_card][group] + equal_card[second_card][group] - hero_reach;

                let win = win_raw - block_win;
                let tie = tie_raw - block_tie;
                let total = total_raw - block_total;
                values[hero_index] += (pot * (win + 0.5 * tie) - invested * total) / denominator;
            }
        }
        values
    }

    fn full_runout_pair_denominator(public_len: usize) -> usize {
        let missing = 5 - public_len;
        let remaining_after_public_and_pair = Card::COUNT - public_len - 4;
        match missing {
            0 => 1,
            1 => remaining_after_public_and_pair,
            2 => remaining_after_public_and_pair * (remaining_after_public_and_pair - 1) / 2,
            _ => panic!("tests only cover flop or later boards"),
        }
    }

    fn combos_collide(left: GpuPrivateCombo, right: GpuPrivateCombo) -> bool {
        left.cards[0] == right.cards[0]
            || left.cards[0] == right.cards[1]
            || left.cards[1] == right.cards[0]
            || left.cards[1] == right.cards[1]
    }

    fn combo_hits_final_board(combo: GpuPrivateCombo, board: GpuFinalBoard) -> bool {
        board.cards.contains(&combo.cards[0]) || board.cards.contains(&combo.cards[1])
    }

    fn assert_close_vec(label: &str, left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len());
        for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
            assert!(
                (left - right).abs() <= tolerance,
                "{label} mismatch at {index}: left={left} right={right}"
            );
        }
    }

    #[test]
    #[ignore = "GPU tests must run on the main thread; use `cargo test -p pokedr-core --test gpu_smoke`"]
    fn gpu_update_matches_cpu_reference_when_adapter_exists() {
        let backend = match GpuDenseCfrBackend::new() {
            Ok(backend) => backend,
            Err(GpuCfrError::NoAdapter) => return,
            Err(error) => panic!("unexpected GPU init error: {error:?}"),
        };
        let config = DenseCfrConfig {
            infosets: 4,
            actions: 3,
            variant: CfrVariant::CfrPlus,
        };
        let mut cpu = DenseCfrState::new(config.clone());
        let mut gpu = DenseCfrState::new(config);
        let action_values = [
            1.0, -0.5, 0.25, -1.0, 2.0, 0.0, 0.5, 0.25, -0.75, 3.0, 1.0, -2.0,
        ];
        let reach_weights = [1.0, 0.5, 2.0, 0.25];
        let strategy_weights = [1.0, 1.0, 0.5, 2.0];

        cpu.update_all_infosets(&action_values, &reach_weights, &strategy_weights, 1);
        backend
            .update_all_infosets(
                &mut gpu,
                &action_values,
                &reach_weights,
                &strategy_weights,
                1,
            )
            .unwrap();

        for (left, right) in cpu.regrets().iter().zip(gpu.regrets()) {
            assert!((left - right).abs() < 1e-5, "{left} != {right}");
        }
        for (left, right) in cpu.strategy_sum().iter().zip(gpu.strategy_sum()) {
            assert!((left - right).abs() < 1e-5, "{left} != {right}");
        }
    }

    #[test]
    #[ignore = "GPU tests must run on the main thread; use `cargo test -p pokedr-core --test gpu_smoke`"]
    fn resident_gpu_state_matches_cpu_after_multiple_updates() {
        let backend = match GpuDenseCfrBackend::new() {
            Ok(backend) => backend,
            Err(GpuCfrError::NoAdapter) => return,
            Err(error) => panic!("unexpected GPU init error: {error:?}"),
        };
        let config = DenseCfrConfig {
            infosets: 8,
            actions: 4,
            variant: CfrVariant::Discounted,
        };
        let mut cpu = DenseCfrState::new(config.clone());
        let mut gpu = backend.zeroed_state(config);
        let reach_weights = vec![1.0; 8];
        let strategy_weights = vec![0.75; 8];

        for iteration in 1..=5 {
            let action_values: Vec<_> = (0..32)
                .map(|index| ((index as f32 + iteration as f32) * 0.25).sin())
                .collect();
            cpu.update_all_infosets(&action_values, &reach_weights, &strategy_weights, iteration);
            gpu.update_all_infosets(
                &backend,
                &action_values,
                &reach_weights,
                &strategy_weights,
                iteration,
            )
            .unwrap();
        }

        let downloaded = gpu.download(&backend).unwrap();
        for (left, right) in cpu.regrets().iter().zip(downloaded.regrets()) {
            assert!((left - right).abs() < 1e-5, "{left} != {right}");
        }
        for (left, right) in cpu.strategy_sum().iter().zip(downloaded.strategy_sum()) {
            assert!((left - right).abs() < 1e-5, "{left} != {right}");
        }
    }
}
