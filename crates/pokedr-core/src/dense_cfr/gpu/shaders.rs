pub(super) const CFR_UPDATE_SHADER: &str = r#"
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

fn average_strategy_weight_multiplier(iteration: u32) -> f32 {
    let delay = params[7];
    let power = bitcast<f32>(params[8]);
    if delay == 0u && power == 0.0 {
        return 1.0;
    }
    if iteration <= delay {
        return 0.0;
    }
    return pow(f32(iteration - delay), power);
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
        strategy_sum[offset + action] = strategy_sum[offset + action] * average_discount + strategy_weights[infoset] * average_strategy_weight_multiplier(iteration) * strategy;
    }
}
"#;

pub(super) const PUBLIC_TREE_CFR_UPDATE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> regrets: array<f32>;
@group(0) @binding(1) var<storage, read_write> strategy_sum: array<f32>;
@group(0) @binding(2) var<storage, read> action_values: array<f32>;
@group(0) @binding(3) var<storage, read> _unused_action_weights: array<f32>;
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

fn action_value(action_offset: u32, infoset: u32) -> f32 {
    let weight = reach_weights[infoset];
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

fn average_strategy_weight_multiplier(iteration: u32) -> f32 {
    let delay = params[7];
    let power = bitcast<f32>(params[8]);
    if delay == 0u && power == 0.0 {
        return 1.0;
    }
    if iteration <= delay {
        return 0.0;
    }
    return pow(f32(iteration - delay), power);
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
        node_value = node_value + strategy * action_value(offset + action, infoset);
    }

    let discount = regret_discount(variant, iteration, dcfr_alpha);
    let average_discount = average_strategy_discount(variant, iteration, dcfr_gamma);

    let reach_weight = reach_weights[infoset];
    let raw_strategy_weight = strategy_weights[infoset];
    var strategy_weight = raw_strategy_weight * f32(iteration);
    if variant == 2u || variant == 3u || variant == 4u {
        strategy_weight = raw_strategy_weight;
    }
    if params[7] != 0u || bitcast<f32>(params[8]) != 0.0 {
        strategy_weight = raw_strategy_weight * average_strategy_weight_multiplier(iteration);
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
        let regret = (action_value(offset + action, infoset) - node_value) * reach_weight;
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

pub(super) const SHOWDOWN_SHADER: &str = r#"
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

pub(super) const SHOWDOWN_MATRIX_SHADER: &str = r#"
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

pub(super) const PUBLIC_TREE_LAYER_REACH_INIT_SHADER: &str = r#"
struct Params {
    combo_count: u32,
    node_count: u32,
    max_actions: u32,
    output_len: u32,
    variant: u32,
    edge_count: u32,
    aux0: u32,
    eta_bits: u32,
};

@group(0) @binding(0) var<storage, read> root_reach_weights: array<f32>;
@group(0) @binding(1) var<storage, read_write> hero_reaches: array<f32>;
@group(0) @binding(2) var<storage, read_write> villain_reaches: array<f32>;
@group(0) @binding(3) var<storage, read_write> combo_live: array<atomic<u32>>;
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
    let live_word = index >> 5u;
    let live_mask = 1u << (index & 31u);
    atomicAnd(&combo_live[live_word], 0xffffffffu ^ live_mask);
    if params.aux0 == 0u && local_node == 0u && root_reach_weights[combo] >= 0.0 {
        hero_reaches[index] = root_reach_weights[combo];
        villain_reaches[index] = root_reach_weights[params.combo_count + combo];
        atomicOr(&combo_live[live_word], live_mask);
    }
}
"#;

pub(super) const PUBLIC_TREE_LAYER_REACH_SHADER: &str = r#"
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
    aux0: u32,
    eta_bits: u32,
};

@group(0) @binding(0) var<storage, read> parent_nodes: array<TreeNode>;
@group(0) @binding(1) var<storage, read> edges: array<Edge>;
@group(0) @binding(2) var<storage, read> combos: array<Combo>;
@group(0) @binding(3) var<storage, read> root_reach_weights: array<f32>;
@group(0) @binding(4) var<storage, read> regrets: array<f32>;
@group(0) @binding(5) var<storage, read> parent_hero_reaches: array<f32>;
@group(0) @binding(6) var<storage, read> parent_villain_reaches: array<f32>;
@group(0) @binding(7) var<storage, read> parent_combo_live: array<u32>;
@group(0) @binding(8) var<storage, read_write> child_hero_reaches: array<f32>;
@group(0) @binding(9) var<storage, read_write> child_villain_reaches: array<f32>;
@group(0) @binding(10) var<storage, read_write> child_combo_live: array<atomic<u32>>;
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
    let public_base = params.node_count;
    let private_infoset = (node.public_infoset - public_base) * params.combo_count + private_combo;
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
    if node.kind == 0u {
        let probability = strategy_probability(node, combo, edge.action);
        let is_br_player = params.aux0 < 2u && node.acting_player == params.aux0;
        if node.acting_player == 0u {
            child_hero_reaches[child_offset] =
                hero_reach * select(probability, 1.0, is_br_player);
            child_villain_reaches[child_offset] = villain_reach;
        } else {
            child_hero_reaches[child_offset] = hero_reach;
            child_villain_reaches[child_offset] =
                villain_reach * select(probability, 1.0, is_br_player);
        }
    } else if node.kind == 1u {
        if combo_has_card(combos[combo], edge.card) {
            child_hero_reaches[child_offset] = 0.0;
            child_villain_reaches[child_offset] = 0.0;
        } else {
            child_hero_reaches[child_offset] = hero_reach;
            child_villain_reaches[child_offset] = villain_reach;
        }
    }
}
"#;

pub(super) const PUBLIC_TREE_TERMINAL_PARTIAL_SHADER: &str = r#"
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
struct TerminalRef {
    node: u32,
    table: u32,
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
@group(0) @binding(1) var<storage, read> terminal_refs: array<TerminalRef>;
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
    let terminal_ref = terminal_refs[group_terminal_slot];
    let node_index = terminal_ref.node;
    let node_offset = node_index * params.combo_count;
    let table_board = terminal_ref.table * params.board_count + board;
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

pub(super) const PUBLIC_TREE_TERMINAL_PARTIAL_SERIAL_SHADER: &str = r#"
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
struct TerminalRef {
    node: u32,
    table: u32,
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
@group(0) @binding(1) var<storage, read> terminal_refs: array<TerminalRef>;
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
    let terminal_ref = terminal_refs[group_terminal_slot];
    let node_index = terminal_ref.node;
    let node_offset = node_index * params.combo_count;
    let table_board = terminal_ref.table * params.board_count + board;
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

pub(super) const PUBLIC_TREE_TERMINAL_REDUCE_SHADER: &str = r#"
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
struct TerminalRef {
    node: u32,
    table: u32,
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
@group(0) @binding(1) var<storage, read> terminal_refs: array<TerminalRef>;
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
    let terminal_ref = terminal_refs[group_terminal_slot];
    let node_index = terminal_ref.node;
    let node = nodes[node_index];
    let node_offset = node_index * params.combo_count;
    let denom = max(node.showdown_denominator, 1.0);
    var hero_value = 0.0;
    var villain_value = 0.0;
    for (var local_board = 0u; local_board < node.board_count; local_board = local_board + 1u) {
        let table_board = terminal_ref.table * params.board_count + local_board;
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

pub(super) const PUBLIC_TREE_TERMINAL_CARD_PREFIX_SHADER: &str = r#"
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
struct TerminalRef {
    node: u32,
    table: u32,
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
@group(0) @binding(1) var<storage, read> terminal_refs: array<TerminalRef>;
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
    let terminal_ref = terminal_refs[group_terminal_slot];
    let node_index = terminal_ref.node;
    let node_offset = node_index * params.combo_count;
    let table_board = terminal_ref.table * params.board_count + board;
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

pub(super) const PUBLIC_TREE_TERMINAL_CARD_AGGREGATE_REDUCE_SHADER: &str = r#"
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
struct TerminalRef {
    node: u32,
    table: u32,
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
@group(0) @binding(1) var<storage, read> terminal_refs: array<TerminalRef>;
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
    let terminal_ref = terminal_refs[group_terminal_slot];
    let node_index = terminal_ref.node;
    let node = nodes[node_index];
    let node_offset = node_index * params.combo_count;
    let private_combo = combos[combo];
    let denom = max(node.showdown_denominator, 1.0);
    var hero_value = 0.0;
    var villain_value = 0.0;
    for (var local_board = 0u; local_board < node.board_count; local_board = local_board + 1u) {
        let table_board = terminal_ref.table * params.board_count + local_board;
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

pub(super) const PUBLIC_TREE_FOLD_AGGREGATE_SHADER: &str = r#"
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

pub(super) const PUBLIC_TREE_FOLD_VALUE_SHADER: &str = r#"
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
@group(0) @binding(9) var<storage, read> combo_live: array<u32>;
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
    let live_word = value_index >> 5u;
    let live_mask = 1u << (value_index & 31u);
    let structurally_live = (combo_live[live_word] & live_mask) != 0u;
    hero_values[value_index] = select(0.0, villain_noncolliding * hero_payoff * node._pad1, structurally_live);
    villain_values[value_index] = select(0.0, hero_noncolliding * (-hero_payoff) * node._pad1, structurally_live);
}
"#;

pub(super) const PUBLIC_TREE_LAYER_BACKUP_INIT_SHADER: &str = r#"
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
    variant: u32,
    public_infoset_base: u32,
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

pub(super) const PUBLIC_TREE_LAYER_BACKUP_SHADER: &str = r#"
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
    value_player_and_flags: u32,
    child_tile_start: u32,
    child_tile_end: u32,
    public_infoset_end: u32,
    public_infoset_base: u32,
    eta_bits: u32,
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
@group(0) @binding(10) var<storage, read> regrets: array<f32>;
@group(0) @binding(11) var<storage, read> prediction: array<f32>;
@group(0) @binding(12) var<uniform> params: Params;

fn effective_regret(index: u32) -> f32 {
    if variant() == 3u {
        return regrets[index] + bitcast<f32>(params.eta_bits) * prediction[index];
    }
    return regrets[index];
}

fn value_player() -> u32 {
    return params.value_player_and_flags & 0xffu;
}

fn variant() -> u32 {
    return (params.value_player_and_flags >> 8u) & 0xffu;
}

fn include_chance_nodes() -> bool {
    return ((params.value_player_and_flags >> 16u) & 1u) != 0u;
}

fn strategy_probability(node: TreeNode, private_combo: u32, action: u32) -> f32 {
    let private_infoset = (node.public_infoset - params.public_infoset_base) * params.combo_count + private_combo;
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
fn backup_child_tile(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * params.output_len;
    let value_count = params.node_count * params.combo_count;
    if index >= value_count {
        return;
    }
    let combo = index % params.combo_count;
    let node_slot = index / params.combo_count;
    let node = parent_nodes[node_slot];
    if node.kind == 0u {
        if node.public_infoset < params.public_infoset_base || node.public_infoset >= params.public_infoset_end {
            return;
        }
    } else if node.kind == 1u {
        if !include_chance_nodes() {
            return;
        }
    } else {
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
            let probability = strategy_probability(node, combo, action);
            if value_player() == 0u {
                hero_value = max(hero_value, hero_child_value);
            } else {
                hero_value = hero_value + probability * hero_child_value;
            }
            villain_value = villain_value + villain_child_value;
        } else {
            let probability = strategy_probability(node, combo, action);
            hero_value = hero_value + hero_child_value;
            if value_player() == 1u {
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

pub(super) const PUBLIC_TREE_LAYER_OUTPUT_SHADER: &str = r#"
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
@group(0) @binding(7) var<storage, read> combo_live: array<u32>;
@group(0) @binding(8) var<storage, read> edges: array<Edge>;
@group(0) @binding(9) var<storage, read> child_hero_values: array<f32>;
@group(0) @binding(10) var<storage, read> child_villain_values: array<f32>;
@group(0) @binding(11) var<storage, read> root_reach_weights: array<f32>;
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
            let live_word = offset >> 5u;
            let live_mask = 1u << (offset & 31u);
            if (combo_live[live_word] & live_mask) != 0u {
                if params.value_player == 2u {
                    hero_sum = hero_sum + max(root_reach_weights[combo], 0.0);
                    villain_sum = villain_sum + max(root_reach_weights[params.combo_count + combo], 0.0);
                } else {
                    hero_sum = hero_sum + hero_reaches[offset];
                    villain_sum = villain_sum + villain_reaches[offset];
                }
            }
        }
    } else {
        let card = slot - 1u;
        for (var combo = 0u; combo < params.combo_count; combo = combo + 1u) {
            let private_combo = combos[combo];
            if private_combo.cards[0] == card || private_combo.cards[1] == card {
                let offset = node_slot * params.combo_count + combo;
                let live_word = offset >> 5u;
                let live_mask = 1u << (offset & 31u);
                if (combo_live[live_word] & live_mask) != 0u {
                    if params.value_player == 2u {
                        hero_sum = hero_sum + max(root_reach_weights[combo], 0.0);
                        villain_sum = villain_sum + max(root_reach_weights[params.combo_count + combo], 0.0);
                    } else {
                        hero_sum = hero_sum + hero_reaches[offset];
                        villain_sum = villain_sum + villain_reaches[offset];
                    }
                }
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
    if node.public_infoset < params.public_infoset_count || node.public_infoset >= params._pad0 {
        return;
    }
    let live_word = index >> 5u;
    let live_mask = 1u << (index & 31u);
    if (combo_live[live_word] & live_mask) == 0u {
        let private_infoset = node.public_infoset * params.combo_count + combo;
        reach_weights[private_infoset - params.public_infoset_count * params.combo_count] = 0.0;
        return;
    }
    let private_combo = combos[combo];
    let aggregate_base = node.public_infoset * 53u;
    var own_root_weight = root_reach_weights[combo];
    var value_weight = 0.0;
    if node.acting_player == 0u {
        let self_reach = villain_reaches[index];
        value_weight = villain_aggregates[aggregate_base]
            - villain_aggregates[aggregate_base + private_combo.cards[0] + 1u]
            - villain_aggregates[aggregate_base + private_combo.cards[1] + 1u]
            + self_reach;
    } else {
        own_root_weight = root_reach_weights[params.combo_count + combo];
        let self_reach = hero_reaches[index];
        value_weight = hero_aggregates[aggregate_base]
            - hero_aggregates[aggregate_base + private_combo.cards[0] + 1u]
            - hero_aggregates[aggregate_base + private_combo.cards[1] + 1u]
            + self_reach;
    }
    let private_infoset = node.public_infoset * params.combo_count + combo;
    if own_root_weight <= 0.0 || value_weight <= 0.0 {
        reach_weights[private_infoset - params.public_infoset_count * params.combo_count] = 0.0;
        return;
    }
    reach_weights[private_infoset - params.public_infoset_count * params.combo_count] = node._pad1 * value_weight;
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
    let parent_combo_index = edge.parent * params.combo_count + combo;
    let live_word = parent_combo_index >> 5u;
    let live_mask = 1u << (parent_combo_index & 31u);
    if (combo_live[live_word] & live_mask) == 0u {
        return;
    }
    let private_infoset = node.public_infoset * params.combo_count + combo;
    let local_private_infoset = private_infoset - params.public_infoset_count * params.combo_count;
    let action_index = local_private_infoset * params.max_actions + edge.action;
    let action_len = params.output_len;
    let child_offset = edge.child * params.combo_count + combo;
    let action_value = select(
        child_villain_values[child_offset],
        child_hero_values[child_offset],
        params.value_player == 0u
    );
    action_values[action_index] = action_value;
    if edge.action == 0u {
        let private_combo = combos[combo];
        let aggregate_base = node.public_infoset * 53u;
        var own_reach = hero_reaches[parent_combo_index];
        var opponent_weight = villain_aggregates[aggregate_base]
            - villain_aggregates[aggregate_base + private_combo.cards[0] + 1u]
            - villain_aggregates[aggregate_base + private_combo.cards[1] + 1u]
            + max(root_reach_weights[params.combo_count + combo], 0.0);
        if node.acting_player == 1u {
            own_reach = villain_reaches[parent_combo_index];
            opponent_weight = hero_aggregates[aggregate_base]
                - hero_aggregates[aggregate_base + private_combo.cards[0] + 1u]
                - hero_aggregates[aggregate_base + private_combo.cards[1] + 1u]
                + max(root_reach_weights[combo], 0.0);
        }
        if own_reach <= 0.0 || opponent_weight <= 0.0 {
            strategy_weights[local_private_infoset] = 0.0;
        } else {
            strategy_weights[local_private_infoset] = node._pad1 * own_reach * opponent_weight;
        }
    }
}
"#;
