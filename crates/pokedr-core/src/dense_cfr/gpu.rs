use std::{collections::BTreeMap, sync::mpsc, time::Instant};

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

fn positive(value: f32) -> f32 {
    return max(value, 0.0);
}

fn strategy_at(offset: u32, action: u32, actions: u32, normalizer: f32) -> f32 {
    if legal_actions[offset + action] == 0u {
        return 0.0;
    }
    if normalizer > 0.0 {
        return positive(regrets[offset + action]) / normalizer;
    }
    return 1.0 / f32(actions);
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let infoset = id.x;
    let infosets = params[0];
    let actions = params[1];
    let variant = params[2];
    let iteration = params[3];
    if infoset >= infosets {
        return;
    }

    let offset = infoset * actions;
    var normalizer = 0.0;
    var legal_count = 0u;
    for (var action = 0u; action < actions; action = action + 1u) {
        if legal_actions[offset + action] != 0u {
            legal_count = legal_count + 1u;
            normalizer = normalizer + positive(regrets[offset + action]);
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

    var discount = 1.0;
    if variant == 1u {
        let t = f32(max(iteration, 1u));
        discount = t / (t + 1.0);
    }

    for (var action = 0u; action < actions; action = action + 1u) {
        if legal_actions[offset + action] == 0u {
            regrets[offset + action] = 0.0;
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
        if variant == 0u {
            updated = max(updated, 0.0);
        }
        regrets[offset + action] = updated;
        strategy_sum[offset + action] = strategy_sum[offset + action] + strategy_weights[infoset] * strategy;
    }
}
"#;

const PUBLIC_TREE_CFR_UPDATE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> regrets: array<f32>;
@group(0) @binding(1) var<storage, read_write> strategy_sum: array<f32>;
@group(0) @binding(2) var<storage, read> output: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@group(0) @binding(4) var<storage, read> legal_actions: array<u32>;

fn positive(value: f32) -> f32 {
    return max(value, 0.0);
}

fn strategy_at(offset: u32, action: u32, actions: u32, normalizer: f32) -> f32 {
    if legal_actions[offset + action] == 0u {
        return 0.0;
    }
    if normalizer > 0.0 {
        return positive(regrets[offset + action]) / normalizer;
    }
    return 1.0 / f32(actions);
}

fn action_value(action_offset: u32, action_len: u32) -> f32 {
    let weight = output[action_len + action_offset];
    if weight > 0.0 {
        return output[action_offset] / weight;
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let infoset = id.x;
    let infosets = params[0];
    let actions = params[1];
    let variant = params[2];
    let iteration = params[3];
    let action_len = params[4];
    let reach_start = params[5];
    let strategy_start = params[6];
    if infoset >= infosets {
        return;
    }

    let offset = infoset * actions;
    var normalizer = 0.0;
    var legal_count = 0u;
    for (var action = 0u; action < actions; action = action + 1u) {
        if legal_actions[offset + action] != 0u {
            legal_count = legal_count + 1u;
            normalizer = normalizer + positive(regrets[offset + action]);
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
        node_value = node_value + strategy * action_value(offset + action, action_len);
    }

    var discount = 1.0;
    if variant == 1u {
        let t = f32(max(iteration, 1u));
        discount = t / (t + 1.0);
    }

    let reach_weight = output[reach_start + infoset];
    let strategy_weight = output[strategy_start + infoset] * f32(iteration);
    for (var action = 0u; action < actions; action = action + 1u) {
        if legal_actions[offset + action] == 0u {
            regrets[offset + action] = 0.0;
            strategy_sum[offset + action] = 0.0;
            continue;
        }
        let strategy = select(
            1.0 / f32(max(legal_count, 1u)),
            strategy_at(offset, action, actions, normalizer),
            normalizer > 0.0
        );
        let regret = (action_value(offset + action, action_len) - node_value) * reach_weight;
        var updated = regrets[offset + action] * discount + regret;
        if variant == 0u {
            updated = max(updated, 0.0);
        }
        regrets[offset + action] = updated;
        strategy_sum[offset + action] = strategy_sum[offset + action] + strategy_weight * strategy;
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

const PUBLIC_TREE_REACH_SHADER: &str = r#"
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
    target_node: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> nodes: array<TreeNode>;
@group(0) @binding(1) var<storage, read> edges: array<Edge>;
@group(0) @binding(2) var<storage, read> combos: array<Combo>;
@group(0) @binding(3) var<storage, read> root_weights: array<f32>;
@group(0) @binding(4) var<storage, read> regrets: array<f32>;
@group(0) @binding(5) var<storage, read_write> hero_reaches: array<f32>;
@group(0) @binding(6) var<storage, read_write> villain_reaches: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

fn combo_has_card(combo: Combo, card: u32) -> bool {
    return combo.cards[0] == card || combo.cards[1] == card;
}

fn strategy_probability(node: TreeNode, private_combo: u32, action: u32) -> f32 {
    let private_infoset = node.public_infoset * params.combo_count * 2u
        + node.acting_player * params.combo_count
        + private_combo;
    let offset = private_infoset * params.max_actions;
    var normalizer = 0.0;
    for (var i = 0u; i < node.child_count; i = i + 1u) {
        normalizer = normalizer + max(regrets[offset + i], 0.0);
    }
    if normalizer > 0.0 {
        return max(regrets[offset + action], 0.0) / normalizer;
    }
    return 1.0 / f32(max(node.child_count, 1u));
}

@compute @workgroup_size(64)
fn reach_init(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * params.output_len;
    let value_count = params.node_count * params.combo_count;
    if index >= value_count {
        return;
    }
    let combo = index % params.combo_count;
    hero_reaches[index] = 0.0;
    villain_reaches[index] = 0.0;
    if index >= params.combo_count || root_weights[combo] < 0.0 {
        return;
    }
    hero_reaches[combo] = 1.0;
    villain_reaches[combo] = root_weights[combo];
}

@compute @workgroup_size(64)
fn reach_edges(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * params.output_len;
    let layer_edge_count = params._pad2;
    let value_count = layer_edge_count * params.combo_count;
    if index >= value_count {
        return;
    }
    let combo = index % params.combo_count;
    let edge_slot = index / params.combo_count;
    let edge = edges[params.target_node + edge_slot];
    let node = nodes[edge.parent];
    let node_offset = edge.parent * params.combo_count + combo;
    let child_offset = edge.child * params.combo_count + combo;
    let hero_reach = hero_reaches[node_offset];
    let villain_reach = villain_reaches[node_offset];
    if node.kind == 0u {
        let probability = strategy_probability(node, combo, edge.action);
        if node.acting_player == 0u {
            hero_reaches[child_offset] = hero_reach * probability;
            villain_reaches[child_offset] = villain_reach;
        } else {
            hero_reaches[child_offset] = hero_reach;
            villain_reaches[child_offset] = villain_reach * probability;
        }
    } else if node.kind == 1u {
        if combo_has_card(combos[combo], edge.card) {
            hero_reaches[child_offset] = 0.0;
            villain_reaches[child_offset] = 0.0;
        } else {
            hero_reaches[child_offset] = hero_reach;
            villain_reaches[child_offset] = villain_reach;
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
    board_base: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> nodes: array<TreeNode>;
@group(0) @binding(1) var<storage, read> terminal_nodes: array<u32>;
@group(0) @binding(2) var<storage, read> combo_order: array<u32>;
@group(0) @binding(3) var<storage, read> combo_bounds: array<Bounds>;
@group(0) @binding(4) var<storage, read> hero_reaches: array<f32>;
@group(0) @binding(5) var<storage, read> villain_reaches: array<f32>;
@group(0) @binding(6) var<storage, read_write> prefix_pairs: array<PrefixPair>;
@group(0) @binding(7) var<uniform> params: Params;

@compute @workgroup_size(64)
fn terminal_partial(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * params.x_invocations;
    let output_count = params.terminal_count * params.board_count;
    if index >= output_count {
        return;
    }
    let board = index % params.board_count;
    let terminal_slot = index / params.board_count;
    let node_index = terminal_nodes[terminal_slot];
    let node_offset = node_index * params.combo_count;
    let order_base = board * params.order_stride;
    let prefix_base = (terminal_slot * params.board_count + board) * params.prefix_stride;
    var hero_sum = 0.0;
    var villain_sum = 0.0;
    prefix_pairs[prefix_base] = PrefixPair(0.0, 0.0);
    for (var position = 0u; position < params.combo_count; position = position + 1u) {
        let combo = combo_order[order_base + position];
        if combo != 0xffffffffu {
            let bounds = combo_bounds[board * params.combo_count + combo];
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
    board_base: u32,
    _pad3: u32,
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

@compute @workgroup_size(64)
fn terminal_reduce(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * params.x_invocations;
    let output_count = params.terminal_count * params.combo_count;
    if index >= output_count {
        return;
    }
    let combo = index % params.combo_count;
    let terminal_slot = index / params.combo_count;
    let node_index = terminal_nodes[terminal_slot];
    let node = nodes[node_index];
    let node_offset = node_index * params.combo_count;
    let denom = max(node.showdown_denominator, 1.0);
    var hero_value = 0.0;
    var villain_value = 0.0;
    for (var board = 0u; board < node.board_count; board = board + 1u) {
        let board_index = node.showdown_offset + board;
        let local_board = board_index - params.board_base;
        let bounds = combo_bounds[local_board * params.combo_count + combo];
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
            let opponent_bounds = combo_bounds[local_board * params.combo_count + opponent];
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
@group(0) @binding(9) var<uniform> params: Params;

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
    hero_values[value_index] = villain_noncolliding * hero_payoff * node._pad1;
    villain_values[value_index] = hero_noncolliding * (-hero_payoff) * node._pad1;
}
"#;

const PUBLIC_TREE_BACKUP_SHADER: &str = r#"
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
    node_count: u32,
    max_actions: u32,
    output_len: u32,
    value_player: u32,
    target_node: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> nodes: array<TreeNode>;
@group(0) @binding(1) var<storage, read> children: array<u32>;
@group(0) @binding(2) var<storage, read> backup_nodes: array<u32>;
@group(0) @binding(3) var<storage, read> hero_reaches: array<f32>;
@group(0) @binding(4) var<storage, read> villain_reaches: array<f32>;
@group(0) @binding(5) var<storage, read_write> hero_values: array<f32>;
@group(0) @binding(6) var<storage, read_write> villain_values: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

fn strategy_from_reach(node_index: u32, child: u32, combo: u32, acting_player: u32) -> f32 {
    let parent_offset = node_index * params.combo_count + combo;
    let child_offset = child * params.combo_count + combo;
    if acting_player == 0u {
        let parent = hero_reaches[parent_offset];
        if parent > 0.0 {
            return hero_reaches[child_offset] / parent;
        }
    } else {
        let parent = villain_reaches[parent_offset];
        if parent > 0.0 {
            return villain_reaches[child_offset] / parent;
        }
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn backup_layer(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * params.output_len;
    let layer_node_count = params._pad2;
    let value_count = layer_node_count * params.combo_count;
    if index >= value_count {
        return;
    }
    let combo = index % params.combo_count;
    let node_slot = index / params.combo_count;
    let node_index = backup_nodes[params.target_node + node_slot];
    let node = nodes[node_index];
    var hero_value = 0.0;
    var villain_value = 0.0;
    if node.kind == 0u {
        for (var action = 0u; action < node.child_count; action = action + 1u) {
            let child = children[node.first_child + action];
            let child_offset = child * params.combo_count + combo;
            let hero_child_value = hero_values[child_offset];
            let villain_child_value = villain_values[child_offset];
            if node.acting_player == 0u {
                let probability = strategy_from_reach(node_index, child, combo, 0u);
                hero_value = hero_value + probability * hero_child_value;
                villain_value = villain_value + villain_child_value;
            } else {
                let probability = strategy_from_reach(node_index, child, combo, 1u);
                hero_value = hero_value + hero_child_value;
                villain_value = villain_value + probability * villain_child_value;
            }
        }
    } else if node.kind == 1u {
        for (var action = 0u; action < node.child_count; action = action + 1u) {
            let child = children[node.first_child + action];
            let child_offset = child * params.combo_count + combo;
            hero_value = hero_value + hero_values[child_offset];
            villain_value = villain_value + villain_values[child_offset];
        }
    }
    let value_index = node_index * params.combo_count + combo;
    hero_values[value_index] = hero_value;
    villain_values[value_index] = villain_value;
}
"#;

const PUBLIC_TREE_AGGREGATE_SHADER: &str = r#"
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
    node_count: u32,
    max_actions: u32,
    output_len: u32,
    value_player: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> nodes: array<TreeNode>;
@group(0) @binding(1) var<storage, read> children: array<u32>;
@group(0) @binding(2) var<storage, read> combos: array<Combo>;
@group(0) @binding(3) var<storage, read> decision_nodes: array<u32>;
@group(0) @binding(4) var<storage, read> hero_reaches: array<f32>;
@group(0) @binding(5) var<storage, read> villain_reaches: array<f32>;
@group(0) @binding(6) var<storage, read> values: array<f32>;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

fn collide(left: Combo, right: Combo) -> bool {
    return left.cards[0] == right.cards[0] || left.cards[0] == right.cards[1]
        || left.cards[1] == right.cards[0] || left.cards[1] == right.cards[1];
}

@compute @workgroup_size(64)
fn tree_aggregate(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= params.output_len {
        return;
    }
    let action_len = params.output_len;
    let private_infoset = index / params.max_actions;
    let action = index % params.max_actions;
    let public_infoset = private_infoset / (params.combo_count * 2u);
    let player_slot = (private_infoset / params.combo_count) % 2u;
    let acting_combo = private_infoset % params.combo_count;
    if player_slot != params.value_player {
        return;
    }

    let node_index = decision_nodes[public_infoset];
    if node_index == 0xffffffffu {
        output[index] = 0.0;
        output[action_len + index] = 0.0;
        return;
    }
    let node = nodes[node_index];
    if action >= node.child_count || player_slot != node.acting_player {
        output[index] = 0.0;
        output[action_len + index] = 0.0;
        return;
    }

    let child = children[node.first_child + action];
    let node_base = node_index * params.combo_count;
    let value_weight = output[action_len * 2u + private_infoset];
    let value_offset = child * params.combo_count + acting_combo;
    let action_value = values[value_offset];
    var strategy_weight = node._pad1 * hero_reaches[node_base + acting_combo];
    if node.acting_player == 1u {
        strategy_weight = node._pad1 * villain_reaches[node_base + acting_combo];
    }
    output[index] = action_value;
    output[action_len + index] = value_weight;
    if action == 0u {
        let infoset_count = action_len / params.max_actions;
        output[action_len * 2u + private_infoset] = value_weight;
        output[action_len * 2u + infoset_count + private_infoset] = strategy_weight;
    }
}
"#;

const PUBLIC_TREE_DECISION_AGGREGATE_SHADER: &str = r#"
struct Combo { cards: array<u32, 2>, };
struct Params {
    combo_count: u32,
    infoset_count: u32,
    slots_per_infoset: u32,
    output_len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> decision_nodes: array<u32>;
@group(0) @binding(1) var<storage, read> combos: array<Combo>;
@group(0) @binding(2) var<storage, read> hero_reaches: array<f32>;
@group(0) @binding(3) var<storage, read> villain_reaches: array<f32>;
@group(0) @binding(4) var<storage, read_write> hero_aggregates: array<f32>;
@group(0) @binding(5) var<storage, read_write> villain_aggregates: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn decision_aggregate(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * params.output_len;
    let output_count = params.infoset_count * params.slots_per_infoset;
    if index >= output_count {
        return;
    }
    let slot = index % params.slots_per_infoset;
    let public_infoset = index / params.slots_per_infoset;
    let node_index = decision_nodes[public_infoset];
    if node_index == 0xffffffffu {
        hero_aggregates[index] = 0.0;
        villain_aggregates[index] = 0.0;
        return;
    }
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

const PUBLIC_TREE_DENOMINATOR_SHADER: &str = r#"
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
    node_count: u32,
    max_actions: u32,
    output_len: u32,
    value_player: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> nodes: array<TreeNode>;
@group(0) @binding(1) var<storage, read> combos: array<Combo>;
@group(0) @binding(2) var<storage, read> decision_nodes: array<u32>;
@group(0) @binding(3) var<storage, read> hero_reaches: array<f32>;
@group(0) @binding(4) var<storage, read> villain_reaches: array<f32>;
@group(0) @binding(5) var<storage, read> hero_aggregates: array<f32>;
@group(0) @binding(6) var<storage, read> villain_aggregates: array<f32>;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

@compute @workgroup_size(64)
fn denominator_mass(@builtin(global_invocation_id) id: vec3<u32>) {
    let private_infoset = id.x;
    if private_infoset >= params.output_len {
        return;
    }
    let public_infoset = private_infoset / (params.combo_count * 2u);
    let player_slot = (private_infoset / params.combo_count) % 2u;
    let acting_combo = private_infoset % params.combo_count;
    if player_slot != params.value_player {
        return;
    }

    let node_index = decision_nodes[public_infoset];
    if node_index == 0xffffffffu {
        output[params.output_len * params.max_actions * 2u + private_infoset] = 0.0;
        return;
    }
    let node = nodes[node_index];
    if node.kind != 0u || node.acting_player != player_slot {
        output[params.output_len * params.max_actions * 2u + private_infoset] = 0.0;
        return;
    }

    let node_base = node_index * params.combo_count;
    let private_combo = combos[acting_combo];
    let aggregate_base = public_infoset * 53u;
    var value_weight = 0.0;
    if node.acting_player == 0u {
        let self_reach = villain_reaches[node_base + acting_combo];
        value_weight = villain_aggregates[aggregate_base]
            - villain_aggregates[aggregate_base + private_combo.cards[0] + 1u]
            - villain_aggregates[aggregate_base + private_combo.cards[1] + 1u]
            + self_reach;
    } else {
        let self_reach = hero_reaches[node_base + acting_combo];
        value_weight = hero_aggregates[aggregate_base]
            - hero_aggregates[aggregate_base + private_combo.cards[0] + 1u]
            - hero_aggregates[aggregate_base + private_combo.cards[1] + 1u]
            + self_reach;
    }
    output[params.output_len * params.max_actions * 2u + private_infoset] = node._pad1 * value_weight;
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
    final_board_strength_pipeline: wgpu::ComputePipeline,
    showdown_matrix_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_reach_init_pipeline: wgpu::ComputePipeline,
    public_tree_reach_edge_pipeline: wgpu::ComputePipeline,
    public_tree_reach_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_terminal_partial_pipeline: wgpu::ComputePipeline,
    public_tree_terminal_partial_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_terminal_reduce_pipeline: wgpu::ComputePipeline,
    public_tree_terminal_reduce_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_fold_aggregate_pipeline: wgpu::ComputePipeline,
    public_tree_fold_aggregate_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_fold_value_pipeline: wgpu::ComputePipeline,
    public_tree_fold_value_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_backup_pipeline: wgpu::ComputePipeline,
    public_tree_backup_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_decision_aggregate_pipeline: wgpu::ComputePipeline,
    public_tree_decision_aggregate_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_denominator_pipeline: wgpu::ComputePipeline,
    public_tree_denominator_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_aggregate_pipeline: wgpu::ComputePipeline,
    public_tree_aggregate_bind_group_layout: wgpu::BindGroupLayout,
}

pub struct GpuDenseCfrState {
    infosets: usize,
    actions: usize,
    variant: super::CfrVariant,
    legal_actions: Vec<u32>,
    legal_actions_buffer: wgpu::Buffer,
    regrets: wgpu::Buffer,
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
    infosets: usize,
    actions: usize,
    action_len: usize,
    output_len: usize,
    node_combo_len: usize,
    public_infoset_count: usize,
    node_buffer: wgpu::Buffer,
    child_buffer: wgpu::Buffer,
    reach_edge_buffer: wgpu::Buffer,
    reach_layer_ranges: Vec<(usize, usize)>,
    backup_nodes_buffer: wgpu::Buffer,
    backup_layer_ranges: Vec<(usize, usize)>,
    combo_buffer: wgpu::Buffer,
    root_weights_buffer: wgpu::Buffer,
    hero_reaches_buffer: wgpu::Buffer,
    villain_reaches_buffer: wgpu::Buffer,
    hero_values_buffer: wgpu::Buffer,
    villain_values_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    fold_terminal_nodes: Vec<u32>,
    showdown_terminal_nodes: Vec<u32>,
    showdown_terminal_groups: Vec<GpuTerminalGroupCache>,
    decision_nodes_buffer: wgpu::Buffer,
    terminal_tile_count: usize,
    terminal_board_tile_count: usize,
    terminal_chunk_size: usize,
    terminal_blocker_neighbors_buffer: wgpu::Buffer,
    terminal_blocker_neighbor_stride: usize,
    terminal_prefix_pair_budget: usize,
    terminal_prefix_pairs_buffer: wgpu::Buffer,
    hero_decision_aggregates_buffer: wgpu::Buffer,
    villain_decision_aggregates_buffer: wgpu::Buffer,
}

struct GpuTerminalGroupCache {
    board_base: usize,
    board_count: usize,
    terminal_nodes: Vec<u32>,
    combo_order: Vec<u32>,
    combo_bounds: Vec<GpuShowdownComboBounds>,
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
    let mut groups: BTreeMap<(usize, usize), Vec<u32>> = BTreeMap::new();
    for &node_index in terminal_nodes {
        let node = nodes[node_index as usize];
        groups
            .entry((node.showdown_offset as usize, node._pad0 as usize))
            .or_default()
            .push(node_index);
    }
    groups
        .into_iter()
        .filter_map(|((board_base, board_count), terminal_nodes)| {
            if board_count == 0 {
                return None;
            }
            let boards = &showdown_boards[board_base..board_base + board_count];
            let (combo_order, combo_bounds) = showdown_strength_order_data(combos, boards);
            Some(GpuTerminalGroupCache {
                board_base,
                board_count,
                terminal_nodes,
                combo_order,
                combo_bounds,
            })
        })
        .collect()
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
}

impl GpuDenseCfrBackend {
    pub fn new() -> Result<Self, GpuCfrError> {
        pollster::block_on(Self::new_async())
    }

    pub async fn new_async() -> Result<Self, GpuCfrError> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        descriptor.backends = wgpu::Backends::VULKAN;
        descriptor
            .flags
            .insert(wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER);
        let instance = wgpu::Instance::new(descriptor);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|_| GpuCfrError::NoAdapter)?;
        let adapter_info = adapter.get_info();
        let adapter_features = adapter.features();
        let required_limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("pokedr dense CFR device"),
                required_features: wgpu::Features::empty(),
                required_limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|error| GpuCfrError::RequestDevice(error.to_string()))?;
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
        let final_board_strength_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("final board strength pipeline"),
                layout: Some(&showdown_matrix_pipeline_layout),
                module: &showdown_matrix_shader,
                entry_point: Some("strengths"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let public_tree_reach_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("public tree reach shader"),
            source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_REACH_SHADER.into()),
        });
        let public_tree_reach_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("public tree reach bind group layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, true),
                    storage_entry(5, false),
                    storage_entry(6, false),
                    uniform_entry(7),
                ],
            });
        let public_tree_reach_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("public tree reach pipeline layout"),
                bind_group_layouts: &[Some(&public_tree_reach_bind_group_layout)],
                immediate_size: 0,
            });
        let public_tree_reach_init_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree reach init pipeline"),
                layout: Some(&public_tree_reach_pipeline_layout),
                module: &public_tree_reach_shader,
                entry_point: Some("reach_init"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let public_tree_reach_edge_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree reach edge pipeline"),
                layout: Some(&public_tree_reach_pipeline_layout),
                module: &public_tree_reach_shader,
                entry_point: Some("reach_edges"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let public_tree_terminal_partial_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree terminal partial shader"),
                source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_TERMINAL_PARTIAL_SHADER.into()),
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
                immediate_size: 0,
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
                immediate_size: 0,
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
                    uniform_entry(9),
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
        let public_tree_backup_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("public tree backup shader"),
            source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_BACKUP_SHADER.into()),
        });
        let public_tree_backup_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("public tree backup bind group layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, true),
                    storage_entry(5, false),
                    storage_entry(6, false),
                    uniform_entry(7),
                ],
            });
        let public_tree_backup_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("public tree backup pipeline layout"),
                bind_group_layouts: &[Some(&public_tree_backup_bind_group_layout)],
                immediate_size: 0,
            });
        let public_tree_backup_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree backup pipeline"),
                layout: Some(&public_tree_backup_pipeline_layout),
                module: &public_tree_backup_shader,
                entry_point: Some("backup_layer"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let public_tree_denominator_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree denominator shader"),
                source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_DENOMINATOR_SHADER.into()),
            });
        let public_tree_decision_aggregate_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree decision aggregate shader"),
                source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_DECISION_AGGREGATE_SHADER.into()),
            });
        let public_tree_decision_aggregate_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("public tree decision aggregate bind group layout"),
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
        let public_tree_decision_aggregate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("public tree decision aggregate pipeline layout"),
                bind_group_layouts: &[Some(&public_tree_decision_aggregate_bind_group_layout)],
                immediate_size: 0,
            });
        let public_tree_decision_aggregate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree decision aggregate pipeline"),
                layout: Some(&public_tree_decision_aggregate_pipeline_layout),
                module: &public_tree_decision_aggregate_shader,
                entry_point: Some("decision_aggregate"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let public_tree_denominator_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("public tree denominator bind group layout"),
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
        let public_tree_denominator_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("public tree denominator pipeline layout"),
                bind_group_layouts: &[Some(&public_tree_denominator_bind_group_layout)],
                immediate_size: 0,
            });
        let public_tree_denominator_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree denominator pipeline"),
                layout: Some(&public_tree_denominator_pipeline_layout),
                module: &public_tree_denominator_shader,
                entry_point: Some("denominator_mass"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let public_tree_aggregate_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree aggregate shader"),
                source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_AGGREGATE_SHADER.into()),
            });
        let public_tree_aggregate_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("public tree aggregate bind group layout"),
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
        let public_tree_aggregate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("public tree aggregate pipeline layout"),
                bind_group_layouts: &[Some(&public_tree_aggregate_bind_group_layout)],
                immediate_size: 0,
            });
        let public_tree_aggregate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree aggregate pipeline"),
                layout: Some(&public_tree_aggregate_pipeline_layout),
                module: &public_tree_aggregate_shader,
                entry_point: Some("tree_aggregate"),
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
            final_board_strength_pipeline,
            showdown_matrix_bind_group_layout,
            public_tree_reach_init_pipeline,
            public_tree_reach_edge_pipeline,
            public_tree_reach_bind_group_layout,
            public_tree_terminal_partial_pipeline,
            public_tree_terminal_partial_bind_group_layout,
            public_tree_terminal_reduce_pipeline,
            public_tree_terminal_reduce_bind_group_layout,
            public_tree_fold_aggregate_pipeline,
            public_tree_fold_aggregate_bind_group_layout,
            public_tree_fold_value_pipeline,
            public_tree_fold_value_bind_group_layout,
            public_tree_backup_pipeline,
            public_tree_backup_bind_group_layout,
            public_tree_decision_aggregate_pipeline,
            public_tree_decision_aggregate_bind_group_layout,
            public_tree_denominator_pipeline,
            public_tree_denominator_bind_group_layout,
            public_tree_aggregate_pipeline,
            public_tree_aggregate_bind_group_layout,
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
        ];
        let regrets = storage_buffer(&self.device, "regrets", &state.regrets);
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
        let strategy_readback = readback_buffer(&self.device, state.strategy_sum.len());
        copy_buffer(
            &mut encoder,
            &regrets,
            &regret_readback,
            state.regrets.len(),
        );
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

    fn final_board_strength_buffer(
        &self,
        combos: &[GpuPrivateCombo],
        final_boards: &[GpuFinalBoard],
    ) -> Result<wgpu::Buffer, GpuCfrError> {
        if combos.is_empty() || final_boards.is_empty() {
            return Ok(storage_buffer(
                &self.device,
                "final board strengths",
                &[0.0f32],
            ));
        }
        let output_count = combos.len() * final_boards.len();
        let combo_buffer = readonly_buffer(&self.device, "strength combos", combos);
        let board_buffer = readonly_buffer(&self.device, "strength boards", final_boards);
        let output_buffer =
            uninit_storage_buffer(&self.device, "final board strengths", output_count, false);
        let params = readonly_buffer(
            &self.device,
            "strength params",
            &[
                combos.len() as u32,
                final_boards.len() as u32,
                0,
                output_count as u32,
            ],
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("final board strength bind group"),
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
                label: Some("final board strength encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("final board strength pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.final_board_strength_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (output_count as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups.min(65_535), groups.div_ceil(65_535), 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(output_buffer)
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
                bind_entry(9, &value_params),
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

    #[allow(clippy::too_many_arguments)]
    fn fill_terminal_values_streaming(
        &self,
        node_buffer: &wgpu::Buffer,
        terminal_groups: &[GpuTerminalGroupCache],
        blocker_neighbors_buffer: &wgpu::Buffer,
        hero_reaches_buffer: &wgpu::Buffer,
        villain_reaches_buffer: &wgpu::Buffer,
        hero_values_buffer: &wgpu::Buffer,
        villain_values_buffer: &wgpu::Buffer,
        terminal_prefix_pairs_buffer: &wgpu::Buffer,
        combo_count: usize,
        blocker_neighbor_stride: usize,
        max_terminal_prefix_pairs: usize,
    ) -> Result<(), GpuCfrError> {
        if terminal_groups.is_empty() || combo_count == 0 {
            return Ok(());
        }
        let poll_interval = std::env::var("POKEDR_GPU_TERMINAL_STREAM_POLL")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(16)
            .max(1);
        let mut submitted_chunks = 0usize;
        for group in terminal_groups {
            let combo_order_buffer = readonly_buffer(
                &self.device,
                "public tree streamed terminal combo strength order",
                &group.combo_order,
            );
            let combo_bounds_buffer = readonly_buffer(
                &self.device,
                "public tree streamed terminal combo strength bounds",
                &group.combo_bounds,
            );
            let prefix_pairs_per_terminal = group.board_count * (combo_count + 1);
            let terminal_chunk_size = (max_terminal_prefix_pairs / prefix_pairs_per_terminal)
                .max(1)
                .min(group.terminal_nodes.len().max(1));
            for terminal_chunk in group.terminal_nodes.chunks(terminal_chunk_size) {
                let terminal_count = terminal_chunk.len();
                let terminal_nodes_buffer = readonly_buffer(
                    &self.device,
                    "public tree streamed terminal nodes",
                    terminal_chunk,
                );
                let partial_invocations = terminal_count * group.board_count;
                let (partial_x_groups, partial_y_groups, partial_x_invocations) =
                    dispatch_grid(partial_invocations);
                let partial_params = uniform_buffer(
                    &self.device,
                    "public tree streamed terminal partial params",
                    &[GpuPublicTreeParams {
                        combo_count: combo_count as u32,
                        node_count: terminal_count as u32,
                        max_actions: group.board_count as u32,
                        output_len: (combo_count + 1) as u32,
                        pair_start: combo_count as u32,
                        chunk_pairs: partial_x_invocations,
                        _pad0: group.board_base as u32,
                        _pad1: 0,
                    }],
                );
                let partial_bind_group =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("public tree streamed terminal partial bind group"),
                        layout: &self.public_tree_terminal_partial_bind_group_layout,
                        entries: &[
                            bind_entry(0, node_buffer),
                            bind_entry(1, &terminal_nodes_buffer),
                            bind_entry(2, &combo_order_buffer),
                            bind_entry(3, &combo_bounds_buffer),
                            bind_entry(4, hero_reaches_buffer),
                            bind_entry(5, villain_reaches_buffer),
                            bind_entry(6, terminal_prefix_pairs_buffer),
                            bind_entry(7, &partial_params),
                        ],
                    });

                let reduce_invocations = terminal_count * combo_count;
                let (reduce_x_groups, reduce_y_groups, reduce_x_invocations) =
                    dispatch_grid(reduce_invocations);
                let reduce_params = uniform_buffer(
                    &self.device,
                    "public tree streamed terminal reduce params",
                    &[GpuPublicTreeParams {
                        combo_count: combo_count as u32,
                        node_count: terminal_count as u32,
                        max_actions: group.board_count as u32,
                        output_len: (combo_count + 1) as u32,
                        pair_start: blocker_neighbor_stride as u32,
                        chunk_pairs: reduce_x_invocations,
                        _pad0: group.board_base as u32,
                        _pad1: 0,
                    }],
                );
                let reduce_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("public tree streamed terminal reduce bind group"),
                    layout: &self.public_tree_terminal_reduce_bind_group_layout,
                    entries: &[
                        bind_entry(0, node_buffer),
                        bind_entry(1, &terminal_nodes_buffer),
                        bind_entry(2, &combo_bounds_buffer),
                        bind_entry(3, blocker_neighbors_buffer),
                        bind_entry(4, hero_reaches_buffer),
                        bind_entry(5, villain_reaches_buffer),
                        bind_entry(6, terminal_prefix_pairs_buffer),
                        bind_entry(7, hero_values_buffer),
                        bind_entry(8, villain_values_buffer),
                        bind_entry(9, &reduce_params),
                    ],
                });
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("public tree streamed terminal encoder"),
                        });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("public tree streamed terminal partial pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.public_tree_terminal_partial_pipeline);
                    pass.set_bind_group(0, &partial_bind_group, &[]);
                    pass.dispatch_workgroups(partial_x_groups, partial_y_groups, 1);
                }

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("public tree streamed terminal reduce pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.public_tree_terminal_reduce_pipeline);
                    pass.set_bind_group(0, &reduce_bind_group, &[]);
                    pass.dispatch_workgroups(reduce_x_groups, reduce_y_groups, 1);
                }
                self.queue.submit(Some(encoder.finish()));
                submitted_chunks += 1;
                if submitted_chunks % poll_interval == 0 {
                    self.profile_poll()?;
                }
            }
        }
        self.profile_poll()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn backup_nonterminal_values(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        node_buffer: &wgpu::Buffer,
        child_buffer: &wgpu::Buffer,
        backup_nodes_buffer: &wgpu::Buffer,
        hero_reaches_buffer: &wgpu::Buffer,
        villain_reaches_buffer: &wgpu::Buffer,
        hero_values_buffer: &wgpu::Buffer,
        villain_values_buffer: &wgpu::Buffer,
        backup_layer_ranges: &[(usize, usize)],
        combo_count: usize,
        node_count: usize,
        max_actions: usize,
    ) -> Result<(), GpuCfrError> {
        for &(layer_start, layer_end) in backup_layer_ranges.iter().rev() {
            let layer_node_count = layer_end - layer_start;
            if layer_node_count == 0 {
                continue;
            }
            let invocation_count = layer_node_count * combo_count;
            let (x_groups, y_groups, x_invocations) = dispatch_grid(invocation_count);
            let params = uniform_buffer(
                &self.device,
                "public tree backup params",
                &[GpuPublicTreeParams {
                    combo_count: combo_count as u32,
                    node_count: node_count as u32,
                    max_actions: max_actions as u32,
                    output_len: x_invocations,
                    pair_start: 0,
                    chunk_pairs: layer_start as u32,
                    _pad0: layer_node_count as u32,
                    _pad1: 0,
                }],
            );
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("public tree backup bind group"),
                layout: &self.public_tree_backup_bind_group_layout,
                entries: &[
                    bind_entry(0, node_buffer),
                    bind_entry(1, child_buffer),
                    bind_entry(2, backup_nodes_buffer),
                    bind_entry(3, hero_reaches_buffer),
                    bind_entry(4, villain_reaches_buffer),
                    bind_entry(5, hero_values_buffer),
                    bind_entry(6, villain_values_buffer),
                    bind_entry(7, &params),
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("public tree backup pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.public_tree_backup_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(x_groups, y_groups, 1);
            }
        }
        Ok(())
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
        let (reach_edges, reach_layer_ranges) =
            public_tree_reach_layers(nodes, children, child_cards);
        let (backup_nodes, backup_layer_ranges) = public_tree_backup_layers(nodes, children);

        let node_buffer = readonly_buffer(&self.device, "public tree nodes", nodes);
        let child_buffer = readonly_buffer(&self.device, "public tree children", children);
        let reach_edge_buffer =
            readonly_buffer(&self.device, "public tree reach edges", &reach_edges);
        let backup_nodes_buffer =
            readonly_buffer(&self.device, "public tree backup nodes", &backup_nodes);
        let combo_buffer = readonly_buffer(&self.device, "public tree combos", combos);
        let root_weights: Vec<_> = combo_legal
            .iter()
            .zip(villain_weights)
            .map(|(is_legal, weight)| if *is_legal != 0 { *weight } else { -1.0 })
            .collect();
        let root_weights_buffer =
            readonly_buffer(&self.device, "public tree root weights", &root_weights);
        let hero_reaches_buffer = uninit_storage_buffer(
            &self.device,
            "public tree hero reaches",
            node_combo_len,
            false,
        );
        let villain_reaches_buffer = uninit_storage_buffer(
            &self.device,
            "public tree villain reaches",
            node_combo_len,
            false,
        );
        let hero_values_buffer = uninit_storage_buffer(
            &self.device,
            "public tree hero private values",
            node_combo_len,
            false,
        );
        let villain_values_buffer = uninit_storage_buffer(
            &self.device,
            "public tree villain private values",
            node_combo_len,
            false,
        );
        let output_buffer = uninit_storage_buffer(
            &self.device,
            "public tree iteration output",
            output_len,
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
        let showdown_terminal_groups =
            terminal_group_caches(nodes, &showdown_terminal_nodes, combos, showdown_boards);
        let public_infoset_count = nodes_public_infoset_count(nodes);
        let mut decision_nodes = vec![u32::MAX; public_infoset_count];
        for (index, node) in nodes.iter().enumerate() {
            if node.kind == 0 {
                decision_nodes[node.public_infoset as usize] = index as u32;
            }
        }
        let decision_nodes_buffer =
            readonly_buffer(&self.device, "public tree decision nodes", &decision_nodes);
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

        GpuPublicTreeIterationContext {
            nodes_len: nodes.len(),
            combos_len: combos.len(),
            infosets,
            actions,
            action_len,
            output_len,
            node_combo_len,
            public_infoset_count,
            node_buffer,
            child_buffer,
            reach_edge_buffer,
            reach_layer_ranges,
            backup_nodes_buffer,
            backup_layer_ranges,
            combo_buffer,
            root_weights_buffer,
            hero_reaches_buffer,
            villain_reaches_buffer,
            hero_values_buffer,
            villain_values_buffer,
            output_buffer,
            fold_terminal_nodes,
            showdown_terminal_nodes,
            showdown_terminal_groups,
            decision_nodes_buffer,
            terminal_tile_count,
            terminal_board_tile_count,
            terminal_chunk_size,
            terminal_blocker_neighbors_buffer,
            terminal_blocker_neighbor_stride: blocker_neighbor_stride,
            terminal_prefix_pair_budget: max_terminal_prefix_pairs,
            terminal_prefix_pairs_buffer,
            hero_decision_aggregates_buffer,
            villain_decision_aggregates_buffer,
        }
    }

    fn public_tree_iteration_output_with_context(
        &self,
        ctx: &GpuPublicTreeIterationContext,
        regrets_buffer: &wgpu::Buffer,
    ) -> Result<(wgpu::Buffer, usize, usize), GpuCfrError> {
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
        let (init_x_groups, init_y_groups, init_x_invocations) = dispatch_grid(ctx.node_combo_len);
        let reach_init_params = uniform_buffer(
            &self.device,
            "public tree reach init params",
            &[GpuPublicTreeParams {
                combo_count: ctx.combos_len as u32,
                node_count: ctx.nodes_len as u32,
                max_actions: ctx.actions as u32,
                output_len: init_x_invocations,
                pair_start: 0,
                chunk_pairs: 0,
                _pad0: 0,
                _pad1: 0,
            }],
        );
        let reach_init_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("public tree reach init bind group"),
            layout: &self.public_tree_reach_bind_group_layout,
            entries: &[
                bind_entry(0, &ctx.node_buffer),
                bind_entry(1, &ctx.reach_edge_buffer),
                bind_entry(2, &ctx.combo_buffer),
                bind_entry(3, &ctx.root_weights_buffer),
                bind_entry(4, regrets_buffer),
                bind_entry(5, &ctx.hero_reaches_buffer),
                bind_entry(6, &ctx.villain_reaches_buffer),
                bind_entry(7, &reach_init_params),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("public tree reach init pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.public_tree_reach_init_pipeline);
            pass.set_bind_group(0, &reach_init_bind_group, &[]);
            pass.dispatch_workgroups(init_x_groups, init_y_groups, 1);
        }

        for &(layer_start, layer_end) in &ctx.reach_layer_ranges {
            let layer_edge_count = layer_end - layer_start;
            if layer_edge_count == 0 {
                continue;
            }
            let invocation_count = layer_edge_count * ctx.combos_len;
            let (x_groups, y_groups, x_invocations) = dispatch_grid(invocation_count);
            let reach_edge_params = uniform_buffer(
                &self.device,
                "public tree reach edge params",
                &[GpuPublicTreeParams {
                    combo_count: ctx.combos_len as u32,
                    node_count: ctx.nodes_len as u32,
                    max_actions: ctx.actions as u32,
                    output_len: x_invocations,
                    pair_start: 0,
                    chunk_pairs: layer_start as u32,
                    _pad0: layer_edge_count as u32,
                    _pad1: 0,
                }],
            );
            let reach_edge_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("public tree reach edge bind group"),
                layout: &self.public_tree_reach_bind_group_layout,
                entries: &[
                    bind_entry(0, &ctx.node_buffer),
                    bind_entry(1, &ctx.reach_edge_buffer),
                    bind_entry(2, &ctx.combo_buffer),
                    bind_entry(3, &ctx.root_weights_buffer),
                    bind_entry(4, regrets_buffer),
                    bind_entry(5, &ctx.hero_reaches_buffer),
                    bind_entry(6, &ctx.villain_reaches_buffer),
                    bind_entry(7, &reach_edge_params),
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("public tree reach edge pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.public_tree_reach_edge_pipeline);
                pass.set_bind_group(0, &reach_edge_bind_group, &[]);
                pass.dispatch_workgroups(x_groups, y_groups, 1);
            }
        }
        encoder = self.finish_profile_phase(encoder, "cfv_reach", phase_start)?;
        phase_start = profile.then(Instant::now);

        self.fill_fold_values(
            &mut encoder,
            &ctx.node_buffer,
            &ctx.fold_terminal_nodes,
            &ctx.combo_buffer,
            &ctx.hero_reaches_buffer,
            &ctx.villain_reaches_buffer,
            &ctx.hero_values_buffer,
            &ctx.villain_values_buffer,
            ctx.combos_len,
        )?;
        self.queue.submit(Some(encoder.finish()));
        self.fill_terminal_values_streaming(
            &ctx.node_buffer,
            &ctx.showdown_terminal_groups,
            &ctx.terminal_blocker_neighbors_buffer,
            &ctx.hero_reaches_buffer,
            &ctx.villain_reaches_buffer,
            &ctx.hero_values_buffer,
            &ctx.villain_values_buffer,
            &ctx.terminal_prefix_pairs_buffer,
            ctx.combos_len,
            ctx.terminal_blocker_neighbor_stride,
            ctx.terminal_prefix_pair_budget,
        )?;
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

        self.backup_nonterminal_values(
            &mut encoder,
            &ctx.node_buffer,
            &ctx.child_buffer,
            &ctx.backup_nodes_buffer,
            &ctx.hero_reaches_buffer,
            &ctx.villain_reaches_buffer,
            &ctx.hero_values_buffer,
            &ctx.villain_values_buffer,
            &ctx.backup_layer_ranges,
            ctx.combos_len,
            ctx.nodes_len,
            ctx.actions,
        )?;
        encoder = self.finish_profile_phase(encoder, "cfv_backup", phase_start)?;
        phase_start = profile.then(Instant::now);

        const DECISION_AGGREGATE_SLOTS: usize = 53;
        let (
            decision_aggregate_x_groups,
            decision_aggregate_y_groups,
            decision_aggregate_x_invocations,
        ) = dispatch_grid(ctx.public_infoset_count * DECISION_AGGREGATE_SLOTS);
        let decision_aggregate_params = uniform_buffer(
            &self.device,
            "public tree decision aggregate params",
            &[GpuPublicTreeParams {
                combo_count: ctx.combos_len as u32,
                node_count: ctx.public_infoset_count as u32,
                max_actions: DECISION_AGGREGATE_SLOTS as u32,
                output_len: decision_aggregate_x_invocations,
                pair_start: 0,
                chunk_pairs: 0,
                _pad0: 0,
                _pad1: 0,
            }],
        );
        let decision_aggregate_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("public tree decision aggregate bind group"),
                layout: &self.public_tree_decision_aggregate_bind_group_layout,
                entries: &[
                    bind_entry(0, &ctx.decision_nodes_buffer),
                    bind_entry(1, &ctx.combo_buffer),
                    bind_entry(2, &ctx.hero_reaches_buffer),
                    bind_entry(3, &ctx.villain_reaches_buffer),
                    bind_entry(4, &ctx.hero_decision_aggregates_buffer),
                    bind_entry(5, &ctx.villain_decision_aggregates_buffer),
                    bind_entry(6, &decision_aggregate_params),
                ],
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("public tree decision aggregate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.public_tree_decision_aggregate_pipeline);
            pass.set_bind_group(0, &decision_aggregate_bind_group, &[]);
            pass.dispatch_workgroups(decision_aggregate_x_groups, decision_aggregate_y_groups, 1);
        }
        for (label, value_player) in [("hero", 0u32), ("villain", 1u32)] {
            let denominator_params = uniform_buffer(
                &self.device,
                &format!("public tree {label} denominator params"),
                &[GpuPublicTreeParams {
                    combo_count: ctx.combos_len as u32,
                    node_count: ctx.nodes_len as u32,
                    max_actions: ctx.actions as u32,
                    output_len: ctx.infosets as u32,
                    pair_start: value_player,
                    chunk_pairs: 0,
                    _pad0: 0,
                    _pad1: 0,
                }],
            );
            let denominator_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("public tree denominator bind group"),
                    layout: &self.public_tree_denominator_bind_group_layout,
                    entries: &[
                        bind_entry(0, &ctx.node_buffer),
                        bind_entry(1, &ctx.combo_buffer),
                        bind_entry(2, &ctx.decision_nodes_buffer),
                        bind_entry(3, &ctx.hero_reaches_buffer),
                        bind_entry(4, &ctx.villain_reaches_buffer),
                        bind_entry(5, &ctx.hero_decision_aggregates_buffer),
                        bind_entry(6, &ctx.villain_decision_aggregates_buffer),
                        bind_entry(7, &ctx.output_buffer),
                        bind_entry(8, &denominator_params),
                    ],
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("public tree denominator pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.public_tree_denominator_pipeline);
                pass.set_bind_group(0, &denominator_bind_group, &[]);
                pass.dispatch_workgroups((ctx.infosets as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
            }
        }
        encoder = self.finish_profile_phase(encoder, "cfv_decision_denominator", phase_start)?;
        phase_start = profile.then(Instant::now);

        for (label, value_player, value_buffer) in [
            ("hero", 0u32, &ctx.hero_values_buffer),
            ("villain", 1u32, &ctx.villain_values_buffer),
        ] {
            let aggregate_params = uniform_buffer(
                &self.device,
                &format!("public tree {label} aggregate params"),
                &[GpuPublicTreeParams {
                    combo_count: ctx.combos_len as u32,
                    node_count: ctx.nodes_len as u32,
                    max_actions: ctx.actions as u32,
                    output_len: ctx.action_len as u32,
                    pair_start: value_player,
                    chunk_pairs: 0,
                    _pad0: 0,
                    _pad1: 0,
                }],
            );
            let aggregate_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("public tree aggregate bind group"),
                layout: &self.public_tree_aggregate_bind_group_layout,
                entries: &[
                    bind_entry(0, &ctx.node_buffer),
                    bind_entry(1, &ctx.child_buffer),
                    bind_entry(2, &ctx.combo_buffer),
                    bind_entry(3, &ctx.decision_nodes_buffer),
                    bind_entry(4, &ctx.hero_reaches_buffer),
                    bind_entry(5, &ctx.villain_reaches_buffer),
                    bind_entry(6, value_buffer),
                    bind_entry(7, &ctx.output_buffer),
                    bind_entry(8, &aggregate_params),
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("public tree aggregate pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.public_tree_aggregate_pipeline);
                pass.set_bind_group(0, &aggregate_bind_group, &[]);
                pass.dispatch_workgroups((ctx.action_len as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
            }
        }
        self.submit_final_profile_phase(encoder, "cfv_action_aggregate", phase_start)?;
        Ok((ctx.output_buffer.clone(), ctx.output_len, ctx.action_len))
    }

    #[allow(clippy::too_many_arguments)]
    fn public_tree_iteration_output(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        villain_weights: &[f32],
        showdown_boards: &[GpuFinalBoard],
        _strength_buffer: &wgpu::Buffer,
        regrets_buffer: &wgpu::Buffer,
        infosets: usize,
        actions: usize,
    ) -> Result<(wgpu::Buffer, usize, usize), GpuCfrError> {
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
        let profile = Self::gpu_profile_enabled();
        let setup_start = profile.then(Instant::now);
        let (reach_edges, reach_layer_ranges) =
            public_tree_reach_layers(nodes, children, child_cards);
        let (backup_nodes, backup_layer_ranges) = public_tree_backup_layers(nodes, children);

        let node_buffer = readonly_buffer(&self.device, "public tree nodes", nodes);
        let child_buffer = readonly_buffer(&self.device, "public tree children", children);
        let reach_edge_buffer =
            readonly_buffer(&self.device, "public tree reach edges", &reach_edges);
        let backup_nodes_buffer =
            readonly_buffer(&self.device, "public tree backup nodes", &backup_nodes);
        let _board_buffer =
            readonly_buffer(&self.device, "public tree showdown boards", showdown_boards);
        let combo_buffer = readonly_buffer(&self.device, "public tree combos", combos);
        let root_weights: Vec<_> = combo_legal
            .iter()
            .zip(villain_weights)
            .map(|(is_legal, weight)| if *is_legal != 0 { *weight } else { -1.0 })
            .collect();
        let root_weights_buffer =
            readonly_buffer(&self.device, "public tree root weights", &root_weights);
        let hero_reaches_buffer = uninit_storage_buffer(
            &self.device,
            "public tree hero reaches",
            node_combo_len,
            false,
        );
        let villain_reaches_buffer = uninit_storage_buffer(
            &self.device,
            "public tree villain reaches",
            node_combo_len,
            false,
        );
        let hero_values_buffer = uninit_storage_buffer(
            &self.device,
            "public tree hero private values",
            node_combo_len,
            false,
        );
        let villain_values_buffer = uninit_storage_buffer(
            &self.device,
            "public tree villain private values",
            node_combo_len,
            false,
        );
        let output_buffer = uninit_storage_buffer(
            &self.device,
            "public tree iteration output",
            output_len,
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
        let showdown_terminal_groups =
            terminal_group_caches(nodes, &showdown_terminal_nodes, combos, showdown_boards);
        let public_infoset_count = nodes_public_infoset_count(nodes);
        let mut decision_nodes = vec![u32::MAX; public_infoset_count];
        for (index, node) in nodes.iter().enumerate() {
            if node.kind == 0 {
                decision_nodes[node.public_infoset as usize] = index as u32;
            }
        }
        let decision_nodes_buffer =
            readonly_buffer(&self.device, "public tree decision nodes", &decision_nodes);
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
        if std::env::var_os("POKEDR_SOLVER_PROGRESS_OFF").is_none() {
            eprintln!(
                "pokedr: gpu public tree cfv nodes={} combos={} node_combo_values={} folds={} showdowns={} terminal_tiles={} board_tiles={} terminal_chunk={}",
                nodes.len(),
                combos.len(),
                node_combo_len,
                fold_terminal_nodes.len(),
                showdown_terminal_nodes.len(),
                terminal_tile_count,
                terminal_board_tile_count,
                terminal_chunk_size
            );
        }
        if let Some(start) = setup_start {
            eprintln!(
                "pokedr: gpu profile phase=cfv_setup elapsed_ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("public tree iteration encoder"),
            });
        let mut phase_start = profile.then(Instant::now);
        let (init_x_groups, init_y_groups, init_x_invocations) = dispatch_grid(node_combo_len);
        let reach_init_params = uniform_buffer(
            &self.device,
            "public tree reach init params",
            &[GpuPublicTreeParams {
                combo_count: combos.len() as u32,
                node_count: nodes.len() as u32,
                max_actions: actions as u32,
                output_len: init_x_invocations,
                pair_start: 0,
                chunk_pairs: 0,
                _pad0: 0,
                _pad1: 0,
            }],
        );
        let reach_init_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("public tree reach init bind group"),
            layout: &self.public_tree_reach_bind_group_layout,
            entries: &[
                bind_entry(0, &node_buffer),
                bind_entry(1, &reach_edge_buffer),
                bind_entry(2, &combo_buffer),
                bind_entry(3, &root_weights_buffer),
                bind_entry(4, &regrets_buffer),
                bind_entry(5, &hero_reaches_buffer),
                bind_entry(6, &villain_reaches_buffer),
                bind_entry(7, &reach_init_params),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("public tree reach init pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.public_tree_reach_init_pipeline);
            pass.set_bind_group(0, &reach_init_bind_group, &[]);
            pass.dispatch_workgroups(init_x_groups, init_y_groups, 1);
        }

        for &(layer_start, layer_end) in &reach_layer_ranges {
            let layer_edge_count = layer_end - layer_start;
            if layer_edge_count == 0 {
                continue;
            }
            let invocation_count = layer_edge_count * combos.len();
            let (x_groups, y_groups, x_invocations) = dispatch_grid(invocation_count);
            let reach_edge_params = uniform_buffer(
                &self.device,
                "public tree reach edge params",
                &[GpuPublicTreeParams {
                    combo_count: combos.len() as u32,
                    node_count: nodes.len() as u32,
                    max_actions: actions as u32,
                    output_len: x_invocations,
                    pair_start: 0,
                    chunk_pairs: layer_start as u32,
                    _pad0: layer_edge_count as u32,
                    _pad1: 0,
                }],
            );
            let reach_edge_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("public tree reach edge bind group"),
                layout: &self.public_tree_reach_bind_group_layout,
                entries: &[
                    bind_entry(0, &node_buffer),
                    bind_entry(1, &reach_edge_buffer),
                    bind_entry(2, &combo_buffer),
                    bind_entry(3, &root_weights_buffer),
                    bind_entry(4, &regrets_buffer),
                    bind_entry(5, &hero_reaches_buffer),
                    bind_entry(6, &villain_reaches_buffer),
                    bind_entry(7, &reach_edge_params),
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("public tree reach edge pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.public_tree_reach_edge_pipeline);
                pass.set_bind_group(0, &reach_edge_bind_group, &[]);
                pass.dispatch_workgroups(x_groups, y_groups, 1);
            }
        }
        encoder = self.finish_profile_phase(encoder, "cfv_reach", phase_start)?;
        phase_start = profile.then(Instant::now);

        self.fill_fold_values(
            &mut encoder,
            &node_buffer,
            &fold_terminal_nodes,
            &combo_buffer,
            &hero_reaches_buffer,
            &villain_reaches_buffer,
            &hero_values_buffer,
            &villain_values_buffer,
            combos.len(),
        )?;

        self.queue.submit(Some(encoder.finish()));
        self.fill_terminal_values_streaming(
            &node_buffer,
            &showdown_terminal_groups,
            &terminal_blocker_neighbors_buffer,
            &hero_reaches_buffer,
            &villain_reaches_buffer,
            &hero_values_buffer,
            &villain_values_buffer,
            &terminal_prefix_pairs_buffer,
            combos.len(),
            blocker_neighbor_stride,
            max_terminal_prefix_pairs,
        )?;
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

        self.backup_nonterminal_values(
            &mut encoder,
            &node_buffer,
            &child_buffer,
            &backup_nodes_buffer,
            &hero_reaches_buffer,
            &villain_reaches_buffer,
            &hero_values_buffer,
            &villain_values_buffer,
            &backup_layer_ranges,
            combos.len(),
            nodes.len(),
            actions,
        )?;
        encoder = self.finish_profile_phase(encoder, "cfv_backup", phase_start)?;
        phase_start = profile.then(Instant::now);
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
        let (
            decision_aggregate_x_groups,
            decision_aggregate_y_groups,
            decision_aggregate_x_invocations,
        ) = dispatch_grid(decision_aggregate_len);
        let decision_aggregate_params = uniform_buffer(
            &self.device,
            "public tree decision aggregate params",
            &[GpuPublicTreeParams {
                combo_count: combos.len() as u32,
                node_count: public_infoset_count as u32,
                max_actions: DECISION_AGGREGATE_SLOTS as u32,
                output_len: decision_aggregate_x_invocations,
                pair_start: 0,
                chunk_pairs: 0,
                _pad0: 0,
                _pad1: 0,
            }],
        );
        let decision_aggregate_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("public tree decision aggregate bind group"),
                layout: &self.public_tree_decision_aggregate_bind_group_layout,
                entries: &[
                    bind_entry(0, &decision_nodes_buffer),
                    bind_entry(1, &combo_buffer),
                    bind_entry(2, &hero_reaches_buffer),
                    bind_entry(3, &villain_reaches_buffer),
                    bind_entry(4, &hero_decision_aggregates_buffer),
                    bind_entry(5, &villain_decision_aggregates_buffer),
                    bind_entry(6, &decision_aggregate_params),
                ],
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("public tree decision aggregate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.public_tree_decision_aggregate_pipeline);
            pass.set_bind_group(0, &decision_aggregate_bind_group, &[]);
            pass.dispatch_workgroups(decision_aggregate_x_groups, decision_aggregate_y_groups, 1);
        }
        for (label, value_player) in [("hero", 0u32), ("villain", 1u32)] {
            let denominator_params = uniform_buffer(
                &self.device,
                &format!("public tree {label} denominator params"),
                &[GpuPublicTreeParams {
                    combo_count: combos.len() as u32,
                    node_count: nodes.len() as u32,
                    max_actions: actions as u32,
                    output_len: infosets as u32,
                    pair_start: value_player,
                    chunk_pairs: 0,
                    _pad0: 0,
                    _pad1: 0,
                }],
            );
            let denominator_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("public tree denominator bind group"),
                    layout: &self.public_tree_denominator_bind_group_layout,
                    entries: &[
                        bind_entry(0, &node_buffer),
                        bind_entry(1, &combo_buffer),
                        bind_entry(2, &decision_nodes_buffer),
                        bind_entry(3, &hero_reaches_buffer),
                        bind_entry(4, &villain_reaches_buffer),
                        bind_entry(5, &hero_decision_aggregates_buffer),
                        bind_entry(6, &villain_decision_aggregates_buffer),
                        bind_entry(7, &output_buffer),
                        bind_entry(8, &denominator_params),
                    ],
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("public tree denominator pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.public_tree_denominator_pipeline);
                pass.set_bind_group(0, &denominator_bind_group, &[]);
                pass.dispatch_workgroups((infosets as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
            }
        }
        encoder = self.finish_profile_phase(encoder, "cfv_decision_denominator", phase_start)?;
        phase_start = profile.then(Instant::now);
        let hero_aggregate_params = uniform_buffer(
            &self.device,
            "public tree hero aggregate params",
            &[GpuPublicTreeParams {
                combo_count: combos.len() as u32,
                node_count: nodes.len() as u32,
                max_actions: actions as u32,
                output_len: action_len as u32,
                pair_start: 0,
                chunk_pairs: 0,
                _pad0: 0,
                _pad1: 0,
            }],
        );
        let hero_aggregate_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("public tree hero aggregate bind group"),
            layout: &self.public_tree_aggregate_bind_group_layout,
            entries: &[
                bind_entry(0, &node_buffer),
                bind_entry(1, &child_buffer),
                bind_entry(2, &combo_buffer),
                bind_entry(3, &decision_nodes_buffer),
                bind_entry(4, &hero_reaches_buffer),
                bind_entry(5, &villain_reaches_buffer),
                bind_entry(6, &hero_values_buffer),
                bind_entry(7, &output_buffer),
                bind_entry(8, &hero_aggregate_params),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("public tree hero aggregate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.public_tree_aggregate_pipeline);
            pass.set_bind_group(0, &hero_aggregate_bind_group, &[]);
            pass.dispatch_workgroups((action_len as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        let villain_aggregate_params = uniform_buffer(
            &self.device,
            "public tree villain aggregate params",
            &[GpuPublicTreeParams {
                combo_count: combos.len() as u32,
                node_count: nodes.len() as u32,
                max_actions: actions as u32,
                output_len: action_len as u32,
                pair_start: 1,
                chunk_pairs: 0,
                _pad0: 0,
                _pad1: 0,
            }],
        );
        let villain_aggregate_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("public tree villain aggregate bind group"),
                layout: &self.public_tree_aggregate_bind_group_layout,
                entries: &[
                    bind_entry(0, &node_buffer),
                    bind_entry(1, &child_buffer),
                    bind_entry(2, &combo_buffer),
                    bind_entry(3, &decision_nodes_buffer),
                    bind_entry(4, &hero_reaches_buffer),
                    bind_entry(5, &villain_reaches_buffer),
                    bind_entry(6, &villain_values_buffer),
                    bind_entry(7, &output_buffer),
                    bind_entry(8, &villain_aggregate_params),
                ],
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("public tree villain aggregate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.public_tree_aggregate_pipeline);
            pass.set_bind_group(0, &villain_aggregate_bind_group, &[]);
            pass.dispatch_workgroups((action_len as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        self.submit_final_profile_phase(encoder, "cfv_action_aggregate", phase_start)?;
        Ok((output_buffer, output_len, action_len))
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
        let regrets_buffer = readonly_buffer(&self.device, "public tree regrets", &state.regrets);
        let strength_buffer = self.final_board_strength_buffer(combos, showdown_boards)?;
        let (output_buffer, output_len, action_len) = self.public_tree_iteration_output(
            nodes,
            children,
            child_cards,
            combos,
            combo_legal,
            villain_weights,
            showdown_boards,
            &strength_buffer,
            &regrets_buffer,
            state.infosets,
            state.actions,
        )?;
        let readback = readback_buffer(&self.device, output_len);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("public tree readback encoder"),
            });
        copy_buffer(&mut encoder, &output_buffer, &readback, output_len);
        let submission = self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;
        let output = read_f32_buffer(&self.device, &readback, output_len)?;
        let mut action_values = output[..action_len].to_vec();
        let value_weights = &output[action_len..action_len * 2];
        for (value, weight) in action_values.iter_mut().zip(value_weights) {
            if *weight > 0.0 {
                *value /= *weight;
            } else {
                *value = 0.0;
            }
        }
        let reach_start = action_len * 2;
        let strategy_start = reach_start + state.infosets;
        Ok(GpuRootTerminalValues {
            action_values,
            reach_weights: output[reach_start..strategy_start].to_vec(),
            strategy_weights: output[strategy_start..strategy_start + state.infosets].to_vec(),
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
        let strength_buffer = self.final_board_strength_buffer(combos, showdown_boards)?;
        self.public_tree_update_state_with_strengths(
            nodes,
            children,
            child_cards,
            combos,
            combo_legal,
            villain_weights,
            showdown_boards,
            &strength_buffer,
            state,
            iteration,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn public_tree_update_state_with_strengths(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        villain_weights: &[f32],
        showdown_boards: &[GpuFinalBoard],
        strength_buffer: &wgpu::Buffer,
        state: &mut GpuDenseCfrState,
        iteration: usize,
    ) -> Result<(), GpuCfrError> {
        let profile = Self::gpu_profile_enabled();
        let cfv_start = profile.then(Instant::now);
        let (output_buffer, _output_len, action_len) = self.public_tree_iteration_output(
            nodes,
            children,
            child_cards,
            combos,
            combo_legal,
            villain_weights,
            showdown_boards,
            strength_buffer,
            &state.regrets,
            state.infosets,
            state.actions,
        )?;
        if let Some(start) = cfv_start {
            self.profile_poll()?;
            eprintln!(
                "pokedr: gpu profile iteration={} phase=cfv elapsed_ms={:.3}",
                iteration,
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        let reach_start = action_len * 2;
        let strategy_start = reach_start + state.infosets;
        let params = readonly_buffer(
            &self.device,
            "public tree CFR update params",
            &[
                state.infosets as u32,
                state.actions as u32,
                variant_code(state.variant),
                iteration as u32,
                action_len as u32,
                reach_start as u32,
                strategy_start as u32,
            ],
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("public tree CFR update bind group"),
            layout: &self.public_tree_cfr_update_bind_group_layout,
            entries: &[
                bind_entry(0, &state.regrets),
                bind_entry(1, &state.strategy_sum),
                bind_entry(2, &output_buffer),
                bind_entry(3, &params),
                bind_entry(4, &state.legal_actions_buffer),
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

    fn public_tree_update_state_with_context(
        &self,
        context: &GpuPublicTreeIterationContext,
        state: &mut GpuDenseCfrState,
        iteration: usize,
    ) -> Result<(), GpuCfrError> {
        let profile = Self::gpu_profile_enabled();
        let cfv_start = profile.then(Instant::now);
        let (output_buffer, _output_len, action_len) =
            self.public_tree_iteration_output_with_context(context, &state.regrets)?;
        if let Some(start) = cfv_start {
            self.profile_poll()?;
            eprintln!(
                "pokedr: gpu profile iteration={} phase=cfv elapsed_ms={:.3}",
                iteration,
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        let reach_start = action_len * 2;
        let strategy_start = reach_start + state.infosets;
        let params = readonly_buffer(
            &self.device,
            "public tree CFR update params",
            &[
                state.infosets as u32,
                state.actions as u32,
                variant_code(state.variant),
                iteration as u32,
                action_len as u32,
                reach_start as u32,
                strategy_start as u32,
            ],
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("public tree CFR update bind group"),
            layout: &self.public_tree_cfr_update_bind_group_layout,
            entries: &[
                bind_entry(0, &state.regrets),
                bind_entry(1, &state.strategy_sum),
                bind_entry(2, &output_buffer),
                bind_entry(3, &params),
                bind_entry(4, &state.legal_actions_buffer),
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
        for iteration in 1..=iterations.max(1) {
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
        Ok(())
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
        let strategy_readback = readback_buffer(&backend.device, len);
        let mut encoder = backend
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident dense CFR download encoder"),
            });
        copy_buffer(&mut encoder, &self.regrets, &regret_readback, len);
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
    }
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

fn public_tree_reach_layers(
    nodes: &[GpuPublicTreeNode],
    children: &[u32],
    child_cards: &[u32],
) -> (Vec<GpuPublicTreeEdge>, Vec<(usize, usize)>) {
    let mut depths = vec![0usize; nodes.len()];
    let mut layered_edges: Vec<Vec<GpuPublicTreeEdge>> = Vec::new();
    for (parent, node) in nodes.iter().enumerate() {
        if node.kind != 0 && node.kind != 1 {
            continue;
        }
        let depth = depths[parent];
        if layered_edges.len() <= depth {
            layered_edges.resize_with(depth + 1, Vec::new);
        }
        for action in 0..node.child_count as usize {
            let child_offset = node.first_child as usize + action;
            let child = children[child_offset] as usize;
            depths[child] = depths[child].max(depth + 1);
            let card = if node.kind == 1 {
                child_cards.get(child_offset).copied().unwrap_or(u32::MAX)
            } else {
                u32::MAX
            };
            layered_edges[depth].push(GpuPublicTreeEdge {
                parent: parent as u32,
                child: child as u32,
                action: action as u32,
                card,
            });
        }
    }

    let mut edges = Vec::new();
    let mut ranges = Vec::new();
    for layer in layered_edges {
        let start = edges.len();
        edges.extend(layer);
        let end = edges.len();
        if start != end {
            ranges.push((start, end));
        }
    }
    (edges, ranges)
}

fn public_tree_backup_layers(
    nodes: &[GpuPublicTreeNode],
    children: &[u32],
) -> (Vec<u32>, Vec<(usize, usize)>) {
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
    let mut layers = vec![Vec::new(); max_depth + 1];
    for (node_index, node) in nodes.iter().enumerate() {
        if node.kind == 0 || node.kind == 1 {
            layers[depths[node_index]].push(node_index as u32);
        }
    }

    let mut backup_nodes = Vec::new();
    let mut ranges = Vec::new();
    for layer in layers {
        let start = backup_nodes.len();
        backup_nodes.extend(layer);
        let end = backup_nodes.len();
        if start != end {
            ranges.push((start, end));
        }
    }
    (backup_nodes, ranges)
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
        assert_close_vec("turn board-major cfv", &brute, &board_major, 1.0e-4);
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
        assert_close_vec("flop board-major cfv", &brute, &board_major, 1.0e-4);
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
