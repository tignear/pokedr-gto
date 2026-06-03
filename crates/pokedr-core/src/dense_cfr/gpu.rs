use std::sync::mpsc;

use wgpu::util::DeviceExt;

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
struct Combo { cards: array<u32, 2>, };
struct FinalBoard { cards: array<u32, 5>, };
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
    tile_count: u32,
    tile_size: u32,
    value_player: u32,
    x_invocations: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> nodes: array<TreeNode>;
@group(0) @binding(1) var<storage, read> terminal_nodes: array<u32>;
@group(0) @binding(2) var<storage, read> boards: array<FinalBoard>;
@group(0) @binding(3) var<storage, read> combos: array<Combo>;
@group(0) @binding(4) var<storage, read> strengths: array<f32>;
@group(0) @binding(5) var<storage, read> hero_reaches: array<f32>;
@group(0) @binding(6) var<storage, read> villain_reaches: array<f32>;
@group(0) @binding(7) var<storage, read_write> partials: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

fn collide(left: Combo, right: Combo) -> bool {
    return left.cards[0] == right.cards[0] || left.cards[0] == right.cards[1]
        || left.cards[1] == right.cards[0] || left.cards[1] == right.cards[1];
}

fn combo_hits_board(combo: Combo, board: FinalBoard) -> bool {
    return combo.cards[0] == board.cards[0]
        || combo.cards[0] == board.cards[1]
        || combo.cards[0] == board.cards[2]
        || combo.cards[0] == board.cards[3]
        || combo.cards[0] == board.cards[4]
        || combo.cards[1] == board.cards[0]
        || combo.cards[1] == board.cards[1]
        || combo.cards[1] == board.cards[2]
        || combo.cards[1] == board.cards[3]
        || combo.cards[1] == board.cards[4];
}

@compute @workgroup_size(64)
fn terminal_partial(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * params.x_invocations;
    let output_count = params.terminal_count * params.combo_count * params.tile_count;
    if index >= output_count {
        return;
    }
    let tile = index % params.tile_count;
    let combo = (index / params.tile_count) % params.combo_count;
    let terminal_slot = index / (params.tile_count * params.combo_count);
    let node_index = terminal_nodes[terminal_slot];
    let node = nodes[node_index];
    let node_offset = node_index * params.combo_count;
    let opponent_start = tile * params.tile_size;
    let opponent_end = min(opponent_start + params.tile_size, params.combo_count);
    var value = 0.0;

    if node.terminal_kind == 0u {
        let payoff = -node.hero_invested;
        for (var opponent = opponent_start; opponent < opponent_end; opponent = opponent + 1u) {
            if collide(combos[combo], combos[opponent]) {
                continue;
            }
            if params.value_player == 0u {
                value = value + villain_reaches[node_offset + opponent] * payoff;
            } else {
                value = value + hero_reaches[node_offset + opponent] * (-payoff);
            }
        }
    } else if node.terminal_kind == 1u {
        let payoff = node.pot - node.hero_invested;
        for (var opponent = opponent_start; opponent < opponent_end; opponent = opponent + 1u) {
            if collide(combos[combo], combos[opponent]) {
                continue;
            }
            if params.value_player == 0u {
                value = value + villain_reaches[node_offset + opponent] * payoff;
            } else {
                value = value + hero_reaches[node_offset + opponent] * (-payoff);
            }
        }
    } else {
        let board_count = node._pad0;
        for (var opponent = opponent_start; opponent < opponent_end; opponent = opponent + 1u) {
            if collide(combos[combo], combos[opponent]) {
                continue;
            }
            var equity_sum = 0.0;
            var valid_boards = 0u;
            for (var board = 0u; board < board_count; board = board + 1u) {
                let final_board_index = node.showdown_offset + board;
                let final_board = boards[final_board_index];
                if combo_hits_board(combos[combo], final_board) || combo_hits_board(combos[opponent], final_board) {
                    continue;
                }
                valid_boards = valid_boards + 1u;
                let strength_offset = final_board_index * params.combo_count;
                let combo_strength = strengths[strength_offset + combo];
                let opponent_strength = strengths[strength_offset + opponent];
                if combo_strength > opponent_strength {
                    equity_sum = equity_sum + 1.0;
                } else if combo_strength == opponent_strength {
                    equity_sum = equity_sum + 0.5;
                }
            }
            let equity = select(0.5, equity_sum / f32(valid_boards), valid_boards > 0u);
            if params.value_player == 0u {
                let hero_payoff = equity * node.pot - node.hero_invested;
                value = value + villain_reaches[node_offset + opponent] * hero_payoff;
            } else {
                let villain_payoff = equity * node.pot - (node.pot - node.hero_invested);
                value = value + hero_reaches[node_offset + opponent] * villain_payoff;
            }
        }
    }
    partials[index] = value * node._pad1;
}
"#;

const PUBLIC_TREE_TERMINAL_REDUCE_SHADER: &str = r#"
struct Params {
    combo_count: u32,
    terminal_count: u32,
    tile_count: u32,
    output_len: u32,
    value_player: u32,
    x_invocations: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> terminal_nodes: array<u32>;
@group(0) @binding(1) var<storage, read> partials: array<f32>;
@group(0) @binding(2) var<storage, read_write> values: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

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
    let partial_base = (terminal_slot * params.combo_count + combo) * params.tile_count;
    var value = 0.0;
    for (var tile = 0u; tile < params.tile_count; tile = tile + 1u) {
        value = value + partials[partial_base + tile];
    }
    values[node_index * params.combo_count + combo] = value;
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
@group(0) @binding(5) var<storage, read_write> values: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

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
    var value = 0.0;
    if node.kind == 0u {
        for (var action = 0u; action < node.child_count; action = action + 1u) {
            let child = children[node.first_child + action];
            let child_value = values[child * params.combo_count + combo];
            if node.acting_player == params.value_player {
                let probability = strategy_from_reach(node_index, child, combo, node.acting_player);
                value = value + probability * child_value;
            } else {
                value = value + child_value;
            }
        }
    } else if node.kind == 1u {
        for (var action = 0u; action < node.child_count; action = action + 1u) {
            let child = children[node.first_child + action];
            value = value + values[child * params.combo_count + combo];
        }
    }
    values[node_index * params.combo_count + combo] = value;
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
    var value_weight = 0.0;
    let node_base = node_index * params.combo_count;
    for (var opponent = 0u; opponent < params.combo_count; opponent = opponent + 1u) {
        if collide(combos[acting_combo], combos[opponent]) {
            continue;
        }
        if node.acting_player == 0u {
            value_weight = value_weight + villain_reaches[node_base + opponent];
        } else {
            value_weight = value_weight + hero_reaches[node_base + opponent];
        }
    }
    let value_offset = child * params.combo_count + acting_combo;
    let action_value = values[value_offset];
    var strategy_weight = node._pad1 * hero_reaches[node_base + acting_combo];
    if node.acting_player == 1u {
        strategy_weight = node._pad1 * villain_reaches[node_base + acting_combo];
    }
    output[index] = action_value;
    output[action_len + index] = node._pad1 * value_weight;
    if action == 0u {
        let infoset_count = action_len / params.max_actions;
        output[action_len * 2u + private_infoset] = node._pad1 * value_weight;
        output[action_len * 2u + infoset_count + private_infoset] = strategy_weight;
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
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
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
    public_tree_backup_pipeline: wgpu::ComputePipeline,
    public_tree_backup_bind_group_layout: wgpu::BindGroupLayout,
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
                    storage_entry(6, true),
                    storage_entry(7, false),
                    uniform_entry(8),
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
                    storage_entry(2, false),
                    uniform_entry(3),
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
                    uniform_entry(6),
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
            pipeline,
            bind_group_layout,
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
            public_tree_backup_pipeline,
            public_tree_backup_bind_group_layout,
            public_tree_aggregate_pipeline,
            public_tree_aggregate_bind_group_layout,
        })
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
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

    fn final_board_strengths(
        &self,
        combos: &[GpuPrivateCombo],
        final_boards: &[GpuFinalBoard],
    ) -> Result<Vec<f32>, GpuCfrError> {
        if combos.is_empty() || final_boards.is_empty() {
            return Ok(Vec::new());
        }
        let output_count = combos.len() * final_boards.len();
        let combo_buffer = readonly_buffer(&self.device, "strength combos", combos);
        let board_buffer = readonly_buffer(&self.device, "strength boards", final_boards);
        let output_buffer = storage_buffer(
            &self.device,
            "final board strengths",
            &vec![0.0f32; output_count],
        );
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
    fn fill_terminal_values(
        &self,
        node_buffer: &wgpu::Buffer,
        terminal_nodes: &[u32],
        board_buffer: &wgpu::Buffer,
        combo_buffer: &wgpu::Buffer,
        strength_buffer: &wgpu::Buffer,
        hero_reaches_buffer: &wgpu::Buffer,
        villain_reaches_buffer: &wgpu::Buffer,
        values_buffer: &wgpu::Buffer,
        terminal_partials_buffer: &wgpu::Buffer,
        combo_count: usize,
        tile_count: usize,
        tile_size: usize,
        terminal_chunk_size: usize,
        value_player: u32,
    ) -> Result<(), GpuCfrError> {
        if terminal_nodes.is_empty() || combo_count == 0 || tile_count == 0 {
            return Ok(());
        }
        for terminal_chunk in terminal_nodes.chunks(terminal_chunk_size.max(1)) {
            let terminal_count = terminal_chunk.len();
            let terminal_nodes_buffer =
                readonly_buffer(&self.device, "public tree terminal nodes", terminal_chunk);
            let partial_invocations = terminal_count * combo_count * tile_count;
            let (partial_x_groups, partial_y_groups, partial_x_invocations) =
                dispatch_grid(partial_invocations);
            let partial_params = uniform_buffer(
                &self.device,
                "public tree terminal partial params",
                &[GpuPublicTreeParams {
                    combo_count: combo_count as u32,
                    node_count: terminal_count as u32,
                    max_actions: tile_count as u32,
                    output_len: tile_size as u32,
                    pair_start: value_player,
                    chunk_pairs: partial_x_invocations,
                    _pad0: 0,
                    _pad1: 0,
                }],
            );
            let partial_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("public tree terminal partial bind group"),
                layout: &self.public_tree_terminal_partial_bind_group_layout,
                entries: &[
                    bind_entry(0, node_buffer),
                    bind_entry(1, &terminal_nodes_buffer),
                    bind_entry(2, board_buffer),
                    bind_entry(3, combo_buffer),
                    bind_entry(4, strength_buffer),
                    bind_entry(5, hero_reaches_buffer),
                    bind_entry(6, villain_reaches_buffer),
                    bind_entry(7, terminal_partials_buffer),
                    bind_entry(8, &partial_params),
                ],
            });

            let reduce_invocations = terminal_count * combo_count;
            let (reduce_x_groups, reduce_y_groups, reduce_x_invocations) =
                dispatch_grid(reduce_invocations);
            let reduce_params = uniform_buffer(
                &self.device,
                "public tree terminal reduce params",
                &[GpuPublicTreeParams {
                    combo_count: combo_count as u32,
                    node_count: terminal_count as u32,
                    max_actions: tile_count as u32,
                    output_len: 0,
                    pair_start: value_player,
                    chunk_pairs: reduce_x_invocations,
                    _pad0: 0,
                    _pad1: 0,
                }],
            );
            let reduce_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("public tree terminal reduce bind group"),
                layout: &self.public_tree_terminal_reduce_bind_group_layout,
                entries: &[
                    bind_entry(0, &terminal_nodes_buffer),
                    bind_entry(1, terminal_partials_buffer),
                    bind_entry(2, values_buffer),
                    bind_entry(3, &reduce_params),
                ],
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("public tree terminal partial encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("public tree terminal partial pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.public_tree_terminal_partial_pipeline);
                pass.set_bind_group(0, &partial_bind_group, &[]);
                pass.dispatch_workgroups(partial_x_groups, partial_y_groups, 1);
            }
            let submission = self.queue.submit(Some(encoder.finish()));
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: None,
                })
                .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("public tree terminal reduce encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("public tree terminal reduce pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.public_tree_terminal_reduce_pipeline);
                pass.set_bind_group(0, &reduce_bind_group, &[]);
                pass.dispatch_workgroups(reduce_x_groups, reduce_y_groups, 1);
            }
            let submission = self.queue.submit(Some(encoder.finish()));
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: None,
                })
                .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn backup_nonterminal_values(
        &self,
        node_buffer: &wgpu::Buffer,
        child_buffer: &wgpu::Buffer,
        backup_nodes_buffer: &wgpu::Buffer,
        hero_reaches_buffer: &wgpu::Buffer,
        villain_reaches_buffer: &wgpu::Buffer,
        values_buffer: &wgpu::Buffer,
        backup_layer_ranges: &[(usize, usize)],
        combo_count: usize,
        node_count: usize,
        max_actions: usize,
        value_player: u32,
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
                    pair_start: value_player,
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
                    bind_entry(5, values_buffer),
                    bind_entry(6, &params),
                ],
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("public tree backup encoder"),
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
            let submission = self.queue.submit(Some(encoder.finish()));
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: None,
                })
                .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;
        }
        Ok(())
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
        assert!(!nodes.is_empty());
        assert_eq!(combo_legal.len(), combos.len());
        assert_eq!(villain_weights.len(), combos.len());
        assert_eq!(
            state.infosets,
            nodes_public_infoset_count(nodes) * combos.len() * 2
        );

        let action_len = state.infosets * state.actions;
        let output_len = action_len * 2 + state.infosets * 2;
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
        let board_buffer =
            readonly_buffer(&self.device, "public tree showdown boards", showdown_boards);
        let combo_buffer = readonly_buffer(&self.device, "public tree combos", combos);
        let root_weights: Vec<_> = combo_legal
            .iter()
            .zip(villain_weights)
            .map(|(is_legal, weight)| if *is_legal != 0 { *weight } else { -1.0 })
            .collect();
        let root_weights_buffer =
            readonly_buffer(&self.device, "public tree root weights", &root_weights);
        let regrets_buffer = readonly_buffer(&self.device, "public tree regrets", &state.regrets);
        let strength_values = self.final_board_strengths(combos, showdown_boards)?;
        let strength_buffer = readonly_buffer(
            &self.device,
            "public tree final board strengths",
            &strength_values,
        );
        let hero_reaches_buffer = storage_buffer(
            &self.device,
            "public tree hero reaches",
            &vec![0.0f32; node_combo_len],
        );
        let villain_reaches_buffer = storage_buffer(
            &self.device,
            "public tree villain reaches",
            &vec![0.0f32; node_combo_len],
        );
        let values_buffer = storage_buffer(
            &self.device,
            "public tree private values",
            &vec![0.0f32; node_combo_len],
        );
        let output_buffer = storage_buffer(
            &self.device,
            "public tree iteration output",
            &vec![0.0f32; output_len],
        );
        let terminal_nodes: Vec<_> = nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| (node.kind != 0 && node.kind != 1).then_some(index as u32))
            .collect();
        let public_infoset_count = nodes_public_infoset_count(nodes);
        let mut decision_nodes = vec![u32::MAX; public_infoset_count];
        for (index, node) in nodes.iter().enumerate() {
            if node.kind == 0 {
                decision_nodes[node.public_infoset as usize] = index as u32;
            }
        }
        let decision_nodes_buffer =
            readonly_buffer(&self.device, "public tree decision nodes", &decision_nodes);
        let terminal_tile_count = std::env::var("POKEDR_GPU_TERMINAL_TILES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(8)
            .max(1)
            .min(combos.len().max(1));
        let terminal_tile_size = combos.len().div_ceil(terminal_tile_count);
        let max_terminal_partial_values = 32_000_000usize;
        let terminal_chunk_size =
            (max_terminal_partial_values / (combos.len() * terminal_tile_count)).max(1);
        let terminal_partial_len = terminal_chunk_size.min(terminal_nodes.len()).max(1)
            * combos.len()
            * terminal_tile_count;
        if std::env::var_os("POKEDR_SOLVER_PROGRESS_OFF").is_none() {
            eprintln!(
                "pokedr: gpu public tree cfv nodes={} combos={} node_combo_values={} terminals={} terminal_tiles={} terminal_chunk={}",
                nodes.len(),
                combos.len(),
                node_combo_len,
                terminal_nodes.len(),
                terminal_tile_count,
                terminal_chunk_size
            );
        }
        let terminal_partials_buffer = storage_buffer(
            &self.device,
            "public tree terminal partial values",
            &vec![0.0f32; terminal_partial_len],
        );
        let (init_x_groups, init_y_groups, init_x_invocations) = dispatch_grid(node_combo_len);
        let reach_init_params = uniform_buffer(
            &self.device,
            "public tree reach init params",
            &[GpuPublicTreeParams {
                combo_count: combos.len() as u32,
                node_count: nodes.len() as u32,
                max_actions: state.actions as u32,
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
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("public tree reach init encoder"),
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
        let submission = self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;

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
                    max_actions: state.actions as u32,
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
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("public tree reach edge encoder"),
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
            let submission = self.queue.submit(Some(encoder.finish()));
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: None,
                })
                .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;
        }

        self.fill_terminal_values(
            &node_buffer,
            &terminal_nodes,
            &board_buffer,
            &combo_buffer,
            &strength_buffer,
            &hero_reaches_buffer,
            &villain_reaches_buffer,
            &values_buffer,
            &terminal_partials_buffer,
            combos.len(),
            terminal_tile_count,
            terminal_tile_size,
            terminal_chunk_size,
            0,
        )?;

        self.backup_nonterminal_values(
            &node_buffer,
            &child_buffer,
            &backup_nodes_buffer,
            &hero_reaches_buffer,
            &villain_reaches_buffer,
            &values_buffer,
            &backup_layer_ranges,
            combos.len(),
            nodes.len(),
            state.actions,
            0,
        )?;
        let hero_aggregate_params = uniform_buffer(
            &self.device,
            "public tree hero aggregate params",
            &[GpuPublicTreeParams {
                combo_count: combos.len() as u32,
                node_count: nodes.len() as u32,
                max_actions: state.actions as u32,
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
                bind_entry(6, &values_buffer),
                bind_entry(7, &output_buffer),
                bind_entry(8, &hero_aggregate_params),
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("public tree hero aggregate encoder"),
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
        let submission = self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;

        self.fill_terminal_values(
            &node_buffer,
            &terminal_nodes,
            &board_buffer,
            &combo_buffer,
            &strength_buffer,
            &hero_reaches_buffer,
            &villain_reaches_buffer,
            &values_buffer,
            &terminal_partials_buffer,
            combos.len(),
            terminal_tile_count,
            terminal_tile_size,
            terminal_chunk_size,
            1,
        )?;

        self.backup_nonterminal_values(
            &node_buffer,
            &child_buffer,
            &backup_nodes_buffer,
            &hero_reaches_buffer,
            &villain_reaches_buffer,
            &values_buffer,
            &backup_layer_ranges,
            combos.len(),
            nodes.len(),
            state.actions,
            1,
        )?;
        let villain_aggregate_params = uniform_buffer(
            &self.device,
            "public tree villain aggregate params",
            &[GpuPublicTreeParams {
                combo_count: combos.len() as u32,
                node_count: nodes.len() as u32,
                max_actions: state.actions as u32,
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
                    bind_entry(6, &values_buffer),
                    bind_entry(7, &output_buffer),
                    bind_entry(8, &villain_aggregate_params),
                ],
            });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("public tree final encoder"),
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
        let readback = readback_buffer(&self.device, output_len);
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
