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
    let pair_count = combo_count * combo_count;
    if index >= pair_count {
        return;
    }
    let hero_index = index / combo_count;
    let villain_index = index % combo_count;
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
"#;

const ROOT_TERMINAL_VALUES_SHADER: &str = r#"
struct Combo {
    cards: array<u32, 2>,
};

struct ActionSpec {
    kind: u32,
    pot: f32,
    hero_invested: f32,
    _pad: u32,
};

struct RootTerminalParams {
    combo_count: u32,
    action_count: u32,
    max_actions: u32,
    acting_player: u32,
};

@group(0) @binding(0) var<storage, read> combos: array<Combo>;
@group(0) @binding(1) var<storage, read> combo_legal: array<u32>;
@group(0) @binding(2) var<storage, read> villain_weights: array<f32>;
@group(0) @binding(3) var<storage, read> showdown_matrix: array<f32>;
@group(0) @binding(4) var<storage, read> actions: array<ActionSpec>;
@group(0) @binding(5) var<storage, read_write> action_values: array<f32>;
@group(0) @binding(6) var<storage, read_write> reach_weights: array<f32>;
@group(0) @binding(7) var<storage, read_write> strategy_weights: array<f32>;
@group(0) @binding(8) var<uniform> params: RootTerminalParams;

fn collide(left: Combo, right: Combo) -> bool {
    return left.cards[0] == right.cards[0]
        || left.cards[0] == right.cards[1]
        || left.cards[1] == right.cards[0]
        || left.cards[1] == right.cards[1];
}

fn hero_utility(action: ActionSpec, hero_combo: u32, villain_combo: u32, combo_count: u32) -> f32 {
    if action.kind == 0u {
        return -action.hero_invested;
    }
    if action.kind == 1u {
        return action.pot - action.hero_invested;
    }
    let equity = showdown_matrix[hero_combo * combo_count + villain_combo];
    return equity * action.pot - action.hero_invested;
}

@compute @workgroup_size(64)
fn root_values(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let combo_count = params.combo_count;
    let action_count = params.action_count;
    let max_actions = params.max_actions;
    let acting_player = params.acting_player;
    if index >= combo_count * action_count {
        return;
    }

    let acting_combo = index / action_count;
    let action_index = index % action_count;
    let infoset_offset = acting_player * combo_count + acting_combo;
    let value_offset = infoset_offset * max_actions + action_index;
    if combo_legal[acting_combo] == 0u {
        action_values[value_offset] = 0.0;
        return;
    }

    let action = actions[action_index];
    var weighted_value = 0.0;
    var value_weight = 0.0;
    var strategy_weight = 0.0;
    var opponent = 0u;
    loop {
        if opponent >= combo_count {
            break;
        }
        if combo_legal[opponent] != 0u && !collide(combos[acting_combo], combos[opponent]) {
            var hero_combo = acting_combo;
            var villain_combo = opponent;
            var opponent_reach = villain_weights[opponent];
            var own_reach = 1.0;
            if acting_player == 1u {
                hero_combo = opponent;
                villain_combo = acting_combo;
                opponent_reach = 1.0;
                own_reach = villain_weights[acting_combo];
            }
            var value = hero_utility(action, hero_combo, villain_combo, combo_count);
            if acting_player == 1u {
                value = -value;
            }
            weighted_value = weighted_value + opponent_reach * value;
            value_weight = value_weight + opponent_reach;
            strategy_weight = strategy_weight + own_reach;
        }
        opponent = opponent + 1u;
    }

    if value_weight > 0.0 {
        action_values[value_offset] = weighted_value / value_weight;
    } else {
        action_values[value_offset] = 0.0;
    }
    if action_index == 0u {
        reach_weights[infoset_offset] = value_weight;
        strategy_weights[infoset_offset] = strategy_weight;
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
    showdown_matrix_bind_group_layout: wgpu::BindGroupLayout,
    root_terminal_values_pipeline: wgpu::ComputePipeline,
    root_terminal_values_bind_group_layout: wgpu::BindGroupLayout,
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
pub struct GpuTerminalAction {
    pub kind: u32,
    pub pot: f32,
    pub hero_invested: f32,
    pub _pad: u32,
}

unsafe impl bytemuck::Zeroable for GpuTerminalAction {}
unsafe impl bytemuck::Pod for GpuTerminalAction {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GpuRootTerminalParams {
    combo_count: u32,
    action_count: u32,
    max_actions: u32,
    acting_player: u32,
}

unsafe impl bytemuck::Zeroable for GpuRootTerminalParams {}
unsafe impl bytemuck::Pod for GpuRootTerminalParams {}

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
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("pokedr dense CFR device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
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
        let root_terminal_values_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("root terminal values shader"),
                source: wgpu::ShaderSource::Wgsl(ROOT_TERMINAL_VALUES_SHADER.into()),
            });
        let root_terminal_values_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("root terminal values bind group layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, true),
                    storage_entry(5, false),
                    storage_entry(6, false),
                    storage_entry(7, false),
                    uniform_entry(8),
                ],
            });
        let root_terminal_values_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("root terminal values pipeline layout"),
                bind_group_layouts: &[Some(&root_terminal_values_bind_group_layout)],
                immediate_size: 0,
            });
        let root_terminal_values_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("root terminal values pipeline"),
                layout: Some(&root_terminal_values_pipeline_layout),
                module: &root_terminal_values_shader,
                entry_point: Some("root_values"),
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
            showdown_matrix_bind_group_layout,
            root_terminal_values_pipeline,
            root_terminal_values_bind_group_layout,
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
        if combos.is_empty() {
            return Ok(Vec::new());
        }
        assert!(
            !final_boards.is_empty(),
            "showdown matrix needs at least one final board"
        );

        let pair_count = combos.len() * combos.len();
        let combo_buffer = readonly_buffer(&self.device, "showdown matrix combos", combos);
        let board_buffer = readonly_buffer(&self.device, "showdown matrix boards", final_boards);
        let output = vec![0.0f32; pair_count];
        let output_buffer = storage_buffer(&self.device, "showdown matrix equities", &output);
        let params = readonly_buffer(
            &self.device,
            "showdown matrix params",
            &[combos.len() as u32, final_boards.len() as u32],
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
            let groups = (pair_count as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        let readback = readback_buffer(&self.device, pair_count);
        copy_buffer(&mut encoder, &output_buffer, &readback, pair_count);
        let submission = self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;
        read_f32_buffer(&self.device, &readback, pair_count)
    }

    pub fn root_terminal_values(
        &self,
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        villain_weights: &[f32],
        showdown_matrix: &[f32],
        actions: &[GpuTerminalAction],
        max_actions: usize,
        acting_player: u32,
    ) -> Result<GpuRootTerminalValues, GpuCfrError> {
        assert_eq!(combo_legal.len(), combos.len());
        assert_eq!(villain_weights.len(), combos.len());
        assert_eq!(showdown_matrix.len(), combos.len() * combos.len());
        assert!(actions.len() <= max_actions);
        if combos.is_empty() || actions.is_empty() {
            return Ok(GpuRootTerminalValues {
                action_values: Vec::new(),
                reach_weights: Vec::new(),
                strategy_weights: Vec::new(),
            });
        }

        let infosets = combos.len() * 2;
        let action_value_len = infosets * max_actions;
        let combo_buffer = readonly_buffer(&self.device, "root terminal combos", combos);
        let combo_legal_buffer =
            readonly_buffer(&self.device, "root terminal combo legal", combo_legal);
        let villain_weights_buffer = readonly_buffer(
            &self.device,
            "root terminal villain weights",
            villain_weights,
        );
        let matrix_buffer = readonly_buffer(
            &self.device,
            "root terminal showdown matrix",
            showdown_matrix,
        );
        let actions_buffer = readonly_buffer(&self.device, "root terminal actions", actions);
        let action_values = vec![0.0f32; action_value_len];
        let reach_weights = vec![0.0f32; infosets];
        let strategy_weights = vec![0.0f32; infosets];
        let action_values_buffer =
            storage_buffer(&self.device, "root terminal action values", &action_values);
        let reach_weights_buffer =
            storage_buffer(&self.device, "root terminal reach weights", &reach_weights);
        let strategy_weights_buffer = storage_buffer(
            &self.device,
            "root terminal strategy weights",
            &strategy_weights,
        );
        let params = uniform_buffer(
            &self.device,
            "root terminal params",
            &[GpuRootTerminalParams {
                combo_count: combos.len() as u32,
                action_count: actions.len() as u32,
                max_actions: max_actions as u32,
                acting_player,
            }],
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("root terminal values bind group"),
            layout: &self.root_terminal_values_bind_group_layout,
            entries: &[
                bind_entry(0, &combo_buffer),
                bind_entry(1, &combo_legal_buffer),
                bind_entry(2, &villain_weights_buffer),
                bind_entry(3, &matrix_buffer),
                bind_entry(4, &actions_buffer),
                bind_entry(5, &action_values_buffer),
                bind_entry(6, &reach_weights_buffer),
                bind_entry(7, &strategy_weights_buffer),
                bind_entry(8, &params),
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("root terminal values encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("root terminal values pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.root_terminal_values_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = ((combos.len() * actions.len()) as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        let action_values_readback = readback_buffer(&self.device, action_value_len);
        let reach_weights_readback = readback_buffer(&self.device, infosets);
        let strategy_weights_readback = readback_buffer(&self.device, infosets);
        copy_buffer(
            &mut encoder,
            &action_values_buffer,
            &action_values_readback,
            action_value_len,
        );
        copy_buffer(
            &mut encoder,
            &reach_weights_buffer,
            &reach_weights_readback,
            infosets,
        );
        copy_buffer(
            &mut encoder,
            &strategy_weights_buffer,
            &strategy_weights_readback,
            infosets,
        );
        let submission = self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;
        Ok(GpuRootTerminalValues {
            action_values: read_f32_buffer(
                &self.device,
                &action_values_readback,
                action_value_len,
            )?,
            reach_weights: read_f32_buffer(&self.device, &reach_weights_readback, infosets)?,
            strategy_weights: read_f32_buffer(&self.device, &strategy_weights_readback, infosets)?,
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
