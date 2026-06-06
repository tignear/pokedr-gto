use std::{
    collections::BTreeMap,
    num::NonZeroU64,
    sync::mpsc,
    time::{Duration, Instant},
};

use wgpu::util::DeviceExt;

use crate::cards::{Card, evaluate};

use super::{
    CompactPrivateCfrChunk, CompactPrivateCfrConfig, DenseCfrConfig, DenseCfrIteration,
    DenseCfrRunStats, DenseCfrState,
};

const WORKGROUP_SIZE: u32 = 64;
const SHOWDOWN_CARDS: usize = 9;

mod shaders;

use shaders::*;

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
    public_tree_compact_reach_edge_pipeline: wgpu::ComputePipeline,
    public_tree_layer_reach_init_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_layer_reach_edge_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_compact_reach_edge_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_terminal_partial_pipeline: wgpu::ComputePipeline,
    public_tree_terminal_partial_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_terminal_reduce_pipeline: wgpu::ComputePipeline,
    public_tree_terminal_reduce_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_fold_aggregate_pipeline: wgpu::ComputePipeline,
    public_tree_fold_aggregate_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_fold_value_pipeline: wgpu::ComputePipeline,
    public_tree_fold_value_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_layer_backup_init_pipeline: wgpu::ComputePipeline,
    public_tree_layer_backup_child_pipeline: wgpu::ComputePipeline,
    public_tree_layer_compact_backup_child_pipeline: wgpu::ComputePipeline,
    public_tree_layer_backup_init_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_layer_backup_child_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_layer_compact_backup_child_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_layer_decision_aggregate_pipeline: wgpu::ComputePipeline,
    public_tree_layer_denominator_pipeline: wgpu::ComputePipeline,
    public_tree_layer_action_edge_pipeline: wgpu::ComputePipeline,
    public_tree_layer_fused_update_pipeline: wgpu::ComputePipeline,
    public_tree_layer_compact_fused_update_pipeline: wgpu::ComputePipeline,
    public_tree_layer_output_bind_group_layout: wgpu::BindGroupLayout,
    public_tree_layer_fused_update_bind_group_layout: wgpu::BindGroupLayout,
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

pub struct GpuCompactPrivateCfrState {
    public_infosets: usize,
    public_actions: usize,
    combos: usize,
    variant: super::CfrVariant,
    chunks: Vec<GpuCompactPrivateCfrChunkState>,
}

pub struct GpuCompactPrivateCfrChunkState {
    chunk: CompactPrivateCfrChunk,
    regrets: wgpu::Buffer,
    prediction: Option<wgpu::Buffer>,
    strategy_sum: Option<wgpu::Buffer>,
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

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GpuPublicTreeEdgeGroup {
    parent: u32,
    first_edge: u32,
    edge_count: u32,
    _pad0: u32,
}

unsafe impl bytemuck::Zeroable for GpuPublicTreeEdgeGroup {}
unsafe impl bytemuck::Pod for GpuPublicTreeEdgeGroup {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GpuTerminalRef {
    node: u32,
    table: u32,
}

unsafe impl bytemuck::Zeroable for GpuTerminalRef {}
unsafe impl bytemuck::Pod for GpuTerminalRef {}

struct GpuPublicTreeIterationContext {
    nodes_len: usize,
    combos_len: usize,
    actions: usize,
    public_action_offsets: Vec<u32>,
    action_len: usize,
    output_len: usize,
    node_combo_len: usize,
    layered: GpuPublicTreeLayered,
    combo_buffer: wgpu::Buffer,
    root_weights_buffer: wgpu::Buffer,
    public_action_offsets_buffer: wgpu::Buffer,
    action_values_buffer: wgpu::Buffer,
    reach_weights_buffer: wgpu::Buffer,
    strategy_weights_buffer: wgpu::Buffer,
    empty_storage_buffer: wgpu::Buffer,
    layer_tiles: Vec<Vec<GpuPublicTreeLayerTileBuffers>>,
    reach_edge_buffers: Vec<GpuPublicTreeLayerReachBuffers>,
    fold_terminal_nodes: Vec<u32>,
    showdown_terminal_nodes: Vec<u32>,
    terminal_chunk_size: usize,
    terminal_blocker_neighbors_buffer: wgpu::Buffer,
    terminal_blocker_neighbor_stride: usize,
    terminal_prefix_pair_budget: usize,
    terminal_prefix_pairs_buffer: wgpu::Buffer,
    hero_decision_aggregates_buffer: wgpu::Buffer,
    villain_decision_aggregates_buffer: wgpu::Buffer,
    fused_public_infoset_mask_buffer: wgpu::Buffer,
    split_public_infosets: Vec<u32>,
    materializes_dense_outputs: bool,
}

struct GpuPublicTreeLayerTileBuffers {
    node_start: usize,
    node_end: usize,
    node_buffer: wgpu::Buffer,
    child_buffer: wgpu::Buffer,
    decision_node_buffer: wgpu::Buffer,
    decision_node_count: usize,
    fold_terminal_nodes: Vec<u32>,
    showdown_terminal_groups: Vec<GpuTerminalGroupCache>,
    hero_reaches_buffer: wgpu::Buffer,
    villain_reaches_buffer: wgpu::Buffer,
    combo_live_buffer: wgpu::Buffer,
    hero_values_buffer: wgpu::Buffer,
    villain_values_buffer: wgpu::Buffer,
}

struct GpuPublicTreeOutputBuffers {
    action_values: wgpu::Buffer,
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
    split_edges: Vec<GpuPublicTreeEdge>,
    complete_decision_groups: Vec<GpuPublicTreeEdgeGroup>,
    groups: Vec<GpuPublicTreeEdgeGroup>,
}

#[derive(Debug, Clone)]
struct GpuPublicTreeLayerTilePair {
    parent_layer: usize,
    child_layer: usize,
    parent_tile: GpuPublicTreeLayerTile,
    child_tile: GpuPublicTreeLayerTile,
}

#[derive(Debug, Clone)]
struct GpuPublicTreeLayered {
    layers: Vec<GpuPublicTreeLayer>,
    max_layer_nodes: usize,
    node_tile_size: usize,
    max_layer_tiles: usize,
    reach_edge_tiles: Vec<GpuPublicTreeLayerEdgeTile>,
    backup_tile_pairs: Vec<GpuPublicTreeLayerTilePair>,
}

struct GpuPublicTreeLayerReachBuffers {
    edges: wgpu::Buffer,
    split_edges: wgpu::Buffer,
    complete_decision_groups: wgpu::Buffer,
    groups: wgpu::Buffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GpuPublicTreeFusedUpdateParams {
    combo_count: u32,
    group_count: u32,
    max_actions: u32,
    output_len: u32,
    variant: u32,
    public_infoset_base: u32,
    iteration: u32,
    eta_bits: u32,
    alpha_bits: u32,
    gamma_bits: u32,
    beta_bits: u32,
    avg_delay: u32,
    avg_power_bits: u32,
}

unsafe impl bytemuck::Zeroable for GpuPublicTreeFusedUpdateParams {}
unsafe impl bytemuck::Pod for GpuPublicTreeFusedUpdateParams {}

struct GpuTerminalGroupCache {
    board_count: usize,
    terminal_count: usize,
    table_count: usize,
    strength_group_count_sum: usize,
    terminal_strength_group_count_sum: usize,
    strength_group_count_max: usize,
    terminal_refs_buffer: wgpu::Buffer,
    combo_order_buffer: wgpu::Buffer,
    combo_bounds_buffer: wgpu::Buffer,
}

struct GpuTerminalGroupData {
    board_count: usize,
    table_count: usize,
    strength_group_count_sum: usize,
    terminal_strength_group_count_sum: usize,
    strength_group_count_max: usize,
    terminal_refs: Vec<GpuTerminalRef>,
    combo_order: Vec<u32>,
    combo_bounds: Vec<u32>,
}

impl GpuPublicTreeIterationContext {
    fn compact_private_action_slots(&self) -> usize {
        self.public_action_offsets.last().copied().unwrap_or(0) as usize * self.combos_len
    }

    fn public_action_offsets_buffer(&self) -> &wgpu::Buffer {
        &self.public_action_offsets_buffer
    }
}

fn pack_showdown_bounds(bounds: GpuShowdownComboBounds) -> u32 {
    debug_assert!(bounds.group_start < 4096);
    debug_assert!(bounds.group_end < 4096);
    (bounds.group_start & 0x0fff)
        | ((bounds.group_end & 0x0fff) << 12)
        | ((bounds.legal & 0x0001) << 24)
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
    _pad2: u32,
    _pad3: u32,
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

fn terminal_group_data(
    nodes: &[GpuPublicTreeNode],
    terminal_nodes: &[u32],
    combos: &[GpuPrivateCombo],
    showdown_boards: &[GpuFinalBoard],
) -> Vec<GpuTerminalGroupData> {
    const MAX_TERMINAL_GROUP_TABLE_BYTES: usize = 124 * 1024 * 1024;
    let collect_group_stats = std::env::var_os("POKEDR_GPU_TERMINAL_GROUP_TRACE").is_some();

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

        let table_bytes_per_unique_board = board_count
            .saturating_mul(combos.len())
            .saturating_mul(std::mem::size_of::<u32>() + std::mem::size_of::<u32>());
        let max_unique_tables_per_group =
            (MAX_TERMINAL_GROUP_TABLE_BYTES / table_bytes_per_unique_board).max(1);

        let mut index = 0usize;
        while index < terminal_nodes.len() {
            let mut table_slots = BTreeMap::<usize, u32>::new();
            let mut terminal_refs = Vec::new();
            let mut combo_order = Vec::new();
            let mut combo_bounds = Vec::new();
            let mut strength_group_count_sum = 0usize;
            let mut terminal_strength_group_count_sum = 0usize;
            let mut strength_group_count_max = 0usize;
            let mut table_strength_group_counts = Vec::<usize>::new();
            while index < terminal_nodes.len() {
                let node_index = terminal_nodes[index];
                let node = nodes[node_index as usize];
                let board_base = node.showdown_offset as usize;
                let existing_slot = table_slots.get(&board_base).copied();
                if existing_slot.is_none()
                    && table_slots.len() >= max_unique_tables_per_group
                    && !terminal_refs.is_empty()
                {
                    break;
                }
                let table = match existing_slot {
                    Some(table) => table,
                    None => {
                        let table = table_slots.len() as u32;
                        let boards = &showdown_boards[board_base..board_base + board_count];
                        let (node_combo_order, node_combo_bounds) =
                            showdown_strength_order_data(combos, boards);
                        if collect_group_stats {
                            let (sum, max) = showdown_strength_group_stats(
                                &node_combo_bounds,
                                board_count,
                                combos.len(),
                            );
                            strength_group_count_sum += sum;
                            strength_group_count_max = strength_group_count_max.max(max);
                            table_strength_group_counts.push(sum);
                        } else {
                            table_strength_group_counts.push(0);
                        }
                        combo_order.extend(node_combo_order);
                        combo_bounds
                            .extend(node_combo_bounds.into_iter().map(pack_showdown_bounds));
                        table_slots.insert(board_base, table);
                        table
                    }
                };
                if collect_group_stats {
                    terminal_strength_group_count_sum +=
                        table_strength_group_counts[table as usize];
                }
                terminal_refs.push(GpuTerminalRef {
                    node: node_index,
                    table,
                });
                index += 1;
            }
            caches.push(GpuTerminalGroupData {
                board_count,
                table_count: table_slots.len(),
                strength_group_count_sum,
                terminal_strength_group_count_sum,
                strength_group_count_max,
                terminal_refs,
                combo_order,
                combo_bounds,
            });
        }
    }
    caches
}

fn showdown_strength_group_stats(
    bounds: &[GpuShowdownComboBounds],
    board_count: usize,
    combo_count: usize,
) -> (usize, usize) {
    let mut sum = 0usize;
    let mut max = 0usize;
    let mut seen = vec![false; combo_count];
    for board in 0..board_count {
        seen.fill(false);
        let board_bounds = &bounds[board * combo_count..(board + 1) * combo_count];
        for bound in board_bounds {
            if bound.legal == 0 {
                continue;
            }
            seen[bound.group_start as usize] = true;
        }
        let count = seen.iter().filter(|&&value| value).count();
        sum += count;
        max = max.max(count);
    }
    (sum, max)
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
                    storage_entry(9, true),
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
        let public_tree_compact_reach_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree compact reach shader"),
                source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_COMPACT_REACH_SHADER.into()),
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
                    storage_entry(8, true),
                    storage_entry(9, false),
                    storage_entry(10, false),
                    storage_entry(11, false),
                    storage_entry(12, true),
                    uniform_entry(13),
                ],
            });
        let public_tree_compact_reach_edge_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("public tree compact reach edge bind group layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, true),
                    storage_entry(5, true),
                    storage_entry(6, true),
                    storage_entry(7, true),
                    storage_entry(8, true),
                    storage_entry(9, false),
                    storage_entry(10, false),
                    storage_entry(11, false),
                    storage_entry(12, true),
                    uniform_entry(13),
                    storage_entry(14, true),
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
        let public_tree_compact_reach_edge_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("public tree compact reach edge pipeline layout"),
                bind_group_layouts: &[Some(&public_tree_compact_reach_edge_bind_group_layout)],
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
        let public_tree_compact_reach_edge_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree compact reach edge pipeline"),
                layout: Some(&public_tree_compact_reach_edge_pipeline_layout),
                module: &public_tree_compact_reach_shader,
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
        let public_tree_layer_compact_backup_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree layer compact backup shader"),
                source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_LAYER_COMPACT_BACKUP_SHADER.into()),
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
                    storage_entry(10, true),
                    storage_entry(11, true),
                    uniform_entry(12),
                ],
            });
        let public_tree_layer_compact_backup_child_bind_group_layout = device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("public tree layer compact backup child bind group layout"),
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
                    storage_entry(10, true),
                    storage_entry(11, true),
                    uniform_entry(12),
                    storage_entry(13, true),
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
        let public_tree_layer_compact_backup_child_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("public tree layer compact backup child pipeline layout"),
                bind_group_layouts: &[Some(
                    &public_tree_layer_compact_backup_child_bind_group_layout,
                )],
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
        let public_tree_layer_compact_backup_child_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree layer compact backup child pipeline"),
                layout: Some(&public_tree_layer_compact_backup_child_pipeline_layout),
                module: &public_tree_layer_compact_backup_shader,
                entry_point: Some("backup_child_tile"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let public_tree_layer_output_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree layer output shader"),
                source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_LAYER_OUTPUT_SHADER.into()),
            });
        let public_tree_layer_fused_update_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree layer fused update shader"),
                source: wgpu::ShaderSource::Wgsl(PUBLIC_TREE_LAYER_FUSED_UPDATE_SHADER.into()),
            });
        let public_tree_layer_compact_fused_update_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("public tree layer compact fused update shader"),
                source: wgpu::ShaderSource::Wgsl(
                    PUBLIC_TREE_LAYER_COMPACT_FUSED_UPDATE_SHADER.into(),
                ),
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
                    storage_entry(11, true),
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
        let public_tree_layer_fused_update_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("public tree layer fused update bind group layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, true),
                    storage_entry(5, true),
                    storage_entry(6, true),
                    storage_entry(7, true),
                    storage_entry(8, true),
                    storage_entry(9, true),
                    storage_entry(10, true),
                    storage_entry(11, true),
                    storage_entry(12, false),
                    storage_entry(13, false),
                    storage_entry(14, true),
                    storage_entry(15, false),
                    uniform_entry(16),
                ],
            });
        let public_tree_layer_fused_update_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("public tree layer fused update pipeline layout"),
                bind_group_layouts: &[Some(&public_tree_layer_fused_update_bind_group_layout)],
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
        let public_tree_layer_fused_update_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree layer fused update pipeline"),
                layout: Some(&public_tree_layer_fused_update_pipeline_layout),
                module: &public_tree_layer_fused_update_shader,
                entry_point: Some("fused_update_tile"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let public_tree_layer_compact_fused_update_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("public tree layer compact fused update pipeline"),
                layout: Some(&public_tree_layer_fused_update_pipeline_layout),
                module: &public_tree_layer_compact_fused_update_shader,
                entry_point: Some("fused_update_tile"),
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
            public_tree_compact_reach_edge_pipeline,
            public_tree_layer_reach_init_bind_group_layout,
            public_tree_layer_reach_edge_bind_group_layout,
            public_tree_compact_reach_edge_bind_group_layout,
            public_tree_terminal_partial_pipeline,
            public_tree_terminal_partial_bind_group_layout,
            public_tree_terminal_reduce_pipeline,
            public_tree_terminal_reduce_bind_group_layout,
            public_tree_fold_aggregate_pipeline,
            public_tree_fold_aggregate_bind_group_layout,
            public_tree_fold_value_pipeline,
            public_tree_fold_value_bind_group_layout,
            public_tree_layer_backup_init_pipeline,
            public_tree_layer_backup_child_pipeline,
            public_tree_layer_compact_backup_child_pipeline,
            public_tree_layer_backup_init_bind_group_layout,
            public_tree_layer_backup_child_bind_group_layout,
            public_tree_layer_compact_backup_child_bind_group_layout,
            public_tree_layer_decision_aggregate_pipeline,
            public_tree_layer_denominator_pipeline,
            public_tree_layer_action_edge_pipeline,
            public_tree_layer_fused_update_pipeline,
            public_tree_layer_compact_fused_update_pipeline,
            public_tree_layer_output_bind_group_layout,
            public_tree_layer_fused_update_bind_group_layout,
        })
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    pub fn max_storage_buffer_binding_size(&self) -> u64 {
        self.device.limits().max_storage_buffer_binding_size
    }

    pub fn max_buffer_size(&self) -> u64 {
        self.device.limits().max_buffer_size
    }

    pub fn adapter_features(&self) -> wgpu::Features {
        self.adapter_features
    }

    pub fn supports_shader_float32_atomic(&self) -> bool {
        self.adapter_features
            .contains(wgpu::Features::SHADER_FLOAT32_ATOMIC)
    }

    pub fn has_compact_reach_pipeline(&self) -> bool {
        let _ = &self.public_tree_compact_reach_edge_pipeline;
        let _ = &self.public_tree_compact_reach_edge_bind_group_layout;
        true
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
        self.queue.submit(Some(encoder.finish()));
        self.profile_poll()?;
        if let Some(start) = start {
            eprintln!(
                "pokedr: gpu profile phase={} elapsed_ms={:.3}",
                phase,
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
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
        self.profile_poll()?;
        if let Some(start) = start {
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
            variant_prediction_eta(state.variant, iteration).to_bits(),
            super::average_strategy_delay() as u32,
            super::average_strategy_power().to_bits(),
            variant_dcfr_beta(state.variant, iteration).to_bits(),
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
                _pad2: 0,
                _pad3: 0,
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
                _pad2: 0,
                _pad3: 0,
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

    #[allow(clippy::too_many_arguments)]
    // Default exact terminal CFV path. It uses rank-prefix sums plus explicit
    // blocker correction and remains the validation baseline for approximate
    // terminal evaluators.
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
        let submit_batch = std::env::var("POKEDR_GPU_TERMINAL_SUBMIT_BATCH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(64)
            .max(1);
        let stage_profile = std::env::var_os("POKEDR_GPU_TERMINAL_STAGE_PROFILE").is_some();
        let mut partial_elapsed = Duration::ZERO;
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
                    node_count: group.terminal_count as u32,
                    max_actions: group.board_count as u32,
                    output_len: (combo_count + 1) as u32,
                    pair_start: combo_count as u32,
                    chunk_pairs: 0,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                    _pad3: 0,
                }],
            );
            let partial_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("public tree streamed terminal partial bind group"),
                layout: &self.public_tree_terminal_partial_bind_group_layout,
                entries: &[
                    bind_entry(0, node_buffer),
                    bind_entry(1, &group.terminal_refs_buffer),
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
                    node_count: group.terminal_count as u32,
                    max_actions: group.board_count as u32,
                    output_len: (combo_count + 1) as u32,
                    pair_start: blocker_neighbor_stride as u32,
                    chunk_pairs: 0,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                    _pad3: 0,
                }],
            );
            let reduce_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("public tree streamed terminal reduce bind group"),
                layout: &self.public_tree_terminal_reduce_bind_group_layout,
                entries: &[
                    bind_entry(0, node_buffer),
                    bind_entry(1, &group.terminal_refs_buffer),
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
            let prefix_chunk_size = max_terminal_prefix_pairs / prefix_pairs_per_terminal;
            let terminal_chunk_size = prefix_chunk_size.max(1).min(group.terminal_count.max(1));
            for terminal_start in (0..group.terminal_count).step_by(terminal_chunk_size) {
                let terminal_count = terminal_chunk_size.min(group.terminal_count - terminal_start);
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
                        pass.set_pipeline(&self.public_tree_terminal_reduce_pipeline);
                        pass.set_bind_group(0, &reduce_bind_group, &[]);
                        pass.set_immediates(0, bytemuck::bytes_of(&reduce_chunk_params));
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

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("public tree streamed terminal reduce pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.public_tree_terminal_reduce_pipeline);
                    pass.set_bind_group(0, &reduce_bind_group, &[]);
                    pass.set_immediates(0, bytemuck::bytes_of(&reduce_chunk_params));
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
        combo_live_masks: &[Vec<u32>],
        include_terminal_work: bool,
    ) -> Vec<Vec<GpuPublicTreeLayerTileBuffers>> {
        layered
            .layers
            .iter()
            .enumerate()
            .map(|(layer_index, layer)| {
                (0..layer.nodes.len())
                    .step_by(layered.node_tile_size)
                    .map(|node_start| {
                        let node_end = (node_start + layered.node_tile_size).min(layer.nodes.len());
                        let mut tile_nodes = Vec::with_capacity(node_end - node_start);
                        let mut tile_children = Vec::new();
                        let mut tile_child_cards = Vec::new();
                        let mut decision_nodes = Vec::new();
                        let mut fold_terminal_nodes = Vec::new();
                        let mut showdown_terminal_nodes = Vec::new();

                        for source_slot in node_start..node_end {
                            let local_slot = (source_slot - node_start) as u32;
                            let source = layer.nodes[source_slot];
                            if source.kind == 0 {
                                decision_nodes.push(GpuPublicTreeEdge {
                                    parent: local_slot,
                                    child: 0,
                                    action: 0,
                                    card: u32::MAX,
                                });
                            }
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
                        let tile_combo_live = tile_combo_live_words(
                            combo_live_masks
                                .get(layer_index)
                                .map(Vec::as_slice)
                                .unwrap_or(&[]),
                            node_start,
                            node_end,
                            combos.len(),
                        );
                        let showdown_terminal_group_data = if include_terminal_work {
                            terminal_group_data(
                                &tile_nodes,
                                &showdown_terminal_nodes,
                                combos,
                                showdown_boards,
                            )
                        } else {
                            Vec::new()
                        };
                        let node_buffer = readonly_buffer(
                            &self.device,
                            "public tree layer tile nodes",
                            &tile_nodes,
                        );
                        let child_buffer = readonly_buffer(
                            &self.device,
                            "public tree layer tile children",
                            &tile_children,
                        );
                        let decision_node_buffer = readonly_buffer(
                            &self.device,
                            "public tree layer tile decision nodes",
                            if decision_nodes.is_empty() {
                                &[GpuPublicTreeEdge {
                                    parent: 0,
                                    child: 0,
                                    action: 0,
                                    card: u32::MAX,
                                }]
                            } else {
                                &decision_nodes
                            },
                        );
                        let hero_reaches_buffer = uninit_storage_buffer(
                            &self.device,
                            "public tree layer tile hero reaches",
                            value_len,
                            false,
                        );
                        let villain_reaches_buffer = uninit_storage_buffer(
                            &self.device,
                            "public tree layer tile villain reaches",
                            value_len,
                            false,
                        );
                        let combo_live_buffer = readonly_buffer(
                            &self.device,
                            "public tree layer tile combo live mask",
                            &tile_combo_live,
                        );
                        let value_buffer_len = if include_terminal_work { value_len } else { 1 };
                        let hero_values_buffer = uninit_storage_buffer(
                            &self.device,
                            "public tree layer tile hero values",
                            value_buffer_len,
                            true,
                        );
                        let villain_values_buffer = uninit_storage_buffer(
                            &self.device,
                            "public tree layer tile villain values",
                            value_buffer_len,
                            true,
                        );
                        let showdown_terminal_groups = showdown_terminal_group_data
                            .into_iter()
                            .map(|group| {
                                let terminal_count = group.terminal_refs.len();
                                GpuTerminalGroupCache {
                                    board_count: group.board_count,
                                    terminal_count,
                                    table_count: group.table_count,
                                    strength_group_count_sum: group.strength_group_count_sum,
                                    terminal_strength_group_count_sum: group
                                        .terminal_strength_group_count_sum,
                                    strength_group_count_max: group.strength_group_count_max,
                                    terminal_refs_buffer: readonly_buffer(
                                        &self.device,
                                        "public tree resident terminal refs",
                                        &group.terminal_refs,
                                    ),
                                    combo_order_buffer: readonly_buffer(
                                        &self.device,
                                        "public tree resident terminal combo strength order",
                                        &group.combo_order,
                                    ),
                                    combo_bounds_buffer: readonly_buffer(
                                        &self.device,
                                        "public tree resident terminal combo strength bounds",
                                        &group.combo_bounds,
                                    ),
                                }
                            })
                            .collect();

                        GpuPublicTreeLayerTileBuffers {
                            node_start,
                            node_end,
                            node_buffer,
                            child_buffer,
                            decision_node_buffer,
                            decision_node_count: decision_nodes.len(),
                            fold_terminal_nodes,
                            showdown_terminal_groups,
                            hero_reaches_buffer,
                            villain_reaches_buffer,
                            combo_live_buffer,
                            hero_values_buffer,
                            villain_values_buffer,
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
        br_player: u32,
        iteration: usize,
    ) {
        self.propagate_layer_reach_inits(encoder, ctx, variant, iteration);
        self.propagate_layer_reach_edges(
            encoder,
            ctx,
            regrets_buffer,
            prediction_buffer,
            variant,
            br_player,
            iteration,
        );
    }

    fn propagate_layer_reach_inits(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &GpuPublicTreeIterationContext,
        variant: super::CfrVariant,
        iteration: usize,
    ) {
        let full_init = std::env::var_os("POKEDR_GPU_FULL_REACH_INIT").is_some();
        for (layer_index, layer_tiles) in ctx.layer_tiles.iter().enumerate() {
            for (tile_index, tile) in layer_tiles.iter().enumerate() {
                if !full_init && !(layer_index == 0 && tile_index == 0) {
                    continue;
                }
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
                        _pad1: variant_prediction_eta(variant, iteration).to_bits(),
                        _pad2: 0,
                        _pad3: 0,
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
    }

    fn propagate_layer_reach_edges(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &GpuPublicTreeIterationContext,
        regrets_buffer: &wgpu::Buffer,
        prediction_buffer: &wgpu::Buffer,
        variant: super::CfrVariant,
        br_player: u32,
        iteration: usize,
    ) {
        for (edge_tile, reach_buffers) in ctx
            .layered
            .reach_edge_tiles
            .iter()
            .zip(&ctx.reach_edge_buffers)
        {
            let parent_tile_index = edge_tile.parent_tile.node_start / ctx.layered.node_tile_size;
            let child_tile_index = edge_tile.child_tile.node_start / ctx.layered.node_tile_size;
            let parent_tile = &ctx.layer_tiles[edge_tile.parent_layer][parent_tile_index];
            let child_tile = &ctx.layer_tiles[edge_tile.child_layer][child_tile_index];
            let parent_layer_nodes = &ctx.layered.layers[edge_tile.parent_layer].nodes
                [parent_tile.node_start..parent_tile.node_end];
            // Chance-only edge tiles still have to propagate reach/live state. They do not need
            // strategy buffers, so bind a one-float placeholder instead of skipping the tile.
            let public_range = public_infoset_range_for_edges(parent_layer_nodes, &edge_tile.edges);
            let (public_base, public_end) = public_range.unwrap_or((0, 0));
            let (action_base, action_len) = if public_range.is_some() {
                let infoset_base = public_base * ctx.combos_len;
                let infoset_len = (public_end - public_base) * ctx.combos_len;
                (infoset_base * ctx.actions, infoset_len * ctx.actions)
            } else {
                (0, 0)
            };
            let invocation_count = edge_tile.groups.len() * ctx.combos_len;
            let (x_groups, y_groups, x_invocations) = dispatch_grid(invocation_count);
            let params = uniform_buffer(
                &self.device,
                "public tree layer reach edge params",
                &[GpuPublicTreeParams {
                    combo_count: ctx.combos_len as u32,
                    node_count: public_base as u32,
                    max_actions: ctx.actions as u32,
                    output_len: x_invocations,
                    pair_start: variant_code(variant),
                    chunk_pairs: edge_tile.groups.len() as u32,
                    _pad0: br_player,
                    _pad1: variant_prediction_eta(variant, iteration).to_bits(),
                    _pad2: 0,
                    _pad3: 0,
                }],
            );
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("public tree layer reach edge bind group"),
                layout: &self.public_tree_layer_reach_edge_bind_group_layout,
                entries: &[
                    bind_entry(0, &parent_tile.node_buffer),
                    bind_entry(1, &reach_buffers.edges),
                    bind_entry(2, &reach_buffers.groups),
                    bind_entry(3, &ctx.combo_buffer),
                    bind_entry(4, &ctx.root_weights_buffer),
                    if action_len > 0 {
                        bind_entry_range(
                            5,
                            regrets_buffer,
                            f32_range_byte_offset(action_base),
                            f32_range_byte_size(action_len),
                        )
                    } else {
                        bind_entry(5, &ctx.empty_storage_buffer)
                    },
                    bind_entry(6, &parent_tile.hero_reaches_buffer),
                    bind_entry(7, &parent_tile.villain_reaches_buffer),
                    bind_entry(8, &parent_tile.combo_live_buffer),
                    bind_entry(9, &child_tile.hero_reaches_buffer),
                    bind_entry(10, &child_tile.villain_reaches_buffer),
                    bind_entry(11, &child_tile.combo_live_buffer),
                    if action_len > 0 {
                        bind_entry_range(
                            12,
                            prediction_buffer,
                            f32_range_byte_offset(action_base),
                            f32_range_byte_size(action_len),
                        )
                    } else {
                        bind_entry(12, &ctx.empty_storage_buffer)
                    },
                    bind_entry(13, &params),
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

    fn submit_compact_layer_reach_edges_batched(
        &self,
        ctx: &GpuPublicTreeIterationContext,
        state: &GpuCompactPrivateCfrState,
        br_player: u32,
        iteration: usize,
    ) -> usize {
        const SUBMIT_BATCH_SLICES: usize = 64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compact public tree reach edge smoke encoder"),
            });
        let mut dispatch_slices = 0usize;
        let mut pending_slices = 0usize;
        for edge_tile in &ctx.layered.reach_edge_tiles {
            let parent_tile_index = edge_tile.parent_tile.node_start / ctx.layered.node_tile_size;
            let child_tile_index = edge_tile.child_tile.node_start / ctx.layered.node_tile_size;
            let parent_tile = &ctx.layer_tiles[edge_tile.parent_layer][parent_tile_index];
            let child_tile = &ctx.layer_tiles[edge_tile.child_layer][child_tile_index];
            let parent_layer_nodes = &ctx.layered.layers[edge_tile.parent_layer].nodes
                [parent_tile.node_start..parent_tile.node_end];
            let slices = compact_reach_slices_for_tile(edge_tile, parent_layer_nodes, state);
            for slice in slices {
                if slice.groups.is_empty() {
                    continue;
                }
                let (regrets_buffer, prediction_buffer, public_action_base) =
                    if let Some(chunk_index) = slice.chunk_index {
                        let chunk = &state.chunks[chunk_index];
                        (
                            &chunk.regrets,
                            chunk
                                .prediction
                                .as_ref()
                                .unwrap_or(&ctx.empty_storage_buffer),
                            chunk.chunk.public_action_start as u32,
                        )
                    } else {
                        (&ctx.empty_storage_buffer, &ctx.empty_storage_buffer, 0)
                    };
                let edge_buffer = readonly_buffer(
                    &self.device,
                    "public tree compact reach sliced edges",
                    &slice.edges,
                );
                let group_buffer = readonly_buffer(
                    &self.device,
                    "public tree compact reach sliced groups",
                    &slice.groups,
                );
                let invocation_count = slice.groups.len() * ctx.combos_len;
                let (x_groups, y_groups, x_invocations) = dispatch_grid(invocation_count);
                let params = uniform_buffer(
                    &self.device,
                    "public tree compact reach edge params",
                    &[GpuPublicTreeParams {
                        combo_count: ctx.combos_len as u32,
                        node_count: 0,
                        max_actions: ctx.actions as u32,
                        output_len: x_invocations,
                        pair_start: variant_code(state.variant),
                        chunk_pairs: slice.groups.len() as u32,
                        _pad0: br_player,
                        _pad1: variant_prediction_eta(state.variant, iteration).to_bits(),
                        _pad2: public_action_base,
                        _pad3: 0,
                    }],
                );
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("public tree compact reach edge bind group"),
                    layout: &self.public_tree_compact_reach_edge_bind_group_layout,
                    entries: &[
                        bind_entry(0, &parent_tile.node_buffer),
                        bind_entry(1, &edge_buffer),
                        bind_entry(2, &group_buffer),
                        bind_entry(3, &ctx.combo_buffer),
                        bind_entry(4, &ctx.root_weights_buffer),
                        bind_entry(5, regrets_buffer),
                        bind_entry(6, &parent_tile.hero_reaches_buffer),
                        bind_entry(7, &parent_tile.villain_reaches_buffer),
                        bind_entry(8, &parent_tile.combo_live_buffer),
                        bind_entry(9, &child_tile.hero_reaches_buffer),
                        bind_entry(10, &child_tile.villain_reaches_buffer),
                        bind_entry(11, &child_tile.combo_live_buffer),
                        bind_entry(12, prediction_buffer),
                        bind_entry(13, &params),
                        bind_entry(14, ctx.public_action_offsets_buffer()),
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("public tree compact reach edge pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.public_tree_compact_reach_edge_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(x_groups, y_groups, 1);
                }
                dispatch_slices += 1;
                pending_slices += 1;
                if pending_slices >= SUBMIT_BATCH_SLICES {
                    self.queue.submit(Some(encoder.finish()));
                    encoder = self
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("compact public tree reach edge smoke encoder"),
                        });
                    pending_slices = 0;
                }
            }
        }
        if pending_slices > 0 {
            self.queue.submit(Some(encoder.finish()));
        }
        dispatch_slices
    }

    fn backup_layer_values(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &GpuPublicTreeIterationContext,
        regrets_buffer: &wgpu::Buffer,
        prediction_buffer: &wgpu::Buffer,
        variant: super::CfrVariant,
        br_player: u32,
        iteration: usize,
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
                        _pad2: 0,
                        _pad3: 0,
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

                for tile_pair in ctx.layered.backup_tile_pairs.iter().filter(|tile_pair| {
                    tile_pair.parent_layer == parent_layer_index
                        && tile_pair.parent_tile.node_start == parent_tile.node_start
                }) {
                    debug_assert_eq!(tile_pair.child_layer, child_layer_index);
                    let child_tile_index =
                        tile_pair.child_tile.node_start / ctx.layered.node_tile_size;
                    let child_tile = &ctx.layer_tiles[child_layer_index][child_tile_index];
                    let parent_layer_nodes = &ctx.layered.layers[parent_layer_index].nodes
                        [parent_tile.node_start..parent_tile.node_end];
                    let public_range = public_infoset_range_for_nodes(parent_layer_nodes);
                    let max_binding_f32 =
                        (self.device.limits().max_storage_buffer_binding_size as usize / 4).max(1);
                    let max_chunk_publics =
                        (max_binding_f32 / (ctx.combos_len.max(1) * ctx.actions.max(1))).max(1);
                    let max_chunk_publics = (max_chunk_publics / 32).max(1) * 32;
                    let mut include_chance = true;
                    if let Some((public_base, public_end)) = public_range {
                        for chunk_public_base in
                            (public_base..public_end).step_by(max_chunk_publics)
                        {
                            let chunk_public_end =
                                (chunk_public_base + max_chunk_publics).min(public_end);
                            let infoset_base = chunk_public_base * ctx.combos_len;
                            let infoset_len =
                                (chunk_public_end - chunk_public_base) * ctx.combos_len;
                            let action_base = infoset_base * ctx.actions;
                            let action_len = infoset_len * ctx.actions;
                            let flags = br_player
                                | (variant_code(variant) << 8)
                                | ((include_chance as u32) << 16);
                            include_chance = false;
                            let params = uniform_buffer(
                                &self.device,
                                "public tree layer backup child params",
                                &[GpuPublicTreeParams {
                                    combo_count: ctx.combos_len as u32,
                                    node_count: (parent_tile.node_end - parent_tile.node_start)
                                        as u32,
                                    max_actions: ctx.actions as u32,
                                    output_len: x_invocations,
                                    pair_start: flags,
                                    chunk_pairs: child_tile.node_start as u32,
                                    _pad0: child_tile.node_end as u32,
                                    _pad1: chunk_public_end as u32,
                                    _pad2: chunk_public_base as u32,
                                    _pad3: variant_prediction_eta(variant, iteration).to_bits(),
                                }],
                            );
                            let bind_group =
                                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                                        bind_entry_range(
                                            10,
                                            regrets_buffer,
                                            f32_range_byte_offset(action_base),
                                            f32_range_byte_size(action_len),
                                        ),
                                        bind_entry_range(
                                            11,
                                            prediction_buffer,
                                            f32_range_byte_offset(action_base),
                                            f32_range_byte_size(action_len),
                                        ),
                                        bind_entry(12, &params),
                                    ],
                                });
                            let mut pass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("public tree layer backup child pass"),
                                    timestamp_writes: None,
                                });
                            pass.set_pipeline(&self.public_tree_layer_backup_child_pipeline);
                            pass.set_bind_group(0, &bind_group, &[]);
                            pass.dispatch_workgroups(x_groups, y_groups, 1);
                        }
                    } else {
                        let flags = br_player | (variant_code(variant) << 8) | (1u32 << 16);
                        let params = uniform_buffer(
                            &self.device,
                            "public tree layer backup child params",
                            &[GpuPublicTreeParams {
                                combo_count: ctx.combos_len as u32,
                                node_count: (parent_tile.node_end - parent_tile.node_start) as u32,
                                max_actions: ctx.actions as u32,
                                output_len: x_invocations,
                                pair_start: flags,
                                chunk_pairs: child_tile.node_start as u32,
                                _pad0: child_tile.node_end as u32,
                                _pad1: 0,
                                _pad2: 0,
                                _pad3: variant_prediction_eta(variant, iteration).to_bits(),
                            }],
                        );
                        let bind_group =
                            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                                    bind_entry(10, &ctx.empty_storage_buffer),
                                    bind_entry(11, &ctx.empty_storage_buffer),
                                    bind_entry(12, &params),
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
    }

    fn backup_layer_values_compact(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &GpuPublicTreeIterationContext,
        state: &GpuCompactPrivateCfrState,
        br_player: u32,
        iteration: usize,
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
                    "public tree compact backup init params",
                    &[GpuPublicTreeParams {
                        combo_count: ctx.combos_len as u32,
                        node_count: (parent_tile.node_end - parent_tile.node_start) as u32,
                        max_actions: ctx.actions as u32,
                        output_len: x_invocations,
                        pair_start: br_player,
                        chunk_pairs: 0,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                        _pad3: 0,
                    }],
                );
                let init_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("public tree compact backup init bind group"),
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
                        label: Some("public tree compact backup init pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.public_tree_layer_backup_init_pipeline);
                    pass.set_bind_group(0, &init_bind_group, &[]);
                    pass.dispatch_workgroups(x_groups, y_groups, 1);
                }

                for tile_pair in ctx.layered.backup_tile_pairs.iter().filter(|tile_pair| {
                    tile_pair.parent_layer == parent_layer_index
                        && tile_pair.parent_tile.node_start == parent_tile.node_start
                }) {
                    debug_assert_eq!(tile_pair.child_layer, child_layer_index);
                    let child_tile_index =
                        tile_pair.child_tile.node_start / ctx.layered.node_tile_size;
                    let child_tile = &ctx.layer_tiles[child_layer_index][child_tile_index];
                    let parent_layer_nodes = &ctx.layered.layers[parent_layer_index].nodes
                        [parent_tile.node_start..parent_tile.node_end];
                    let public_range = public_infoset_range_for_nodes(parent_layer_nodes);
                    let mut include_chance = true;
                    if let Some((public_base, public_end)) = public_range {
                        for chunk in state.chunks.iter().filter(|chunk| {
                            chunk.chunk.public_start < public_end
                                && public_base < chunk.chunk.public_end
                        }) {
                            let chunk_public_base = public_base.max(chunk.chunk.public_start);
                            let chunk_public_end = public_end.min(chunk.chunk.public_end);
                            let flags = br_player
                                | (variant_code(state.variant) << 8)
                                | ((include_chance as u32) << 16);
                            include_chance = false;
                            let params = uniform_buffer(
                                &self.device,
                                "public tree compact backup child params",
                                &[GpuPublicTreeParams {
                                    combo_count: ctx.combos_len as u32,
                                    node_count: (parent_tile.node_end - parent_tile.node_start)
                                        as u32,
                                    max_actions: chunk.chunk.public_action_start as u32,
                                    output_len: x_invocations,
                                    pair_start: flags,
                                    chunk_pairs: child_tile.node_start as u32,
                                    _pad0: child_tile.node_end as u32,
                                    _pad1: chunk_public_end as u32,
                                    _pad2: chunk_public_base as u32,
                                    _pad3: variant_prediction_eta(state.variant, iteration)
                                        .to_bits(),
                                }],
                            );
                            let prediction = chunk
                                .prediction
                                .as_ref()
                                .unwrap_or(&ctx.empty_storage_buffer);
                            let bind_group =
                                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("public tree compact backup child bind group"),
                                    layout: &self
                                        .public_tree_layer_compact_backup_child_bind_group_layout,
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
                                        bind_entry(10, &chunk.regrets),
                                        bind_entry(11, prediction),
                                        bind_entry(12, &params),
                                        bind_entry(13, &ctx.public_action_offsets_buffer),
                                    ],
                                });
                            let mut pass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("public tree compact backup child pass"),
                                    timestamp_writes: None,
                                });
                            pass.set_pipeline(
                                &self.public_tree_layer_compact_backup_child_pipeline,
                            );
                            pass.set_bind_group(0, &bind_group, &[]);
                            pass.dispatch_workgroups(x_groups, y_groups, 1);
                        }
                    } else {
                        let flags = br_player | (variant_code(state.variant) << 8) | (1u32 << 16);
                        let params = uniform_buffer(
                            &self.device,
                            "public tree compact backup child params",
                            &[GpuPublicTreeParams {
                                combo_count: ctx.combos_len as u32,
                                node_count: (parent_tile.node_end - parent_tile.node_start) as u32,
                                max_actions: 0,
                                output_len: x_invocations,
                                pair_start: flags,
                                chunk_pairs: child_tile.node_start as u32,
                                _pad0: child_tile.node_end as u32,
                                _pad1: 0,
                                _pad2: 0,
                                _pad3: variant_prediction_eta(state.variant, iteration).to_bits(),
                            }],
                        );
                        let bind_group =
                            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("public tree compact backup child bind group"),
                                layout: &self
                                    .public_tree_layer_compact_backup_child_bind_group_layout,
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
                                    bind_entry(10, &ctx.empty_storage_buffer),
                                    bind_entry(11, &ctx.empty_storage_buffer),
                                    bind_entry(12, &params),
                                    bind_entry(13, &ctx.public_action_offsets_buffer),
                                ],
                            });
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("public tree compact backup child pass"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.public_tree_layer_compact_backup_child_pipeline);
                        pass.set_bind_group(0, &bind_group, &[]);
                        pass.dispatch_workgroups(x_groups, y_groups, 1);
                    }
                }
            }
        }
    }

    fn write_layer_outputs(
        &self,
        mut encoder: wgpu::CommandEncoder,
        ctx: &GpuPublicTreeIterationContext,
        split_only_action_edges: bool,
    ) -> Result<wgpu::CommandEncoder, GpuCfrError> {
        let stage_profile = Self::gpu_profile_enabled()
            && std::env::var_os("POKEDR_GPU_OUTPUT_STAGE_PROFILE").is_some();
        let mut stage_start = stage_profile.then(Instant::now);
        for (layer_index, layer_tiles) in ctx.layer_tiles.iter().enumerate() {
            for tile in layer_tiles {
                let tile_nodes =
                    &ctx.layered.layers[layer_index].nodes[tile.node_start..tile.node_end];
                let Some((public_base, _public_end)) = public_infoset_range_for_nodes(tile_nodes)
                else {
                    continue;
                };
                let decision_invocations = tile.decision_node_count * 53usize;
                if decision_invocations == 0 {
                    continue;
                }
                let (x_groups, y_groups, x_invocations) = dispatch_grid(decision_invocations);
                let params = uniform_buffer(
                    &self.device,
                    "public tree layer decision aggregate params",
                    &[GpuPublicTreeParams {
                        combo_count: ctx.combos_len as u32,
                        node_count: tile.decision_node_count as u32,
                        max_actions: ctx.actions as u32,
                        output_len: x_invocations,
                        pair_start: 0,
                        chunk_pairs: public_base as u32,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                        _pad3: 0,
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
                        bind_entry(6, &ctx.empty_storage_buffer),
                        bind_entry(7, &tile.combo_live_buffer),
                        bind_entry(8, &tile.decision_node_buffer),
                        bind_entry(9, &tile.hero_values_buffer),
                        bind_entry(10, &tile.villain_values_buffer),
                        bind_entry(11, &ctx.root_weights_buffer),
                        bind_entry(12, &ctx.empty_storage_buffer),
                        bind_entry(13, &ctx.empty_storage_buffer),
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
        if stage_profile {
            encoder =
                self.finish_profile_phase(encoder, "cfv_output_decision_aggregate", stage_start)?;
            stage_start = Some(Instant::now());
        }
        if split_only_action_edges && ctx.split_public_infosets.is_empty() {
            return Ok(encoder);
        }

        for (layer_index, layer_tiles) in ctx.layer_tiles.iter().enumerate() {
            for tile in layer_tiles {
                let tile_nodes =
                    &ctx.layered.layers[layer_index].nodes[tile.node_start..tile.node_end];
                let Some((public_base, public_end)) = public_infoset_range_for_nodes(tile_nodes)
                else {
                    continue;
                };
                let invocations = tile.decision_node_count * ctx.combos_len;
                if invocations == 0 {
                    continue;
                }
                let (x_groups, y_groups, x_invocations) = dispatch_grid(invocations);
                let max_binding_f32 =
                    (self.device.limits().max_storage_buffer_binding_size as usize / 4).max(1);
                let max_chunk_publics = (max_binding_f32 / ctx.combos_len.max(1)).max(1);
                let max_chunk_publics = (max_chunk_publics / 32).max(1) * 32;
                for chunk_public_base in (public_base..public_end).step_by(max_chunk_publics) {
                    let chunk_public_end = (chunk_public_base + max_chunk_publics).min(public_end);
                    let chunk_infoset_base = chunk_public_base * ctx.combos_len;
                    let chunk_infoset_len = (chunk_public_end - chunk_public_base) * ctx.combos_len;
                    let params = uniform_buffer(
                        &self.device,
                        "public tree layer denominator params",
                        &[GpuPublicTreeParams {
                            combo_count: ctx.combos_len as u32,
                            node_count: tile.decision_node_count as u32,
                            max_actions: ctx.actions as u32,
                            output_len: x_invocations,
                            pair_start: 0,
                            chunk_pairs: chunk_public_base as u32,
                            _pad0: chunk_public_end as u32,
                            _pad1: 0,
                            _pad2: 0,
                            _pad3: 0,
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
                            bind_entry(6, &ctx.empty_storage_buffer),
                            bind_entry(7, &tile.combo_live_buffer),
                            bind_entry(8, &tile.decision_node_buffer),
                            bind_entry(9, &tile.hero_values_buffer),
                            bind_entry(10, &tile.villain_values_buffer),
                            bind_entry(11, &ctx.root_weights_buffer),
                            bind_entry_range(
                                12,
                                &ctx.reach_weights_buffer,
                                f32_range_byte_offset(chunk_infoset_base),
                                f32_range_byte_size(chunk_infoset_len),
                            ),
                            bind_entry(13, &ctx.empty_storage_buffer),
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
        if stage_profile {
            encoder = self.finish_profile_phase(encoder, "cfv_output_denominator", stage_start)?;
            stage_start = Some(Instant::now());
        }

        for (layer_index, layer_tiles) in ctx.layer_tiles.iter().enumerate() {
            for tile in layer_tiles {
                let tile_nodes =
                    &ctx.layered.layers[layer_index].nodes[tile.node_start..tile.node_end];
                let Some((public_base, _public_end)) = public_infoset_range_for_nodes(tile_nodes)
                else {
                    continue;
                };
                let decision_invocations = tile.decision_node_count * 53usize;
                if decision_invocations == 0 {
                    continue;
                }
                let (x_groups, y_groups, x_invocations) = dispatch_grid(decision_invocations);
                let params = uniform_buffer(
                    &self.device,
                    "public tree layer strategy aggregate params",
                    &[GpuPublicTreeParams {
                        combo_count: ctx.combos_len as u32,
                        node_count: tile.decision_node_count as u32,
                        max_actions: ctx.actions as u32,
                        output_len: x_invocations,
                        pair_start: 2,
                        chunk_pairs: public_base as u32,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                        _pad3: 0,
                    }],
                );
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("public tree layer strategy aggregate bind group"),
                    layout: &self.public_tree_layer_output_bind_group_layout,
                    entries: &[
                        bind_entry(0, &tile.node_buffer),
                        bind_entry(1, &ctx.combo_buffer),
                        bind_entry(2, &tile.hero_reaches_buffer),
                        bind_entry(3, &tile.villain_reaches_buffer),
                        bind_entry(4, &ctx.hero_decision_aggregates_buffer),
                        bind_entry(5, &ctx.villain_decision_aggregates_buffer),
                        bind_entry(6, &ctx.empty_storage_buffer),
                        bind_entry(7, &tile.combo_live_buffer),
                        bind_entry(8, &tile.decision_node_buffer),
                        bind_entry(9, &tile.hero_values_buffer),
                        bind_entry(10, &tile.villain_values_buffer),
                        bind_entry(11, &ctx.root_weights_buffer),
                        bind_entry(12, &ctx.empty_storage_buffer),
                        bind_entry(13, &ctx.empty_storage_buffer),
                        bind_entry(14, &params),
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("public tree layer strategy aggregate pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.public_tree_layer_decision_aggregate_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(x_groups, y_groups, 1);
            }
        }
        if stage_profile {
            encoder =
                self.finish_profile_phase(encoder, "cfv_output_strategy_aggregate", stage_start)?;
            stage_start = Some(Instant::now());
        }

        for (edge_tile, reach_buffers) in ctx
            .layered
            .reach_edge_tiles
            .iter()
            .zip(&ctx.reach_edge_buffers)
        {
            let parent_tile_index = edge_tile.parent_tile.node_start / ctx.layered.node_tile_size;
            let child_tile_index = edge_tile.child_tile.node_start / ctx.layered.node_tile_size;
            let parent_tile = &ctx.layer_tiles[edge_tile.parent_layer][parent_tile_index];
            let child_tile = &ctx.layer_tiles[edge_tile.child_layer][child_tile_index];
            let parent_layer_nodes = &ctx.layered.layers[edge_tile.parent_layer].nodes
                [parent_tile.node_start..parent_tile.node_end];
            let Some((public_base, public_end)) =
                public_infoset_range_for_edges(parent_layer_nodes, &edge_tile.edges)
            else {
                continue;
            };
            let infoset_base = public_base * ctx.combos_len;
            let infoset_len = (public_end - public_base) * ctx.combos_len;
            let action_base = infoset_base * ctx.actions;
            let action_len = infoset_len * ctx.actions;
            let action_edge_count = if split_only_action_edges {
                edge_tile.split_edges.len()
            } else {
                edge_tile.edges.len()
            };
            if action_edge_count == 0 {
                continue;
            }
            let invocations = action_edge_count * ctx.combos_len;
            let (x_groups, y_groups, x_invocations) = dispatch_grid(invocations);
            let params = uniform_buffer(
                &self.device,
                "public tree layer action edge params",
                &[GpuPublicTreeParams {
                    combo_count: ctx.combos_len as u32,
                    node_count: action_edge_count as u32,
                    max_actions: ctx.actions as u32,
                    output_len: x_invocations,
                    pair_start: 0,
                    chunk_pairs: public_base as u32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                    _pad3: 0,
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
                    bind_entry_range(
                        6,
                        &ctx.action_values_buffer,
                        f32_range_byte_offset(action_base),
                        f32_range_byte_size(action_len),
                    ),
                    bind_entry(7, &parent_tile.combo_live_buffer),
                    bind_entry(
                        8,
                        if split_only_action_edges {
                            &reach_buffers.split_edges
                        } else {
                            &reach_buffers.edges
                        },
                    ),
                    bind_entry(9, &child_tile.hero_values_buffer),
                    bind_entry(10, &child_tile.villain_values_buffer),
                    bind_entry(11, &ctx.root_weights_buffer),
                    bind_entry_range(
                        12,
                        &ctx.reach_weights_buffer,
                        f32_range_byte_offset(infoset_base),
                        f32_range_byte_size(infoset_len),
                    ),
                    bind_entry_range(
                        13,
                        &ctx.strategy_weights_buffer,
                        f32_range_byte_offset(infoset_base),
                        f32_range_byte_size(infoset_len),
                    ),
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
        if stage_profile {
            encoder = self.finish_profile_phase(encoder, "cfv_output_action_edge", stage_start)?;
        }
        Ok(encoder)
    }

    fn write_compact_update_aggregates(
        &self,
        mut encoder: wgpu::CommandEncoder,
        ctx: &GpuPublicTreeIterationContext,
    ) -> wgpu::CommandEncoder {
        for (layer_index, layer_tiles) in ctx.layer_tiles.iter().enumerate() {
            for tile in layer_tiles {
                let tile_nodes =
                    &ctx.layered.layers[layer_index].nodes[tile.node_start..tile.node_end];
                let Some((public_base, _public_end)) = public_infoset_range_for_nodes(tile_nodes)
                else {
                    continue;
                };
                let decision_invocations = tile.decision_node_count * 53usize;
                if decision_invocations == 0 {
                    continue;
                }
                let (x_groups, y_groups, x_invocations) = dispatch_grid(decision_invocations);
                let params = uniform_buffer(
                    &self.device,
                    "public tree compact update aggregate params",
                    &[GpuPublicTreeParams {
                        combo_count: ctx.combos_len as u32,
                        node_count: tile.decision_node_count as u32,
                        max_actions: ctx.actions as u32,
                        output_len: x_invocations,
                        pair_start: 2,
                        chunk_pairs: public_base as u32,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                        _pad3: 0,
                    }],
                );
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("public tree compact update aggregate bind group"),
                    layout: &self.public_tree_layer_output_bind_group_layout,
                    entries: &[
                        bind_entry(0, &tile.node_buffer),
                        bind_entry(1, &ctx.combo_buffer),
                        bind_entry(2, &tile.hero_reaches_buffer),
                        bind_entry(3, &tile.villain_reaches_buffer),
                        bind_entry(4, &ctx.hero_decision_aggregates_buffer),
                        bind_entry(5, &ctx.villain_decision_aggregates_buffer),
                        bind_entry(6, &ctx.empty_storage_buffer),
                        bind_entry(7, &tile.combo_live_buffer),
                        bind_entry(8, &tile.decision_node_buffer),
                        bind_entry(9, &tile.hero_values_buffer),
                        bind_entry(10, &tile.villain_values_buffer),
                        bind_entry(11, &ctx.root_weights_buffer),
                        bind_entry(12, &ctx.empty_storage_buffer),
                        bind_entry(13, &ctx.empty_storage_buffer),
                        bind_entry(14, &params),
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("public tree compact update aggregate pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.public_tree_layer_decision_aggregate_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(x_groups, y_groups, 1);
            }
        }
        encoder
    }

    #[allow(clippy::too_many_arguments)]
    fn public_tree_iteration_context(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        hero_weights: &[f32],
        villain_weights: &[f32],
        showdown_boards: &[GpuFinalBoard],
        infosets: usize,
        actions: usize,
        materialize_dense_outputs: bool,
        include_terminal_work: bool,
    ) -> GpuPublicTreeIterationContext {
        assert!(!nodes.is_empty());
        assert_eq!(combo_legal.len(), combos.len());
        assert_eq!(hero_weights.len(), combos.len());
        assert_eq!(villain_weights.len(), combos.len());
        assert_eq!(infosets, nodes_public_infoset_count(nodes) * combos.len());

        let action_len = infosets * actions;
        let output_len = action_len * 2 + infosets * 2;
        let node_combo_len = nodes.len() * combos.len();
        let mut layered = public_tree_layered(
            nodes,
            children,
            child_cards,
            combos.len(),
            self.device.limits().max_storage_buffer_binding_size,
        );
        if std::env::var_os("POKEDR_GPU_LAYER_TRACE").is_some() {
            eprintln!(
                "pokedr: gpu public tree layers={} max_layer_nodes={} node_tile_size={} max_layer_tiles={} reach_edge_tiles={} backup_tile_pairs={} max_layer_node_combos={} full_node_combos={}",
                layered.layers.len(),
                layered.max_layer_nodes,
                layered.node_tile_size,
                layered.max_layer_tiles,
                layered.reach_edge_tiles.len(),
                layered.backup_tile_pairs.len(),
                layered.max_layer_nodes * combos.len(),
                node_combo_len,
            );
        }
        let (fused_public_infoset_mask, split_public_infosets) =
            public_tree_partition_fused_public_infosets(
                &mut layered,
                nodes_public_infoset_count(nodes),
            );
        let public_action_offsets = public_action_offsets_from_nodes(nodes);

        let combo_buffer = readonly_buffer(&self.device, "public tree combos", combos);
        let mut root_weights = Vec::with_capacity(combos.len() * 2);
        root_weights.extend(
            combo_legal
                .iter()
                .zip(hero_weights)
                .map(|(is_legal, weight)| if *is_legal != 0 { *weight } else { -1.0 }),
        );
        root_weights.extend(
            combo_legal
                .iter()
                .zip(villain_weights)
                .map(|(is_legal, weight)| if *is_legal != 0 { *weight } else { -1.0 }),
        );
        let root_weights_buffer = readonly_buffer(
            &self.device,
            "public tree root reach weights",
            &root_weights,
        );
        let public_action_offsets_buffer = readonly_buffer(
            &self.device,
            "public tree public action offsets",
            &public_action_offsets,
        );
        let dense_action_output_len = if materialize_dense_outputs {
            action_len
        } else {
            1
        };
        let dense_infoset_output_len = if materialize_dense_outputs {
            infosets
        } else {
            1
        };
        let action_values_buffer = uninit_storage_buffer(
            &self.device,
            "public tree action values output",
            dense_action_output_len,
            true,
        );
        let reach_weights_buffer = uninit_storage_buffer(
            &self.device,
            "public tree reach weights output",
            dense_infoset_output_len,
            true,
        );
        let strategy_weights_buffer = uninit_storage_buffer(
            &self.device,
            "public tree strategy weights output",
            dense_infoset_output_len,
            true,
        );
        let empty_storage_buffer = uninit_storage_buffer(
            &self.device,
            "public tree empty storage placeholder",
            1,
            false,
        );
        let fold_terminal_nodes: Vec<_> = if include_terminal_work {
            nodes
                .iter()
                .enumerate()
                .filter_map(|(index, node)| {
                    (node.kind != 0 && node.kind != 1 && node.terminal_kind != 2)
                        .then_some(index as u32)
                })
                .collect()
        } else {
            Vec::new()
        };
        let showdown_terminal_nodes: Vec<_> = if include_terminal_work {
            nodes
                .iter()
                .enumerate()
                .filter_map(|(index, node)| {
                    (node.kind != 0 && node.kind != 1 && node.terminal_kind == 2)
                        .then_some(index as u32)
                })
                .collect()
        } else {
            Vec::new()
        };
        let public_infoset_count = nodes_public_infoset_count(nodes);
        let max_showdown_boards = showdown_terminal_nodes
            .iter()
            .map(|node_index| nodes[*node_index as usize]._pad0 as usize)
            .max()
            .unwrap_or(1)
            .max(1);
        let (blocker_neighbors, blocker_neighbor_stride) = if include_terminal_work {
            showdown_blocker_neighbors(combos)
        } else {
            (vec![0u32], 1)
        };
        let terminal_blocker_neighbors_buffer = readonly_buffer(
            &self.device,
            "public tree terminal blocker neighbors",
            &blocker_neighbors,
        );
        let default_max_terminal_prefix_pairs = 8_000_000usize;
        let max_terminal_prefix_pairs = if include_terminal_work {
            std::env::var("POKEDR_GPU_MAX_TERMINAL_PREFIX_PAIRS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(default_max_terminal_prefix_pairs)
                .max(max_showdown_boards * (combos.len() + 1))
        } else {
            1
        };
        let terminal_chunk_size =
            (max_terminal_prefix_pairs / (max_showdown_boards * (combos.len() + 1))).max(1);
        let terminal_prefix_pairs_buffer = uninit_storage_buffer(
            &self.device,
            "public tree streamed terminal prefix pairs scratch",
            max_terminal_prefix_pairs * 2,
            false,
        );
        const DECISION_AGGREGATE_SLOTS: usize = 53;
        let decision_aggregate_len = if include_terminal_work {
            public_infoset_count * DECISION_AGGREGATE_SLOTS
        } else {
            1
        };
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
        let reach_edge_buffers = layered
            .reach_edge_tiles
            .iter()
            .map(|edge_tile| GpuPublicTreeLayerReachBuffers {
                edges: readonly_buffer(
                    &self.device,
                    "public tree layer resident edges",
                    &edge_tile.edges,
                ),
                split_edges: readonly_buffer(
                    &self.device,
                    "public tree layer resident split edges",
                    if edge_tile.split_edges.is_empty() {
                        &[GpuPublicTreeEdge {
                            parent: 0,
                            child: 0,
                            action: 0,
                            card: u32::MAX,
                        }]
                    } else {
                        &edge_tile.split_edges
                    },
                ),
                complete_decision_groups: readonly_buffer(
                    &self.device,
                    "public tree layer resident complete decision groups",
                    if edge_tile.complete_decision_groups.is_empty() {
                        &[GpuPublicTreeEdgeGroup {
                            parent: 0,
                            first_edge: 0,
                            edge_count: 0,
                            _pad0: 0,
                        }]
                    } else {
                        &edge_tile.complete_decision_groups
                    },
                ),
                groups: readonly_buffer(
                    &self.device,
                    "public tree layer resident edge groups",
                    &edge_tile.groups,
                ),
            })
            .collect();
        let combo_live_masks = public_tree_static_combo_live_masks(&layered, combos, combo_legal);
        let layer_tiles = self.public_tree_layer_tile_buffers(
            &layered,
            combos,
            showdown_boards,
            &combo_live_masks,
            include_terminal_work,
        );
        if std::env::var_os("POKEDR_GPU_TERMINAL_GROUP_TRACE").is_some() {
            let mut groups_by_board_count = BTreeMap::<usize, (usize, usize)>::new();
            let mut tables_by_board_count = BTreeMap::<usize, usize>::new();
            let mut total_groups = 0usize;
            let mut total_terminals = 0usize;
            let mut total_tables = 0usize;
            let mut total_prefix_rows = 0usize;
            let mut total_table_rows = 0usize;
            let mut total_reduce_lanes = 0usize;
            let mut total_strength_groups = 0usize;
            let mut total_terminal_strength_groups = 0usize;
            let mut max_strength_groups = 0usize;
            for layer in &layer_tiles {
                for tile in layer {
                    for group in &tile.showdown_terminal_groups {
                        let entry = groups_by_board_count
                            .entry(group.board_count)
                            .or_insert((0, 0));
                        entry.0 += 1;
                        entry.1 += group.terminal_count;
                        *tables_by_board_count.entry(group.board_count).or_insert(0) +=
                            group.table_count;
                        total_groups += 1;
                        total_terminals += group.terminal_count;
                        total_tables += group.table_count;
                        total_prefix_rows += group.terminal_count * group.board_count;
                        total_table_rows += group.table_count * group.board_count;
                        total_reduce_lanes += group.terminal_count * combos.len();
                        total_strength_groups += group.strength_group_count_sum;
                        total_terminal_strength_groups += group.terminal_strength_group_count_sum;
                        max_strength_groups =
                            max_strength_groups.max(group.strength_group_count_max);
                    }
                }
            }
            let static_card_prefix_cells = total_strength_groups * (Card::COUNT + 1);
            let terminal_card_prefix_cells = total_terminal_strength_groups * (Card::COUNT + 1);
            eprintln!(
                "pokedr: gpu terminal groups summary groups={} terminals={} unique_tables={} terminals_per_table={:.2} prefix_rows={} static_table_rows={} prefix_row_reuse={:.2} reduce_lanes={} strength_groups={} terminal_strength_groups={} max_strength_groups_per_board={} static_card_prefix_cells={} terminal_card_prefix_cells={} terminal_card_prefix_bytes_f32_pair={}",
                total_groups,
                total_terminals,
                total_tables,
                total_terminals as f64 / total_tables.max(1) as f64,
                total_prefix_rows,
                total_table_rows,
                total_prefix_rows as f64 / total_table_rows.max(1) as f64,
                total_reduce_lanes,
                total_strength_groups,
                total_terminal_strength_groups,
                max_strength_groups,
                static_card_prefix_cells,
                terminal_card_prefix_cells,
                terminal_card_prefix_cells * std::mem::size_of::<[f32; 2]>()
            );
            eprintln!("pokedr: gpu terminal groups by board_count:");
            for (board_count, (group_count, terminal_count)) in groups_by_board_count {
                let table_count = tables_by_board_count
                    .get(&board_count)
                    .copied()
                    .unwrap_or(0);
                eprintln!(
                    "pokedr: gpu terminal group board_count={} groups={} tables={} terminals={} terminals_per_table={:.2}",
                    board_count,
                    group_count,
                    table_count,
                    terminal_count,
                    terminal_count as f64 / table_count.max(1) as f64
                );
            }
        }
        let fused_public_infoset_mask_buffer = readonly_buffer(
            &self.device,
            "public tree fused public infoset mask",
            &fused_public_infoset_mask,
        );

        GpuPublicTreeIterationContext {
            nodes_len: nodes.len(),
            combos_len: combos.len(),
            actions,
            public_action_offsets,
            action_len,
            output_len,
            node_combo_len,
            layered,
            combo_buffer,
            root_weights_buffer,
            public_action_offsets_buffer,
            action_values_buffer,
            reach_weights_buffer,
            strategy_weights_buffer,
            empty_storage_buffer,
            layer_tiles,
            reach_edge_buffers,
            fold_terminal_nodes,
            showdown_terminal_nodes,
            terminal_chunk_size,
            terminal_blocker_neighbors_buffer,
            terminal_blocker_neighbor_stride: blocker_neighbor_stride,
            terminal_prefix_pair_budget: max_terminal_prefix_pairs,
            terminal_prefix_pairs_buffer,
            hero_decision_aggregates_buffer,
            villain_decision_aggregates_buffer,
            fused_public_infoset_mask_buffer,
            split_public_infosets,
            materializes_dense_outputs: materialize_dense_outputs,
        }
    }

    fn public_tree_iteration_output_with_context(
        &self,
        ctx: &GpuPublicTreeIterationContext,
        regrets_buffer: &wgpu::Buffer,
        prediction_buffer: &wgpu::Buffer,
        variant: super::CfrVariant,
        br_player: u32,
        iteration: usize,
        split_only_action_edges: bool,
    ) -> Result<(GpuPublicTreeOutputBuffers, usize, usize), GpuCfrError> {
        assert!(
            ctx.materializes_dense_outputs,
            "dense output materialization is disabled for compact public-tree context"
        );
        if std::env::var_os("POKEDR_SOLVER_PROGRESS_OFF").is_none() {
            eprintln!(
                "pokedr: gpu public tree cfv nodes={} combos={} node_combo_values={} folds={} showdowns={} terminal_chunk={}",
                ctx.nodes_len,
                ctx.combos_len,
                ctx.node_combo_len,
                ctx.fold_terminal_nodes.len(),
                ctx.showdown_terminal_nodes.len(),
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
        if profile && std::env::var_os("POKEDR_GPU_REACH_PROFILE").is_some() {
            let split_trace = std::env::var_os("POKEDR_GPU_SPLIT_TRACE").is_some();
            let mut decision_edges = 0usize;
            let mut chance_edges = 0usize;
            let mut chance_only_tiles = 0usize;
            let mut complete_decision_groups = 0usize;
            let mut complete_decision_edges = 0usize;
            let mut split_decision_groups = 0usize;
            let mut split_decision_edges = 0usize;
            for edge_tile in &ctx.layered.reach_edge_tiles {
                let parent = &ctx.layered.layers[edge_tile.parent_layer];
                let mut tile_decision_edges = 0usize;
                let mut tile_chance_edges = 0usize;
                for edge in &edge_tile.edges {
                    let node =
                        parent.nodes[edge_tile.parent_tile.node_start + edge.parent as usize];
                    if node.kind == 0 {
                        tile_decision_edges += 1;
                    } else if node.kind == 1 {
                        tile_chance_edges += 1;
                    }
                }
                for group in &edge_tile.groups {
                    let node =
                        parent.nodes[edge_tile.parent_tile.node_start + group.parent as usize];
                    if node.kind == 0 {
                        if group.edge_count == node.child_count {
                            complete_decision_groups += 1;
                            complete_decision_edges += group.edge_count as usize;
                        } else {
                            split_decision_groups += 1;
                            split_decision_edges += group.edge_count as usize;
                            if split_trace {
                                let start = group.first_edge as usize;
                                let end = start + group.edge_count as usize;
                                let actions: Vec<u32> = edge_tile.edges[start..end]
                                    .iter()
                                    .map(|edge| edge.action)
                                    .collect();
                                eprintln!(
                                    "pokedr: gpu split group parent_layer={} child_layer={} parent_tile_start={} child_tile_start={} parent_local={} public_infoset={} child_count={} edge_count={} actions={:?}",
                                    edge_tile.parent_layer,
                                    edge_tile.child_layer,
                                    edge_tile.parent_tile.node_start,
                                    edge_tile.child_tile.node_start,
                                    group.parent,
                                    node.public_infoset,
                                    node.child_count,
                                    group.edge_count,
                                    actions,
                                );
                            }
                        }
                    }
                }
                if tile_decision_edges == 0 && tile_chance_edges > 0 {
                    chance_only_tiles += 1;
                }
                decision_edges += tile_decision_edges;
                chance_edges += tile_chance_edges;
            }
            eprintln!(
                "pokedr: gpu profile reach edge_tiles={} chance_only_tiles={} decision_edges={} chance_edges={} complete_decision_groups={} complete_decision_edges={} split_decision_groups={} split_decision_edges={}",
                ctx.layered.reach_edge_tiles.len(),
                chance_only_tiles,
                decision_edges,
                chance_edges,
                complete_decision_groups,
                complete_decision_edges,
                split_decision_groups,
                split_decision_edges,
            );
            self.propagate_layer_reach_inits(&mut encoder, ctx, variant, iteration);
            encoder = self.finish_profile_phase(encoder, "cfv_reach_init", phase_start)?;
            phase_start = profile.then(Instant::now);
            self.propagate_layer_reach_edges(
                &mut encoder,
                ctx,
                regrets_buffer,
                prediction_buffer,
                variant,
                br_player,
                iteration,
            );
            encoder = self.finish_profile_phase(encoder, "cfv_reach_edges", phase_start)?;
        } else {
            self.propagate_layer_reaches(
                &mut encoder,
                ctx,
                regrets_buffer,
                prediction_buffer,
                variant,
                br_player,
                iteration,
            );
            encoder = self.finish_profile_phase(encoder, "cfv_reach", phase_start)?;
        }
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
        self.profile_poll()?;
        for layer_tiles in &ctx.layer_tiles {
            for tile in layer_tiles {
                self.fill_terminal_values_streaming(
                    &tile.node_buffer,
                    &tile.showdown_terminal_groups,
                    &ctx.terminal_blocker_neighbors_buffer,
                    &tile.hero_reaches_buffer,
                    &tile.villain_reaches_buffer,
                    &tile.hero_values_buffer,
                    &tile.villain_values_buffer,
                    &ctx.terminal_prefix_pairs_buffer,
                    ctx.combos_len,
                    ctx.terminal_blocker_neighbor_stride,
                    ctx.terminal_prefix_pair_budget,
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

        self.backup_layer_values(
            &mut encoder,
            ctx,
            regrets_buffer,
            prediction_buffer,
            variant,
            br_player,
            iteration,
        );
        encoder = self.finish_profile_phase(encoder, "cfv_backup", phase_start)?;
        phase_start = profile.then(Instant::now);

        encoder = self.write_layer_outputs(encoder, ctx, split_only_action_edges)?;
        encoder = self.finish_profile_phase(encoder, "cfv_decision_denominator", phase_start)?;
        phase_start = profile.then(Instant::now);
        self.submit_final_profile_phase(encoder, "cfv_action_aggregate", phase_start)?;
        Ok((
            GpuPublicTreeOutputBuffers {
                action_values: ctx.action_values_buffer.clone(),
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
        hero_weights: &[f32],
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
            hero_weights,
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
        hero_weights: &[f32],
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
            hero_weights,
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
        hero_weights: &[f32],
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
            hero_weights,
            villain_weights,
            showdown_boards,
            state.infosets,
            state.actions,
            true,
            true,
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
                usize::MAX,
                false,
            )?;
        let action_values_readback = readback_buffer(&self.device, action_len);
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
        let reach_weights = read_f32_buffer(&self.device, &reach_weights_readback, state.infosets)?;
        for (action_index, value) in action_values.iter_mut().enumerate() {
            let infoset = action_index / state.actions;
            let reach_weight = reach_weights[infoset];
            if reach_weight > 0.0 {
                *value /= reach_weight;
            } else {
                *value = 0.0;
            }
        }
        Ok(GpuRootTerminalValues {
            action_values,
            reach_weights,
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
        hero_weights: &[f32],
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
            hero_weights,
            villain_weights,
            showdown_boards,
            state.infosets,
            state.actions,
            true,
            true,
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
                iteration,
                true,
            )?;
        if let Some(start) = cfv_start {
            self.profile_poll()?;
            eprintln!(
                "pokedr: gpu profile iteration={} phase=cfv elapsed_ms={:.3}",
                iteration,
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        let update_start = profile.then(Instant::now);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("public tree CFR update encoder"),
            });
        self.fused_complete_group_updates(&mut encoder, context, state, iteration);
        let mut split_update_ranges = Vec::with_capacity(context.split_public_infosets.len());
        for &public_infoset in &context.split_public_infosets {
            let target_start = public_infoset as usize * context.combos_len;
            if target_start >= state.infosets {
                continue;
            }
            let aligned_start = (target_start / 8) * 8;
            let target_end = (target_start + context.combos_len).min(state.infosets);
            split_update_ranges.push((aligned_start, target_end));
        }
        split_update_ranges.sort_unstable();
        let mut merged_split_update_ranges: Vec<(usize, usize)> = Vec::new();
        for (start, end) in split_update_ranges {
            if let Some((_, last_end)) = merged_split_update_ranges.last_mut() {
                if start <= *last_end {
                    *last_end = (*last_end).max(end);
                    continue;
                }
            }
            merged_split_update_ranges.push((start, end));
        }
        for (infoset_start, infoset_end) in merged_split_update_ranges {
            let infoset_count = infoset_end - infoset_start;
            if infoset_count == 0 {
                continue;
            }
            let action_start = infoset_start * state.actions;
            let action_len = infoset_count * state.actions;
            let params = readonly_buffer(
                &self.device,
                "public tree CFR update params",
                &[
                    infoset_count as u32,
                    state.actions as u32,
                    variant_code(state.variant),
                    iteration as u32,
                    variant_dcfr_alpha(state.variant, iteration).to_bits(),
                    variant_dcfr_gamma(state.variant, iteration).to_bits(),
                    variant_prediction_eta(state.variant, iteration).to_bits(),
                    super::average_strategy_delay() as u32,
                    super::average_strategy_power().to_bits(),
                    infoset_start as u32,
                    context.combos_len as u32,
                    variant_dcfr_beta(state.variant, iteration).to_bits(),
                ],
            );
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("public tree CFR update bind group"),
                layout: &self.public_tree_cfr_update_bind_group_layout,
                entries: &[
                    bind_entry_range(
                        0,
                        &state.regrets,
                        f32_range_byte_offset(action_start),
                        f32_range_byte_size(action_len),
                    ),
                    bind_entry_range(
                        1,
                        &state.strategy_sum,
                        f32_range_byte_offset(action_start),
                        f32_range_byte_size(action_len),
                    ),
                    bind_entry_range(
                        2,
                        &output_buffers.action_values,
                        f32_range_byte_offset(action_start),
                        f32_range_byte_size(action_len),
                    ),
                    bind_entry_range(3, &context.empty_storage_buffer, 0, f32_range_byte_size(1)),
                    bind_entry_range(
                        4,
                        &output_buffers.reach_weights,
                        f32_range_byte_offset(infoset_start),
                        f32_range_byte_size(infoset_count),
                    ),
                    bind_entry_range(
                        5,
                        &output_buffers.strategy_weights,
                        f32_range_byte_offset(infoset_start),
                        f32_range_byte_size(infoset_count),
                    ),
                    bind_entry(6, &params),
                    bind_entry_range(
                        7,
                        &state.legal_actions_buffer,
                        (action_start * std::mem::size_of::<u32>()) as u64,
                        (action_len.max(1) * std::mem::size_of::<u32>()) as u64,
                    ),
                    bind_entry_range(
                        8,
                        &state.prediction,
                        f32_range_byte_offset(action_start),
                        f32_range_byte_size(action_len),
                    ),
                    bind_entry(9, &context.fused_public_infoset_mask_buffer),
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("public tree CFR update pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.public_tree_cfr_update_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (infoset_count as u32).div_ceil(WORKGROUP_SIZE);
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

    fn fused_complete_group_updates(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &GpuPublicTreeIterationContext,
        state: &GpuDenseCfrState,
        iteration: usize,
    ) {
        for (edge_tile, reach_buffers) in ctx
            .layered
            .reach_edge_tiles
            .iter()
            .zip(&ctx.reach_edge_buffers)
        {
            if edge_tile.complete_decision_groups.is_empty() {
                continue;
            }
            let parent_tile_index = edge_tile.parent_tile.node_start / ctx.layered.node_tile_size;
            let child_tile_index = edge_tile.child_tile.node_start / ctx.layered.node_tile_size;
            let parent_tile = &ctx.layer_tiles[edge_tile.parent_layer][parent_tile_index];
            let child_tile = &ctx.layer_tiles[edge_tile.child_layer][child_tile_index];
            let parent_layer_nodes = &ctx.layered.layers[edge_tile.parent_layer].nodes
                [parent_tile.node_start..parent_tile.node_end];
            let Some((public_base, public_end)) =
                public_infoset_range_for_edges(parent_layer_nodes, &edge_tile.edges)
            else {
                continue;
            };
            let infoset_base = public_base * ctx.combos_len;
            let infoset_len = (public_end - public_base) * ctx.combos_len;
            let action_base = infoset_base * ctx.actions;
            let action_len = infoset_len * ctx.actions;
            let invocation_count = edge_tile.complete_decision_groups.len() * ctx.combos_len;
            if invocation_count == 0 {
                continue;
            }
            let (x_groups, y_groups, x_invocations) = dispatch_grid(invocation_count);
            let params = uniform_buffer(
                &self.device,
                "public tree fused update params",
                &[GpuPublicTreeFusedUpdateParams {
                    combo_count: ctx.combos_len as u32,
                    group_count: edge_tile.complete_decision_groups.len() as u32,
                    max_actions: ctx.actions as u32,
                    output_len: x_invocations,
                    variant: variant_code(state.variant),
                    public_infoset_base: public_base as u32,
                    iteration: iteration as u32,
                    eta_bits: variant_prediction_eta(state.variant, iteration).to_bits(),
                    alpha_bits: variant_dcfr_alpha(state.variant, iteration).to_bits(),
                    gamma_bits: variant_dcfr_gamma(state.variant, iteration).to_bits(),
                    beta_bits: variant_dcfr_beta(state.variant, iteration).to_bits(),
                    avg_delay: super::average_strategy_delay() as u32,
                    avg_power_bits: super::average_strategy_power().to_bits(),
                }],
            );
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("public tree fused update bind group"),
                layout: &self.public_tree_layer_fused_update_bind_group_layout,
                entries: &[
                    bind_entry(0, &parent_tile.node_buffer),
                    bind_entry(1, &ctx.combo_buffer),
                    bind_entry(2, &parent_tile.hero_reaches_buffer),
                    bind_entry(3, &parent_tile.villain_reaches_buffer),
                    bind_entry(4, &ctx.hero_decision_aggregates_buffer),
                    bind_entry(5, &ctx.villain_decision_aggregates_buffer),
                    bind_entry(6, &parent_tile.combo_live_buffer),
                    bind_entry(7, &reach_buffers.edges),
                    bind_entry(8, &reach_buffers.complete_decision_groups),
                    bind_entry(9, &child_tile.hero_values_buffer),
                    bind_entry(10, &child_tile.villain_values_buffer),
                    bind_entry(11, &ctx.root_weights_buffer),
                    bind_entry_range(
                        12,
                        &state.regrets,
                        f32_range_byte_offset(action_base),
                        f32_range_byte_size(action_len),
                    ),
                    bind_entry_range(
                        13,
                        &state.strategy_sum,
                        f32_range_byte_offset(action_base),
                        f32_range_byte_size(action_len),
                    ),
                    bind_entry_range(
                        14,
                        &state.legal_actions_buffer,
                        (action_base * std::mem::size_of::<u32>()) as u64,
                        (action_len.max(1) * std::mem::size_of::<u32>()) as u64,
                    ),
                    bind_entry_range(
                        15,
                        &state.prediction,
                        f32_range_byte_offset(action_base),
                        f32_range_byte_size(action_len),
                    ),
                    bind_entry(16, &params),
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("public tree fused update pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.public_tree_layer_fused_update_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(x_groups, y_groups, 1);
        }
    }

    fn submit_compact_complete_group_updates_batched(
        &self,
        ctx: &GpuPublicTreeIterationContext,
        state: &GpuCompactPrivateCfrState,
        iteration: usize,
    ) -> usize {
        const SUBMIT_BATCH_SLICES: usize = 64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compact public tree update encoder"),
            });
        let mut dispatch_slices = 0usize;
        let mut pending_slices = 0usize;
        for edge_tile in &ctx.layered.reach_edge_tiles {
            let parent_tile_index = edge_tile.parent_tile.node_start / ctx.layered.node_tile_size;
            let child_tile_index = edge_tile.child_tile.node_start / ctx.layered.node_tile_size;
            let parent_tile = &ctx.layer_tiles[edge_tile.parent_layer][parent_tile_index];
            let child_tile = &ctx.layer_tiles[edge_tile.child_layer][child_tile_index];
            let parent_layer_nodes = &ctx.layered.layers[edge_tile.parent_layer].nodes
                [parent_tile.node_start..parent_tile.node_end];
            for slice in
                compact_complete_group_slices_for_tile(edge_tile, parent_layer_nodes, state)
            {
                if slice.groups.is_empty() {
                    continue;
                }
                let chunk = &state.chunks[slice.chunk_index.expect("decision group chunk")];
                let strategy_sum = chunk
                    .strategy_sum
                    .as_ref()
                    .unwrap_or(&ctx.empty_storage_buffer);
                let prediction = chunk
                    .prediction
                    .as_ref()
                    .unwrap_or(&ctx.empty_storage_buffer);
                let edge_buffer = readonly_buffer(
                    &self.device,
                    "public tree compact update sliced edges",
                    &slice.edges,
                );
                let group_buffer = readonly_buffer(
                    &self.device,
                    "public tree compact update sliced groups",
                    &slice.groups,
                );
                let invocation_count = slice.groups.len() * ctx.combos_len;
                let (x_groups, y_groups, x_invocations) = dispatch_grid(invocation_count);
                let params = uniform_buffer(
                    &self.device,
                    "public tree compact fused update params",
                    &[GpuPublicTreeFusedUpdateParams {
                        combo_count: ctx.combos_len as u32,
                        group_count: slice.groups.len() as u32,
                        max_actions: ctx.actions as u32,
                        output_len: x_invocations,
                        variant: variant_code(state.variant),
                        public_infoset_base: chunk.chunk.public_action_start as u32,
                        iteration: iteration as u32,
                        eta_bits: variant_prediction_eta(state.variant, iteration).to_bits(),
                        alpha_bits: variant_dcfr_alpha(state.variant, iteration).to_bits(),
                        gamma_bits: variant_dcfr_gamma(state.variant, iteration).to_bits(),
                        beta_bits: variant_dcfr_beta(state.variant, iteration).to_bits(),
                        avg_delay: super::average_strategy_delay() as u32,
                        avg_power_bits: super::average_strategy_power().to_bits(),
                    }],
                );
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("public tree compact fused update bind group"),
                    layout: &self.public_tree_layer_fused_update_bind_group_layout,
                    entries: &[
                        bind_entry(0, &parent_tile.node_buffer),
                        bind_entry(1, &ctx.combo_buffer),
                        bind_entry(2, &parent_tile.hero_reaches_buffer),
                        bind_entry(3, &parent_tile.villain_reaches_buffer),
                        bind_entry(4, &ctx.hero_decision_aggregates_buffer),
                        bind_entry(5, &ctx.villain_decision_aggregates_buffer),
                        bind_entry(6, &parent_tile.combo_live_buffer),
                        bind_entry(7, &edge_buffer),
                        bind_entry(8, &group_buffer),
                        bind_entry(9, &child_tile.hero_values_buffer),
                        bind_entry(10, &child_tile.villain_values_buffer),
                        bind_entry(11, &ctx.root_weights_buffer),
                        bind_entry(12, &chunk.regrets),
                        bind_entry(13, strategy_sum),
                        bind_entry(14, &ctx.public_action_offsets_buffer),
                        bind_entry(15, prediction),
                        bind_entry(16, &params),
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("public tree compact fused update pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.public_tree_layer_compact_fused_update_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(x_groups, y_groups, 1);
                }
                dispatch_slices += 1;
                pending_slices += 1;
                if pending_slices >= SUBMIT_BATCH_SLICES {
                    self.queue.submit(Some(encoder.finish()));
                    encoder = self
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("compact public tree update encoder"),
                        });
                    pending_slices = 0;
                }
            }
        }
        if pending_slices > 0 {
            self.queue.submit(Some(encoder.finish()));
        }
        dispatch_slices
    }

    #[allow(clippy::too_many_arguments)]
    pub fn public_tree_run_iterations(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        hero_weights: &[f32],
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
            hero_weights,
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
        hero_weights: &[f32],
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
            hero_weights,
            villain_weights,
            showdown_boards,
            state.infosets,
            state.actions,
            true,
            true,
        );
        if let Some(start) = setup_start {
            eprintln!(
                "pokedr: gpu profile phase=cfv_setup elapsed_ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        if std::env::var_os("POKEDR_GPU_COMPACT_TRACE").is_some() {
            let _ = context.public_action_offsets_buffer();
            eprintln!(
                "pokedr: gpu compact context public_actions={} private_action_slots={}",
                context.public_action_offsets.last().copied().unwrap_or(0),
                context.compact_private_action_slots()
            );
        }
        let flush_interval = std::env::var("POKEDR_GPU_ITERATION_FLUSH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(64)
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
        hero_weights: &[f32],
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
            hero_weights,
            villain_weights,
            showdown_boards,
            state.infosets,
            state.actions,
            true,
            true,
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
            .unwrap_or(64)
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

    pub fn zeroed_compact_private_state(
        &self,
        config: CompactPrivateCfrConfig,
        max_chunk_bytes: usize,
    ) -> GpuCompactPrivateCfrState {
        let include_prediction = matches!(config.variant, super::CfrVariant::PdcfrPlus { .. });
        self.zeroed_compact_private_state_with_buffers(
            config,
            max_chunk_bytes,
            include_prediction,
            true,
        )
    }

    pub fn zeroed_compact_private_regret_state(
        &self,
        config: CompactPrivateCfrConfig,
        max_chunk_bytes: usize,
    ) -> GpuCompactPrivateCfrState {
        self.zeroed_compact_private_state_with_buffers(config, max_chunk_bytes, false, false)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compact_public_tree_context_smoke(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        hero_weights: &[f32],
        villain_weights: &[f32],
        showdown_boards: &[GpuFinalBoard],
    ) -> usize {
        self.compact_public_tree_context_smoke_with_state(
            nodes,
            children,
            child_cards,
            combos,
            combo_legal,
            hero_weights,
            villain_weights,
            showdown_boards,
            None,
        )
        .0
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compact_public_tree_context_smoke_with_state(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        hero_weights: &[f32],
        villain_weights: &[f32],
        showdown_boards: &[GpuFinalBoard],
        state: Option<&GpuCompactPrivateCfrState>,
    ) -> (usize, usize, usize) {
        let chunks = state.map(|state| {
            state
                .chunks
                .iter()
                .map(|chunk| chunk.chunk)
                .collect::<Vec<_>>()
        });
        self.compact_public_tree_context_smoke_with_chunks(
            nodes,
            children,
            child_cards,
            combos,
            combo_legal,
            hero_weights,
            villain_weights,
            showdown_boards,
            chunks.as_deref(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compact_public_tree_context_smoke_with_chunks(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        hero_weights: &[f32],
        villain_weights: &[f32],
        showdown_boards: &[GpuFinalBoard],
        chunks: Option<&[CompactPrivateCfrChunk]>,
    ) -> (usize, usize, usize) {
        let context = self.public_tree_iteration_context(
            nodes,
            children,
            child_cards,
            combos,
            combo_legal,
            hero_weights,
            villain_weights,
            showdown_boards,
            nodes_public_infoset_count(nodes) * combos.len(),
            nodes_max_action_count(nodes),
            false,
            false,
        );
        let uncovered_reach_tiles = chunks
            .map(|chunks| compact_uncovered_reach_tiles(&context, chunks))
            .unwrap_or(0);
        (
            context.compact_private_action_slots(),
            uncovered_reach_tiles,
            context.split_public_infosets.len(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compact_public_tree_reach_smoke_with_state(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        hero_weights: &[f32],
        villain_weights: &[f32],
        showdown_boards: &[GpuFinalBoard],
        state: &GpuCompactPrivateCfrState,
        br_player: u32,
        iteration: usize,
    ) -> (usize, usize, usize) {
        let context = self.public_tree_iteration_context(
            nodes,
            children,
            child_cards,
            combos,
            combo_legal,
            hero_weights,
            villain_weights,
            showdown_boards,
            nodes_public_infoset_count(nodes) * combos.len(),
            nodes_max_action_count(nodes),
            false,
            false,
        );
        let chunks: Vec<_> = state.chunks.iter().map(|chunk| chunk.chunk).collect();
        let uncovered_reach_tiles = compact_uncovered_reach_tiles(&context, &chunks);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compact public tree reach init smoke encoder"),
            });
        self.propagate_layer_reach_inits(&mut encoder, &context, state.variant, iteration);
        self.queue.submit(Some(encoder.finish()));
        let dispatch_slices =
            self.submit_compact_layer_reach_edges_batched(&context, state, br_player, iteration);
        (
            context.compact_private_action_slots(),
            uncovered_reach_tiles,
            dispatch_slices,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compact_public_tree_reach_smoke_with_chunk_plan(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        hero_weights: &[f32],
        villain_weights: &[f32],
        showdown_boards: &[GpuFinalBoard],
        config: CompactPrivateCfrConfig,
        max_chunk_bytes: usize,
        br_player: u32,
        iteration: usize,
    ) -> (usize, usize, usize) {
        let chunks = config.chunk_by_action_bytes(max_chunk_bytes);
        let largest_chunk_slots = chunks
            .iter()
            .map(|chunk| chunk.action_slots)
            .max()
            .unwrap_or(1)
            .max(1);
        let zeros = vec![0.0; largest_chunk_slots];
        let scratch_regrets = storage_buffer(
            &self.device,
            "compact private streamed regret scratch",
            &zeros,
        );
        let state = GpuCompactPrivateCfrState {
            public_infosets: config.public_infosets(),
            public_actions: config.public_actions(),
            combos: config.combos,
            variant: config.variant,
            chunks: chunks
                .into_iter()
                .map(|chunk| GpuCompactPrivateCfrChunkState {
                    chunk,
                    regrets: scratch_regrets.clone(),
                    prediction: None,
                    strategy_sum: None,
                })
                .collect(),
        };
        self.compact_public_tree_reach_smoke_with_state(
            nodes,
            children,
            child_cards,
            combos,
            combo_legal,
            hero_weights,
            villain_weights,
            showdown_boards,
            &state,
            br_player,
            iteration,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compact_public_tree_iteration_smoke_with_chunk_plan(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        hero_weights: &[f32],
        villain_weights: &[f32],
        showdown_boards: &[GpuFinalBoard],
        config: CompactPrivateCfrConfig,
        max_chunk_bytes: usize,
        br_player: u32,
        iteration: usize,
    ) -> Result<(usize, usize, usize, usize), GpuCfrError> {
        trace_pipeline_step("compact_iteration:chunk_plan:start");
        let chunks = config.chunk_by_action_bytes(max_chunk_bytes);
        let largest_chunk_slots = chunks
            .iter()
            .map(|chunk| chunk.action_slots)
            .max()
            .unwrap_or(1)
            .max(1);
        trace_pipeline_step("compact_iteration:scratch_buffers:start");
        let zeros = vec![0.0; largest_chunk_slots];
        let scratch_regrets = storage_buffer(
            &self.device,
            "compact private streamed iteration regret scratch",
            &zeros,
        );
        let scratch_prediction = storage_buffer(
            &self.device,
            "compact private streamed iteration prediction scratch",
            &zeros,
        );
        let scratch_strategy_sum = storage_buffer(
            &self.device,
            "compact private streamed iteration strategy scratch",
            &zeros,
        );
        let state = GpuCompactPrivateCfrState {
            public_infosets: config.public_infosets(),
            public_actions: config.public_actions(),
            combos: config.combos,
            variant: config.variant,
            chunks: chunks
                .into_iter()
                .map(|chunk| GpuCompactPrivateCfrChunkState {
                    chunk,
                    regrets: scratch_regrets.clone(),
                    prediction: Some(scratch_prediction.clone()),
                    strategy_sum: Some(scratch_strategy_sum.clone()),
                })
                .collect(),
        };
        self.compact_public_tree_run_iteration_with_state(
            nodes,
            children,
            child_cards,
            combos,
            combo_legal,
            hero_weights,
            villain_weights,
            showdown_boards,
            &state,
            br_player,
            iteration,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compact_public_tree_run_iteration_with_state(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        hero_weights: &[f32],
        villain_weights: &[f32],
        showdown_boards: &[GpuFinalBoard],
        state: &GpuCompactPrivateCfrState,
        br_player: u32,
        iteration: usize,
    ) -> Result<(usize, usize, usize, usize), GpuCfrError> {
        assert!(
            state
                .chunks
                .iter()
                .all(|chunk| chunk.strategy_sum.is_some()),
            "compact public tree iteration requires strategy_sum buffers"
        );
        assert!(
            !matches!(state.variant, super::CfrVariant::PdcfrPlus { .. })
                || state.chunks.iter().all(|chunk| chunk.prediction.is_some()),
            "PDCFR+ compact public tree iteration requires prediction buffers"
        );
        trace_pipeline_step("compact_iteration:context:start");
        let context = self.public_tree_iteration_context(
            nodes,
            children,
            child_cards,
            combos,
            combo_legal,
            hero_weights,
            villain_weights,
            showdown_boards,
            nodes_public_infoset_count(nodes) * combos.len(),
            nodes_max_action_count(nodes),
            false,
            true,
        );
        trace_pipeline_step("compact_iteration:context:done");
        let chunk_plan: Vec<_> = state.chunks.iter().map(|chunk| chunk.chunk).collect();
        let uncovered_reach_tiles = compact_uncovered_reach_tiles(&context, &chunk_plan);

        trace_pipeline_step("compact_iteration:reach_init:start");
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compact public tree iteration pre-terminal encoder"),
            });
        self.propagate_layer_reach_inits(&mut encoder, &context, state.variant, iteration);
        self.queue.submit(Some(encoder.finish()));
        trace_pipeline_step("compact_iteration:reach_edges:start");
        let reach_slices =
            self.submit_compact_layer_reach_edges_batched(&context, &state, br_player, iteration);

        trace_pipeline_step("compact_iteration:fold:start");
        for layer_tiles in &context.layer_tiles {
            for tile in layer_tiles {
                if tile.fold_terminal_nodes.is_empty() {
                    continue;
                }
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("compact public tree iteration fold encoder"),
                        });
                self.fill_fold_values(
                    &mut encoder,
                    &tile.node_buffer,
                    &tile.fold_terminal_nodes,
                    &context.combo_buffer,
                    &tile.hero_reaches_buffer,
                    &tile.villain_reaches_buffer,
                    &tile.combo_live_buffer,
                    &tile.hero_values_buffer,
                    &tile.villain_values_buffer,
                    context.combos_len,
                )?;
                self.queue.submit(Some(encoder.finish()));
            }
        }
        self.profile_poll()?;
        trace_pipeline_step("compact_iteration:showdown:start");
        for layer_tiles in &context.layer_tiles {
            for tile in layer_tiles {
                self.fill_terminal_values_streaming(
                    &tile.node_buffer,
                    &tile.showdown_terminal_groups,
                    &context.terminal_blocker_neighbors_buffer,
                    &tile.hero_reaches_buffer,
                    &tile.villain_reaches_buffer,
                    &tile.hero_values_buffer,
                    &tile.villain_values_buffer,
                    &context.terminal_prefix_pairs_buffer,
                    context.combos_len,
                    context.terminal_blocker_neighbor_stride,
                    context.terminal_prefix_pair_budget,
                )?;
            }
        }

        trace_pipeline_step("compact_iteration:backup:start");
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compact public tree iteration update prep encoder"),
            });
        self.backup_layer_values_compact(&mut encoder, &context, &state, br_player, iteration);
        trace_pipeline_step("compact_iteration:aggregate:start");
        let encoder = self.write_compact_update_aggregates(encoder, &context);
        self.queue.submit(Some(encoder.finish()));
        trace_pipeline_step("compact_iteration:update:start");
        let update_slices =
            self.submit_compact_complete_group_updates_batched(&context, &state, iteration);
        trace_pipeline_step("compact_iteration:done");

        Ok((
            context.compact_private_action_slots(),
            uncovered_reach_tiles,
            reach_slices,
            update_slices,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compact_public_tree_run_iterations_with_state(
        &self,
        nodes: &[GpuPublicTreeNode],
        children: &[u32],
        child_cards: &[u32],
        combos: &[GpuPrivateCombo],
        combo_legal: &[u32],
        hero_weights: &[f32],
        villain_weights: &[f32],
        showdown_boards: &[GpuFinalBoard],
        state: &GpuCompactPrivateCfrState,
        first_iteration: usize,
        iterations: usize,
    ) -> Result<(usize, usize, usize, usize), GpuCfrError> {
        let mut last = (0, 0, 0, 0);
        for offset in 0..iterations {
            last = self.compact_public_tree_run_iteration_with_state(
                nodes,
                children,
                child_cards,
                combos,
                combo_legal,
                hero_weights,
                villain_weights,
                showdown_boards,
                state,
                2,
                first_iteration + offset,
            )?;
        }
        Ok(last)
    }

    fn zeroed_compact_private_state_with_buffers(
        &self,
        config: CompactPrivateCfrConfig,
        max_chunk_bytes: usize,
        include_prediction: bool,
        include_strategy_sum: bool,
    ) -> GpuCompactPrivateCfrState {
        config.validate();
        let chunks = config
            .chunk_by_action_bytes(max_chunk_bytes)
            .into_iter()
            .map(|chunk| {
                let zeros = vec![0.0; chunk.action_slots];
                let prediction = include_prediction
                    .then(|| storage_buffer(&self.device, "compact private prediction", &zeros));
                let strategy_sum = include_strategy_sum
                    .then(|| storage_buffer(&self.device, "compact private strategy sum", &zeros));
                GpuCompactPrivateCfrChunkState {
                    chunk,
                    regrets: storage_buffer(&self.device, "compact private regrets", &zeros),
                    prediction,
                    strategy_sum,
                }
            })
            .collect();
        GpuCompactPrivateCfrState {
            public_infosets: config.public_infosets(),
            public_actions: config.public_actions(),
            combos: config.combos,
            variant: config.variant,
            chunks,
        }
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
            variant_prediction_eta(self.variant, iteration).to_bits(),
            super::average_strategy_delay() as u32,
            super::average_strategy_power().to_bits(),
            variant_dcfr_beta(self.variant, iteration).to_bits(),
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

    pub fn download_strategy_sum_prefix(
        &self,
        backend: &GpuDenseCfrBackend,
        len: usize,
    ) -> Result<Vec<f32>, GpuCfrError> {
        assert!(len <= self.infosets * self.actions);
        let readback = readback_buffer(&backend.device, len);
        let mut encoder = backend
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident dense CFR strategy prefix download encoder"),
            });
        copy_buffer(&mut encoder, &self.strategy_sum, &readback, len);
        let submission = backend.queue.submit(Some(encoder.finish()));
        backend
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| GpuCfrError::MapFailed(error.to_string()))?;
        read_f32_buffer(&backend.device, &readback, len)
    }
}

impl GpuCompactPrivateCfrState {
    pub fn public_infosets(&self) -> usize {
        self.public_infosets
    }

    pub fn public_actions(&self) -> usize {
        self.public_actions
    }

    pub fn combos(&self) -> usize {
        self.combos
    }

    pub fn variant(&self) -> super::CfrVariant {
        self.variant
    }

    pub fn chunks(&self) -> &[GpuCompactPrivateCfrChunkState] {
        &self.chunks
    }

    pub fn total_action_slots(&self) -> usize {
        self.chunks
            .iter()
            .map(|chunk| chunk.chunk.action_slots)
            .sum()
    }
}

impl GpuCompactPrivateCfrChunkState {
    pub fn chunk(&self) -> CompactPrivateCfrChunk {
        self.chunk
    }

    pub fn regrets_buffer(&self) -> &wgpu::Buffer {
        &self.regrets
    }

    pub fn prediction_buffer(&self) -> Option<&wgpu::Buffer> {
        self.prediction.as_ref()
    }

    pub fn strategy_sum_buffer(&self) -> Option<&wgpu::Buffer> {
        self.strategy_sum.as_ref()
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

fn bind_entry_range(
    binding: u32,
    buffer: &wgpu::Buffer,
    offset: u64,
    size: u64,
) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset,
            size: NonZeroU64::new(size),
        }),
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
    uninit_storage_buffer_typed::<f32>(device, label, len, copy_src, false)
}

fn uninit_storage_buffer_typed<T>(
    device: &wgpu::Device,
    label: &str,
    len: usize,
    copy_src: bool,
    copy_dst: bool,
) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE;
    if copy_src {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    if copy_dst {
        usage |= wgpu::BufferUsages::COPY_DST;
    }
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: byte_len::<T>(len),
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
        super::CfrVariant::Dcfr { .. } => 5,
    }
}

fn variant_dcfr_alpha(variant: super::CfrVariant, iteration: usize) -> f32 {
    match variant {
        super::CfrVariant::Dcfr { alpha, .. } => alpha,
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

fn variant_dcfr_beta(variant: super::CfrVariant, _iteration: usize) -> f32 {
    match variant {
        super::CfrVariant::Dcfr { beta, .. } => beta,
        _ => super::DEFAULT_DCFR_BETA,
    }
}

fn variant_dcfr_gamma(variant: super::CfrVariant, iteration: usize) -> f32 {
    match variant {
        super::CfrVariant::Dcfr { gamma, .. } => gamma,
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

fn variant_prediction_eta(variant: super::CfrVariant, iteration: usize) -> f32 {
    match variant {
        super::CfrVariant::PdcfrPlus {
            eta_start,
            eta_end,
            eta_horizon,
            ..
        } => scheduled_value(eta_start, eta_end, iteration, eta_horizon),
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

fn nodes_max_action_count(nodes: &[GpuPublicTreeNode]) -> usize {
    nodes
        .iter()
        .filter(|node| node.kind == 0)
        .map(|node| node.child_count as usize)
        .max()
        .unwrap_or(1)
}

fn compact_uncovered_reach_tiles(
    ctx: &GpuPublicTreeIterationContext,
    chunks: &[CompactPrivateCfrChunk],
) -> usize {
    ctx.layered
        .reach_edge_tiles
        .iter()
        .filter(|edge_tile| {
            let parent_tile_index = edge_tile.parent_tile.node_start / ctx.layered.node_tile_size;
            let parent_tile = &ctx.layer_tiles[edge_tile.parent_layer][parent_tile_index];
            let parent_layer_nodes = &ctx.layered.layers[edge_tile.parent_layer].nodes
                [parent_tile.node_start..parent_tile.node_end];
            let Some((public_start, public_end)) =
                public_infoset_exact_range_for_edges(parent_layer_nodes, &edge_tile.edges)
            else {
                return false;
            };
            compact_chunk_covering_public_range(chunks, public_start, public_end).is_none()
        })
        .count()
}

struct CompactReachSlice {
    chunk_index: Option<usize>,
    edges: Vec<GpuPublicTreeEdge>,
    groups: Vec<GpuPublicTreeEdgeGroup>,
}

fn compact_reach_slices_for_tile(
    edge_tile: &GpuPublicTreeLayerEdgeTile,
    parent_layer_nodes: &[GpuPublicTreeNode],
    state: &GpuCompactPrivateCfrState,
) -> Vec<CompactReachSlice> {
    let mut by_chunk: BTreeMap<Option<usize>, CompactReachSlice> = BTreeMap::new();
    for group in &edge_tile.groups {
        let node = parent_layer_nodes[group.parent as usize];
        let chunk_index = if node.kind == 0 {
            Some(compact_chunk_index_for_public_infoset(
                &state.chunks,
                node.public_infoset as usize,
            ))
        } else {
            None
        };
        let slice = by_chunk
            .entry(chunk_index)
            .or_insert_with(|| CompactReachSlice {
                chunk_index,
                edges: Vec::new(),
                groups: Vec::new(),
            });
        let first_edge = slice.edges.len() as u32;
        let edge_start = group.first_edge as usize;
        let edge_end = edge_start + group.edge_count as usize;
        slice
            .edges
            .extend_from_slice(&edge_tile.edges[edge_start..edge_end]);
        slice.groups.push(GpuPublicTreeEdgeGroup {
            first_edge,
            ..*group
        });
    }
    by_chunk.into_values().collect()
}

fn compact_complete_group_slices_for_tile(
    edge_tile: &GpuPublicTreeLayerEdgeTile,
    parent_layer_nodes: &[GpuPublicTreeNode],
    state: &GpuCompactPrivateCfrState,
) -> Vec<CompactReachSlice> {
    let mut by_chunk: BTreeMap<Option<usize>, CompactReachSlice> = BTreeMap::new();
    for group in &edge_tile.complete_decision_groups {
        let node = parent_layer_nodes[group.parent as usize];
        if node.kind != 0 {
            continue;
        }
        let chunk_index = Some(compact_chunk_index_for_public_infoset(
            &state.chunks,
            node.public_infoset as usize,
        ));
        let slice = by_chunk
            .entry(chunk_index)
            .or_insert_with(|| CompactReachSlice {
                chunk_index,
                edges: Vec::new(),
                groups: Vec::new(),
            });
        let first_edge = slice.edges.len() as u32;
        let edge_start = group.first_edge as usize;
        let edge_end = edge_start + group.edge_count as usize;
        slice
            .edges
            .extend_from_slice(&edge_tile.edges[edge_start..edge_end]);
        slice.groups.push(GpuPublicTreeEdgeGroup {
            first_edge,
            ..*group
        });
    }
    by_chunk.into_values().collect()
}

fn compact_chunk_index_for_public_infoset(
    chunks: &[GpuCompactPrivateCfrChunkState],
    public_infoset: usize,
) -> usize {
    chunks
        .iter()
        .position(|chunk| {
            chunk.chunk.public_start <= public_infoset && public_infoset < chunk.chunk.public_end
        })
        .expect("public infoset must be covered by compact chunk")
}

fn compact_chunk_covering_public_range(
    chunks: &[CompactPrivateCfrChunk],
    public_start: usize,
    public_end: usize,
) -> Option<&CompactPrivateCfrChunk> {
    chunks
        .iter()
        .find(|chunk| chunk.public_start <= public_start && public_end <= chunk.public_end)
}

fn public_action_offsets_from_nodes(nodes: &[GpuPublicTreeNode]) -> Vec<u32> {
    let public_infoset_count = nodes_public_infoset_count(nodes);
    let mut action_counts = vec![None; public_infoset_count];
    for node in nodes.iter().filter(|node| node.kind == 0) {
        let public_infoset = node.public_infoset as usize;
        let previous = action_counts[public_infoset].replace(node.child_count);
        assert!(
            previous.is_none() || previous == Some(node.child_count),
            "public infoset must have a stable action count"
        );
    }

    let mut offsets = Vec::with_capacity(public_infoset_count + 1);
    offsets.push(0);
    for action_count in action_counts {
        let action_count = action_count.expect("public infoset ids must be contiguous");
        offsets.push(offsets.last().copied().unwrap_or(0) + action_count);
    }
    offsets
}

fn public_infoset_bind_base(public_infoset: usize) -> usize {
    public_infoset - public_infoset % 32
}

fn public_infoset_range_for_nodes(nodes: &[GpuPublicTreeNode]) -> Option<(usize, usize)> {
    let mut min_infoset = usize::MAX;
    let mut max_infoset = 0usize;
    let mut found = false;
    for node in nodes {
        if node.kind != 0 {
            continue;
        }
        let infoset = node.public_infoset as usize;
        min_infoset = min_infoset.min(infoset);
        max_infoset = max_infoset.max(infoset);
        found = true;
    }
    found.then_some((public_infoset_bind_base(min_infoset), max_infoset + 1))
}

fn public_infoset_range_for_edges(
    nodes: &[GpuPublicTreeNode],
    edges: &[GpuPublicTreeEdge],
) -> Option<(usize, usize)> {
    let mut min_infoset = usize::MAX;
    let mut max_infoset = 0usize;
    let mut found = false;
    for edge in edges {
        let node = nodes[edge.parent as usize];
        if node.kind != 0 {
            continue;
        }
        let infoset = node.public_infoset as usize;
        min_infoset = min_infoset.min(infoset);
        max_infoset = max_infoset.max(infoset);
        found = true;
    }
    found.then_some((public_infoset_bind_base(min_infoset), max_infoset + 1))
}

fn public_infoset_exact_range_for_edges(
    nodes: &[GpuPublicTreeNode],
    edges: &[GpuPublicTreeEdge],
) -> Option<(usize, usize)> {
    let mut min_infoset = usize::MAX;
    let mut max_infoset = 0usize;
    let mut found = false;
    for edge in edges {
        let node = nodes[edge.parent as usize];
        if node.kind != 0 {
            continue;
        }
        let infoset = node.public_infoset as usize;
        min_infoset = min_infoset.min(infoset);
        max_infoset = max_infoset.max(infoset);
        found = true;
    }
    found.then_some((min_infoset, max_infoset + 1))
}

fn f32_range_byte_offset(elements: usize) -> u64 {
    byte_len::<f32>(elements)
}

fn f32_range_byte_size(elements: usize) -> u64 {
    byte_len::<f32>(elements.max(1))
}

fn public_tree_static_combo_live_masks(
    layered: &GpuPublicTreeLayered,
    combos: &[GpuPrivateCombo],
    combo_legal: &[u32],
) -> Vec<Vec<u32>> {
    let combo_count = combos.len();
    let mut masks = layered
        .layers
        .iter()
        .map(|layer| vec![0u32; (layer.nodes.len() * combo_count).div_ceil(32).max(1)])
        .collect::<Vec<_>>();
    if layered.layers.is_empty() || combo_count == 0 {
        return masks;
    }

    for combo in 0..combo_count {
        if combo_legal.get(combo).copied().unwrap_or(0) != 0 {
            set_combo_live_bit(&mut masks[0], combo);
        }
    }

    for layer_index in 0..layered.layers.len().saturating_sub(1) {
        let child_layer_index = layer_index + 1;
        let layer = &layered.layers[layer_index];
        for (parent_slot, node) in layer.nodes.iter().copied().enumerate() {
            if node.kind != 0 && node.kind != 1 {
                continue;
            }
            for action in 0..node.child_count as usize {
                let child_offset = node.first_child as usize + action;
                let child_slot = layer.children[child_offset] as usize;
                let chance_card = layer.child_cards.get(child_offset).copied().unwrap_or(52);
                for (combo, private_combo) in combos.iter().enumerate() {
                    if !combo_live_bit(&masks[layer_index], parent_slot * combo_count + combo) {
                        continue;
                    }
                    if node.kind == 1
                        && (private_combo.cards[0] == chance_card
                            || private_combo.cards[1] == chance_card)
                    {
                        continue;
                    }
                    set_combo_live_bit(
                        &mut masks[child_layer_index],
                        child_slot * combo_count + combo,
                    );
                }
            }
        }
    }

    masks
}

fn tile_combo_live_words(
    layer_words: &[u32],
    node_start: usize,
    node_end: usize,
    combo_count: usize,
) -> Vec<u32> {
    let value_len = (node_end - node_start) * combo_count;
    let mut words = vec![0u32; value_len.div_ceil(32).max(1)];
    for source_slot in node_start..node_end {
        for combo in 0..combo_count {
            let source_index = source_slot * combo_count + combo;
            if combo_live_bit(layer_words, source_index) {
                let local_slot = source_slot - node_start;
                set_combo_live_bit(&mut words, local_slot * combo_count + combo);
            }
        }
    }
    words
}

fn combo_live_bit(words: &[u32], index: usize) -> bool {
    words
        .get(index >> 5)
        .map(|word| (word & (1u32 << (index & 31))) != 0)
        .unwrap_or(false)
}

fn set_combo_live_bit(words: &mut [u32], index: usize) {
    if let Some(word) = words.get_mut(index >> 5) {
        *word |= 1u32 << (index & 31);
    }
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
    for (node_index, &depth) in depths.iter().enumerate() {
        if depth == 0 {
            layer_globals[0].push(node_index as u32);
        }
    }
    for depth in 0..max_depth {
        let mut seen_next = vec![false; nodes.len()];
        let parent_globals = layer_globals[depth].clone();
        for parent_index in parent_globals {
            let parent = nodes[parent_index as usize];
            if parent.kind != 0 && parent.kind != 1 {
                continue;
            }
            for action in 0..parent.child_count as usize {
                let child = children[parent.first_child as usize + action] as usize;
                if depths[child] == depth + 1 && !seen_next[child] {
                    seen_next[child] = true;
                    layer_globals[depth + 1].push(child as u32);
                }
            }
        }
        for (node_index, &node_depth) in depths.iter().enumerate() {
            if node_depth == depth + 1 && !seen_next[node_index] {
                seen_next[node_index] = true;
                layer_globals[depth + 1].push(node_index as u32);
            }
        }
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

    let node_tile_alignment = decision_child_count_alignment(nodes);
    let node_tile_size = layer_node_tile_size(
        combo_count,
        max_storage_buffer_binding_size,
        node_tile_alignment,
    );
    let max_layer_tiles = layers
        .iter()
        .map(|layer| layer.nodes.len().div_ceil(node_tile_size))
        .max()
        .unwrap_or(0);
    let reach_edge_tiles = public_tree_layer_edge_tiles(&layers, node_tile_size);
    let backup_tile_pairs = public_tree_layer_tile_pairs(&reach_edge_tiles);

    GpuPublicTreeLayered {
        node_tile_size,
        max_layer_tiles,
        reach_edge_tiles,
        backup_tile_pairs,
        layers,
        max_layer_nodes,
    }
}

fn layer_node_tile_size(
    combo_count: usize,
    max_storage_buffer_binding_size: u64,
    node_tile_alignment: usize,
) -> usize {
    let bytes_per_node = combo_count.max(1) as u64 * std::mem::size_of::<f32>() as u64;
    let value_tile_nodes = (max_storage_buffer_binding_size / bytes_per_node)
        .max(1)
        .try_into()
        .unwrap_or(usize::MAX);
    let state_action_bytes_per_public =
        combo_count.max(1) as u64 * 4 * std::mem::size_of::<f32>() as u64;
    let state_tile_publics = (max_storage_buffer_binding_size / state_action_bytes_per_public)
        .max(1)
        .try_into()
        .unwrap_or(usize::MAX);
    let tile_nodes = value_tile_nodes.min(state_tile_publics).max(1);
    if node_tile_alignment > 1 && tile_nodes >= node_tile_alignment {
        (tile_nodes / node_tile_alignment).max(1) * node_tile_alignment
    } else {
        tile_nodes
    }
}

fn decision_child_count_alignment(nodes: &[GpuPublicTreeNode]) -> usize {
    nodes
        .iter()
        .filter(|node| node.kind == 0 && node.child_count > 1)
        .map(|node| node.child_count as usize)
        .fold(1usize, |alignment, child_count| {
            lcm_usize(alignment, child_count).min(64)
        })
}

fn lcm_usize(left: usize, right: usize) -> usize {
    if left == 0 || right == 0 {
        return 1;
    }
    left / gcd_usize(left, right) * right
}

fn gcd_usize(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let next = left % right;
        left = right;
        right = next;
    }
    left
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
                let mut edges_by_bucket: BTreeMap<usize, Vec<GpuPublicTreeEdge>> = BTreeMap::new();
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
                        let bucket = if node.kind == 0 {
                            (node.public_infoset as usize / 4096) * 4096
                        } else {
                            0
                        };
                        edges_by_bucket
                            .entry(bucket)
                            .or_default()
                            .push(GpuPublicTreeEdge {
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
                for edges in edges_by_bucket.into_values() {
                    let groups = public_tree_edge_groups(&edges, parent, parent_start);
                    let split_edges =
                        public_tree_split_decision_edges(&edges, &groups, parent, parent_start);
                    let complete_decision_groups =
                        public_tree_complete_decision_groups(&groups, parent, parent_start);
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
                        split_edges,
                        complete_decision_groups,
                        groups,
                    });
                }
            }
        }
    }
    tiles
}

fn public_tree_complete_decision_groups(
    groups: &[GpuPublicTreeEdgeGroup],
    parent_layer: &GpuPublicTreeLayer,
    parent_start: usize,
) -> Vec<GpuPublicTreeEdgeGroup> {
    groups
        .iter()
        .copied()
        .filter(|group| {
            let node = parent_layer.nodes[parent_start + group.parent as usize];
            node.kind == 0 && group.edge_count == node.child_count
        })
        .collect()
}

fn public_tree_split_decision_edges(
    edges: &[GpuPublicTreeEdge],
    groups: &[GpuPublicTreeEdgeGroup],
    parent_layer: &GpuPublicTreeLayer,
    parent_start: usize,
) -> Vec<GpuPublicTreeEdge> {
    let mut split_edges = Vec::new();
    for group in groups {
        let node = parent_layer.nodes[parent_start + group.parent as usize];
        if node.kind != 0 || group.edge_count == node.child_count {
            continue;
        }
        let start = group.first_edge as usize;
        let end = start + group.edge_count as usize;
        split_edges.extend_from_slice(&edges[start..end]);
    }
    split_edges
}

fn public_tree_edge_groups(
    edges: &[GpuPublicTreeEdge],
    parent_layer: &GpuPublicTreeLayer,
    parent_start: usize,
) -> Vec<GpuPublicTreeEdgeGroup> {
    let mut groups = Vec::new();
    let mut index = 0usize;
    while index < edges.len() {
        let parent = edges[index].parent;
        let first_edge = index;
        index += 1;
        let node = parent_layer.nodes[parent_start + parent as usize];
        if node.kind == 0 {
            while index < edges.len() && edges[index].parent == parent {
                index += 1;
            }
        }
        groups.push(GpuPublicTreeEdgeGroup {
            parent,
            first_edge: first_edge as u32,
            edge_count: (index - first_edge) as u32,
            _pad0: 0,
        });
    }
    groups
}

fn public_tree_partition_fused_public_infosets(
    layered: &mut GpuPublicTreeLayered,
    public_infoset_count: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut mask = vec![0u32; public_infoset_count.max(1)];
    let mut split = vec![false; public_infoset_count.max(1)];
    for edge_tile in &layered.reach_edge_tiles {
        let parent_layer = &layered.layers[edge_tile.parent_layer];
        for group in &edge_tile.groups {
            let node = parent_layer.nodes[edge_tile.parent_tile.node_start + group.parent as usize];
            if node.kind != 0 || (node.public_infoset as usize) >= public_infoset_count {
                continue;
            }
            let public_infoset = node.public_infoset as usize;
            if group.edge_count == node.child_count {
                mask[public_infoset] = 1;
            } else {
                split[public_infoset] = true;
            }
        }
    }
    let split_public_infosets = split
        .iter()
        .enumerate()
        .filter_map(|(public_infoset, &has_split)| {
            if public_infoset < public_infoset_count && has_split {
                Some(public_infoset as u32)
            } else {
                None
            }
        })
        .collect();
    for (slot, has_split) in split.into_iter().enumerate() {
        if has_split {
            mask[slot] = 0;
        }
    }
    for edge_tile in &mut layered.reach_edge_tiles {
        let parent_layer = &layered.layers[edge_tile.parent_layer];
        edge_tile.complete_decision_groups.retain(|group| {
            let node = parent_layer.nodes[edge_tile.parent_tile.node_start + group.parent as usize];
            (node.public_infoset as usize) < public_infoset_count
                && mask[node.public_infoset as usize] != 0
        });
    }
    (mask, split_public_infosets)
}

fn public_tree_layer_tile_pairs(
    edge_tiles: &[GpuPublicTreeLayerEdgeTile],
) -> Vec<GpuPublicTreeLayerTilePair> {
    let mut keys = BTreeMap::new();
    for edge_tile in edge_tiles {
        keys.entry((
            edge_tile.parent_layer,
            edge_tile.child_layer,
            edge_tile.parent_tile.node_start,
            edge_tile.child_tile.node_start,
        ))
        .or_insert(());
    }
    keys.into_keys()
        .map(
            |(parent_layer, child_layer, parent_node_start, child_node_start)| {
                GpuPublicTreeLayerTilePair {
                    parent_layer,
                    child_layer,
                    parent_tile: GpuPublicTreeLayerTile {
                        node_start: parent_node_start,
                    },
                    child_tile: GpuPublicTreeLayerTile {
                        node_start: child_node_start,
                    },
                }
            },
        )
        .collect()
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
    fn public_action_offsets_follow_decision_child_counts() {
        let nodes = [
            GpuPublicTreeNode {
                kind: 0,
                acting_player: 0,
                public_infoset: 0,
                first_child: 0,
                child_count: 2,
                terminal_kind: 0,
                showdown_offset: 0,
                _pad0: 0,
                pot: 0.0,
                hero_invested: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            },
            GpuPublicTreeNode {
                kind: 1,
                acting_player: 0,
                public_infoset: 0,
                first_child: 0,
                child_count: 0,
                terminal_kind: 0,
                showdown_offset: 0,
                _pad0: 0,
                pot: 0.0,
                hero_invested: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            },
            GpuPublicTreeNode {
                kind: 0,
                acting_player: 1,
                public_infoset: 1,
                first_child: 0,
                child_count: 3,
                terminal_kind: 0,
                showdown_offset: 0,
                _pad0: 0,
                pot: 0.0,
                hero_invested: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            },
            GpuPublicTreeNode {
                kind: 0,
                acting_player: 0,
                public_infoset: 2,
                first_child: 0,
                child_count: 1,
                terminal_kind: 0,
                showdown_offset: 0,
                _pad0: 0,
                pot: 0.0,
                hero_invested: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            },
        ];

        assert_eq!(public_action_offsets_from_nodes(&nodes), vec![0, 2, 5, 6]);
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
