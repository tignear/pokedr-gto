use std::sync::mpsc;

use wgpu::util::DeviceExt;

use super::DenseCfrState;

const WORKGROUP_SIZE: u32 = 64;

const CFR_UPDATE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> regrets: array<f32>;
@group(0) @binding(1) var<storage, read_write> strategy_sum: array<f32>;
@group(0) @binding(2) var<storage, read> action_values: array<f32>;
@group(0) @binding(3) var<storage, read> reach_weights: array<f32>;
@group(0) @binding(4) var<storage, read> strategy_weights: array<f32>;
@group(0) @binding(5) var<storage, read> params: array<u32>;

fn positive(value: f32) -> f32 {
    return max(value, 0.0);
}

fn strategy_at(offset: u32, action: u32, actions: u32, normalizer: f32) -> f32 {
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
    for (var action = 0u; action < actions; action = action + 1u) {
        normalizer = normalizer + positive(regrets[offset + action]);
    }

    var node_value = 0.0;
    for (var action = 0u; action < actions; action = action + 1u) {
        let strategy = strategy_at(offset, action, actions, normalizer);
        node_value = node_value + strategy * action_values[offset + action];
    }

    var discount = 1.0;
    if variant == 1u {
        let t = f32(max(iteration, 1u));
        discount = t / (t + 1.0);
    }

    for (var action = 0u; action < actions; action = action + 1u) {
        let strategy = strategy_at(offset, action, actions, normalizer);
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

#[derive(Debug)]
pub enum GpuCfrError {
    NoAdapter,
    RequestDevice(String),
    MapFailed(String),
}

pub struct GpuDenseCfrBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuDenseCfrBackend {
    pub fn new() -> Result<Self, GpuCfrError> {
        pollster::block_on(Self::new_async())
    }

    pub async fn new_async() -> Result<Self, GpuCfrError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|_| GpuCfrError::NoAdapter)?;
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
        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense_cfr::{CfrVariant, DenseCfrConfig};

    #[test]
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
}
