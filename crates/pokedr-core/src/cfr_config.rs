#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealCfrConfig {
    pub iterations: u32,
    pub variant: RealCfrVariant,
    pub average_strategy: RealCfrAverageStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RealCfrVariant {
    CfrPlus,
    Dcfr { alpha: f32, beta: f32, gamma: f32 },
    DcfrPlus { alpha: f32, gamma: f32 },
}

impl Default for RealCfrVariant {
    fn default() -> Self {
        Self::CfrPlus
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealCfrAverageStrategy {
    ReachWeighted,
    Local,
}

impl Default for RealCfrAverageStrategy {
    fn default() -> Self {
        Self::ReachWeighted
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealCfrExploitability {
    pub profile_oop_value: f32,
    pub profile_ip_value: f32,
    pub oop_best_response_value: f32,
    pub ip_best_response_value: f32,
    pub oop_gain: f32,
    pub ip_gain: f32,
    pub nash_conv_chips: f32,
    pub exploitability_chips: f32,
    pub exploitability_bb_per_100: f32,
}
