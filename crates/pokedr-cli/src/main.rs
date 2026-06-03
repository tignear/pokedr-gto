use pokedr_core::{
    cards::{Board, Card, Rank, Suit},
    dense_cfr::{
        CfrVariant, DenseCfrConfig, DenseCfrIteration, DenseCfrSolver, DenseCfrState,
        gpu::{GpuCfrError, GpuDenseCfrBackend},
    },
    postflop::{ActionSetConfig, Player, PublicState, Street, SubgameTree, SubgameTreeConfig},
    postflop_dense::PostflopDenseLayout,
};

fn main() {
    let mut args = std::env::args();
    let program = args.next().unwrap_or_else(|| "pokedr".to_string());
    match args.next().as_deref() {
        Some("gpu-info") => print_gpu_info(),
        Some("gpu-smoke") => run_gpu_smoke(),
        Some("postflop-smoke") => run_postflop_smoke(),
        Some("solve-flop") => run_solve_flop(args.next().as_deref()),
        Some("solve-flop-metrics") => run_solve_flop_metrics(args.next().as_deref()),
        Some("rs-poker-smoke") => run_rs_poker_smoke(),
        Some("rs-poker-trace") => run_rs_poker_trace(),
        _ => {
            eprintln!(
                "usage: {program} <gpu-info|gpu-smoke|postflop-smoke|solve-flop|solve-flop-metrics|rs-poker-smoke|rs-poker-trace>"
            );
            std::process::exit(2);
        }
    }
}

fn print_gpu_info() {
    let backend = match GpuDenseCfrBackend::new() {
        Ok(backend) => backend,
        Err(GpuCfrError::NoAdapter) => {
            println!("no GPU adapter visible to wgpu");
            return;
        }
        Err(error) => {
            eprintln!("failed to initialize GPU backend: {error:?}");
            std::process::exit(1);
        }
    };
    let info = backend.adapter_info();
    println!("name: {}", info.name);
    println!("backend: {:?}", info.backend);
    println!("device_type: {:?}", info.device_type);
    println!("vendor: 0x{:x}", info.vendor);
    println!("device: 0x{:x}", info.device);
}

fn run_gpu_smoke() {
    println!("initializing GPU backend");
    let backend = match GpuDenseCfrBackend::new() {
        Ok(backend) => backend,
        Err(GpuCfrError::NoAdapter) => {
            eprintln!("no GPU adapter visible to wgpu");
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!("failed to initialize GPU backend: {error:?}");
            std::process::exit(1);
        }
    };
    let info = backend.adapter_info();
    println!("adapter: {} ({:?})", info.name, info.backend);

    println!("running one-shot update");
    run_one_shot_update(&backend);
    println!("one-shot update passed");

    println!("running resident updates");
    run_resident_updates(&backend);
    println!("GPU smoke passed");
}

fn run_one_shot_update(backend: &GpuDenseCfrBackend) {
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
        .unwrap_or_else(|error| {
            eprintln!("GPU one-shot update failed: {error:?}");
            std::process::exit(1);
        });
    assert_close("one-shot regret", cpu.regrets(), gpu.regrets());
    assert_close(
        "one-shot strategy_sum",
        cpu.strategy_sum(),
        gpu.strategy_sum(),
    );
}

fn run_resident_updates(backend: &GpuDenseCfrBackend) {
    let config = DenseCfrConfig {
        infosets: 8,
        actions: 4,
        variant: CfrVariant::Discounted,
    };
    let mut cpu = DenseCfrSolver::new(config.clone());
    let mut gpu = backend.resident_solver(config);

    cpu.run_iterations(5, fill_fixture_iteration_with_state);
    gpu.run_iterations(&backend, 5, |iteration, batch| {
        println!("dispatch iteration {iteration}");
        fill_fixture_iteration(iteration, batch);
    })
    .unwrap_or_else(|error| {
        eprintln!("GPU update failed: {error:?}");
        std::process::exit(1);
    });

    println!("downloading result");
    let downloaded = gpu.download(&backend).unwrap_or_else(|error| {
        eprintln!("GPU download failed: {error:?}");
        std::process::exit(1);
    });
    assert_close("regret", cpu.state().regrets(), downloaded.regrets());
    assert_close(
        "strategy_sum",
        cpu.state().strategy_sum(),
        downloaded.strategy_sum(),
    );
}

fn run_postflop_smoke() {
    let backend = match GpuDenseCfrBackend::new() {
        Ok(backend) => backend,
        Err(GpuCfrError::NoAdapter) => {
            eprintln!("no GPU adapter visible to wgpu");
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!("failed to initialize GPU backend: {error:?}");
            std::process::exit(1);
        }
    };
    let info = backend.adapter_info();
    println!("adapter: {} ({:?})", info.name, info.backend);

    let tree = smoke_postflop_tree();
    let layout = PostflopDenseLayout::from_tree(&tree);
    let config = layout.dense_config(CfrVariant::CfrPlus);
    println!(
        "public nodes: {}, infosets: {}, max actions: {}",
        tree.nodes().len(),
        layout.infoset_count(),
        layout.max_actions()
    );

    let mut solver =
        backend.resident_solver_with_legal_actions(config.clone(), layout.legal_actions().to_vec());
    solver
        .run_iterations(&backend, 8, |iteration, batch| {
            fill_postflop_fixture_iteration(iteration, &layout, batch);
        })
        .unwrap_or_else(|error| {
            eprintln!("postflop GPU update failed: {error:?}");
            std::process::exit(1);
        });
    let state = solver.download(&backend).unwrap_or_else(|error| {
        eprintln!("postflop GPU download failed: {error:?}");
        std::process::exit(1);
    });
    assert!(
        state.regrets().iter().all(|value| value.is_finite()),
        "regrets must stay finite"
    );
    assert!(
        state.strategy_sum().iter().all(|value| value.is_finite()),
        "strategy sums must stay finite"
    );
    println!("postflop GPU CFR smoke passed");
}

fn run_rs_poker_smoke() {
    let summary = pokedr_agent::run_heads_up_match_with_config(smoke_hands(), 7, match_config());
    println!("hands: {}", summary.hands);
    println!("hero_net: {:.2}", summary.hero_net);
    println!("villain_net: {:.2}", summary.villain_net);
}

fn run_rs_poker_trace() {
    for trace in
        pokedr_agent::run_traced_heads_up_match_with_config(smoke_hands(), 7, match_config())
    {
        println!(
            "hand {} hero [{}] villain [{}] board [{}] hero_net {:.2} villain_net {:.2}",
            trace.hand_index,
            trace.hero_cards,
            trace.villain_cards,
            trace.board,
            trace.hero_net,
            trace.villain_net
        );
        for action in &trace.actions {
            println!("  {action}");
        }
        for award in &trace.awards {
            println!("  {award}");
        }
    }
}

fn run_solve_flop(flop_arg: Option<&str>) {
    let flop = parse_flop(flop_arg.unwrap_or("As7h2c")).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let config = fixed_flop_config();
    println!(
        "solving fixed flop iterations={} depth={} runouts={}",
        config.cfr_iterations, config.max_depth, config.max_showdown_runouts
    );
    let summary = pokedr_agent::solve_fixed_flop_once(flop, config);
    println!("board: {}", summary.board);
    println!("iterations: {}", summary.iterations);
    println!("public_decisions: {}", summary.decisions);
    println!("chance_nodes: {}", summary.chance);
    println!("terminals: {}", summary.terminals);
    println!("public_infosets: {}", summary.public_infosets);
    println!("private_infosets: {}", summary.private_infosets);
    println!("max_actions: {}", summary.max_actions);
    println!("elapsed: {:.2}s", summary.elapsed_secs);
}

fn run_solve_flop_metrics(flop_arg: Option<&str>) {
    let flop = parse_flop(flop_arg.unwrap_or("As7h2c")).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let config = match_config();
    let iterations = metric_iterations();
    println!(
        "solving fixed flop metrics depth={} runouts={} iterations={:?}",
        config.max_depth, config.max_showdown_runouts, iterations
    );
    for row in pokedr_agent::solve_fixed_flop_metrics(flop, config, &iterations) {
        let delta = row
            .root_strategy_l1_delta
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string());
        let root_actions = row
            .root_action_probabilities
            .iter()
            .map(|value| format!("{value:.4}"))
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "board={} iterations={} elapsed={:.2}s root_l1_delta={} root_actions=[{}] regret_mass={:.3} illegal_mass={:.6} current_norm_err={:.6} avg_norm_err={:.6} finite={}",
            row.board,
            row.iterations,
            row.elapsed_secs,
            delta,
            root_actions,
            row.positive_regret_mass,
            row.illegal_strategy_mass,
            row.current_strategy_norm_error,
            row.average_strategy_norm_error,
            row.finite
        );
    }
}

fn smoke_hands() -> usize {
    std::env::var("POKEDR_HANDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(16)
}

fn metric_iterations() -> Vec<usize> {
    std::env::var("POKEDR_METRIC_ITERATIONS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|value| *value > 0)
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32])
}

fn fixed_flop_config() -> pokedr_agent::PokedrAgentConfig {
    let mut config = match_config();
    config.cfr_iterations = env_usize("POKEDR_CFR_ITERATIONS").unwrap_or(1);
    config
}

fn match_config() -> pokedr_agent::PokedrAgentConfig {
    let mut config = pokedr_agent::PokedrAgentConfig::default();
    config.cfr_iterations = env_usize("POKEDR_CFR_ITERATIONS").unwrap_or(config.cfr_iterations);
    config.max_depth = env_usize("POKEDR_MAX_DEPTH").unwrap_or(config.max_depth);
    config.max_showdown_runouts =
        env_usize("POKEDR_MAX_SHOWDOWN_RUNOUTS").unwrap_or(config.max_showdown_runouts);
    config
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

fn parse_flop(input: &str) -> Result<[Card; 3], String> {
    let normalized = input
        .split(|character: char| character == ',' || character.is_whitespace())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    let tokens = if normalized.len() == 3 {
        normalized
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>()
    } else {
        split_compact_cards(input)?
    };
    if tokens.len() != 3 {
        return Err("flop must contain exactly 3 cards, for example As7h2c".to_string());
    }
    let cards = [
        parse_card(&tokens[0])?,
        parse_card(&tokens[1])?,
        parse_card(&tokens[2])?,
    ];
    let mask = cards
        .iter()
        .fold(0u64, |mask, card| mask | card.deck_mask());
    if mask.count_ones() != 3 {
        return Err("flop contains duplicate cards".to_string());
    }
    Ok(cards)
}

fn split_compact_cards(input: &str) -> Result<Vec<String>, String> {
    let chars = input
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<Vec<_>>();
    let mut cards = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        if index + 1 >= chars.len() {
            return Err("card must include rank and suit".to_string());
        }
        let rank_len = if chars[index] == '1' && chars.get(index + 1) == Some(&'0') {
            2
        } else {
            1
        };
        if index + rank_len >= chars.len() {
            return Err("card must include rank and suit".to_string());
        }
        let token = chars[index..=index + rank_len].iter().collect();
        cards.push(token);
        index += rank_len + 1;
    }
    Ok(cards)
}

fn parse_card(input: &str) -> Result<Card, String> {
    let lower = input.trim().to_ascii_lowercase();
    if lower.len() < 2 {
        return Err(format!("invalid card: {input}"));
    }
    let (rank_part, suit_part) = lower.split_at(lower.len() - 1);
    let rank = match rank_part {
        "2" => Rank::Two,
        "3" => Rank::Three,
        "4" => Rank::Four,
        "5" => Rank::Five,
        "6" => Rank::Six,
        "7" => Rank::Seven,
        "8" => Rank::Eight,
        "9" => Rank::Nine,
        "t" | "10" => Rank::Ten,
        "j" => Rank::Jack,
        "q" => Rank::Queen,
        "k" => Rank::King,
        "a" => Rank::Ace,
        _ => return Err(format!("invalid rank in card: {input}")),
    };
    let suit = match suit_part {
        "c" => Suit::Clubs,
        "d" => Suit::Diamonds,
        "h" => Suit::Hearts,
        "s" => Suit::Spades,
        _ => return Err(format!("invalid suit in card: {input}")),
    };
    Ok(Card::new(rank, suit))
}

fn smoke_postflop_tree() -> SubgameTree {
    SubgameTree::build(
        PublicState {
            street: Street::Flop,
            board: Board::new(vec![
                Card::new(Rank::Ace, Suit::Spades),
                Card::new(Rank::Seven, Suit::Hearts),
                Card::new(Rank::Two, Suit::Clubs),
            ]),
            pot: 100,
            hero_invested: 50,
            villain_invested: 50,
            effective_stack: 300,
            to_call: 0,
            min_aggressive_amount: 50,
            acting_player: Player::Hero,
            raises_this_street: 0,
            checks_this_street: 0,
        },
        SubgameTreeConfig {
            action_set: ActionSetConfig {
                max_aggressive_actions: 2,
                flop_bet_fractions: vec![0.5],
                turn_bet_fractions: vec![0.5],
                river_bet_fractions: vec![0.5],
                raise_fractions: vec![1.0],
                ..ActionSetConfig::default()
            },
            max_raises_per_street: 1,
            max_depth: 5,
        },
    )
}

fn fill_postflop_fixture_iteration(
    iteration: usize,
    layout: &PostflopDenseLayout,
    batch: &mut DenseCfrIteration,
) {
    batch.action_values.fill(0.0);
    batch.reach_weights.fill(1.0);
    batch.strategy_weights.fill(1.0);
    for infoset in 0..layout.infoset_count() {
        let offset = infoset * layout.max_actions();
        for action in 0..layout.action_count(infoset) {
            batch.action_values[offset + action] =
                ((infoset as f32 * 0.17) + (action as f32 * 0.31) + iteration as f32).sin();
        }
    }
}

fn fill_fixture_iteration(iteration: usize, batch: &mut DenseCfrIteration) {
    for (index, value) in batch.action_values.iter_mut().enumerate() {
        *value = ((index as f32 + iteration as f32) * 0.25).sin();
    }
    batch.reach_weights.fill(1.0);
    batch.strategy_weights.fill(0.75);
}

fn fill_fixture_iteration_with_state(
    iteration: usize,
    _state: &DenseCfrState,
    batch: &mut DenseCfrIteration,
) {
    fill_fixture_iteration(iteration, batch);
}

fn assert_close(label: &str, expected: &[f32], actual: &[f32]) {
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        if (expected - actual).abs() >= 1e-5 {
            eprintln!("{label}[{index}] mismatch: expected {expected}, actual {actual}");
            std::process::exit(1);
        }
    }
}
