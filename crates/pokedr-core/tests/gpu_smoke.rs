use pokedr_core::cards::{Card, Rank, Suit, evaluate};
use pokedr_core::dense_cfr::{
    CfrVariant, DenseCfrConfig, DenseCfrSolver, DenseCfrState,
    gpu::{GpuCfrError, GpuDenseCfrBackend, GpuPublicTreeNode, GpuShowdownTask},
    gpu::{GpuFinalBoard, GpuPrivateCombo},
};

fn main() {
    let backend = match GpuDenseCfrBackend::new() {
        Ok(backend) => backend,
        Err(GpuCfrError::NoAdapter) => {
            println!("skipping GPU smoke: no GPU adapter visible to wgpu");
            return;
        }
        Err(error) => {
            eprintln!("failed to initialize GPU backend: {error:?}");
            std::process::exit(1);
        }
    };

    let info = backend.adapter_info();
    println!("adapter: {} ({:?})", info.name, info.backend);

    run_one_shot_update(&backend);
    run_masked_one_shot_update(&backend);
    run_resident_updates(&backend);
    run_showdown_equities(&backend);
    run_showdown_matrix(&backend);
    run_public_tree_terminal_values(&backend);
    println!("GPU smoke passed");
}

fn run_one_shot_update(backend: &GpuDenseCfrBackend) {
    let action_values = [
        1.0, -0.5, 0.25, -1.0, 2.0, 0.0, 0.5, 0.25, -0.75, 3.0, 1.0, -2.0,
    ];
    let reach_weights = [1.0, 0.5, 2.0, 0.25];
    let strategy_weights = [1.0, 1.0, 0.5, 2.0];

    for variant in [
        CfrVariant::CfrPlus,
        CfrVariant::Discounted,
        CfrVariant::DcfrPlus,
    ] {
        let config = DenseCfrConfig {
            infosets: 4,
            actions: 3,
            variant,
        };
        let mut cpu = DenseCfrState::new(config.clone());
        let mut gpu = DenseCfrState::new(config);
        for iteration in 1..=3 {
            cpu.update_all_infosets(&action_values, &reach_weights, &strategy_weights, iteration);
            backend
                .update_all_infosets(
                    &mut gpu,
                    &action_values,
                    &reach_weights,
                    &strategy_weights,
                    iteration,
                )
                .unwrap_or_else(|error| fail(&format!("GPU one-shot update failed: {error:?}")));
        }

        assert_close("one-shot regret", cpu.regrets(), gpu.regrets());
        assert_close(
            "one-shot strategy_sum",
            cpu.strategy_sum(),
            gpu.strategy_sum(),
        );
    }
}

fn run_masked_one_shot_update(backend: &GpuDenseCfrBackend) {
    let config = DenseCfrConfig {
        infosets: 3,
        actions: 4,
        variant: CfrVariant::CfrPlus,
    };
    let legal_actions = vec![
        true, true, false, false, true, false, true, false, false, true, true, true,
    ];
    let mut cpu = DenseCfrState::new_with_legal_actions(config.clone(), legal_actions.clone());
    let mut gpu = DenseCfrState::new_with_legal_actions(config, legal_actions);
    let action_values = [
        1.0, -0.5, 100.0, 100.0, -1.0, 100.0, 2.0, 100.0, 100.0, 0.25, 0.75, -0.25,
    ];
    let reach_weights = [1.0, 0.5, 2.0];
    let strategy_weights = [1.0, 0.75, 0.25];

    cpu.update_all_infosets(&action_values, &reach_weights, &strategy_weights, 1);
    backend
        .update_all_infosets(
            &mut gpu,
            &action_values,
            &reach_weights,
            &strategy_weights,
            1,
        )
        .unwrap_or_else(|error| fail(&format!("GPU masked update failed: {error:?}")));

    assert_close("masked regret", cpu.regrets(), gpu.regrets());
    assert_close(
        "masked strategy_sum",
        cpu.strategy_sum(),
        gpu.strategy_sum(),
    );
}

fn run_resident_updates(backend: &GpuDenseCfrBackend) {
    for variant in [
        CfrVariant::CfrPlus,
        CfrVariant::Discounted,
        CfrVariant::DcfrPlus,
    ] {
        let config = DenseCfrConfig {
            infosets: 8,
            actions: 4,
            variant,
        };
        let mut cpu = DenseCfrSolver::new(config.clone());
        let mut gpu = backend.resident_solver(config);

        cpu.run_iterations(5, fill_fixture_iteration_with_state);
        gpu.run_iterations(backend, 5, fill_fixture_iteration)
            .unwrap_or_else(|error| fail(&format!("GPU resident update failed: {error:?}")));

        let downloaded = gpu
            .download(backend)
            .unwrap_or_else(|error| fail(&format!("GPU download failed: {error:?}")));
        assert_close(
            "resident regret",
            cpu.state().regrets(),
            downloaded.regrets(),
        );
        assert_close(
            "resident strategy_sum",
            cpu.state().strategy_sum(),
            downloaded.strategy_sum(),
        );
    }
}

fn fill_fixture_iteration(iteration: usize, batch: &mut pokedr_core::dense_cfr::DenseCfrIteration) {
    for (index, value) in batch.action_values.iter_mut().enumerate() {
        *value = ((index as f32 + iteration as f32) * 0.25).sin();
    }
    batch.reach_weights.fill(1.0);
    batch.strategy_weights.fill(0.75);
}

fn fill_fixture_iteration_with_state(
    iteration: usize,
    _state: &pokedr_core::dense_cfr::DenseCfrState,
    batch: &mut pokedr_core::dense_cfr::DenseCfrIteration,
) {
    fill_fixture_iteration(iteration, batch);
}

fn run_showdown_equities(backend: &GpuDenseCfrBackend) {
    let fixtures = vec![
        (
            [
                Card::new(Rank::Ace, Suit::Spades),
                Card::new(Rank::King, Suit::Spades),
                Card::new(Rank::Queen, Suit::Spades),
                Card::new(Rank::Two, Suit::Clubs),
                Card::new(Rank::Three, Suit::Diamonds),
            ],
            [
                Card::new(Rank::Jack, Suit::Spades),
                Card::new(Rank::Ten, Suit::Spades),
            ],
            [
                Card::new(Rank::Ace, Suit::Clubs),
                Card::new(Rank::Ace, Suit::Diamonds),
            ],
        ),
        (
            [
                Card::new(Rank::Nine, Suit::Clubs),
                Card::new(Rank::Nine, Suit::Diamonds),
                Card::new(Rank::Four, Suit::Spades),
                Card::new(Rank::Four, Suit::Hearts),
                Card::new(Rank::Two, Suit::Clubs),
            ],
            [
                Card::new(Rank::Nine, Suit::Hearts),
                Card::new(Rank::Nine, Suit::Spades),
            ],
            [
                Card::new(Rank::Four, Suit::Clubs),
                Card::new(Rank::Ace, Suit::Diamonds),
            ],
        ),
        (
            [
                Card::new(Rank::Two, Suit::Spades),
                Card::new(Rank::Three, Suit::Spades),
                Card::new(Rank::Four, Suit::Spades),
                Card::new(Rank::Five, Suit::Spades),
                Card::new(Rank::King, Suit::Clubs),
            ],
            [
                Card::new(Rank::Six, Suit::Spades),
                Card::new(Rank::Seven, Suit::Clubs),
            ],
            [
                Card::new(Rank::Ace, Suit::Hearts),
                Card::new(Rank::Ace, Suit::Diamonds),
            ],
        ),
        (
            [
                Card::new(Rank::Ace, Suit::Clubs),
                Card::new(Rank::King, Suit::Diamonds),
                Card::new(Rank::Queen, Suit::Hearts),
                Card::new(Rank::Jack, Suit::Spades),
                Card::new(Rank::Ten, Suit::Clubs),
            ],
            [
                Card::new(Rank::Two, Suit::Clubs),
                Card::new(Rank::Three, Suit::Diamonds),
            ],
            [
                Card::new(Rank::Four, Suit::Hearts),
                Card::new(Rank::Five, Suit::Spades),
            ],
        ),
    ];
    let tasks: Vec<_> = fixtures
        .iter()
        .map(|(board, hero, villain)| showdown_task(*board, *hero, *villain))
        .collect();
    let gpu = backend
        .showdown_equities(&tasks)
        .unwrap_or_else(|error| fail(&format!("GPU showdown failed: {error:?}")));
    let expected: Vec<_> = fixtures
        .iter()
        .map(|(board, hero, villain)| cpu_showdown(*board, *hero, *villain))
        .collect();
    assert_close("showdown equity", &expected, &gpu);
}

fn showdown_task(board: [Card; 5], hero: [Card; 2], villain: [Card; 2]) -> GpuShowdownTask {
    GpuShowdownTask {
        cards: [
            hero[0].index() as u32,
            hero[1].index() as u32,
            villain[0].index() as u32,
            villain[1].index() as u32,
            board[0].index() as u32,
            board[1].index() as u32,
            board[2].index() as u32,
            board[3].index() as u32,
            board[4].index() as u32,
        ],
    }
}

fn cpu_showdown(board: [Card; 5], hero: [Card; 2], villain: [Card; 2]) -> f32 {
    let mut hero_cards = hero.to_vec();
    hero_cards.extend(board);
    let mut villain_cards = villain.to_vec();
    villain_cards.extend(board);
    match evaluate(&hero_cards).cmp(&evaluate(&villain_cards)) {
        std::cmp::Ordering::Greater => 1.0,
        std::cmp::Ordering::Equal => 0.5,
        std::cmp::Ordering::Less => 0.0,
    }
}

fn run_showdown_matrix(backend: &GpuDenseCfrBackend) {
    let combos = vec![
        gpu_combo([
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::King, Suit::Spades),
        ]),
        gpu_combo([
            Card::new(Rank::Ace, Suit::Clubs),
            Card::new(Rank::Ace, Suit::Diamonds),
        ]),
        gpu_combo([
            Card::new(Rank::Jack, Suit::Spades),
            Card::new(Rank::Ten, Suit::Spades),
        ]),
    ];
    let boards = vec![GpuFinalBoard {
        cards: [
            Card::new(Rank::Queen, Suit::Spades).index() as u32,
            Card::new(Rank::Nine, Suit::Spades).index() as u32,
            Card::new(Rank::Two, Suit::Clubs).index() as u32,
            Card::new(Rank::Three, Suit::Diamonds).index() as u32,
            Card::new(Rank::Four, Suit::Hearts).index() as u32,
        ],
    }];
    let matrix = backend
        .showdown_matrix(&combos, &boards)
        .unwrap_or_else(|error| fail(&format!("GPU showdown matrix failed: {error:?}")));
    let board = [
        Card::new(Rank::Queen, Suit::Spades),
        Card::new(Rank::Nine, Suit::Spades),
        Card::new(Rank::Two, Suit::Clubs),
        Card::new(Rank::Three, Suit::Diamonds),
        Card::new(Rank::Four, Suit::Hearts),
    ];
    let combo_cards = [
        [
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::King, Suit::Spades),
        ],
        [
            Card::new(Rank::Ace, Suit::Clubs),
            Card::new(Rank::Ace, Suit::Diamonds),
        ],
        [
            Card::new(Rank::Jack, Suit::Spades),
            Card::new(Rank::Ten, Suit::Spades),
        ],
    ];
    for hero in 0..combo_cards.len() {
        for villain in 0..combo_cards.len() {
            let expected = if hero == villain {
                0.5
            } else {
                cpu_showdown(board, combo_cards[hero], combo_cards[villain])
            };
            let actual = matrix[hero * combo_cards.len() + villain];
            if (expected - actual).abs() >= 1e-5 {
                fail(&format!(
                    "showdown matrix[{hero},{villain}] mismatch: expected {expected}, actual {actual}"
                ));
            }
        }
    }
}

fn run_public_tree_terminal_values(backend: &GpuDenseCfrBackend) {
    let combos = vec![
        gpu_combo([
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::King, Suit::Spades),
        ]),
        gpu_combo([
            Card::new(Rank::Ace, Suit::Clubs),
            Card::new(Rank::Ace, Suit::Diamonds),
        ]),
        gpu_combo([
            Card::new(Rank::Queen, Suit::Hearts),
            Card::new(Rank::Jack, Suit::Hearts),
        ]),
    ];
    let nodes = vec![
        GpuPublicTreeNode {
            kind: 0,
            acting_player: 0,
            public_infoset: 0,
            first_child: 0,
            child_count: 2,
            terminal_kind: 0,
            showdown_offset: 0,
            _pad0: 0,
            pot: 10.0,
            hero_invested: 4.0,
            _pad1: 1.0,
            _pad2: 0.0,
        },
        GpuPublicTreeNode {
            kind: 2,
            acting_player: 0,
            public_infoset: 0,
            first_child: 0,
            child_count: 0,
            terminal_kind: 0,
            showdown_offset: 0,
            _pad0: 0,
            pot: 10.0,
            hero_invested: 4.0,
            _pad1: 1.0,
            _pad2: 0.0,
        },
        GpuPublicTreeNode {
            kind: 2,
            acting_player: 0,
            public_infoset: 0,
            first_child: 0,
            child_count: 0,
            terminal_kind: 1,
            showdown_offset: 0,
            _pad0: 0,
            pot: 10.0,
            hero_invested: 4.0,
            _pad1: 1.0,
            _pad2: 0.0,
        },
    ];
    let children = vec![1, 2];
    let boards = vec![GpuFinalBoard {
        cards: [
            Card::new(Rank::Two, Suit::Clubs).index() as u32,
            Card::new(Rank::Three, Suit::Diamonds).index() as u32,
            Card::new(Rank::Four, Suit::Hearts).index() as u32,
            Card::new(Rank::Five, Suit::Spades).index() as u32,
            Card::new(Rank::Six, Suit::Clubs).index() as u32,
        ],
    }];
    let state = DenseCfrState::new_with_legal_actions(
        DenseCfrConfig {
            infosets: combos.len() * 2,
            actions: 2,
            variant: CfrVariant::CfrPlus,
        },
        vec![true; combos.len() * 2 * 2],
    );
    let values = backend
        .public_tree_iteration_values(
            &nodes,
            &children,
            &[0],
            &combos,
            &vec![1; combos.len()],
            &vec![1.0; combos.len()],
            &boards,
            &state,
        )
        .unwrap_or_else(|error| fail(&format!("GPU public tree values failed: {error:?}")));

    for combo in 0..combos.len() {
        let offset = combo * 2;
        let fold = values.action_values[offset];
        let win = values.action_values[offset + 1];
        if (fold + 4.0).abs() >= 1e-5 || (win - 6.0).abs() >= 1e-5 {
            fail(&format!(
                "public tree action values combo {combo} mismatch: fold={fold}, win={win}, reach_weight={}, strategy_weight={}",
                values.reach_weights[combo], values.strategy_weights[combo],
            ));
        }
    }
}

fn gpu_combo(cards: [Card; 2]) -> GpuPrivateCombo {
    GpuPrivateCombo {
        cards: [cards[0].index() as u32, cards[1].index() as u32],
    }
}

fn assert_close(label: &str, expected: &[f32], actual: &[f32]) {
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        if (expected - actual).abs() >= 1e-5 {
            fail(&format!(
                "{label}[{index}] mismatch: expected {expected}, actual {actual}"
            ));
        }
    }
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}
