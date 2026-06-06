use pokedr_core::cards::{Card, Rank, Suit, evaluate};
use pokedr_core::dense_cfr::{
    BatchedPrivateCfrConfig, BatchedPrivateCfrState, CfrVariant, DenseCfrConfig, DenseCfrSolver,
    DenseCfrState,
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
    run_batched_private_update(&backend);
    run_showdown_equities(&backend);
    run_showdown_matrix(&backend);
    run_public_tree_terminal_values(&backend);
    run_public_tree_chance_blocks_fold_values(&backend);
    run_public_tree_showdown_values_match_bruteforce(&backend);
    run_public_tree_multistreet_values_match_cpu_exact(&backend);
    run_public_tree_average_strategy_br_matches_cpu_exact(&backend);
    run_public_tree_iterations_match_cpu_exact(&backend);
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
        CfrVariant::dcfr_plus_default(),
        CfrVariant::dcfr_schedule_default(),
        CfrVariant::pdcfr_plus_default(),
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
        assert_close("one-shot prediction", cpu.prediction(), gpu.prediction());
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
        CfrVariant::dcfr_plus_default(),
        CfrVariant::dcfr_schedule_default(),
        CfrVariant::pdcfr_plus_default(),
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
        assert_close(
            "resident prediction",
            cpu.state().prediction(),
            downloaded.prediction(),
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

fn run_batched_private_update(backend: &GpuDenseCfrBackend) {
    for variant in [
        CfrVariant::CfrPlus,
        CfrVariant::Discounted,
        CfrVariant::dcfr_plus_default(),
        CfrVariant::dcfr_schedule_default(),
        CfrVariant::pdcfr_plus_default(),
    ] {
        let batch_config = BatchedPrivateCfrConfig {
            batches: 3,
            public_infosets: 5,
            combos: 7,
            actions: 4,
            variant,
        };
        let legal_actions_per_public = vec![
            true, true, false, false, true, false, true, false, true, true, true, false, false,
            true, true, true, true, false, false, true,
        ];
        let batched = BatchedPrivateCfrState::new(batch_config.clone(), &legal_actions_per_public);
        let dense_config = DenseCfrConfig {
            infosets: batch_config.private_infosets(),
            actions: batch_config.actions,
            variant,
        };
        let mut cpu =
            DenseCfrState::new_with_legal_actions(dense_config, batched.legal_actions().to_vec());
        let mut gpu = backend.upload_batched_private_state(&batched);
        let mut action_values = vec![0.0; batch_config.action_slots()];
        let mut reach_weights = vec![0.0; batch_config.private_infosets()];
        let mut strategy_weights = vec![0.0; batch_config.private_infosets()];
        for iteration in 1..=4 {
            for (index, value) in action_values.iter_mut().enumerate() {
                *value = ((index + 3 * iteration) as f32 * 0.125).cos();
            }
            for (index, value) in reach_weights.iter_mut().enumerate() {
                *value = 0.25 + ((index + iteration) % 5) as f32 * 0.1;
            }
            for (index, value) in strategy_weights.iter_mut().enumerate() {
                *value = 0.5 + ((index + 2 * iteration) % 7) as f32 * 0.05;
            }
            cpu.update_all_infosets(&action_values, &reach_weights, &strategy_weights, iteration);
            gpu.update_all_private_infosets(
                backend,
                &action_values,
                &reach_weights,
                &strategy_weights,
                iteration,
            )
            .unwrap_or_else(|error| fail(&format!("GPU batched private update failed: {error:?}")));
        }
        let downloaded = gpu.download(backend).unwrap_or_else(|error| {
            fail(&format!("GPU batched private download failed: {error:?}"))
        });

        assert_eq!(downloaded.config().batches, batch_config.batches);
        assert_eq!(
            downloaded.config().public_infosets,
            batch_config.public_infosets
        );
        assert_close(
            "batched private regret",
            cpu.regrets(),
            downloaded.regrets(),
        );
        assert_close(
            "batched private strategy_sum",
            cpu.strategy_sum(),
            downloaded.strategy_sum(),
        );
        assert_close(
            "batched private prediction",
            cpu.prediction(),
            downloaded.prediction(),
        );
    }
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
            infosets: combos.len(),
            actions: 2,
            variant: CfrVariant::CfrPlus,
        },
        vec![true; combos.len() * 2],
    );
    let values = backend
        .public_tree_iteration_values(
            &nodes,
            &children,
            &[0],
            &combos,
            &vec![1; combos.len()],
            &vec![1.0; combos.len()],
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

fn run_public_tree_chance_blocks_fold_values(backend: &GpuDenseCfrBackend) {
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
            kind: 1,
            acting_player: 0,
            public_infoset: 0,
            first_child: 2,
            child_count: 2,
            terminal_kind: 0,
            showdown_offset: 0,
            _pad0: 0,
            pot: 0.0,
            hero_invested: 0.0,
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
    let children = vec![1, 2, 3, 4];
    let child_cards = vec![
        52,
        52,
        Card::new(Rank::Ace, Suit::Spades).index() as u32,
        Card::new(Rank::Queen, Suit::Hearts).index() as u32,
    ];
    let state = DenseCfrState::new_with_legal_actions(
        DenseCfrConfig {
            infosets: combos.len(),
            actions: 2,
            variant: CfrVariant::CfrPlus,
        },
        vec![true; combos.len() * 2],
    );
    let values = backend
        .public_tree_iteration_values(
            &nodes,
            &children,
            &child_cards,
            &combos,
            &vec![1; combos.len()],
            &vec![1.0; combos.len()],
            &vec![1.0, 0.0, 1.0],
            &[],
            &state,
        )
        .unwrap_or_else(|error| fail(&format!("GPU public tree chance values failed: {error:?}")));

    let blocked_combo_offset = 0;
    let blocked_chance_action = values.action_values[blocked_combo_offset + 1];
    if blocked_chance_action.abs() >= 1e-5 {
        fail(&format!(
            "chance-blocked fold value leaked invalid hero/villain branch: expected 0, actual {blocked_chance_action}"
        ));
    }
}

fn run_public_tree_showdown_values_match_bruteforce(backend: &GpuDenseCfrBackend) {
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
            terminal_kind: 2,
            showdown_offset: 0,
            _pad0: 1,
            pot: 10.0,
            hero_invested: 4.0,
            _pad1: 1.0,
            _pad2: 1.0,
        },
    ];
    let children = vec![1, 2];
    let board = GpuFinalBoard {
        cards: [
            Card::new(Rank::Two, Suit::Clubs).index() as u32,
            Card::new(Rank::Three, Suit::Diamonds).index() as u32,
            Card::new(Rank::Four, Suit::Hearts).index() as u32,
            Card::new(Rank::Five, Suit::Spades).index() as u32,
            Card::new(Rank::Six, Suit::Clubs).index() as u32,
        ],
    };
    let villain_weights = vec![0.2, 0.7, 1.3];
    let state = DenseCfrState::new_with_legal_actions(
        DenseCfrConfig {
            infosets: combos.len(),
            actions: 2,
            variant: CfrVariant::CfrPlus,
        },
        vec![true; combos.len() * 2],
    );
    let values = backend
        .public_tree_iteration_values(
            &nodes,
            &children,
            &[52, 52],
            &combos,
            &vec![1; combos.len()],
            &vec![1.0; combos.len()],
            &villain_weights,
            &[board],
            &state,
        )
        .unwrap_or_else(|error| fail(&format!("GPU public tree showdown failed: {error:?}")));

    for hero in 0..combos.len() {
        let expected = expected_showdown_action_value(&combos, &villain_weights, board, hero);
        let actual = values.action_values[hero * 2 + 1];
        if (expected - actual).abs() >= 1e-5 {
            fail(&format!(
                "public tree showdown action combo {hero} mismatch: expected {expected}, actual {actual}"
            ));
        }
    }
}

fn run_public_tree_multistreet_values_match_cpu_exact(backend: &GpuDenseCfrBackend) {
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
    let chance_cards = vec![
        Card::new(Rank::Ace, Suit::Spades),
        Card::new(Rank::King, Suit::Spades),
        Card::new(Rank::Ace, Suit::Clubs),
        Card::new(Rank::Ace, Suit::Diamonds),
        Card::new(Rank::Queen, Suit::Hearts),
        Card::new(Rank::Jack, Suit::Hearts),
        Card::new(Rank::Two, Suit::Clubs),
        Card::new(Rank::Three, Suit::Diamonds),
    ];
    let chance_scale = 1.0 / (chance_cards.len() - 4) as f32;
    let mut nodes = vec![
        GpuPublicTreeNode {
            kind: 0,
            acting_player: 0,
            public_infoset: 0,
            first_child: 0,
            child_count: 2,
            terminal_kind: 0,
            showdown_offset: 0,
            _pad0: 0,
            pot: 12.0,
            hero_invested: 5.0,
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
            pot: 12.0,
            hero_invested: 5.0,
            _pad1: 1.0,
            _pad2: 0.0,
        },
        GpuPublicTreeNode {
            kind: 1,
            acting_player: 0,
            public_infoset: 0,
            first_child: 2,
            child_count: chance_cards.len() as u32,
            terminal_kind: 0,
            showdown_offset: 0,
            _pad0: 0,
            pot: 0.0,
            hero_invested: 0.0,
            _pad1: 1.0,
            _pad2: 0.0,
        },
    ];
    let mut children = vec![1, 2];
    let mut child_cards = vec![52, 52];
    let chance_first_child = children.len();
    children.resize(chance_first_child + chance_cards.len(), u32::MAX);
    child_cards.resize(chance_first_child + chance_cards.len(), 52);
    let mut boards = Vec::with_capacity(chance_cards.len());
    for (chance_index, chance_card) in chance_cards.iter().copied().enumerate() {
        let public_infoset = 1 + chance_index as u32;
        let decision_index = nodes.len() as u32;
        children[chance_first_child + chance_index] = decision_index;
        child_cards[chance_first_child + chance_index] = chance_card.index() as u32;

        let first_child = children.len() as u32;
        let fold_index = decision_index + 1;
        let showdown_index = decision_index + 2;
        children.extend([fold_index, showdown_index]);
        child_cards.extend([52, 52]);
        nodes.push(GpuPublicTreeNode {
            kind: 0,
            acting_player: 1,
            public_infoset,
            first_child,
            child_count: 2,
            terminal_kind: 0,
            showdown_offset: 0,
            _pad0: 0,
            pot: 16.0,
            hero_invested: 6.0,
            _pad1: chance_scale,
            _pad2: 0.0,
        });
        nodes.push(GpuPublicTreeNode {
            kind: 2,
            acting_player: 0,
            public_infoset: 0,
            first_child: 0,
            child_count: 0,
            terminal_kind: 1,
            showdown_offset: 0,
            _pad0: 0,
            pot: 16.0,
            hero_invested: 6.0,
            _pad1: chance_scale,
            _pad2: 0.0,
        });
        nodes.push(GpuPublicTreeNode {
            kind: 2,
            acting_player: 0,
            public_infoset: 0,
            first_child: 0,
            child_count: 0,
            terminal_kind: 2,
            showdown_offset: chance_index as u32,
            _pad0: 1,
            pot: 16.0,
            hero_invested: 6.0,
            _pad1: chance_scale,
            _pad2: 1.0,
        });
        boards.push(final_board_with_turn(chance_card));
    }

    let public_infosets = 1 + chance_cards.len();
    let mut state = DenseCfrState::new_with_legal_actions(
        DenseCfrConfig {
            infosets: public_infosets * combos.len(),
            actions: 2,
            variant: CfrVariant::CfrPlus,
        },
        vec![true; public_infosets * combos.len() * 2],
    );
    let seed_action_values: Vec<_> = (0..state.infosets() * state.actions())
        .map(|index| ((index as f32 + 0.5) * 0.37).sin())
        .collect();
    state.update_all_infosets(
        &seed_action_values,
        &vec![1.0; state.infosets()],
        &vec![1.0; state.infosets()],
        1,
    );
    let villain_weights = vec![0.35, 1.1, 0.8];
    let values = backend
        .public_tree_iteration_values(
            &nodes,
            &children,
            &child_cards,
            &combos,
            &vec![1; combos.len()],
            &vec![1.0; combos.len()],
            &villain_weights,
            &boards,
            &state,
        )
        .unwrap_or_else(|error| {
            fail(&format!(
                "GPU public tree multistreet values failed: {error:?}"
            ))
        });
    let expected = cpu_exact_public_tree_values(
        &nodes,
        &children,
        &child_cards,
        &combos,
        &villain_weights,
        &boards,
        &state,
        public_infosets,
        2,
    );

    assert_close(
        "multistreet action value",
        &expected.action_values,
        &values.action_values,
    );
    assert_close(
        "multistreet reach weight",
        &expected.reach_weights,
        &values.reach_weights,
    );
    assert_close(
        "multistreet strategy weight",
        &expected.strategy_weights,
        &values.strategy_weights,
    );
}

fn run_public_tree_average_strategy_br_matches_cpu_exact(backend: &GpuDenseCfrBackend) {
    let fixture = multistreet_public_tree_fixture();
    let mut state = DenseCfrState::new_with_legal_actions(
        DenseCfrConfig {
            infosets: fixture.public_infosets * fixture.combos.len(),
            actions: 2,
            variant: CfrVariant::DcfrPlus {
                alpha: 1.5,
                gamma: 4.0,
            },
        },
        vec![true; fixture.public_infosets * fixture.combos.len() * 2],
    );
    for iteration in 1..=4 {
        let action_values: Vec<_> = (0..state.infosets() * state.actions())
            .map(|index| ((index as f32 + iteration as f32 * 0.41) * 0.29).cos())
            .collect();
        let reach_weights: Vec<_> = (0..state.infosets())
            .map(|index| 0.25 + ((index + iteration) % 5) as f32 * 0.2)
            .collect();
        let strategy_weights: Vec<_> = (0..state.infosets())
            .map(|index| 0.5 + ((index * 3 + iteration) % 7) as f32 * 0.15)
            .collect();
        state.update_all_infosets(&action_values, &reach_weights, &strategy_weights, iteration);
    }
    let average_state = state.average_strategy_profile_state();

    for br_player in [0u32, 1u32] {
        let gpu = backend
            .public_tree_best_response_values(
                &fixture.nodes,
                &fixture.children,
                &fixture.child_cards,
                &fixture.combos,
                &vec![1; fixture.combos.len()],
                &vec![1.0; fixture.combos.len()],
                &fixture.villain_weights,
                &fixture.boards,
                &average_state,
                br_player,
            )
            .unwrap_or_else(|error| {
                fail(&format!(
                    "GPU public tree average strategy BR failed: {error:?}"
                ))
            });
        let expected = cpu_exact_public_tree_values_with_br(
            &fixture.nodes,
            &fixture.children,
            &fixture.child_cards,
            &fixture.combos,
            &fixture.villain_weights,
            &fixture.boards,
            &average_state,
            fixture.public_infosets,
            2,
            Some(br_player as usize),
        );
        assert_close_player_actions(
            &format!("average strategy BR {br_player} action value"),
            &expected.action_values,
            &gpu.action_values,
            &fixture.nodes,
            fixture.combos.len(),
            2,
            br_player as usize,
        );
        assert_close_player_infosets(
            &format!("average strategy BR {br_player} reach weight"),
            &expected.reach_weights,
            &gpu.reach_weights,
            &fixture.nodes,
            fixture.combos.len(),
            br_player as usize,
        );
    }
}

fn run_public_tree_iterations_match_cpu_exact(backend: &GpuDenseCfrBackend) {
    let fixture = multistreet_public_tree_fixture();
    let config = DenseCfrConfig {
        infosets: fixture.public_infosets * fixture.combos.len(),
        actions: 2,
        variant: CfrVariant::DcfrPlus {
            alpha: 1.5,
            gamma: 4.0,
        },
    };
    let legal_actions = vec![true; config.infosets * config.actions];
    let mut cpu = DenseCfrState::new_with_legal_actions(config.clone(), legal_actions.clone());
    let mut gpu = backend.zeroed_state_with_legal_actions(config, legal_actions);

    for iteration in 1..=5 {
        let values = cpu_exact_public_tree_values(
            &fixture.nodes,
            &fixture.children,
            &fixture.child_cards,
            &fixture.combos,
            &fixture.villain_weights,
            &fixture.boards,
            &cpu,
            fixture.public_infosets,
            2,
        );
        cpu.update_all_infosets(
            &values.action_values,
            &values.reach_weights,
            &values.strategy_weights,
            iteration,
        );
    }

    backend
        .public_tree_run_iterations(
            &fixture.nodes,
            &fixture.children,
            &fixture.child_cards,
            &fixture.combos,
            &vec![1; fixture.combos.len()],
            &vec![1.0; fixture.combos.len()],
            &fixture.villain_weights,
            &fixture.boards,
            &mut gpu,
            5,
        )
        .unwrap_or_else(|error| fail(&format!("GPU public tree CFR iterations failed: {error:?}")));
    let downloaded = gpu
        .download(backend)
        .unwrap_or_else(|error| fail(&format!("GPU public tree CFR download failed: {error:?}")));

    assert_close(
        "public tree CFR regret",
        cpu.regrets(),
        downloaded.regrets(),
    );
    assert_close(
        "public tree CFR strategy sum",
        cpu.strategy_sum(),
        downloaded.strategy_sum(),
    );
    assert_close(
        "public tree CFR prediction",
        cpu.prediction(),
        downloaded.prediction(),
    );
}

struct PublicTreeFixture {
    nodes: Vec<GpuPublicTreeNode>,
    children: Vec<u32>,
    child_cards: Vec<u32>,
    combos: Vec<GpuPrivateCombo>,
    villain_weights: Vec<f32>,
    boards: Vec<GpuFinalBoard>,
    public_infosets: usize,
}

fn multistreet_public_tree_fixture() -> PublicTreeFixture {
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
    let chance_cards = vec![
        Card::new(Rank::Ace, Suit::Spades),
        Card::new(Rank::King, Suit::Spades),
        Card::new(Rank::Ace, Suit::Clubs),
        Card::new(Rank::Ace, Suit::Diamonds),
        Card::new(Rank::Queen, Suit::Hearts),
        Card::new(Rank::Jack, Suit::Hearts),
        Card::new(Rank::Two, Suit::Clubs),
        Card::new(Rank::Three, Suit::Diamonds),
    ];
    let chance_scale = 1.0 / (chance_cards.len() - 4) as f32;
    let mut nodes = vec![
        GpuPublicTreeNode {
            kind: 0,
            acting_player: 0,
            public_infoset: 0,
            first_child: 0,
            child_count: 2,
            terminal_kind: 0,
            showdown_offset: 0,
            _pad0: 0,
            pot: 12.0,
            hero_invested: 5.0,
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
            pot: 12.0,
            hero_invested: 5.0,
            _pad1: 1.0,
            _pad2: 0.0,
        },
        GpuPublicTreeNode {
            kind: 1,
            acting_player: 0,
            public_infoset: 0,
            first_child: 2,
            child_count: chance_cards.len() as u32,
            terminal_kind: 0,
            showdown_offset: 0,
            _pad0: 0,
            pot: 0.0,
            hero_invested: 0.0,
            _pad1: 1.0,
            _pad2: 0.0,
        },
    ];
    let mut children = vec![1, 2];
    let mut child_cards = vec![52, 52];
    let chance_first_child = children.len();
    children.resize(chance_first_child + chance_cards.len(), u32::MAX);
    child_cards.resize(chance_first_child + chance_cards.len(), 52);
    let mut boards = Vec::with_capacity(chance_cards.len());
    for (chance_index, chance_card) in chance_cards.iter().copied().enumerate() {
        let public_infoset = 1 + chance_index as u32;
        let decision_index = nodes.len() as u32;
        children[chance_first_child + chance_index] = decision_index;
        child_cards[chance_first_child + chance_index] = chance_card.index() as u32;

        let first_child = children.len() as u32;
        let fold_index = decision_index + 1;
        let showdown_index = decision_index + 2;
        children.extend([fold_index, showdown_index]);
        child_cards.extend([52, 52]);
        nodes.push(GpuPublicTreeNode {
            kind: 0,
            acting_player: 1,
            public_infoset,
            first_child,
            child_count: 2,
            terminal_kind: 0,
            showdown_offset: 0,
            _pad0: 0,
            pot: 16.0,
            hero_invested: 6.0,
            _pad1: chance_scale,
            _pad2: 0.0,
        });
        nodes.push(GpuPublicTreeNode {
            kind: 2,
            acting_player: 0,
            public_infoset: 0,
            first_child: 0,
            child_count: 0,
            terminal_kind: 1,
            showdown_offset: 0,
            _pad0: 0,
            pot: 16.0,
            hero_invested: 6.0,
            _pad1: chance_scale,
            _pad2: 0.0,
        });
        nodes.push(GpuPublicTreeNode {
            kind: 2,
            acting_player: 0,
            public_infoset: 0,
            first_child: 0,
            child_count: 0,
            terminal_kind: 2,
            showdown_offset: chance_index as u32,
            _pad0: 1,
            pot: 16.0,
            hero_invested: 6.0,
            _pad1: chance_scale,
            _pad2: 1.0,
        });
        boards.push(final_board_with_turn(chance_card));
    }

    PublicTreeFixture {
        nodes,
        children,
        child_cards,
        combos,
        villain_weights: vec![0.35, 1.1, 0.8],
        boards,
        public_infosets: 1 + chance_cards.len(),
    }
}

fn final_board_with_turn(turn: Card) -> GpuFinalBoard {
    let mut cards = [
        turn.index() as u32,
        Card::new(Rank::Four, Suit::Hearts).index() as u32,
        Card::new(Rank::Five, Suit::Spades).index() as u32,
        Card::new(Rank::Six, Suit::Clubs).index() as u32,
        Card::new(Rank::Seven, Suit::Diamonds).index() as u32,
    ];
    if cards[0] == cards[1] {
        cards[1] = Card::new(Rank::Eight, Suit::Hearts).index() as u32;
    }
    GpuFinalBoard { cards }
}

struct ExpectedPublicTreeValues {
    action_values: Vec<f32>,
    reach_weights: Vec<f32>,
    strategy_weights: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn cpu_exact_public_tree_values(
    nodes: &[GpuPublicTreeNode],
    children: &[u32],
    child_cards: &[u32],
    combos: &[GpuPrivateCombo],
    villain_weights: &[f32],
    boards: &[GpuFinalBoard],
    state: &DenseCfrState,
    public_infosets: usize,
    actions: usize,
) -> ExpectedPublicTreeValues {
    cpu_exact_public_tree_values_with_br(
        nodes,
        children,
        child_cards,
        combos,
        villain_weights,
        boards,
        state,
        public_infosets,
        actions,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn cpu_exact_public_tree_values_with_br(
    nodes: &[GpuPublicTreeNode],
    children: &[u32],
    child_cards: &[u32],
    combos: &[GpuPrivateCombo],
    villain_weights: &[f32],
    boards: &[GpuFinalBoard],
    state: &DenseCfrState,
    public_infosets: usize,
    actions: usize,
    br_player: Option<usize>,
) -> ExpectedPublicTreeValues {
    let infosets = public_infosets * combos.len();
    let mut values = ExpectedPublicTreeValues {
        action_values: vec![0.0; infosets * actions],
        reach_weights: vec![0.0; infosets],
        strategy_weights: vec![0.0; infosets],
    };
    let mut value_weights = vec![0.0; infosets * actions];
    for hero in 0..combos.len() {
        for villain in 0..combos.len() {
            if private_combos_collide(combos[hero], combos[villain]) {
                continue;
            }
            traverse_cpu_public_tree(
                nodes,
                children,
                child_cards,
                combos,
                boards,
                hero,
                villain,
                0,
                1.0,
                villain_weights[villain],
                1.0,
                state,
                actions,
                br_player,
                &mut values,
                &mut value_weights,
            );
        }
    }
    for (value, weight) in values.action_values.iter_mut().zip(value_weights) {
        if weight > 0.0 {
            *value /= weight;
        }
    }
    values.strategy_weights = cpu_exact_strategy_weights(
        nodes,
        children,
        child_cards,
        combos,
        villain_weights,
        state,
        public_infosets,
    );
    values
}

fn cpu_exact_strategy_weights(
    nodes: &[GpuPublicTreeNode],
    children: &[u32],
    child_cards: &[u32],
    combos: &[GpuPrivateCombo],
    villain_weights: &[f32],
    state: &DenseCfrState,
    public_infosets: usize,
) -> Vec<f32> {
    let combo_count = combos.len();
    let mut hero_reaches = vec![0.0f32; nodes.len() * combo_count];
    let mut villain_reaches = vec![0.0f32; nodes.len() * combo_count];
    for combo in 0..combo_count {
        hero_reaches[combo] = 1.0;
        villain_reaches[combo] = villain_weights[combo];
    }
    for node_index in 0..nodes.len() {
        let node = nodes[node_index];
        for combo in 0..combo_count {
            let hero_reach = hero_reaches[node_index * combo_count + combo];
            let villain_reach = villain_reaches[node_index * combo_count + combo];
            match node.kind {
                0 => {
                    let private_infoset = node.public_infoset as usize * combo_count + combo;
                    let mut strategy = vec![0.0; state.actions()];
                    state.strategy_for(private_infoset, &mut strategy);
                    for action in 0..node.child_count as usize {
                        let child = children[node.first_child as usize + action] as usize;
                        let probability = strategy[action];
                        if node.acting_player == 0 {
                            hero_reaches[child * combo_count + combo] = hero_reach * probability;
                            villain_reaches[child * combo_count + combo] = villain_reach;
                        } else {
                            hero_reaches[child * combo_count + combo] = hero_reach;
                            villain_reaches[child * combo_count + combo] =
                                villain_reach * probability;
                        }
                    }
                }
                1 => {
                    for action in 0..node.child_count as usize {
                        let child = children[node.first_child as usize + action] as usize;
                        let card = child_cards[node.first_child as usize + action];
                        if combo_has_card(combos[combo], card) {
                            continue;
                        }
                        hero_reaches[child * combo_count + combo] = hero_reach;
                        villain_reaches[child * combo_count + combo] = villain_reach;
                    }
                }
                _ => {}
            }
        }
    }

    let mut weights = vec![0.0; public_infosets * combo_count];
    for (node_index, node) in nodes.iter().copied().enumerate() {
        if node.kind != 0 {
            continue;
        }
        for combo in 0..combo_count {
            let infoset = node.public_infoset as usize * combo_count + combo;
            weights[infoset] = node._pad1
                * if node.acting_player == 0 {
                    hero_reaches[node_index * combo_count + combo]
                } else {
                    villain_reaches[node_index * combo_count + combo]
                };
        }
    }
    weights
}

#[allow(clippy::too_many_arguments)]
fn traverse_cpu_public_tree(
    nodes: &[GpuPublicTreeNode],
    children: &[u32],
    child_cards: &[u32],
    combos: &[GpuPrivateCombo],
    boards: &[GpuFinalBoard],
    hero: usize,
    villain: usize,
    node_index: usize,
    hero_reach: f32,
    villain_reach: f32,
    public_reach: f32,
    state: &DenseCfrState,
    actions: usize,
    br_player: Option<usize>,
    output: &mut ExpectedPublicTreeValues,
    value_weights: &mut [f32],
) -> f32 {
    let node = nodes[node_index];
    match node.kind {
        0 => {
            let acting_player = node.acting_player as usize;
            let acting_combo = if acting_player == 0 { hero } else { villain };
            let infoset = node.public_infoset as usize * combos.len() + acting_combo;
            let offset = infoset * actions;
            let mut strategy = vec![0.0; state.actions()];
            state.strategy_for(infoset, &mut strategy);
            let mut action_values = vec![0.0; node.child_count as usize];
            for action in 0..node.child_count as usize {
                let child = children[node.first_child as usize + action] as usize;
                let probability = strategy[action];
                let (next_hero_reach, next_villain_reach) = if acting_player == 0 {
                    (hero_reach * probability, villain_reach)
                } else {
                    (hero_reach, villain_reach * probability)
                };
                action_values[action] = traverse_cpu_public_tree(
                    nodes,
                    children,
                    child_cards,
                    combos,
                    boards,
                    hero,
                    villain,
                    child,
                    next_hero_reach,
                    next_villain_reach,
                    public_reach,
                    state,
                    actions,
                    br_player,
                    output,
                    value_weights,
                );
            }
            let opponent_reach = public_reach
                * if acting_player == 0 {
                    villain_reach
                } else {
                    hero_reach
                };
            let own_reach = public_reach
                * if acting_player == 0 {
                    hero_reach
                } else {
                    villain_reach
                };
            if opponent_reach > 0.0 || own_reach > 0.0 {
                for (action, &hero_value) in action_values.iter().enumerate() {
                    let player_value = if acting_player == 0 {
                        hero_value
                    } else {
                        -hero_value
                    };
                    output.action_values[offset + action] += opponent_reach * player_value;
                    value_weights[offset + action] += opponent_reach;
                }
                output.reach_weights[infoset] += opponent_reach;
                output.strategy_weights[infoset] += own_reach;
            }
            if br_player == Some(acting_player) {
                if acting_player == 0 {
                    action_values
                        .iter()
                        .copied()
                        .reduce(f32::max)
                        .unwrap_or(0.0)
                } else {
                    action_values
                        .iter()
                        .copied()
                        .reduce(f32::min)
                        .unwrap_or(0.0)
                }
            } else {
                action_values
                    .iter()
                    .zip(strategy)
                    .map(|(value, probability)| value * probability)
                    .sum()
            }
        }
        1 => {
            let mut valid_children = Vec::new();
            for action in 0..node.child_count as usize {
                let card = child_cards[node.first_child as usize + action];
                if combo_has_card(combos[hero], card) || combo_has_card(combos[villain], card) {
                    continue;
                }
                valid_children.push(children[node.first_child as usize + action] as usize);
            }
            if valid_children.is_empty() {
                return 0.0;
            }
            let valid_count = valid_children.len();
            let child_public_reach = public_reach / valid_children.len() as f32;
            let mut sum = 0.0;
            for child in valid_children {
                sum += traverse_cpu_public_tree(
                    nodes,
                    children,
                    child_cards,
                    combos,
                    boards,
                    hero,
                    villain,
                    child,
                    hero_reach,
                    villain_reach,
                    child_public_reach,
                    state,
                    actions,
                    br_player,
                    output,
                    value_weights,
                );
            }
            sum / valid_count as f32
        }
        _ => match node.terminal_kind {
            0 => -node.hero_invested,
            1 => node.pot - node.hero_invested,
            2 => {
                let board = boards[node.showdown_offset as usize];
                node.pot * showdown_equity(combos[hero], combos[villain], board)
                    - node.hero_invested
            }
            _ => 0.0,
        },
    }
}

fn combo_has_card(combo: GpuPrivateCombo, card: u32) -> bool {
    combo.cards[0] == card || combo.cards[1] == card
}

fn expected_showdown_action_value(
    combos: &[GpuPrivateCombo],
    villain_weights: &[f32],
    board: GpuFinalBoard,
    hero: usize,
) -> f32 {
    let mut value = 0.0;
    let mut weight = 0.0;
    for (villain, villain_combo) in combos.iter().copied().enumerate() {
        if private_combos_collide(combos[hero], villain_combo) {
            continue;
        }
        let villain_weight = villain_weights[villain];
        value +=
            villain_weight * (10.0 * showdown_equity(combos[hero], villain_combo, board) - 4.0);
        weight += villain_weight;
    }
    value / weight
}

fn private_combos_collide(left: GpuPrivateCombo, right: GpuPrivateCombo) -> bool {
    left.cards[0] == right.cards[0]
        || left.cards[0] == right.cards[1]
        || left.cards[1] == right.cards[0]
        || left.cards[1] == right.cards[1]
}

fn showdown_equity(hero: GpuPrivateCombo, villain: GpuPrivateCombo, board: GpuFinalBoard) -> f32 {
    let hero_cards = [
        Card::from_index(hero.cards[0] as u8),
        Card::from_index(hero.cards[1] as u8),
    ];
    let villain_cards = [
        Card::from_index(villain.cards[0] as u8),
        Card::from_index(villain.cards[1] as u8),
    ];
    let board_cards = [
        Card::from_index(board.cards[0] as u8),
        Card::from_index(board.cards[1] as u8),
        Card::from_index(board.cards[2] as u8),
        Card::from_index(board.cards[3] as u8),
        Card::from_index(board.cards[4] as u8),
    ];
    let hero_strength = evaluate(&[
        hero_cards[0],
        hero_cards[1],
        board_cards[0],
        board_cards[1],
        board_cards[2],
        board_cards[3],
        board_cards[4],
    ]);
    let villain_strength = evaluate(&[
        villain_cards[0],
        villain_cards[1],
        board_cards[0],
        board_cards[1],
        board_cards[2],
        board_cards[3],
        board_cards[4],
    ]);
    if hero_strength > villain_strength {
        1.0
    } else if hero_strength == villain_strength {
        0.5
    } else {
        0.0
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

fn assert_close_player_actions(
    label: &str,
    expected: &[f32],
    actual: &[f32],
    nodes: &[GpuPublicTreeNode],
    combo_count: usize,
    actions: usize,
    player: usize,
) {
    let actors = public_infoset_actors(nodes);
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        let private_infoset = index / actions;
        let public_infoset = private_infoset / combo_count;
        if actors.get(public_infoset) == Some(&player) && (expected - actual).abs() >= 1e-5 {
            fail(&format!(
                "{label}[{index}] mismatch: expected {expected}, actual {actual}"
            ));
        }
    }
}

fn assert_close_player_infosets(
    label: &str,
    expected: &[f32],
    actual: &[f32],
    nodes: &[GpuPublicTreeNode],
    combo_count: usize,
    player: usize,
) {
    let actors = public_infoset_actors(nodes);
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        let public_infoset = index / combo_count;
        if actors.get(public_infoset) == Some(&player) && (expected - actual).abs() >= 1e-5 {
            fail(&format!(
                "{label}[{index}] mismatch: expected {expected}, actual {actual}"
            ));
        }
    }
}

fn public_infoset_actors(nodes: &[GpuPublicTreeNode]) -> Vec<usize> {
    let count = nodes
        .iter()
        .filter(|node| node.kind == 0)
        .map(|node| node.public_infoset as usize + 1)
        .max()
        .unwrap_or(0);
    let mut actors = vec![usize::MAX; count];
    for node in nodes {
        if node.kind == 0 {
            actors[node.public_infoset as usize] = node.acting_player as usize;
        }
    }
    actors
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}
