use std::env;
use std::time::Instant;

use pokedr_core::blinds::blind_level;
use pokedr_core::cards::Card;
use pokedr_core::cfr::{CfrResult, KuhnCfrSolver, LeducCfrSolver};
use pokedr_core::postflop::{
    PostflopCfrConfig, PostflopCfrResult, PostflopComboFeatures, PostflopEquityReport,
    PostflopNode, PostflopNodeReport, PostflopRole, parse_range, postflop_combos,
    postflop_equity_reports, solve_postflop_check_bet,
};
use pokedr_core::river::{
    RiverCfrConfig, RiverCfrResult, RiverNode, RiverRole, river_blocker_reports, river_combos,
    solve_river_check_bet,
};
use pokedr_core::short_stack::{
    ShortStackConfig, ShortStackReport, analyze_open_2bb_defense, analyze_short_stack,
};
use pokedr_core::structure::STARTING_STACK;

const DEFAULT_MAX_BOARDS_PER_COMBO: usize = 32;
const DEFAULT_RANGE_SAMPLE_LIMIT: usize = 8;
const DEFAULT_MAX_ITERATIONS: usize = 8;
const DEFAULT_MAX_SPOT_ITERATIONS: usize = 2;
const DEFAULT_LEVEL: u32 = 9;
const DEFAULT_POSTFLOP_REALIZATION: f64 = 0.4;
const DEFAULT_FLAT_CALL_FRACTION: f64 = 0.25;

fn main() {
    let args: Vec<String> = env::args().collect();
    let solve_kuhn = has_flag(&args, "--solve-kuhn");
    let solve_leduc = has_flag(&args, "--solve-leduc");
    let river_blockers = has_flag(&args, "--river-blockers");
    let solve_river = has_flag(&args, "--solve-river");
    let postflop_equity = has_flag(&args, "--postflop-equity");
    let solve_postflop = has_flag(&args, "--solve-postflop");
    let scan_open2bb = has_flag(&args, "--scan-open2bb");
    let scan_defense2bb = has_flag(&args, "--scan-defense2bb");
    let level = parse_arg(&args, "--level").unwrap_or(DEFAULT_LEVEL);
    let stack = parse_arg(&args, "--stack").unwrap_or(STARTING_STACK);
    let alive_players = parse_arg(&args, "--alive").unwrap_or(6) as u8;
    let stacks = parse_stacks(&args).unwrap_or_else(|| vec![stack; alive_players as usize]);
    let players_behind =
        parse_arg(&args, "--behind").unwrap_or(alive_players.saturating_sub(1) as u32) as u8;
    let default_boards = if scan_open2bb || scan_defense2bb {
        1
    } else {
        DEFAULT_MAX_BOARDS_PER_COMBO as u32
    };
    let default_range_sample = if scan_open2bb || scan_defense2bb {
        1
    } else {
        DEFAULT_RANGE_SAMPLE_LIMIT as u32
    };
    let default_iterations = if scan_open2bb || scan_defense2bb {
        0
    } else {
        DEFAULT_MAX_ITERATIONS as u32
    };
    let default_spot_iterations = if scan_open2bb || scan_defense2bb {
        0
    } else {
        DEFAULT_MAX_SPOT_ITERATIONS as u32
    };
    let max_boards_per_combo = parse_arg(&args, "--boards").unwrap_or(default_boards) as usize;
    let range_sample_limit =
        parse_arg(&args, "--range-sample").unwrap_or(default_range_sample) as usize;
    let iterations = parse_arg(&args, "--iterations").unwrap_or(default_iterations) as usize;
    let spot_iterations =
        parse_arg(&args, "--spot-iterations").unwrap_or(default_spot_iterations) as usize;
    let elapsed_in_level_seconds = parse_arg(&args, "--elapsed").unwrap_or(0);
    let hand_duration_seconds = parse_arg(&args, "--hand-seconds").unwrap_or(20);
    let postflop_realization =
        parse_f64_arg(&args, "--postflop-realization").unwrap_or(DEFAULT_POSTFLOP_REALIZATION);
    let flat_call_fraction =
        parse_f64_arg(&args, "--flat-call-fraction").unwrap_or(DEFAULT_FLAT_CALL_FRACTION);
    let defender_jam_fraction_override = parse_f64_arg(&args, "--defender-jam-fraction");
    let include_overcall = has_flag(&args, "--overcall");
    let format = parse_string_arg(&args, "--format").unwrap_or("text");

    if solve_kuhn || solve_leduc {
        let iterations = parse_arg(&args, "--cfr-iterations").unwrap_or(100_000) as usize;
        let started_at = Instant::now();
        let result = if solve_leduc {
            let mut solver = LeducCfrSolver::new();
            solver.train(iterations)
        } else {
            let mut solver = KuhnCfrSolver::new();
            solver.train(iterations)
        };
        let elapsed = started_at.elapsed();
        match format {
            "json" => print_cfr_json(&result, elapsed.as_secs_f64()),
            "text" => print_cfr_report(&result, elapsed.as_secs_f64()),
            _ => {
                eprintln!("invalid --format; expected text or json");
                std::process::exit(2);
            }
        }
        return;
    }

    if river_blockers || solve_river {
        let Some(board) = parse_board_arg(&args) else {
            eprintln!("invalid --board; expected ten chars like AsKsQsJs2d");
            std::process::exit(2);
        };
        let top_fraction = parse_f64_arg(&args, "--top-fraction").unwrap_or(0.05);
        let limit = parse_arg(&args, "--limit").unwrap_or(40) as usize;
        if solve_river {
            let cfr_iterations = parse_arg(&args, "--cfr-iterations").unwrap_or(100_000) as usize;
            let pot = parse_f64_arg(&args, "--pot").unwrap_or(100.0);
            let bet = parse_f64_arg(&args, "--bet").unwrap_or(pot * 0.75);
            let result = solve_river_check_bet(RiverCfrConfig {
                board,
                pot,
                bet,
                iterations: cfr_iterations,
                top_fraction,
            });
            print_river_solution(&result, limit, format);
        } else {
            print_river_blockers(board, top_fraction, limit, format);
        }
        return;
    }

    if postflop_equity || solve_postflop {
        let Some(board) = parse_cards_arg(&args, "--board") else {
            eprintln!("invalid --board; expected 3 to 5 cards like AhKd7c");
            std::process::exit(2);
        };
        if !(3..=5).contains(&board.len()) {
            eprintln!("invalid --board; expected 3 to 5 cards");
            std::process::exit(2);
        }
        let Some(oop_range) = parse_string_arg(&args, "--oop-range") else {
            eprintln!("missing --oop-range");
            std::process::exit(2);
        };
        let Some(ip_range) = parse_string_arg(&args, "--ip-range") else {
            eprintln!("missing --ip-range");
            std::process::exit(2);
        };
        let max_runouts = parse_arg(&args, "--runouts").unwrap_or(256) as usize;
        let limit = parse_arg(&args, "--limit").unwrap_or(40) as usize;
        if solve_postflop {
            let cfr_iterations = parse_arg(&args, "--cfr-iterations").unwrap_or(100_000) as usize;
            let pot = parse_f64_arg(&args, "--pot").unwrap_or(100.0);
            let bet = parse_f64_arg(&args, "--bet").unwrap_or(pot * 0.75);
            let raise = parse_f64_arg(&args, "--raise").unwrap_or(bet * 3.0);
            let reraise = parse_f64_arg(&args, "--reraise").unwrap_or(raise * 2.0);
            print_postflop_solution(
                &board,
                oop_range,
                ip_range,
                max_runouts,
                cfr_iterations,
                pot,
                bet,
                raise,
                reraise,
                limit,
                format,
            );
        } else {
            print_postflop_equity(&board, oop_range, ip_range, max_runouts, limit, format);
        }
        return;
    }

    let Some(level) = blind_level(level as u8) else {
        eprintln!("invalid --level; expected 1..=16");
        std::process::exit(2);
    };

    if scan_open2bb {
        print_open2bb_scan(
            &args,
            alive_players,
            max_boards_per_combo,
            range_sample_limit,
            iterations,
            spot_iterations,
            elapsed_in_level_seconds,
            hand_duration_seconds,
            include_overcall,
        );
        return;
    }

    if scan_defense2bb {
        print_defense2bb_scan(
            &args,
            level,
            alive_players,
            stack,
            stacks.clone(),
            players_behind,
            max_boards_per_combo,
            range_sample_limit,
            iterations,
            spot_iterations,
            elapsed_in_level_seconds,
            hand_duration_seconds,
            include_overcall,
            postflop_realization,
            flat_call_fraction,
            defender_jam_fraction_override,
        );
        return;
    }

    let report = analyze_short_stack(&ShortStackConfig {
        level,
        alive_players: stacks.len() as u8,
        stack,
        stacks: stacks.clone(),
        players_behind,
        elapsed_in_level_seconds,
        hand_duration_seconds,
        max_boards_per_combo,
        range_sample_limit,
        iterations,
        spot_iterations,
        include_overcall,
        postflop_realization,
        flat_call_fraction,
        defender_jam_fraction_override,
    });

    match format {
        "json" => print_json_report(level.level, &stacks, players_behind, &report),
        "text" => print_report(level.level, &stacks, players_behind, &report),
        _ => {
            eprintln!("invalid --format; expected text or json");
            std::process::exit(2);
        }
    }
}

fn parse_arg(args: &[String], name: &str) -> Option<u32> {
    args.windows(2)
        .find(|window| window[0] == name)
        .and_then(|window| window[1].parse().ok())
}

fn parse_f64_arg(args: &[String], name: &str) -> Option<f64> {
    args.windows(2)
        .find(|window| window[0] == name)
        .and_then(|window| window[1].parse().ok())
}

fn parse_string_arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|window| window[0] == name)
        .map(|window| window[1].as_str())
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

fn parse_stacks(args: &[String]) -> Option<Vec<u32>> {
    let value = parse_string_arg(args, "--stacks")?;
    let stacks: Vec<u32> = value
        .split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect();

    (!stacks.is_empty()).then_some(stacks)
}

fn parse_board_arg(args: &[String]) -> Option<[Card; 5]> {
    let value = parse_string_arg(args, "--board")?;
    let cards = parse_cards(value)?;
    (cards.len() == 5).then_some([cards[0], cards[1], cards[2], cards[3], cards[4]])
}

fn parse_cards_arg(args: &[String], name: &str) -> Option<Vec<Card>> {
    let value = parse_string_arg(args, name)?;
    parse_cards(value)
}

fn parse_cards(value: &str) -> Option<Vec<Card>> {
    if value.len() % 2 != 0 {
        return None;
    }

    let chars: Vec<_> = value.chars().collect();
    let mut cards = Vec::with_capacity(chars.len() / 2);
    for index in (0..chars.len()).step_by(2) {
        cards.push(parse_card(chars[index], chars[index + 1])?);
    }

    let mut mask = 0_u64;
    for card in &cards {
        if mask & card.mask() != 0 {
            return None;
        }
        mask |= card.mask();
    }

    Some(cards)
}

fn parse_card(rank: char, suit: char) -> Option<Card> {
    let rank = match rank.to_ascii_uppercase() {
        'A' => 14,
        'K' => 13,
        'Q' => 12,
        'J' => 11,
        'T' => 10,
        '9' => 9,
        '8' => 8,
        '7' => 7,
        '6' => 6,
        '5' => 5,
        '4' => 4,
        '3' => 3,
        '2' => 2,
        _ => return None,
    };
    let suit = match suit.to_ascii_lowercase() {
        'c' => 0,
        'd' => 1,
        'h' => 2,
        's' => 3,
        _ => return None,
    };
    Some(Card::new(rank, suit))
}

fn print_river_blockers(board: [Card; 5], top_fraction: f64, limit: usize, format: &str) {
    let combos = river_combos(board);
    let reports = river_blocker_reports(board, &combos, &combos, top_fraction);
    let mut reports = reports;
    reports.sort_by(|left, right| {
        right
            .blocked_top_combos
            .cmp(&left.blocked_top_combos)
            .then_with(|| right.hero.value.cmp(&left.hero.value))
            .then_with(|| left.hero.label().cmp(&right.hero.label()))
    });

    match format {
        "json" => {
            println!("{{");
            println!("  \"game\": \"river-blockers\",");
            println!("  \"combo_count\": {},", combos.len());
            println!("  \"top_fraction\": {:.6},", top_fraction);
            println!("  \"reports\": [");
            for (index, report) in reports.iter().take(limit).enumerate() {
                let comma = if index + 1 == limit.min(reports.len()) {
                    ""
                } else {
                    ","
                };
                println!(
                    "    {{\"combo\":\"{}\",\"hand_value\":{},\"blocked_villain_combos\":{},\"blocked_top_combos\":{},\"total_villain_combos\":{},\"top_villain_combos\":{}}}{}",
                    report.hero.label(),
                    report.hero.value,
                    report.blocked_villain_combos,
                    report.blocked_top_combos,
                    report.total_villain_combos,
                    report.top_villain_combos,
                    comma
                );
            }
            println!("  ]");
            println!("}}");
        }
        "text" => {
            println!("river blocker report");
            println!("combo count: {}", combos.len());
            println!("top fraction: {:.3}", top_fraction);
            println!("combo,hand_value,blocked_top,total_top,blocked_all,total_all");
            for report in reports.iter().take(limit) {
                println!(
                    "{},{},{},{},{},{}",
                    report.hero.label(),
                    report.hero.value,
                    report.blocked_top_combos,
                    report.top_villain_combos,
                    report.blocked_villain_combos,
                    report.total_villain_combos
                );
            }
        }
        _ => {
            eprintln!("invalid --format; expected text or json");
            std::process::exit(2);
        }
    }
}

fn print_postflop_equity(
    board: &[Card],
    oop_range: &str,
    ip_range: &str,
    max_runouts: usize,
    limit: usize,
    format: &str,
) {
    let oop_classes = match parse_range(oop_range) {
        Ok(classes) => classes,
        Err(_) => {
            eprintln!("invalid --oop-range");
            std::process::exit(2);
        }
    };
    let ip_classes = match parse_range(ip_range) {
        Ok(classes) => classes,
        Err(_) => {
            eprintln!("invalid --ip-range");
            std::process::exit(2);
        }
    };
    let oop_combos = postflop_combos(&oop_classes, board);
    let ip_combos = postflop_combos(&ip_classes, board);
    let oop_reports = postflop_equity_reports(board, &oop_combos, &ip_combos, max_runouts);
    let ip_reports = postflop_equity_reports(board, &ip_combos, &oop_combos, max_runouts);

    match format {
        "json" => {
            println!("{{");
            println!("  \"game\": \"postflop-equity\",");
            println!("  \"board_cards\": {},", board.len());
            println!("  \"oop_combo_count\": {},", oop_combos.len());
            println!("  \"ip_combo_count\": {},", ip_combos.len());
            println!("  \"max_runouts\": {},", max_runouts);
            println!("  \"oop\": [");
            print_postflop_json_reports(&oop_reports, limit, "    ");
            println!("  ],");
            println!("  \"ip\": [");
            print_postflop_json_reports(&ip_reports, limit, "    ");
            println!("  ]");
            println!("}}");
        }
        "text" => {
            println!("postflop equity");
            println!("board cards: {}", board.len());
            println!("OOP combos: {}", oop_combos.len());
            println!("IP combos: {}", ip_combos.len());
            println!("max runouts per matchup: {}", max_runouts);
            println!();
            println!("OOP");
            print_postflop_text_reports(&oop_reports, limit);
            println!();
            println!("IP");
            print_postflop_text_reports(&ip_reports, limit);
        }
        _ => {
            eprintln!("invalid --format; expected text or json");
            std::process::exit(2);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn print_postflop_solution(
    board: &[Card],
    oop_range: &str,
    ip_range: &str,
    max_runouts: usize,
    iterations: usize,
    pot: f64,
    bet: f64,
    raise: f64,
    reraise: f64,
    limit: usize,
    format: &str,
) {
    let oop_classes = match parse_range(oop_range) {
        Ok(classes) => classes,
        Err(_) => {
            eprintln!("invalid --oop-range");
            std::process::exit(2);
        }
    };
    let ip_classes = match parse_range(ip_range) {
        Ok(classes) => classes,
        Err(_) => {
            eprintln!("invalid --ip-range");
            std::process::exit(2);
        }
    };
    let result = solve_postflop_check_bet(PostflopCfrConfig {
        board: board.to_vec(),
        oop_range: postflop_combos(&oop_classes, board),
        ip_range: postflop_combos(&ip_classes, board),
        pot,
        bet,
        raise,
        reraise,
        iterations,
        max_runouts,
    });
    print_postflop_solution_result(&result, limit, format);
}

fn print_postflop_solution_result(result: &PostflopCfrResult, limit: usize, format: &str) {
    let mut strategies = result.strategies.clone();
    strategies.sort_by(|left, right| {
        postflop_action_frequency(right, "bet")
            .total_cmp(&postflop_action_frequency(left, "bet"))
            .then_with(|| {
                postflop_action_frequency(right, "call")
                    .total_cmp(&postflop_action_frequency(left, "call"))
            })
            .then_with(|| right.equity.total_cmp(&left.equity))
            .then_with(|| left.combo.label().cmp(&right.combo.label()))
    });

    match format {
        "json" => {
            println!("{{");
            println!("  \"game\": \"postflop-check-bet\",");
            println!("  \"iterations\": {},", result.iterations);
            println!(
                "  \"expected_value_oop\": {:.9},",
                result.expected_value_oop
            );
            println!("  \"pot\": {:.6},", result.pot);
            println!("  \"bet\": {:.6},", result.bet);
            println!("  \"raise\": {:.6},", result.raise);
            println!("  \"reraise\": {:.6},", result.reraise);
            println!("  \"board_cards\": {},", result.board_cards);
            println!("  \"oop_combo_count\": {},", result.oop_combo_count);
            println!("  \"ip_combo_count\": {},", result.ip_combo_count);
            println!("  \"limitations\": [");
            for (index, limitation) in result.limitations.iter().enumerate() {
                let comma = if index + 1 == result.limitations.len() {
                    ""
                } else {
                    ","
                };
                println!("    \"{}\"{}", limitation, comma);
            }
            println!("  ],");
            println!("  \"node_reports\": [");
            let node_reports = postflop_display_node_reports(result);
            for (index, report) in node_reports.iter().enumerate() {
                let comma = if index + 1 == node_reports.len() {
                    ""
                } else {
                    ","
                };
                print_postflop_node_report_json(report, limit, "    ", comma);
            }
            println!("  ],");
            println!("  \"strategies\": [");
            for (index, strategy) in strategies.iter().take(limit).enumerate() {
                let comma = if index + 1 == limit.min(strategies.len()) {
                    ""
                } else {
                    ","
                };
                println!(
                    "    {{\"role\":\"{}\",\"node\":\"{}\",\"board\":\"{}\",\"combo\":\"{}\",\"class\":\"{}\",\"equity\":{:.9},\"features\":{},\"actions\":[",
                    postflop_role_label(strategy.role),
                    postflop_node_label(strategy.node),
                    cards_label(&strategy.board),
                    strategy.combo.label(),
                    strategy.combo.class.label(),
                    strategy.equity,
                    postflop_features_json(strategy.features)
                );
                for (action_index, action) in strategy.actions.iter().enumerate() {
                    let action_comma = if action_index + 1 == strategy.actions.len() {
                        ""
                    } else {
                        ","
                    };
                    println!(
                        "      {{\"action\":\"{}\",\"frequency\":{:.9}}}{}",
                        action.action, action.frequency, action_comma
                    );
                }
                println!("    ]}}{}", comma);
            }
            println!("  ]");
            println!("}}");
        }
        "text" => {
            println!("postflop check-bet CFR");
            println!("iterations: {}", result.iterations);
            println!("expected value OOP: {:.6}", result.expected_value_oop);
            println!("pot: {:.3}", result.pot);
            println!("bet: {:.3}", result.bet);
            println!("raise: {:.3}", result.raise);
            println!("reraise: {:.3}", result.reraise);
            println!("board cards: {}", result.board_cards);
            println!("OOP combos: {}", result.oop_combo_count);
            println!("IP combos: {}", result.ip_combo_count);
            println!("limitations:");
            for limitation in &result.limitations {
                println!("- {}", limitation);
            }
            println!("node reports:");
            for report in postflop_display_node_reports(result) {
                println!(
                    "{} {} {} combos={} top_entries",
                    postflop_role_label(report.role),
                    postflop_node_label(report.node),
                    cards_label(&report.board),
                    report.combo_count
                );
                for entry in report.entries.iter().take(limit.min(12)) {
                    let actions = entry
                        .actions
                        .iter()
                        .map(|action| format!("{}={:.3}", action.action, action.frequency))
                        .collect::<Vec<_>>()
                        .join(" ");
                    println!(
                        "  {:<5} {:<4} weight={:.3} eq={:.3} sc={} gap={} pair_block={} flush_block={} top_block={} {}",
                        entry.combo.label(),
                        entry.combo.class.label(),
                        entry.range_weight,
                        entry.equity,
                        entry.features.is_suited_connector,
                        entry.features.connector_gap,
                        entry.features.blocks_board_pair,
                        entry.features.blocks_board_flush_suit,
                        entry.features.blocks_top_villain,
                        actions
                    );
                }
            }
            println!("strategy rows:");
            println!(
                "role,node,board,combo,class,equity,suited,connector_gap,suited_connector,blocks_board_pair,blocks_board_flush_suit,blocks_top_villain,actions"
            );
            for strategy in strategies.iter().take(limit) {
                let actions = strategy
                    .actions
                    .iter()
                    .map(|action| format!("{}={:.6}", action.action, action.frequency))
                    .collect::<Vec<_>>()
                    .join(" ");
                println!(
                    "{},{},{},{},{},{:.6},{},{},{},{},{},{},{}",
                    postflop_role_label(strategy.role),
                    postflop_node_label(strategy.node),
                    cards_label(&strategy.board),
                    strategy.combo.label(),
                    strategy.combo.class.label(),
                    strategy.equity,
                    strategy.features.suited,
                    strategy.features.connector_gap,
                    strategy.features.is_suited_connector,
                    strategy.features.blocks_board_pair,
                    strategy.features.blocks_board_flush_suit,
                    strategy.features.blocks_top_villain,
                    actions
                );
            }
        }
        _ => {
            eprintln!("invalid --format; expected text or json");
            std::process::exit(2);
        }
    }
}

fn postflop_display_node_reports(result: &PostflopCfrResult) -> Vec<&PostflopNodeReport> {
    let reports: Vec<_> = result
        .node_reports
        .iter()
        .filter(|report| report.board.len() == result.board_cards)
        .collect();
    if reports.is_empty() {
        result.node_reports.iter().collect()
    } else {
        reports
    }
}

fn print_postflop_node_report_json(
    report: &PostflopNodeReport,
    limit: usize,
    indent: &str,
    trailing_comma: &str,
) {
    println!("{indent}{{");
    println!(
        "{indent}  \"role\": \"{}\",",
        postflop_role_label(report.role)
    );
    println!(
        "{indent}  \"node\": \"{}\",",
        postflop_node_label(report.node)
    );
    println!("{indent}  \"board\": \"{}\",", cards_label(&report.board));
    println!("{indent}  \"combo_count\": {},", report.combo_count);
    println!("{indent}  \"entries\": [");
    for (index, entry) in report.entries.iter().take(limit).enumerate() {
        let comma = if index + 1 == limit.min(report.entries.len()) {
            ""
        } else {
            ","
        };
        println!(
            "{indent}    {{\"combo\":\"{}\",\"class\":\"{}\",\"equity\":{:.9},\"range_weight\":{:.9},\"features\":{},\"actions\":[",
            entry.combo.label(),
            entry.combo.class.label(),
            entry.equity,
            entry.range_weight,
            postflop_features_json(entry.features)
        );
        for (action_index, action) in entry.actions.iter().enumerate() {
            let action_comma = if action_index + 1 == entry.actions.len() {
                ""
            } else {
                ","
            };
            println!(
                "{indent}      {{\"action\":\"{}\",\"frequency\":{:.9}}}{}",
                action.action, action.frequency, action_comma
            );
        }
        println!("{indent}    ]}}{}", comma);
    }
    println!("{indent}  ]");
    println!("{indent}}}{trailing_comma}");
}

fn postflop_features_json(features: PostflopComboFeatures) -> String {
    format!(
        "{{\"suited\":{},\"connector_gap\":{},\"suited_connector\":{},\"blocks_board_pair\":{},\"blocks_board_flush_suit\":{},\"blocks_top_villain\":{}}}",
        features.suited,
        features.connector_gap,
        features.is_suited_connector,
        features.blocks_board_pair,
        features.blocks_board_flush_suit,
        features.blocks_top_villain
    )
}

fn cards_label(cards: &[Card]) -> String {
    cards.iter().map(|&card| card_label(card)).collect()
}

fn card_label(card: Card) -> String {
    let rank = match card.rank() {
        14 => 'A',
        13 => 'K',
        12 => 'Q',
        11 => 'J',
        10 => 'T',
        9 => '9',
        8 => '8',
        7 => '7',
        6 => '6',
        5 => '5',
        4 => '4',
        3 => '3',
        2 => '2',
        _ => '?',
    };
    let suit = match card.suit() {
        0 => 'c',
        1 => 'd',
        2 => 'h',
        3 => 's',
        _ => '?',
    };
    format!("{rank}{suit}")
}

fn postflop_action_frequency(
    strategy: &pokedr_core::postflop::PostflopComboStrategy,
    action: &str,
) -> f64 {
    strategy
        .actions
        .iter()
        .find(|frequency| frequency.action == action)
        .map(|frequency| frequency.frequency)
        .unwrap_or(0.0)
}

fn postflop_role_label(role: PostflopRole) -> &'static str {
    match role {
        PostflopRole::Oop => "OOP",
        PostflopRole::Ip => "IP",
    }
}

fn postflop_node_label(node: PostflopNode) -> &'static str {
    match node {
        PostflopNode::FacingBet => "facing-bet",
        PostflopNode::FacingRaise => "facing-raise",
        PostflopNode::FacingReraise => "facing-reraise",
        PostflopNode::AfterCheck => "after-check",
    }
}

fn print_postflop_text_reports(reports: &[PostflopEquityReport], limit: usize) {
    println!("combo,class,equity,win,tie,total");
    for report in reports.iter().take(limit) {
        println!(
            "{},{},{:.6},{:.0},{:.0},{:.0}",
            report.combo.label(),
            report.combo.class.label(),
            report.equity,
            report.win,
            report.tie,
            report.total
        );
    }
}

fn print_postflop_json_reports(reports: &[PostflopEquityReport], limit: usize, indent: &str) {
    for (index, report) in reports.iter().take(limit).enumerate() {
        let comma = if index + 1 == limit.min(reports.len()) {
            ""
        } else {
            ","
        };
        println!(
            "{indent}{{\"combo\":\"{}\",\"class\":\"{}\",\"equity\":{:.9},\"win\":{:.0},\"tie\":{:.0},\"total\":{:.0}}}{}",
            report.combo.label(),
            report.combo.class.label(),
            report.equity,
            report.win,
            report.tie,
            report.total,
            comma
        );
    }
}

fn print_river_solution(result: &RiverCfrResult, limit: usize, format: &str) {
    let mut strategies = result.strategies.clone();
    strategies.sort_by(|left, right| {
        river_action_frequency(right, "bet")
            .total_cmp(&river_action_frequency(left, "bet"))
            .then_with(|| {
                river_action_frequency(right, "call")
                    .total_cmp(&river_action_frequency(left, "call"))
            })
            .then_with(|| right.blocked_top_combos.cmp(&left.blocked_top_combos))
            .then_with(|| right.combo.value.cmp(&left.combo.value))
            .then_with(|| left.combo.label().cmp(&right.combo.label()))
    });

    match format {
        "json" => {
            println!("{{");
            println!("  \"game\": \"river-check-bet\",");
            println!("  \"iterations\": {},", result.iterations);
            println!(
                "  \"expected_value_oop\": {:.9},",
                result.expected_value_oop
            );
            println!("  \"pot\": {:.6},", result.pot);
            println!("  \"bet\": {:.6},", result.bet);
            println!("  \"combo_count\": {},", result.combo_count);
            println!("  \"strategies\": [");
            for (index, strategy) in strategies.iter().take(limit).enumerate() {
                let comma = if index + 1 == limit.min(strategies.len()) {
                    ""
                } else {
                    ","
                };
                println!(
                    "    {{\"role\":\"{}\",\"node\":\"{}\",\"combo\":\"{}\",\"hand_value\":{},\"blocked_top_combos\":{},\"top_villain_combos\":{},\"actions\":[",
                    river_role_label(strategy.role),
                    river_node_label(strategy.node),
                    strategy.combo.label(),
                    strategy.combo.value,
                    strategy.blocked_top_combos,
                    strategy.top_villain_combos
                );
                for (action_index, action) in strategy.actions.iter().enumerate() {
                    let action_comma = if action_index + 1 == strategy.actions.len() {
                        ""
                    } else {
                        ","
                    };
                    println!(
                        "      {{\"action\":\"{}\",\"frequency\":{:.9}}}{}",
                        action.action, action.frequency, action_comma
                    );
                }
                println!("    ]}}{}", comma);
            }
            println!("  ]");
            println!("}}");
        }
        "text" => {
            println!("river check-bet CFR");
            println!("iterations: {}", result.iterations);
            println!("expected value OOP: {:.6}", result.expected_value_oop);
            println!("pot: {:.3}", result.pot);
            println!("bet: {:.3}", result.bet);
            println!("combo count: {}", result.combo_count);
            println!("role,node,combo,hand_value,blocked_top,total_top,actions");
            for strategy in strategies.iter().take(limit) {
                let actions = strategy
                    .actions
                    .iter()
                    .map(|action| format!("{}={:.6}", action.action, action.frequency))
                    .collect::<Vec<_>>()
                    .join(" ");
                println!(
                    "{},{},{},{},{},{},{}",
                    river_role_label(strategy.role),
                    river_node_label(strategy.node),
                    strategy.combo.label(),
                    strategy.combo.value,
                    strategy.blocked_top_combos,
                    strategy.top_villain_combos,
                    actions
                );
            }
        }
        _ => {
            eprintln!("invalid --format; expected text or json");
            std::process::exit(2);
        }
    }
}

fn river_action_frequency(strategy: &pokedr_core::river::RiverComboStrategy, action: &str) -> f64 {
    strategy
        .actions
        .iter()
        .find(|frequency| frequency.action == action)
        .map(|frequency| frequency.frequency)
        .unwrap_or(0.0)
}

fn river_role_label(role: RiverRole) -> &'static str {
    match role {
        RiverRole::Oop => "OOP",
        RiverRole::Ip => "IP",
    }
}

fn river_node_label(node: RiverNode) -> &'static str {
    match node {
        RiverNode::FacingBet => "facing-bet",
        RiverNode::AfterCheck => "after-check",
    }
}

fn print_cfr_report(result: &CfrResult, elapsed_seconds: f64) {
    let iterations_per_second = if elapsed_seconds > 0.0 {
        result.iterations as f64 / elapsed_seconds
    } else {
        0.0
    };

    println!("{} poker CFR", result.game);
    println!("iterations: {}", result.iterations);
    println!("expected value P0: {:.6}", result.expected_value_p0);
    if result.game == "kuhn" {
        println!("known value P0: {:.6}", -1.0 / 18.0);
    }
    println!("elapsed seconds: {:.6}", elapsed_seconds);
    println!("iterations/sec: {:.0}", iterations_per_second);
    println!();
    println!("infoset,actions");
    for strategy in &result.infosets {
        let actions = strategy
            .actions
            .iter()
            .map(|action| format!("{}={:.6}", action.label, action.probability))
            .collect::<Vec<_>>()
            .join(" ");
        println!("{},{}", strategy.key, actions);
    }
}

fn print_cfr_json(result: &CfrResult, elapsed_seconds: f64) {
    let iterations_per_second = if elapsed_seconds > 0.0 {
        result.iterations as f64 / elapsed_seconds
    } else {
        0.0
    };

    println!("{{");
    println!("  \"game\": \"{}\",", result.game);
    println!("  \"iterations\": {},", result.iterations);
    println!("  \"expected_value_p0\": {:.9},", result.expected_value_p0);
    if result.game == "kuhn" {
        println!("  \"known_value_p0\": {:.9},", -1.0 / 18.0);
    }
    println!("  \"elapsed_seconds\": {:.9},", elapsed_seconds);
    println!("  \"iterations_per_second\": {:.3},", iterations_per_second);
    println!("  \"infosets\": [");
    for (index, strategy) in result.infosets.iter().enumerate() {
        let comma = if index + 1 == result.infosets.len() {
            ""
        } else {
            ","
        };
        println!("    {{\"key\":\"{}\",\"actions\":[", strategy.key);
        for (action_index, action) in strategy.actions.iter().enumerate() {
            let action_comma = if action_index + 1 == strategy.actions.len() {
                ""
            } else {
                ","
            };
            println!(
                "      {{\"label\":\"{}\",\"probability\":{:.9}}}{}",
                action.label, action.probability, action_comma
            );
        }
        println!("    ]}}{}", comma);
    }
    println!("  ]");
    println!("}}");
}

fn parse_u32_list(args: &[String], name: &str, default: &[u32]) -> Vec<u32> {
    parse_string_arg(args, name)
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse().ok())
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn parse_f64_list(args: &[String], name: &str, default: &[f64]) -> Vec<f64> {
    parse_string_arg(args, name)
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse().ok())
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn parse_string_list(args: &[String], name: &str, default: &[&str]) -> Vec<String> {
    parse_string_arg(args, name)
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| default.iter().map(|value| value.to_string()).collect())
}

#[allow(clippy::too_many_arguments)]
fn print_open2bb_scan(
    args: &[String],
    alive_players: u8,
    max_boards_per_combo: usize,
    range_sample_limit: usize,
    iterations: usize,
    spot_iterations: usize,
    elapsed_in_level_seconds: u32,
    hand_duration_seconds: u32,
    include_overcall: bool,
) {
    let levels = parse_u32_list(args, "--scan-levels", &[8, 9, 10]);
    let stack_bbs = parse_f64_list(args, "--scan-bbs", &[3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
    let realizations = parse_f64_list(args, "--scan-realizations", &[0.4]);
    let flat_fractions = parse_f64_list(args, "--scan-flat-fractions", &[0.0, 0.25]);
    let jam_fractions = parse_scan_jam_fractions(args);

    println!(
        "level,stack,stack_bb,postflop_realization,flat_call_fraction,defender_jam_fraction,seat,posted,open2bb_wfrac,ai_wfrac,fold_wfrac,open2bb_classes,ai_classes,top_open2bb,top_ai"
    );

    for level_number in levels {
        let Some(level) = blind_level(level_number as u8) else {
            continue;
        };
        for stack_bb in &stack_bbs {
            let stack = (stack_bb * level.big_blind as f64).round().max(1.0) as u32;
            let stacks = vec![stack; alive_players as usize];
            for &postflop_realization in &realizations {
                for &flat_call_fraction in &flat_fractions {
                    for &jam_fraction in &jam_fractions {
                        let report = analyze_short_stack(&ShortStackConfig {
                            level,
                            alive_players,
                            stack,
                            stacks: stacks.clone(),
                            players_behind: alive_players.saturating_sub(1),
                            elapsed_in_level_seconds,
                            hand_duration_seconds,
                            max_boards_per_combo,
                            range_sample_limit,
                            iterations,
                            spot_iterations,
                            include_overcall,
                            postflop_realization,
                            flat_call_fraction,
                            defender_jam_fraction_override: jam_fraction,
                        });

                        for seat in report.seats.iter().filter(|seat| seat.players_behind > 0) {
                            let open_wfrac =
                                clean_fraction(weighted_combo_fraction(&seat.open_2bb_range));
                            let ai_wfrac =
                                clean_fraction(weighted_combo_fraction(&seat.shove_range));
                            let fold_wfrac = clean_fraction(1.0 - open_wfrac - ai_wfrac);
                            println!(
                                "{},{},{:.3},{:.3},{:.3},{},{},{},{:.6},{:.6},{:.6},{},{},{},{}",
                                level.level,
                                stack,
                                *stack_bb,
                                postflop_realization,
                                flat_call_fraction,
                                jam_fraction_label(jam_fraction),
                                seat.seat_index,
                                seat.posted_amount,
                                open_wfrac,
                                ai_wfrac,
                                fold_wfrac,
                                seat.open_2bb_range.len(),
                                seat.shove_range.len(),
                                top_hands(&seat.open_2bb_range, 5),
                                top_hands(&seat.shove_range, 5)
                            );
                        }
                    }
                }
            }
        }
    }
}

fn parse_scan_jam_fractions(args: &[String]) -> Vec<Option<f64>> {
    let Some(value) = parse_string_arg(args, "--scan-jam-fractions") else {
        return vec![None, Some(0.0), Some(0.05), Some(0.1), Some(0.2)];
    };
    let values: Vec<_> = value
        .split(',')
        .filter_map(|part| {
            let part = part.trim();
            if part.eq_ignore_ascii_case("solver") {
                Some(None)
            } else {
                part.parse().ok().map(Some)
            }
        })
        .collect();

    if values.is_empty() {
        vec![None]
    } else {
        values
    }
}

fn jam_fraction_label(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "solver".to_string())
}

fn defense_profile_stacks(
    profile: &str,
    base_stack: u32,
    big_blind: u32,
    player_count: usize,
    opener_seat: usize,
    defender_seat: usize,
) -> Vec<u32> {
    let mut stacks = vec![base_stack; player_count];
    let critical = (3.0 * big_blind as f64).round() as u32;
    let medium = (6.0 * big_blind as f64).round() as u32;
    let defender_short = (8.0 * big_blind as f64).round() as u32;
    let opener_big = (30.0 * big_blind as f64).round() as u32;
    let short_seats: Vec<_> = (0..player_count)
        .filter(|seat| *seat != opener_seat && *seat != defender_seat)
        .collect();

    match profile {
        "one-critical-short" => {
            if let Some(&seat) = short_seats.first() {
                stacks[seat] = critical.min(base_stack);
            }
        }
        "one-medium-short" => {
            if let Some(&seat) = short_seats.first() {
                stacks[seat] = medium.min(base_stack);
            }
        }
        "two-shorts" => {
            if let Some(&seat) = short_seats.first() {
                stacks[seat] = critical.min(base_stack);
            }
            if let Some(&seat) = short_seats.get(1) {
                stacks[seat] = medium.min(base_stack);
            }
        }
        "defender-short" => {
            stacks[defender_seat] = defender_short.min(base_stack);
        }
        "opener-big" => {
            stacks[opener_seat] = opener_big.max(base_stack);
        }
        _ => {}
    }

    stacks
}

#[allow(clippy::too_many_arguments)]
fn print_defense2bb_scan(
    args: &[String],
    level: pokedr_core::blinds::BlindLevel,
    alive_players: u8,
    stack: u32,
    stacks: Vec<u32>,
    players_behind: u8,
    max_boards_per_combo: usize,
    range_sample_limit: usize,
    iterations: usize,
    spot_iterations: usize,
    elapsed_in_level_seconds: u32,
    hand_duration_seconds: u32,
    include_overcall: bool,
    postflop_realization: f64,
    flat_call_fraction: f64,
    defender_jam_fraction_override: Option<f64>,
) {
    let opener_seat = parse_arg(args, "--opener-seat").unwrap_or(0) as usize;
    let defender_seat = parse_arg(args, "--defender-seat")
        .unwrap_or(alive_players.saturating_sub(1) as u32) as usize;
    let opener_open_fractions = parse_f64_list(args, "--opener-open-fractions", &[0.25, 0.5, 1.0]);
    let opener_call_fractions =
        parse_f64_list(args, "--opener-call-fractions", &[0.0, 0.05, 0.1, 0.2]);
    let flat_realizations = parse_f64_list(args, "--defender-flat-realizations", &[0.2, 0.4, 0.6]);
    let stacks = if stacks.is_empty() {
        vec![stack; alive_players as usize]
    } else {
        stacks
    };

    println!(
        "level,stack_profile,stack,stack_bb,opener_seat,defender_seat,opener_open_fraction,opener_call_fraction,defender_flat_realization,hand,best_action,fold_value,flat_value,jam_value,flat_ev,jam_ev,flat_equity,jam_equity,bb_win_bb_delta,bb_win_opener_delta,bb_win_others_delta,bb_lose_bb_delta,bb_lose_opener_delta,bb_lose_others_delta"
    );
    let stack_bbs = parse_f64_list(args, "--scan-defense-bbs", &[]);
    let stack_profiles = parse_string_list(args, "--stack-profiles", &["equal"]);
    let stack_points: Vec<(u32, f64)> = if stack_bbs.is_empty() {
        let defender_stack = stacks.get(defender_seat).copied().unwrap_or(stack);
        vec![(
            defender_stack,
            defender_stack as f64 / level.big_blind as f64,
        )]
    } else {
        stack_bbs
            .iter()
            .map(|stack_bb| {
                (
                    (stack_bb * level.big_blind as f64).round().max(1.0) as u32,
                    *stack_bb,
                )
            })
            .collect()
    };

    for profile in stack_profiles {
        for (scan_stack, stack_bb) in stack_points.iter().copied() {
            let scan_stacks = defense_profile_stacks(
                &profile,
                scan_stack,
                level.big_blind,
                alive_players as usize,
                opener_seat,
                defender_seat,
            );
            for &opener_open_fraction in &opener_open_fractions {
                for &opener_call_fraction in &opener_call_fractions {
                    for &defender_flat_realization in &flat_realizations {
                        let config = ShortStackConfig {
                            level,
                            alive_players: scan_stacks.len() as u8,
                            stack: scan_stack,
                            stacks: scan_stacks.clone(),
                            players_behind,
                            elapsed_in_level_seconds,
                            hand_duration_seconds,
                            max_boards_per_combo,
                            range_sample_limit,
                            iterations,
                            spot_iterations,
                            include_overcall,
                            postflop_realization,
                            flat_call_fraction,
                            defender_jam_fraction_override,
                        };
                        for result in analyze_open_2bb_defense(
                            &config,
                            opener_seat,
                            defender_seat,
                            opener_open_fraction,
                            opener_call_fraction,
                            defender_flat_realization,
                        ) {
                            println!(
                                "{},{},{},{:.3},{},{},{:.3},{:.3},{:.3},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                                level.level,
                                profile,
                                scan_stack,
                                stack_bb,
                                opener_seat,
                                defender_seat,
                                opener_open_fraction,
                                opener_call_fraction,
                                defender_flat_realization,
                                result.hand.label(),
                                result.best_action.label(),
                                result.fold_value,
                                result.flat_value,
                                result.jam_value,
                                result.flat_value - result.fold_value,
                                result.jam_value - result.fold_value,
                                result.flat_equity,
                                result.jam_equity,
                                result.bb_win_bb_delta,
                                result.bb_win_opener_delta,
                                result.bb_win_others_delta,
                                result.bb_lose_bb_delta,
                                result.bb_lose_opener_delta,
                                result.bb_lose_others_delta
                            );
                        }
                    }
                }
            }
        }
    }
}

fn weighted_combo_fraction(range: &[pokedr_core::short_stack::HandResult]) -> f64 {
    range
        .iter()
        .map(|result| result.hand.combos().len() as f64 * result.frequency)
        .sum::<f64>()
        / 1326.0
}

fn clean_fraction(value: f64) -> f64 {
    if value.abs() < 1.0e-9 {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

fn top_hands(range: &[pokedr_core::short_stack::HandResult], limit: usize) -> String {
    range
        .iter()
        .take(limit)
        .map(|result| result.hand.label())
        .collect::<Vec<_>>()
        .join("|")
}

fn print_report(level: u8, stacks: &[u32], players_behind: u8, report: &ShortStackReport) {
    let stack = stacks.first().copied().unwrap_or_default();
    println!("level: {level}");
    println!("stacks: {:?}", stacks);
    println!(
        "seat 0 stack: {stack} ({:.2} BB)",
        report.stack_in_big_blinds
    );
    println!("alive players: {}", stacks.len());
    println!("players behind for first-in shove: {players_behind}");
    println!(
        "overall convergence: {}",
        if report.converged {
            "converged"
        } else {
            "not converged"
        }
    );
    println!("max iterations: {}", report.max_iterations);
    println!("max spot iterations: {}", report.max_spot_iterations);
    println!("overcall analyzed: {}", report.overcall_analyzed);
    println!(
        "postflop realization: {:.2}, flat call fraction: {:.1}%",
        report.postflop_realization,
        report.flat_call_fraction * 100.0
    );
    println!(
        "defender jam fraction: {}",
        report
            .defender_jam_fraction_override
            .map(|value| format!("{:.1}%", value * 100.0))
            .unwrap_or_else(|| "solver".to_string())
    );
    println!("dead pot: {}", report.dead_pot);
    println!(
        "orbit cost if everyone folds: {} ({:.1}% of stack)",
        report.orbit_cost,
        report.orbit_cost as f64 / stack as f64 * 100.0
    );
    println!(
        "chip pot odds: call {:.1}%, overcall {:.1}%",
        report.single_call_required_equity * 100.0,
        report.overcall_required_equity * 100.0
    );
    println!();
    for seat in &report.seats {
        println!(
            "seat {}: players behind {}, posted {}",
            seat.seat_index, seat.players_behind, seat.posted_amount
        );
        println!(
            "  response solve: {} / {} iterations, {}",
            seat.iterations_run,
            report.max_iterations,
            if seat.converged {
                "converged"
            } else {
                "not converged"
            }
        );
        println!(
            "  death-race required equity: call {:.1}%, overcall {:.1}%",
            seat.call_required_equity * 100.0,
            seat.overcall_required_equity * 100.0
        );
        if seat.players_behind == 0 {
            println!("  first-in all-in range: n/a (no players behind)");
            println!();
        } else {
            print_range("  first-in all-in range", &seat.shove_range, 40);
            print_range("  first-in open 2bb range", &seat.open_2bb_range, 40);
        }
        print_range("  call vs one all-in range", &seat.call_range, 40);
        for spot in &seat.call_spots {
            println!(
                "  call vs seat {} AI: effective {}, required {:.1}%, {} / {} iterations, {}",
                spot.opener_seat,
                spot.effective_all_in_cost,
                spot.required_equity * 100.0,
                spot.iterations_run,
                report.max_spot_iterations,
                if spot.converged {
                    "converged"
                } else {
                    "not converged"
                }
            );
            print_range("    range", &spot.range, 40);
            println!("    patterns: {}", spot.patterns.len());
            if let Some(next_response) = &spot.next_response {
                print_range(
                    &format!("    next seat {} response range", next_response.actor_seat),
                    &next_response.range,
                    40,
                );
            }
        }
        if report.overcall_analyzed {
            print_range("  overcall range vs jam+call", &seat.overcall_range, 40);
        } else {
            println!("  overcall range vs jam+call: skipped (pass --overcall to analyze)");
            println!();
        }
    }
}

fn print_json_report(level: u8, stacks: &[u32], players_behind: u8, report: &ShortStackReport) {
    let stack = stacks.first().copied().unwrap_or_default();
    println!("{{");
    println!("  \"level\": {level},");
    println!("  \"stack\": {stack},");
    println!(
        "  \"stacks\": [{}],",
        stacks
            .iter()
            .map(|stack| stack.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "  \"stack_in_big_blinds\": {:.6},",
        report.stack_in_big_blinds
    );
    println!("  \"alive_players\": {},", stacks.len());
    println!("  \"players_behind\": {players_behind},");
    println!("  \"converged\": {},", report.converged);
    println!("  \"max_iterations\": {},", report.max_iterations);
    println!("  \"max_spot_iterations\": {},", report.max_spot_iterations);
    println!("  \"overcall_analyzed\": {},", report.overcall_analyzed);
    println!(
        "  \"postflop_realization\": {:.6},",
        report.postflop_realization
    );
    println!(
        "  \"flat_call_fraction\": {:.6},",
        report.flat_call_fraction
    );
    println!(
        "  \"defender_jam_fraction_override\": {},",
        json_optional_f64(report.defender_jam_fraction_override)
    );
    println!("  \"dead_pot\": {},", report.dead_pot);
    println!("  \"orbit_cost\": {},", report.orbit_cost);
    println!(
        "  \"single_call_required_equity\": {:.6},",
        report.single_call_required_equity
    );
    println!(
        "  \"overcall_required_equity\": {:.6},",
        report.overcall_required_equity
    );
    println!("  \"seats\": [");
    for (index, seat) in report.seats.iter().enumerate() {
        let comma = if index + 1 == report.seats.len() {
            ""
        } else {
            ","
        };
        println!("    {{");
        println!("      \"seat_index\": {},", seat.seat_index);
        println!("      \"players_behind\": {},", seat.players_behind);
        println!("      \"posted_amount\": {},", seat.posted_amount);
        println!("      \"iterations_run\": {},", seat.iterations_run);
        println!("      \"converged\": {},", seat.converged);
        println!(
            "      \"call_required_equity\": {:.6},",
            seat.call_required_equity
        );
        println!(
            "      \"overcall_required_equity\": {:.6},",
            seat.overcall_required_equity
        );
        println!("      \"call_spots\": [");
        for (spot_index, spot) in seat.call_spots.iter().enumerate() {
            let spot_comma = if spot_index + 1 == seat.call_spots.len() {
                ""
            } else {
                ","
            };
            println!("        {{");
            println!("          \"opener_seat\": {},", spot.opener_seat);
            println!(
                "          \"effective_all_in_cost\": {},",
                spot.effective_all_in_cost
            );
            println!("          \"iterations_run\": {},", spot.iterations_run);
            println!("          \"converged\": {},", spot.converged);
            println!(
                "          \"required_equity\": {:.6},",
                spot.required_equity
            );
            print_json_range("range", &spot.range, true, 10);
            println!("          \"patterns\": [");
            for (pattern_index, pattern) in spot.patterns.iter().enumerate() {
                let pattern_comma = if pattern_index + 1 == spot.patterns.len() {
                    ""
                } else {
                    ","
                };
                println!("            {{");
                println!(
                    "              \"callers\": [{}],",
                    pattern
                        .callers
                        .iter()
                        .map(|seat| seat.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                );
                println!("              \"way\": {},", pattern.callers.len() + 2);
                println!("              \"probability\": {:.6},", pattern.probability);
                print_json_range("range", &pattern.range, false, 14);
                println!("            }}{pattern_comma}");
            }
            println!("          ],");
            print_json_response_node("next_response", spot.next_response.as_ref(), false, 10);
            println!("        }}{spot_comma}");
        }
        println!("      ],");
        println!("      \"ranges\": {{");
        print_json_range("first_in_all_in", &seat.shove_range, true, 8);
        print_json_range("first_in_open_2bb", &seat.open_2bb_range, true, 8);
        print_json_range("call_vs_one_all_in", &seat.call_range, true, 8);
        print_json_range("overcall_vs_jam_call", &seat.overcall_range, false, 8);
        println!("      }}");
        println!("    }}{comma}");
    }
    println!("  ]");
    println!("}}");
}

fn print_json_response_node(
    name: &str,
    node: Option<&pokedr_core::short_stack::ResponseNode>,
    trailing_comma: bool,
    indent: usize,
) {
    let pad = " ".repeat(indent);
    let Some(node) = node else {
        println!(
            "{pad}\"{name}\": null{}",
            if trailing_comma { "," } else { "" }
        );
        return;
    };

    println!("{pad}\"{name}\": {{");
    println!("{pad}  \"actor_seat\": {},", node.actor_seat);
    println!(
        "{pad}  \"prior_callers\": [{}],",
        node.prior_callers
            .iter()
            .map(|seat| seat.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    print_json_range("range", &node.range, true, indent + 2);
    print_json_response_node(
        "next_response",
        node.next_response.as_deref(),
        true,
        indent + 2,
    );
    print_json_response_node(
        "fold_response",
        node.fold_response.as_deref(),
        false,
        indent + 2,
    );
    println!("{pad}}}{}", if trailing_comma { "," } else { "" });
}

fn print_json_range(
    name: &str,
    range: &[pokedr_core::short_stack::HandResult],
    trailing_comma: bool,
    indent: usize,
) {
    let pad = " ".repeat(indent);
    let combo_count: usize = range.iter().map(|result| result.hand.combos().len()).sum();
    let weighted_combos: f64 = range
        .iter()
        .map(|result| result.hand.combos().len() as f64 * result.frequency)
        .sum::<f64>()
        .max(0.0);
    println!("{pad}\"{name}\": {{");
    println!("{pad}  \"classes\": {},", range.len());
    println!("{pad}  \"combos\": {combo_count},");
    println!(
        "{pad}  \"combo_fraction\": {:.6},",
        combo_count as f64 / 1326.0
    );
    println!("{pad}  \"weighted_combos\": {:.3},", weighted_combos);
    println!(
        "{pad}  \"weighted_combo_fraction\": {:.6},",
        weighted_combos / 1326.0
    );
    println!("{pad}  \"hands\": [");

    for (index, result) in range.iter().enumerate() {
        let comma = if index + 1 == range.len() { "" } else { "," };
        println!(
            "{pad}    {{\"hand\":\"{}\",\"equity\":{:.6},\"ev\":{:.6},\"frequency\":{:.6},\"call_value\":{},\"fold_value\":{}}}{comma}",
            result.hand.label(),
            result.equity,
            result.ev,
            result.frequency,
            json_optional_f64(result.call_value),
            json_optional_f64(result.fold_value)
        );
    }

    println!("{pad}  ]");
    println!("{pad}}}{}", if trailing_comma { "," } else { "" });
}

fn json_optional_f64(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "null".to_string())
}

fn print_range(title: &str, range: &[pokedr_core::short_stack::HandResult], limit: usize) {
    let combo_count: usize = range.iter().map(|result| result.hand.combos().len()).sum();

    println!(
        "{title}: {} classes, {} combos ({:.1}%)",
        range.len(),
        combo_count,
        combo_count as f64 / 1326.0 * 100.0
    );

    for chunk in range.chunks(16).take(limit.div_ceil(16)) {
        let line = chunk
            .iter()
            .map(|result| result.hand.label())
            .collect::<Vec<_>>()
            .join(" ");
        println!("  {line}");
    }

    println!();
}
