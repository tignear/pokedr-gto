use std::env;

use pokedr_core::blinds::blind_level;
use pokedr_core::short_stack::{ShortStackConfig, ShortStackReport, analyze_short_stack};

fn main() {
    let args: Vec<String> = env::args().collect();
    let level = parse_arg(&args, "--level").unwrap_or(11);
    let stack = parse_arg(&args, "--stack").unwrap_or(40_000);
    let alive_players = parse_arg(&args, "--alive").unwrap_or(6) as u8;
    let players_behind =
        parse_arg(&args, "--behind").unwrap_or(alive_players.saturating_sub(1) as u32) as u8;
    let max_boards_per_combo = parse_arg(&args, "--boards").unwrap_or(4) as usize;
    let range_sample_limit = parse_arg(&args, "--range-sample").unwrap_or(18) as usize;
    let iterations = parse_arg(&args, "--iterations").unwrap_or(2) as usize;

    let Some(level) = blind_level(level as u8) else {
        eprintln!("invalid --level; expected 1..=16");
        std::process::exit(2);
    };

    let report = analyze_short_stack(&ShortStackConfig {
        level,
        alive_players,
        stack,
        players_behind,
        max_boards_per_combo,
        range_sample_limit,
        iterations,
    });

    print_report(level.level, stack, alive_players, players_behind, &report);
}

fn parse_arg(args: &[String], name: &str) -> Option<u32> {
    args.windows(2)
        .find(|window| window[0] == name)
        .and_then(|window| window[1].parse().ok())
}

fn print_report(
    level: u8,
    stack: u32,
    alive_players: u8,
    players_behind: u8,
    report: &ShortStackReport,
) {
    println!("level: {level}");
    println!("stack: {stack} ({:.2} BB)", report.stack_in_big_blinds);
    println!("alive players: {alive_players}");
    println!("players behind for first-in shove: {players_behind}");
    println!("dead pot: {}", report.dead_pot);
    println!(
        "orbit cost if everyone folds: {} ({:.1}% of stack)",
        report.orbit_cost,
        report.orbit_cost as f64 / stack as f64 * 100.0
    );
    println!(
        "required equity: call {:.1}%, overcall {:.1}%",
        report.single_call_required_equity * 100.0,
        report.overcall_required_equity * 100.0
    );
    println!();
    print_range("first-in all-in range", &report.shove_range, 40);
    print_range("call vs one all-in range", &report.call_range, 40);
    print_range("overcall range vs jam+call", &report.overcall_range, 40);
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
