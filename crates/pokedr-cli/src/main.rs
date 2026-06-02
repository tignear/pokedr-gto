use std::env;

use pokedr_core::blinds::blind_level;
use pokedr_core::short_stack::{ShortStackConfig, ShortStackReport, analyze_short_stack};
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
    let level = parse_arg(&args, "--level").unwrap_or(DEFAULT_LEVEL);
    let stack = parse_arg(&args, "--stack").unwrap_or(STARTING_STACK);
    let alive_players = parse_arg(&args, "--alive").unwrap_or(6) as u8;
    let stacks = parse_stacks(&args).unwrap_or_else(|| vec![stack; alive_players as usize]);
    let players_behind =
        parse_arg(&args, "--behind").unwrap_or(alive_players.saturating_sub(1) as u32) as u8;
    let max_boards_per_combo =
        parse_arg(&args, "--boards").unwrap_or(DEFAULT_MAX_BOARDS_PER_COMBO as u32) as usize;
    let range_sample_limit =
        parse_arg(&args, "--range-sample").unwrap_or(DEFAULT_RANGE_SAMPLE_LIMIT as u32) as usize;
    let iterations =
        parse_arg(&args, "--iterations").unwrap_or(DEFAULT_MAX_ITERATIONS as u32) as usize;
    let spot_iterations = parse_arg(&args, "--spot-iterations")
        .unwrap_or(DEFAULT_MAX_SPOT_ITERATIONS as u32) as usize;
    let elapsed_in_level_seconds = parse_arg(&args, "--elapsed").unwrap_or(0);
    let hand_duration_seconds = parse_arg(&args, "--hand-seconds").unwrap_or(20);
    let postflop_realization =
        parse_f64_arg(&args, "--postflop-realization").unwrap_or(DEFAULT_POSTFLOP_REALIZATION);
    let flat_call_fraction =
        parse_f64_arg(&args, "--flat-call-fraction").unwrap_or(DEFAULT_FLAT_CALL_FRACTION);
    let include_overcall = has_flag(&args, "--overcall");
    let format = parse_string_arg(&args, "--format").unwrap_or("text");

    let Some(level) = blind_level(level as u8) else {
        eprintln!("invalid --level; expected 1..=16");
        std::process::exit(2);
    };

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
