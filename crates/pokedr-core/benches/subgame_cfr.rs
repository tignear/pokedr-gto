use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use pokedr_core::cards::Card;
use pokedr_core::postflop::parse_range;
use pokedr_core::subgame::{
    ActionAbstraction, BetSize, ChancePolicy, PotState, RangeState, SubgameSpec,
};

fn c(rank: u8, suit: u8) -> Card {
    Card::new(rank, suit)
}

fn small_subgame() -> SubgameSpec {
    SubgameSpec::postflop(
        vec![c(14, 0), c(13, 1), c(2, 2)],
        PotState::new(100.0, [1_000.0, 1_000.0]),
        RangeState::new(
            parse_range("AA,AKs,AKo").unwrap(),
            parse_range("QQ,JJ,AQs").unwrap(),
        ),
        ActionAbstraction {
            bet_sizes: vec![BetSize::PotFraction(0.5), BetSize::PotFraction(1.0)],
            raise_sizes: vec![BetSize::CurrentBetMultiple(2.5)],
            reraise_sizes: vec![BetSize::CurrentBetMultiple(2.0)],
            allow_all_in: false,
            max_raises: 2,
        },
        ChancePolicy::Sample(8),
    )
    .unwrap()
}

fn medium_subgame() -> SubgameSpec {
    SubgameSpec::postflop(
        vec![c(14, 0), c(13, 1), c(2, 2)],
        PotState::new(100.0, [1_000.0, 1_000.0]),
        RangeState::new(
            parse_range("AA,AKs,AKo,AQs,KQs").unwrap(),
            parse_range("QQ,JJ,AQs,KQs,QJs").unwrap(),
        ),
        ActionAbstraction {
            bet_sizes: vec![
                BetSize::PotFraction(0.33),
                BetSize::PotFraction(0.75),
                BetSize::PotFraction(1.25),
            ],
            raise_sizes: vec![
                BetSize::CurrentBetMultiple(2.5),
                BetSize::CurrentBetMultiple(3.5),
            ],
            reraise_sizes: vec![BetSize::CurrentBetMultiple(2.2)],
            allow_all_in: true,
            max_raises: 2,
        },
        ChancePolicy::Sample(16),
    )
    .unwrap()
}

fn wide_subgame() -> SubgameSpec {
    SubgameSpec::postflop(
        vec![c(14, 0), c(13, 1), c(2, 2)],
        PotState::new(100.0, [1_000.0, 1_000.0]),
        RangeState::new(
            parse_range(
                "22+,A2s+,K2s+,Q5s+,J7s+,T7s+,97s+,86s+,75s+,64s+,54s,A2o+,K8o+,Q9o+,J9o+,T9o",
            )
            .unwrap(),
            parse_range(
                "22+,A2s+,K2s+,Q4s+,J7s+,T7s+,97s+,86s+,75s+,64s+,54s,A2o+,K7o+,Q9o+,J9o+,T9o",
            )
            .unwrap(),
        ),
        ActionAbstraction {
            bet_sizes: vec![
                BetSize::PotFraction(0.25),
                BetSize::PotFraction(0.5),
                BetSize::PotFraction(0.75),
                BetSize::PotFraction(1.25),
            ],
            raise_sizes: vec![
                BetSize::CurrentBetMultiple(2.25),
                BetSize::CurrentBetMultiple(3.0),
                BetSize::CurrentBetMultiple(4.0),
            ],
            reraise_sizes: vec![
                BetSize::CurrentBetMultiple(2.2),
                BetSize::CurrentBetMultiple(3.0),
            ],
            allow_all_in: true,
            max_raises: 2,
        },
        ChancePolicy::Sample(16),
    )
    .unwrap()
}

fn bench_subgame_cfr(c: &mut Criterion) {
    let mut group = c.benchmark_group("subgame_cfr");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));

    let small = small_subgame();
    group.bench_function("small_500_iters", |b| {
        b.iter(|| small.solve_cfr(500).unwrap())
    });

    let medium = medium_subgame();
    group.bench_function("medium_5000_iters", |b| {
        b.iter(|| medium.solve_cfr(5_000).unwrap())
    });

    let wide = wide_subgame();
    group.bench_function("wide_1000_iters", |b| {
        b.iter(|| wide.solve_cfr(1_000).unwrap())
    });

    group.finish();
}

criterion_group!(benches, bench_subgame_cfr);
criterion_main!(benches);
