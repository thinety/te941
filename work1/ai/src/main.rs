#![feature(generic_arg_infer)]

use std::f64::consts::{PI, TAU};

use ai::metaheuristics::{evolutionary_computation::*, Problem, Range};

use rand::rngs::Xoshiro256Plus;

enum TubularColumn {}

#[rustfmt::skip]
impl Problem<2> for TubularColumn {
    fn f(&[d, t]: &[f64; 2]) -> f64 {
        9.82 * d * t + 2.0 * d
    }

    const RANGES: [Range<f64>; 2] = [
        Range::new(2.0, 14.0),
        Range::new(0.2, 0.8),
    ];

    const INEQUALITIES: &'static [fn(&[f64; 2]) -> f64] = &[
        |&[d, t]| {
            5.0 / (PI * d * t) - 1.0
        },
        |&[d, t]| {
            1.0 / (6.8e-4 * PI * PI * PI * d * t * (d * d + t * t)) - 1.0
        },
        |&[d, _]| {
            2.0 / d - 1.0
        },
        |&[d, _]| {
            d / 14.0 - 1.0
        },
        |&[_, t]| {
            0.2 / t - 1.0
        },
        |&[_, t]| {
            t / 0.8 - 1.0
        },
    ];

    const PENALTY_WEIGHT: f64 = 1000.0;
}

enum TensionCompressionSpring {}

#[rustfmt::skip]
impl Problem<3> for TensionCompressionSpring {
    fn f(&[l, d, n]: &[f64; 3]) -> f64 {
        (n + 2.0) * d * l * l
    }

    const RANGES: [Range<f64>; 3] = [
        Range::new(0.05, 2.0),
        Range::new(0.25, 1.3),
        Range::new(2.0, 15.0),
    ];

    const INEQUALITIES: &'static [fn(&[f64; 3]) -> f64] = &[
        |&[l, d, n]| {
            1.0 - (d * d * d * n) / (71785.0 * l * l * l * l)
        },
        |&[l, d, _]| {
            (4.0 * d * d - l * d) / (12566.0 * l * l * l * (d - l)) + 1.0 / (5108.0 * l * l) - 1.0
        },
        |&[l, d, n]| {
            1.0 - (140.45 * l) / (d * d * n)
        },
        |&[l, d, _]| {
            (d + l) / 1.5 - 1.0
        },
    ];

    const PENALTY_WEIGHT: f64 = 100.0;
}

enum FMSound {}

#[rustfmt::skip]
impl Problem<6> for FMSound {
    fn f(&[a1, w1, a2, w2, a3, w3]: &[f64; 6]) -> f64 {
        (0..=100)
            .map(|t| {
                let t = (t as f64) * TAU / 100.0;

                let y = a1 * f64::sin(w1 * t + a2 * f64::sin(w2 * t + a3 * f64::sin(w3 * t)));

                let y_0 = 1.0 * f64::sin(5.0 * t - 1.5 * f64::sin(4.8 * t + 2.0 * f64::sin(4.9 * t)));

                let e = y - y_0;
                e * e
            })
            .sum()
    }

    const RANGES: [Range<f64>; 6] = [Range::new(-6.4, 6.35); 6];

    const INEQUALITIES: &'static [fn(&[f64; 6]) -> f64] = &[];

    const PENALTY_WEIGHT: f64 = 100.0;
}

fn main() {
    let number_of_runs = 30;

    let seed = [
        0x93920339ac7730ac,
        0x8db68f4acc7c22b1,
        0x8b804df6a99a1289,
        0xff5fa2f037375aa9,
    ];

    let mut results = Vec::with_capacity(number_of_runs);

    for i in 0..number_of_runs {
        let mut rng = Xoshiro256Plus::new(seed);
        for _ in 0..i {
            rng.jump();
        }

        // uncomment the desired combination of algorithm/problem
        let result = {
            genetic_algorithm::<_, TubularColumn, _>(
                &mut rng,
                1000,
                20,
                2,
                0.9,
                0.2,
            )
            // genetic_algorithm::<_, TubularColumn, _>(
            //     &mut rng,
            //     1000,
            //     20,
            //     2,
            //     0.9,
            //     0.15,
            // )
            // genetic_algorithm::<_, TubularColumn, _>(
            //     &mut rng,
            //     1000,
            //     20,
            //     2,
            //     0.9,
            //     0.1,
            // )
            // differential_evolution::<_, TubularColumn, _>(
            //     &mut rng,
            //     1000,
            //     20,
            //     0.9,
            //     0.8,
            // )
            // differential_evolution::<_, TubularColumn, _>(
            //     &mut rng,
            //     1000,
            //     20,
            //     0.9,
            //     0.5,
            // )
            // differential_evolution::<_, TubularColumn, _>(
            //     &mut rng,
            //     1000,
            //     20,
            //     0.8,
            //     0.5,
            // )

            // genetic_algorithm::<_, TensionCompressionSpring, _>(
            //     &mut rng,
            //     1000,
            //     30,
            //     3,
            //     0.9,
            //     0.2,
            // )
            // genetic_algorithm::<_, TensionCompressionSpring, _>(
            //     &mut rng,
            //     1000,
            //     30,
            //     3,
            //     0.9,
            //     0.15,
            // )
            // genetic_algorithm::<_, TensionCompressionSpring, _>(
            //     &mut rng,
            //     1000,
            //     30,
            //     3,
            //     0.9,
            //     0.1,
            // )
            // differential_evolution::<_, TensionCompressionSpring, _>(
            //     &mut rng,
            //     1000,
            //     30,
            //     0.9,
            //     0.8,
            // )
            // differential_evolution::<_, TensionCompressionSpring, _>(
            //     &mut rng,
            //     1000,
            //     30,
            //     0.9,
            //     0.5,
            // )
            // differential_evolution::<_, TensionCompressionSpring, _>(
            //     &mut rng,
            //     1000,
            //     30,
            //     0.8,
            //     0.5,
            // )

            // genetic_algorithm::<_, FMSound, _>(
            //     &mut rng,
            //     10000,
            //     60,
            //     6,
            //     0.9,
            //     0.2,
            // )
            // genetic_algorithm::<_, FMSound, _>(
            //     &mut rng,
            //     10000,
            //     60,
            //     6,
            //     0.9,
            //     0.15,
            // )
            // genetic_algorithm::<_, FMSound, _>(
            //     &mut rng,
            //     10000,
            //     60,
            //     6,
            //     0.9,
            //     0.1,
            // )
            // differential_evolution::<_, FMSound, _>(
            //     &mut rng,
            //     10000,
            //     60,
            //     0.9,
            //     0.8,
            // )
            // differential_evolution::<_, FMSound, _>(
            //     &mut rng,
            //     10000,
            //     60,
            //     0.9,
            //     0.5,
            // )
            // differential_evolution::<_, FMSound, _>(
            //     &mut rng,
            //     10000,
            //     60,
            //     0.8,
            //     0.5,
            // )
        };

        // uncomment the line with the desired problem
        results.push(TubularColumn::f(&result));
        // results.push(TensionCompressionSpring::f(&result));
        // results.push(FMSound::f(&result));
    }

    results.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let &min = results.first().unwrap();

    let mean = results.iter().sum::<f64>() / (number_of_runs as f64);

    let median = {
        let half = number_of_runs / 2;
        if number_of_runs % 2 == 0 {
            (results[half - 1] + results[half]) / 2.0
        } else {
            results[half]
        }
    };

    let &max = results.last().unwrap();

    let std_deviation = {
        let variance = results
            .iter()
            .map(|&value| {
                let diff = mean - value;
                diff * diff
            })
            .sum::<f64>()
            / (number_of_runs as f64);

        f64::sqrt(variance)
    };

    dbg!(min, mean, median, max, std_deviation);
}
