use std::ops::Range;

pub fn genetic_algorithm<R, const D: usize>(
    rng: &mut R,
    range: Range<f64>,
    f: fn(&[f64; D]) -> f64,
    number_of_iterations: usize,
    population_size: usize,
    tournament_size: usize,
    mutation_probability: f64,
) -> [f64; D]
where
    R: rand::rng::Rng<f64>,
{
    let mut population = vec![[0.0; D]; population_size];
    for x in &mut population {
        for xi in x {
            *xi = rng.gen_range(range.start..range.end);
        }
    }
    let mut fitnesses = population.iter().map(f).collect::<Vec<_>>();

    let mut new_population = vec![[0.0; D]; population_size];
    let mut new_fitnesses = vec![0.0; population_size];

    let mut indexes = (0..population_size).collect::<Vec<_>>();

    for _ in 0..number_of_iterations {
        for i in 0..population_size {
            let mut parents = [0; 2];
            for parent in &mut parents {
                rand::util::shuffle(rng, &mut indexes);
                let (result, _) = indexes[..tournament_size]
                    .iter()
                    .map(|&i| (i, fitnesses[i]))
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
                *parent = result;
            }

            let [p1, p2] = parents;
            let a = rng.gen_range(0.0..1.0);
            for j in 0..D {
                new_population[i][j] = (1.0 - a) * population[p1][j] + a * population[p2][j];
            }

            if rng.gen_range(0.0..1.0) < mutation_probability {
                let k = rng.gen_range((0 as f64)..(D as f64)) as usize;
                let new_xi = rng.gen_range(range.start..range.end);
                new_population[i][k] = new_xi;
            }

            new_fitnesses[i] = f(&new_population[i]);
        }

        (population, new_population) = (new_population, population);
        (fitnesses, new_fitnesses) = (new_fitnesses, fitnesses);
    }

    let (result, _) = std::iter::zip(population, fitnesses)
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    result
}

pub fn differential_evolution<R, const D: usize>(
    rng: &mut R,
    range: Range<f64>,
    f: fn(&[f64; D]) -> f64,
    number_of_iterations: usize,
    population_size: usize,
    crossover_probability: f64,
    differential_weight: f64,
) -> [f64; D]
where
    R: rand::rng::Rng<f64>,
{
    let mut population = vec![[0.0; D]; population_size];
    for x in &mut population {
        for xi in x {
            *xi = rng.gen_range(range.start..range.end);
        }
    }
    let mut fitnesses = population.iter().map(f).collect::<Vec<_>>();

    let mut new_population = vec![[0.0; D]; population_size];
    let mut new_fitnesses = vec![0.0; population_size];

    let mut indexes = (0..population_size).collect::<Vec<_>>();

    for _ in 0..number_of_iterations {
        rand::util::shuffle(rng, &mut indexes);

        for i in 0..population_size {
            let (r1, r2, r3) = (
                indexes[i % population_size],
                indexes[(i + 1) % population_size],
                indexes[(i + 2) % population_size],
            );

            let r = rng.gen_range((0 as f64)..(D as f64)) as usize;
            for j in 0..D {
                let rj = rng.gen_range(0.0..1.0);

                if rj < crossover_probability || j == r {
                    new_population[i][j] = population[r1][j]
                        + differential_weight * (population[r2][j] - population[r3][j]);
                } else {
                    new_population[i][j] = population[i][j];
                }
            }

            new_fitnesses[i] = f(&new_population[i]);
        }

        for ((x, y), (new_x, new_y)) in std::iter::zip(
            std::iter::zip(&mut population, &mut fitnesses),
            std::iter::zip(&new_population, &new_fitnesses),
        ) {
            if new_y < y {
                *x = *new_x;
                *y = *new_y;
            }
        }
    }

    let (result, _) = std::iter::zip(population, fitnesses)
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    result
}
