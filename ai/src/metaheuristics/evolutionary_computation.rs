use rand::distributions::{Distribution, UniformClosedOpen, UniformClosedOpen01};
use rand::rngs::Rng;

use super::Problem;

pub fn genetic_algorithm<R, P, const D: usize>(
    rng: &mut R,
    number_of_iterations: usize,
    population_size: usize,
    tournament_size: usize,
    crossover_probability: f64,
    mutation_probability: f64,
) -> [f64; D]
where
    R: Rng<<UniformClosedOpen<f64> as Distribution<f64>>::Backend>
        + Rng<<UniformClosedOpen01 as Distribution<f64>>::Backend>,
    P: Problem<D>,
{
    let mut population = vec![[0.0; D]; population_size];
    let mut fitnesses = vec![0.0; population_size];

    let mut best_individual = [0.0; D];
    let mut best_fitness = f64::INFINITY;

    for (individual, fitness) in population.iter_mut().zip(&mut fitnesses) {
        for (x, range) in individual.iter_mut().zip(&P::RANGES) {
            *x = rng.sample(&UniformClosedOpen::new(range.start, range.end));
        }

        *fitness = P::phi(individual);

        if *fitness < best_fitness {
            best_individual = *individual;
            best_fitness = *fitness;
        }
    }

    let mut new_population = vec![[0.0; D]; population_size];
    let mut new_fitnesses = vec![0.0; population_size];

    let mut indexes = (0..population_size).collect::<Vec<_>>();

    for _ in 0..number_of_iterations {
        // selection (tournament)
        for i in 0..population_size {
            let (j, _) = rand::util::partial_shuffle(rng, &mut indexes, tournament_size)
                .iter()
                .map(|&i| (i, fitnesses[i]))
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            new_population[i] = population[j];
        }

        // recombination
        {
            let mut i = 0;

            loop {
                if rng.sample::<f64, _>(&UniformClosedOpen01) < crossover_probability {
                    let p1 = i;

                    loop {
                        if rng.sample::<f64, _>(&UniformClosedOpen01) < crossover_probability {
                            let p2 = i;

                            let a = rng.sample::<f64, _>(&UniformClosedOpen01);
                            for j in 0..D {
                                (new_population[p1][j], new_population[p2][j]) = (
                                    (1.0 - a) * new_population[p1][j] + a * new_population[p2][j],
                                    a * new_population[p1][j] + (1.0 - a) * new_population[p2][j],
                                );
                            }

                            break;
                        }

                        i += 1;
                        if i >= population_size {
                            break;
                        }
                    }
                }

                i += 1;
                if i >= population_size {
                    break;
                }
            }
        }

        // mutation
        for i in 0..population_size {
            for j in 0..D {
                if rng.sample::<f64, _>(&UniformClosedOpen01) < mutation_probability {
                    new_population[i][j] = rng.sample(&UniformClosedOpen::new(
                        P::RANGES[j].start,
                        P::RANGES[j].end,
                    ));
                }
            }
        }

        // evaluation
        for i in 0..population_size {
            new_fitnesses[i] = P::phi(&new_population[i]);

            if new_fitnesses[i] < best_fitness {
                best_individual = new_population[i];
                best_fitness = new_fitnesses[i];
            }
        }

        (population, fitnesses, new_population, new_fitnesses) =
            (new_population, new_fitnesses, population, fitnesses);
    }

    best_individual
}

pub fn differential_evolution<R, P, const D: usize>(
    rng: &mut R,
    number_of_iterations: usize,
    population_size: usize,
    crossover_probability: f64,
    differential_weight: f64,
) -> [f64; D]
where
    R: Rng<<UniformClosedOpen<f64> as Distribution<f64>>::Backend>
        + Rng<<UniformClosedOpen01 as Distribution<f64>>::Backend>,
    P: Problem<D>,
{
    let mut population = vec![[0.0; D]; population_size];
    let mut fitnesses = vec![0.0; population_size];

    for (individual, fitness) in population.iter_mut().zip(&mut fitnesses) {
        for (x, range) in individual.iter_mut().zip(&P::RANGES) {
            *x = rng.sample(&UniformClosedOpen::new(range.start, range.end));
        }

        *fitness = P::phi(individual);
    }

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

            let r = rng.sample(&UniformClosedOpen::new(0 as f64, D as f64)) as usize;
            for j in 0..D {
                let rj = rng.sample::<f64, _>(&UniformClosedOpen01);

                if rj < crossover_probability || j == r {
                    new_population[i][j] = population[r1][j]
                        + differential_weight * (population[r2][j] - population[r3][j]);
                } else {
                    new_population[i][j] = population[i][j];
                }
            }

            new_fitnesses[i] = P::phi(&new_population[i]);
        }

        for ((individual, fitness), (new_individual, new_fitness)) in population
            .iter_mut()
            .zip(&mut fitnesses)
            .zip(new_population.iter().zip(&new_fitnesses))
        {
            if new_fitness < fitness {
                *individual = *new_individual;
                *fitness = *new_fitness;
            }
        }
    }

    let (result, _) = population
        .into_iter()
        .zip(fitnesses)
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    result
}
