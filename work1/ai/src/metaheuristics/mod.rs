pub mod evolutionary_computation;
pub mod swarm_intelligence;

#[derive(Clone, Copy)]
pub struct Range<T> {
    pub start: T,
    pub end: T,
}

impl Range<f64> {
    pub const fn new(start: f64, end: f64) -> Self {
        Self { start, end }
    }
}

pub trait Problem<const D: usize> {
    fn f(x: &[f64; D]) -> f64;

    const RANGES: [Range<f64>; D];

    const INEQUALITIES: &'static [fn(&[f64; D]) -> f64];

    const PENALTY_WEIGHT: f64;

    fn phi(x: &[f64; D]) -> f64 {
        let mut y = Self::f(x);

        for (range, &xi) in Self::RANGES.iter().zip(x) {
            let d = range.start - xi;
            if d > 0.0 {
                y += Self::PENALTY_WEIGHT * d;

                // only one of the conditions can happen, so we can continue early
                continue;
            }

            let d = xi - range.end;
            if d > 0.0 {
                y += Self::PENALTY_WEIGHT * d;
            }
        }

        for inequality in Self::INEQUALITIES {
            let g = inequality(x);
            if g > 0.0 {
                y += Self::PENALTY_WEIGHT * g;
            }
        }

        y
    }
}
