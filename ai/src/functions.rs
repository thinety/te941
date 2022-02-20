pub mod ackley {
    pub const RANGE: std::ops::Range<f64> = -32.768..32.768;
    pub fn f<const D: usize>(x: &[f64; D], a: f64, b: f64, c: f64) -> f64 {
        a + std::f64::consts::E
            - a * f64::exp(
                -b * f64::sqrt(1.0 / (D as f64) * x.iter().map(|xi| xi.powi(2)).sum::<f64>()),
            )
            - f64::exp(1.0 / (D as f64) * x.iter().map(|xi| f64::cos(c * xi)).sum::<f64>())
    }
}

pub mod rastrigin {
    pub const RANGE: std::ops::Range<f64> = -5.12..5.12;
    pub fn f<const D: usize>(x: &[f64; D], a: f64) -> f64 {
        a * (D as f64)
            + x.iter()
                .map(|xi| xi.powi(2) - a * f64::cos(std::f64::consts::TAU * xi))
                .sum::<f64>()
    }
}

pub mod styblinski_tang {
    pub const RANGE: std::ops::Range<f64> = -5.0..5.0;
    pub fn f<const D: usize>(x: &[f64; D]) -> f64 {
        1.0 / 2.0
            * x.iter()
                .map(|xi| xi.powi(4) - 16.0 * xi.powi(2) + 5.0 * xi)
                .sum::<f64>()
    }
}

pub mod sphere {
    pub const RANGE: std::ops::Range<f64> = -100.0..100.0;
    pub fn f<const D: usize>(x: &[f64; D]) -> f64 {
        x.iter().map(|xi| xi.powi(2)).sum()
    }
}

pub mod himmelblau {
    pub const RANGE: std::ops::Range<f64> = -5.0..5.0;
    pub fn f([x, y]: &[f64; 2]) -> f64 {
        (x.powi(2) + y - 11.0).powi(2) + (x + y.powi(2) - 7.0).powi(2)
    }
}
