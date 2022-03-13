use crate::distributions::{Distribution, UniformClosedOpen};
use crate::rngs::Rng;

// Fisher-Yates shuffle
pub fn shuffle<R, T>(rng: &mut R, array: &mut [T])
where
    R: Rng<<UniformClosedOpen<f64> as Distribution<f64>>::Backend>,
{
    let n = array.len();

    for i in 0..n - 1 {
        let j = rng.sample(&UniformClosedOpen::new(i as f64, n as f64)) as usize;
        array.swap(i, j);
    }
}

// partial Fisher-Yates shuffle
pub fn partial_shuffle<'a, R, T>(rng: &mut R, array: &'a mut [T], amount: usize) -> &'a [T]
where
    R: Rng<<UniformClosedOpen<f64> as Distribution<f64>>::Backend>,
{
    let mut n = array.len();
    if amount < n {
        n = amount;
    }

    for i in 0..n - 1 {
        let j = rng.sample(&UniformClosedOpen::new(i as f64, n as f64)) as usize;
        array.swap(i, j);
    }

    &array[..n]
}
