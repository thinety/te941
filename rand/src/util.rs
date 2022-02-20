// Fisher-Yates shuffle
pub fn shuffle<R, T>(rng: &mut R, array: &mut [T])
where
    R: crate::rng::Rng<f64>,
{
    let n = array.len();
    for i in 0..n - 1 {
        let j = rng.gen_range((i as f64)..(n as f64)) as usize;
        array.swap(i, j);
    }
}
