use std::ops::Range;

pub mod xoshiro128plus;
pub mod xoshiro128plusplus;
pub mod xoshiro256plus;
pub mod xoshiro256plusplus;

pub trait Rng<T> {
    fn gen_range(&mut self, range: Range<T>) -> T;
}

impl Rng<f32> for xoshiro128plus::Xoshiro128Plus {
    fn gen_range(&mut self, range: Range<f32>) -> f32 {
        self.next_f32() * (range.end - range.start) + range.start
    }
}

impl Rng<f64> for xoshiro256plus::Xoshiro256Plus {
    fn gen_range(&mut self, range: Range<f64>) -> f64 {
        self.next_f64() * (range.end - range.start) + range.start
    }
}
