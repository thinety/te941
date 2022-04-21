use crate::distributions::Distribution;

mod xoshiro128plus;
mod xoshiro128plusplus;
mod xoshiro256plus;
mod xoshiro256plusplus;

pub use xoshiro128plus::Xoshiro128Plus;
pub use xoshiro128plusplus::Xoshiro128PlusPlus;
pub use xoshiro256plus::Xoshiro256Plus;
pub use xoshiro256plusplus::Xoshiro256PlusPlus;

pub trait Rng<T> {
    fn gen(&mut self) -> T;

    fn sample<U, D>(&mut self, distribution: &D) -> U
    where
        D: Distribution<U, Backend = T>,
    {
        distribution.sample(self)
    }
}
