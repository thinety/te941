use crate::rngs::Rng;

mod uniform;

pub use uniform::{
    UniformClosedOpen, UniformClosedOpen01, UniformOpenClosed, UniformOpenClosed01,
    UniformOpenOpen, UniformOpenOpen01,
};

pub trait Distribution<T> {
    type Backend;

    fn sample<R>(&self, rng: &mut R) -> T
    where
        R: Rng<Self::Backend> + ?Sized;
}
