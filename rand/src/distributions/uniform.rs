use crate::rngs::Rng;

use super::Distribution;

pub struct UniformClosedOpen01;

pub struct UniformOpenClosed01;

pub struct UniformOpenOpen01;

macro unit_uniform_distributions_impl($fty:ty, $uty:ty, $total_bits:expr, $significant_bits:expr) {
    impl Distribution<$fty> for UniformClosedOpen01 {
        type Backend = $uty;

        fn sample<R>(&self, rng: &mut R) -> $fty
        where
            R: Rng<Self::Backend> + ?Sized,
        {
            let value = rng.gen() >> ($total_bits - $significant_bits);

            let scale = 1.0 / (((1 as $uty) << $significant_bits) as $fty);

            scale * (value as $fty)
        }
    }

    impl Distribution<$fty> for UniformOpenClosed01 {
        type Backend = $uty;

        fn sample<R>(&self, rng: &mut R) -> $fty
        where
            R: Rng<Self::Backend> + ?Sized,
        {
            let value = rng.gen() >> ($total_bits - $significant_bits);

            let scale = 1.0 / (((1 as $uty) << $significant_bits) as $fty);

            scale * ((value + 1) as $fty)
        }
    }

    impl Distribution<$fty> for UniformOpenOpen01 {
        type Backend = $uty;

        fn sample<R>(&self, rng: &mut R) -> $fty
        where
            R: Rng<Self::Backend> + ?Sized,
        {
            let value = rng.gen() >> ($total_bits - $significant_bits);

            let scale = 1.0 / (((1 as $uty) << $significant_bits) as $fty);

            scale * ((value | 1) as $fty)
        }
    }
}

unit_uniform_distributions_impl! { f32, u32, 32, 24 }
unit_uniform_distributions_impl! { f64, u64, 64, 53 }

pub struct UniformClosedOpen<T> {
    start: T,
    end: T,
}
impl<T> UniformClosedOpen<T> {
    pub fn new(start: T, end: T) -> Self {
        Self { start, end }
    }
}

pub struct UniformOpenOpen<T> {
    start: T,
    end: T,
}
impl<T> UniformOpenOpen<T> {
    pub fn new(start: T, end: T) -> Self {
        Self { start, end }
    }
}

pub struct UniformOpenClosed<T> {
    start: T,
    end: T,
}
impl<T> UniformOpenClosed<T> {
    pub fn new(start: T, end: T) -> Self {
        Self { start, end }
    }
}

macro uniform_distributions_impl($fty:ty) {
    impl Distribution<$fty> for UniformClosedOpen<$fty> {
        type Backend = <UniformClosedOpen01 as Distribution<$fty>>::Backend;

        fn sample<R>(&self, rng: &mut R) -> $fty
        where
            R: Rng<Self::Backend> + ?Sized,
        {
            rng.sample::<$fty, _>(&UniformClosedOpen01) * (self.end - self.start) + self.start
        }
    }

    impl Distribution<$fty> for UniformOpenClosed<$fty> {
        type Backend = <UniformOpenClosed01 as Distribution<$fty>>::Backend;

        fn sample<R>(&self, rng: &mut R) -> $fty
        where
            R: Rng<Self::Backend> + ?Sized,
        {
            rng.sample::<$fty, _>(&UniformOpenClosed01) * (self.end - self.start) + self.start
        }
    }

    impl Distribution<$fty> for UniformOpenOpen<$fty> {
        type Backend = <UniformOpenOpen01 as Distribution<$fty>>::Backend;

        fn sample<R>(&self, rng: &mut R) -> $fty
        where
            R: Rng<Self::Backend> + ?Sized,
        {
            rng.sample::<$fty, _>(&UniformOpenOpen01) * (self.end - self.start) + self.start
        }
    }
}

uniform_distributions_impl! { f32 }
uniform_distributions_impl! { f64 }
