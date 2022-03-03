use crate::rngs::Rng;

use super::Distribution;

pub struct UniformClosedOpen01;

pub struct UniformClosedOpen<T> {
    start: T,
    end: T,
}
impl<T> UniformClosedOpen<T> {
    pub fn new(start: T, end: T) -> Self {
        Self { start, end }
    }
}

pub struct UniformOpenClosed01;

pub struct UniformOpenClosed<T> {
    start: T,
    end: T,
}
impl<T> UniformOpenClosed<T> {
    pub fn new(start: T, end: T) -> Self {
        Self { start, end }
    }
}

pub struct UniformOpenOpen01;

pub struct UniformOpenOpen<T> {
    start: T,
    end: T,
}
impl<T> UniformOpenOpen<T> {
    pub fn new(start: T, end: T) -> Self {
        Self { start, end }
    }
}

macro uniform_distributions_impl($fty:ty, $uty:ty, $exponent_bits:expr, $significant_bits:expr) {
    impl Distribution<$fty> for UniformClosedOpen01 {
        type Backend = $uty;

        fn sample<R>(&self, rng: &mut R) -> $fty
        where
            R: Rng<Self::Backend> + ?Sized,
        {
            ((rng.gen() >> $exponent_bits) as $fty) / (((1 as $uty) << $significant_bits) as $fty)
        }
    }

    impl Distribution<$fty> for UniformClosedOpen<$fty> {
        type Backend = <UniformClosedOpen01 as Distribution<$fty>>::Backend;

        fn sample<R>(&self, rng: &mut R) -> $fty
        where
            R: Rng<Self::Backend> + ?Sized,
        {
            rng.sample::<$fty, _>(&UniformClosedOpen01) * (self.end - self.start) + self.start
        }
    }

    impl Distribution<$fty> for UniformOpenClosed01 {
        type Backend = $uty;

        fn sample<R>(&self, rng: &mut R) -> $fty
        where
            R: Rng<Self::Backend> + ?Sized,
        {
            (((rng.gen() >> $exponent_bits) + 1) as $fty)
                / (((1 as $uty) << $significant_bits) as $fty)
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

    impl Distribution<$fty> for UniformOpenOpen01 {
        type Backend = $uty;

        fn sample<R>(&self, rng: &mut R) -> $fty
        where
            R: Rng<Self::Backend> + ?Sized,
        {
            (((rng.gen() >> $exponent_bits) | 1) as $fty)
                / (((1 as $uty) << $significant_bits) as $fty)
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

uniform_distributions_impl! { f32, u32,  8, 24 }
uniform_distributions_impl! { f64, u64, 11, 53 }
