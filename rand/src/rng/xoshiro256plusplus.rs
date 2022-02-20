/// This is xoshiro256++ 1.0, one of our all-purpose, rock-solid generators.
/// It has excellent (sub-ns) speed, a state (256 bits) that is large
/// enough for any parallel application, and it passes all tests we are
/// aware of.
///
/// For generating just floating-point numbers, xoshiro256+ is even faster.
///
/// The state must be seeded so that it is not everywhere zero. If you have
/// a 64-bit seed, we suggest to seed a splitmix64 generator and use its
/// output to fill s.
pub struct Xoshiro256PlusPlus {
    s: [u64; 4],
}

impl Xoshiro256PlusPlus {
    pub fn new(s: [u64; 4]) -> Self {
        Self { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = u64::wrapping_add(
            u64::wrapping_add(self.s[0], self.s[3]).rotate_left(23),
            self.s[0],
        );

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;

        self.s[3] = self.s[3].rotate_left(45);

        result
    }

    /// This is the jump function for the generator. It is equivalent
    /// to 2^128 calls to next(); it can be used to generate 2^128
    /// non-overlapping subsequences for parallel computations.
    pub fn jump(&mut self) {
        let mut s = [0; 4];

        for jump in [
            0x180ec6d33cfd0aba,
            0xd5a61266f0c9392c,
            0xa9582618e03fc9aa,
            0x39abdc4529b1661c,
        ] {
            for b in 0..64 {
                if (jump & 1u64 << b) != 0 {
                    s[0] ^= self.s[0];
                    s[1] ^= self.s[1];
                    s[2] ^= self.s[2];
                    s[3] ^= self.s[3];
                }
                self.next_u64();
            }
        }

        self.s = s;
    }

    /// This is the long-jump function for the generator. It is equivalent to
    /// 2^192 calls to next(); it can be used to generate 2^64 starting points,
    /// from each of which jump() will generate 2^64 non-overlapping
    /// subsequences for parallel distributed computations.
    pub fn long_jump(&mut self) {
        let mut s = [0; 4];

        for jump in [
            0x76e15d3efefdcbbf,
            0xc5004e441c522fb3,
            0x77710069854ee241,
            0x39109bb02acbe635,
        ] {
            for b in 0..64 {
                if (jump & 1u64 << b) != 0 {
                    s[0] ^= self.s[0];
                    s[1] ^= self.s[1];
                    s[2] ^= self.s[2];
                    s[3] ^= self.s[3];
                }
                self.next_u64();
            }
        }

        self.s = s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: [u64; 4] = [
        0xce124f618403c393,
        0x28d53c991db633b3,
        0x84e1e11761ad8d8f,
        0x3d51155d3a5e4243,
    ];

    #[test]
    fn next_u64_works() {
        let mut rng = Xoshiro256PlusPlus::new(SEED);

        for expected in [
            0x2d7180646f097545,
            0x86aaf8c154609c97,
            0xf687629849b6a7d3,
            0x8d13a4f86b93777b,
            0x0ed43bc88dd120bb,
            0xab596449d2a29e07,
            0x0463f42b963870b6,
            0x2e9c9e7c390013b0,
            0x34da10f1ab890efe,
            0x25485a30a1df6974,
            0xed8666c23fe61afc,
            0xeb69ac661237eb67,
            0xf97d197dce1e6d3e,
            0xfc489f3c86191389,
            0x3ecda9f634e9bb1e,
            0x94c0edf11713294d,
        ] {
            assert_eq!(rng.next_u64(), expected);
        }
    }

    #[test]
    fn jump_works() {
        let mut rng = Xoshiro256PlusPlus::new(SEED);

        rng.jump();

        for expected in [
            0x7c069472d3e99b3a,
            0x6b20dfe6144c3ee6,
            0x8b92ed074c0f71ce,
            0xbe5d371d73ecb140,
            0x485057ca4edde36e,
            0x9376f5ea133f6304,
            0xa6df8d3565c58840,
            0x94f72b006176d9e2,
            0x66ecbf7fef2d2be7,
            0x2abde98ec70bc1c3,
            0x5109c27b703ef4a3,
            0x4f5e741727ceb28e,
            0x988c4d0a0ee9be7a,
            0x196d403832ea8216,
            0x1e77f9283b2cb767,
            0x1382456f927f7711,
        ] {
            assert_eq!(rng.next_u64(), expected);
        }
    }

    #[test]
    fn long_jump_works() {
        let mut rng = Xoshiro256PlusPlus::new(SEED);

        rng.long_jump();

        for expected in [
            0x813a5ad905882ad7,
            0x3c3268bd26b419ce,
            0xa89e30e2460fd006,
            0x47d49dc700d15a47,
            0xdf4ed294f19f9364,
            0x927ad92a45e1c46d,
            0x2c8ad032cf5cfcc0,
            0x6dac7b4379b415d6,
            0xa76e7803b748cc34,
            0xd22689fe6bc1a018,
            0xd77e47971f6a1f9d,
            0x42535593bce799e1,
            0x366135ca59650266,
            0x0904391d802a2b9d,
            0x562a5eae79f517c9,
            0x4a85b73df86b63c3,
        ] {
            assert_eq!(rng.next_u64(), expected);
        }
    }
}
