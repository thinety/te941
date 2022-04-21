use super::Rng;

/// This is xoshiro128+ 1.0, our best and fastest 32-bit generator for 32-bit
/// floating-point numbers. We suggest to use its upper bits for
/// floating-point generation, as it is slightly faster than xoshiro128**.
/// It passes all tests we are aware of except for
/// linearity tests, as the lowest four bits have low linear complexity, so
/// if low linear complexity is not considered an issue (as it is usually
/// the case) it can be used to generate 32-bit outputs, too.
///
/// We suggest to use a sign test to extract a random Boolean value, and
/// right shifts to extract subsets of bits.
///
/// The state must be seeded so that it is not everywhere zero.
pub struct Xoshiro128Plus {
    s: [u32; 4],
}

impl Xoshiro128Plus {
    pub fn new(s: [u32; 4]) -> Self {
        Self { s }
    }

    pub fn next_u32(&mut self) -> u32 {
        let result = u32::wrapping_add(self.s[0], self.s[3]);

        let t = self.s[1] << 9;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;

        self.s[3] = self.s[3].rotate_left(11);

        result
    }

    /// This is the jump function for the generator. It is equivalent
    /// to 2^64 calls to next(); it can be used to generate 2^64
    /// non-overlapping subsequences for parallel computations.
    pub fn jump(&mut self) {
        let mut s = [0; 4];

        for jump in [0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b] {
            for b in 0..32 {
                if (jump & 1u32 << b) != 0 {
                    s[0] ^= self.s[0];
                    s[1] ^= self.s[1];
                    s[2] ^= self.s[2];
                    s[3] ^= self.s[3];
                }
                self.next_u32();
            }
        }

        self.s = s;
    }

    /// This is the long-jump function for the generator. It is equivalent to
    /// 2^96 calls to next(); it can be used to generate 2^32 starting points,
    /// from each of which jump() will generate 2^32 non-overlapping
    /// subsequences for parallel distributed computations.
    pub fn long_jump(&mut self) {
        let mut s = [0; 4];

        for jump in [0xb523952e, 0x0b6f099f, 0xccf5a0ef, 0x1c580662] {
            for b in 0..32 {
                if (jump & 1u32 << b) != 0 {
                    s[0] ^= self.s[0];
                    s[1] ^= self.s[1];
                    s[2] ^= self.s[2];
                    s[3] ^= self.s[3];
                }
                self.next_u32();
            }
        }

        self.s = s;
    }
}

impl Rng<u32> for Xoshiro128Plus {
    fn gen(&mut self) -> u32 {
        self.next_u32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: [u32; 4] = [0x66781c33, 0x55a6e5fa, 0xb0c0d490, 0x936131b7];

    #[test]
    fn next_u32_works() {
        let mut rng = Xoshiro128Plus::new(SEED);

        for expected in [
            0xf9d94dea, 0xdf6236b4, 0xff1f08fe, 0x94ed7a4e, 0xcba36961, 0xc93db423, 0x1af019f1,
            0xc0947912, 0x02a761a6, 0x8eb42210, 0x8957c55e, 0x92cde6e4, 0x5feeb4d0, 0xc60d1a15,
            0xd181aaa0, 0xba192162, 0x494a7435, 0x6b349197, 0x6cd249ed, 0x1c71c80a, 0x41ecce84,
            0xb4920b83, 0x305c4c5e, 0x7cd3ba2e, 0x162cd6a2, 0xac1082c1, 0x77135a5f, 0xf1d8bc50,
            0xb2e51839, 0x02c8ee06, 0x5e6eb691, 0xaaa37b83,
        ] {
            assert_eq!(rng.next_u32(), expected);
        }
    }

    #[test]
    fn jump_works() {
        let mut rng = Xoshiro128Plus::new(SEED);

        rng.jump();

        for expected in [
            0xda05d9c5, 0x8c314f31, 0xbeae7ee0, 0x6895da7e, 0x3ac125e8, 0xbe2927c3, 0x8bab5385,
            0x0bbeed6f, 0x4c4ae158, 0x876a807e, 0x87154e4f, 0x5064643b, 0xb27d9172, 0x2649478a,
            0xeabd4007, 0x92474eee, 0xc0e98775, 0xc313490f, 0xd9e6cbe9, 0x22adac52, 0xe2d32a80,
            0x4f9cc6ed, 0xb2547b23, 0xeb9d25b9, 0x92490918, 0x3c3b9ae8, 0x27260f92, 0xb46f37a9,
            0xf993453f, 0x8c7d6b48, 0xf7c56b78, 0x0be6ff9d,
        ] {
            assert_eq!(rng.next_u32(), expected);
        }
    }

    #[test]
    fn long_jump_works() {
        let mut rng = Xoshiro128Plus::new(SEED);

        rng.long_jump();

        for expected in [
            0xdf555163, 0x88aa8824, 0xfd729500, 0xb4ccfc4f, 0x62c7bf88, 0x2d6891c2, 0xf21fe249,
            0xe894b3ba, 0x734121f3, 0x1fc94f52, 0x827b4384, 0x6ef623d3, 0x34a64631, 0x1f0b3860,
            0xb07240d5, 0x3566a537, 0x2ccaf8ac, 0xd23bbbef, 0xba05d31a, 0x076a0b1b, 0x25578077,
            0xba874d05, 0x134e34d7, 0xe4473868, 0xc329b958, 0x9856a217, 0xb29e6e64, 0xacdac351,
            0x6f969e3f, 0xc4d33728, 0xdaf8e820, 0x28925ea6,
        ] {
            assert_eq!(rng.next_u32(), expected);
        }
    }
}
