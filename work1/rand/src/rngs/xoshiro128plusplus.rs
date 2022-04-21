use super::Rng;

/// This is xoshiro128++ 1.0, one of our 32-bit all-purpose, rock-solid
/// generators. It has excellent speed, a state size (128 bits) that is
/// large enough for mild parallelism, and it passes all tests we are aware
/// of.
///
/// For generating just single-precision (i.e., 32-bit) floating-point
/// numbers, xoshiro128+ is even faster.
///
/// The state must be seeded so that it is not everywhere zero.
pub struct Xoshiro128PlusPlus {
    s: [u32; 4],
}

impl Xoshiro128PlusPlus {
    pub fn new(s: [u32; 4]) -> Self {
        Self { s }
    }

    pub fn next_u32(&mut self) -> u32 {
        let result = u32::wrapping_add(
            u32::wrapping_add(self.s[0], self.s[3]).rotate_left(7),
            self.s[0],
        );

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

impl Rng<u32> for Xoshiro128PlusPlus {
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
        let mut rng = Xoshiro128PlusPlus::new(SEED);

        for expected in [
            0x531f11af, 0x51db22ed, 0xac880a90, 0xbe8756c2, 0x7a6322cb, 0x6fcb8a45, 0x2489ae9a,
            0x69e534d4, 0x67c64c4a, 0xd0878744, 0x291aac48, 0xd566e893, 0xd37d776d, 0x9ed3c2d5,
            0xac8f5f69, 0x1a40338f, 0x072d8377, 0xc73d7ae4, 0x430be8fb, 0xa514ff6a, 0x8c2cc0d5,
            0x5156b871, 0x41198aa1, 0xbdd35764, 0x30b48d60, 0x3e295a1a, 0x8695ebd2, 0xea3464bf,
            0x780fad50, 0x6a9308eb, 0x94976c67, 0x5a85daa9,
        ] {
            assert_eq!(rng.next_u32(), expected);
        }
    }

    #[test]
    fn jump_works() {
        let mut rng = Xoshiro128PlusPlus::new(SEED);

        rng.jump();

        for expected in [
            0x0f077052, 0x9662dc6a, 0x55a2bb21, 0xb5ca3d0d, 0x193fb372, 0x022a8ef9, 0x11066794,
            0xecba2344, 0xa44b71e2, 0x2fbb979c, 0xcc9b2436, 0x46a69f37, 0xd4dfc6b8, 0xe79f29f6,
            0xf591215c, 0x7e3f5fd4, 0xb87c098c, 0x37bf5983, 0xc9c09513, 0x22e69c91, 0xb9f77e0e,
            0x68980742, 0x957a0574, 0x71484c62, 0x29296831, 0xa4dff2e9, 0xc773f30a, 0xa260f18e,
            0x8964c693, 0xa01ce899, 0x16158562, 0x8d5158ce,
        ] {
            assert_eq!(rng.next_u32(), expected);
        }
    }

    #[test]
    fn long_jump_works() {
        let mut rng = Xoshiro128PlusPlus::new(SEED);

        rng.long_jump();

        for expected in [
            0x8a465242, 0x272f89f5, 0x460fe295, 0xc50dcd97, 0xb1cd031e, 0x1efe499e, 0x2f205b96,
            0xdb9fbcbb, 0xdd740e3f, 0xd88e11e9, 0x0bd2a3d7, 0x4137e54a, 0x31524e0b, 0xd271a3d8,
            0x7ef88b65, 0x7109710e, 0x8b5261ff, 0x03a97dd8, 0x6069a2b5, 0xa5e85333, 0x445398c6,
            0x1f2f05ca, 0xb5c281b7, 0xc53c7762, 0xefa9f5de, 0x8f5ccbed, 0x73f1aeb8, 0x1b674fd2,
            0xc179ab35, 0x07e9f047, 0x98eda47c, 0x85f440b5,
        ] {
            assert_eq!(rng.next_u32(), expected);
        }
    }
}
