/// This is xoshiro256+ 1.0, our best and fastest generator for floating-point
/// numbers. We suggest to use its upper bits for floating-point
/// generation, as it is slightly faster than xoshiro256++/xoshiro256**. It
/// passes all tests we are aware of except for the lowest three bits,
/// which might fail linearity tests (and just those), so if low linear
/// complexity is not considered an issue (as it is usually the case) it
/// can be used to generate 64-bit outputs, too.
///
/// We suggest to use a sign test to extract a random Boolean value, and
/// right shifts to extract subsets of bits.
///
/// The state must be seeded so that it is not everywhere zero. If you have
/// a 64-bit seed, we suggest to seed a splitmix64 generator and use its
/// output to fill s.
pub struct Xoshiro256Plus {
    s: [u64; 4],
}

impl Xoshiro256Plus {
    pub fn new(s: [u64; 4]) -> Self {
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = u64::wrapping_add(self.s[0], self.s[3]);

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

    pub fn next_f64(&mut self) -> f64 {
        let result = self.next_u64();

        ((result >> 11) as f64) * f64::exp2(-53.0)
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
        let mut rng = Xoshiro256Plus::new(SEED);

        for expected in [
            0x0b6364bebe6205d6,
            0xe9d4695629243760,
            0xd6b9447df0d73ad5,
            0x30a46cd47b959eab,
            0xc96bfc5ecebe41bf,
            0x8b09ba9d85345720,
            0xc1ef6f1d5e5464f0,
            0x87c9fe2338dadf3e,
            0x34a83ca578285fda,
            0xb9b8c1e51c64c101,
            0xd7f46e56986cf754,
            0xe8062e78ec570cb7,
            0xabf4e181b8158ccf,
            0xbeea08f5d01afdd3,
            0xdf5a45a3c54407c4,
            0xe118a58b92795236,
        ] {
            assert_eq!(rng.next_u64(), expected);
        }
    }

    #[test]
    fn jump_works() {
        let mut rng = Xoshiro256Plus::new(SEED);

        rng.jump();

        for expected in [
            0x23a6a7693c649791,
            0x81fbc0f4028b8221,
            0x80ab5d2d18a0baa8,
            0x324cb63dd25c5be8,
            0x670ee636ca900eca,
            0xe7eed1a32a43d088,
            0xe66755d0b30ea359,
            0x09cd0200521c9225,
            0x1192240f2269c6ff,
            0x494a32a2874f60f2,
            0x157eb941916cc291,
            0x0de73965dbc06f00,
            0xded0b8edb257a9a5,
            0x298905daaaf82449,
            0x3fc314177c19cd08,
            0x849f459a7294e685,
        ] {
            assert_eq!(rng.next_u64(), expected);
        }
    }

    #[test]
    fn long_jump_works() {
        let mut rng = Xoshiro256Plus::new(SEED);

        rng.long_jump();

        for expected in [
            0xbd46e488520d0c85,
            0xfd4a23bac6a2fc50,
            0x53ca70d144586ec8,
            0x47ddac205da563ba,
            0xba669c0caa68e7a3,
            0xd06b7c7fcd1bf066,
            0xff014197b6e9c4f4,
            0xac84cfae2354b80f,
            0x8fe6bb71fd9760d2,
            0xa06880d03ff376a6,
            0x039189f719e7bdfc,
            0xfc5616d9fe411cd6,
            0x22c14cdb05594157,
            0x2237e07933f275c6,
            0x44dccf0c4ee65e8b,
            0x10834ed6e4549282,
        ] {
            assert_eq!(rng.next_u64(), expected);
        }
    }

    #[test]
    fn next_f64_works() {
        let mut rng = Xoshiro256Plus::new(SEED);

        for expected in [
            0x3fa6c6c97d7cc400,
            0x3fed3a8d2ac52486,
            0x3fead7288fbe1ae7,
            0x3fc852366a3dcacc,
            0x3fe92d7f8bd9d7c8,
            0x3fe1613753b0a68a,
            0x3fe83dede3abca8c,
            0x3fe0f93fc4671b5b,
            0x3fca541e52bc142c,
            0x3fe737183ca38c98,
            0x3feafe8dcad30d9e,
            0x3fed00c5cf1d8ae1,
            0x3fe57e9c303702b1,
            0x3fe7dd411eba035f,
            0x3febeb48b478a880,
            0x3fec2314b1724f2a,
        ] {
            assert_eq!(rng.next_f64().to_bits(), expected);
        }
    }
}
