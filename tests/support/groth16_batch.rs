//! Groth16 statements built from known toxic waste, and the random-linear
//! combination batch over them.
//!
//! Every proof here satisfies the real verification equation, so a batch
//! accepts exactly when every member is genuine. Both the soundness test and
//! the batch bench drive this fixture.

#![allow(dead_code)]

use helius_narsil::{
    Fp12, Fr, G1Affine, G1Projective, G2Affine, LivePair, PreparedTerm, PreparedVerifier,
    pairing::multi_pairing,
};

pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub fn word(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn scalar(&mut self) -> Fr {
        Fr::from_raw([self.word(), self.word(), self.word(), self.word()])
    }
}

pub fn g1(scalar: Fr) -> G1Affine {
    G1Projective::generator().mul_scalar(scalar).to_affine()
}

pub fn g2(scalar: Fr) -> G2Affine {
    G2Affine::generator()
        .to_curve()
        .mul_scalar(scalar)
        .to_affine()
}

pub struct Key {
    pub alpha: G1Affine,
    pub beta: G2Affine,
    pub gamma: G2Affine,
    pub delta: G2Affine,
    pub k0: G1Affine,
    pub k1: G1Affine,
    alpha_beta: Fr,
    gamma_scalar: Fr,
    delta_inverse: Fr,
    k0_scalar: Fr,
    k1_scalar: Fr,
    context: PreparedVerifier,
}

impl Key {
    pub fn new(rng: &mut Rng) -> Self {
        let (alpha, beta, gamma, delta) = (rng.scalar(), rng.scalar(), rng.scalar(), rng.scalar());
        let (k0, k1) = (rng.scalar(), rng.scalar());
        let gamma_point = g2(gamma);
        let delta_point = g2(delta);
        let context = PreparedVerifier::new(
            &[gamma_point.negate(), delta_point.negate()],
            &[helius_narsil::FixedPair {
                g1: g1(alpha).negate(),
                g2: g2(beta),
            }],
        )
        .expect("generator multiples are valid verifier material");
        Self {
            alpha: g1(alpha),
            beta: g2(beta),
            gamma: gamma_point,
            delta: delta_point,
            k0: g1(k0),
            k1: g1(k1),
            alpha_beta: alpha * beta,
            gamma_scalar: gamma,
            delta_inverse: delta.invert().expect("delta is nonzero"),
            k0_scalar: k0,
            k1_scalar: k1,
            context,
        }
    }
}

#[derive(Clone)]
pub struct Member {
    pub key: usize,
    pub a: G1Affine,
    pub b: G2Affine,
    pub c: G1Affine,
    pub input: Fr,
    pub challenge: Fr,
}

impl Member {
    /// Solve for `C` so the Groth16 equation holds over the known exponents.
    pub fn new(rng: &mut Rng, keys: &[Key], key: usize) -> Self {
        let k = &keys[key];
        let (a, b, input) = (rng.scalar(), rng.scalar(), rng.scalar());
        let witness = k.k0_scalar + input * k.k1_scalar;
        let c = (a * b + (k.alpha_beta + witness * k.gamma_scalar).negate()) * k.delta_inverse;
        Self {
            key,
            a: g1(a),
            b: g2(b),
            c: g1(c),
            input,
            challenge: rng.scalar(),
        }
    }
}

pub struct Batch {
    pub keys: Vec<Key>,
    pub members: Vec<Member>,
}

impl Batch {
    pub fn new(seed: u64, size: usize, keys: usize) -> Self {
        let mut rng = Rng::new(seed);
        let key_set: Vec<Key> = (0..keys).map(|_| Key::new(&mut rng)).collect();
        let members = (0..size)
            .map(|index| Member::new(&mut rng, &key_set, index % keys))
            .collect();
        Self {
            keys: key_set,
            members,
        }
    }

    /// Redraw every coefficient. Batch soundness rests on the coefficients
    /// being unpredictable to whoever chose the proofs, so a rejection test
    /// must hold over independent draws and not over one fixed vector.
    pub fn redraw_challenges(&mut self, seed: u64) {
        let mut rng = Rng::new(seed);
        for member in &mut self.members {
            member.challenge = rng.scalar();
        }
    }

    /// Pairing product of the whole batch under the drawn coefficients.
    pub fn product(&self) -> Fp12 {
        let keys = self.keys.len();
        let mut input_sum = vec![G1Projective::identity(); keys];
        let mut c_sum = vec![G1Projective::identity(); keys];
        let mut challenge_sum = vec![Fr::ZERO; keys];
        let mut pairs: Vec<(G1Affine, G2Affine)> =
            Vec::with_capacity(self.members.len() + 3 * keys);
        for member in &self.members {
            let key = &self.keys[member.key];
            let vk_x = key
                .k1
                .to_curve()
                .mul_scalar(member.input)
                .add_mixed(key.k0)
                .mul_scalar(member.challenge);
            input_sum[member.key] = input_sum[member.key].add_projective(vk_x);
            c_sum[member.key] =
                c_sum[member.key].add_projective(member.c.to_curve().mul_scalar(member.challenge));
            challenge_sum[member.key] += member.challenge;
            pairs.push((
                member.a.to_curve().mul_scalar(member.challenge).to_affine(),
                member.b,
            ));
        }
        for (index, key) in self.keys.iter().enumerate() {
            pairs.push((input_sum[index].negate().to_affine(), key.gamma));
            pairs.push((c_sum[index].negate().to_affine(), key.delta));
            pairs.push((
                key.alpha
                    .to_curve()
                    .mul_scalar(challenge_sum[index])
                    .negate()
                    .to_affine(),
                key.beta,
            ));
        }
        let borrowed: Vec<(&G1Affine, &G2Affine)> = pairs.iter().map(|(p, q)| (p, q)).collect();
        multi_pairing(&borrowed)
    }

    pub fn verify(&self) -> bool {
        self.product().is_one()
    }

    /// Verify every member on its own, the reference the batch must beat.
    pub fn verify_sequential(&self) -> bool {
        self.members.iter().all(|member| {
            let key = &self.keys[member.key];
            let accumulator = key
                .k1
                .to_curve()
                .mul_scalar(member.input)
                .add_mixed(key.k0)
                .to_affine();
            key.context
                .verify_prevalidated(
                    &[
                        PreparedTerm {
                            g1: accumulator,
                            prepared_g2: 0,
                        },
                        PreparedTerm {
                            g1: member.c,
                            prepared_g2: 1,
                        },
                    ],
                    &[LivePair {
                        g1: member.a,
                        g2: member.b,
                    }],
                )
                .expect("fixture terms are in bounds")
                .0
        })
    }
}
