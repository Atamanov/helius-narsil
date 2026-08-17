//! Fixture pools and lane adapters for the four-lane timing runner.
//!
//! Every fixture builds a helius, an arkworks, and an mcl context over the
//! same statement and refuses to exist unless the three agree. A lane that
//! computes something else cannot reach a timed region.

use std::{
    collections::HashSet, ffi::c_void, fmt::Write as _, marker::PhantomData, ptr::NonNull,
    sync::OnceLock,
};

use ark_bn254::{
    Bn254, Fq, Fq2, Fq6 as ArkFq6, Fq12 as ArkFq12, Fr as ArkFr, G1Affine as ArkG1,
    G1Projective as ArkG1Projective, G2Affine as ArkG2, G2Projective as ArkG2Projective,
};
use ark_ec::{
    AffineRepr, CurveGroup, VariableBaseMSM,
    pairing::{Pairing, PairingOutput},
};
use ark_ff::{BigInteger, CyclotomicMultSubgroup, Field, One, PrimeField, UniformRand, Zero};
use ark_groth16::{Groth16, PreparedVerifyingKey, Proof, prepare_verifying_key};
use helius_narsil::{
    FixedPair, Fp, Fp2, Fp6, Fp12, Fr, G1Affine, G1Bytes, G1Projective, G2Affine, G2Bytes,
    LivePair, PreparedTerm, PreparedVerifier, ScalarBytes, g1_msm, msm_variable_time_affine,
    pairing::{
        G2Prepared as HeliusG2Prepared, MillerTerm, miller_loop_prepared,
        multi_miller_loop_mixed as helius_multi_miller_loop_mixed,
        multi_pairing as helius_multi_pairing, prepare_g2, prepare_g2_unchecked,
    },
};
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// Base fixture seed. Every pool derives from it and from the run seed.
pub const SEED: u64 = 0x0a17_b428;
pub const GNARK_FRESH_POOL_SHA256: &str =
    "26aa7e990eeeac633c3146a7756b5bd459d34185601af422d18756d0706b5704";
pub const GNARK_FRESH_POOL_SIZE: usize = 64;
pub const PRODUCTION_POOL_SHA256: &str =
    "f0126fc8f1e10dbf99d042732ab9f48f7cd6323f9279b5af5fe6521fa67c0f9c";
pub const PRODUCTION_STANDARD_POOL_SIZE: usize = 64;
pub const PRODUCTION_COMMITTED_POOL_SIZE: usize = 16;
/// A block that carries several transactions of one circuit is the common
/// case, so the same-key batch takes the largest size the campaign times.
pub const PRODUCTION_BATCH_SIZE: usize = 8;
/// The size the project goal is written against, kept beside the larger one
/// because the two sit on opposite sides of the batching crossover.
pub const PRODUCTION_SMALL_BATCH_SIZE: usize = 3;
/// Batch pool width, per batch size.
///
/// The standard pool holds 64 vectors over two plain keys, so a same-key pool
/// of `count` batches of `size` needs `(count / 2) * size` vectors per key and
/// a mixed pool needs `count * ceil(size / 2)`. Both bind at 8 batches of 8 and
/// at 16 batches of 3. The size 3 rows draw 8 fixtures a round, so 16 is what
/// lets their four rounds draw four different sets.
pub fn production_batch_pool_size(batch_size: usize) -> usize {
    if batch_size >= PRODUCTION_BATCH_SIZE {
        8
    } else {
        16
    }
}

/// Arkworks' reusable G2 line schedule type for the prepared-only comparator.
pub type ArkG2Prepared = <Bn254 as Pairing>::G2Prepared;

/// One typed nonidentity pair used to compare only the Miller loop.
pub struct MillerFixture {
    helius_g1: G1Affine,
    helius_g2: G2Affine,
    helius_g2_prepared: HeliusG2Prepared,
    ark_g1: ArkG1,
    ark_g2: ArkG2,
    ark_g2_prepared: ArkG2Prepared,
    helius_miller_output: Fp12,
    ark_miller_output: ark_ec::pairing::MillerLoopOutput<Bn254>,
    fixture_sha256: String,
    mcl: NonNull<c_void>,
}

impl MillerFixture {
    pub fn from_seed(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let ark_g1 = ArkG1Projective::rand(&mut rng).into_affine();
        let ark_g2 = ArkG2Projective::rand(&mut rng).into_affine();
        let g1_bytes = encode_ark_g1(ark_g1.into_group());
        let g2_bytes = encode_ark_g2(ark_g2.into_group());
        let helius_g1 = g1_bytes.to_affine().expect("valid Helius Miller G1");
        let helius_g2 = g2_bytes.to_affine().expect("valid Helius Miller G2");
        let helius_g2_prepared =
            prepare_g2(&helius_g2).expect("valid Helius G2 prepares successfully");
        let ark_g2_prepared = ark_g2.into();
        let helius_miller_output = helius_narsil::pairing::miller_loop(&helius_g1, &helius_g2);
        let ark_miller_output = Bn254::multi_miller_loop([ark_g1], [ark_g2]);
        ensure_mcl();
        let mut handle = core::ptr::null_mut();
        // SAFETY: both encodings are 64 and 128 initialized bytes and handle is
        // a valid out-pointer.
        let status = unsafe {
            ffi::mcl_miller_create(g1_bytes.0.as_ptr(), g2_bytes.0.as_ptr(), &mut handle)
        };
        assert_eq!(
            status, 0,
            "MCL Miller context creation failed with {status}"
        );
        let fixture = Self {
            helius_g1,
            helius_g2,
            helius_g2_prepared,
            ark_g1,
            ark_g2,
            ark_g2_prepared,
            helius_miller_output,
            ark_miller_output,
            fixture_sha256: format!(
                "{:x}",
                Sha256::digest([g1_bytes.0.as_slice(), g2_bytes.0.as_slice()].concat())
            ),
            mcl: NonNull::new(handle).expect("MCL returned a Miller context"),
        };
        assert_miller_equivalent(&fixture);
        fixture
    }

    pub fn helius_g2(&self) -> G2Affine {
        self.helius_g2
    }

    pub fn ark_g2(&self) -> ArkG2 {
        self.ark_g2
    }

    pub fn sha256(&self) -> &str {
        &self.fixture_sha256
    }
}

impl Drop for MillerFixture {
    fn drop(&mut self) {
        // SAFETY: this handle is uniquely owned and was returned by create.
        unsafe { ffi::mcl_miller_destroy(self.mcl.as_ptr()) };
    }
}

/// One decoded plain Groth16 statement and its prepared verifier state.
pub struct Groth16Fixture {
    ark_pvk: PreparedVerifyingKey<Bn254>,
    ark_proof: Proof<Bn254>,
    ark_public_inputs: Vec<ArkFr>,
    helius_context: PreparedVerifier,
    helius_gamma_abc: Vec<G1Affine>,
    helius_public_input_limbs: Vec<[u64; 4]>,
    helius_proof_a: G1Affine,
    helius_proof_b: G2Affine,
    helius_proof_c: G1Affine,
    fixture_sha256: String,
    mcl: NonNull<c_void>,
}

impl Groth16Fixture {
    /// Build all three lanes' contexts over one already decoded statement.
    ///
    /// No verdict is asserted, so a deliberately invalid vector is built the
    /// same way a valid one is.
    pub fn from_parts(
        vk: &ark_groth16::VerifyingKey<Bn254>,
        ark_proof: Proof<Bn254>,
        ark_public_inputs: Vec<ArkFr>,
    ) -> Self {
        assert_eq!(
            vk.gamma_abc_g1.len(),
            ark_public_inputs.len() + 1,
            "gamma_abc holds the constant wire plus one point per public input"
        );
        let ark_pvk = prepare_verifying_key(vk);

        let alpha = helius_g1(vk.alpha_g1);
        let beta = helius_g2(vk.beta_g2);
        let gamma = helius_g2(vk.gamma_g2);
        let delta = helius_g2(vk.delta_g2);
        let helius_gamma_abc = vk
            .gamma_abc_g1
            .iter()
            .copied()
            .map(helius_g1)
            .collect::<Vec<_>>();
        // Precomputed outside every timed region, the same way the other
        // lanes hold their decoded scalars.
        let helius_public_input_limbs = ark_public_inputs
            .iter()
            .copied()
            .map(|value| scalar_limbs(&encode_ark_fr(value)))
            .collect::<Vec<_>>();
        let helius_proof_a = helius_g1(ark_proof.a);
        let helius_proof_b = helius_g2(ark_proof.b);
        let helius_proof_c = helius_g1(ark_proof.c);
        let helius_context = PreparedVerifier::new(
            &[gamma, delta],
            &[FixedPair {
                g1: alpha.negate(),
                g2: beta,
            }],
        )
        .expect("valid Helius Groth16 context");

        let blob = groth16_blob(vk, &ark_proof, &ark_public_inputs);
        let fixture_sha256 = format!("{:x}", Sha256::digest(&blob));
        ensure_mcl();
        let mut handle = core::ptr::null_mut();
        // SAFETY: blob holds the documented fixed-width records and handle is a
        // valid out-pointer.
        let status = unsafe {
            ffi::mcl_groth16_create(
                blob.as_ptr(),
                vk.gamma_abc_g1.len(),
                ark_public_inputs.len(),
                &mut handle,
            )
        };
        assert_eq!(
            status, 0,
            "mcl Groth16 context creation failed with {status}"
        );
        let mcl = NonNull::new(handle).expect("MCL returned a non-null Groth16 context");
        Self {
            ark_pvk,
            ark_proof,
            ark_public_inputs,
            helius_context,
            helius_gamma_abc,
            helius_public_input_limbs,
            helius_proof_a,
            helius_proof_b,
            helius_proof_c,
            fixture_sha256,
            mcl,
        }
    }

    pub fn sha256(&self) -> &str {
        &self.fixture_sha256
    }

    /// Build the three online Groth16 pairing terms.
    fn helius_online_terms(&self) -> ([PreparedTerm; 2], [LivePair; 1]) {
        let public_input = helius_public_input_accumulator(self);
        (
            [
                PreparedTerm {
                    g1: public_input.negate(),
                    prepared_g2: 0,
                },
                PreparedTerm {
                    g1: self.helius_proof_c.negate(),
                    prepared_g2: 1,
                },
            ],
            [LivePair {
                g1: self.helius_proof_a,
                g2: self.helius_proof_b,
            }],
        )
    }
}

impl Drop for Groth16Fixture {
    fn drop(&mut self) {
        // SAFETY: this handle is uniquely owned and was returned by create.
        unsafe { ffi::mcl_groth16_destroy(self.mcl.as_ptr()) };
    }
}

/// One committed-wires (gnark BSB22) Groth16 statement, all lanes typed.
pub struct GnarkCommittedGroth16Fixture {
    ark_alpha_beta: PairingOutput<Bn254>,
    ark_gamma_neg: ArkG2,
    ark_delta_neg: ArkG2,
    ark_k: [ArkG1; 3],
    ark_commitment_key_g: ArkG2,
    ark_commitment_key_g_sigma_neg: ArkG2,
    ark_proof_a: ArkG1,
    ark_proof_b: ArkG2,
    ark_proof_c: ArkG1,
    ark_commitment: ArkG1,
    ark_commitment_pok: ArkG1,
    ark_public_input: ArkFr,
    helius_alpha_beta: Fp12,
    helius_gamma_neg: G2Affine,
    helius_delta_neg: G2Affine,
    helius_k: [G1Affine; 3],
    helius_commitment_key_g: G2Affine,
    helius_commitment_key_g_sigma_neg: G2Affine,
    helius_proof_a: G1Affine,
    helius_proof_b: G2Affine,
    helius_proof_c: G1Affine,
    helius_commitment: G1Affine,
    helius_commitment_pok: G1Affine,
    helius_public_input: Fr,
    // The four fixed G2 points are verifying-key material, so both Rust
    // lanes own their line schedules from construction. Only proof_b stays
    // live in the timed paths.
    helius_pok_context: PreparedVerifier,
    helius_main_context: PreparedVerifier,
    ark_commitment_key_g_sigma_neg_prepared: ArkG2Prepared,
    ark_commitment_key_g_prepared: ArkG2Prepared,
    ark_delta_neg_prepared: ArkG2Prepared,
    ark_gamma_neg_prepared: ArkG2Prepared,
    // Transcript hash of the fixture, both lanes typed. The timed verify paths
    // take it as verifier-key material, computed outside the timer.
    ark_commitment_hash: ArkFr,
    helius_commitment_hash: Fr,
    commitment_hash_bytes: [u8; 32],
    fixture_sha256: String,
    mcl: NonNull<c_void>,
}

/// Decoded, still-encoded inputs for one committed-wires Groth16 statement.
pub struct CommittedParts {
    pub alpha: G1Bytes,
    pub beta: G2Bytes,
    pub gamma: G2Bytes,
    pub delta: G2Bytes,
    pub k: [G1Bytes; 3],
    pub commitment_key_g: G2Bytes,
    pub commitment_key_g_sigma_neg: G2Bytes,
    pub proof_a: G1Bytes,
    pub proof_b: G2Bytes,
    pub proof_c: G1Bytes,
    pub commitment: G1Bytes,
    pub commitment_pok: G1Bytes,
    pub public_input: [u8; 32],
    /// Public-input values the commitment binds, gnark's
    /// `PublicAndCommitmentCommitted` for this key. Empty is normal, a
    /// commitment that binds no public wire still has a transcript.
    pub transcript_inputs: Vec<[u8; 32]>,
}

impl GnarkCommittedGroth16Fixture {
    /// Build every lane's context over one already decoded committed statement.
    ///
    /// No verdict is asserted, so a deliberately invalid vector is built the
    /// same way a valid one is.
    pub fn from_parts(parts: CommittedParts) -> Self {
        let CommittedParts {
            alpha: alpha_bytes,
            beta: beta_bytes,
            gamma: gamma_bytes,
            delta: delta_bytes,
            k: k_bytes,
            commitment_key_g: commitment_key_g_bytes,
            commitment_key_g_sigma_neg: commitment_key_g_sigma_neg_bytes,
            proof_a: proof_a_bytes,
            proof_b: proof_b_bytes,
            proof_c: proof_c_bytes,
            commitment: commitment_bytes,
            commitment_pok: commitment_pok_bytes,
            public_input: public_input_bytes,
            transcript_inputs,
        } = parts;

        let ark_alpha = decode_ark_g1(&alpha_bytes.0).expect("valid gnark alpha G1");
        let ark_beta = decode_ark_g2(&beta_bytes.0).expect("valid gnark beta G2");
        let ark_gamma = decode_ark_g2(&gamma_bytes.0).expect("valid gnark gamma G2");
        let ark_delta = decode_ark_g2(&delta_bytes.0).expect("valid gnark delta G2");
        let ark_k = k_bytes.map(|point| decode_ark_g1(&point.0).expect("valid gnark K G1"));
        let ark_commitment_key_g =
            decode_ark_g2(&commitment_key_g_bytes.0).expect("valid gnark commitment-key G");
        let ark_commitment_key_g_sigma_neg = decode_ark_g2(&commitment_key_g_sigma_neg_bytes.0)
            .expect("valid gnark commitment-key negative sigma G");
        let ark_proof_a = decode_ark_g1(&proof_a_bytes.0).expect("valid gnark proof A");
        let ark_proof_b = decode_ark_g2(&proof_b_bytes.0).expect("valid gnark proof B");
        let ark_proof_c = decode_ark_g1(&proof_c_bytes.0).expect("valid gnark proof C");
        let ark_commitment = decode_ark_g1(&commitment_bytes.0).expect("valid gnark commitment");
        let ark_commitment_pok =
            decode_ark_g1(&commitment_pok_bytes.0).expect("valid gnark commitment PoK");
        let ark_public_input =
            decode_ark_fr(&public_input_bytes).expect("valid gnark public input");

        let helius_alpha = alpha_bytes.to_affine().expect("valid Helius alpha G1");
        let helius_beta = beta_bytes.to_affine().expect("valid Helius beta G2");
        let helius_gamma = gamma_bytes.to_affine().expect("valid Helius gamma G2");
        let helius_delta = delta_bytes.to_affine().expect("valid Helius delta G2");
        let helius_k = k_bytes.map(|point| point.to_affine().expect("valid Helius K G1"));
        let helius_commitment_key_g = commitment_key_g_bytes
            .to_affine()
            .expect("valid Helius commitment-key G");
        let helius_commitment_key_g_sigma_neg = commitment_key_g_sigma_neg_bytes
            .to_affine()
            .expect("valid Helius commitment-key negative sigma G");
        let helius_proof_a = proof_a_bytes.to_affine().expect("valid Helius proof A");
        let helius_proof_b = proof_b_bytes.to_affine().expect("valid Helius proof B");
        let helius_proof_c = proof_c_bytes.to_affine().expect("valid Helius proof C");
        let helius_commitment = commitment_bytes
            .to_affine()
            .expect("valid Helius commitment");
        let helius_commitment_pok = commitment_pok_bytes
            .to_affine()
            .expect("valid Helius commitment PoK");
        let helius_public_input = ScalarBytes(public_input_bytes)
            .to_fr()
            .expect("valid Helius public input");

        let ark_commitment_hash = gnark_commitment_hash(commitment_bytes.0, &transcript_inputs);
        let commitment_hash_bytes = encode_ark_fr(ark_commitment_hash).0;
        let helius_commitment_hash = ScalarBytes(commitment_hash_bytes)
            .to_fr()
            .expect("gnark commitment hash is canonical");

        let ark_alpha_beta = Bn254::pairing(ark_alpha, ark_beta);
        let helius_alpha_beta = helius_narsil::pairing(&helius_alpha, &helius_beta);
        let ark_gamma_neg = -ark_gamma;
        let ark_delta_neg = -ark_delta;
        let helius_gamma_neg = helius_gamma.negate();
        let helius_delta_neg = helius_delta.negate();
        let helius_pok_context = PreparedVerifier::new(
            &[helius_commitment_key_g_sigma_neg, helius_commitment_key_g],
            &[],
        )
        .expect("valid gnark commitment-key G2 points");
        // The fixed e(-alpha, beta) term inverts the main-equation target, so
        // the context verdict tests the online product against e(alpha, beta).
        let helius_main_context = PreparedVerifier::new(
            &[helius_delta_neg, helius_gamma_neg],
            &[FixedPair {
                g1: helius_alpha.negate(),
                g2: helius_beta,
            }],
        )
        .expect("valid gnark verifying-key points");

        let blob = gnark_committed_groth16_blob(
            &alpha_bytes,
            &beta_bytes,
            &gamma_bytes,
            &delta_bytes,
            &k_bytes,
            &commitment_key_g_bytes,
            &commitment_key_g_sigma_neg_bytes,
            &proof_a_bytes,
            &proof_b_bytes,
            &proof_c_bytes,
            &commitment_bytes,
            &commitment_pok_bytes,
            public_input_bytes,
        );
        // The same preimage the originating generator digests, so a vector's
        // identity is comparable against the manifest it shipped in.
        let fixture_sha256 = format!(
            "{:x}",
            Sha256::digest(
                [
                    proof_a_bytes.0.as_slice(),
                    proof_b_bytes.0.as_slice(),
                    proof_c_bytes.0.as_slice(),
                    public_input_bytes.as_slice(),
                    commitment_bytes.0.as_slice(),
                    commitment_pok_bytes.0.as_slice(),
                ]
                .concat()
            )
        );
        ensure_mcl();
        let mut handle = core::ptr::null_mut();
        // SAFETY: blob holds the documented 1312-byte committed layout and
        // handle is a valid out-pointer.
        let status = unsafe { ffi::mcl_gnark_committed_groth16_create(blob.as_ptr(), &mut handle) };
        assert_eq!(status, 0, "MCL committed Groth16 context creation failed");

        Self {
            ark_alpha_beta,
            ark_gamma_neg,
            ark_delta_neg,
            ark_k,
            ark_commitment_key_g,
            ark_commitment_key_g_sigma_neg,
            ark_proof_a,
            ark_proof_b,
            ark_proof_c,
            ark_commitment,
            ark_commitment_pok,
            ark_public_input,
            helius_alpha_beta,
            helius_gamma_neg,
            helius_delta_neg,
            helius_k,
            helius_commitment_key_g,
            helius_commitment_key_g_sigma_neg,
            helius_proof_a,
            helius_proof_b,
            helius_proof_c,
            helius_commitment,
            helius_commitment_pok,
            helius_public_input,
            helius_pok_context,
            helius_main_context,
            ark_commitment_key_g_sigma_neg_prepared: ArkG2Prepared::from(
                ark_commitment_key_g_sigma_neg,
            ),
            ark_commitment_key_g_prepared: ArkG2Prepared::from(ark_commitment_key_g),
            ark_delta_neg_prepared: ArkG2Prepared::from(ark_delta_neg),
            ark_gamma_neg_prepared: ArkG2Prepared::from(ark_gamma_neg),
            ark_commitment_hash,
            helius_commitment_hash,
            commitment_hash_bytes,
            fixture_sha256,
            mcl: NonNull::new(handle).expect("MCL returned a committed Groth16 context"),
        }
    }

    pub fn sha256(&self) -> &str {
        &self.fixture_sha256
    }

    /// The transcript hash every lane takes as prepared verifier material.
    pub fn commitment_hash_bytes(&self) -> &[u8; 32] {
        &self.commitment_hash_bytes
    }
}

impl Drop for GnarkCommittedGroth16Fixture {
    fn drop(&mut self) {
        // SAFETY: this handle is uniquely owned and was returned by create.
        unsafe { ffi::mcl_gnark_committed_groth16_destroy(self.mcl.as_ptr()) };
    }
}

#[derive(Deserialize)]
struct GnarkFreshPoolDocument {
    schema: String,
    gnark_version: String,
    gnark_crypto_version: String,
    run_seed: String,
    verifying_key: GnarkFreshKeyDocument,
    valid: Vec<GnarkFreshVectorDocument>,
    invalid: Vec<GnarkFreshVectorDocument>,
}

#[derive(Deserialize)]
struct GnarkFreshKeyDocument {
    alpha_g1: String,
    beta_g2: String,
    gamma_g2: String,
    delta_g2: String,
    k: Vec<String>,
}

#[derive(Deserialize)]
struct GnarkFreshVectorDocument {
    public_inputs: Vec<String>,
    proof_a: String,
    proof_b: String,
    proof_c: String,
    gnark_verdict: String,
}

// ---------------------------------------------------------------------------
// Per-session draw order over the pinned external pools.
// ---------------------------------------------------------------------------

const GNARK_FRESH_DRAW_DOMAIN: u64 = 0x676e_6172_6b5f_7631;
const PRODUCTION_STANDARD_DRAW_DOMAIN: u64 = 0x7a6f_6c5f_7374_6431;
const PRODUCTION_COMMITTED_DRAW_DOMAIN: u64 = 0x7a6f_6c5f_636d_7431;
const SPLITMIX_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// SplitMix64 finalizer. It maps zero to zero, which is what keeps an unset
/// run seed on the frozen pools.
pub fn mix64(value: u64) -> u64 {
    let mut mixed = value;
    mixed ^= mixed >> 30;
    mixed = mixed.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed ^= mixed >> 27;
    mixed = mixed.wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

/// One pool entry's seed. The run seed enters as a mixed constant, so domain
/// separation and within-run uniqueness hold exactly as before, and a zero run
/// seed reproduces the frozen default pools byte for byte.
pub fn pool_seed(fixture_seed: u64, domain: u64, index: usize) -> u64 {
    SEED ^ mix64(fixture_seed)
        ^ domain.rotate_left(17)
        ^ (index as u64).wrapping_mul(SPLITMIX_GAMMA)
}

/// A seeded pool of `size` fixtures under one domain.
pub fn seeded_pool<T>(
    fixture_seed: u64,
    domain: u64,
    size: usize,
    build: impl Fn(u64) -> T,
) -> Vec<T> {
    (0..size)
        .map(|index| build(pool_seed(fixture_seed, domain, index)))
        .collect()
}

/// Pool domain separators. The campaign runner and the Criterion anchor bench
/// must build one pool from one seed, so both read these.
pub const FIELD_POOL_DOMAIN: u64 = 0x0066_6965_6c64;
pub const MILLER_POOL_DOMAIN: u64 = 0x6d69_6c6c_6572;
pub const MSM_POOL_DOMAIN: u64 = 0x6d73_6d5f;
pub const SUBGROUP_POOL_DOMAIN: u64 = 0x7375_6267_7270;

/// Fixture pool sizes, shared by the runner and the anchor bench.
///
/// The pairing rows draw 16 fixtures a round, so 64 lets four rounds draw four
/// disjoint sets. The field, subgroup and MSM rows draw hundreds to thousands
/// of iterations a round; no pool of a sane size can give those rows a fresh
/// set, and their rotation exists to defeat a per-fixture cache instead.
pub const MILLER_POOL_SIZE: usize = 64;
pub const FIELD_POOL_SIZE: usize = 8;
pub const SUBGROUP_POOL_SIZE: usize = 8;
pub const MSM_POOL_SIZE: usize = 8;

/// Fisher-Yates order over `0..len`, keyed by the run seed and a pool domain.
///
/// An external pool ships pinned proof bytes, so a session's freshness is
/// which vectors it draws and in what order, never the vectors themselves.
/// The order is a pure function of the seed the provenance line records, so a
/// session is reproducible from that seed alone. A zero seed is the identity,
/// which keeps an unseeded run on the frozen order.
pub fn seeded_order(seed: u64, domain: u64, len: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..len).collect();
    if seed == 0 {
        return order;
    }
    let mut state = mix64(seed) ^ domain.rotate_left(17);
    for high in (1..len).rev() {
        state = state.wrapping_add(SPLITMIX_GAMMA);
        let draw = (mix64(state) % (high as u64 + 1)) as usize;
        order.swap(high, draw);
    }
    order
}

/// Move a pool into its per-session draw order.
fn draw_order<T>(order: &[usize], values: Vec<T>) -> Vec<T> {
    assert_eq!(order.len(), values.len());
    let mut slots: Vec<Option<T>> = values.into_iter().map(Some).collect();
    order
        .iter()
        .map(|index| slots[*index].take().expect("a draw order is a permutation"))
        .collect()
}

/// A pool of Groth16 statements gnark proved, decoded into harness fixtures.
///
/// No code in this repository made these proofs. The pool file hash is pinned
/// and construction fails unless all three lanes reproduce the verdict gnark
/// recorded for every vector, so a lane that accepts an invalid proof or
/// rejects a valid one cannot reach a timed region.
pub struct GnarkFreshGroth16Pool {
    valid: Vec<Groth16Fixture>,
    invalid: Vec<Groth16Fixture>,
}

impl GnarkFreshGroth16Pool {
    /// The frozen draw order. A campaign passes its run seed instead.
    pub fn deterministic() -> Self {
        Self::from_seed(0)
    }

    /// Decode and validate the repository-owned gnark pool once, then place it
    /// in this session's draw order.
    pub fn from_seed(fixture_seed: u64) -> Self {
        let pool_bytes = include_bytes!("../fixtures/gnark-fresh-20260816/fixture.json");
        assert_eq!(
            format!("{:x}", Sha256::digest(pool_bytes)),
            GNARK_FRESH_POOL_SHA256,
            "gnark fresh pool hash"
        );
        let document: GnarkFreshPoolDocument =
            serde_json::from_slice(pool_bytes).expect("valid gnark fresh pool JSON");
        assert_eq!(document.schema, "helius.gnark-fresh-groth16-pool.v1");
        assert_eq!(document.gnark_version, "v0.15.0");
        assert_eq!(document.gnark_crypto_version, "v0.20.1");
        assert_eq!(document.run_seed, "0x474e41524b465348");
        assert_eq!(document.valid.len(), GNARK_FRESH_POOL_SIZE);
        assert!(!document.invalid.is_empty());

        let vk = document.verifying_key.to_ark();
        let valid = document
            .valid
            .iter()
            .enumerate()
            .map(|(index, vector)| vector.checked_fixture(&vk, index))
            .collect::<Vec<_>>();
        let invalid = document
            .invalid
            .iter()
            .enumerate()
            .map(|(index, vector)| vector.checked_fixture(&vk, GNARK_FRESH_POOL_SIZE + index))
            .collect();
        // A repeated statement replays one gamma_abc MSM, which is the part of
        // the verifier most open to memoization.
        assert_eq!(
            valid
                .iter()
                .map(Groth16Fixture::sha256)
                .collect::<HashSet<_>>()
                .len(),
            GNARK_FRESH_POOL_SIZE,
            "gnark pool vectors must be distinct"
        );
        let valid = draw_order(
            &seeded_order(fixture_seed, GNARK_FRESH_DRAW_DOMAIN, valid.len()),
            valid,
        );
        Self { valid, invalid }
    }

    /// Digest of the pinned pool artifact and this session's draw order.
    pub fn order_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"helius-gnark-fresh-draw-order-v1");
        hasher.update(GNARK_FRESH_POOL_SHA256.as_bytes());
        for fixture in &self.valid {
            hasher.update(fixture.sha256().as_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    /// The vectors gnark verified, the only ones a timed row may draw.
    pub fn valid(&self) -> &[Groth16Fixture] {
        &self.valid
    }

    /// The vectors gnark rejected, for the tamper pass only.
    pub fn invalid(&self) -> &[Groth16Fixture] {
        &self.invalid
    }

    pub fn len(&self) -> usize {
        self.valid.len()
    }

    pub fn is_empty(&self) -> bool {
        self.valid.is_empty()
    }

    /// First pool index of one round's draw window.
    pub fn window_start(&self, iterations: usize, rotation: usize) -> usize {
        rotation_window_start(self.valid.len(), iterations, rotation)
    }
}

impl GnarkFreshKeyDocument {
    fn to_ark(&self) -> ark_groth16::VerifyingKey<Bn254> {
        assert_eq!(
            self.k.len(),
            4,
            "one constant wire plus three public inputs"
        );
        ark_groth16::VerifyingKey {
            alpha_g1: decode_ark_g1(&decode_hex(&self.alpha_g1)).expect("gnark alpha G1"),
            beta_g2: decode_ark_g2(&decode_hex(&self.beta_g2)).expect("gnark beta G2"),
            gamma_g2: decode_ark_g2(&decode_hex(&self.gamma_g2)).expect("gnark gamma G2"),
            delta_g2: decode_ark_g2(&decode_hex(&self.delta_g2)).expect("gnark delta G2"),
            gamma_abc_g1: self
                .k
                .iter()
                .map(|point| decode_ark_g1(&decode_hex(point)).expect("gnark K G1"))
                .collect(),
        }
    }
}

impl GnarkFreshVectorDocument {
    fn checked_fixture(
        &self,
        vk: &ark_groth16::VerifyingKey<Bn254>,
        index: usize,
    ) -> Groth16Fixture {
        let proof = Proof {
            a: decode_ark_g1(&decode_hex(&self.proof_a)).expect("gnark proof A"),
            b: decode_ark_g2(&decode_hex(&self.proof_b)).expect("gnark proof B"),
            c: decode_ark_g1(&decode_hex(&self.proof_c)).expect("gnark proof C"),
        };
        let public_inputs = self
            .public_inputs
            .iter()
            .map(|value| {
                decode_ark_fr(&decode_hex(value)).expect("gnark public input is canonical")
            })
            .collect();
        let expected = match self.gnark_verdict.as_str() {
            "PASS" => true,
            "FAIL" => false,
            other => panic!("gnark verdict on vector {index} is {other}"),
        };
        let fixture = Groth16Fixture::from_parts(vk, proof, public_inputs);
        for (lane, verdict) in [
            ("helius", helius_groth16_verify(&fixture)),
            ("arkworks", ark_groth16_verify(&fixture)),
            ("mcl", mcl_groth16_verify(&fixture)),
        ] {
            assert_eq!(
                verdict, expected,
                "{lane} disagrees with gnark on vector {index}"
            );
        }
        fixture
    }
}

// ---------------------------------------------------------------------------
// Production Groth16 pool.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ProductionPoolDocument {
    schema: String,
    gnark_version: String,
    gnark_crypto_version: String,
    source_revision: String,
    run_seed: String,
    keys: Vec<ProductionKeyDocument>,
    standard: Vec<ProductionVectorDocument>,
    committed: Vec<ProductionVectorDocument>,
    invalid: Vec<ProductionVectorDocument>,
}

#[derive(Deserialize)]
struct ProductionKeyDocument {
    name: String,
    committed: bool,
    public_input_count: usize,
    alpha_g1: String,
    beta_g2: String,
    gamma_g2: String,
    delta_g2: String,
    k: Vec<String>,
    #[serde(default)]
    commitment_key_g: String,
    #[serde(default)]
    commitment_key_g_sigma_neg: String,
    public_and_commitment_committed: Vec<Vec<usize>>,
}

#[derive(Deserialize)]
struct ProductionVectorDocument {
    key: String,
    class: String,
    public_inputs: Vec<String>,
    proof_a: String,
    proof_b: String,
    proof_c: String,
    #[serde(default)]
    commitment: String,
    #[serde(default)]
    commitment_pok: String,
    #[serde(default)]
    commitment_hash: String,
    gnark_verdict: String,
}

/// One plain production verifying key, in both the Arkworks form the standard
/// fixtures need and the flat record the batch contexts consume.
struct ProductionVerifyingKey {
    name: String,
    ark: ark_groth16::VerifyingKey<Bn254>,
    record: Vec<u8>,
}

/// The production Groth16 statements a validator verifies.
///
/// No code in this repository proved anything here. The circuits, the
/// constraint systems, the proving keys, and the accepting verdict come from
/// the originating prover and gnark. Construction fails unless all three lanes reproduce the
/// verdict the originating verifier recorded for every vector, so a lane that
/// accepts an invalid proof or rejects a valid one cannot reach a timed region.
pub struct ProductionGroth16Pool {
    keys: Vec<ProductionVerifyingKey>,
    standard: Vec<Groth16Fixture>,
    standard_key: Vec<usize>,
    committed: Vec<GnarkCommittedGroth16Fixture>,
    invalid_standard: Vec<Groth16Fixture>,
    invalid_standard_key: Vec<usize>,
    invalid_committed: Vec<GnarkCommittedGroth16Fixture>,
}

impl ProductionGroth16Pool {
    /// The frozen draw order. A campaign passes its run seed instead.
    pub fn deterministic() -> Self {
        Self::from_seed(0)
    }

    /// Decode and validate the repository-owned production pool once, then place
    /// both timed pools in this session's draw order.
    pub fn from_seed(fixture_seed: u64) -> Self {
        let pool_bytes = include_bytes!("../fixtures/production-20260816/fixture.json");
        assert_eq!(
            format!("{:x}", Sha256::digest(pool_bytes)),
            PRODUCTION_POOL_SHA256,
            "production pool hash"
        );
        let document: ProductionPoolDocument =
            serde_json::from_slice(pool_bytes).expect("valid production pool JSON");
        assert_eq!(document.schema, "helius.production-groth16-pool.v1");
        assert_eq!(document.gnark_version, "v0.15.0");
        assert_eq!(document.gnark_crypto_version, "v0.20.1");
        assert_eq!(
            document.source_revision,
            "5330a112cb10e7622585f61cde2397d26721fdb6"
        );
        assert_eq!(document.run_seed, "0x005a4f4c414e4131");
        assert_eq!(document.standard.len(), PRODUCTION_STANDARD_POOL_SIZE);
        assert_eq!(document.committed.len(), PRODUCTION_COMMITTED_POOL_SIZE);
        assert!(!document.invalid.is_empty());

        let keys: Vec<ProductionVerifyingKey> = document
            .keys
            .iter()
            .filter(|key| !key.committed)
            .map(ProductionKeyDocument::to_plain_key)
            .collect();
        assert!(
            keys.len() >= 2,
            "the different-key batch row needs at least two plain verifying keys"
        );

        let committed_key = document
            .keys
            .iter()
            .find(|key| key.committed)
            .expect("a committed verifying key");
        assert_eq!(committed_key.public_input_count, 1);
        assert_eq!(committed_key.k.len(), 3);

        let standard_index = |name: &str| {
            keys.iter()
                .position(|key| key.name == name)
                .unwrap_or_else(|| panic!("vector names unknown key {name}"))
        };
        let mut standard = Vec::with_capacity(document.standard.len());
        let mut standard_key = Vec::with_capacity(document.standard.len());
        for vector in &document.standard {
            let index = standard_index(&vector.key);
            standard.push(vector.checked_standard(&keys[index].ark));
            standard_key.push(index);
        }
        let committed = document
            .committed
            .iter()
            .map(|vector| vector.checked_committed(committed_key))
            .collect::<Vec<_>>();

        let mut invalid_standard = Vec::new();
        let mut invalid_standard_key = Vec::new();
        let mut invalid_committed = Vec::new();
        for vector in &document.invalid {
            assert_eq!(vector.gnark_verdict, "FAIL");
            match vector.class.as_str() {
                "standard" => {
                    let index = standard_index(&vector.key);
                    invalid_standard.push(vector.checked_standard(&keys[index].ark));
                    invalid_standard_key.push(index);
                }
                "committed" => invalid_committed.push(vector.checked_committed(committed_key)),
                other => panic!("unknown vector class {other}"),
            }
        }
        assert!(!invalid_standard.is_empty() && !invalid_committed.is_empty());

        // A repeated statement replays one gamma_abc MSM, which is the part of
        // the verifier most open to memoization.
        assert_unique_digests(standard.iter().map(Groth16Fixture::sha256));
        assert_unique_digests(committed.iter().map(GnarkCommittedGroth16Fixture::sha256));

        // The key of a vector travels with it, so a batch row keeps its key
        // structure whatever order the session draws.
        let order = seeded_order(
            fixture_seed,
            PRODUCTION_STANDARD_DRAW_DOMAIN,
            standard.len(),
        );
        let standard_key: Vec<usize> = order.iter().map(|index| standard_key[*index]).collect();
        let standard = draw_order(&order, standard);
        let committed = draw_order(
            &seeded_order(
                fixture_seed,
                PRODUCTION_COMMITTED_DRAW_DOMAIN,
                committed.len(),
            ),
            committed,
        );

        Self {
            keys,
            standard,
            standard_key,
            committed,
            invalid_standard,
            invalid_standard_key,
            invalid_committed,
        }
    }

    /// Digest of the pinned pool artifact and this session's draw order.
    pub fn order_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        // Frozen byte domain. It separates this draw from the gnark draw and
        // it is escaped so that a rename cannot move the sealed digests.
        hasher.update(b"helius-\x7a\x6f\x6c\x61\x6e\x61-draw-order-v1");
        hasher.update(PRODUCTION_POOL_SHA256.as_bytes());
        for fixture in &self.standard {
            hasher.update(fixture.sha256().as_bytes());
        }
        for fixture in &self.committed {
            hasher.update(fixture.sha256().as_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    pub fn standard(&self) -> &[Groth16Fixture] {
        &self.standard
    }

    pub fn committed(&self) -> &[GnarkCommittedGroth16Fixture] {
        &self.committed
    }

    pub fn invalid_standard(&self) -> &[Groth16Fixture] {
        &self.invalid_standard
    }

    pub fn invalid_committed(&self) -> &[GnarkCommittedGroth16Fixture] {
        &self.invalid_committed
    }

    /// The transcript hashes the committed pool takes as verifier material,
    /// flattened for the bridge's pooled loop.
    pub fn committed_hashes(&self) -> Vec<u8> {
        self.committed
            .iter()
            .flat_map(|fixture| *fixture.commitment_hash_bytes())
            .collect()
    }

    /// First pool index of one round's draw window.
    pub fn window_start(&self, length: usize, iterations: usize, rotation: usize) -> usize {
        rotation_window_start(length, iterations, rotation)
    }

    /// Run the mcl committed-wires loop over the rotating pool. Each context
    /// carries its own transcript hash, so the loop never replays one proof.
    pub fn mcl_committed_run(&self, hashes: &[u8], iterations: usize, rotation: usize) -> u64 {
        let handles: Vec<*mut c_void> = self.committed.iter().map(|f| f.mcl.as_ptr()).collect();
        assert_eq!(hashes.len(), handles.len() * 32);
        let mut digest = 0;
        // SAFETY: the pool owns every context and every hash for the call.
        let status = unsafe {
            ffi::mcl_gnark_committed_pool_run(
                handles.as_ptr(),
                hashes.as_ptr(),
                handles.len(),
                iterations,
                rotation,
                &mut digest,
            )
        };
        assert_eq!(status, 0, "MCL committed pool loop failed with {status}");
        digest
    }

    /// Batches of `size` proofs, every member under one verifying key.
    pub fn same_key_batches(&self, size: usize, count: usize) -> Vec<Groth16BatchFixture> {
        (0..count)
            .map(|batch| {
                let key = batch % self.keys.len();
                let members = self.same_key_members(size, key, batch / self.keys.len());
                self.batch(&members)
            })
            .collect()
    }

    /// Batches of `size` proofs spread over every verifying key. A size the
    /// key count does not divide gives the leading keys the extra member.
    pub fn different_key_batches(&self, size: usize, count: usize) -> Vec<Groth16BatchFixture> {
        (0..count)
            .map(|batch| {
                let members = self.mixed_key_members(size, batch);
                self.batch(&members)
            })
            .collect()
    }

    /// Every placement of every gnark-rejected vector the pool ships, at every
    /// index of both batch shapes. No lane may accept one of these.
    pub fn invalid_batches(&self, size: usize, mixed: bool) -> Vec<Groth16BatchFixture> {
        let mut batches = Vec::new();
        for (bad, bad_key) in self.invalid_standard.iter().zip(&self.invalid_standard_key) {
            let template = if mixed {
                self.mixed_key_members(size, 0)
            } else {
                self.same_key_members(size, *bad_key, 0)
            };
            for position in 0..size {
                let mut members = template.clone();
                members[position] = (*bad_key, bad);
                batches.push(self.assemble(&members));
            }
        }
        batches
    }

    /// `size` proofs under `key`, taken from the `window`th disjoint block.
    fn same_key_members(
        &self,
        size: usize,
        key: usize,
        window: usize,
    ) -> Vec<(usize, &Groth16Fixture)> {
        let members: Vec<(usize, &Groth16Fixture)> = self
            .standard
            .iter()
            .enumerate()
            .filter(|(index, _)| self.standard_key[*index] == key)
            .map(|(_, fixture)| (key, fixture))
            .skip(window * size)
            .take(size)
            .collect();
        assert_eq!(members.len(), size, "the pool has enough proofs per key");
        members
    }

    fn mixed_key_members(&self, size: usize, window: usize) -> Vec<(usize, &Groth16Fixture)> {
        let keys = self.keys.len();
        let mut members = Vec::with_capacity(size);
        for key in 0..keys {
            let share = size / keys + usize::from(key < size % keys);
            members.extend(self.same_key_members(share, key, window));
        }
        assert_eq!(members.len(), size);
        members
    }

    fn batch(&self, members: &[(usize, &Groth16Fixture)]) -> Groth16BatchFixture {
        let (keys, entries) = self.localize(members);
        Groth16BatchFixture::new(&keys, &entries)
    }

    fn assemble(&self, members: &[(usize, &Groth16Fixture)]) -> Groth16BatchFixture {
        let (keys, entries) = self.localize(members);
        Groth16BatchFixture::assemble(&keys, &entries)
    }

    /// Map the pool-wide key index of every member onto the batch's own key
    /// list, which the bridge indexes.
    fn localize<'a>(
        &'a self,
        members: &[(usize, &'a Groth16Fixture)],
    ) -> (
        Vec<&'a ProductionVerifyingKey>,
        Vec<(usize, &'a Groth16Fixture)>,
    ) {
        let used: Vec<usize> = {
            let mut used: Vec<usize> = members.iter().map(|(key, _)| *key).collect();
            used.sort_unstable();
            used.dedup();
            used
        };
        let entries = members
            .iter()
            .map(|(key, fixture)| {
                let local = used
                    .iter()
                    .position(|candidate| candidate == key)
                    .expect("member keys were collected from the same list");
                (local, *fixture)
            })
            .collect();
        (used.iter().map(|key| &self.keys[*key]).collect(), entries)
    }
}

impl ProductionKeyDocument {
    fn to_plain_key(&self) -> ProductionVerifyingKey {
        assert_eq!(
            self.public_input_count, 1,
            "production keys carry one public input"
        );
        assert_eq!(self.k.len(), 2, "one constant wire plus one public input");
        assert!(self.public_and_commitment_committed.is_empty());
        let alpha = G1Bytes(decode_hex(&self.alpha_g1));
        let beta = G2Bytes(decode_hex(&self.beta_g2));
        let gamma = G2Bytes(decode_hex(&self.gamma_g2));
        let delta = G2Bytes(decode_hex(&self.delta_g2));
        let k: Vec<G1Bytes> = self
            .k
            .iter()
            .map(|point| G1Bytes(decode_hex(point)))
            .collect();
        let mut record = Vec::with_capacity(576);
        record.extend_from_slice(&alpha.0);
        record.extend_from_slice(&beta.0);
        record.extend_from_slice(&gamma.0);
        record.extend_from_slice(&delta.0);
        for point in &k {
            record.extend_from_slice(&point.0);
        }
        ProductionVerifyingKey {
            name: self.name.clone(),
            ark: ark_groth16::VerifyingKey {
                alpha_g1: decode_ark_g1(&alpha.0).expect("production alpha G1"),
                beta_g2: decode_ark_g2(&beta.0).expect("production beta G2"),
                gamma_g2: decode_ark_g2(&gamma.0).expect("production gamma G2"),
                delta_g2: decode_ark_g2(&delta.0).expect("production delta G2"),
                gamma_abc_g1: k
                    .iter()
                    .map(|point| decode_ark_g1(&point.0).expect("production K G1"))
                    .collect(),
            },
            record,
        }
    }
}

impl ProductionVectorDocument {
    fn expected(&self) -> bool {
        match self.gnark_verdict.as_str() {
            "PASS" => true,
            "FAIL" => false,
            other => panic!("production verdict is {other}"),
        }
    }

    fn checked_standard(&self, vk: &ark_groth16::VerifyingKey<Bn254>) -> Groth16Fixture {
        let proof = Proof {
            a: decode_ark_g1(&decode_hex(&self.proof_a)).expect("production proof A"),
            b: decode_ark_g2(&decode_hex(&self.proof_b)).expect("production proof B"),
            c: decode_ark_g1(&decode_hex(&self.proof_c)).expect("production proof C"),
        };
        let public_inputs = self
            .public_inputs
            .iter()
            .map(|value| decode_ark_fr(&decode_hex(value)).expect("production public input"))
            .collect();
        let fixture = Groth16Fixture::from_parts(vk, proof, public_inputs);
        let expected = self.expected();
        for (lane, accepted) in [
            ("helius", helius_groth16_verify(&fixture)),
            ("arkworks", ark_groth16_verify(&fixture)),
            ("mcl", mcl_groth16_verify(&fixture)),
        ] {
            assert_eq!(
                accepted, expected,
                "{lane} disagrees with the originating verifier on a production vector"
            );
        }
        fixture
    }

    fn checked_committed(&self, key: &ProductionKeyDocument) -> GnarkCommittedGroth16Fixture {
        let public_input: [u8; 32] = decode_hex(&self.public_inputs[0]);
        let committed = key
            .public_and_commitment_committed
            .first()
            .expect("a committed key names its committed wires");
        let fixture = GnarkCommittedGroth16Fixture::from_parts(CommittedParts {
            alpha: G1Bytes(decode_hex(&key.alpha_g1)),
            beta: G2Bytes(decode_hex(&key.beta_g2)),
            gamma: G2Bytes(decode_hex(&key.gamma_g2)),
            delta: G2Bytes(decode_hex(&key.delta_g2)),
            k: key
                .k
                .iter()
                .map(|point| G1Bytes(decode_hex(point)))
                .collect::<Vec<_>>()
                .try_into()
                .expect("three production K points"),
            commitment_key_g: G2Bytes(decode_hex(&key.commitment_key_g)),
            commitment_key_g_sigma_neg: G2Bytes(decode_hex(&key.commitment_key_g_sigma_neg)),
            proof_a: G1Bytes(decode_hex(&self.proof_a)),
            proof_b: G2Bytes(decode_hex(&self.proof_b)),
            proof_c: G1Bytes(decode_hex(&self.proof_c)),
            commitment: G1Bytes(decode_hex(&self.commitment)),
            commitment_pok: G1Bytes(decode_hex(&self.commitment_pok)),
            public_input,
            transcript_inputs: committed
                .iter()
                .map(|wire| {
                    assert_eq!(*wire, 1, "this key commits only to its first public wire");
                    public_input
                })
                .collect(),
        });
        assert_eq!(
            fixture.commitment_hash_bytes(),
            &decode_hex::<32>(&self.commitment_hash),
            "the harness transcript must reproduce gnark's"
        );
        let expected = self.expected();
        for (lane, accepted) in [
            ("helius", helius_gnark_committed_groth16_verify(&fixture)),
            ("arkworks", ark_gnark_committed_groth16_verify(&fixture)),
            ("mcl", mcl_gnark_committed_groth16_verify(&fixture)),
        ] {
            assert_eq!(
                accepted, expected,
                "{lane} disagrees with the originating verifier on a production committed vector"
            );
        }
        fixture
    }
}

fn assert_unique_digests<'a>(values: impl Iterator<Item = &'a str>) {
    let values: Vec<&str> = values.collect();
    assert_eq!(
        values.iter().collect::<HashSet<_>>().len(),
        values.len(),
        "pool vectors must be distinct"
    );
}

/// One random-linear-combination batch of production Groth16 proofs.
///
/// The challenges are fixture data, drawn once outside every timer, the way a
/// verifier draws them from its own randomness before it starts. Every scalar
/// multiplication the combination needs runs inside the timed region, so the
/// row measures a batch verification and not a replay of precomputed points.
pub struct Groth16BatchFixture {
    keys: Vec<BatchKey>,
    members: Vec<BatchMember>,
    fixture_sha256: String,
    // Kept for the tamper pass, which rebuilds one mcl context off the timed
    // path. No timed region touches these.
    key_blob: Vec<u8>,
    member_keys: Vec<u32>,
    mcl: NonNull<c_void>,
}

struct BatchKey {
    ark_alpha: ArkG1,
    ark_k: [ArkG1; 2],
    helius_alpha: G1Affine,
    helius_k: [G1Affine; 2],
    /// The key-owned G2 line schedules, in gamma, delta, beta order. A
    /// verifying key is fixed, so every lane builds these once here and
    /// replays them inside the timed region. Only a member's own B stays live.
    helius_prepared: PreparedVerifier,
    ark_prepared: [ArkG2Prepared; 3],
}

/// Slot of each key-owned G2 in a batch key's prepared schedules.
const BATCH_GAMMA: usize = 0;
const BATCH_DELTA: usize = 1;
const BATCH_BETA: usize = 2;

#[derive(Clone)]
struct BatchMember {
    key: usize,
    ark_a: ArkG1,
    ark_b: ArkG2,
    ark_c: ArkG1,
    ark_input: ArkFr,
    ark_challenge: ArkFr,
    helius_a: G1Affine,
    helius_b: G2Affine,
    helius_c: G1Affine,
    helius_input: Fr,
    helius_challenge: Fr,
}

impl Groth16BatchFixture {
    /// A batch every lane must accept.
    fn new(keys: &[&ProductionVerifyingKey], entries: &[(usize, &Groth16Fixture)]) -> Self {
        let fixture = Self::assemble(keys, entries);
        assert!(helius_groth16_batch_verify(&fixture));
        assert!(ark_groth16_batch_verify(&fixture));
        assert!(mcl_groth16_batch_verify(&fixture));
        fixture
    }

    /// A batch with no verdict expectation, so a negative pass can carry a
    /// member the originating verifier rejected.
    fn assemble(keys: &[&ProductionVerifyingKey], entries: &[(usize, &Groth16Fixture)]) -> Self {
        assert!(!keys.is_empty() && !entries.is_empty());
        let mut key_blob = Vec::with_capacity(keys.len() * 576);
        let mut member_blob = Vec::with_capacity(entries.len() * 320);
        let mut member_keys = Vec::with_capacity(entries.len());
        // The digest binds membership alone, and the uniqueness assertion over
        // these digests is what proves the builder never reuses a member.
        let mut hasher = Sha256::new();
        // Frozen byte domain, escaped so that a rename cannot move the sealed
        // membership digests.
        hasher.update(b"helius-\x7a\x6f\x6c\x61\x6e\x61-groth16-batch-membership-v1");
        for key in keys {
            hasher.update(key.name.as_bytes());
            hasher.update(&key.record);
        }
        for (key, fixture) in entries {
            hasher.update((*key as u64).to_be_bytes());
            hasher.update(fixture.sha256().as_bytes());
        }
        let transcript: [u8; 32] = hasher.finalize().into();

        let batch_keys = keys
            .iter()
            .map(|key| {
                key_blob.extend_from_slice(&key.record);
                BatchKey {
                    ark_alpha: key.ark.alpha_g1,
                    ark_k: [key.ark.gamma_abc_g1[0], key.ark.gamma_abc_g1[1]],
                    helius_alpha: helius_g1(key.ark.alpha_g1),
                    helius_k: [
                        helius_g1(key.ark.gamma_abc_g1[0]),
                        helius_g1(key.ark.gamma_abc_g1[1]),
                    ],
                    helius_prepared: PreparedVerifier::new(
                        &[
                            helius_g2(key.ark.gamma_g2),
                            helius_g2(key.ark.delta_g2),
                            helius_g2(key.ark.beta_g2),
                        ],
                        &[],
                    )
                    .expect("a verifying key carries valid nonidentity G2"),
                    ark_prepared: [
                        ArkG2Prepared::from(key.ark.gamma_g2),
                        ArkG2Prepared::from(key.ark.delta_g2),
                        ArkG2Prepared::from(key.ark.beta_g2),
                    ],
                }
            })
            .collect::<Vec<_>>();

        let members = entries
            .iter()
            .enumerate()
            .map(|(position, (key, fixture))| {
                let challenge = groth16_batch_coefficient(&transcript, position);
                let input = fixture.ark_public_inputs[0];
                member_blob.extend_from_slice(&encode_ark_g1(fixture.ark_proof.a.into_group()).0);
                member_blob.extend_from_slice(&encode_ark_g2(fixture.ark_proof.b.into_group()).0);
                member_blob.extend_from_slice(&encode_ark_g1(fixture.ark_proof.c.into_group()).0);
                member_blob.extend_from_slice(&encode_ark_fr(input).0);
                member_blob.extend_from_slice(&encode_ark_fr(challenge).0);
                member_keys.push(*key as u32);
                BatchMember {
                    key: *key,
                    ark_a: fixture.ark_proof.a,
                    ark_b: fixture.ark_proof.b,
                    ark_c: fixture.ark_proof.c,
                    ark_input: input,
                    ark_challenge: challenge,
                    helius_a: fixture.helius_proof_a,
                    helius_b: fixture.helius_proof_b,
                    helius_c: fixture.helius_proof_c,
                    helius_input: encode_ark_fr(input).to_fr().expect("canonical input"),
                    helius_challenge: encode_ark_fr(challenge)
                        .to_fr()
                        .expect("canonical challenge"),
                }
            })
            .collect::<Vec<_>>();

        assert_eq!(
            entries
                .iter()
                .map(|(_, fixture)| fixture.sha256())
                .collect::<HashSet<_>>()
                .len(),
            entries.len(),
            "a batch never repeats a member"
        );
        // Two members under one coefficient cancel against each other, so the
        // combination proves nothing about either proof.
        assert_eq!(
            members
                .iter()
                .map(|member| member.ark_challenge)
                .collect::<HashSet<_>>()
                .len(),
            members.len(),
            "batch coefficients must be pairwise distinct"
        );

        ensure_mcl();
        let mut handle = core::ptr::null_mut();
        // SAFETY: both blobs hold the documented fixed-width records and the
        // key index of every member is in range.
        let status = unsafe {
            ffi::mcl_groth16_batch_create(
                key_blob.as_ptr(),
                batch_keys.len(),
                member_blob.as_ptr(),
                member_keys.as_ptr(),
                members.len(),
                &mut handle,
            )
        };
        assert_eq!(status, 0, "mcl Groth16 batch context failed with {status}");

        let mut fixture_sha256 = String::with_capacity(64);
        for byte in transcript {
            write!(&mut fixture_sha256, "{byte:02x}").expect("a String never fails to format");
        }
        Self {
            keys: batch_keys,
            members,
            fixture_sha256,
            key_blob,
            member_keys,
            mcl: NonNull::new(handle).expect("MCL returned a Groth16 batch context"),
        }
    }

    pub fn sha256(&self) -> &str {
        &self.fixture_sha256
    }

    pub fn proof_count(&self) -> usize {
        self.members.len()
    }

    pub fn key_count(&self) -> usize {
        self.keys.len()
    }
}

impl Drop for Groth16BatchFixture {
    fn drop(&mut self) {
        // SAFETY: this handle is uniquely owned and was returned by create.
        unsafe { ffi::mcl_groth16_batch_destroy(self.mcl.as_ptr()) };
    }
}

/// How many distinct draw starts a pool serves one session.
///
/// A round draws `min(iterations, pool_len)` entries from its start, so two
/// rounds take the same start only past this count. The capacity is the number
/// of block-aligned starts times the block count, which is every pool index
/// that a whole block still fits behind.
pub fn rotation_window_capacity(pool_len: usize, iterations: usize) -> usize {
    assert!(pool_len > 1, "a rotation needs more than one fixture");
    assert!(iterations > 0, "a round draws at least one fixture");
    let span = iterations.min(pool_len);
    pool_len - pool_len % span
}

/// Whether the rounds of a row can draw different fixture sets at all.
///
/// A round of `iterations >= pool_len` walks the whole pool, so every round of
/// such a row draws the same set and the rounds differ only in the order they
/// draw it. Set freshness is reachable only below that point, and a row that
/// wants it has to hold a pool wider than one round's draw. The campaign's
/// primitive and group rows run thousands of iterations against pools of eight,
/// so they rotate an order and nothing more. `bench/README.md` lists which
/// rows are on which side of this line.
pub fn rotation_draws_distinct_sets(pool_len: usize, iterations: usize) -> bool {
    iterations < pool_len
}

/// First pool index of one round's draw window.
///
/// The walk is block major, so the first `pool_len / span` rounds draw
/// mutually disjoint windows and every later round takes a start no earlier
/// round took. Distinct starts give distinct fixture sets only while a round
/// draws a proper subset of the pool, which is what
/// [`rotation_draws_distinct_sets`] reports. Past the capacity the pool has no
/// unused start left, and the run stops instead of wrapping in silence.
pub fn rotation_window_start(pool_len: usize, iterations: usize, rotation: usize) -> usize {
    let capacity = rotation_window_capacity(pool_len, iterations);
    assert!(
        rotation < capacity,
        "round {rotation} has no fresh window: a pool of {pool_len} serves {capacity} windows at {iterations} iterations"
    );
    let span = iterations.min(pool_len);
    let blocks = capacity / span;
    (rotation % blocks) * span + rotation / blocks
}

/// Random-linear-combination coefficient width. A forged batch survives with
/// probability at most `2^-BATCH_COEFFICIENT_BITS`, so 128 bits buys the same
/// 128-bit soundness BN254 itself targets and a wider draw buys nothing.
const BATCH_COEFFICIENT_BITS: usize = 128;

/// One batch coefficient.
///
/// `transcript` is the membership digest over every verifying key and every
/// member, so a coefficient exists only after the statements are fixed and no
/// party that chose the proofs can predict it. Zero would drop its member from
/// the combination, so a zero draw maps to one.
fn groth16_batch_coefficient(transcript: &[u8; 32], position: usize) -> ArkFr {
    const BYTES: usize = BATCH_COEFFICIENT_BITS / 8;
    let mut hasher = Sha256::new();
    // Frozen byte domain. It binds every batch coefficient, so a rename must
    // not touch it. The escapes keep the bytes out of a text substitution.
    hasher.update(b"helius-\x7a\x6f\x6c\x61\x6e\x61-groth16-batch-coefficient-v2");
    hasher.update(transcript);
    hasher.update((position as u64).to_be_bytes());
    let digest = hasher.finalize();
    let mut wide = [0_u8; 32];
    wide[32 - BYTES..].copy_from_slice(&digest[..BYTES]);
    let value = ArkFr::from_be_bytes_mod_order(&wide);
    if value.is_zero() { ArkFr::one() } else { value }
}

fn decode_hex<const N: usize>(value: &str) -> [u8; N] {
    assert_eq!(value.len(), N * 2, "fixed-width fixture hex");
    let mut output = [0_u8; N];
    for (index, byte) in output.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .expect("lowercase fixture hex");
    }
    output
}

/// gnark's `hashToField` over SHA-256, expand_message_xmd with one output.
fn gnark_hash_to_field(message: &[u8], dst: &[u8]) -> ArkFr {
    assert!(dst.len() <= u8::MAX as usize, "gnark XMD domain length");
    const WIDE_BYTES: usize = 48;
    let mut hasher = Sha256::new();
    hasher.update([0_u8; 64]);
    hasher.update(message);
    hasher.update((WIDE_BYTES as u16).to_be_bytes());
    hasher.update([0_u8]);
    hasher.update(dst);
    hasher.update([dst.len() as u8]);
    let b0 = hasher.finalize_reset();

    hasher.update(b0);
    hasher.update([1_u8]);
    hasher.update(dst);
    hasher.update([dst.len() as u8]);
    let b1 = hasher.finalize_reset();
    let mut xor = [0_u8; 32];
    for index in 0..xor.len() {
        xor[index] = b0[index] ^ b1[index];
    }
    hasher.update(xor);
    hasher.update([2_u8]);
    hasher.update(dst);
    hasher.update([dst.len() as u8]);
    let b2 = hasher.finalize();

    let mut wide = [0_u8; WIDE_BYTES];
    wide[..32].copy_from_slice(&b1);
    wide[32..].copy_from_slice(&b2[..16]);
    ArkFr::from_be_bytes_mod_order(&wide)
}

/// gnark's BSB22 transcript hash. `committed` holds the public-input values
/// the commitment binds, in `PublicAndCommitmentCommitted` order. A wrong
/// preimage moves the hash, which enters the verifier's gamma_abc MSM as an
/// extra public input, so it rejects a sound proof.
fn gnark_commitment_hash(commitment: [u8; 64], committed: &[[u8; 32]]) -> ArkFr {
    let mut preimage = Vec::with_capacity(64 + committed.len() * 32);
    preimage.extend_from_slice(&commitment);
    for value in committed {
        preimage.extend_from_slice(value);
    }
    gnark_hash_to_field(&preimage, b"bsb22-commitment")
}

#[allow(clippy::too_many_arguments)]
fn gnark_committed_groth16_blob(
    alpha: &G1Bytes,
    beta: &G2Bytes,
    gamma: &G2Bytes,
    delta: &G2Bytes,
    k: &[G1Bytes; 3],
    commitment_key_g: &G2Bytes,
    commitment_key_g_sigma_neg: &G2Bytes,
    proof_a: &G1Bytes,
    proof_b: &G2Bytes,
    proof_c: &G1Bytes,
    commitment: &G1Bytes,
    commitment_pok: &G1Bytes,
    public_input: [u8; 32],
) -> Vec<u8> {
    let mut blob = Vec::with_capacity(1312);
    blob.extend_from_slice(&alpha.0);
    blob.extend_from_slice(&beta.0);
    blob.extend_from_slice(&gamma.0);
    blob.extend_from_slice(&delta.0);
    for point in k {
        blob.extend_from_slice(&point.0);
    }
    blob.extend_from_slice(&commitment_key_g.0);
    blob.extend_from_slice(&commitment_key_g_sigma_neg.0);
    blob.extend_from_slice(&proof_a.0);
    blob.extend_from_slice(&proof_b.0);
    blob.extend_from_slice(&proof_c.0);
    blob.extend_from_slice(&commitment.0);
    blob.extend_from_slice(&commitment_pok.0);
    blob.extend_from_slice(&public_input);
    assert_eq!(blob.len(), 1312, "committed blob layout is fixed width");
    blob
}

fn groth16_blob(
    vk: &ark_groth16::VerifyingKey<Bn254>,
    proof: &Proof<Bn254>,
    public_inputs: &[ArkFr],
) -> Vec<u8> {
    let mut blob = Vec::new();
    blob.extend_from_slice(&encode_ark_g1(vk.alpha_g1.into_group()).0);
    blob.extend_from_slice(&encode_ark_g2(vk.beta_g2.into_group()).0);
    blob.extend_from_slice(&encode_ark_g2(vk.gamma_g2.into_group()).0);
    blob.extend_from_slice(&encode_ark_g2(vk.delta_g2.into_group()).0);
    for point in &vk.gamma_abc_g1 {
        blob.extend_from_slice(&encode_ark_g1(point.into_group()).0);
    }
    blob.extend_from_slice(&encode_ark_g1(proof.a.into_group()).0);
    blob.extend_from_slice(&encode_ark_g2(proof.b.into_group()).0);
    blob.extend_from_slice(&encode_ark_g1(proof.c.into_group()).0);
    for input in public_inputs {
        blob.extend_from_slice(&encode_ark_fr(*input).0);
    }
    blob
}

fn helius_g1(point: ArkG1) -> G1Affine {
    G1Bytes(encode_ark_g1(point.into_group()).0)
        .to_affine()
        .expect("Ark G1 is valid Helius G1")
}

fn helius_g2(point: ArkG2) -> G2Affine {
    G2Bytes(encode_ark_g2(point.into_group()).0)
        .to_affine()
        .expect("Ark G2 is valid Helius G2")
}

const FP_MODULUS: [u8; 32] = [
    0x30, 0x64, 0x4e, 0x72, 0xe1, 0x31, 0xa0, 0x29, 0xb8, 0x50, 0x45, 0xb6, 0x81, 0x81, 0x58, 0x5d,
    0x97, 0x81, 0x6a, 0x91, 0x68, 0x71, 0xca, 0x8d, 0x3c, 0x20, 0x8c, 0x16, 0xd8, 0x7c, 0xfd, 0x47,
];

const FR_MODULUS: [u8; 32] = [
    0x30, 0x64, 0x4e, 0x72, 0xe1, 0x31, 0xa0, 0x29, 0xb8, 0x50, 0x45, 0xb6, 0x81, 0x81, 0x58, 0x5d,
    0x28, 0x33, 0xe8, 0x48, 0x79, 0xb9, 0x70, 0x91, 0x43, 0xe1, 0xf5, 0x93, 0xf0, 0x00, 0x00, 0x01,
];

fn field_be<F: PrimeField>(value: F) -> [u8; 32] {
    let encoded = value.into_bigint().to_bytes_be();
    let mut output = [0_u8; 32];
    output[32 - encoded.len()..].copy_from_slice(&encoded);
    output
}

fn encode_ark_fr(value: ArkFr) -> ScalarBytes {
    ScalarBytes(field_be(value))
}

fn encode_ark_g1(point: ArkG1Projective) -> G1Bytes {
    let mut output = [0_u8; 64];
    if let Some((x, y)) = point.into_affine().xy() {
        output[..32].copy_from_slice(&field_be(x));
        output[32..].copy_from_slice(&field_be(y));
    }
    G1Bytes(output)
}

fn encode_ark_g2(point: ArkG2Projective) -> G2Bytes {
    let mut output = [0_u8; 128];
    if let Some((x, y)) = point.into_affine().xy() {
        output[..32].copy_from_slice(&field_be(x.c1));
        output[32..64].copy_from_slice(&field_be(x.c0));
        output[64..96].copy_from_slice(&field_be(y.c1));
        output[96..].copy_from_slice(&field_be(y.c0));
    }
    G2Bytes(output)
}

fn flatten_g1(values: &[G1Bytes]) -> Vec<u8> {
    values.iter().flat_map(|value| value.0).collect()
}

fn flatten_scalars(values: &[ScalarBytes]) -> Vec<u8> {
    values.iter().flat_map(|value| value.0).collect()
}

struct FixtureHasher(Sha256);

impl FixtureHasher {
    fn new(operation: &str, count: usize, pool_size: usize) -> Self {
        let mut hasher = Self(Sha256::new());
        // This byte domain predates the product rename. Keep its bytes stable so
        // branding cannot silently change the workload's golden identity.
        hasher.field(b"\x68\x65\x6c\x69\x6f\x73_exact_\x61\x67\x61\x76\x65_fixture_v1");
        hasher.field(operation.as_bytes());
        hasher.field(&(count as u64).to_be_bytes());
        hasher.field(&(pool_size as u64).to_be_bytes());
        hasher
    }

    fn field(&mut self, bytes: &[u8]) {
        self.0.update((bytes.len() as u64).to_be_bytes());
        self.0.update(bytes);
    }

    fn finish(self) -> String {
        let mut output = String::with_capacity(64);
        for byte in self.0.finalize() {
            write!(output, "{byte:02x}").expect("writing to String cannot fail");
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Miller-loop lane adapters.
// ---------------------------------------------------------------------------

pub fn helius_miller(input: &MillerFixture) -> Fp12 {
    helius_narsil::pairing::miller_loop(&input.helius_g1, &input.helius_g2)
}

pub fn ark_miller(input: &MillerFixture) -> ark_ec::pairing::MillerLoopOutput<Bn254> {
    Bn254::multi_miller_loop([input.ark_g1], [input.ark_g2])
}

pub fn helius_final_exponentiation(input: &MillerFixture) -> Fp12 {
    helius_narsil::pairing::final_exponentiation(&input.helius_miller_output)
}

pub fn ark_final_exponentiation(input: &MillerFixture) -> ark_ec::pairing::PairingOutput<Bn254> {
    Bn254::final_exponentiation(input.ark_miller_output)
        .expect("a Miller result has a final exponentiation")
}

pub fn helius_full_pairing(input: &MillerFixture) -> Fp12 {
    let miller = helius_narsil::pairing::miller_loop(&input.helius_g1, &input.helius_g2);
    helius_narsil::pairing::final_exponentiation(&miller)
}

pub fn ark_full_pairing(input: &MillerFixture) -> ark_ec::pairing::PairingOutput<Bn254> {
    Bn254::pairing(input.ark_g1, input.ark_g2)
}

/// Replay Helius' validated, precomputed G2 line schedule.
pub fn helius_miller_prepared(input: &MillerFixture) -> Fp12 {
    miller_loop_prepared(&input.helius_g1, &input.helius_g2_prepared)
}

/// Prepared-schedule Miller replay and final exponentiation in one call.
pub fn helius_full_pairing_prepared(input: &MillerFixture) -> Fp12 {
    let miller = miller_loop_prepared(&input.helius_g1, &input.helius_g2_prepared);
    helius_narsil::pairing::final_exponentiation(&miller)
}

/// Replay Arkworks' `G2Prepared` schedule.
///
/// Arkworks 0.5 consumes `G2Prepared` by value, so the public API requires a
/// schedule clone for every repeated call. Construction of the schedule itself
/// remains outside the measured operation.
pub fn ark_miller_prepared(input: &MillerFixture) -> ark_ec::pairing::MillerLoopOutput<Bn254> {
    Bn254::multi_miller_loop([input.ark_g1], [input.ark_g2_prepared.clone()])
}

/// Clone Arkworks' prepared schedule for the untimed loop setup.
pub fn ark_miller_prepared_setup(input: &MillerFixture) -> ArkG2Prepared {
    input.ark_g2_prepared.clone()
}

/// Consume one already-cloned Arkworks prepared schedule in the timed replay.
pub fn ark_miller_prepared_replay(
    input: &MillerFixture,
    prepared: ArkG2Prepared,
) -> ark_ec::pairing::MillerLoopOutput<Bn254> {
    Bn254::multi_miller_loop([input.ark_g1], [prepared])
}

/// Consume one already-cloned prepared schedule, then final-exponentiate.
pub fn ark_full_pairing_prepared_replay(
    input: &MillerFixture,
    schedule: ArkG2Prepared,
) -> ark_ec::pairing::PairingOutput<Bn254> {
    let miller = Bn254::multi_miller_loop([input.ark_g1], [schedule]);
    Bn254::final_exponentiation(miller).expect("a Miller result has a final exponentiation")
}

/// Encode a Helius Fp12 value in the comparator's canonical coefficient order.
pub fn encode_helius_fp12(value: Fp12) -> [u8; 384] {
    let coefficients = [
        value.c0.c0.c0,
        value.c0.c0.c1,
        value.c0.c1.c0,
        value.c0.c1.c1,
        value.c0.c2.c0,
        value.c0.c2.c1,
        value.c1.c0.c0,
        value.c1.c0.c1,
        value.c1.c1.c0,
        value.c1.c1.c1,
        value.c1.c2.c0,
        value.c1.c2.c1,
    ];
    let mut output = [0u8; 384];
    for (chunk, coefficient) in output.chunks_exact_mut(32).zip(coefficients) {
        chunk.copy_from_slice(&coefficient.to_bytes_be());
    }
    output
}

/// Encode an Arkworks Fq12 value in the comparator's canonical coefficient order.
pub fn encode_ark_fp12(value: ArkFq12) -> [u8; 384] {
    let coefficients = [
        value.c0.c0.c0,
        value.c0.c0.c1,
        value.c0.c1.c0,
        value.c0.c1.c1,
        value.c0.c2.c0,
        value.c0.c2.c1,
        value.c1.c0.c0,
        value.c1.c0.c1,
        value.c1.c1.c0,
        value.c1.c1.c1,
        value.c1.c2.c0,
        value.c1.c2.c1,
    ];
    let mut output = [0u8; 384];
    for (chunk, coefficient) in output.chunks_exact_mut(32).zip(coefficients) {
        chunk.copy_from_slice(&field_be(coefficient));
    }
    output
}

/// Require the three lanes to agree on the Miller product of one fixture,
/// raw and after final exponentiation, live and prepared.
pub fn assert_miller_equivalent(input: &MillerFixture) {
    let helius_raw = helius_miller(input);
    let ark_raw = ark_miller(input);
    let helius_raw_bytes = encode_helius_fp12(helius_raw);
    assert_eq!(
        helius_raw_bytes,
        encode_ark_fp12(ark_raw.0),
        "Helius and Ark raw Miller outputs",
    );
    let mut mcl_raw = [0u8; 384];
    // SAFETY: the fixture owns a live context and the buffer holds the 384
    // bytes the bridge writes.
    let raw_status = unsafe { ffi::mcl_miller_raw(input.mcl.as_ptr(), mcl_raw.as_mut_ptr()) };
    assert_eq!(
        raw_status, 0,
        "MCL raw Miller export failed with {raw_status}"
    );
    assert_eq!(
        helius_raw_bytes, mcl_raw,
        "Helius and MCL raw Miller outputs"
    );

    let helius = encode_helius_fp12(helius_narsil::pairing::final_exponentiation(&helius_raw));
    let ark = encode_ark_fp12(
        Bn254::final_exponentiation(ark_raw)
            .expect("nonzero Ark Miller output")
            .0,
    );
    let mut mcl = [0u8; 384];
    // SAFETY: as above.
    let status = unsafe { ffi::mcl_miller_final(input.mcl.as_ptr(), mcl.as_mut_ptr()) };
    assert_eq!(status, 0, "MCL Miller finalization failed with {status}");
    assert_eq!(helius, ark, "Helius and Ark Miller/final-exp outputs");
    assert_eq!(helius, mcl, "Helius and MCL Miller/final-exp outputs");

    assert_prepared_miller_equivalent(input);
}

/// Prove that all prepared paths retain the live Miller loop's pairing value.
pub fn assert_prepared_miller_equivalent(input: &MillerFixture) {
    let helius_live_raw = helius_miller(input);
    let ark_live_raw = ark_miller(input);
    let helius_prepared_raw = helius_miller_prepared(input);
    let ark_prepared_raw = ark_miller_prepared(input);

    assert_eq!(
        encode_helius_fp12(helius_prepared_raw),
        encode_helius_fp12(helius_live_raw),
        "Helius prepared and live raw Miller outputs",
    );
    assert_eq!(
        encode_ark_fp12(ark_prepared_raw.0),
        encode_ark_fp12(ark_live_raw.0),
        "Arkworks prepared and live raw Miller outputs",
    );

    let helius = encode_helius_fp12(helius_narsil::pairing::final_exponentiation(
        &helius_prepared_raw,
    ));
    let ark = encode_ark_fp12(
        Bn254::final_exponentiation(ark_prepared_raw)
            .expect("nonzero prepared Ark Miller output")
            .0,
    );
    let mut mcl_live = [0u8; 384];
    // SAFETY: the fixture owns a live context and each buffer holds the 384
    // bytes the bridge writes.
    let live_status = unsafe { ffi::mcl_miller_final(input.mcl.as_ptr(), mcl_live.as_mut_ptr()) };
    assert_eq!(live_status, 0, "MCL live Miller finalization failed");
    let mut mcl_prepared = [0u8; 384];
    // SAFETY: as above.
    let prepared_status =
        unsafe { ffi::mcl_miller_prepared_final(input.mcl.as_ptr(), mcl_prepared.as_mut_ptr()) };
    assert_eq!(
        prepared_status, 0,
        "MCL prepared Miller finalization failed"
    );
    assert_eq!(
        helius, ark,
        "Helius and Ark prepared Miller/final-exp outputs"
    );
    assert_eq!(
        helius, mcl_prepared,
        "Helius and MCL prepared Miller/final-exp outputs"
    );
    assert_eq!(
        mcl_prepared, mcl_live,
        "MCL prepared and live Miller/final-exp outputs"
    );
}

/// The live Miller product and the three lanes' fresh-schedule replays of it,
/// all in canonical coefficient order.
pub struct G2PrepareReplays {
    pub live: [u8; 384],
    pub helius: [u8; 384],
    pub arkworks: [u8; 384],
    pub mcl: [u8; 384],
}

/// Prepare one G2 line schedule per lane exactly the way the timed
/// `g2_prepare_87_lines` row prepares it, then replay each schedule through
/// that same lane's prepared Miller entry.
pub fn g2_prepare_replays(input: &MillerFixture) -> Result<G2PrepareReplays, String> {
    let helius_schedule = prepare_g2_unchecked(&input.helius_g2);
    let ark_schedule = ArkG2Prepared::from(input.ark_g2);
    let mut mcl = [0_u8; 384];
    // SAFETY: the fixture uniquely owns a live Miller context and the output
    // buffer holds the 384 bytes the bridge writes.
    let status = unsafe { ffi::mcl_g2_prepare_replay_raw(input.mcl.as_ptr(), mcl.as_mut_ptr()) };
    if status != 0 {
        return Err(format!("mcl fresh G2 schedule replay failed with {status}"));
    }
    Ok(G2PrepareReplays {
        live: encode_helius_fp12(helius_miller(input)),
        helius: encode_helius_fp12(miller_loop_prepared(&input.helius_g1, &helius_schedule)),
        arkworks: encode_ark_fp12(Bn254::multi_miller_loop([input.ark_g1], [ark_schedule]).0),
        mcl,
    })
}

/// Time one shape of mcl's prepared Miller entry over a resident schedule.
///
/// `pairs` is 1 or 2, the only widths mcl's public prepared entries offer. The
/// pair of measurements is what prices the two-pair ceiling the disclosures
/// name, so both must run on one host in one session.
pub fn mcl_prepared_shape_run(input: &MillerFixture, pairs: usize, iterations: usize) -> u64 {
    let mut digest = 0_u64;
    // SAFETY: the fixture owns a live Miller context with a prepared schedule
    // for the whole call.
    let status =
        unsafe { ffi::mcl_prepared_shape_run(input.mcl.as_ptr(), pairs, iterations, &mut digest) };
    assert_eq!(status, 0, "mcl prepared shape probe failed with {status}");
    digest
}

/// Prove that the three lanes prepare equivalent G2 line schedules.
///
/// `g2_prepare_87_lines` folds a lane-local schedule layout, so no digest can
/// cross lanes for that row. Each lane instead replays its own freshly built
/// schedule and all three replays must equal the live Miller product bit for
/// bit. That is what makes the timed row a comparison of the same artifact.
pub fn g2_prepare_schedules_equivalent(input: &MillerFixture) -> Result<(), String> {
    let replays = g2_prepare_replays(input)?;
    for (lane, replay) in [
        ("helius", replays.helius),
        ("arkworks", replays.arkworks),
        ("mcl", replays.mcl),
    ] {
        if replay != replays.live {
            return Err(format!(
                "{lane} fresh G2 schedule replays to another Miller value"
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Plain Groth16 lane adapters.
// ---------------------------------------------------------------------------

/// Verify one Groth16 statement through Arkworks' native verifier.
pub fn ark_groth16_verify(input: &Groth16Fixture) -> bool {
    Groth16::<Bn254>::verify_proof(&input.ark_pvk, &input.ark_proof, &input.ark_public_inputs)
        .expect("well-formed Arkworks Groth16 verifier input")
}

pub fn helius_public_input_accumulator(input: &Groth16Fixture) -> G1Affine {
    // One multi-scalar multiplication over the variable bases, then the
    // constant wire. A per-input scalar multiplication loop costs several
    // times this for the public-input counts real circuits carry, and the
    // comparator lanes do not pay that shape.
    let variable = msm_variable_time_affine(
        &input.helius_gamma_abc[1..],
        &input.helius_public_input_limbs,
    );
    G1Projective::from(input.helius_gamma_abc[0])
        .add_mixed(variable)
        .to_affine()
}

/// Verify the same Groth16 equation using Helius' prepared verifier context.
pub fn helius_groth16_verify(input: &Groth16Fixture) -> bool {
    let (prepared, live) = input.helius_online_terms();
    input
        .helius_context
        .verify(&prepared, &live)
        .expect("well-formed Helius Groth16 verifier input")
}

/// Verify the same typed Groth16 equation through MCL.
pub fn mcl_groth16_verify(input: &Groth16Fixture) -> bool {
    let mut output = 0u8;
    // SAFETY: the fixture uniquely owns a live context for the call.
    let status = unsafe { ffi::mcl_groth16_verify(input.mcl.as_ptr(), &mut output) };
    assert_eq!(status, 0, "mcl Groth16 verification failed with {status}");
    output != 0
}

/// Helius Groth16 verification returning the online target-group product.
///
/// The Miller product consumes the freshly computed public-input accumulator,
/// so the MSM has a real data dependency into the pairing. Point validation
/// happened at fixture construction, outside every timer. The returned Fp12 is
/// the online product e(A,B) e(-vk_x,gamma) e(-C,delta). Every accepting proof
/// under one key drives it to e(alpha,beta), so it binds the verdict alone. A
/// timed row folds the public-input accumulator with it.
pub fn helius_groth16_verify_digest(input: &Groth16Fixture) -> (bool, Fp12) {
    let accumulator = helius_public_input_accumulator(input);
    let prepared = [
        PreparedTerm {
            g1: accumulator.negate(),
            prepared_g2: 0,
        },
        PreparedTerm {
            g1: input.helius_proof_c.negate(),
            prepared_g2: 1,
        },
    ];
    let live = [LivePair {
        g1: input.helius_proof_a,
        g2: input.helius_proof_b,
    }];
    input
        .helius_context
        .verify_prevalidated(&prepared, &live)
        .expect("fixture terms are in bounds")
}

/// Clone the prepared verifier-key G2 schedules for the timed Arkworks loop.
pub fn ark_groth16_prepared_schedules(input: &Groth16Fixture) -> (ArkG2Prepared, ArkG2Prepared) {
    (
        input.ark_pvk.gamma_g2_neg_pc.clone(),
        input.ark_pvk.delta_g2_neg_pc.clone(),
    )
}

/// Hand-rolled Arkworks twin of [`helius_groth16_verify_digest`], with the
/// same limit on what its Fp12 binds.
///
/// Check-for-check identical to ark-groth16 verify_proof_with_prepared_inputs,
/// e(A,B) e(vk_x,-gamma) e(C,-delta) == e(alpha,beta) with the prepared
/// negated schedules consumed by value.
pub fn ark_groth16_verify_digest(
    input: &Groth16Fixture,
    gamma_neg: ArkG2Prepared,
    delta_neg: ArkG2Prepared,
) -> (bool, ArkFq12) {
    let mut vk_x = input.ark_pvk.vk.gamma_abc_g1[0].into_group();
    for (scalar, base) in input
        .ark_public_inputs
        .iter()
        .zip(input.ark_pvk.vk.gamma_abc_g1.iter().skip(1))
    {
        vk_x += base.mul_bigint(scalar.into_bigint());
    }
    let miller = Bn254::multi_miller_loop(
        [
            <Bn254 as Pairing>::G1Prepared::from(input.ark_proof.a),
            vk_x.into_affine().into(),
            input.ark_proof.c.into(),
        ],
        [input.ark_proof.b.into(), gamma_neg, delta_neg],
    );
    let test = Bn254::final_exponentiation(miller)
        .expect("a Groth16 Miller product has a final exponentiation");
    (test.0 == input.ark_pvk.alpha_g1_beta_g2, test.0)
}

// ---------------------------------------------------------------------------
// Production row semantics. Every lane runs the same equation over the same
// statement, and every digest binds a value the lane had to compute.
// ---------------------------------------------------------------------------

/// Timed helius verification of one production proof.
///
/// The returned accumulator is the freshly computed public-input point, so a
/// digest over it detects a memoized gamma_abc MSM. The Fp12 is the online
/// product every lane tests against e(alpha, beta).
pub fn helius_groth16_verify_accumulated(input: &Groth16Fixture) -> (bool, G1Affine, Fp12) {
    let accumulator = helius_public_input_accumulator(input);
    let prepared = [
        PreparedTerm {
            g1: accumulator.negate(),
            prepared_g2: 0,
        },
        PreparedTerm {
            g1: input.helius_proof_c.negate(),
            prepared_g2: 1,
        },
    ];
    let live = [LivePair {
        g1: input.helius_proof_a,
        g2: input.helius_proof_b,
    }];
    let (accepted, product) = input
        .helius_context
        .verify_prevalidated(&prepared, &live)
        .expect("fixture terms are in bounds");
    (accepted, accumulator, product)
}

/// Arkworks twin of [`helius_groth16_verify_accumulated`], consuming the
/// already-cloned prepared key schedules.
pub fn ark_groth16_verify_accumulated(
    input: &Groth16Fixture,
    gamma_neg: ArkG2Prepared,
    delta_neg: ArkG2Prepared,
) -> (bool, ArkG1, ArkFq12) {
    let mut vk_x = input.ark_pvk.vk.gamma_abc_g1[0].into_group();
    for (scalar, base) in input
        .ark_public_inputs
        .iter()
        .zip(input.ark_pvk.vk.gamma_abc_g1.iter().skip(1))
    {
        vk_x += base.mul_bigint(scalar.into_bigint());
    }
    let accumulator = vk_x.into_affine();
    let miller = Bn254::multi_miller_loop(
        [
            <Bn254 as Pairing>::G1Prepared::from(input.ark_proof.a),
            accumulator.into(),
            input.ark_proof.c.into(),
        ],
        [input.ark_proof.b.into(), gamma_neg, delta_neg],
    );
    let test = Bn254::final_exponentiation(miller)
        .expect("a Groth16 Miller product has a final exponentiation");
    (
        test.0 == input.ark_pvk.alpha_g1_beta_g2,
        accumulator,
        test.0,
    )
}

/// Timed helius verification of one committed-wires production proof.
pub fn helius_committed_verify_accumulated(
    input: &GnarkCommittedGroth16Fixture,
) -> (bool, G1Affine, Fp12) {
    let commitment = input.helius_commitment;
    let k_sum = input.helius_k[0]
        .to_curve()
        .add_projective(
            input.helius_k[1]
                .to_curve()
                .mul_scalar(input.helius_public_input),
        )
        .add_projective(
            input.helius_k[2]
                .to_curve()
                .mul_scalar(input.helius_commitment_hash),
        )
        .add_mixed(commitment)
        .to_affine();
    let (pok_ok, pok_result) = input
        .helius_pok_context
        .verify_prevalidated(
            &[
                PreparedTerm {
                    g1: commitment,
                    prepared_g2: 0,
                },
                PreparedTerm {
                    g1: input.helius_commitment_pok,
                    prepared_g2: 1,
                },
            ],
            &[],
        )
        .expect("fixture terms are in bounds");
    if !pok_ok {
        return (false, k_sum, pok_result);
    }
    let (main_ok, product) = input
        .helius_main_context
        .verify_prevalidated(
            &[
                PreparedTerm {
                    g1: input.helius_proof_c,
                    prepared_g2: 0,
                },
                PreparedTerm {
                    g1: k_sum,
                    prepared_g2: 1,
                },
            ],
            &[LivePair {
                g1: input.helius_proof_a,
                g2: input.helius_proof_b,
            }],
        )
        .expect("fixture terms are in bounds");
    (main_ok, k_sum, product)
}

/// Clone the four prepared fixed-key G2 schedules for the timed Arkworks
/// loop, ordered sigma_neg, key_g, delta_neg, gamma_neg.
pub fn ark_committed_prepared_schedules(
    input: &GnarkCommittedGroth16Fixture,
) -> [ArkG2Prepared; 4] {
    [
        input.ark_commitment_key_g_sigma_neg_prepared.clone(),
        input.ark_commitment_key_g_prepared.clone(),
        input.ark_delta_neg_prepared.clone(),
        input.ark_gamma_neg_prepared.clone(),
    ]
}

/// Arkworks twin of [`helius_committed_verify_accumulated`].
pub fn ark_committed_verify_accumulated(
    input: &GnarkCommittedGroth16Fixture,
    schedules: [ArkG2Prepared; 4],
) -> (bool, ArkG1, ArkFq12) {
    let [sigma_neg, key_g, delta_neg, gamma_neg] = schedules;
    let commitment = input.ark_commitment;
    let k_sum = (input.ark_k[0].into_group()
        + input.ark_k[1] * input.ark_public_input
        + input.ark_k[2] * input.ark_commitment_hash
        + commitment)
        .into_affine();
    let pok_miller = Bn254::multi_miller_loop(
        [
            <Bn254 as Pairing>::G1Prepared::from(commitment),
            input.ark_commitment_pok.into(),
        ],
        [sigma_neg, key_g],
    );
    let pok_result = Bn254::final_exponentiation(pok_miller)
        .expect("a committed Miller product has a final exponentiation");
    if !pok_result.is_zero() {
        return (false, k_sum, pok_result.0);
    }
    let miller = Bn254::multi_miller_loop(
        [
            <Bn254 as Pairing>::G1Prepared::from(input.ark_proof_a),
            input.ark_proof_c.into(),
            k_sum.into(),
        ],
        [input.ark_proof_b.into(), delta_neg, gamma_neg],
    );
    let product = Bn254::final_exponentiation(miller)
        .expect("a committed Miller product has a final exponentiation");
    (product == input.ark_alpha_beta, k_sum, product.0)
}

/// Verify gnark's committed-wires equation through Helius public arithmetic.
pub fn helius_gnark_committed_groth16_verify(input: &GnarkCommittedGroth16Fixture) -> bool {
    let k_sum = input.helius_k[0]
        .to_curve()
        .add_projective(
            input.helius_k[1]
                .to_curve()
                .mul_scalar(input.helius_public_input),
        )
        .add_projective(
            input.helius_k[2]
                .to_curve()
                .mul_scalar(input.helius_commitment_hash),
        )
        .add_mixed(input.helius_commitment)
        .to_affine();
    let pok_result = helius_multi_pairing(&[
        (
            &input.helius_commitment,
            &input.helius_commitment_key_g_sigma_neg,
        ),
        (&input.helius_commitment_pok, &input.helius_commitment_key_g),
    ]);
    if !pok_result.is_one() {
        return false;
    }
    helius_multi_pairing(&[
        (&input.helius_proof_a, &input.helius_proof_b),
        (&input.helius_proof_c, &input.helius_delta_neg),
        (&k_sum, &input.helius_gamma_neg),
    ]) == input.helius_alpha_beta
}

/// Verify gnark's committed-wires equation through Arkworks 0.5 low-level APIs.
pub fn ark_gnark_committed_groth16_verify(input: &GnarkCommittedGroth16Fixture) -> bool {
    let k_sum = (input.ark_k[0].into_group()
        + input.ark_k[1] * input.ark_public_input
        + input.ark_k[2] * input.ark_commitment_hash
        + input.ark_commitment)
        .into_affine();
    if !Bn254::multi_pairing(
        [input.ark_commitment, input.ark_commitment_pok],
        [
            input.ark_commitment_key_g_sigma_neg,
            input.ark_commitment_key_g,
        ],
    )
    .is_zero()
    {
        return false;
    }
    Bn254::multi_pairing(
        [input.ark_proof_a, input.ark_proof_c, k_sum],
        [input.ark_proof_b, input.ark_delta_neg, input.ark_gamma_neg],
    ) == input.ark_alpha_beta
}

/// Verify gnark's committed-wires equation through pinned MCL low-level APIs.
pub fn mcl_gnark_committed_groth16_verify(input: &GnarkCommittedGroth16Fixture) -> bool {
    let mut output = 0_u8;
    // SAFETY: the fixture owns a live context and the hash is 32 bytes.
    let status = unsafe {
        ffi::mcl_gnark_committed_groth16_verify(
            input.mcl.as_ptr(),
            input.commitment_hash_bytes.as_ptr(),
            &mut output,
        )
    };
    assert_eq!(status, 0, "MCL committed Groth16 verification failed");
    output != 0
}

/// Helius batch verification.
///
/// Under one key `sum_i r_i (k0 + x_i k1)` is `(sum_i r_i) k0 + (sum_i r_i
/// x_i) k1`, so the public inputs of a whole key cost two weighted points, not
/// two per member. Every remaining weighting that shares a G2 partner is one
/// multi-scalar multiplication. What stays per member is `r_i A_i`, which
/// pairs against that member's own B.
///
/// The three key-owned G2 replay the schedules the key prepared once. Those
/// replays and every member's live B share one multi-Miller accumulator, so
/// the batch pays one Fp12 square chain and one final exponentiation whatever
/// the key count. A Miller product over a set is the product over any
/// partition of it, so the grouping cannot move the result.
fn helius_batch_product(keys: &[BatchKey], members: &[BatchMember]) -> (G1Affine, Fp12, Fp12) {
    let count = keys.len();
    let mut coefficient_sum = vec![Fr::ZERO; count];
    let mut weighted_input_sum = vec![Fr::ZERO; count];
    let mut c_bases: Vec<Vec<G1Affine>> = vec![Vec::new(); count];
    let mut c_scalars: Vec<Vec<[u64; 4]>> = vec![Vec::new(); count];
    let mut live: Vec<(G1Affine, G2Affine)> = Vec::with_capacity(members.len());
    for member in members {
        let coefficient = member.helius_challenge;
        coefficient_sum[member.key] += coefficient;
        weighted_input_sum[member.key] += coefficient * member.helius_input;
        c_bases[member.key].push(member.helius_c);
        c_scalars[member.key].push(coefficient.to_raw());
        live.push((
            member
                .helius_a
                .to_curve()
                .mul_scalar(coefficient)
                .to_affine(),
            member.helius_b,
        ));
    }
    let mut total = G1Projective::identity();
    let mut key_terms: Vec<[PreparedTerm; 3]> = Vec::with_capacity(keys.len());
    for (index, key) in keys.iter().enumerate() {
        let input_sum = msm_variable_time_affine(
            &key.helius_k,
            &[
                coefficient_sum[index].to_raw(),
                weighted_input_sum[index].to_raw(),
            ],
        );
        total = total.add_mixed(input_sum);
        key_terms.push([
            PreparedTerm {
                g1: input_sum.negate(),
                prepared_g2: BATCH_GAMMA,
            },
            PreparedTerm {
                g1: msm_variable_time_affine(&c_bases[index], &c_scalars[index]).negate(),
                prepared_g2: BATCH_DELTA,
            },
            PreparedTerm {
                g1: key
                    .helius_alpha
                    .to_curve()
                    .mul_scalar(coefficient_sum[index])
                    .to_affine()
                    .negate(),
                prepared_g2: BATCH_BETA,
            },
        ]);
    }
    let validated: Vec<_> = keys
        .iter()
        .zip(&key_terms)
        .map(|(key, terms)| {
            key.helius_prepared
                .validate_online(terms, &[])
                .expect("a batch key owns three prepared schedules")
        })
        .collect();
    let mut terms: Vec<MillerTerm<'_>> = Vec::with_capacity(3 * keys.len() + live.len());
    terms.extend(validated.iter().flat_map(|key| key.terms()));
    terms.extend(live.iter().map(|(g1, g2)| MillerTerm::live(g1, g2)));
    let miller = helius_multi_miller_loop_mixed(&terms);
    let product = if miller.is_one() {
        Fp12::ONE
    } else {
        helius_narsil::pairing::final_exponentiation(&miller)
    };
    (total.to_affine(), miller, product)
}

/// Batch product and the value-bearing accumulator through Helius.
pub fn helius_groth16_batch_verify_digest(
    input: &Groth16BatchFixture,
) -> (bool, G1Affine, Fp12, Fp12) {
    let (accumulator, miller, product) = helius_batch_product(&input.keys, &input.members);
    (product.is_one(), accumulator, miller, product)
}

pub fn helius_groth16_batch_verify(input: &Groth16BatchFixture) -> bool {
    helius_groth16_batch_verify_digest(input).0
}

/// Arkworks twin of [`helius_batch_product`], consuming the already-cloned
/// key schedules. ark-ec 0.5 takes `G2Prepared` by value, so the clones stay
/// outside the timer. Each member's B is prepared inside, it is the live G2.
fn ark_batch_product(
    keys: &[BatchKey],
    members: &[BatchMember],
    prepared: Vec<ArkG2Prepared>,
) -> (ArkG1, ArkFq12, ArkFq12) {
    let count = keys.len();
    assert_eq!(prepared.len(), 3 * count, "one schedule triple per key");
    let mut coefficient_sum = vec![ArkFr::zero(); count];
    let mut weighted_input_sum = vec![ArkFr::zero(); count];
    let mut c_bases: Vec<Vec<ArkG1>> = vec![Vec::new(); count];
    let mut c_scalars: Vec<Vec<ArkFr>> = vec![Vec::new(); count];
    let mut g1: Vec<ArkG1> = Vec::with_capacity(members.len() + 3 * count);
    let mut g2: Vec<ArkG2Prepared> = Vec::with_capacity(members.len() + 3 * count);
    for member in members {
        let coefficient = member.ark_challenge;
        coefficient_sum[member.key] += coefficient;
        weighted_input_sum[member.key] += coefficient * member.ark_input;
        c_bases[member.key].push(member.ark_c);
        c_scalars[member.key].push(coefficient);
        g1.push(
            member
                .ark_a
                .mul_bigint(coefficient.into_bigint())
                .into_affine(),
        );
        g2.push(member.ark_b.into());
    }
    let mut total = ArkG1Projective::zero();
    let mut schedules = prepared.into_iter();
    for (index, key) in keys.iter().enumerate() {
        let input_sum = ArkG1Projective::msm_unchecked(
            &key.ark_k,
            &[coefficient_sum[index], weighted_input_sum[index]],
        );
        total += input_sum;
        g1.push((-input_sum).into_affine());
        g1.push(
            (-ArkG1Projective::msm_unchecked(&c_bases[index], &c_scalars[index])).into_affine(),
        );
        g1.push(
            (-key
                .ark_alpha
                .mul_bigint(coefficient_sum[index].into_bigint()))
            .into_affine(),
        );
        g2.extend(schedules.by_ref().take(3));
    }
    let miller = Bn254::multi_miller_loop(g1.iter().copied(), g2);
    let product = Bn254::final_exponentiation(miller)
        .expect("a batch Miller product has a final exponentiation")
        .0;
    (total.into_affine(), miller.0, product)
}

/// The key-owned schedules one batch replays, cloned outside the timer.
pub fn ark_batch_prepared_schedules(input: &Groth16BatchFixture) -> Vec<ArkG2Prepared> {
    input
        .keys
        .iter()
        .flat_map(|key| key.ark_prepared.clone())
        .collect()
}

/// Batch product and the value-bearing accumulator through Arkworks.
pub fn ark_batch_verify_accumulated(
    input: &Groth16BatchFixture,
    prepared: Vec<ArkG2Prepared>,
) -> (bool, ArkG1, ArkFq12, ArkFq12) {
    let (accumulator, miller, product) = ark_batch_product(&input.keys, &input.members, prepared);
    (product.is_one(), accumulator, miller, product)
}

pub fn ark_groth16_batch_verify_digest(
    input: &Groth16BatchFixture,
) -> (bool, ArkG1, ArkFq12, ArkFq12) {
    ark_batch_verify_accumulated(input, ark_batch_prepared_schedules(input))
}

pub fn ark_groth16_batch_verify(input: &Groth16BatchFixture) -> bool {
    ark_groth16_batch_verify_digest(input).0
}

/// Batch product through MCL.
pub fn mcl_groth16_batch_verify(input: &Groth16BatchFixture) -> bool {
    let mut output = 0_u8;
    // SAFETY: the fixture owns a live context for the duration of the call.
    let status = unsafe { ffi::mcl_groth16_batch_verify(input.mcl.as_ptr(), &mut output) };
    assert_eq!(
        status, 0,
        "mcl Groth16 batch verification failed with {status}"
    );
    output != 0
}

// ---------------------------------------------------------------------------
// Negative passes. Tampered or rejected inputs are never timed.
// ---------------------------------------------------------------------------

/// Require every lane to reject the invalid vectors gnark shipped.
///
/// One moves proof A by a generator step and one moves a public input by one,
/// so both stay decodable and the rejection comes from the pairing equation.
pub fn gnark_fresh_tamper_rejected(pool: &GnarkFreshGroth16Pool) -> Result<(), String> {
    for (index, fixture) in pool.invalid().iter().enumerate() {
        for (lane, accepted) in [
            ("helius", helius_groth16_verify(fixture)),
            ("arkworks", ark_groth16_verify(fixture)),
            ("mcl", mcl_groth16_verify(fixture)),
        ] {
            if accepted {
                return Err(format!("{lane} accepted gnark invalid vector {index}"));
            }
        }
    }
    Ok(())
}

/// Require every lane to reject the invalid vectors the originating verifier
/// rejected, on both production rails.
pub fn production_tamper_rejected(pool: &ProductionGroth16Pool) -> Result<(), String> {
    for (index, fixture) in pool.invalid_standard().iter().enumerate() {
        for (lane, accepted) in [
            ("helius", helius_groth16_verify(fixture)),
            ("arkworks", ark_groth16_verify(fixture)),
            ("mcl", mcl_groth16_verify(fixture)),
        ] {
            if accepted {
                return Err(format!("{lane} accepted production invalid vector {index}"));
            }
        }
    }
    for (index, fixture) in pool.invalid_committed().iter().enumerate() {
        for (lane, accepted) in [
            ("helius", helius_gnark_committed_groth16_verify(fixture)),
            ("arkworks", ark_gnark_committed_groth16_verify(fixture)),
            ("mcl", mcl_gnark_committed_groth16_verify(fixture)),
        ] {
            if accepted {
                return Err(format!(
                    "{lane} accepted production invalid committed vector {index}"
                ));
            }
        }
    }
    Ok(())
}

/// Every way one member of a batch can be wrong while staying on the curve, so
/// the rejection comes from the pairing product and never from decode.
#[derive(Clone, Copy, Debug)]
enum BatchCorruption {
    NegateA,
    NegateC,
    ScaleA,
    ShiftInput,
    StealNeighbourA,
    StealNeighbourB,
}

const BATCH_CORRUPTIONS: [BatchCorruption; 6] = [
    BatchCorruption::NegateA,
    BatchCorruption::NegateC,
    BatchCorruption::ScaleA,
    BatchCorruption::ShiftInput,
    BatchCorruption::StealNeighbourA,
    BatchCorruption::StealNeighbourB,
];

impl BatchCorruption {
    fn apply(self, members: &mut [BatchMember], index: usize) {
        let neighbour = members[(index + 1) % members.len()].clone();
        let member = &mut members[index];
        match self {
            Self::NegateA => member.ark_a = -member.ark_a,
            Self::NegateC => member.ark_c = -member.ark_c,
            Self::ScaleA => {
                member.ark_a = member
                    .ark_a
                    .mul_bigint(ArkFr::from(3_u64).into_bigint())
                    .into_affine();
            }
            Self::ShiftInput => member.ark_input += ArkFr::one(),
            Self::StealNeighbourA => member.ark_a = neighbour.ark_a,
            Self::StealNeighbourB => member.ark_b = neighbour.ark_b,
        }
        sync_batch_member(&mut members[index]);
    }

    /// A borrowing from the only member leaves the batch unchanged.
    fn is_meaningful(self, size: usize) -> bool {
        !matches!(self, Self::StealNeighbourA | Self::StealNeighbourB) || size > 1
    }
}

/// Re-derive a member's Helius values from its Arkworks values, so a
/// corruption reaches every lane as the same point.
fn sync_batch_member(member: &mut BatchMember) {
    member.helius_a = encode_ark_g1(member.ark_a.into_group())
        .to_affine()
        .expect("a corrupted A stays on the curve");
    member.helius_b = encode_ark_g2(member.ark_b.into_group())
        .to_affine()
        .expect("a corrupted B stays on the curve");
    member.helius_c = encode_ark_g1(member.ark_c.into_group())
        .to_affine()
        .expect("a corrupted C stays on the curve");
    member.helius_input = encode_ark_fr(member.ark_input)
        .to_fr()
        .expect("a shifted input stays canonical");
}

/// The member records the bridge decodes, rebuilt from a corrupted member list.
fn encode_batch_members(members: &[BatchMember]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(members.len() * 320);
    for member in members {
        blob.extend_from_slice(&encode_ark_g1(member.ark_a.into_group()).0);
        blob.extend_from_slice(&encode_ark_g2(member.ark_b.into_group()).0);
        blob.extend_from_slice(&encode_ark_g1(member.ark_c.into_group()).0);
        blob.extend_from_slice(&encode_ark_fr(member.ark_input).0);
        blob.extend_from_slice(&encode_ark_fr(member.ark_challenge).0);
    }
    blob
}

/// mcl's verdict on a corrupted member list, through a context built from the
/// same bytes the other lanes hold.
fn mcl_batch_accepts(input: &Groth16BatchFixture, members: &[BatchMember]) -> Result<bool, String> {
    let member_blob = encode_batch_members(members);
    let mut handle = core::ptr::null_mut();
    // SAFETY: the rebuilt blob keeps the documented fixed-width records and
    // the key index of every member is unchanged.
    let status = unsafe {
        ffi::mcl_groth16_batch_create(
            input.key_blob.as_ptr(),
            input.keys.len(),
            member_blob.as_ptr(),
            input.member_keys.as_ptr(),
            members.len(),
            &mut handle,
        )
    };
    if status != 0 {
        return Err(format!("tampered batch mcl context failed with {status}"));
    }
    let mut verdict = 0_u8;
    // SAFETY: handle came from a successful create just above.
    let status = unsafe { ffi::mcl_groth16_batch_verify(handle, &mut verdict) };
    // SAFETY: as above, and the handle is dropped here.
    unsafe { ffi::mcl_groth16_batch_destroy(handle) };
    if status != 0 {
        return Err(format!("tampered batch mcl verify failed with {status}"));
    }
    Ok(verdict != 0)
}

fn batch_must_reject(
    input: &Groth16BatchFixture,
    members: &[BatchMember],
    what: &str,
) -> Result<(), String> {
    if helius_batch_product(&input.keys, members).1.is_one() {
        return Err(format!("helius accepted a batch with {what}"));
    }
    if ark_batch_product(&input.keys, members, ark_batch_prepared_schedules(input))
        .1
        .is_one()
    {
        return Err(format!("Arkworks accepted a batch with {what}"));
    }
    if mcl_batch_accepts(input, members)? {
        return Err(format!("mcl accepted a batch with {what}"));
    }
    Ok(())
}

/// Require every lane to reject one bad member at every index, and to reject a
/// pair placed to cancel inside the combination.
pub fn batch_tamper_rejected(input: &Groth16BatchFixture) -> Result<(), String> {
    let size = input.members.len();
    for index in 0..size {
        for corruption in BATCH_CORRUPTIONS {
            if !corruption.is_meaningful(size) {
                continue;
            }
            let mut members = input.members.clone();
            corruption.apply(&mut members, index);
            batch_must_reject(input, &members, &format!("{corruption:?} at index {index}"))?;
        }
    }
    // A forger who knows the other members still cannot cancel against them.
    // The coefficients weight each member independently, so the cancellation
    // would have to survive a coefficient the forger never saw.
    if size >= 2 {
        let offset = (ArkG1::generator() * ArkFr::from(0x9e37_79b9_u64)).into_affine();
        let mut members = input.members.clone();
        members[0].ark_a = (members[0].ark_a + offset).into_affine();
        members[size - 1].ark_a = (members[size - 1].ark_a - offset).into_affine();
        sync_batch_member(&mut members[0]);
        sync_batch_member(&mut members[size - 1]);
        batch_must_reject(input, &members, "a cancelling pair at the ends")?;
    }
    Ok(())
}

/// Require every lane to reject a batch that carries one gnark-rejected vector,
/// at every index of both batch shapes.
pub fn production_batch_invalid_rejected(
    pool: &ProductionGroth16Pool,
    size: usize,
    mixed: bool,
) -> Result<(), String> {
    let batches = pool.invalid_batches(size, mixed);
    // A silently empty pass would report success without testing anything.
    assert_eq!(
        batches.len(),
        pool.invalid_standard().len() * size,
        "every invalid vector must reach every index"
    );
    for (index, fixture) in batches.iter().enumerate() {
        for (lane, accepted) in [
            ("helius", helius_groth16_batch_verify(fixture)),
            ("arkworks", ark_groth16_batch_verify(fixture)),
            ("mcl", mcl_groth16_batch_verify(fixture)),
        ] {
            if accepted {
                let shape = if mixed { "mixed-key" } else { "same-key" };
                return Err(format!(
                    "{lane} accepted {shape} batch {index} of size {size} carrying a production invalid vector"
                ));
            }
        }
    }
    Ok(())
}

/// Require every lane to reject the G1 encoding with one flipped bit.
pub fn g1_validate_tamper_rejected(fixture: &SubgroupFixture) -> Result<(), String> {
    let mut tampered = fixture.g1_bytes.0;
    tampered[31] ^= 1;
    if G1Bytes(tampered).to_affine().is_ok() {
        return Err("helius accepted a tampered G1 encoding".to_owned());
    }
    if decode_ark_g1(&tampered).is_some() {
        return Err("Arkworks accepted a tampered G1 encoding".to_owned());
    }
    let mut raw = std::ptr::null_mut();
    // SAFETY: tampered holds 64 initialized bytes and raw is a valid
    // out-pointer. A successful call returns an owned allocation.
    let status = unsafe { ffi::mcl_g1_create(tampered.as_ptr(), &mut raw) };
    if status == 0 {
        // SAFETY: raw came from a successful create just above.
        unsafe { ffi::mcl_g1_destroy(raw) };
        return Err("mcl accepted a tampered G1 encoding".to_owned());
    }
    Ok(())
}

/// Require every lane to reject an on-curve point outside the G2 subgroup.
///
/// This exercises the order check itself, not coordinate decode. mcl rejects
/// at typed construction because its decode enforces G2 order.
pub fn g2_subgroup_tamper_rejected() -> Result<(), String> {
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x0ff5_1b67);
    let point = loop {
        let x = Fq2::rand(&mut rng);
        if let Some(point) = ArkG2::get_point_from_x_unchecked(x, false)
            && !point.is_in_correct_subgroup_assuming_on_curve()
        {
            break point;
        }
    };
    if point.is_in_correct_subgroup_assuming_on_curve() {
        return Err("Arkworks accepted an off-subgroup G2 point".to_owned());
    }
    let helius_point = G2Affine {
        x: helius_fp2(point.x),
        y: helius_fp2(point.y),
        infinity: false,
    };
    if !helius_point.is_on_curve() {
        return Err("off-subgroup G2 probe must stay on the curve".to_owned());
    }
    if helius_point.is_in_correct_subgroup_assuming_on_curve() {
        return Err("helius accepted an off-subgroup G2 point".to_owned());
    }
    let encoded = encode_ark_g2(point.into_group());
    let mut raw = std::ptr::null_mut();
    // SAFETY: encoded holds 128 initialized bytes and raw is a valid
    // out-pointer. A successful call returns an owned allocation.
    let status = unsafe { ffi::mcl_g2_create(encoded.0.as_ptr(), &mut raw) };
    if status == 0 {
        // SAFETY: raw came from a successful create just above.
        unsafe { ffi::mcl_g2_destroy(raw) };
        return Err("mcl accepted an off-subgroup G2 point".to_owned());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared digest folds. The C++ bridge mirrors every step bit for bit.
// ---------------------------------------------------------------------------

/// Fold words into the four-lane digest.
///
/// Word folds run over raw Montgomery limbs. All four lanes store field
/// elements as 4x64 little-endian limbs with `R = 2^256`, so equal values
/// fold equally across lanes.
pub fn fold_words(acc: u64, words: impl IntoIterator<Item = u64>) -> u64 {
    words.into_iter().fold(acc, fold_word)
}

/// One digest chain step. The odd multiplier makes every step a bijection,
/// so repeated inputs never cancel the way a plain rotate-xor chain does.
/// The C++ bridge mirrors this step bit for bit.
pub const FOLD_MULTIPLIER: u64 = 0x9e37_79b9_7f4a_7c15;

pub fn fold_word(acc: u64, word: u64) -> u64 {
    (acc.rotate_left(7) ^ word).wrapping_mul(FOLD_MULTIPLIER)
}

/// Steps in the gate E reference chain. Long enough that one call sits in the
/// microsecond range the timed rows live in, short enough to cost nothing.
pub const LANE_REFERENCE_STEPS: u64 = 1_024;

/// Fixed-work reference chain, gate E's anchor.
///
/// Every lane times this one function, so the four columns of a campaign share
/// one code path and one clock. The chain is a dependent 64-bit multiply that
/// touches no field, no curve and no comparator library, so no lane can be
/// faster at it than another and its cost follows the core frequency alone.
/// It exists to tie the four columns to each other, never to compare them.
pub fn lane_reference_chain(seed: u64) -> u64 {
    (0..LANE_REFERENCE_STEPS).fold(seed, fold_word)
}

/// Fold a boolean verdict into the four-lane digest.
pub fn fold_bool(acc: u64, value: bool) -> u64 {
    fold_word(acc, u64::from(value))
}

pub fn fold_helius_fp(acc: u64, value: &Fp) -> u64 {
    fold_words(acc, value.mont_limbs())
}

pub fn fold_helius_fp2(acc: u64, value: &Fp2) -> u64 {
    fold_helius_fp(fold_helius_fp(acc, &value.c0), &value.c1)
}

pub fn fold_helius_fp12(acc: u64, value: &Fp12) -> u64 {
    [&value.c0, &value.c1].into_iter().fold(acc, |acc, half| {
        [&half.c0, &half.c1, &half.c2]
            .into_iter()
            .fold(acc, fold_helius_fp2)
    })
}

pub fn fold_ark_fq(acc: u64, value: &Fq) -> u64 {
    fold_words(acc, value.0.0)
}

pub fn fold_ark_fq2(acc: u64, value: &Fq2) -> u64 {
    fold_ark_fq(fold_ark_fq(acc, &value.c0), &value.c1)
}

pub fn fold_ark_fq12(acc: u64, value: &ArkFq12) -> u64 {
    [&value.c0, &value.c1].into_iter().fold(acc, |acc, half| {
        [&half.c0, &half.c1, &half.c2]
            .into_iter()
            .fold(acc, fold_ark_fq2)
    })
}

// Value-bearing digest folds. Coordinates fold before the verdict, and only
// for a decoded nonidentity point, identically in every lane.

pub fn fold_helius_g1_point(acc: u64, value: &G1Affine) -> u64 {
    if value.is_identity() {
        return acc;
    }
    fold_helius_fp(fold_helius_fp(acc, &value.x), &value.y)
}

pub fn fold_ark_g1_point(acc: u64, value: &ArkG1) -> u64 {
    match value.xy() {
        Some((x, y)) => fold_ark_fq(fold_ark_fq(acc, &x), &y),
        None => acc,
    }
}

pub fn fold_helius_g1_validate(acc: u64, value: &(bool, Option<G1Affine>)) -> u64 {
    let acc = value
        .1
        .as_ref()
        .map_or(acc, |point| fold_helius_g1_point(acc, point));
    fold_bool(acc, value.0)
}

pub fn fold_ark_g1_validate(acc: u64, value: &(bool, Option<ArkG1>)) -> u64 {
    let acc = value
        .1
        .as_ref()
        .map_or(acc, |point| fold_ark_g1_point(acc, point));
    fold_bool(acc, value.0)
}

pub fn fold_helius_g2_check(acc: u64, value: &(bool, G2Affine)) -> u64 {
    let acc = if value.1.is_identity() {
        acc
    } else {
        fold_helius_fp2(fold_helius_fp2(acc, &value.1.x), &value.1.y)
    };
    fold_bool(acc, value.0)
}

pub fn fold_ark_g2_check(acc: u64, value: &(bool, ArkG2)) -> u64 {
    let acc = match value.1.xy() {
        Some((x, y)) => fold_ark_fq2(fold_ark_fq2(acc, &x), &y),
        None => acc,
    };
    fold_bool(acc, value.0)
}

pub fn fold_helius_verdict_fp12(acc: u64, value: &(bool, Fp12)) -> u64 {
    fold_bool(fold_helius_fp12(acc, &value.1), value.0)
}

pub fn fold_ark_verdict_fq12(acc: u64, value: &(bool, ArkFq12)) -> u64 {
    fold_bool(fold_ark_fq12(acc, &value.1), value.0)
}

// The accumulator folds before the target-group product, which folds before
// the verdict. The bridge folds the same three values in the same order.

pub fn fold_helius_accumulated_verdict(acc: u64, value: &(bool, G1Affine, Fp12)) -> u64 {
    fold_bool(
        fold_helius_fp12(fold_helius_g1_point(acc, &value.1), &value.2),
        value.0,
    )
}

pub fn fold_ark_accumulated_verdict(acc: u64, value: &(bool, ArkG1, ArkFq12)) -> u64 {
    fold_bool(
        fold_ark_fq12(fold_ark_g1_point(acc, &value.1), &value.2),
        value.0,
    )
}

// Batch verdict folds. The target-group product of an accepting batch is
// Fp12::ONE whatever the batch held, so the digest binds the Miller product
// that precedes the final exponentiation as well. That value is specific to
// the drawn proofs and the drawn coefficients, and no lane can name it without
// running every pair of the batch.

pub fn fold_helius_batch_verdict(acc: u64, value: &(bool, G1Affine, Fp12, Fp12)) -> u64 {
    fold_bool(
        fold_helius_fp12(
            fold_helius_fp12(fold_helius_g1_point(acc, &value.1), &value.2),
            &value.3,
        ),
        value.0,
    )
}

pub fn fold_ark_batch_verdict(acc: u64, value: &(bool, ArkG1, ArkFq12, ArkFq12)) -> u64 {
    fold_bool(
        fold_ark_fq12(
            fold_ark_fq12(fold_ark_g1_point(acc, &value.1), &value.2),
            &value.3,
        ),
        value.0,
    )
}

// ---------------------------------------------------------------------------
// MCL loop runner.
// ---------------------------------------------------------------------------

/// Probe whether the linked mcl engaged its Xbyak JIT backend.
///
/// The first `isEnableJIT` call is not thread safe, so the probe runs once
/// behind a lock and before the harness spawns any thread.
pub fn mcl_runtime_jit() -> bool {
    static PROBE: OnceLock<bool> = OnceLock::new();
    *PROBE.get_or_init(|| {
        ensure_mcl();
        // SAFETY: the probe takes no pointers and OnceLock invokes it once.
        let status = unsafe { ffi::mcl_runtime_jit() };
        assert!(status >= 0, "mcl JIT probe failed with {status}");
        status == 1
    })
}

/// Operations served by the bridge's four-lane loop runner.
///
/// The discriminants are the bridge's switch cases. They must not move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum MclFourLaneOp {
    FpMul = 0,
    FpSquare = 1,
    Fp2Mul = 2,
    Fp2Square = 3,
    Fp12Mul = 4,
    Fp12Square = 5,
    Fp12Sparse034 = 6,
    G1LineScaling = 7,
    G2Prepare = 8,
    MillerLive = 9,
    MillerPrepared = 10,
    FinalExponentiation = 11,
    FullPairing = 12,
    G1Validate = 13,
    G2Subgroup = 14,
    G1Msm = 15,
    Groth16Single = 16,
    Fp12CyclotomicSquare = 18,
    FullPairingPrepared = 19,
    Groth16Accumulated = 20,
    Groth16Batch = 21,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MclFourLanePoolKind {
    Field,
    Miller,
    G1Point,
    G2Point,
    Msm,
    Groth16,
    Batch,
}

impl MclFourLaneOp {
    fn pool_kind(self) -> MclFourLanePoolKind {
        match self {
            Self::FpMul
            | Self::FpSquare
            | Self::Fp2Mul
            | Self::Fp2Square
            | Self::Fp12Mul
            | Self::Fp12Square
            | Self::Fp12Sparse034
            | Self::Fp12CyclotomicSquare
            | Self::G1LineScaling => MclFourLanePoolKind::Field,
            Self::G2Prepare
            | Self::MillerLive
            | Self::MillerPrepared
            | Self::FinalExponentiation
            | Self::FullPairing
            | Self::FullPairingPrepared => MclFourLanePoolKind::Miller,
            Self::G1Validate => MclFourLanePoolKind::G1Point,
            Self::G2Subgroup => MclFourLanePoolKind::G2Point,
            Self::G1Msm => MclFourLanePoolKind::Msm,
            Self::Groth16Single | Self::Groth16Accumulated => MclFourLanePoolKind::Groth16,
            Self::Groth16Batch => MclFourLanePoolKind::Batch,
        }
    }
}

/// Borrowed handle pool for the bridge's four-lane loop runner.
///
/// The kind tag prevents running an operation against contexts of another
/// C++ type, which would be undefined behavior across the FFI boundary.
pub struct MclFourLanePool<'a> {
    handles: Vec<*mut c_void>,
    kind: MclFourLanePoolKind,
    fixtures: PhantomData<&'a ()>,
}

impl<'a> MclFourLanePool<'a> {
    fn from_handles(handles: Vec<*mut c_void>, kind: MclFourLanePoolKind) -> Self {
        assert!(!handles.is_empty(), "four-lane pool must not be empty");
        Self {
            handles,
            kind,
            fixtures: PhantomData,
        }
    }

    pub fn field(fixtures: &'a [FieldFixture]) -> Self {
        Self::from_handles(
            fixtures.iter().map(|f| f.mcl.as_ptr()).collect(),
            MclFourLanePoolKind::Field,
        )
    }

    pub fn miller(fixtures: &'a [MillerFixture]) -> Self {
        Self::from_handles(
            fixtures.iter().map(|f| f.mcl.as_ptr()).collect(),
            MclFourLanePoolKind::Miller,
        )
    }

    /// The G1 validate pool hands the bridge raw 64-byte encodings, so the
    /// C++ loop decodes per iteration exactly like the Rust lanes.
    pub fn g1_blobs(fixtures: &'a [SubgroupFixture]) -> Self {
        Self::from_handles(
            fixtures
                .iter()
                .map(|f| f.g1_bytes.0.as_ptr() as *mut c_void)
                .collect(),
            MclFourLanePoolKind::G1Point,
        )
    }

    pub fn g2_points(fixtures: &'a [SubgroupFixture]) -> Self {
        Self::from_handles(
            fixtures.iter().map(|f| f.mcl_g2.raw.as_ptr()).collect(),
            MclFourLanePoolKind::G2Point,
        )
    }

    pub fn msm(fixtures: &'a [MsmPubInputsFixture]) -> Self {
        Self::from_handles(
            fixtures.iter().map(|f| f.mcl.as_ptr()).collect(),
            MclFourLanePoolKind::Msm,
        )
    }

    pub fn groth16(fixtures: &'a [Groth16Fixture]) -> Self {
        Self::from_handles(
            fixtures.iter().map(|f| f.mcl.as_ptr()).collect(),
            MclFourLanePoolKind::Groth16,
        )
    }

    pub fn batch(fixtures: &'a [Groth16BatchFixture]) -> Self {
        Self::from_handles(
            fixtures.iter().map(|f| f.mcl.as_ptr()).collect(),
            MclFourLanePoolKind::Batch,
        )
    }

    /// Run the complete iteration loop inside the bridge. One FFI crossing
    /// per loop; the C++ side folds every iteration's output identically to
    /// the Rust lanes.
    pub fn run(&self, operation: MclFourLaneOp, iterations: usize, rotation: usize) -> u64 {
        assert_eq!(
            operation.pool_kind(),
            self.kind,
            "four-lane pool kind must match the operation"
        );
        let mut digest = 0;
        // SAFETY: every handle outlives this call and the kind tag above
        // proves each one has the C++ type this operation casts to.
        let status = unsafe {
            ffi::mcl_four_lane_run(
                self.handles.as_ptr(),
                self.handles.len(),
                operation as u32,
                iterations,
                rotation,
                &mut digest,
            )
        };
        assert_eq!(status, 0, "MCL four-lane loop failed with {status}");
        digest
    }
}

// ---------------------------------------------------------------------------
// Per-operation fixture pools the timing binary rotates over.
// ---------------------------------------------------------------------------

/// One seeded entry of the four-lane field-operation pool, all lanes typed.
///
/// `sparse` holds the `(c0, c3, c4)` line coefficients of `mul_by_034`. The
/// cyclotomic input is a final-exponentiation image, so every correct
/// cyclotomic squaring formula agrees on it.
pub struct FieldFixture {
    pub h_fp: (Fp, Fp),
    pub h_fp2: (Fp2, Fp2),
    pub h_fp12: (Fp12, Fp12),
    pub h_fp12_cyclo: Fp12,
    pub h_sparse: (Fp2, Fp2, Fp2),
    pub h_scale: Fp,
    pub a_fp: (Fq, Fq),
    pub a_fp2: (Fq2, Fq2),
    pub a_fp12: (ArkFq12, ArkFq12),
    pub a_fp12_cyclo: ArkFq12,
    pub a_sparse: (Fq2, Fq2, Fq2),
    pub a_scale: Fq,
    fixture_sha256: String,
    mcl: NonNull<c_void>,
}

impl FieldFixture {
    pub fn from_seed(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let a_fp = (Fq::rand(&mut rng), Fq::rand(&mut rng));
        let a_fp2 = (Fq2::rand(&mut rng), Fq2::rand(&mut rng));
        let a_fp12 = (ArkFq12::rand(&mut rng), ArkFq12::rand(&mut rng));
        let a_sparse = (
            Fq2::rand(&mut rng),
            Fq2::rand(&mut rng),
            Fq2::rand(&mut rng),
        );
        let a_scale = Fq::rand(&mut rng);
        let cyclo_preimage = helius_fp12(ArkFq12::rand(&mut rng));
        let h_fp12_cyclo = helius_narsil::pairing::final_exponentiation(&cyclo_preimage);
        let a_fp12_cyclo = ark_fp12_from_canonical(&encode_helius_fp12(h_fp12_cyclo));

        let h_fp = (helius_fp(a_fp.0), helius_fp(a_fp.1));
        let h_fp2 = (helius_fp2(a_fp2.0), helius_fp2(a_fp2.1));
        let h_fp12 = (helius_fp12(a_fp12.0), helius_fp12(a_fp12.1));
        let h_sparse = (
            helius_fp2(a_sparse.0),
            helius_fp2(a_sparse.1),
            helius_fp2(a_sparse.2),
        );
        let h_scale = helius_fp(a_scale);

        let mut blob = [0_u8; 1568];
        let mut next = 0;
        let mut write = |bytes: &[u8]| {
            blob[next..next + bytes.len()].copy_from_slice(bytes);
            next += bytes.len();
        };
        write(&h_fp.0.to_bytes_be());
        write(&h_fp.1.to_bytes_be());
        write(&encode_helius_fp2(h_fp2.0));
        write(&encode_helius_fp2(h_fp2.1));
        write(&encode_helius_fp12(h_fp12.0));
        write(&encode_helius_fp2(h_sparse.0));
        write(&encode_helius_fp2(h_sparse.1));
        write(&encode_helius_fp2(h_sparse.2));
        write(&h_scale.to_bytes_be());
        write(&encode_helius_fp12(h_fp12.1));
        write(&encode_helius_fp12(h_fp12_cyclo));
        assert_eq!(next, blob.len());

        let mut hasher = FixtureHasher::new("four_lane_field_ops", 1, 1);
        hasher.field(&blob);
        let fixture_sha256 = hasher.finish();

        ensure_mcl();
        let mut handle = core::ptr::null_mut();
        // SAFETY: blob holds the documented 1568-byte extended direct layout.
        let status = unsafe { ffi::mcl_direct_create2(blob.as_ptr(), &mut handle) };
        assert_eq!(status, 0, "MCL direct context v2 failed with {status}");
        Self {
            h_fp,
            h_fp2,
            h_fp12,
            h_fp12_cyclo,
            h_sparse,
            h_scale,
            a_fp,
            a_fp2,
            a_fp12,
            a_fp12_cyclo,
            a_sparse,
            a_scale,
            fixture_sha256,
            mcl: NonNull::new(handle).expect("MCL returned a direct context"),
        }
    }

    pub fn sha256(&self) -> &str {
        &self.fixture_sha256
    }

    fn mcl_check(&self, operation: usize) -> [u8; 384] {
        let mut output = [0_u8; 384];
        // SAFETY: the fixture owns a live context and the buffer holds the
        // widest result the bridge writes.
        let status = unsafe {
            ffi::mcl_direct_check(self.mcl.as_ptr(), operation as u32, output.as_mut_ptr())
        };
        assert_eq!(status, 0, "MCL direct check failed with {status}");
        output
    }

    /// Compare every field operation across the three implementations.
    pub fn check_cross_lane(&self) -> Result<(), String> {
        let compare = |name: &str, helius: &[u8], ark: &[u8], mcl: Option<usize>| {
            if helius != ark {
                return Err(format!("{name}: Helius and Arkworks outputs differ"));
            }
            if let Some(operation) = mcl
                && helius != &self.mcl_check(operation)[..helius.len()]
            {
                return Err(format!("{name}: Helius and MCL outputs differ"));
            }
            Ok(())
        };
        compare(
            "fp_mul",
            &(self.h_fp.0 * self.h_fp.1).to_bytes_be(),
            &field_be(self.a_fp.0 * self.a_fp.1),
            Some(0),
        )?;
        compare(
            "fp_square",
            &self.h_fp.0.square().to_bytes_be(),
            &field_be(self.a_fp.0.square()),
            Some(1),
        )?;
        compare(
            "fp2_mul",
            &encode_helius_fp2(self.h_fp2.0 * self.h_fp2.1),
            &encode_ark_fq2(self.a_fp2.0 * self.a_fp2.1),
            Some(4),
        )?;
        compare(
            "fp2_square",
            &encode_helius_fp2(self.h_fp2.0.square()),
            &encode_ark_fq2(self.a_fp2.0.square()),
            Some(5),
        )?;
        let mut a_scaled = self.a_fp2.0;
        a_scaled.mul_assign_by_fp(&self.a_scale);
        compare(
            "g1_line_scaling",
            &encode_helius_fp2(self.h_fp2.0.mul_by_fp(self.h_scale)),
            &encode_ark_fq2(a_scaled),
            Some(14),
        )?;
        compare(
            "fp12_square",
            &encode_helius_fp12(self.h_fp12.0.square()),
            &encode_ark_fp12(self.a_fp12.0.square()),
            Some(15),
        )?;
        let mut a_sparse = self.a_fp12.0;
        a_sparse.mul_by_034(&self.a_sparse.0, &self.a_sparse.1, &self.a_sparse.2);
        compare(
            "fp12_sparse_034",
            &encode_helius_fp12(self.h_fp12.0.mul_by_034(
                self.h_sparse.0,
                self.h_sparse.1,
                self.h_sparse.2,
            )),
            &encode_ark_fp12(a_sparse),
            Some(16),
        )?;
        compare(
            "fp12_mul",
            &encode_helius_fp12(self.h_fp12.0 * self.h_fp12.1),
            &encode_ark_fp12(self.a_fp12.0 * self.a_fp12.1),
            Some(17),
        )?;
        compare(
            "fp12_cyclotomic_square",
            &encode_helius_fp12(self.h_fp12_cyclo.cyclotomic_square()),
            &encode_ark_fp12(self.a_fp12_cyclo.cyclotomic_square()),
            None,
        )
    }
}

impl Drop for FieldFixture {
    fn drop(&mut self) {
        // SAFETY: this handle is uniquely owned and was returned by create.
        unsafe { ffi::mcl_direct_destroy(self.mcl.as_ptr()) };
    }
}

/// One seeded valid point pair for the subgroup-check operations.
pub struct SubgroupFixture {
    pub helius_g1: G1Affine,
    pub helius_g2: G2Affine,
    pub ark_g1: ArkG1,
    pub ark_g2: ArkG2,
    pub g1_bytes: G1Bytes,
    pub g2_bytes: G2Bytes,
    fixture_sha256: String,
    mcl_g2: MclG2Handle,
}

impl SubgroupFixture {
    pub fn from_seed(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let ark_g1 = ArkG1Projective::rand(&mut rng).into_affine();
        let ark_g2 = ArkG2Projective::rand(&mut rng).into_affine();
        let g1_bytes = encode_ark_g1(ark_g1.into_group());
        let g2_bytes = encode_ark_g2(ark_g2.into_group());
        let helius_g1 = g1_bytes.to_affine().expect("valid Helius subgroup G1");
        let helius_g2 = g2_bytes.to_affine().expect("valid Helius subgroup G2");
        Self {
            helius_g1,
            helius_g2,
            ark_g1,
            ark_g2,
            g1_bytes,
            g2_bytes,
            fixture_sha256: format!(
                "{:x}",
                Sha256::digest([g1_bytes.0.as_slice(), g2_bytes.0.as_slice()].concat())
            ),
            mcl_g2: MclG2Handle::new(&g2_bytes),
        }
    }

    pub fn sha256(&self) -> &str {
        &self.fixture_sha256
    }

    /// BN254 G1 has cofactor one, so the real proof-point workload is byte
    /// validation, canonical range plus on-curve. All lanes decode the same
    /// 64 canonical bytes per call.
    pub fn helius_g1_validate(&self) -> bool {
        self.g1_bytes.to_affine().is_ok()
    }

    /// Decode verdict plus the decoded point, so the digest binds the
    /// coordinates a short-circuiting lane could not reproduce.
    pub fn helius_g1_validate_digest(&self) -> (bool, Option<G1Affine>) {
        match self.g1_bytes.to_affine() {
            Ok(point) => (true, Some(point)),
            Err(_) => (false, None),
        }
    }

    pub fn ark_g1_validate(&self) -> bool {
        decode_ark_g1(&self.g1_bytes.0).is_some()
    }

    pub fn ark_g1_validate_digest(&self) -> (bool, Option<ArkG1>) {
        let point = decode_ark_g1(&self.g1_bytes.0);
        (point.is_some(), point)
    }

    pub fn helius_g2_check(&self) -> bool {
        self.helius_g2.is_in_correct_subgroup_assuming_on_curve()
    }

    /// Subgroup verdict plus the checked point for the value-bearing digest.
    pub fn helius_g2_check_digest(&self) -> (bool, G2Affine) {
        (self.helius_g2_check(), self.helius_g2)
    }

    pub fn ark_g2_check(&self) -> bool {
        self.ark_g2.is_in_correct_subgroup_assuming_on_curve()
    }

    pub fn ark_g2_check_digest(&self) -> (bool, ArkG2) {
        (self.ark_g2_check(), self.ark_g2)
    }

    pub fn mcl_g2_check(&self) -> bool {
        self.mcl_g2.is_valid_order()
    }

    /// The mcl decode verdict through the same bridge path the timed loop
    /// takes. A fresh handle round-trip decodes canonical bytes and checks
    /// the curve equation.
    pub fn mcl_g1_validate(&self) -> bool {
        let mut raw = std::ptr::null_mut();
        // SAFETY: g1_bytes points to 64 initialized bytes and raw is a valid
        // out-pointer. A successful call returns an owned allocation.
        let status = unsafe { ffi::mcl_g1_create(self.g1_bytes.0.as_ptr(), &mut raw) };
        if status != 0 {
            return false;
        }
        // SAFETY: raw came from a successful create just above.
        unsafe { ffi::mcl_g1_destroy(raw) };
        true
    }

    /// Require agreement of all validation verdicts on this valid fixture.
    pub fn check_cross_lane(&self) -> Result<(), String> {
        for (name, verdict) in [
            ("helius g1", self.helius_g1_validate()),
            ("arkworks g1", self.ark_g1_validate()),
            ("mcl g1", self.mcl_g1_validate()),
            ("helius g2", self.helius_g2_check()),
            ("arkworks g2", self.ark_g2_check()),
            ("mcl g2", self.mcl_g2_check()),
        ] {
            if !verdict {
                return Err(format!("{name}: valid point rejected"));
            }
        }
        Ok(())
    }
}

/// One seeded verifier-key public-input MSM (`vk_x` preparation) fixture.
///
/// Every lane consumes typed pre-decoded inputs, so the timed loop contains
/// the MSM arithmetic only, never byte decode or validation.
pub struct MsmPubInputsFixture {
    pub count: usize,
    pub points: Vec<G1Bytes>,
    pub scalars: Vec<ScalarBytes>,
    pub ark_bases: Vec<ArkG1>,
    pub ark_scalars: Vec<ArkFr>,
    helius_bases: Vec<G1Affine>,
    helius_scalar_limbs: Vec<[u64; 4]>,
    fixture_sha256: String,
    mcl: NonNull<c_void>,
}

impl MsmPubInputsFixture {
    pub fn from_seed(count: usize, seed: u64) -> Self {
        assert!((1..=3).contains(&count), "public-input MSM size");
        let mut rng = StdRng::seed_from_u64(seed);
        let ark_bases: Vec<ArkG1> = (0..count)
            .map(|_| ArkG1Projective::rand(&mut rng).into_affine())
            .collect();
        let ark_scalars: Vec<ArkFr> = (0..count).map(|_| ArkFr::rand(&mut rng)).collect();
        let points: Vec<G1Bytes> = ark_bases
            .iter()
            .map(|point| encode_ark_g1(point.into_group()))
            .collect();
        let scalars: Vec<ScalarBytes> = ark_scalars.iter().map(|s| encode_ark_fr(*s)).collect();
        let points_flat = flatten_g1(&points);
        let scalars_flat = flatten_scalars(&scalars);
        let mut hasher = FixtureHasher::new("four_lane_g1_msm_pub_inputs", count, 1);
        hasher.field(&points_flat);
        hasher.field(&scalars_flat);
        let fixture_sha256 = hasher.finish();
        ensure_mcl();
        let mut handle = core::ptr::null_mut();
        // SAFETY: both flat buffers hold `count` fixed-width records.
        let status = unsafe {
            ffi::mcl_msm_create(
                points_flat.as_ptr(),
                scalars_flat.as_ptr(),
                count,
                &mut handle,
            )
        };
        assert_eq!(status, 0, "MCL MSM context failed with {status}");
        let helius_bases = points
            .iter()
            .map(|point| point.to_affine().expect("valid four-lane MSM base"))
            .collect();
        let helius_scalar_limbs = scalars.iter().map(scalar_limbs).collect();
        Self {
            count,
            points,
            scalars,
            ark_bases,
            ark_scalars,
            helius_bases,
            helius_scalar_limbs,
            fixture_sha256,
            mcl: NonNull::new(handle).expect("MCL returned an MSM context"),
        }
    }

    pub fn sha256(&self) -> &str {
        &self.fixture_sha256
    }

    pub fn helius_msm(&self) -> [u8; 64] {
        g1_msm(&self.points, &self.scalars)
            .expect("valid four-lane MSM input")
            .0
    }

    /// Typed helius MSM over the pre-decoded bases and scalar limbs.
    pub fn helius_msm_typed(&self) -> G1Affine {
        msm_variable_time_affine(&self.helius_bases, &self.helius_scalar_limbs)
    }

    pub fn ark_msm(&self) -> [u8; 64] {
        encode_ark_g1(ArkG1Projective::msm_unchecked(
            &self.ark_bases,
            &self.ark_scalars,
        ))
        .0
    }

    /// Typed Arkworks MSM with the affine conversion inside, like every lane.
    pub fn ark_msm_typed(&self) -> ArkG1 {
        ArkG1Projective::msm_unchecked(&self.ark_bases, &self.ark_scalars).into_affine()
    }

    pub fn mcl_msm(&self) -> [u8; 64] {
        let mut output = [0_u8; 64];
        // SAFETY: the fixture owns a live context and the buffer holds one
        // 64-byte G1 encoding.
        let status = unsafe { ffi::mcl_msm_check(self.mcl.as_ptr(), output.as_mut_ptr()) };
        assert_eq!(status, 0, "MCL MSM check failed with {status}");
        output
    }

    /// Compare the MSM result bytes across the three implementations and
    /// pin the typed paths to the byte facade result.
    pub fn check_cross_lane(&self) -> Result<(), String> {
        let expected = self.helius_msm();
        if self.ark_msm() != expected {
            return Err(format!(
                "g1_msm_pub_inputs_{}: Helius and Arkworks outputs differ",
                self.count
            ));
        }
        if self.mcl_msm() != expected {
            return Err(format!(
                "g1_msm_pub_inputs_{}: Helius and MCL outputs differ",
                self.count
            ));
        }
        let typed = self.helius_msm_typed();
        let mut typed_bytes = [0_u8; 64];
        if !typed.is_identity() {
            typed_bytes[..32].copy_from_slice(&typed.x.to_bytes_be());
            typed_bytes[32..].copy_from_slice(&typed.y.to_bytes_be());
        }
        if typed_bytes != expected {
            return Err(format!(
                "g1_msm_pub_inputs_{}: typed Helius MSM differs from the byte facade",
                self.count
            ));
        }
        if encode_ark_g1(self.ark_msm_typed().into_group()).0 != expected {
            return Err(format!(
                "g1_msm_pub_inputs_{}: typed Arkworks MSM differs from the byte facade",
                self.count
            ));
        }
        Ok(())
    }
}

impl Drop for MsmPubInputsFixture {
    fn drop(&mut self) {
        // SAFETY: this handle is uniquely owned and was returned by create.
        unsafe { ffi::mcl_msm_destroy(self.mcl.as_ptr()) };
    }
}

struct MclG2Handle {
    raw: NonNull<c_void>,
}

impl MclG2Handle {
    fn new(encoded: &G2Bytes) -> Self {
        ensure_mcl();
        let mut raw = std::ptr::null_mut();
        // SAFETY: encoded points to 128 initialized bytes and `raw` is a valid
        // out-pointer. A successful bridge call returns an owned allocation.
        let status = unsafe { ffi::mcl_g2_create(encoded.0.as_ptr(), &mut raw) };
        assert_eq!(status, 0, "mcl typed G2 construction failed with {status}");
        Self {
            raw: NonNull::new(raw).expect("successful mcl G2 construction returned null"),
        }
    }

    fn is_valid_order(&self) -> bool {
        let mut output = 0_u8;
        // SAFETY: `raw` is owned by this handle and remains live for the call.
        let status = unsafe { ffi::mcl_g2_is_valid_order(self.raw.as_ptr(), &mut output) };
        assert_eq!(
            status, 0,
            "mcl typed G2 subgroup check failed with {status}"
        );
        output != 0
    }
}

impl Drop for MclG2Handle {
    fn drop(&mut self) {
        // SAFETY: this is the unique owning handle and Drop runs once.
        unsafe { ffi::mcl_g2_destroy(self.raw.as_ptr()) };
    }
}

// ---------------------------------------------------------------------------
// Typed conversions and canonical decode.
// ---------------------------------------------------------------------------

fn scalar_limbs(bytes: &ScalarBytes) -> [u64; 4] {
    let bytes = &bytes.0;
    [
        u64::from_be_bytes(bytes[24..32].try_into().expect("8-byte limb")),
        u64::from_be_bytes(bytes[16..24].try_into().expect("8-byte limb")),
        u64::from_be_bytes(bytes[8..16].try_into().expect("8-byte limb")),
        u64::from_be_bytes(bytes[0..8].try_into().expect("8-byte limb")),
    ]
}

fn helius_fp(value: Fq) -> Fp {
    Fp::from_bytes_be(&field_be(value)).expect("Ark Fq is canonical Helius Fp")
}

fn helius_fp2(value: Fq2) -> Fp2 {
    Fp2::new(helius_fp(value.c0), helius_fp(value.c1))
}

fn helius_fp12(value: ArkFq12) -> Fp12 {
    Fp12::new(
        Fp6::new(
            helius_fp2(value.c0.c0),
            helius_fp2(value.c0.c1),
            helius_fp2(value.c0.c2),
        ),
        Fp6::new(
            helius_fp2(value.c1.c0),
            helius_fp2(value.c1.c1),
            helius_fp2(value.c1.c2),
        ),
    )
}

fn ark_fp12_from_canonical(bytes: &[u8; 384]) -> ArkFq12 {
    let coefficients: Vec<Fq> = bytes
        .chunks_exact(32)
        .map(|chunk| {
            decode_ark_fq(chunk.try_into().expect("32-byte chunk"))
                .expect("canonical Fp12 coefficient")
        })
        .collect();
    ArkFq12::new(
        ArkFq6::new(
            Fq2::new(coefficients[0], coefficients[1]),
            Fq2::new(coefficients[2], coefficients[3]),
            Fq2::new(coefficients[4], coefficients[5]),
        ),
        ArkFq6::new(
            Fq2::new(coefficients[6], coefficients[7]),
            Fq2::new(coefficients[8], coefficients[9]),
            Fq2::new(coefficients[10], coefficients[11]),
        ),
    )
}

fn encode_helius_fp2(value: Fp2) -> [u8; 64] {
    let mut output = [0_u8; 64];
    output[..32].copy_from_slice(&value.c0.to_bytes_be());
    output[32..].copy_from_slice(&value.c1.to_bytes_be());
    output
}

fn encode_ark_fq2(value: Fq2) -> [u8; 64] {
    let mut output = [0_u8; 64];
    output[..32].copy_from_slice(&field_be(value.c0));
    output[32..].copy_from_slice(&field_be(value.c1));
    output
}

/// Big-endian canonical range check. Values at or above the modulus decode to
/// nothing in every lane, so a noncanonical encoding can never enter a pool.
fn canonical(input: &[u8; 32], modulus: &[u8; 32]) -> bool {
    input.as_slice() < modulus.as_slice()
}

fn decode_ark_fq(input: &[u8; 32]) -> Option<Fq> {
    canonical(input, &FP_MODULUS).then(|| Fq::from_be_bytes_mod_order(input))
}

fn decode_ark_fr(input: &[u8; 32]) -> Option<ArkFr> {
    canonical(input, &FR_MODULUS).then(|| ArkFr::from_be_bytes_mod_order(input))
}

fn decode_ark_g1(input: &[u8; 64]) -> Option<ArkG1> {
    if input.iter().all(|byte| *byte == 0) {
        return Some(ArkG1::zero());
    }
    let x = decode_ark_fq(input[..32].try_into().ok()?)?;
    let y = decode_ark_fq(input[32..].try_into().ok()?)?;
    let point = ArkG1::new_unchecked(x, y);
    point.is_on_curve().then_some(point)
}

fn decode_ark_g2(input: &[u8; 128]) -> Option<ArkG2> {
    if input.iter().all(|byte| *byte == 0) {
        return Some(ArkG2::zero());
    }
    let x1 = decode_ark_fq(input[0..32].try_into().ok()?)?;
    let x0 = decode_ark_fq(input[32..64].try_into().ok()?)?;
    let y1 = decode_ark_fq(input[64..96].try_into().ok()?)?;
    let y0 = decode_ark_fq(input[96..128].try_into().ok()?)?;
    let point = ArkG2::new_unchecked(Fq2::new(x0, x1), Fq2::new(y0, y1));
    (point.is_on_curve() && point.is_in_correct_subgroup_assuming_on_curve()).then_some(point)
}

/// The C bridge over the comparator MCL build. Every symbol here is defined in
/// src/mcl_bridge.cpp and shares the mcl_ prefix with the linked archive, so
/// call sites name the module and never shadow a safe lane adapter.
mod ffi {
    use core::ffi::c_void;

    unsafe extern "C" {
        pub(super) fn mcl_init() -> i32;
        pub(super) fn mcl_runtime_jit() -> i32;
        pub(super) fn mcl_g1_create(encoded: *const u8, output: *mut *mut c_void) -> i32;
        pub(super) fn mcl_g1_destroy(handle: *mut c_void);
        pub(super) fn mcl_g2_create(encoded: *const u8, output: *mut *mut c_void) -> i32;
        pub(super) fn mcl_g2_destroy(handle: *mut c_void);
        pub(super) fn mcl_g2_is_valid_order(handle: *const c_void, output: *mut u8) -> i32;
        pub(super) fn mcl_direct_create2(blob: *const u8, output: *mut *mut c_void) -> i32;
        pub(super) fn mcl_direct_destroy(handle: *mut c_void);
        pub(super) fn mcl_direct_check(handle: *mut c_void, operation: u32, output: *mut u8)
        -> i32;
        pub(super) fn mcl_msm_create(
            points: *const u8,
            scalars: *const u8,
            count: usize,
            output: *mut *mut c_void,
        ) -> i32;
        pub(super) fn mcl_msm_destroy(handle: *mut c_void);
        pub(super) fn mcl_msm_check(handle: *mut c_void, output: *mut u8) -> i32;
        pub(super) fn mcl_miller_create(
            g1: *const u8,
            g2: *const u8,
            output: *mut *mut c_void,
        ) -> i32;
        pub(super) fn mcl_miller_destroy(handle: *mut c_void);
        pub(super) fn mcl_miller_raw(handle: *mut c_void, output: *mut u8) -> i32;
        pub(super) fn mcl_miller_final(handle: *mut c_void, output: *mut u8) -> i32;
        pub(super) fn mcl_miller_prepared_final(handle: *mut c_void, output: *mut u8) -> i32;
        pub(super) fn mcl_g2_prepare_replay_raw(handle: *mut c_void, output: *mut u8) -> i32;
        pub(super) fn mcl_prepared_shape_run(
            handle: *mut c_void,
            pairs: usize,
            iterations: usize,
            digest: *mut u64,
        ) -> i32;
        pub(super) fn mcl_groth16_create(
            blob: *const u8,
            gamma_abc_count: usize,
            public_input_count: usize,
            output: *mut *mut c_void,
        ) -> i32;
        pub(super) fn mcl_groth16_destroy(handle: *mut c_void);
        pub(super) fn mcl_groth16_verify(handle: *mut c_void, output: *mut u8) -> i32;
        pub(super) fn mcl_gnark_committed_groth16_create(
            blob: *const u8,
            output: *mut *mut c_void,
        ) -> i32;
        pub(super) fn mcl_gnark_committed_groth16_destroy(handle: *mut c_void);
        pub(super) fn mcl_gnark_committed_groth16_verify(
            handle: *mut c_void,
            commitment_hash: *const u8,
            output: *mut u8,
        ) -> i32;
        pub(super) fn mcl_gnark_committed_pool_run(
            handles: *const *mut c_void,
            hashes: *const u8,
            handle_count: usize,
            iterations: usize,
            rotation: usize,
            digest: *mut u64,
        ) -> i32;
        pub(super) fn mcl_groth16_batch_create(
            keys: *const u8,
            key_count: usize,
            members: *const u8,
            member_keys: *const u32,
            member_count: usize,
            output: *mut *mut c_void,
        ) -> i32;
        pub(super) fn mcl_groth16_batch_destroy(handle: *mut c_void);
        pub(super) fn mcl_groth16_batch_verify(handle: *mut c_void, output: *mut u8) -> i32;
        pub(super) fn mcl_four_lane_run(
            handles: *const *mut c_void,
            handle_count: usize,
            operation: u32,
            iterations: usize,
            rotation: usize,
            digest: *mut u64,
        ) -> i32;
    }
}

fn ensure_mcl() {
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| {
        // SAFETY: initialization takes no pointers and OnceLock invokes it once.
        let status = unsafe { ffi::mcl_init() };
        assert_eq!(status, 0, "mcl BN_SNARK1 initialization failed");
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn four_lane_seed(domain: u64, index: usize) -> u64 {
        SEED ^ domain.rotate_left(17) ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
    }

    #[test]
    fn four_lane_field_fixtures_agree_and_fold_identically() {
        let pool: Vec<FieldFixture> = (0..2)
            .map(|index| FieldFixture::from_seed(four_lane_seed(0x0066_6965_6c64, index)))
            .collect();
        assert_ne!(pool[0].sha256(), pool[1].sha256());
        for fixture in &pool {
            fixture.check_cross_lane().unwrap();
            assert_eq!(
                fold_helius_fp12(7, &fixture.h_fp12.0),
                fold_ark_fq12(7, &fixture.a_fp12.0),
            );
            assert_eq!(
                fold_helius_fp2(7, &fixture.h_fp2.1),
                fold_ark_fq2(7, &fixture.a_fp2.1),
            );
        }
        let mcl = MclFourLanePool::field(&pool);
        let mut fp_expected = 0;
        let mut fp12_expected = 0;
        for counter in 0..6 {
            let fixture = &pool[(counter + 3) % pool.len()];
            fp_expected = fold_helius_fp(fp_expected, &(fixture.h_fp.0 * fixture.h_fp.1));
            fp12_expected = fold_helius_fp12(fp12_expected, &(fixture.h_fp12.0 * fixture.h_fp12.1));
        }
        assert_eq!(mcl.run(MclFourLaneOp::FpMul, 6, 3), fp_expected);
        assert_eq!(mcl.run(MclFourLaneOp::Fp12Mul, 6, 3), fp12_expected);
    }

    #[test]
    fn four_lane_subgroup_fixtures_agree_and_reject_off_subgroup_g2() {
        let pool: Vec<SubgroupFixture> = (0..2)
            .map(|index| SubgroupFixture::from_seed(four_lane_seed(0x7375_6267_7270, index)))
            .collect();
        assert_ne!(pool[0].sha256(), pool[1].sha256());
        for fixture in &pool {
            fixture.check_cross_lane().unwrap();
        }
        let mut g2_expected = 0;
        let mut g1_expected = 0;
        for counter in 0..4 {
            let fixture = &pool[(counter + 1) % pool.len()];
            g2_expected = fold_helius_g2_check(g2_expected, &fixture.helius_g2_check_digest());
            g1_expected =
                fold_helius_g1_validate(g1_expected, &fixture.helius_g1_validate_digest());
        }
        assert_eq!(
            MclFourLanePool::g2_points(&pool).run(MclFourLaneOp::G2Subgroup, 4, 1),
            g2_expected
        );
        assert_eq!(
            MclFourLanePool::g1_blobs(&pool).run(MclFourLaneOp::G1Validate, 4, 1),
            g1_expected
        );
        for fixture in &pool {
            assert_eq!(
                fold_ark_g1_validate(3, &fixture.ark_g1_validate_digest()),
                fold_helius_g1_validate(3, &fixture.helius_g1_validate_digest()),
            );
            assert_eq!(
                fold_ark_g2_check(3, &fixture.ark_g2_check_digest()),
                fold_helius_g2_check(3, &fixture.helius_g2_check_digest()),
            );
        }
        g1_validate_tamper_rejected(&pool[0]).unwrap();
        g2_subgroup_tamper_rejected().unwrap();
    }

    #[test]
    fn four_lane_msm_fixtures_agree_across_sizes() {
        for count in 1..=3 {
            let pool: Vec<MsmPubInputsFixture> = (0..2)
                .map(|index| {
                    MsmPubInputsFixture::from_seed(
                        count,
                        four_lane_seed(0x6d73_6d5f ^ count as u64, index),
                    )
                })
                .collect();
            assert_ne!(pool[0].sha256(), pool[1].sha256());
            for fixture in &pool {
                fixture.check_cross_lane().unwrap();
            }
            let mut expected = 0;
            for counter in 0..3 {
                expected = fold_helius_g1_point(
                    expected,
                    &pool[(counter + 1) % pool.len()].helius_msm_typed(),
                );
            }
            assert_eq!(
                MclFourLanePool::msm(&pool).run(MclFourLaneOp::G1Msm, 3, 1),
                expected
            );
            for fixture in &pool {
                assert_eq!(
                    fold_ark_g1_point(5, &fixture.ark_msm_typed()),
                    fold_helius_g1_point(5, &fixture.helius_msm_typed()),
                );
            }
        }
    }

    #[test]
    fn four_lane_miller_ops_fold_identically_across_lanes() {
        let pool: Vec<MillerFixture> = (0..2)
            .map(|index| MillerFixture::from_seed(four_lane_seed(0x6d69_6c6c_6572, index)))
            .collect();
        let mcl = MclFourLanePool::miller(&pool);
        let mut helius_fold = 0;
        let mut ark_fold = 0;
        for counter in 0..3 {
            let fixture = &pool[counter % pool.len()];
            helius_fold = fold_helius_fp12(helius_fold, &helius_miller(fixture));
            ark_fold = fold_ark_fq12(ark_fold, &ark_miller(fixture).0);
        }
        assert_eq!(helius_fold, ark_fold);
        assert_eq!(mcl.run(MclFourLaneOp::MillerLive, 3, 0), helius_fold);

        let mut final_fold = 0;
        for counter in 0..3 {
            let fixture = &pool[counter % pool.len()];
            final_fold = fold_helius_fp12(final_fold, &helius_final_exponentiation(fixture));
        }
        assert_eq!(
            mcl.run(MclFourLaneOp::FinalExponentiation, 3, 0),
            final_fold
        );
        // The prepare digest is lane local. Require determinism, not
        // cross-lane equality.
        let prepare_digest = mcl.run(MclFourLaneOp::G2Prepare, 3, 0);
        assert_ne!(prepare_digest, 0);
        assert_eq!(mcl.run(MclFourLaneOp::G2Prepare, 3, 0), prepare_digest);
    }

    #[test]
    fn fresh_g2_schedules_replay_alike_and_bind_their_own_point() {
        let pool: Vec<MillerFixture> = (0..2)
            .map(|index| MillerFixture::from_seed(four_lane_seed(0x6d69_6c6c_6572, index)))
            .collect();
        for fixture in &pool {
            g2_prepare_schedules_equivalent(fixture).unwrap();
        }
        // Each lane's replay must depend on the point it prepared, otherwise
        // the gate would hold for a schedule of any other G2.
        let first = g2_prepare_replays(&pool[0]).unwrap();
        let second = g2_prepare_replays(&pool[1]).unwrap();
        for (lane, left, right) in [
            ("live", first.live, second.live),
            ("helius", first.helius, second.helius),
            ("arkworks", first.arkworks, second.arkworks),
            ("mcl", first.mcl, second.mcl),
        ] {
            assert_ne!(left, right, "{lane}");
        }
    }

    #[test]
    fn gnark_fresh_pool_folds_identically_across_lanes_and_rejects_its_invalid_vectors() {
        let pool = GnarkFreshGroth16Pool::deterministic();
        assert_eq!(pool.len(), GNARK_FRESH_POOL_SIZE);
        assert_eq!(pool.invalid().len(), 2);

        let count = 8;
        let start = pool.window_start(count, 3);
        assert_eq!(start, 24);
        let window = |rotation: usize| {
            (0..count)
                .map(|step| {
                    pool.valid()[(step + pool.window_start(count, rotation)) % pool.len()].sha256()
                })
                .collect::<HashSet<_>>()
        };
        assert_eq!(window(3).len(), count);
        assert!(window(3).is_disjoint(&window(4)));

        let mut expected = 0;
        let mut accumulators = HashSet::new();
        for step in 0..count {
            let fixture = &pool.valid()[(step + start) % pool.len()];
            let digest = helius_groth16_verify_accumulated(fixture);
            assert!(digest.0);
            accumulators.insert(fold_helius_g1_point(0, &digest.1));
            let (gamma_neg, delta_neg) = ark_groth16_prepared_schedules(fixture);
            assert_eq!(
                fold_ark_accumulated_verdict(
                    7,
                    &ark_groth16_verify_accumulated(fixture, gamma_neg, delta_neg)
                ),
                fold_helius_accumulated_verdict(7, &digest),
            );
            expected = fold_helius_accumulated_verdict(expected, &digest);
        }
        // One verifying key gives every accepting vector the same target-group
        // value, so only the public-input accumulator separates the statements.
        assert_eq!(accumulators.len(), count);
        assert_eq!(
            pool.valid()[..count]
                .iter()
                .map(|fixture| fold_helius_fp12(0, &helius_groth16_verify_digest(fixture).1))
                .collect::<HashSet<_>>()
                .len(),
            1
        );
        assert_eq!(
            MclFourLanePool::groth16(pool.valid()).run(
                MclFourLaneOp::Groth16Accumulated,
                count,
                start
            ),
            expected
        );
        gnark_fresh_tamper_rejected(&pool).unwrap();
    }

    #[test]
    fn production_pool_folds_identically_across_lanes_and_moves_with_the_statement() {
        let pool = ProductionGroth16Pool::deterministic();
        assert_eq!(pool.standard().len(), PRODUCTION_STANDARD_POOL_SIZE);
        assert_eq!(pool.committed().len(), PRODUCTION_COMMITTED_POOL_SIZE);

        let count = 8;
        let start = pool.window_start(pool.standard().len(), count, 3);
        let mut expected = 0;
        let mut accumulators = HashSet::new();
        for step in 0..count {
            let fixture = &pool.standard()[(step + start) % pool.standard().len()];
            let digest = helius_groth16_verify_accumulated(fixture);
            assert!(digest.0);
            accumulators.insert(fold_helius_g1_point(0, &digest.1));
            let (gamma_neg, delta_neg) = ark_groth16_prepared_schedules(fixture);
            assert_eq!(
                fold_ark_accumulated_verdict(
                    7,
                    &ark_groth16_verify_accumulated(fixture, gamma_neg, delta_neg)
                ),
                fold_helius_accumulated_verdict(7, &digest),
            );
            expected = fold_helius_accumulated_verdict(expected, &digest);
        }
        // The public-input accumulator is the value that separates one
        // statement from another under the same key.
        assert_eq!(accumulators.len(), count);
        assert_eq!(
            MclFourLanePool::groth16(pool.standard()).run(
                MclFourLaneOp::Groth16Accumulated,
                count,
                start
            ),
            expected
        );

        let hashes = pool.committed_hashes();
        let start = pool.window_start(pool.committed().len(), count, 1);
        let mut expected = 0;
        for step in 0..count {
            let fixture = &pool.committed()[(step + start) % pool.committed().len()];
            let digest = helius_committed_verify_accumulated(fixture);
            assert!(digest.0);
            assert_eq!(
                fold_ark_accumulated_verdict(
                    7,
                    &ark_committed_verify_accumulated(
                        fixture,
                        ark_committed_prepared_schedules(fixture)
                    )
                ),
                fold_helius_accumulated_verdict(7, &digest),
            );
            expected = fold_helius_accumulated_verdict(expected, &digest);
        }
        assert_eq!(pool.mcl_committed_run(&hashes, count, start), expected);

        production_tamper_rejected(&pool).unwrap();
    }

    #[test]
    fn production_batches_agree_across_lanes_and_reject_a_negated_member() {
        let pool = ProductionGroth16Pool::deterministic();
        for batches in [
            pool.same_key_batches(
                PRODUCTION_BATCH_SIZE,
                production_batch_pool_size(PRODUCTION_BATCH_SIZE),
            ),
            pool.different_key_batches(
                PRODUCTION_BATCH_SIZE,
                production_batch_pool_size(PRODUCTION_BATCH_SIZE),
            ),
        ] {
            assert_eq!(
                batches.len(),
                production_batch_pool_size(PRODUCTION_BATCH_SIZE)
            );
            let mut expected = 0;
            for fixture in &batches {
                assert_eq!(fixture.proof_count(), PRODUCTION_BATCH_SIZE);
                let digest = helius_groth16_batch_verify_digest(fixture);
                assert!(digest.0);
                assert_eq!(
                    fold_ark_batch_verdict(7, &ark_groth16_batch_verify_digest(fixture)),
                    fold_helius_batch_verdict(7, &digest),
                );
                expected = fold_helius_batch_verdict(expected, &digest);
                batch_tamper_rejected(fixture).unwrap();
            }
            assert_eq!(
                MclFourLanePool::batch(&batches).run(MclFourLaneOp::Groth16Batch, batches.len(), 0),
                expected
            );
        }
        // A mixed batch spans every key, so it cannot collapse into the
        // single-key aggregation the same-key row measures.
        assert_eq!(
            pool.different_key_batches(PRODUCTION_BATCH_SIZE, 1)[0].key_count(),
            2
        );
        assert_eq!(
            pool.same_key_batches(PRODUCTION_BATCH_SIZE, 1)[0].key_count(),
            1
        );
        // Two batches that hold the same proofs are the same fixture. The
        // uniqueness assertion over these digests says nothing unless the
        // batch counter is unable to separate them.
        assert_eq!(
            pool.different_key_batches(PRODUCTION_BATCH_SIZE, 1)[0].sha256(),
            pool.different_key_batches(
                PRODUCTION_BATCH_SIZE,
                production_batch_pool_size(PRODUCTION_BATCH_SIZE)
            )[0]
            .sha256()
        );
    }

    const DRAW_TEST_SEED: u64 = 0x5eed_0001;

    #[test]
    fn a_seeded_draw_order_is_a_reproducible_permutation() {
        for len in [1_usize, 2, 8, 16, 64] {
            let identity: Vec<usize> = (0..len).collect();
            assert_eq!(seeded_order(0, GNARK_FRESH_DRAW_DOMAIN, len), identity);
            let order = seeded_order(DRAW_TEST_SEED, GNARK_FRESH_DRAW_DOMAIN, len);
            assert_eq!(
                order,
                seeded_order(DRAW_TEST_SEED, GNARK_FRESH_DRAW_DOMAIN, len)
            );
            assert_eq!(order.iter().copied().collect::<HashSet<_>>().len(), len);
        }
        let order = seeded_order(DRAW_TEST_SEED, GNARK_FRESH_DRAW_DOMAIN, 64);
        assert_ne!(order, (0..64).collect::<Vec<_>>());
        assert_ne!(
            order,
            seeded_order(DRAW_TEST_SEED + 1, GNARK_FRESH_DRAW_DOMAIN, 64)
        );
        assert_ne!(
            order,
            seeded_order(DRAW_TEST_SEED, PRODUCTION_STANDARD_DRAW_DOMAIN, 64)
        );
    }

    /// Two sessions must not verify the same statements in the same order, or
    /// a warm result or one lucky vector can carry a headline number.
    #[test]
    fn a_run_seed_redraws_the_gnark_row() {
        let frozen = GnarkFreshGroth16Pool::deterministic();
        let seeded = GnarkFreshGroth16Pool::from_seed(DRAW_TEST_SEED);
        assert_ne!(frozen.order_sha256(), seeded.order_sha256());

        // The pool applies the pure order and nothing else, so the seed on the
        // provenance line replays the session.
        let order = seeded_order(DRAW_TEST_SEED, GNARK_FRESH_DRAW_DOMAIN, frozen.len());
        for (position, index) in order.iter().enumerate() {
            assert_eq!(
                seeded.valid()[position].sha256(),
                frozen.valid()[*index].sha256()
            );
        }

        // One round draws a different set, not only a different order.
        let count = 8;
        let start = frozen.window_start(count, 1);
        let window = |pool: &GnarkFreshGroth16Pool| {
            (0..count)
                .map(|step| {
                    pool.valid()[(step + start) % pool.len()]
                        .sha256()
                        .to_owned()
                })
                .collect::<HashSet<String>>()
        };
        assert_ne!(window(&frozen), window(&seeded));

        let mut expected = 0;
        for step in 0..count {
            let fixture = &seeded.valid()[(step + start) % seeded.len()];
            let digest = helius_groth16_verify_digest(fixture);
            assert!(digest.0);
            let (gamma_neg, delta_neg) = ark_groth16_prepared_schedules(fixture);
            assert_eq!(
                fold_ark_verdict_fq12(7, &ark_groth16_verify_digest(fixture, gamma_neg, delta_neg)),
                fold_helius_verdict_fp12(7, &digest),
            );
            expected = fold_helius_verdict_fp12(expected, &digest);
        }
        assert_eq!(
            MclFourLanePool::groth16(seeded.valid()).run(
                MclFourLaneOp::Groth16Single,
                count,
                start
            ),
            expected
        );
    }

    #[test]
    fn a_run_seed_redraws_the_production_rows_and_their_batches() {
        let frozen = ProductionGroth16Pool::deterministic();
        let seeded = ProductionGroth16Pool::from_seed(DRAW_TEST_SEED);
        assert_ne!(frozen.order_sha256(), seeded.order_sha256());

        let order = seeded_order(
            DRAW_TEST_SEED,
            PRODUCTION_STANDARD_DRAW_DOMAIN,
            frozen.standard().len(),
        );
        for (position, index) in order.iter().enumerate() {
            assert_eq!(
                seeded.standard()[position].sha256(),
                frozen.standard()[*index].sha256()
            );
        }
        let order = seeded_order(
            DRAW_TEST_SEED,
            PRODUCTION_COMMITTED_DRAW_DOMAIN,
            frozen.committed().len(),
        );
        for (position, index) in order.iter().enumerate() {
            assert_eq!(
                seeded.committed()[position].sha256(),
                frozen.committed()[*index].sha256()
            );
        }

        let count = 8;
        let start = seeded.window_start(seeded.standard().len(), count, 1);
        let mut expected = 0;
        for step in 0..count {
            let fixture = &seeded.standard()[(step + start) % seeded.standard().len()];
            let digest = helius_groth16_verify_accumulated(fixture);
            assert!(digest.0);
            let (gamma_neg, delta_neg) = ark_groth16_prepared_schedules(fixture);
            assert_eq!(
                fold_ark_accumulated_verdict(
                    7,
                    &ark_groth16_verify_accumulated(fixture, gamma_neg, delta_neg)
                ),
                fold_helius_accumulated_verdict(7, &digest),
            );
            expected = fold_helius_accumulated_verdict(expected, &digest);
        }
        assert_eq!(
            MclFourLanePool::groth16(seeded.standard()).run(
                MclFourLaneOp::Groth16Accumulated,
                count,
                start
            ),
            expected
        );

        let hashes = seeded.committed_hashes();
        let start = seeded.window_start(seeded.committed().len(), count, 1);
        let mut expected = 0;
        for step in 0..count {
            let fixture = &seeded.committed()[(step + start) % seeded.committed().len()];
            let digest = helius_committed_verify_accumulated(fixture);
            assert!(digest.0);
            assert_eq!(
                fold_ark_accumulated_verdict(
                    7,
                    &ark_committed_verify_accumulated(
                        fixture,
                        ark_committed_prepared_schedules(fixture)
                    )
                ),
                fold_helius_accumulated_verdict(7, &digest),
            );
            expected = fold_helius_accumulated_verdict(expected, &digest);
        }
        assert_eq!(seeded.mcl_committed_run(&hashes, count, start), expected);

        // Membership follows the draw. Each row keeps the key structure it is
        // named for, and no batch repeats a member or another batch.
        let identities = |batches: &[Groth16BatchFixture]| {
            batches
                .iter()
                .map(|batch| batch.sha256().to_owned())
                .collect::<Vec<String>>()
        };
        for (keys, frozen_batches, seeded_batches) in [
            (
                1,
                frozen.same_key_batches(
                    PRODUCTION_BATCH_SIZE,
                    production_batch_pool_size(PRODUCTION_BATCH_SIZE),
                ),
                seeded.same_key_batches(
                    PRODUCTION_BATCH_SIZE,
                    production_batch_pool_size(PRODUCTION_BATCH_SIZE),
                ),
            ),
            (
                2,
                frozen.different_key_batches(
                    PRODUCTION_BATCH_SIZE,
                    production_batch_pool_size(PRODUCTION_BATCH_SIZE),
                ),
                seeded.different_key_batches(
                    PRODUCTION_BATCH_SIZE,
                    production_batch_pool_size(PRODUCTION_BATCH_SIZE),
                ),
            ),
        ] {
            assert_ne!(identities(&frozen_batches), identities(&seeded_batches));
            assert_eq!(
                identities(&seeded_batches)
                    .iter()
                    .collect::<HashSet<_>>()
                    .len(),
                production_batch_pool_size(PRODUCTION_BATCH_SIZE)
            );
            for batch in &seeded_batches {
                assert_eq!(batch.key_count(), keys);
                assert_eq!(batch.proof_count(), PRODUCTION_BATCH_SIZE);
                assert!(helius_groth16_batch_verify(batch));
                assert!(ark_groth16_batch_verify(batch));
                assert!(mcl_groth16_batch_verify(batch));
            }
        }
    }

    #[test]
    fn rotation_windows_move_with_the_round() {
        // The early rounds of a campaign still start one window apart.
        assert_eq!(rotation_window_start(64, 8, 3), 24);
        assert_eq!(rotation_window_start(16, 8, 1), 8);
        assert_eq!(rotation_window_start(8, 4, 1), 4);
        // A window that covers the pool leaves only the phase to move.
        for pool in [8_usize, 16] {
            let starts: HashSet<usize> = (0..pool)
                .map(|rotation| rotation_window_start(pool, pool, rotation))
                .collect();
            assert_eq!(starts.len(), pool);
        }
    }

    /// Every round of a campaign must draw a window no earlier round drew.
    /// A repeat lets a prepared schedule or an MSM cache carry state from one
    /// round into the next, and the timed digest cannot tell the two apart.
    #[test]
    fn a_campaign_never_repeats_a_rotation_window() {
        // The batch rows draw 4 of 8 and the production row 16 of 64. Both used
        // to fold back onto an earlier round.
        for (pool, iterations) in [
            (8_usize, 4_usize),
            (64, 16),
            (16, 8),
            (8, 16),
            (64, 10),
            (64, 64),
            (16, 3),
        ] {
            let capacity = rotation_window_capacity(pool, iterations);
            let starts: Vec<usize> = (0..capacity)
                .map(|rotation| rotation_window_start(pool, iterations, rotation))
                .collect();
            assert_eq!(
                starts.iter().copied().collect::<HashSet<_>>().len(),
                capacity,
                "{pool}/{iterations}"
            );
            assert!(starts.iter().all(|start| *start < pool));

            // The block-aligned rounds come first, so the opening rounds of a
            // campaign draw fixtures no other opening round drew.
            let span = iterations.min(pool);
            let blocks = capacity / span;
            let drawn = |start: usize| {
                (0..span)
                    .map(|step| (step + start) % pool)
                    .collect::<HashSet<_>>()
            };
            for first in 0..blocks {
                for second in (first + 1)..blocks {
                    assert!(
                        drawn(starts[first]).is_disjoint(&drawn(starts[second])),
                        "{pool}/{iterations} rounds {first} {second}"
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "no fresh window")]
    fn a_pool_too_small_for_the_round_stops_the_run() {
        let capacity = rotation_window_capacity(8, 4);
        assert_eq!(capacity, 8);
        rotation_window_start(8, 4, capacity);
    }
}
