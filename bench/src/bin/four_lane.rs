//! Four-lane timing runner over one stdout protocol.
//!
//! `four_lane LANE OPERATION ITERATIONS ROTATION` times one operation in one
//! lane and prints one provenance line and one result line. Deterministic
//! operations fold every iteration's output into a digest that is
//! bit-identical across lanes, so the caller can cross-check runs.
//!
//! Exit codes: 0 ok, 2 usage, 3 build-regime gate failure, 4
//! cross-implementation mismatch, 5 timer self-test failure, 6 unsupported
//! (operation, lane) cell.

use std::{
    collections::{HashMap, HashSet},
    hint::black_box,
    process::exit,
    time::{Duration, Instant},
};

use helius_narsil::pairing::prepare_g2_unchecked;
use helius_narsil_bench::{
    ArkG2Prepared, FIELD_POOL_DOMAIN, FIELD_POOL_SIZE, FieldFixture, GnarkFreshGroth16Pool,
    Groth16BatchFixture, MILLER_POOL_DOMAIN, MILLER_POOL_SIZE, MSM_POOL_DOMAIN, MSM_POOL_SIZE,
    MclFourLaneOp, MclFourLanePool, MillerFixture, MsmPubInputsFixture, PRODUCTION_BATCH_SIZE,
    PRODUCTION_SMALL_BATCH_SIZE, ProductionGroth16Pool, SUBGROUP_POOL_DOMAIN, SUBGROUP_POOL_SIZE,
    SubgroupFixture, ark_batch_prepared_schedules, ark_batch_verify_accumulated,
    ark_committed_prepared_schedules, ark_committed_verify_accumulated, ark_final_exponentiation,
    ark_full_pairing, ark_full_pairing_prepared_replay, ark_groth16_batch_verify,
    ark_groth16_prepared_schedules, ark_groth16_verify_accumulated, ark_miller,
    ark_miller_prepared_replay, ark_miller_prepared_setup, batch_tamper_rejected,
    fold_ark_accumulated_verdict, fold_ark_batch_verdict, fold_ark_fq, fold_ark_fq2, fold_ark_fq12,
    fold_ark_g1_point, fold_ark_g1_validate, fold_ark_g2_check, fold_helius_accumulated_verdict,
    fold_helius_batch_verdict, fold_helius_fp, fold_helius_fp2, fold_helius_fp12,
    fold_helius_g1_point, fold_helius_g1_validate, fold_helius_g2_check, fold_word, fold_words,
    g1_validate_tamper_rejected, g2_prepare_schedules_equivalent, g2_subgroup_tamper_rejected,
    gnark_fresh_tamper_rejected, helius_committed_verify_accumulated, helius_final_exponentiation,
    helius_full_pairing, helius_full_pairing_prepared, helius_groth16_batch_verify,
    helius_groth16_batch_verify_digest, helius_groth16_verify_accumulated, helius_miller,
    helius_miller_prepared, lane_reference_chain, mcl_groth16_batch_verify, mcl_runtime_jit,
    production_batch_invalid_rejected, production_batch_pool_size, production_tamper_rejected,
    rotation_draws_distinct_sets, rotation_window_capacity, rotation_window_start, seeded_pool,
};
use sha2::{Digest, Sha256};

// Upstream e107c70e plus two commits that expose mcl::bn::cyclotomicSqr and
// mcl::bn::mulSparseLine. bench/scripts/mcl-bench-patches.patch reproduces them.
const MCL_REVISION: &str = "b70288469cd065b28a31764450f85238508c4d1e";
const MCL_MANIFEST_SCHEMA: &str = "helius-bn254-mcl-native-build-v2";
/// Run-scoped fixture seed. A claim campaign draws one value per session and
/// passes it to every invocation, so no session measures the points of an
/// earlier session and no cached result can stand in for a fresh one. Unset,
/// or `0`, keeps the frozen default pools.
const FIXTURE_SEED_VAR: &str = "FOUR_LANE_FIXTURE_SEED";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane {
    Helius,
    Arkworks,
    ArkworksPortable,
    Mcl,
}

impl Lane {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "helius" => Some(Self::Helius),
            "arkworks" => Some(Self::Arkworks),
            "arkworks-portable" => Some(Self::ArkworksPortable),
            "mcl" => Some(Self::Mcl),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Helius => "helius",
            Self::Arkworks => "arkworks",
            Self::ArkworksPortable => "arkworks-portable",
            Self::Mcl => "mcl",
        }
    }
}

/// Which implementation executes. Both arkworks lanes run the same code;
/// the build-regime gate decides which lane name this binary may serve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Impl {
    Helius,
    Ark,
    Mcl,
}

impl Lane {
    fn implementation(self) -> Impl {
        match self {
            Self::Helius => Impl::Helius,
            Self::Arkworks | Self::ArkworksPortable => Impl::Ark,
            Self::Mcl => Impl::Mcl,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    FpMul,
    FpSquare,
    Fp2Mul,
    Fp2Square,
    Fp12Mul,
    Fp12Square,
    Fp12Sparse034,
    Fp12CyclotomicSquare,
    G1LineScaling,
    G2Prepare87Lines,
    G1MsmPubInputs1,
    G1MsmPubInputs2,
    G1MsmPubInputs3,
    G1Validate,
    G2SubgroupCheck,
    MillerLive,
    MillerPrepared,
    FinalExponentiation,
    FullPairing,
    FullPairingPrepared,
    Groth16Gnark,
    Groth16OneInput,
    Groth16CommittedRails,
    Groth16Batch8OneKey,
    Groth16Batch8MixedKeys,
    Groth16Batch3OneKey,
    Groth16Batch3MixedKeys,
    LaneReference,
}

const OPERATIONS: [(Op, &str); 28] = [
    (Op::FpMul, "fp_mul"),
    (Op::FpSquare, "fp_square"),
    (Op::Fp2Mul, "fp2_mul"),
    (Op::Fp2Square, "fp2_square"),
    (Op::Fp12Mul, "fp12_mul"),
    (Op::Fp12Square, "fp12_square"),
    (Op::Fp12Sparse034, "fp12_sparse_034"),
    (Op::Fp12CyclotomicSquare, "fp12_cyclotomic_square"),
    (Op::G1LineScaling, "g1_line_scaling"),
    (Op::G2Prepare87Lines, "g2_prepare_87_lines"),
    (Op::G1MsmPubInputs1, "g1_msm_pub_inputs_1"),
    (Op::G1MsmPubInputs2, "g1_msm_pub_inputs_2"),
    (Op::G1MsmPubInputs3, "g1_msm_pub_inputs_3"),
    (Op::G1Validate, "g1_validate"),
    (Op::G2SubgroupCheck, "g2_subgroup_check"),
    (Op::MillerLive, "miller_live"),
    (Op::MillerPrepared, "miller_prepared"),
    (Op::FinalExponentiation, "final_exponentiation"),
    (Op::FullPairing, "full_pairing"),
    (Op::FullPairingPrepared, "full_pairing_prepared"),
    (Op::Groth16Gnark, "groth16_gnark"),
    (Op::Groth16OneInput, "groth16_one_input"),
    (Op::Groth16CommittedRails, "groth16_committed_rails"),
    (Op::Groth16Batch8OneKey, "groth16_batch8_one_key"),
    (Op::Groth16Batch8MixedKeys, "groth16_batch8_mixed_keys"),
    (Op::Groth16Batch3OneKey, "groth16_batch3_one_key"),
    (Op::Groth16Batch3MixedKeys, "groth16_batch3_mixed_keys"),
    (Op::LaneReference, "lane_reference"),
];

impl Op {
    fn parse(value: &str) -> Option<Self> {
        OPERATIONS
            .iter()
            .find(|(_, name)| *name == value)
            .map(|(op, _)| *op)
    }

    /// The batch size and key spread a batch row measures. `None` for a row
    /// that is not a batch.
    fn batch_shape(self) -> Option<(usize, bool)> {
        match self {
            Self::Groth16Batch8OneKey => Some((PRODUCTION_BATCH_SIZE, false)),
            Self::Groth16Batch8MixedKeys => Some((PRODUCTION_BATCH_SIZE, true)),
            Self::Groth16Batch3OneKey => Some((PRODUCTION_SMALL_BATCH_SIZE, false)),
            Self::Groth16Batch3MixedKeys => Some((PRODUCTION_SMALL_BATCH_SIZE, true)),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        OPERATIONS
            .iter()
            .find(|(op, _)| *op == self)
            .map(|(_, name)| *name)
            .expect("every operation is listed")
    }

    fn msm_count(self) -> Option<usize> {
        match self {
            Self::G1MsmPubInputs1 => Some(1),
            Self::G1MsmPubInputs2 => Some(2),
            Self::G1MsmPubInputs3 => Some(3),
            _ => None,
        }
    }

    /// Line schedules are lane-local layouts. The digest binds each lane's
    /// own schedule bits and must never be compared across lanes.
    fn lane_local_digest(self) -> bool {
        matches!(self, Self::G2Prepare87Lines)
    }
}

fn main() {
    let (lane, op, iterations, rotation) = parse_arguments();
    let fixture_seed = read_fixture_seed();

    let gate_inputs = GateInputs {
        lane,
        x86: cfg!(target_arch = "x86_64"),
        target_cpu: env!("FOUR_LANE_BUILD_TARGET_CPU"),
        cxx_native_flag: env!("FOUR_LANE_BUILD_CXX_NATIVE_FLAG"),
        ark_asm: build_ark_asm(),
        force_portable: cfg!(feature = "force-portable"),
        helius_backend: helius_narsil::backend_report_for_harness(),
        mcl_jit: mcl_runtime_jit(),
        manifest: read_manifest(),
        linked_manifest_sha256: env!("FOUR_LANE_BUILD_MCL_MANIFEST_SHA256"),
    };
    let failures = gate_failures(&gate_inputs);
    if !failures.is_empty() {
        for failure in &failures {
            eprintln!("gate: {failure}");
        }
        exit(3);
    }

    let timer = match timer_self_test() {
        Ok(report) => report,
        Err(reason) => {
            eprintln!("timer self-test: {reason}");
            exit(5);
        }
    };

    let pools = Pools::for_op(op, fixture_seed);

    if let Err(reason) = check_equivalence(op, &pools) {
        eprintln!("equivalence: {reason}");
        exit(4);
    }

    let warmups = (iterations / 32).clamp(4, 128);
    if let Err(reason) = check_rotation_window(op, &pools, iterations, rotation) {
        eprintln!("rotation: {reason}");
        exit(4);
    }

    print_provenance(
        &gate_inputs,
        &timer,
        fixture_seed,
        &pools.fixture_set_sha256(),
    );

    let implementation = lane.implementation();
    let length = pool_len(op, &pools);
    let start = rotation_window_start(length, iterations, rotation);
    // The warmup takes the entries directly behind the timed window, so it
    // preloads no fixture the timed loop then measures. That holds only while
    // the two windows fit in the pool together. A row whose round already
    // walks the whole pool warms on the fixtures it is about to time, and
    // nothing can change that.
    let warmup_start = (start + iterations.min(length)) % length;
    let (warmup_elapsed, warmup_digest) =
        run_lane(op, implementation, &pools, warmups, warmup_start);
    black_box(warmup_digest);
    black_box(warmup_elapsed);

    let (elapsed, digest) = run_lane(op, implementation, &pools, iterations, start);
    println!(
        "result|{}|{}|{}|{}|{:016x}",
        op.name(),
        lane.name(),
        iterations,
        elapsed.as_nanos(),
        digest
    );
}

fn parse_arguments() -> (Lane, Op, usize, usize) {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.len() != 4 {
        usage();
    }
    let Some(lane) = Lane::parse(&arguments[0]) else {
        usage()
    };
    let Some(op) = Op::parse(&arguments[1]) else {
        usage()
    };
    let Some(iterations) = arguments[2].parse().ok().filter(|value| *value >= 1) else {
        usage()
    };
    let Ok(rotation) = arguments[3].parse() else {
        usage()
    };
    (lane, op, iterations, rotation)
}

fn usage() -> ! {
    eprintln!("usage: four_lane LANE OPERATION ITERATIONS ROTATION");
    eprintln!("  LANE: helius | arkworks | arkworks-portable | mcl");
    eprintln!("  OPERATION: one of {} names", OPERATIONS.len());
    eprintln!("  ITERATIONS >= 1, ROTATION >= 0");
    eprintln!("  {FIXTURE_SEED_VAR}: optional u64, decimal or 0x hex");
    exit(2);
}

/// Read the run-scoped fixture seed. A malformed value aborts, because a
/// silently ignored seed would let one campaign mix two fixture sets.
fn read_fixture_seed() -> u64 {
    let Some(raw) = std::env::var_os(FIXTURE_SEED_VAR) else {
        return 0;
    };
    match raw.to_str().and_then(parse_fixture_seed) {
        Some(seed) => seed,
        None => {
            eprintln!("{FIXTURE_SEED_VAR} must be a u64, decimal or 0x hex");
            exit(2);
        }
    }
}

fn parse_fixture_seed(value: &str) -> Option<u64> {
    let value = value.trim();
    match value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => value.parse().ok(),
    }
}

fn build_ark_asm() -> bool {
    env!("FOUR_LANE_BUILD_ARK_ASM") == "1"
}

// ---------------------------------------------------------------------------
// Build-regime gates.
// ---------------------------------------------------------------------------

struct ManifestFacts {
    disk_sha256: String,
    revision: String,
    schema: String,
    native_flag: String,
}

struct GateInputs {
    lane: Lane,
    x86: bool,
    target_cpu: &'static str,
    cxx_native_flag: &'static str,
    ark_asm: bool,
    force_portable: bool,
    helius_backend: &'static str,
    mcl_jit: bool,
    manifest: Result<ManifestFacts, String>,
    linked_manifest_sha256: &'static str,
}

fn read_manifest() -> Result<ManifestFacts, String> {
    let path = env!("FOUR_LANE_BUILD_MCL_MANIFEST_PATH");
    let bytes = std::fs::read(path).map_err(|error| format!("cannot read {path}: {error}"))?;
    let disk_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let document: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid MCL manifest JSON: {error}"))?;
    let field = |name: &str| -> Result<String, String> {
        document
            .get(name)
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| format!("MCL manifest misses string field {name}"))
    };
    Ok(ManifestFacts {
        disk_sha256,
        revision: field("revision")?,
        schema: field("schema")?,
        native_flag: field("native_flag")?,
    })
}

/// Pure gate predicate. An empty result admits the run.
fn gate_failures(inputs: &GateInputs) -> Vec<String> {
    let mut failures = Vec::new();
    if inputs.lane == Lane::Arkworks && !inputs.ark_asm {
        failures.push("lane arkworks requires the ark-asm build regime".to_owned());
    }
    if inputs.lane == Lane::ArkworksPortable && inputs.ark_asm {
        failures.push("lane arkworks-portable requires the non-ark-asm build regime".to_owned());
    }
    if !inputs.x86 {
        return failures;
    }
    if inputs.target_cpu != "native" {
        failures.push(format!(
            "build target-cpu must be native, got {}",
            inputs.target_cpu
        ));
    }
    if inputs.cxx_native_flag == "NONE" {
        failures.push("the C++ bridge was built without a native flag".to_owned());
    }
    match &inputs.manifest {
        Err(reason) => failures.push(format!("MCL manifest unavailable: {reason}")),
        Ok(manifest) => {
            if manifest.disk_sha256 != inputs.linked_manifest_sha256 {
                failures.push(format!(
                    "MCL manifest changed after the build: disk={} linked={}",
                    manifest.disk_sha256, inputs.linked_manifest_sha256
                ));
            }
            if manifest.revision != MCL_REVISION {
                failures.push(format!("MCL revision must be {MCL_REVISION}"));
            }
            if manifest.schema != MCL_MANIFEST_SCHEMA {
                failures.push(format!("MCL manifest schema must be {MCL_MANIFEST_SCHEMA}"));
            }
            if manifest.native_flag != "-march=native" {
                failures.push("MCL native flag must be -march=native".to_owned());
            }
        }
    }
    if inputs.lane == Lane::Helius
        && !inputs.force_portable
        && inputs.helius_backend != "avx512-ifma"
    {
        failures.push(format!(
            "lane helius requires the avx512-ifma backend, got {}",
            inputs.helius_backend
        ));
    }
    if inputs.lane == Lane::Mcl && !inputs.mcl_jit {
        failures.push("lane mcl requires the Xbyak JIT backend on x86-64".to_owned());
    }
    failures
}

// ---------------------------------------------------------------------------
// Timer self-test.
// ---------------------------------------------------------------------------

struct TimerReport {
    overhead_ns_per_iter: f64,
    tsc_agreement_pct: f64,
}

#[inline(never)]
fn empty_rotated_loop(iterations: usize, rotation: usize, pool_len: usize) -> u64 {
    let mut digest = 0_u64;
    for counter in 0..iterations {
        let index = black_box((counter + rotation) % pool_len);
        digest = fold_word(digest, black_box(index as u64));
    }
    digest
}

fn measure_overhead_ns_per_iter() -> f64 {
    let iterations = 1 << 22;
    black_box(empty_rotated_loop(iterations / 16, 3, FIELD_POOL_SIZE));
    let started = Instant::now();
    black_box(empty_rotated_loop(iterations, 3, FIELD_POOL_SIZE));
    started.elapsed().as_secs_f64() * 1e9 / iterations as f64
}

#[cfg(target_arch = "x86_64")]
fn tsc_agreement_pct() -> Result<f64, String> {
    fn spin_for(duration: Duration) -> (Duration, u64) {
        let started = Instant::now();
        // SAFETY: RDTSC has no preconditions on x86-64.
        let first = unsafe { core::arch::x86_64::_rdtsc() };
        while started.elapsed() < duration {
            std::hint::spin_loop();
        }
        // SAFETY: as above.
        let last = unsafe { core::arch::x86_64::_rdtsc() };
        (started.elapsed(), last.saturating_sub(first))
    }

    let (calibration_elapsed, calibration_cycles) = spin_for(Duration::from_millis(100));
    if calibration_cycles == 0 {
        return Err("TSC did not advance during calibration".to_owned());
    }
    let cycles_per_second = calibration_cycles as f64 / calibration_elapsed.as_secs_f64();
    let (elapsed, cycles) = spin_for(Duration::from_millis(50));
    let tsc_seconds = cycles as f64 / cycles_per_second;
    let instant_seconds = elapsed.as_secs_f64();
    let agreement = 100.0 * (1.0 - ((tsc_seconds - instant_seconds).abs() / instant_seconds));
    if agreement < 99.0 {
        return Err(format!(
            "monotonic clock and TSC disagree: agreement {agreement:.3}%"
        ));
    }
    Ok(agreement)
}

#[cfg(not(target_arch = "x86_64"))]
fn tsc_agreement_pct() -> Result<f64, String> {
    Ok(0.0)
}

fn timer_self_test() -> Result<TimerReport, String> {
    let tsc_agreement_pct = tsc_agreement_pct()?;
    let overhead_ns_per_iter = measure_overhead_ns_per_iter();
    if !overhead_ns_per_iter.is_finite() || overhead_ns_per_iter <= 0.0 {
        return Err(format!(
            "implausible loop overhead {overhead_ns_per_iter} ns"
        ));
    }
    Ok(TimerReport {
        overhead_ns_per_iter,
        tsc_agreement_pct,
    })
}

// ---------------------------------------------------------------------------
// Fixture pools.
// ---------------------------------------------------------------------------

struct Pools {
    field: Option<Vec<FieldFixture>>,
    miller: Option<Vec<MillerFixture>>,
    subgroup: Option<Vec<SubgroupFixture>>,
    msm: Option<Vec<MsmPubInputsFixture>>,
    gnark: Option<GnarkFreshGroth16Pool>,
    production: Option<ProductionGroth16Pool>,
    committed_hashes: Vec<u8>,
    batches: Option<Vec<Groth16BatchFixture>>,
}

fn assert_unique<'a>(values: impl Iterator<Item = &'a str>, expected: usize) {
    assert_eq!(
        values.collect::<HashSet<_>>().len(),
        expected,
        "fixture pool digests must be unique"
    );
}

impl Pools {
    /// The seeded pools regenerate their fixtures from the run seed. The gnark
    /// and production pool files are hash-checked external artifacts whose proof
    /// bytes never move, so there the run seed sets which vectors the session
    /// draws and in what order. A pool of genuinely new proofs needs the
    /// generator rerun under a new seed, and the production pool additionally
    /// needs the originating prover's proving keys, which no repository here
    /// holds.
    fn for_op(op: Op, fixture_seed: u64) -> Self {
        let mut pools = Self {
            field: None,
            miller: None,
            subgroup: None,
            msm: None,
            gnark: None,
            production: None,
            committed_hashes: Vec::new(),
            batches: None,
        };
        match op {
            Op::FpMul
            | Op::FpSquare
            | Op::Fp2Mul
            | Op::Fp2Square
            | Op::Fp12Mul
            | Op::Fp12Square
            | Op::Fp12Sparse034
            | Op::Fp12CyclotomicSquare
            | Op::G1LineScaling
            | Op::LaneReference => {
                let pool = seeded_pool(
                    fixture_seed,
                    FIELD_POOL_DOMAIN,
                    FIELD_POOL_SIZE,
                    FieldFixture::from_seed,
                );
                assert_unique(pool.iter().map(FieldFixture::sha256), FIELD_POOL_SIZE);
                pools.field = Some(pool);
            }
            Op::G2Prepare87Lines
            | Op::MillerLive
            | Op::MillerPrepared
            | Op::FinalExponentiation
            | Op::FullPairing
            | Op::FullPairingPrepared => {
                let pool = seeded_pool(
                    fixture_seed,
                    MILLER_POOL_DOMAIN,
                    MILLER_POOL_SIZE,
                    MillerFixture::from_seed,
                );
                assert_unique(pool.iter().map(MillerFixture::sha256), MILLER_POOL_SIZE);
                pools.miller = Some(pool);
            }
            Op::G1MsmPubInputs1 | Op::G1MsmPubInputs2 | Op::G1MsmPubInputs3 => {
                let count = op.msm_count().expect("MSM operations carry a count");
                let pool = seeded_pool(
                    fixture_seed,
                    MSM_POOL_DOMAIN ^ count as u64,
                    MSM_POOL_SIZE,
                    |seed| MsmPubInputsFixture::from_seed(count, seed),
                );
                assert_unique(pool.iter().map(MsmPubInputsFixture::sha256), MSM_POOL_SIZE);
                pools.msm = Some(pool);
            }
            Op::G1Validate | Op::G2SubgroupCheck => {
                let pool = seeded_pool(
                    fixture_seed,
                    SUBGROUP_POOL_DOMAIN,
                    SUBGROUP_POOL_SIZE,
                    SubgroupFixture::from_seed,
                );
                assert_unique(pool.iter().map(SubgroupFixture::sha256), SUBGROUP_POOL_SIZE);
                pools.subgroup = Some(pool);
            }
            Op::Groth16Gnark => {
                pools.gnark = Some(GnarkFreshGroth16Pool::from_seed(fixture_seed));
            }
            Op::Groth16OneInput => {
                pools.production = Some(ProductionGroth16Pool::from_seed(fixture_seed));
            }
            Op::Groth16CommittedRails => {
                let pool = ProductionGroth16Pool::from_seed(fixture_seed);
                pools.committed_hashes = pool.committed_hashes();
                pools.production = Some(pool);
            }
            Op::Groth16Batch8OneKey
            | Op::Groth16Batch8MixedKeys
            | Op::Groth16Batch3OneKey
            | Op::Groth16Batch3MixedKeys => {
                let (size, mixed) = op.batch_shape().expect("a batch operation has a shape");
                let pool = ProductionGroth16Pool::from_seed(fixture_seed);
                let count = production_batch_pool_size(size);
                let batches = if mixed {
                    pool.different_key_batches(size, count)
                } else {
                    pool.same_key_batches(size, count)
                };
                assert_unique(batches.iter().map(Groth16BatchFixture::sha256), count);
                pools.production = Some(pool);
                pools.batches = Some(batches);
            }
        }
        pools
    }

    fn fixture_set_sha256(&self) -> String {
        let mut digests: Vec<String> = Vec::new();
        if let Some(pool) = &self.field {
            digests.extend(pool.iter().map(|f| f.sha256().to_owned()));
        }
        if let Some(pool) = &self.miller {
            digests.extend(pool.iter().map(|f| f.sha256().to_owned()));
        }
        if let Some(pool) = &self.subgroup {
            digests.extend(pool.iter().map(|f| f.sha256().to_owned()));
        }
        if let Some(pool) = &self.msm {
            digests.extend(pool.iter().map(|f| f.sha256().to_owned()));
        }
        if let Some(pool) = &self.gnark {
            digests.push(pool.order_sha256());
        }
        if let Some(pool) = &self.production {
            digests.push(pool.order_sha256());
        }
        if let Some(pool) = &self.batches {
            digests.extend(pool.iter().map(|f| f.sha256().to_owned()));
        }
        digests.sort_unstable();
        format!("{:x}", Sha256::digest(digests.concat().as_bytes()))
    }
}

// ---------------------------------------------------------------------------
// Cross-implementation equivalence, before any timing.
// ---------------------------------------------------------------------------

fn check_equivalence(op: Op, pools: &Pools) -> Result<(), String> {
    if let Some(pool) = &pools.field {
        for fixture in pool {
            fixture.check_cross_lane()?;
        }
    }
    if let Some(pool) = &pools.subgroup {
        for fixture in pool {
            fixture.check_cross_lane()?;
        }
    }
    if let Some(pool) = &pools.msm {
        for fixture in pool {
            fixture.check_cross_lane()?;
        }
    }

    check_prepared_schedules(op, pools)?;
    check_accepts(op, pools)?;
    check_tamper_rejection(op, pools)?;

    // One untimed pass per implementation over the exact timed semantics.
    // Identical folds prove that the lanes compute the same outputs. Lane-
    // local schedule digests are exempt because they bind lane-own layouts,
    // and check_prepared_schedules carries their cross-lane statement instead.
    let passes = pool_len(op, pools);
    let (_, helius_digest) = run_lane(op, Impl::Helius, pools, passes, 0);
    let (_, ark_digest) = run_lane(op, Impl::Ark, pools, passes, 0);
    let (_, mcl_digest) = run_lane(op, Impl::Mcl, pools, passes, 0);
    if op.lane_local_digest() {
        return Ok(());
    }
    if helius_digest != ark_digest {
        return Err(format!(
            "{}: Helius and Arkworks digests differ over one pool pass",
            op.name()
        ));
    }
    if mcl_digest != helius_digest {
        return Err(format!(
            "{}: Helius and MCL digests differ over one pool pass",
            op.name()
        ));
    }
    Ok(())
}

/// The g2_prepare row folds a lane-local layout, so its digest never crosses
/// lanes. Instead each lane replays its own freshly prepared schedule and all
/// three replays must equal the live Miller product bit for bit, which is what
/// makes the timed row a comparison of the same artifact.
fn check_prepared_schedules(op: Op, pools: &Pools) -> Result<(), String> {
    if op != Op::G2Prepare87Lines {
        return Ok(());
    }
    let pool = pools.miller.as_deref().expect("miller pool built");
    for (index, fixture) in pool.iter().enumerate() {
        g2_prepare_schedules_equivalent(fixture)
            .map_err(|reason| format!("g2_prepare_87_lines fixture {index}: {reason}"))?;
    }
    Ok(())
}

/// Digest agreement alone cannot separate three accepting lanes from three
/// rejecting lanes, so every verifying operation proves the accept verdict
/// on each fixture before any timing.
fn check_accepts(op: Op, pools: &Pools) -> Result<(), String> {
    match op {
        Op::Groth16Batch8OneKey
        | Op::Groth16Batch8MixedKeys
        | Op::Groth16Batch3OneKey
        | Op::Groth16Batch3MixedKeys => {
            let pool = pools.batches.as_deref().expect("batch pool built");
            for (index, fixture) in pool.iter().enumerate() {
                for (lane, accepted) in [
                    ("helius", helius_groth16_batch_verify(fixture)),
                    ("arkworks", ark_groth16_batch_verify(fixture)),
                    ("mcl", mcl_groth16_batch_verify(fixture)),
                ] {
                    if !accepted {
                        return Err(format!("{}: {lane} rejected batch {index}", op.name()));
                    }
                }
            }
        }
        // G1Validate and G2SubgroupCheck accepts run in check_cross_lane. The
        // gnark and production pool constructors admit a vector only when all
        // three lanes reproduce the originating verifier's own verdict.
        _ => {}
    }
    Ok(())
}

/// Non-timed proof that this round draws a window no earlier round drew.
///
/// Every row folds the identities of the fixtures one round draws, in draw
/// order. Round `r` must differ from every round before it, not only from its
/// neighbour. A campaign that revisits a window measures one draw twice and
/// reports the two rounds as independent, and the timed digest cannot tell.
///
/// Order freshness is all a row can have when one round already draws the
/// whole pool. A row whose round draws a proper subset gets the stronger
/// property, a fixture set no earlier round drew, and this checks that
/// separately so the two are never confused.
fn check_rotation_window(
    op: Op,
    pools: &Pools,
    iterations: usize,
    rotation: usize,
) -> Result<(), String> {
    let pool_len = pool_len(op, pools);
    let capacity = rotation_window_capacity(pool_len, iterations);
    if rotation >= capacity {
        return Err(format!(
            "{}: round {rotation} has no fresh window, a pool of {pool_len} serves {capacity} windows at {iterations} iterations",
            op.name()
        ));
    }

    if let Some((earlier, round)) = first_repeated_round(rotation + 1, |round| {
        window_identity(op, pools, iterations, round)
    }) {
        return Err(format!(
            "{}: rounds {earlier} and {round} draw the same window",
            op.name()
        ));
    }

    if !rotation_draws_distinct_sets(pool_len, iterations) {
        return Ok(());
    }
    match first_repeated_round(rotation + 1, |round| {
        window_set_identity(op, pools, iterations, round)
    }) {
        Some((earlier, round)) => Err(format!(
            "{}: rounds {earlier} and {round} draw the same fixture set",
            op.name()
        )),
        None => Ok(()),
    }
}

/// The earliest pair of rounds of a session that fold to one identity.
fn first_repeated_round(rounds: usize, identity: impl Fn(usize) -> u64) -> Option<(usize, usize)> {
    let mut seen: HashMap<u64, usize> = HashMap::with_capacity(rounds);
    (0..rounds).find_map(|round| {
        seen.insert(identity(round), round)
            .map(|first| (first, round))
    })
}

/// The set a round draws, folded so the draw order cannot move it.
///
/// `window_identity` binds the order and therefore calls a pure rotation
/// fresh. This binds only which fixtures were drawn, which is what separates a
/// genuinely new set from the same set walked from another start.
fn window_set_identity(op: Op, pools: &Pools, iterations: usize, rotation: usize) -> u64 {
    let identities = fixture_identities(op, pools);
    let start = rotation_window_start(identities.len(), iterations, rotation);
    let span = iterations.min(identities.len());
    let mut drawn: Vec<usize> = draw(identities.len(), span, start).collect();
    drawn.sort_unstable();
    drawn.dedup();
    drawn.into_iter().fold(0_u64, |acc, index| {
        fold_words(acc, identities[index].bytes().map(u64::from))
    })
}

fn window_identity(op: Op, pools: &Pools, iterations: usize, rotation: usize) -> u64 {
    let identities = fixture_identities(op, pools);
    let start = rotation_window_start(identities.len(), iterations, rotation);
    // A longer round only repeats the pool, so one pass over the drawn entries
    // carries the whole window. The fold binds the draw position, and its
    // multiplier step makes it order sensitive, so a window that only rotates
    // its order still moves.
    let span = iterations.min(identities.len());
    draw(identities.len(), span, start)
        .enumerate()
        .fold(0_u64, |acc, (position, index)| {
            fold_words(
                acc,
                std::iter::once(position as u64).chain(identities[index].bytes().map(u64::from)),
            )
        })
}

/// The pinned digest of every fixture the row can draw, in pool order.
fn fixture_identities(op: Op, pools: &Pools) -> Vec<&str> {
    match op {
        Op::FpMul
        | Op::FpSquare
        | Op::Fp2Mul
        | Op::Fp2Square
        | Op::Fp12Mul
        | Op::Fp12Square
        | Op::Fp12Sparse034
        | Op::Fp12CyclotomicSquare
        | Op::G1LineScaling
        | Op::LaneReference => pools
            .field
            .as_deref()
            .expect("field pool built for this op")
            .iter()
            .map(FieldFixture::sha256)
            .collect(),
        Op::G2Prepare87Lines
        | Op::MillerLive
        | Op::MillerPrepared
        | Op::FinalExponentiation
        | Op::FullPairing
        | Op::FullPairingPrepared => pools
            .miller
            .as_deref()
            .expect("miller pool built for this op")
            .iter()
            .map(MillerFixture::sha256)
            .collect(),
        Op::G1MsmPubInputs1 | Op::G1MsmPubInputs2 | Op::G1MsmPubInputs3 => pools
            .msm
            .as_deref()
            .expect("msm pool built for this op")
            .iter()
            .map(MsmPubInputsFixture::sha256)
            .collect(),
        Op::G1Validate | Op::G2SubgroupCheck => pools
            .subgroup
            .as_deref()
            .expect("subgroup pool built for this op")
            .iter()
            .map(SubgroupFixture::sha256)
            .collect(),
        Op::Groth16Gnark => pools
            .gnark
            .as_ref()
            .expect("gnark pool built for this op")
            .valid()
            .iter()
            .map(|fixture| fixture.sha256())
            .collect(),
        Op::Groth16OneInput => pools
            .production
            .as_ref()
            .expect("production pool built for this op")
            .standard()
            .iter()
            .map(|fixture| fixture.sha256())
            .collect(),
        Op::Groth16CommittedRails => pools
            .production
            .as_ref()
            .expect("production pool built for this op")
            .committed()
            .iter()
            .map(|fixture| fixture.sha256())
            .collect(),
        Op::Groth16Batch8OneKey
        | Op::Groth16Batch8MixedKeys
        | Op::Groth16Batch3OneKey
        | Op::Groth16Batch3MixedKeys => pools
            .batches
            .as_deref()
            .expect("batch pool built for this op")
            .iter()
            .map(Groth16BatchFixture::sha256)
            .collect(),
    }
}

fn draw(pool_len: usize, iterations: usize, start: usize) -> impl Iterator<Item = usize> {
    (0..iterations).map(move |counter| (counter + start) % pool_len)
}

/// Negative pass over one tampered instance per operation class. Tampered
/// inputs are never timed.
fn check_tamper_rejection(op: Op, pools: &Pools) -> Result<(), String> {
    match op {
        Op::G1Validate => {
            let pool = pools.subgroup.as_deref().expect("subgroup pool built");
            g1_validate_tamper_rejected(&pool[0])
        }
        Op::G2SubgroupCheck => g2_subgroup_tamper_rejected(),
        Op::Groth16Gnark => {
            let pool = pools.gnark.as_ref().expect("gnark pool built");
            gnark_fresh_tamper_rejected(pool)
        }
        Op::Groth16OneInput | Op::Groth16CommittedRails => {
            let pool = pools.production.as_ref().expect("production pool built");
            production_tamper_rejected(pool)
        }
        Op::Groth16Batch8OneKey
        | Op::Groth16Batch8MixedKeys
        | Op::Groth16Batch3OneKey
        | Op::Groth16Batch3MixedKeys => {
            let (size, mixed) = op.batch_shape().expect("a batch operation has a shape");
            let batches = pools.batches.as_deref().expect("batch pool built");
            batch_tamper_rejected(&batches[0])?;
            let pool = pools.production.as_ref().expect("production pool built");
            production_batch_invalid_rejected(pool, size, mixed)
        }
        _ => Ok(()),
    }
}

fn pool_len(op: Op, pools: &Pools) -> usize {
    match op {
        Op::FpMul
        | Op::FpSquare
        | Op::Fp2Mul
        | Op::Fp2Square
        | Op::Fp12Mul
        | Op::Fp12Square
        | Op::Fp12Sparse034
        | Op::Fp12CyclotomicSquare
        | Op::G1LineScaling
        | Op::LaneReference => pools.field.as_ref().map_or(0, Vec::len),
        Op::G2Prepare87Lines
        | Op::MillerLive
        | Op::MillerPrepared
        | Op::FinalExponentiation
        | Op::FullPairing
        | Op::FullPairingPrepared => pools.miller.as_ref().map_or(0, Vec::len),
        Op::G1MsmPubInputs1 | Op::G1MsmPubInputs2 | Op::G1MsmPubInputs3 => {
            pools.msm.as_ref().map_or(0, Vec::len)
        }
        Op::G1Validate | Op::G2SubgroupCheck => pools.subgroup.as_ref().map_or(0, Vec::len),
        Op::Groth16Gnark => pools.gnark.as_ref().map_or(0, GnarkFreshGroth16Pool::len),
        Op::Groth16OneInput => pools
            .production
            .as_ref()
            .map_or(0, |pool| pool.standard().len()),
        Op::Groth16CommittedRails => pools
            .production
            .as_ref()
            .map_or(0, |pool| pool.committed().len()),
        Op::Groth16Batch8OneKey
        | Op::Groth16Batch8MixedKeys
        | Op::Groth16Batch3OneKey
        | Op::Groth16Batch3MixedKeys => pools.batches.as_ref().map_or(0, Vec::len),
    }
}

// ---------------------------------------------------------------------------
// Timed execution.
// ---------------------------------------------------------------------------

/// One timed rotated loop. Generic so each (operation, lane) call site
/// monomorphizes into its own `#[inline(never)]` symbol for clean `perf`
/// attribution. The fold runs every iteration, identically to the bridge.
#[inline(never)]
fn run_rotated<T, R>(
    pool: &[T],
    iterations: usize,
    rotation: usize,
    mut operation: impl FnMut(&T) -> R,
    mut fold: impl FnMut(u64, &R) -> u64,
) -> (Duration, u64) {
    let mut digest = 0_u64;
    let started = Instant::now();
    for counter in 0..iterations {
        let input = black_box(&pool[(counter + rotation) % pool.len()]);
        let output = black_box(operation(input));
        digest = fold(digest, &output);
    }
    (started.elapsed(), digest)
}

/// Prepared BN254 G2 schedules held live at once by [`run_ark_replay`].
///
/// One schedule is 87 lines of three Fq2, about 16 KiB, so this bounds the
/// replay buffer near 8 MiB whatever the iteration count and whatever a row
/// prepares per iteration. Every campaign row draws fewer iterations than one
/// chunk, so the bound costs nothing there.
const ARK_REPLAY_LIVE_SCHEDULES: usize = 512;

/// One timed prepared-schedule replay loop for the Arkworks lane.
///
/// ark-ec 0.5 has no borrowed prepared entry. Every `multi_miller_loop`
/// argument is `Into<G2Prepared>` and `Bn::ell` is private, so one replay needs
/// one owned schedule. Every schedule clone and every container deallocation
/// therefore stays outside the timer, which is everything the harness controls.
/// `multi_miller_loop` still frees the schedule buffer inside its own call.
/// The protocol document states that residual.
///
/// `per_iteration` is how many schedules one replay consumes. The loop builds
/// them a chunk at a time and adds the chunks' timed spans, so peak memory
/// does not grow with the iteration count. The chunk boundary sits outside
/// every timed span and cannot move a digest.
#[inline(never)]
fn run_ark_replay<S, R>(
    pool_len: usize,
    iterations: usize,
    rotation: usize,
    per_iteration: usize,
    build: impl Fn(usize) -> S,
    mut replay: impl FnMut(usize, S) -> R,
    mut fold: impl FnMut(u64, &R) -> u64,
) -> (Duration, u64) {
    let chunk = (ARK_REPLAY_LIVE_SCHEDULES / per_iteration.max(1)).max(1);
    let mut digest = 0_u64;
    let mut elapsed = Duration::ZERO;
    let mut done = 0;
    while done < iterations {
        let span = chunk.min(iterations - done);
        let mut schedules: Vec<S> = (done..done + span)
            .map(|counter| build((counter + rotation) % pool_len))
            .collect();
        let started = Instant::now();
        for (offset, schedule) in schedules.drain(..).enumerate() {
            let index = black_box((done + offset + rotation) % pool_len);
            let output = black_box(replay(index, schedule));
            digest = fold(digest, &output);
        }
        elapsed += started.elapsed();
        drop(schedules);
        done += span;
    }
    (elapsed, digest)
}

fn run_mcl_pool(
    pool: MclFourLanePool<'_>,
    op: MclFourLaneOp,
    iterations: usize,
    rotation: usize,
) -> (Duration, u64) {
    let started = Instant::now();
    let digest = pool.run(op, iterations, rotation);
    (started.elapsed(), digest)
}

/// One timed pass over the draw window that begins at `start`.
///
/// The caller derives `start` from the round index, so every row of a session
/// walks the pool through the one traversal `check_rotation_window` audits.
fn run_lane(
    op: Op,
    implementation: Impl,
    pools: &Pools,
    iterations: usize,
    start: usize,
) -> (Duration, u64) {
    // Pools are built for the requested operation before any run.
    let field = || {
        pools
            .field
            .as_deref()
            .expect("field pool built for this op")
    };
    let miller = || {
        pools
            .miller
            .as_deref()
            .expect("miller pool built for this op")
    };
    let subgroup = || {
        pools
            .subgroup
            .as_deref()
            .expect("subgroup pool built for this op")
    };
    let msm = || pools.msm.as_deref().expect("msm pool built for this op");
    let gnark = || pools.gnark.as_ref().expect("gnark pool built for this op");
    let production = || {
        pools
            .production
            .as_ref()
            .expect("production pool built for this op")
    };
    let batches = || {
        pools
            .batches
            .as_deref()
            .expect("batch pool built for this op")
    };

    match (op, implementation) {
        (Op::FpMul, Impl::Helius) => run_rotated(
            field(),
            iterations,
            start,
            |f| f.h_fp.0 * f.h_fp.1,
            fold_helius_fp,
        ),
        (Op::FpMul, Impl::Ark) => run_rotated(
            field(),
            iterations,
            start,
            |f| f.a_fp.0 * f.a_fp.1,
            fold_ark_fq,
        ),
        (Op::FpMul, Impl::Mcl) => run_mcl_pool(
            MclFourLanePool::field(field()),
            MclFourLaneOp::FpMul,
            iterations,
            start,
        ),
        (Op::FpSquare, Impl::Helius) => run_rotated(
            field(),
            iterations,
            start,
            |f| f.h_fp.0.square(),
            fold_helius_fp,
        ),
        (Op::FpSquare, Impl::Ark) => run_rotated(
            field(),
            iterations,
            start,
            |f| {
                use ark_ff::Field;
                f.a_fp.0.square()
            },
            fold_ark_fq,
        ),
        (Op::FpSquare, Impl::Mcl) => run_mcl_pool(
            MclFourLanePool::field(field()),
            MclFourLaneOp::FpSquare,
            iterations,
            start,
        ),
        (Op::Fp2Mul, Impl::Helius) => run_rotated(
            field(),
            iterations,
            start,
            |f| f.h_fp2.0 * f.h_fp2.1,
            fold_helius_fp2,
        ),
        (Op::Fp2Mul, Impl::Ark) => run_rotated(
            field(),
            iterations,
            start,
            |f| f.a_fp2.0 * f.a_fp2.1,
            fold_ark_fq2,
        ),
        (Op::Fp2Mul, Impl::Mcl) => run_mcl_pool(
            MclFourLanePool::field(field()),
            MclFourLaneOp::Fp2Mul,
            iterations,
            start,
        ),
        (Op::Fp2Square, Impl::Helius) => run_rotated(
            field(),
            iterations,
            start,
            |f| f.h_fp2.0.square(),
            fold_helius_fp2,
        ),
        (Op::Fp2Square, Impl::Ark) => run_rotated(
            field(),
            iterations,
            start,
            |f| {
                use ark_ff::Field;
                f.a_fp2.0.square()
            },
            fold_ark_fq2,
        ),
        (Op::Fp2Square, Impl::Mcl) => run_mcl_pool(
            MclFourLanePool::field(field()),
            MclFourLaneOp::Fp2Square,
            iterations,
            start,
        ),
        (Op::Fp12Mul, Impl::Helius) => run_rotated(
            field(),
            iterations,
            start,
            |f| f.h_fp12.0 * f.h_fp12.1,
            fold_helius_fp12,
        ),
        (Op::Fp12Mul, Impl::Ark) => run_rotated(
            field(),
            iterations,
            start,
            |f| f.a_fp12.0 * f.a_fp12.1,
            fold_ark_fq12,
        ),
        (Op::Fp12Mul, Impl::Mcl) => run_mcl_pool(
            MclFourLanePool::field(field()),
            MclFourLaneOp::Fp12Mul,
            iterations,
            start,
        ),
        (Op::Fp12Square, Impl::Helius) => run_rotated(
            field(),
            iterations,
            start,
            |f| f.h_fp12.0.square(),
            fold_helius_fp12,
        ),
        (Op::Fp12Square, Impl::Ark) => run_rotated(
            field(),
            iterations,
            start,
            |f| {
                use ark_ff::Field;
                f.a_fp12.0.square()
            },
            fold_ark_fq12,
        ),
        (Op::Fp12Square, Impl::Mcl) => run_mcl_pool(
            MclFourLanePool::field(field()),
            MclFourLaneOp::Fp12Square,
            iterations,
            start,
        ),
        (Op::Fp12Sparse034, Impl::Helius) => run_rotated(
            field(),
            iterations,
            start,
            |f| {
                f.h_fp12
                    .0
                    .mul_by_034(f.h_sparse.0, f.h_sparse.1, f.h_sparse.2)
            },
            fold_helius_fp12,
        ),
        (Op::Fp12Sparse034, Impl::Ark) => run_rotated(
            field(),
            iterations,
            start,
            |f| {
                let mut value = f.a_fp12.0;
                value.mul_by_034(&f.a_sparse.0, &f.a_sparse.1, &f.a_sparse.2);
                value
            },
            fold_ark_fq12,
        ),
        (Op::Fp12Sparse034, Impl::Mcl) => run_mcl_pool(
            MclFourLanePool::field(field()),
            MclFourLaneOp::Fp12Sparse034,
            iterations,
            start,
        ),
        (Op::Fp12CyclotomicSquare, Impl::Helius) => run_rotated(
            field(),
            iterations,
            start,
            |f| f.h_fp12_cyclo.cyclotomic_square(),
            fold_helius_fp12,
        ),
        (Op::Fp12CyclotomicSquare, Impl::Ark) => run_rotated(
            field(),
            iterations,
            start,
            |f| {
                use ark_ff::CyclotomicMultSubgroup;
                f.a_fp12_cyclo.cyclotomic_square()
            },
            fold_ark_fq12,
        ),
        (Op::Fp12CyclotomicSquare, Impl::Mcl) => run_mcl_pool(
            MclFourLanePool::field(field()),
            MclFourLaneOp::Fp12CyclotomicSquare,
            iterations,
            start,
        ),
        (Op::G1LineScaling, Impl::Helius) => run_rotated(
            field(),
            iterations,
            start,
            |f| f.h_fp2.0.mul_by_fp(f.h_scale),
            fold_helius_fp2,
        ),
        (Op::G1LineScaling, Impl::Ark) => run_rotated(
            field(),
            iterations,
            start,
            |f| {
                let mut value = f.a_fp2.0;
                value.mul_assign_by_fp(&f.a_scale);
                value
            },
            fold_ark_fq2,
        ),
        (Op::G1LineScaling, Impl::Mcl) => run_mcl_pool(
            MclFourLanePool::field(field()),
            MclFourLaneOp::G1LineScaling,
            iterations,
            start,
        ),
        // Preparation without validation, like the other lanes. Each lane
        // folds its own last-line bits, a lane-local digest by definition.
        (Op::G2Prepare87Lines, Impl::Helius) => run_rotated(
            miller(),
            iterations,
            start,
            |f| prepare_g2_unchecked(&f.helius_g2()),
            |acc, schedule| fold_words(acc, schedule.last_line_limbs()),
        ),
        (Op::G2Prepare87Lines, Impl::Ark) => run_rotated(
            miller(),
            iterations,
            start,
            |f| ArkG2Prepared::from(f.ark_g2()),
            |acc, schedule| {
                let (c0, c1, c2) = schedule
                    .ell_coeffs
                    .last()
                    .expect("a nonidentity schedule has lines");
                fold_ark_fq2(fold_ark_fq2(fold_ark_fq2(acc, c0), c1), c2)
            },
        ),
        (Op::G2Prepare87Lines, Impl::Mcl) => run_mcl_pool(
            MclFourLanePool::miller(miller()),
            MclFourLaneOp::G2Prepare,
            iterations,
            start,
        ),
        (Op::G1MsmPubInputs1 | Op::G1MsmPubInputs2 | Op::G1MsmPubInputs3, Impl::Helius) => {
            run_rotated(
                msm(),
                iterations,
                start,
                MsmPubInputsFixture::helius_msm_typed,
                fold_helius_g1_point,
            )
        }
        (Op::G1MsmPubInputs1 | Op::G1MsmPubInputs2 | Op::G1MsmPubInputs3, Impl::Ark) => {
            run_rotated(
                msm(),
                iterations,
                start,
                MsmPubInputsFixture::ark_msm_typed,
                fold_ark_g1_point,
            )
        }
        (Op::G1MsmPubInputs1 | Op::G1MsmPubInputs2 | Op::G1MsmPubInputs3, Impl::Mcl) => {
            run_mcl_pool(
                MclFourLanePool::msm(msm()),
                MclFourLaneOp::G1Msm,
                iterations,
                start,
            )
        }
        (Op::G1Validate, Impl::Helius) => run_rotated(
            subgroup(),
            iterations,
            start,
            SubgroupFixture::helius_g1_validate_digest,
            fold_helius_g1_validate,
        ),
        (Op::G1Validate, Impl::Ark) => run_rotated(
            subgroup(),
            iterations,
            start,
            SubgroupFixture::ark_g1_validate_digest,
            fold_ark_g1_validate,
        ),
        (Op::G1Validate, Impl::Mcl) => run_mcl_pool(
            MclFourLanePool::g1_blobs(subgroup()),
            MclFourLaneOp::G1Validate,
            iterations,
            start,
        ),
        (Op::G2SubgroupCheck, Impl::Helius) => run_rotated(
            subgroup(),
            iterations,
            start,
            SubgroupFixture::helius_g2_check_digest,
            fold_helius_g2_check,
        ),
        (Op::G2SubgroupCheck, Impl::Ark) => run_rotated(
            subgroup(),
            iterations,
            start,
            SubgroupFixture::ark_g2_check_digest,
            fold_ark_g2_check,
        ),
        (Op::G2SubgroupCheck, Impl::Mcl) => run_mcl_pool(
            MclFourLanePool::g2_points(subgroup()),
            MclFourLaneOp::G2Subgroup,
            iterations,
            start,
        ),
        (Op::MillerLive, Impl::Helius) => {
            run_rotated(miller(), iterations, start, helius_miller, fold_helius_fp12)
        }
        (Op::MillerLive, Impl::Ark) => {
            run_rotated(miller(), iterations, start, ark_miller, |acc, value| {
                fold_ark_fq12(acc, &value.0)
            })
        }
        (Op::MillerLive, Impl::Mcl) => run_mcl_pool(
            MclFourLanePool::miller(miller()),
            MclFourLaneOp::MillerLive,
            iterations,
            start,
        ),
        (Op::MillerPrepared, Impl::Helius) => run_rotated(
            miller(),
            iterations,
            start,
            helius_miller_prepared,
            fold_helius_fp12,
        ),
        (Op::MillerPrepared, Impl::Ark) => {
            let pool = miller();
            run_ark_replay(
                pool.len(),
                iterations,
                start,
                1,
                |index| ark_miller_prepared_setup(&pool[index]),
                |index, schedule| ark_miller_prepared_replay(&pool[index], schedule),
                |acc, value| fold_ark_fq12(acc, &value.0),
            )
        }
        (Op::MillerPrepared, Impl::Mcl) => run_mcl_pool(
            MclFourLanePool::miller(miller()),
            MclFourLaneOp::MillerPrepared,
            iterations,
            start,
        ),
        (Op::FinalExponentiation, Impl::Helius) => run_rotated(
            miller(),
            iterations,
            start,
            helius_final_exponentiation,
            fold_helius_fp12,
        ),
        (Op::FinalExponentiation, Impl::Ark) => run_rotated(
            miller(),
            iterations,
            start,
            ark_final_exponentiation,
            |acc, value| fold_ark_fq12(acc, &value.0),
        ),
        (Op::FinalExponentiation, Impl::Mcl) => run_mcl_pool(
            MclFourLanePool::miller(miller()),
            MclFourLaneOp::FinalExponentiation,
            iterations,
            start,
        ),
        (Op::FullPairing, Impl::Helius) => run_rotated(
            miller(),
            iterations,
            start,
            helius_full_pairing,
            fold_helius_fp12,
        ),
        (Op::FullPairing, Impl::Ark) => run_rotated(
            miller(),
            iterations,
            start,
            ark_full_pairing,
            |acc, value| fold_ark_fq12(acc, &value.0),
        ),
        (Op::FullPairing, Impl::Mcl) => run_mcl_pool(
            MclFourLanePool::miller(miller()),
            MclFourLaneOp::FullPairing,
            iterations,
            start,
        ),
        (Op::FullPairingPrepared, Impl::Helius) => run_rotated(
            miller(),
            iterations,
            start,
            helius_full_pairing_prepared,
            fold_helius_fp12,
        ),
        (Op::FullPairingPrepared, Impl::Ark) => {
            let pool = miller();
            run_ark_replay(
                pool.len(),
                iterations,
                start,
                1,
                |index| ark_miller_prepared_setup(&pool[index]),
                |index, schedule| ark_full_pairing_prepared_replay(&pool[index], schedule),
                |acc, value| fold_ark_fq12(acc, &value.0),
            )
        }
        (Op::FullPairingPrepared, Impl::Mcl) => run_mcl_pool(
            MclFourLanePool::miller(miller()),
            MclFourLaneOp::FullPairingPrepared,
            iterations,
            start,
        ),
        // The gnark row is the plain equation over proofs gnark made. The
        // whole pool sits under one verifying key, so every accepting vector
        // folds the same target-group value and only the public-input
        // accumulator binds the digest to the statement that was verified.
        (Op::Groth16Gnark, Impl::Helius) => {
            let pool = gnark();
            run_rotated(
                pool.valid(),
                iterations,
                start,
                helius_groth16_verify_accumulated,
                fold_helius_accumulated_verdict,
            )
        }
        (Op::Groth16Gnark, Impl::Ark) => {
            let pool = gnark();
            let vectors = pool.valid();
            run_ark_replay(
                vectors.len(),
                iterations,
                start,
                2,
                |index| ark_groth16_prepared_schedules(&vectors[index]),
                |index, (gamma_neg, delta_neg)| {
                    ark_groth16_verify_accumulated(&vectors[index], gamma_neg, delta_neg)
                },
                fold_ark_accumulated_verdict,
            )
        }
        (Op::Groth16Gnark, Impl::Mcl) => {
            let pool = gnark();
            run_mcl_pool(
                MclFourLanePool::groth16(pool.valid()),
                MclFourLaneOp::Groth16Accumulated,
                iterations,
                start,
            )
        }
        (Op::Groth16OneInput, Impl::Helius) => {
            let pool = production();
            let vectors = pool.standard();
            run_rotated(
                vectors,
                iterations,
                start,
                helius_groth16_verify_accumulated,
                fold_helius_accumulated_verdict,
            )
        }
        (Op::Groth16OneInput, Impl::Ark) => {
            let pool = production();
            let vectors = pool.standard();
            run_ark_replay(
                vectors.len(),
                iterations,
                start,
                2,
                |index| ark_groth16_prepared_schedules(&vectors[index]),
                |index, (gamma_neg, delta_neg)| {
                    ark_groth16_verify_accumulated(&vectors[index], gamma_neg, delta_neg)
                },
                fold_ark_accumulated_verdict,
            )
        }
        (Op::Groth16OneInput, Impl::Mcl) => {
            let pool = production();
            let vectors = pool.standard();
            run_mcl_pool(
                MclFourLanePool::groth16(vectors),
                MclFourLaneOp::Groth16Accumulated,
                iterations,
                start,
            )
        }
        (Op::Groth16CommittedRails, Impl::Helius) => {
            let pool = production();
            let vectors = pool.committed();
            run_rotated(
                vectors,
                iterations,
                start,
                helius_committed_verify_accumulated,
                fold_helius_accumulated_verdict,
            )
        }
        (Op::Groth16CommittedRails, Impl::Ark) => {
            let pool = production();
            let vectors = pool.committed();
            run_ark_replay(
                vectors.len(),
                iterations,
                start,
                4,
                |index| ark_committed_prepared_schedules(&vectors[index]),
                |index, schedules| ark_committed_verify_accumulated(&vectors[index], schedules),
                fold_ark_accumulated_verdict,
            )
        }
        (Op::Groth16CommittedRails, Impl::Mcl) => {
            let pool = production();
            let started = Instant::now();
            let digest = pool.mcl_committed_run(&pools.committed_hashes, iterations, start);
            (started.elapsed(), digest)
        }
        (
            Op::Groth16Batch8OneKey
            | Op::Groth16Batch8MixedKeys
            | Op::Groth16Batch3OneKey
            | Op::Groth16Batch3MixedKeys,
            Impl::Helius,
        ) => {
            let pool = batches();
            run_rotated(
                pool,
                iterations,
                start,
                helius_groth16_batch_verify_digest,
                fold_helius_batch_verdict,
            )
        }
        (
            Op::Groth16Batch8OneKey
            | Op::Groth16Batch8MixedKeys
            | Op::Groth16Batch3OneKey
            | Op::Groth16Batch3MixedKeys,
            Impl::Ark,
        ) => {
            let pool = batches();
            let widest = pool
                .iter()
                .map(Groth16BatchFixture::key_count)
                .max()
                .unwrap_or(1);
            run_ark_replay(
                pool.len(),
                iterations,
                start,
                3 * widest,
                |index| ark_batch_prepared_schedules(&pool[index]),
                |index, prepared| ark_batch_verify_accumulated(&pool[index], prepared),
                fold_ark_batch_verdict,
            )
        }
        (
            Op::Groth16Batch8OneKey
            | Op::Groth16Batch8MixedKeys
            | Op::Groth16Batch3OneKey
            | Op::Groth16Batch3MixedKeys,
            Impl::Mcl,
        ) => {
            let pool = batches();
            run_mcl_pool(
                MclFourLanePool::batch(pool),
                MclFourLaneOp::Groth16Batch,
                iterations,
                start,
            )
        }
        // Gate E's anchor. Every lane runs this arm, so the row measures one
        // code path four times and its cross-lane spread is a property of the
        // host and the clock, not of any library.
        (Op::LaneReference, _) => run_rotated(
            field(),
            iterations,
            start,
            |f| lane_reference_chain(f.h_fp.0.mont_limbs()[0]),
            |acc, value| fold_word(acc, *value),
        ),
    }
}

fn print_provenance(
    inputs: &GateInputs,
    timer: &TimerReport,
    fixture_seed: u64,
    fixture_set_sha256: &str,
) {
    let provenance = serde_json::json!({
        "schema": "four-lane-provenance-v1",
        "rustflags": env!("FOUR_LANE_BUILD_RUSTFLAGS"),
        "target_cpu": inputs.target_cpu,
        "ark_asm": inputs.ark_asm,
        "mcl_archive_sha256": env!("FOUR_LANE_BUILD_MCL_ARCHIVE_SHA256"),
        "mcl_manifest_sha256": inputs.linked_manifest_sha256,
        "cxx": env!("FOUR_LANE_BUILD_CXX"),
        "cxx_native_flag": inputs.cxx_native_flag,
        "helius_backend": inputs.helius_backend,
        "mcl_runtime": if inputs.mcl_jit { "xbyak-jit" } else { "no-jit" },
        "git_rev": env!("FOUR_LANE_BUILD_GIT_REV"),
        "timer_overhead_ns_per_iter": timer.overhead_ns_per_iter,
        "tsc_agreement_pct": timer.tsc_agreement_pct,
        "fixture_seed": format!("0x{fixture_seed:016x}"),
        "fixture_set_sha256": fixture_set_sha256,
    });
    println!("provenance|{provenance}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use helius_narsil_bench::{SEED, mix64, pool_seed};

    fn valid_manifest() -> ManifestFacts {
        ManifestFacts {
            disk_sha256: "abc".to_owned(),
            revision: MCL_REVISION.to_owned(),
            schema: MCL_MANIFEST_SCHEMA.to_owned(),
            native_flag: "-march=native".to_owned(),
        }
    }

    fn valid_inputs(lane: Lane) -> GateInputs {
        GateInputs {
            lane,
            x86: true,
            target_cpu: "native",
            cxx_native_flag: "-march=native",
            ark_asm: true,
            force_portable: false,
            helius_backend: "avx512-ifma",
            mcl_jit: true,
            manifest: Ok(valid_manifest()),
            linked_manifest_sha256: "abc",
        }
    }

    #[test]
    fn gates_admit_a_conforming_x86_build() {
        for lane in [Lane::Helius, Lane::Arkworks, Lane::Mcl] {
            assert!(gate_failures(&valid_inputs(lane)).is_empty(), "{lane:?}");
        }
    }

    #[test]
    fn gates_bind_arkworks_lanes_to_their_build_regime() {
        let mut inputs = valid_inputs(Lane::ArkworksPortable);
        assert_eq!(gate_failures(&inputs).len(), 1);
        inputs.ark_asm = false;
        assert!(gate_failures(&inputs).is_empty());
        inputs.lane = Lane::Arkworks;
        assert_eq!(gate_failures(&inputs).len(), 1);
        // The lane regime binds on every architecture.
        inputs.x86 = false;
        assert_eq!(gate_failures(&inputs).len(), 1);
    }

    #[test]
    fn gates_reject_non_native_regimes_on_x86() {
        let mut inputs = valid_inputs(Lane::Mcl);
        inputs.target_cpu = "znver5";
        inputs.cxx_native_flag = "NONE";
        assert_eq!(gate_failures(&inputs).len(), 2);
    }

    #[test]
    fn gates_pin_the_mcl_manifest() {
        let mut inputs = valid_inputs(Lane::Mcl);
        inputs.linked_manifest_sha256 = "other";
        assert_eq!(gate_failures(&inputs).len(), 1);

        let mut inputs = valid_inputs(Lane::Mcl);
        inputs.manifest = Ok(ManifestFacts {
            revision: "0000".to_owned(),
            schema: "wrong".to_owned(),
            native_flag: "-mcpu=native".to_owned(),
            ..valid_manifest()
        });
        assert_eq!(gate_failures(&inputs).len(), 3);

        let mut inputs = valid_inputs(Lane::Mcl);
        inputs.manifest = Err("missing".to_owned());
        assert_eq!(gate_failures(&inputs).len(), 1);
    }

    #[test]
    fn gates_require_the_fast_backends_per_lane() {
        let mut inputs = valid_inputs(Lane::Helius);
        inputs.helius_backend = "scalar-x86-adx";
        assert_eq!(gate_failures(&inputs).len(), 1);
        inputs.force_portable = true;
        assert!(gate_failures(&inputs).is_empty());

        let mut inputs = valid_inputs(Lane::Mcl);
        inputs.mcl_jit = false;
        assert_eq!(gate_failures(&inputs).len(), 1);
        // Other lanes run even when mcl skipped its JIT.
        inputs.lane = Lane::Arkworks;
        assert!(gate_failures(&inputs).is_empty());
    }

    #[test]
    fn gates_skip_x86_regime_checks_off_x86() {
        let mut inputs = valid_inputs(Lane::Helius);
        inputs.x86 = false;
        inputs.target_cpu = "unspecified";
        inputs.helius_backend = "non-x86 -> aarch64-apple-scalar";
        inputs.mcl_jit = false;
        assert!(gate_failures(&inputs).is_empty());
    }

    #[test]
    fn operation_names_are_unique_and_complete() {
        let names: HashSet<_> = OPERATIONS.iter().map(|(_, name)| *name).collect();
        assert_eq!(names.len(), OPERATIONS.len());
        for (op, name) in OPERATIONS {
            assert_eq!(Op::parse(name), Some(op));
            assert_eq!(op.name(), name);
        }
        // The synthetic rows the audit rejected are gone, not renamed.
        for retired in [
            "groth16_single",
            "groth16_committed",
            "groth16_multiproof_3",
        ] {
            assert_eq!(Op::parse(retired), None, "{retired}");
        }
    }

    #[test]
    fn fixture_seed_parses_both_radices_and_rejects_junk() {
        assert_eq!(parse_fixture_seed("0"), Some(0));
        assert_eq!(parse_fixture_seed(" 42 "), Some(42));
        assert_eq!(parse_fixture_seed("0xff"), Some(255));
        assert_eq!(parse_fixture_seed("0XFF"), Some(255));
        assert_eq!(parse_fixture_seed("0xffffffffffffffff"), Some(u64::MAX));
        for junk in ["", "-1", "0x", "1g", "0x1_0", "18446744073709551616"] {
            assert_eq!(parse_fixture_seed(junk), None, "{junk}");
        }
    }

    #[test]
    fn a_zero_run_seed_keeps_the_frozen_pools() {
        assert_eq!(mix64(0), 0);
        for domain in [FIELD_POOL_DOMAIN, MILLER_POOL_DOMAIN] {
            for index in 0..8 {
                assert_eq!(
                    pool_seed(0, domain, index),
                    SEED ^ domain.rotate_left(17)
                        ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                );
            }
        }
    }

    /// One verifying key covers the whole gnark pool, so every accepting proof
    /// folds the same target-group value. Only the public-input accumulator
    /// tells the digest which statement the lane verified.
    #[test]
    fn the_gnark_row_digest_moves_with_the_seed_and_the_round() {
        let iterations = 4;
        let mut digests = HashSet::new();
        for seed in [0_u64, 0x5eed_0001] {
            let pools = Pools::for_op(Op::Groth16Gnark, seed);
            for rotation in 0..4 {
                let start =
                    rotation_window_start(pool_len(Op::Groth16Gnark, &pools), iterations, rotation);
                let lanes: Vec<u64> = [Impl::Helius, Impl::Ark, Impl::Mcl]
                    .into_iter()
                    .map(|implementation| {
                        run_lane(Op::Groth16Gnark, implementation, &pools, iterations, start).1
                    })
                    .collect();
                assert_eq!(
                    lanes.iter().collect::<HashSet<_>>().len(),
                    1,
                    "seed {seed} round {rotation} disagrees across lanes"
                );
                digests.insert(lanes[0]);
            }
        }
        assert_eq!(digests.len(), 8);
    }

    /// One operation per pool kind. A campaign of rounds must draw a window no
    /// earlier round drew, and must stop once the pool has no fresh start.
    #[test]
    fn a_campaign_draws_a_fresh_window_on_every_row() {
        let iterations = 4;
        for op in [
            Op::FpMul,
            Op::MillerLive,
            Op::G1MsmPubInputs2,
            Op::G1Validate,
            Op::Groth16Gnark,
            Op::Groth16OneInput,
            Op::Groth16CommittedRails,
            Op::Groth16Batch8OneKey,
            Op::Groth16Batch8MixedKeys,
        ] {
            let pools = Pools::for_op(op, 0);
            let capacity = rotation_window_capacity(pool_len(op, &pools), iterations);
            let identities: HashSet<u64> = (0..capacity)
                .map(|rotation| window_identity(op, &pools, iterations, rotation))
                .collect();
            assert_eq!(identities.len(), capacity, "{}", op.name());
            assert!(check_rotation_window(op, &pools, iterations, capacity - 1).is_ok());
            assert!(check_rotation_window(op, &pools, iterations, capacity).is_err());
        }
    }

    /// Four rounds of the campaign schedule, per row, split by what the pool
    /// can deliver. A row whose round draws a proper subset must give four
    /// different fixture sets. A row whose round already walks the whole pool
    /// cannot, and the harness must say so rather than call a rotation fresh.
    #[test]
    fn four_rounds_draw_four_sets_wherever_the_pool_allows_it() {
        const ROUNDS: usize = 4;
        for (op, iterations) in [
            (Op::FpMul, 16_384),
            (Op::G1Validate, 4_096),
            (Op::G1MsmPubInputs2, 512),
            (Op::MillerLive, 16),
            (Op::G2Prepare87Lines, 16),
            (Op::Groth16Gnark, 8),
            (Op::Groth16OneInput, 8),
            (Op::Groth16CommittedRails, 8),
            (Op::Groth16Batch8OneKey, 4),
            (Op::Groth16Batch8MixedKeys, 4),
            (Op::Groth16Batch3OneKey, 8),
            (Op::Groth16Batch3MixedKeys, 8),
        ] {
            let pools = Pools::for_op(op, 0);
            let length = pool_len(op, &pools);
            let sets: HashSet<u64> = (0..ROUNDS)
                .map(|round| window_set_identity(op, &pools, iterations, round))
                .collect();
            let expected = if rotation_draws_distinct_sets(length, iterations) {
                ROUNDS
            } else {
                1
            };
            assert_eq!(sets.len(), expected, "{}", op.name());
            for round in 0..ROUNDS {
                assert!(
                    check_rotation_window(op, &pools, iterations, round).is_ok(),
                    "{}",
                    op.name()
                );
            }
        }
    }

    /// The old check compared one round against its neighbour only, so a
    /// campaign that folded back onto round 0 every second round passed it.
    #[test]
    fn a_repeated_window_is_caught_even_when_it_is_not_adjacent() {
        assert_eq!(
            first_repeated_round(4, |round| (round % 2) as u64),
            Some((0, 2))
        );
        assert_eq!(
            first_repeated_round(4, |round| (round % 3) as u64),
            Some((0, 3))
        );
        assert_eq!(first_repeated_round(64, |round| round as u64), None);
    }

    #[test]
    fn a_run_seed_moves_every_pool_entry_and_keeps_domains_apart() {
        let seeds: HashSet<u64> = [0_u64, 1, u64::MAX]
            .into_iter()
            .flat_map(|run| {
                [FIELD_POOL_DOMAIN, MILLER_POOL_DOMAIN]
                    .into_iter()
                    .flat_map(move |domain| (0..16).map(move |index| pool_seed(run, domain, index)))
            })
            .collect();
        assert_eq!(seeds.len(), 3 * 2 * 16);
    }
}
