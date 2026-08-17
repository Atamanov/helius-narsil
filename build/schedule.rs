//! The Montgomery kernel schedules for 4x64 moduli (BN254's p).
//!
//! These functions are the audited artifact: each call names the semantic
//! value it computes, and the emitted assembly is this text, one instruction
//! per call, in this order. Both kernels receive `(z, x, y, consts)` in the
//! System V argument registers, with `consts = { p[4], -p^-1 mod 2^64 }`
//! owned by Rust (same contract as the AArch64 leaf).
//!
//! # Dual carry chains
//!
//! Products are accumulated with two independent carry chains so consecutive
//! `mulx` results never serialize on one flag:
//!
//! * the **value chain** (`adox`, OF) adds each product's low half into
//!   accumulator word `j`.
//! * the **carry chain** (`adcx`, CF) adds each product's high half into
//!   accumulator word `j + 1`.
//!
//! The chains cannot collide: within a row, word `j` receives exactly one
//! `adox` and one `adcx`, and `mulx`/`mov` between them touch no flags.
//!
//! # Carry bounds (why the chains close without a sixth word)
//!
//! Requires `p < 2^62 * 2^192` (four-limb modulus with two spare top bits.
//! BN254's top limb is `0x3064...` ~ 2^61.6). With `a, b < p` the CIOS
//! accumulator obeys `t < 2p` at every round boundary, so:
//!
//! * after a product row:    `t + a*b_i  < 2p + 2^64*p        < 2^64 * 2^255`
//! * after the cancel row:   `... + m*p  < 2^64*p + 2^64*p    < 2^65 * 2^255`
//!
//! Both stay below `2^320`, so the fifth word absorbs every carry and both
//! chains provably close with CF = OF = 0 -- the `claim_flags_clear` calls
//! make the interpreter check exactly that, and each round relies on it
//! instead of re-clearing flags.

use super::machine::Reg::{
    R8, R9, R10, R11, R12, R13, R14, R15, Rax, Rbp, Rbx, Rcx, Rdi, Rdx, Rsi,
};
use super::machine::{LoopEnd, Machine, Mem, Reg};

/// Register roles for `narsil_mont4_mul_x86`. All fifteen usable GPRs are
/// live. There are no spills.
pub const MUL_REGISTER_MAP: &[(Reg, &str)] = &[
    (Rdi, "z: result pointer (argument 1, live throughout)"),
    (
        Rsi,
        "x pointer on entry; repointed at y after the operand load",
    ),
    (
        Rdx,
        "y pointer on entry; then the implicit mulx multiplicand",
    ),
    (Rcx, "consts pointer: p at +0..+24, -p^-1 at +32"),
    (R8, "a0 (x limb 0, loaded once)"),
    (R9, "a1"),
    (R10, "a2"),
    (R11, "a3"),
    (
        R12,
        "CIOS accumulator (rotates: t_k of round r is ACC[(r+k) % 5])",
    ),
    (R13, "CIOS accumulator"),
    (R14, "CIOS accumulator"),
    (R15, "CIOS accumulator"),
    (Rbp, "CIOS accumulator"),
    (
        Rax,
        "low half of the current product; zero for chain closes",
    ),
    (Rbx, "high half of the current product"),
];

/// Register roles for `narsil_mont4_sqr_x86`.
pub const SQR_REGISTER_MAP: &[(Reg, &str)] = &[
    (Rdi, "z: result pointer (argument 1, live throughout)"),
    (Rsi, "x pointer on entry; then cross-product word C6 / T6"),
    (
        Rdx,
        "unused argument on entry; the implicit mulx multiplicand",
    ),
    (Rcx, "consts pointer: p at +0..+24, -p^-1 at +32"),
    (R8, "x0; then T0 (dies as round 0 cancels it)"),
    (R9, "x1; then high-half scratch of the reduction rows"),
    (R10, "x2; freed by the diagonal row"),
    (R11, "x3; freed by the diagonal row"),
    (R12, "cross-product word C1; then T1"),
    (R13, "cross-product word C2; then T2"),
    (R14, "cross-product word C3; then T3"),
    (R15, "cross-product word C4; then T4"),
    (Rbp, "cross-product word C5; then T5"),
    (Rbx, "doubling carry H; then T7"),
    (
        Rax,
        "low half of the current product; zero for chain closes",
    ),
];

const OUT_PTR: Reg = Rdi;
const CONSTS: Reg = Rcx;
/// The implicit `mulx` multiplicand.
const MULTIPLIER: Reg = Rdx;
/// `x` limbs for mul, `x` limbs for sqr. Loaded once, immutable.
const A: [Reg; 4] = [R8, R9, R10, R11];
/// Rotating five-word CIOS accumulator (mul kernel).
const ACC: [Reg; 5] = [R12, R13, R14, R15, Rbp];
/// Product low half / zero source for chain closes.
const LO: Reg = Rax;

const CALLEE_SAVED: [Reg; 6] = [Rbx, Rbp, R12, R13, R14, R15];

fn p_limb(j: usize) -> Mem {
    Mem::new(CONSTS, 8 * j as i32)
}

fn p_inv() -> Mem {
    Mem::new(CONSTS, 32)
}

/// Accumulator register of round `round`, logical word `k`. Rounds shift the
/// accumulator down one word by renaming registers instead of moving data.
fn acc(round: usize, k: usize) -> Reg {
    ACC[(round + k) % 5]
}

/// `t[j] += lo(product)` on the value chain, `t[j+1] += hi(product)` on the
/// carry chain. `hi` is the per-kernel high-half scratch register.
fn mul_into_columns<M: Machine>(
    m: &mut M,
    hi: Reg,
    src: Reg,
    t_lo: Reg,
    t_hi: Reg,
    product: &str,
    j: usize,
) {
    m.mulx(hi, LO, src, &format!("{product} -> (lo, hi)"));
    m.adox(t_lo, LO, &format!("t{j} += lo({product})   [value chain]"));
    m.adcx(
        t_hi,
        hi,
        &format!("t{} += hi({product})   [carry chain]", j + 1),
    );
}

/// Same column step with the multiplicand taken from the constant table.
fn mul_mem_into_columns<M: Machine>(
    m: &mut M,
    hi: Reg,
    src: Mem,
    t_lo: Reg,
    t_hi: Reg,
    product: &str,
    j: usize,
) {
    m.mulx_mem(hi, LO, src, &format!("{product} -> (lo, hi)"));
    m.adox(t_lo, LO, &format!("t{j} += lo({product})   [value chain]"));
    m.adcx(
        t_hi,
        hi,
        &format!("t{} += hi({product})   [carry chain]", j + 1),
    );
}

/// Montgomery cancel row: `m = t0 * (-p^-1) mod 2^64`, then `t += m*p`, which
/// forces `t0` to exactly zero. Shared verbatim by both kernels. `t` is the
/// five-word window, `hi` the high-half scratch.
///
/// Entry and exit invariant: CF = OF = 0 (see the module bound argument. For
/// the squaring kernel the caller ripples the window carries out instead).
fn cancel_low_word<M: Machine>(m: &mut M, t: [Reg; 5], hi: Reg, round: usize) {
    cancel_low_word_at(m, t, hi, CONSTS, &round.to_string());
}

/// Same cancel row with the constants table at `consts`: the rolled sosd2
/// kernel keeps rcx as its round cursor and reloads the table pointer.
fn cancel_low_word_at<M: Machine>(m: &mut M, t: [Reg; 5], hi: Reg, consts: Reg, tag: &str) {
    m.mov(MULTIPLIER, t[0], &format!("m{tag} multiplicand <- t0"));
    m.mulx_mem(
        hi,
        MULTIPLIER,
        Mem::new(consts, 32),
        &format!("m{tag} = t0 * -p^-1 mod 2^64 (hi half discarded)"),
    );
    for j in 0..4 {
        let product = format!("m{tag}*p{j}");
        mul_mem_into_columns(
            m,
            hi,
            Mem::new(consts, 8 * j as i32),
            t[j],
            t[j + 1],
            &product,
            j,
        );
    }
    m.mov_zero(LO, "zero for the chain closes (flags preserved)");
    m.adox(t[4], LO, "close the value chain into t4");
}

/// Push callee-saved registers and load the four `x` limbs into registers.
fn enter<M: Machine>(m: &mut M, operand: &str) {
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    for (j, reg) in A.iter().enumerate() {
        m.load(*reg, Mem::new(Rsi, 8 * j as i32), &format!("{operand}{j}"));
    }
}

/// Conditionally subtract p and store. `value < 2p` fits four words, so a
/// four-word borrow decides: borrow means `value < p`, keep the original.
fn reduce_and_store<M: Machine>(m: &mut M, value: [Reg; 4], keep: [Reg; 4]) {
    m.comment("final reduction: value < 2p, subtract p once if value >= p");
    for (j, (v, k)) in value.iter().zip(keep).enumerate() {
        m.mov(k, *v, &format!("keep-copy of word {j}"));
    }
    for (j, v) in value.iter().enumerate() {
        let what = format!("word {j} -= p{j}");
        if j == 0 {
            m.sub_mem(*v, p_limb(j), &what);
        } else {
            m.sbb_mem(*v, p_limb(j), &what);
        }
    }
    for (j, (v, k)) in value.iter().zip(keep).enumerate() {
        m.cmov_carry(*v, k, &format!("borrow: value < p, keep word {j}"));
    }
    for (j, v) in value.iter().enumerate() {
        m.store(Mem::new(OUT_PTR, 8 * j as i32), *v, &format!("z{j}"));
    }
    for reg in CALLEE_SAVED.iter().rev() {
        m.pop(*reg);
    }
    m.ret();
}

/// `narsil_mont4_mul_x86`: fully unrolled CIOS Montgomery multiplication.
///
/// Round r: `t += a * b_r` (product row), then one [`cancel_low_word`]
/// (cancel row), then the shift-by-renaming. Round 0 builds `t` directly from
/// the products (single adc chain) instead of adding into zeros.
pub fn mont4_mul<M: Machine>(m: &mut M) {
    let hi = Rbx;
    enter(m, "a");
    m.mov(
        Rsi,
        Rdx,
        "y pointer moves; rdx becomes the mulx multiplicand",
    );

    m.comment("");
    m.comment("round 0: t = a*b0, then cancel t0");
    let (t0, t1, t2, t3, t4) = (acc(0, 0), acc(0, 1), acc(0, 2), acc(0, 3), acc(0, 4));
    m.load(MULTIPLIER, Mem::new(Rsi, 0), "b0");
    m.mulx(t1, t0, A[0], "a0*b0 -> (t0, t1)");
    m.mulx(t2, LO, A[1], "a1*b0 -> (lo, t2)");
    m.add(t1, LO, "t1 += lo(a1*b0)");
    m.mulx(t3, LO, A[2], "a2*b0 -> (lo, t3)");
    m.adc(t2, LO, "t2 += lo(a2*b0)");
    m.mulx(t4, LO, A[3], "a3*b0 -> (lo, t4)");
    m.adc(t3, LO, "t3 += lo(a3*b0)");
    m.adc_zero(t4, "t4 += chain carry; hi(a3*b0) <= 2^64-2 so CF = 0");
    m.xor_clear(
        LO,
        "clear OF (adc left it undefined) before the dual chains",
    );
    cancel_low_word(m, [t0, t1, t2, t3, t4], hi, 0);

    for round in 1..4 {
        m.comment("");
        m.comment(&format!("round {round}: t += a*b{round}, then cancel t0"));
        m.claim_flags_clear("previous round closed both chains under the 2^320 bound");
        let t: [Reg; 5] = core::array::from_fn(|k| acc(round, k));
        m.load(
            MULTIPLIER,
            Mem::new(Rsi, 8 * round as i32),
            &format!("b{round}"),
        );
        for j in 0..4 {
            let product = format!("a{j}*b{round}");
            mul_into_columns(m, hi, A[j], t[j], t[j + 1], &product, j);
        }
        m.mov_zero(LO, "zero for the chain closes (flags preserved)");
        m.adox(t[4], LO, "close the value chain into t4");
        cancel_low_word(m, t, hi, round);
    }
    m.claim_flags_clear("final round closed both chains under the 2^320 bound");

    m.comment("");
    // Round 3's canceled word 0 drops. Its words 1..4 are the result, which
    // in the round-4 renaming frame are exactly acc(4, 0..3).
    let result: [Reg; 4] = core::array::from_fn(|k| acc(4, k));
    reduce_and_store(m, result, A);
}

/// `narsil_mont4_sqr_x86`: dedicated Montgomery squaring.
///
/// Ten products instead of sixteen: six cross products summed once and
/// doubled by an add-to-self chain, then four diagonal squares folded in.
/// The full 512-bit square lives in eight registers. Reduction then runs
/// four [`cancel_low_word`] rows in exactly the mul kernel's style, with the
/// window carries rippled out to the top through both chains.
///
/// Trade-off vs. `mont4_mul`: fewer multiplier uops (port pressure win on
/// wide cores), but the reduction's `m_i` values depend serially on finished
/// words T_i, so the dependent-chain latency win is smaller than the
/// product-count saving suggests.
///
/// Bound: the running value stays below `p^2 + p*2^256 < 2^511`, so no carry
/// ever leaves T7 and every round re-enters with CF = OF = 0.
pub fn mont4_sqr<M: Machine>(m: &mut M) {
    let x = A; // x0..x3, same physical registers as the mul kernel's a.
    let (c1, c2, c3, c4, c5, c6) = (R12, R13, R14, R15, Rbp, Rsi);
    let h = Rbx;
    enter(m, "x");

    m.comment("");
    m.comment("cross products: C = sum of x_i*x_j (i < j), words C1..C6");
    m.mov(MULTIPLIER, x[0], "multiplicand <- x0");
    m.mulx(c4, c3, x[3], "x0*x3 -> (C3, C4)");
    m.mulx(LO, c2, x[2], "x0*x2 -> (C2, hi)");
    m.add(c3, LO, "C3 += hi(x0*x2)");
    m.mov(MULTIPLIER, x[1], "multiplicand <- x1");
    m.mulx(c5, LO, x[3], "x1*x3 -> (lo, C5)");
    m.adc(c4, LO, "C4 += lo(x1*x3)");
    m.adc_zero(c5, "C5 += chain carry; hi(x1*x3) <= 2^64-2 so CF = 0");
    m.mulx(LO, c1, x[0], "x0*x1 -> (C1, hi)");
    m.add(c2, LO, "C2 += hi(x0*x1)   [new chain]");
    m.mulx(h, LO, x[2], "x1*x2 -> (lo, hi)");
    m.adc(c3, LO, "C3 += lo(x1*x2)");
    m.adc(c4, h, "C4 += hi(x1*x2)");
    m.mov(MULTIPLIER, x[3], "multiplicand <- x3");
    m.mulx(c6, LO, x[2], "x2*x3 -> (lo, C6)");
    m.adc(c5, LO, "C5 += lo(x2*x3)");
    m.adc_zero(c6, "C6 += chain carry; hi(x2*x3) <= 2^64-2 so CF = 0");

    m.comment("");
    m.comment("double the cross words: T channel = 2C, carry bit lands in H");
    m.xor_clear(h, "H = 0; also clears CF for the doubling chain");
    for (word, c) in [c1, c2, c3, c4, c5, c6].into_iter().enumerate() {
        let what = format!("C{} *= 2", word + 1);
        if word == 0 {
            m.add(c, c, &what);
        } else {
            m.adc(c, c, &what);
        }
    }
    m.adc(h, h, "H = carry shifted out of 2*C6");

    m.comment("");
    m.comment("fold the diagonal squares: T = 2C + sum of x_i^2 * 2^(128i)");
    let t_of_c = [c1, c2, c3, c4, c5, c6];
    // Diagonal x_j^2 covers words 2j and 2j+1. Word 0 (T0) has no cross term.
    // each x_j frees its own register the moment it becomes the multiplicand.
    m.mov(MULTIPLIER, x[0], "multiplicand <- x0 (r8 freed for T0)");
    m.mulx(LO, x[0], MULTIPLIER, "x0^2 -> (T0, hi)");
    m.add(t_of_c[0], LO, "T1 = 2C1 + hi(x0^2)");
    for j in 1..4 {
        m.mov(
            MULTIPLIER,
            x[j],
            &format!("multiplicand <- x{j} (register freed)"),
        );
        m.mulx(x[j], LO, MULTIPLIER, &format!("x{j}^2 -> (lo, hi)"));
        m.adc(t_of_c[2 * j - 1], LO, &format!("T{} += lo(x{j}^2)", 2 * j));
        if j < 3 {
            m.adc(
                t_of_c[2 * j],
                x[j],
                &format!("T{} += hi(x{j}^2)", 2 * j + 1),
            );
        } else {
            m.adc(h, x[j], "T7 = H + hi(x3^2); x^2 < 2^512 so CF = 0");
        }
    }

    // T0..T7. R9..R11 (former x1..x3) are now scratch.
    let t = [x[0], c1, c2, c3, c4, c5, c6, h];
    let hi = R9;
    m.comment("");
    m.comment("Montgomery reduction: four cancel rows over the 8-word square");
    m.xor_clear(
        LO,
        "clear OF (adc left it undefined) before the dual chains",
    );
    for round in 0..4 {
        if round > 0 {
            m.comment("");
            m.claim_flags_clear("previous row rippled both chains out under the 2^511 bound");
        }
        let window: [Reg; 5] = core::array::from_fn(|k| t[round + k]);
        cancel_low_word(m, window, hi, round);
        for (k, &word) in t.iter().enumerate().skip(round + 5) {
            m.adcx(word, LO, &format!("T{k} += carry-chain ripple"));
            m.adox(word, LO, &format!("T{k} += value-chain ripple"));
        }
    }
    m.claim_flags_clear("last row closed at T7 under the 2^511 bound");

    m.comment("");
    let result: [Reg; 4] = core::array::from_fn(|k| t[4 + k]);
    reduce_and_store(m, result, [x[0], R9, R10, R11]);
}

/// Register roles for the compact direct 4x4 raw-product primitive used by
/// the MCL-shaped sparse tower.
pub const MULPRE_REGISTER_MAP: &[(Reg, &str)] = &[
    (Rdi, "y pointer after entry"),
    (Rsi, "x pointer"),
    (Rdx, "implicit mulx multiplicand"),
    (R8, "raw accumulator word 0"),
    (R9, "raw accumulator word 1"),
    (R10, "raw accumulator word 2"),
    (R11, "raw accumulator word 3"),
    (R12, "raw accumulator word 4"),
    (R13, "raw accumulator word 5"),
    (R14, "x/output byte cursor"),
    (Rbp, "z pointer"),
    (Rax, "product low half"),
    (Rbx, "product high half"),
];

/// `z[0..8] = x[0..4] * y[0..4]`, with direct operands and destination.
/// Four rolled rows keep text compact. No pointer table or operand staging.
pub fn mont4_mulpre_x86<M: Machine>(m: &mut M) {
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    m.mov(Rbp, Rdi, "z");
    m.mov(Rdi, Rdx, "y; rdx becomes mulx multiplicand");
    for (k, t) in T.into_iter().enumerate() {
        m.xor_clear(t, &format!("t{k} = 0"));
    }
    m.xor_clear(R14, "byte cursor 8j");
    m.stride_loop(R14, 8, LoopEnd::Imm(32), ".Lmulpre_j", &mut |m| {
        m.load_indexed(MULTIPLIER, Rsi, R14, "x[j]");
        m.xor_clear(LO, "seed CF = OF = 0");
        fsq_mulpre_row(m, Rdi);
        m.store(Mem::new(Rbp, 0), T[0], "raw product limb j");
        m.add_imm(Rbp, 8, "next output limb");
        for k in 0..5 {
            m.mov(T[k], T[k + 1], &format!("t{k} = t{}", k + 1));
        }
        m.xor_clear(T[5], "top word clears after shift");
    });
    for (k, t) in T[..4].iter().enumerate() {
        m.store(
            Mem::new(Rbp, 8 * k as i32),
            *t,
            &format!("raw product limb {}", k + 4),
        );
    }
    for reg in CALLEE_SAVED.iter().rev() {
        m.pop(*reg);
    }
    m.ret();
}

/// Register roles for the direct REDC primitive.
pub const REDC_REGISTER_MAP: &[(Reg, &str)] = &[
    (Rdi, "canonical destination"),
    (Rsi, "eight-limb input"),
    (Rdx, "consts on entry; then implicit mulx multiplicand"),
    (Rcx, "consts after entry"),
    (R8, "T0 / keep scratch"),
    (R9, "T1 / keep scratch"),
    (R10, "T2 / keep scratch"),
    (R11, "T3 / keep scratch"),
    (R12, "T4 / result word 0"),
    (R13, "T5 / result word 1"),
    (R14, "T6 / result word 2"),
    (R15, "T7 / result word 3"),
    (Rax, "product low half"),
    (Rbx, "product high half"),
];

/// Montgomery-reduce one direct eight-limb value `T < p*2^256`.
pub fn mont4_redc_x86<M: Machine>(m: &mut M) {
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    m.mov(Rcx, Rdx, "consts");
    let t = [R8, R9, R10, R11, R12, R13, R14, R15];
    for (k, reg) in t.into_iter().enumerate() {
        m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("T{k}"));
    }
    m.xor_clear(LO, "seed CF = OF = 0");
    for round in 0..4 {
        if round > 0 {
            m.claim_flags_clear("previous cancellation closed below 2pR");
        }
        let window = core::array::from_fn(|k| t[round + k]);
        cancel_low_word(m, window, Rbx, round);
        for (k, word) in t.iter().enumerate().skip(round + 5) {
            m.adcx(*word, LO, &format!("T{k} carry-chain ripple"));
            m.adox(*word, LO, &format!("T{k} value-chain ripple"));
        }
    }
    m.claim_flags_clear("T < pR implies REDC(T) < 2p");
    reduce_and_store(m, [R12, R13, R14, R15], [R8, R9, R10, R11]);
}

/// Register roles for `narsil_sos_x86`.
pub const SOS_REGISTER_MAP: &[(Reg, &str)] = &[
    (
        Rdi,
        "z on entry (spilled); then the current b_i pointer inside a row",
    ),
    (Rsi, "pair-table base (argument 2, live throughout)"),
    (
        Rdx,
        "T (pair count) on entry; then the implicit mulx multiplicand",
    ),
    (Rcx, "consts pointer: p at +0..+24, -p^-1 at +32"),
    (R8, "accumulator t0"),
    (R9, "accumulator t1"),
    (R10, "accumulator t2"),
    (R11, "accumulator t3"),
    (R12, "accumulator t4"),
    (R13, "accumulator t5 (top carry word)"),
    (
        R14,
        "byte offset 8j of the round's source limb; epilogue scratch",
    ),
    (R15, "pair-table cursor; epilogue scratch"),
    (Rbp, "pair-table end (rsi + 16T)"),
    (
        Rax,
        "low half of the current product; zero for chain closes",
    ),
    (Rbx, "high half of the current product"),
];

/// Six-word SoS accumulator t0..t5.
const T: [Reg; 6] = [R8, R9, R10, R11, R12, R13];
/// Byte offset `8j` of the current CIOS round's source limb.
const JOFF: Reg = R14;
/// Pair-table cursor of the inner product walk.
const CURSOR: Reg = R15;
/// Pair-table end bound (`rsi + 16T`).
const TABLE_END: Reg = Rbp;

/// One dual-chain product row into the six-word accumulator: for the value
/// already in `rdx`, `t[k] += lo_k` on the value chain and `t[k+1] += hi_k`
/// on the carry chain, source limbs from `[b + off + 8k]`. Both chains are
/// then closed into the top words, so the row leaves CF = OF = 0 and nothing
/// crosses the following back edge or row boundary.
fn sos_row_at<M: Machine>(m: &mut M, b: Reg, off: i32, product: &str) {
    for k in 0..4 {
        mul_mem_into_columns(
            m,
            Rbx,
            Mem::new(b, off + 8 * k as i32),
            T[k],
            T[k + 1],
            &format!("{product}[{k}]"),
            k,
        );
    }
    m.mov_zero(LO, "zero for the chain closes (flags preserved)");
    m.adox(T[4], LO, "close the value chain into t4");
    m.adox(T[5], LO, "ripple the t4 close into t5");
    m.adcx(T[5], LO, "close the carry chain into t5");
    m.claim_flags_clear("in-round peak < (T+1)*p*2^64 < 2^325 keeps t5 from wrapping");
}

fn sos_row<M: Machine>(m: &mut M, b: Reg, product: &str) {
    sos_row_at(m, b, 0, product);
}

/// Register roles for `narsil_sosd2_x86`.
pub const SOSD2_REGISTER_MAP: &[(Reg, &str)] = &[
    (Rdi, "z on entry (spilled at once); then lane1 accumulator"),
    (Rsi, "x0 pointer on entry; then reloaded pointer scratch"),
    (
        Rdx,
        "x1 pointer on entry (spilled); the implicit mulx multiplicand",
    ),
    (
        Rcx,
        "y0 pointer on entry; then consts: p at +0..+24, -p^-1 at +32",
    ),
    (R8, "y1 pointer on entry (spilled); then lane0 accumulator"),
    (R9, "consts pointer on entry; then lane0 accumulator"),
    (R10, "lane0 accumulator"),
    (R11, "lane0 accumulator; z pointer in the epilogue"),
    (
        R12,
        "lane0 accumulator (rotates: t_k of round r is L0[(r+k) % 5])",
    ),
    (R13, "lane1 accumulator"),
    (R14, "lane1 accumulator"),
    (R15, "lane1 accumulator"),
    (Rbp, "lane1 accumulator"),
    (
        Rax,
        "low half of the current product; zero for chain closes",
    ),
    (Rbx, "high half of the current product; prologue scratch"),
];

/// Rotating five-word accumulator of sosd2 lane 0 (real part).
const L0: [Reg; 5] = [R8, R9, R10, R11, R12];
/// Rotating five-word accumulator of sosd2 lane 1 (imaginary part).
const L1: [Reg; 5] = [R13, R14, R15, Rbp, Rdi];

/// Five-word window of `bank` at `round`: shift-by-renaming as in the mul
/// kernel, so no data ever moves between rounds.
fn lane_window(bank: [Reg; 5], round: usize) -> [Reg; 5] {
    core::array::from_fn(|k| bank[(round + k) % 5])
}

/// sosd2 frame layout, all rsp + disp8. Limb values first (ny1 and the y0
/// copy feed `mulx` memory operands directly), spilled pointers after.
const NY1_OFF: i32 = 0;
const Y0_COPY_OFF: i32 = 32;
const X0_PTR_OFF: i32 = 64;
const X1_PTR_OFF: i32 = 72;
const Y1_PTR_OFF: i32 = 80;
const Z_PTR_OFF: i32 = 88;
const SOSD2_FRAME: i32 = 96;

/// One dual-chain accumulate row of a lane window: `t[k] += lo_k` on the
/// value chain, `t[k+1] += hi_k` on the carry chain, source limbs from
/// `[base + off + 8k]`. Entry and exit invariant: CF = OF = 0.
fn sosd2_acc_row<M: Machine>(m: &mut M, t: [Reg; 5], base: Reg, off: i32, product: &str) {
    for k in 0..4 {
        mul_mem_into_columns(
            m,
            Rbx,
            Mem::new(base, off + 8 * k as i32),
            t[k],
            t[k + 1],
            &format!("{product}[{k}]"),
            k,
        );
    }
    m.mov_zero(LO, "zero for the chain closes (flags preserved)");
    m.adox(t[4], LO, "close the value chain into the top word");
    m.claim_flags_clear("in-round peak < 3p*2^64 < 2^320 keeps the top word from wrapping");
}

/// Round-0 row that builds a lane window directly from the products (single
/// adc chain), mirroring the mul kernel's round 0. Leaves CF = 0 but OF
/// undefined. The caller re-seeds with `xor_clear` before any dual-chain row.
fn sosd2_build_row<M: Machine>(m: &mut M, t: [Reg; 5], base: Reg, off: i32, product: &str) {
    let src = |k: usize| Mem::new(base, off + 8 * k as i32);
    m.mulx_mem(t[1], t[0], src(0), &format!("{product}[0] -> (t0, t1)"));
    m.mulx_mem(t[2], LO, src(1), &format!("{product}[1] -> (lo, t2)"));
    m.add(t[1], LO, &format!("t1 += lo({product}[1])"));
    m.mulx_mem(t[3], LO, src(2), &format!("{product}[2] -> (lo, t3)"));
    m.adc(t[2], LO, &format!("t2 += lo({product}[2])"));
    m.mulx_mem(t[4], LO, src(3), &format!("{product}[3] -> (lo, t4)"));
    m.adc(t[3], LO, &format!("t3 += lo({product}[3])"));
    m.adc_zero(t[4], "t4 += chain carry; hi <= 2^64-2 so CF = 0");
}

/// Compute `(x0*y0 - x1*y1, x0*y1 + x1*y0)` with one Montgomery reduction
/// per component. Inputs must be canonical. Five words contain each in-round
/// value because the fixed two-term sum stays below `3p*2^64`.
pub fn sosd2_x86<M: Machine>(m: &mut M) {
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    m.alloc_stack(SOSD2_FRAME);
    m.comment("frame: ny1 +0..24, y0 copy +32..56, x0/x1/y1/z pointers +64..88");
    m.store(Mem::new(Reg::Rsp, Z_PTR_OFF), Rdi, "spill z");
    m.store(Mem::new(Reg::Rsp, X1_PTR_OFF), Rdx, "spill the x1 pointer");
    m.store(Mem::new(Reg::Rsp, Y1_PTR_OFF), R8, "spill the y1 pointer");
    m.store(Mem::new(Reg::Rsp, X0_PTR_OFF), Rsi, "spill the x0 pointer");

    // Accumulator registers double as prologue scratch: rounds start later.
    let scratch = [Rax, Rbx, R10, R11];
    m.comment("copy y0 into the frame: two rows per round read it via rsp");
    for (k, s) in scratch.into_iter().enumerate() {
        m.load(s, Mem::new(Rcx, 8 * k as i32), &format!("y0[{k}]"));
    }
    for (k, s) in scratch.into_iter().enumerate() {
        m.store(
            Mem::new(Reg::Rsp, Y0_COPY_OFF + 8 * k as i32),
            s,
            &format!("y0[{k}]"),
        );
    }
    m.comment("ny1 = p - y1: lane0's subtracted term enters as the negp image");
    for (k, s) in scratch.into_iter().enumerate() {
        m.load(s, Mem::new(R9, 8 * k as i32), &format!("p{k}"));
    }
    for (k, s) in scratch.into_iter().enumerate() {
        let what = format!("p{k} - y1[{k}]");
        if k == 0 {
            m.sub_mem(s, Mem::new(R8, 0), &what);
        } else {
            m.sbb_mem(s, Mem::new(R8, 8 * k as i32), &what);
        }
    }
    for (k, s) in scratch.into_iter().enumerate() {
        m.store(
            Mem::new(Reg::Rsp, NY1_OFF + 8 * k as i32),
            s,
            &format!("ny1[{k}]"),
        );
    }
    m.mov(Rcx, R9, "consts pointer takes rcx (p and -p^-1 table)");

    m.comment("");
    m.comment("round 0: build t = x0[0]*y0 and u = x0[0]*y1 directly");
    let t = lane_window(L0, 0);
    let u = lane_window(L1, 0);
    m.load(MULTIPLIER, Mem::new(Rsi, 0), "x0[0] (rsi still holds x0)");
    sosd2_build_row(m, t, Reg::Rsp, Y0_COPY_OFF, "x0[0]*y0");
    m.load(Rbx, Mem::new(Reg::Rsp, Y1_PTR_OFF), "y1 pointer");
    sosd2_build_row(m, u, Rbx, 0, "x0[0]*y1");
    m.xor_clear(
        LO,
        "clear OF (adc left it undefined) before the dual chains",
    );
    m.load(Rsi, Mem::new(Reg::Rsp, X1_PTR_OFF), "x1 pointer");
    m.load(MULTIPLIER, Mem::new(Rsi, 0), "x1[0]");
    sosd2_acc_row(m, t, Reg::Rsp, NY1_OFF, "x1[0]*ny1");
    sosd2_acc_row(m, u, Reg::Rsp, Y0_COPY_OFF, "x1[0]*y0");
    m.comment("lane0 cancel row");
    cancel_low_word(m, t, Rbx, 0);
    m.claim_flags_clear("cancel row closed both chains under the 2^320 bound");
    m.comment("lane1 cancel row");
    cancel_low_word(m, u, Rbx, 0);
    m.claim_flags_clear("cancel row closed both chains under the 2^320 bound");

    for round in 1..4 {
        let t = lane_window(L0, round);
        let u = lane_window(L1, round);
        m.comment("");
        m.comment(&format!(
            "round {round}: t += x0[{round}]*y0 + x1[{round}]*ny1, u += x0[{round}]*y1 + x1[{round}]*y0"
        ));
        m.load(Rsi, Mem::new(Reg::Rsp, X0_PTR_OFF), "x0 pointer");
        m.load(
            MULTIPLIER,
            Mem::new(Rsi, 8 * round as i32),
            &format!("x0[{round}]"),
        );
        sosd2_acc_row(m, t, Reg::Rsp, Y0_COPY_OFF, &format!("x0[{round}]*y0"));
        m.load(Rsi, Mem::new(Reg::Rsp, Y1_PTR_OFF), "y1 pointer");
        sosd2_acc_row(m, u, Rsi, 0, &format!("x0[{round}]*y1"));
        m.load(Rsi, Mem::new(Reg::Rsp, X1_PTR_OFF), "x1 pointer");
        m.load(
            MULTIPLIER,
            Mem::new(Rsi, 8 * round as i32),
            &format!("x1[{round}]"),
        );
        sosd2_acc_row(m, t, Reg::Rsp, NY1_OFF, &format!("x1[{round}]*ny1"));
        sosd2_acc_row(m, u, Reg::Rsp, Y0_COPY_OFF, &format!("x1[{round}]*y0"));
        m.comment("lane0 cancel row");
        cancel_low_word(m, t, Rbx, round);
        m.claim_flags_clear("cancel row closed both chains under the 2^320 bound");
        m.comment("lane1 cancel row");
        cancel_low_word(m, u, Rbx, round);
        m.claim_flags_clear("cancel row closed both chains under the 2^320 bound");
    }

    m.comment("");
    // Round 3's canceled word 0 drops. In the round-4 renaming frame each
    // lane's result is window words 0..3 and the empty top word is word 4.
    let t = lane_window(L0, 4);
    let u = lane_window(L1, 4);
    m.claim_zero(t[4], "lane0 final value < 1.379p fits four words");
    m.claim_zero(u[4], "lane1 final value < 1.379p fits four words");
    let z = t[4];
    m.load(
        z,
        Mem::new(Reg::Rsp, Z_PTR_OFF),
        "reload z into the freed top word",
    );
    m.comment("final reduction per lane: value < 2p, subtract p once if >= p");
    let keep = [Rax, Rbx, Rsi, MULTIPLIER];
    for (lane, (value, out_off)) in [(t, 0i32), (u, 32i32)].into_iter().enumerate() {
        for (k, (v, s)) in value.iter().zip(keep).enumerate() {
            m.mov(s, *v, &format!("lane{lane}: keep-copy of word {k}"));
        }
        for (k, v) in value.iter().take(4).enumerate() {
            let what = format!("lane{lane}: word {k} -= p{k}");
            if k == 0 {
                m.sub_mem(*v, p_limb(k), &what);
            } else {
                m.sbb_mem(*v, p_limb(k), &what);
            }
        }
        for (k, (v, s)) in value.iter().zip(keep).enumerate() {
            m.cmov_carry(*v, s, &format!("borrow: lane{lane} < p, keep word {k}"));
        }
        for (k, v) in value.iter().take(4).enumerate() {
            m.store(
                Mem::new(z, out_off + 8 * k as i32),
                *v,
                &format!("z[{}]", lane * 4 + k),
            );
        }
    }
    m.free_stack(SOSD2_FRAME);
    for reg in CALLEE_SAVED.iter().rev() {
        m.pop(*reg);
    }
    m.ret();
}

/// Register roles for `narsil_sosd2_small_x86`.
pub const SOSD2_SMALL_REGISTER_MAP: &[(Reg, &str)] = &[
    (
        Rdi,
        "z on entry (spilled); lane1 top accumulator; z again at the end",
    ),
    (Rsi, "x0 pointer on entry; then reloaded pointer scratch"),
    (
        Rdx,
        "x1 pointer on entry (spilled); the implicit mulx multiplicand",
    ),
    (
        Rcx,
        "y0 pointer on entry; then the round cursor (byte offset 8j)",
    ),
    (
        R8,
        "y1 pointer on entry (spilled); then lane0 accumulator t0",
    ),
    (
        R9,
        "consts pointer on entry (spilled); then lane0 accumulator t1",
    ),
    (R10, "lane0 accumulator t2"),
    (R11, "lane0 accumulator t3"),
    (R12, "lane0 accumulator t4 (top word)"),
    (R13, "lane1 accumulator u0"),
    (R14, "lane1 accumulator u1"),
    (R15, "lane1 accumulator u2"),
    (Rbp, "lane1 accumulator u3"),
    (
        Rax,
        "low half of the current product; zero for chain closes",
    ),
    (Rbx, "high half of the current product; prologue scratch"),
];

/// Extra frame slot of the rolled variant: the consts pointer, spilled so
/// rcx can serve as the round cursor.
const CONSTS_PTR_OFF: i32 = 96;
const SOSD2_SMALL_FRAME: i32 = 104;

/// `narsil_sosd2_small_x86`: the rolled, size-compact variant of
/// [`sosd2_x86`] -- same lanes, same contract, same spill-frame design, but
/// the four reduction rounds share one loop body so the text stays op-cache
/// friendly inside the combined pairing loop (the unrolled kernel's 2.4 KiB
/// measurably pressured the frontend there).
///
/// Differences from the unrolled schedule, all forced by the back edge:
///
/// * rcx is the round cursor (byte offset 8j), so the constants table lives
///   in the frame and is reloaded into rsi once per round for the cancels.
/// * the shift-by-renaming becomes four movs plus a flag-safe xor per lane
///   (register names must be iteration-invariant).
/// * round 0 cannot build directly from products, so the accumulators are
///   zeroed up front and the first iteration adds into zeros.
/// * the back edge clobbers CF/OF, so each iteration re-seeds with a
///   flag-cutting xor before its first row.
///
/// Register budget, spill plan and carry bounds are otherwise exactly those
/// of [`sosd2_x86`].
pub fn sosd2_small_x86<M: Machine>(m: &mut M) {
    let cursor = Rcx;
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    m.alloc_stack(SOSD2_SMALL_FRAME);
    m.comment("frame: ny1 +0..24, y0 copy +32..56, x0/x1/y1/z/consts pointers +64..96");
    m.store(Mem::new(Reg::Rsp, Z_PTR_OFF), Rdi, "spill z");
    m.store(Mem::new(Reg::Rsp, X1_PTR_OFF), Rdx, "spill the x1 pointer");
    m.store(Mem::new(Reg::Rsp, Y1_PTR_OFF), R8, "spill the y1 pointer");
    m.store(Mem::new(Reg::Rsp, X0_PTR_OFF), Rsi, "spill the x0 pointer");
    m.store(
        Mem::new(Reg::Rsp, CONSTS_PTR_OFF),
        R9,
        "spill the consts pointer",
    );

    let scratch = [Rax, Rbx, R10, R11];
    m.comment("copy y0 into the frame: two rows per round read it via rsp");
    for (k, s) in scratch.into_iter().enumerate() {
        m.load(s, Mem::new(Rcx, 8 * k as i32), &format!("y0[{k}]"));
    }
    for (k, s) in scratch.into_iter().enumerate() {
        m.store(
            Mem::new(Reg::Rsp, Y0_COPY_OFF + 8 * k as i32),
            s,
            &format!("y0[{k}]"),
        );
    }
    m.comment("ny1 = p - y1: lane0's subtracted term enters as the negp image");
    for (k, s) in scratch.into_iter().enumerate() {
        m.load(s, Mem::new(R9, 8 * k as i32), &format!("p{k}"));
    }
    for (k, s) in scratch.into_iter().enumerate() {
        let what = format!("p{k} - y1[{k}]");
        if k == 0 {
            m.sub_mem(s, Mem::new(R8, 0), &what);
        } else {
            m.sbb_mem(s, Mem::new(R8, 8 * k as i32), &what);
        }
    }
    for (k, s) in scratch.into_iter().enumerate() {
        m.store(
            Mem::new(Reg::Rsp, NY1_OFF + 8 * k as i32),
            s,
            &format!("ny1[{k}]"),
        );
    }
    for (k, t) in L0.into_iter().enumerate() {
        m.xor_clear(t, &format!("t{k} = 0"));
    }
    for (k, u) in L1.into_iter().enumerate() {
        m.xor_clear(u, &format!("u{k} = 0"));
    }
    m.xor_clear(cursor, "byte offset of the round's source limb: 8j = 0");

    m.comment("");
    m.stride_loop(cursor, 8, LoopEnd::Imm(32), ".Lsosd2_round", &mut |m| {
        m.comment("product rows: both lanes, sharing each multiplicand load");
        m.load(Rsi, Mem::new(Reg::Rsp, X0_PTR_OFF), "x0 pointer");
        m.load_indexed(MULTIPLIER, Rsi, cursor, "x0[j], the row multiplicand");
        m.xor_clear(LO, "re-seed CF = OF = 0 (back edge clobbered flags)");
        sosd2_acc_row(m, L0, Reg::Rsp, Y0_COPY_OFF, "x0[j]*y0");
        m.load(Rsi, Mem::new(Reg::Rsp, Y1_PTR_OFF), "y1 pointer");
        sosd2_acc_row(m, L1, Rsi, 0, "x0[j]*y1");
        m.load(Rsi, Mem::new(Reg::Rsp, X1_PTR_OFF), "x1 pointer");
        m.load_indexed(MULTIPLIER, Rsi, cursor, "x1[j]");
        sosd2_acc_row(m, L0, Reg::Rsp, NY1_OFF, "x1[j]*ny1");
        sosd2_acc_row(m, L1, Reg::Rsp, Y0_COPY_OFF, "x1[j]*y0");
        m.load(Rsi, Mem::new(Reg::Rsp, CONSTS_PTR_OFF), "consts pointer");
        m.comment("lane0 cancel row");
        cancel_low_word_at(m, L0, Rbx, Rsi, "");
        m.claim_flags_clear("cancel row closed both chains under the 2^320 bound");
        m.comment("lane1 cancel row");
        cancel_low_word_at(m, L1, Rbx, Rsi, "");
        m.claim_flags_clear("cancel row closed both chains under the 2^320 bound");
        m.claim_zero(L0[0], "the Montgomery factor cancels lane0's low word");
        m.claim_zero(L1[0], "the Montgomery factor cancels lane1's low word");
        m.comment("shift both lanes down one word: the canceled zero drops");
        for k in 0..4 {
            m.mov(L0[k], L0[k + 1], &format!("t{k} = t{}", k + 1));
        }
        m.xor_clear(L0[4], "t4 = 0 (CF/OF stay clear)");
        for k in 0..4 {
            m.mov(L1[k], L1[k + 1], &format!("u{k} = u{}", k + 1));
        }
        m.xor_clear(L1[4], "u4 = 0 (CF/OF stay clear)");
    });

    m.comment("");
    m.load(
        Rcx,
        Mem::new(Reg::Rsp, CONSTS_PTR_OFF),
        "consts pointer back in rcx",
    );
    m.load(
        Rdi,
        Mem::new(Reg::Rsp, Z_PTR_OFF),
        "reload z (lane1's zeroed top word)",
    );
    m.comment("final reduction per lane: value < 2p, subtract p once if >= p");
    let keep = [Rax, Rbx, Rsi, MULTIPLIER];
    let lanes: [([Reg; 4], i32); 2] = [([R8, R9, R10, R11], 0), ([R13, R14, R15, Rbp], 32)];
    for (lane, (value, out_off)) in lanes.into_iter().enumerate() {
        for (k, (v, s)) in value.iter().zip(keep).enumerate() {
            m.mov(s, *v, &format!("lane{lane}: keep-copy of word {k}"));
        }
        for (k, v) in value.iter().enumerate() {
            let what = format!("lane{lane}: word {k} -= p{k}");
            if k == 0 {
                m.sub_mem(*v, p_limb(k), &what);
            } else {
                m.sbb_mem(*v, p_limb(k), &what);
            }
        }
        for (k, (v, s)) in value.iter().zip(keep).enumerate() {
            m.cmov_carry(*v, s, &format!("borrow: lane{lane} < p, keep word {k}"));
        }
        for (k, v) in value.iter().enumerate() {
            m.store(
                Mem::new(Rdi, out_off + 8 * k as i32),
                *v,
                &format!("z[{}]", lane * 4 + k),
            );
        }
    }
    m.free_stack(SOSD2_SMALL_FRAME);
    for reg in CALLEE_SAVED.iter().rev() {
        m.pop(*reg);
    }
    m.ret();
}

/// Register roles for `narsil_f2sqr_x86` and its rolled twin. The two lanes
/// reuse the sosd2 accumulator banks, so the same fifteen roles apply.
pub const F2SQR_REGISTER_MAP: &[(Reg, &str)] = &[
    (
        Rdi,
        "z on entry (spilled at once); then lane1 accumulator; z again in the epilogue",
    ),
    (
        Rsi,
        "x0 pointer on entry; the source cursor of the rolled twin; keep-copy scratch",
    ),
    (
        Rdx,
        "x1 pointer on entry; then the implicit mulx multiplicand",
    ),
    (Rcx, "consts pointer: p at +0..+24, -p^-1 at +32"),
    (
        R8,
        "lane0 accumulator (rotates: t_k of round r is L0[(r+k) % 5])",
    ),
    (R9, "lane0 accumulator"),
    (R10, "lane0 accumulator"),
    (R11, "lane0 accumulator"),
    (R12, "lane0 accumulator"),
    (R13, "lane1 accumulator"),
    (R14, "lane1 accumulator"),
    (R15, "lane1 accumulator"),
    (Rbp, "lane1 accumulator"),
    (
        Rax,
        "low half of the current product; zero for chain closes",
    ),
    (Rbx, "high half of the current product; prologue scratch"),
];

/// f2sqr frame layout, all rsp + disp8. `s` and the `x0` copy are adjacent
/// and stride-8 apart so the rolled twin walks both multiplicands with one
/// cursor.
const F2S_S: i32 = 0;
const F2S_X0: i32 = 32;
const F2S_D: i32 = 64;
const F2S_TWO_X1: i32 = 96;
const F2S_Z: i32 = 128;
const F2S_END: i32 = 136;
const F2SQR_FRAME: i32 = 144;

/// Prologue scratch: the four value words and their pre-subtraction copies.
const F2S_V: [Reg; 4] = [R8, R9, R10, R11];
const F2S_KEEP: [Reg; 4] = [R12, R13, R14, R15];

/// `[rsp + out .. +24] = a + b mod p` for operands at most p. Both sources
/// are memory operands and may be the same. `a + b < 2p < 2^255` fits four
/// words, so no fifth word is needed and one guarded subtraction is exact.
fn f2sqr_add_mod_store<M: Machine>(m: &mut M, a: Mem, b: Mem, out: i32, what: &str) {
    let at = |base: Mem, k: usize| Mem::new(base.base, base.offset + 8 * k as i32);
    for (k, v) in F2S_V.iter().enumerate() {
        m.load(*v, at(a, k), &format!("{what}: left limb {k}"));
    }
    for (k, v) in F2S_V.iter().enumerate() {
        let text = format!("{what}: += right limb {k}");
        if k == 0 {
            m.add_mem(*v, at(b, k), &text);
        } else {
            m.adc_mem(*v, at(b, k), &text);
        }
    }
    for (k, (v, keep)) in F2S_V.iter().zip(F2S_KEEP).enumerate() {
        m.mov(keep, *v, &format!("{what}: keep-copy of limb {k}"));
    }
    for (k, v) in F2S_V.iter().enumerate() {
        let text = format!("{what}: limb {k} -= p{k}");
        if k == 0 {
            m.sub_mem(*v, p_limb(k), &text);
        } else {
            m.sbb_mem(*v, p_limb(k), &text);
        }
    }
    for (k, (v, keep)) in F2S_V.iter().zip(F2S_KEEP).enumerate() {
        m.cmov_carry(
            *v,
            keep,
            &format!("{what}: borrow means below p, keep limb {k}"),
        );
    }
    for (k, v) in F2S_V.iter().enumerate() {
        m.store(
            Mem::new(Reg::Rsp, out + 8 * k as i32),
            *v,
            &format!("{what}[{k}]"),
        );
    }
}

/// `[rsp + out .. +24] = p - x1`, the negp image that turns the difference
/// `x0 - x1` into an addition. Maps zero to p, which the guarded addition
/// above accepts (its operand bound is at most p).
fn f2sqr_negp_store<M: Machine>(m: &mut M, x1: Reg, out: i32) {
    for (k, v) in F2S_V.iter().enumerate() {
        m.load(*v, p_limb(k), &format!("p{k}"));
    }
    for (k, v) in F2S_V.iter().enumerate() {
        let text = format!("p{k} - x1[{k}]");
        if k == 0 {
            m.sub_mem(*v, Mem::new(x1, 0), &text);
        } else {
            m.sbb_mem(*v, Mem::new(x1, 8 * k as i32), &text);
        }
    }
    for (k, v) in F2S_V.iter().enumerate() {
        m.store(
            Mem::new(Reg::Rsp, out + 8 * k as i32),
            *v,
            &format!("p - x1 [{k}]"),
        );
    }
}

/// Stage the operands both lanes consume: `s = x0 + x1`, `d = x0 - x1` and
/// `2*x1`, all canonical. `copy_x0` also mirrors `x0` into the frame, which
/// only the rolled twin needs (it walks both multiplicands with one cursor).
fn f2sqr_prologue<M: Machine>(m: &mut M, copy_x0: bool) {
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    m.alloc_stack(F2SQR_FRAME);
    m.comment("frame: s +0, x0 copy +32, d +64, 2*x1 +96, z +128, walk bound +136");
    m.store(Mem::new(Reg::Rsp, F2S_Z), Rdi, "spill z");
    if copy_x0 {
        m.comment("copy x0 into the frame, one stride above s");
        for (k, v) in F2S_V.iter().enumerate() {
            m.load(*v, Mem::new(Rsi, 8 * k as i32), &format!("x0[{k}]"));
        }
        for (k, v) in F2S_V.iter().enumerate() {
            m.store(
                Mem::new(Reg::Rsp, F2S_X0 + 8 * k as i32),
                *v,
                &format!("x0[{k}]"),
            );
        }
    }
    f2sqr_add_mod_store(m, Mem::new(Rsi, 0), Mem::new(Rdx, 0), F2S_S, "s = x0 + x1");
    f2sqr_add_mod_store(
        m,
        Mem::new(Rdx, 0),
        Mem::new(Rdx, 0),
        F2S_TWO_X1,
        "2*x1 (lane1 doubles here, not after the reduction)",
    );
    f2sqr_negp_store(m, Rdx, F2S_D);
    f2sqr_add_mod_store(
        m,
        Mem::new(Rsi, 0),
        Mem::new(Reg::Rsp, F2S_D),
        F2S_D,
        "d = x0 - x1",
    );
}

/// One dual-chain product row of an f2sqr lane: `t[k] += lo_k` on the value
/// chain, `t[k+1] += hi_k` on the carry chain, source limbs from
/// `[rsp + off + 8k]`. Entry and exit invariant: CF = OF = 0.
fn f2sqr_acc_row<M: Machine>(m: &mut M, t: [Reg; 5], off: i32, product: &str) {
    for k in 0..4 {
        mul_mem_into_columns(
            m,
            Rbx,
            Mem::new(Reg::Rsp, off + 8 * k as i32),
            t[k],
            t[k + 1],
            &format!("{product}[{k}]"),
            k,
        );
    }
    m.mov_zero(LO, "zero for the chain closes (flags preserved)");
    m.adox(t[4], LO, "close the value chain into the top word");
    m.claim_flags_clear("single-product round peak < 2p + 2^64*p < 2^320");
}

/// Canonicalize both lanes and store `z[0..4] = lane0`, `z[4..8] = lane1`.
/// `z` is reloaded into the freed top word of lane0.
fn f2sqr_store_lanes<M: Machine>(m: &mut M, lane0: [Reg; 4], lane1: [Reg; 4], z: Reg) {
    m.load(z, Mem::new(Reg::Rsp, F2S_Z), "reload z into the freed word");
    m.comment("final reduction per lane: value < 2p, subtract p once if >= p");
    let keep = [Rax, Rbx, Rsi, MULTIPLIER];
    for (lane, (value, out_off)) in [(lane0, 0i32), (lane1, 32i32)].into_iter().enumerate() {
        for (k, (v, s)) in value.iter().zip(keep).enumerate() {
            m.mov(s, *v, &format!("lane{lane}: keep-copy of word {k}"));
        }
        for (k, v) in value.iter().enumerate() {
            let what = format!("lane{lane}: word {k} -= p{k}");
            if k == 0 {
                m.sub_mem(*v, p_limb(k), &what);
            } else {
                m.sbb_mem(*v, p_limb(k), &what);
            }
        }
        for (k, (v, s)) in value.iter().zip(keep).enumerate() {
            m.cmov_carry(*v, s, &format!("borrow: lane{lane} < p, keep word {k}"));
        }
        for (k, v) in value.iter().enumerate() {
            m.store(
                Mem::new(z, out_off + 8 * k as i32),
                *v,
                &format!("z[{}]", lane * 4 + k),
            );
        }
    }
    m.free_stack(F2SQR_FRAME);
    for reg in CALLEE_SAVED.iter().rev() {
        m.pop(*reg);
    }
    m.ret();
}

/// `narsil_f2sqr_x86`: the complex Fp2 square `(x0 + x1*u)^2` over
/// `Fp2 = Fp[u]/(u^2 + 1)` as one leaf with two interleaved Montgomery
/// chains -- `lane0 = (x0 + x1)(x0 - x1)/R`, `lane1 = x0*(2*x1)/R`.
///
/// Both operand pairs are canonicalized in the prologue, so each lane is a
/// single-product CIOS reduction with exactly the [`mont4_mul`] carry bound:
/// `t < 2p` at every round boundary, `t + a*b_j < 2p + 2^64*p` after a
/// product row and below `2^65*p < 2^320` after the cancel row. The five-word
/// window therefore absorbs every carry in both lanes.
///
/// Two sequential `mont4_mul` calls compute the same value with the same
/// product count, but their CIOS chains are strictly serial. Interleaving the
/// lanes here keeps both dependent chains in flight without adding one
/// multiply.
pub fn f2sqr_x86<M: Machine>(m: &mut M) {
    f2sqr_prologue(m, false);

    m.comment("");
    m.comment("round 0: build t = s[0]*d and u = x0[0]*2x1 directly");
    let t = lane_window(L0, 0);
    let u = lane_window(L1, 0);
    m.load(MULTIPLIER, Mem::new(Reg::Rsp, F2S_S), "s[0]");
    sosd2_build_row(m, t, Reg::Rsp, F2S_D, "s[0]*d");
    m.load(MULTIPLIER, Mem::new(Rsi, 0), "x0[0]");
    sosd2_build_row(m, u, Reg::Rsp, F2S_TWO_X1, "x0[0]*2x1");
    m.xor_clear(
        LO,
        "clear OF (adc left it undefined) before the dual chains",
    );
    m.comment("lane0 cancel row");
    cancel_low_word(m, t, Rbx, 0);
    m.claim_flags_clear("cancel row closed both chains under the 2^320 bound");
    m.comment("lane1 cancel row");
    cancel_low_word(m, u, Rbx, 0);
    m.claim_flags_clear("cancel row closed both chains under the 2^320 bound");

    for round in 1..4 {
        let t = lane_window(L0, round);
        let u = lane_window(L1, round);
        m.comment("");
        m.comment(&format!(
            "round {round}: t += s[{round}]*d, u += x0[{round}]*2x1"
        ));
        m.load(
            MULTIPLIER,
            Mem::new(Reg::Rsp, F2S_S + 8 * round as i32),
            &format!("s[{round}]"),
        );
        f2sqr_acc_row(m, t, F2S_D, &format!("s[{round}]*d"));
        m.load(
            MULTIPLIER,
            Mem::new(Rsi, 8 * round as i32),
            &format!("x0[{round}]"),
        );
        f2sqr_acc_row(m, u, F2S_TWO_X1, &format!("x0[{round}]*2x1"));
        m.comment("lane0 cancel row");
        cancel_low_word(m, t, Rbx, round);
        m.claim_flags_clear("cancel row closed both chains under the 2^320 bound");
        m.comment("lane1 cancel row");
        cancel_low_word(m, u, Rbx, round);
        m.claim_flags_clear("cancel row closed both chains under the 2^320 bound");
    }

    m.comment("");
    // Round 3's canceled word 0 drops. In the round-4 renaming frame each
    // lane's result is window words 0..3 and the empty top word is word 4.
    let t = lane_window(L0, 4);
    let u = lane_window(L1, 4);
    m.claim_zero(t[4], "the Montgomery factor canceled lane0's low word");
    m.claim_zero(u[4], "the Montgomery factor canceled lane1's low word");
    let lane0: [Reg; 4] = core::array::from_fn(|k| t[k]);
    let lane1: [Reg; 4] = core::array::from_fn(|k| u[k]);
    f2sqr_store_lanes(m, lane0, lane1, t[4]);
}

/// `narsil_f2sqr_small_x86`: the rolled twin of [`f2sqr_x86`] -- same lanes,
/// same contract, same bounds, one shared round body so the text stays
/// op-cache compact inside the Miller loop.
///
/// Differences forced by the back edge, exactly as in [`sosd2_small_x86`]:
/// rsi walks `&s[j]` (the `x0` copy sits 32 bytes above it, so one cursor
/// feeds both multiplicands), the accumulators start at zero because round 0
/// cannot build directly from products, the shift-by-renaming becomes four
/// movs plus a flag-safe xor per lane, and each iteration re-seeds CF and OF.
pub fn f2sqr_small_x86<M: Machine>(m: &mut M) {
    f2sqr_prologue(m, true);

    m.comment("");
    m.comment("one cursor for both multiplicands: s[j] at +0, x0[j] at +32");
    m.mov(Rsi, Reg::Rsp, "");
    m.add_imm(Rsi, F2S_S, "cursor = &s[0]");
    m.mov(Rax, Rsi, "");
    m.add_imm(Rax, 32, "walk bound = &s[4]");
    m.store(Mem::new(Reg::Rsp, F2S_END), Rax, "walk bound");
    for (k, t) in L0.into_iter().enumerate() {
        m.xor_clear(t, &format!("t{k} = 0"));
    }
    for (k, u) in L1.into_iter().enumerate() {
        m.xor_clear(u, &format!("u{k} = 0"));
    }

    m.comment("");
    m.stride_loop(
        Rsi,
        8,
        LoopEnd::Mem(Mem::new(Reg::Rsp, F2S_END)),
        ".Lf2sqr_round",
        &mut |m| {
            m.load(MULTIPLIER, Mem::new(Rsi, 0), "s[j], lane0's multiplicand");
            m.xor_clear(LO, "re-seed CF = OF = 0 (back edge clobbered flags)");
            f2sqr_acc_row(m, L0, F2S_D, "s[j]*d");
            m.load(MULTIPLIER, Mem::new(Rsi, 32), "x0[j], lane1's multiplicand");
            f2sqr_acc_row(m, L1, F2S_TWO_X1, "x0[j]*2x1");
            m.comment("lane0 cancel row");
            cancel_low_word_at(m, L0, Rbx, CONSTS, "");
            m.claim_flags_clear("cancel row closed both chains under the 2^320 bound");
            m.comment("lane1 cancel row");
            cancel_low_word_at(m, L1, Rbx, CONSTS, "");
            m.claim_flags_clear("cancel row closed both chains under the 2^320 bound");
            m.claim_zero(L0[0], "the Montgomery factor cancels lane0's low word");
            m.claim_zero(L1[0], "the Montgomery factor cancels lane1's low word");
            m.comment("shift both lanes down one word: the canceled zero drops");
            for k in 0..4 {
                m.mov(L0[k], L0[k + 1], &format!("t{k} = t{}", k + 1));
            }
            m.xor_clear(L0[4], "t4 = 0 (CF/OF stay clear)");
            for k in 0..4 {
                m.mov(L1[k], L1[k + 1], &format!("u{k} = u{}", k + 1));
            }
            m.xor_clear(L1[4], "u4 = 0 (CF/OF stay clear)");
        },
    );

    m.comment("");
    m.claim_zero(L1[4], "lane1's top word is the shift's cleared slot");
    f2sqr_store_lanes(
        m,
        [L0[0], L0[1], L0[2], L0[3]],
        [L1[0], L1[1], L1[2], L1[3]],
        L1[4],
    );
}

/// Register roles for `narsil_g2_ysqr_x86`. The product walk holds a
/// six-word raw accumulator plus three frame pointers, the reduction walk an
/// eight-word double-width value plus the two Montgomery scratch halves, so
/// no computed state ever leaves a register.
pub const G2_YSQR_REGISTER_MAP: &[(Reg, &str)] = &[
    (
        Rdi,
        "z on entry (spilled); product destination; output pointer of a reduction row",
    ),
    (Rsi, "g on entry; product multiplicand row; mask scratch"),
    (
        Rdx,
        "e on entry (moved out at once: rdx is the implicit mulx multiplicand)",
    ),
    (Rcx, "f on entry; product multiplier row; subtrahend row"),
    (R8, "accumulator word 0 / double-width word 0"),
    (R9, "accumulator word 1 / double-width word 1"),
    (R10, "accumulator word 2 / double-width word 2"),
    (R11, "accumulator word 3 / double-width word 3"),
    (
        R12,
        "accumulator word 4 / double-width word 4, result word 0",
    ),
    (
        R13,
        "accumulator word 5 / double-width word 5, result word 1",
    ),
    (
        R14,
        "double-width word 6, result word 2; prologue mask scratch",
    ),
    (
        R15,
        "double-width word 7, result word 3; prologue mask scratch",
    ),
    (Rbp, "e pointer after entry; then the walk row cursor"),
    (
        Rax,
        "low half of the current product; zero for chain closes; borrow mask",
    ),
    (Rbx, "high half of the current product; prologue scratch"),
];

/// The bound the whole lazy schedule rests on: p occupies fewer than 254
/// bits, so `4p < 2^256` and in particular `2p < 2^256`. Uncorrected
/// four-limb sums of canonical values cannot carry out, and every product of
/// a sub-2p value by a sub-p value stays below `p*2^256`, the Montgomery
/// reduction's precondition. This is mcl's `isFullBit == false` (indeed its
/// stronger `isLtQuad`) for BN254. A modulus that broke it would silently
/// corrupt every product below.
pub const G2_YSQR_MODULUS_BOUND: () = assert!(
    super::BN254_P[3] < (1u64 << 62),
    "g2_ysqr needs 4p < 2^256: the top limb of p must stay below 2^62"
);

// g2_ysqr frame layout, all rsp-relative. P at +0 and -p^-1 at +32 mirror
// the consts-table shape, so `cancel_low_word_at` and `fsq_masked_p` address
// the frame as their constants table.
const G2Y_P: i32 = 0;
const G2Y_PINV: i32 = 32;
const G2Y_Z: i32 = 40;
const G2Y_TBL: i32 = 48;
const G2Y_END: i32 = 56;
/// Eight 32-byte product operands. `gs`, `g2`, `es` and `e6` are raw sums
/// below 2p (mcl's `addPre` shape). `gd` and `ed` are canonical.
const G2Y_GS: i32 = 64;
const G2Y_GD: i32 = 96;
const G2Y_G0: i32 = 128;
const G2Y_G2: i32 = 160;
const G2Y_ES: i32 = 192;
const G2Y_ED: i32 = 224;
const G2Y_E0: i32 = 256;
const G2Y_E6: i32 = 288;
/// Four 64-byte raw products, held unreduced until the combining walk.
const G2Y_D: i32 = 320;
const G2Y_FRAME: i32 = 576;

/// Byte offsets of the two table regions inside the g2_ysqr rodata blob.
const G2Y_TB_PROD: i32 = 0; // 4 rows x 3
const G2Y_TB_MOD: i32 = 96; // 2 rows x 3
const G2Y_TB_END: i32 = 144;

const G2Y_TAB_LABEL: &str = ".Lg2y_tab";

/// The g2_ysqr walk tables (rsp-relative frame offsets).
///
/// Product rows are `(x, y, destination)`. Reduction rows are
/// `(minuend, subtrahend, z byte offset)`: each subtracts one raw product
/// from another at double width and Montgomery-reduces the difference.
fn g2_ysqr_tables() -> Vec<u64> {
    let mut t: Vec<u64> = Vec::new();
    let row3 = |t: &mut Vec<u64>, a: i32, b: i32, c: i32| {
        t.extend([a as u64, b as u64, c as u64]);
    };
    let d = |k: i32| G2Y_D + 64 * k;
    for (x, y, dst) in [
        (G2Y_GS, G2Y_GD, d(0)),
        (G2Y_ES, G2Y_ED, d(1)),
        (G2Y_G0, G2Y_G2, d(2)),
        (G2Y_E0, G2Y_E6, d(3)),
    ] {
        row3(&mut t, x, y, dst);
    }
    assert_eq!(t.len() * 8, G2Y_TB_MOD as usize);
    row3(&mut t, d(0), d(1), 0);
    row3(&mut t, d(2), d(3), 32);
    assert_eq!(t.len() * 8, G2Y_TB_END as usize);
    t
}

/// Load the walk table base into `cursor` and set the bound slot.
fn g2y_walk_setup<M: Machine>(m: &mut M, offset: i32, bytes: i32) {
    let rsp = Reg::Rsp;
    m.load(Rbp, Mem::new(rsp, G2Y_TBL), "table base");
    m.add_imm(Rbp, offset, "walk start");
    m.mov(Rax, Rbp, "");
    m.add_imm(Rax, bytes, "walk end");
    m.store(Mem::new(rsp, G2Y_END), Rax, "walk bound");
}

/// `[rsp + dst] = [a] + [b]` as a raw four-limb sum, no reduction. Sound
/// only where both operands are canonical: `2p < 2^256` for BN254, so the
/// sum keeps four limbs (mcl's `!isFullBit` `addPre`).
fn g2y_add_pre<M: Machine>(m: &mut M, dst: i32, a: Mem, b: Mem, what: &str) {
    let v = [R8, R9, R10, R11];
    for (k, reg) in v.into_iter().enumerate() {
        m.load(reg, Mem::new(a.base, a.offset + 8 * k as i32), what);
    }
    for (k, reg) in v.into_iter().enumerate() {
        let addend = Mem::new(b.base, b.offset + 8 * k as i32);
        if k == 0 {
            m.add_mem(reg, addend, what);
        } else {
            m.adc_mem(reg, addend, what);
        }
    }
    m.claim_flags_clear("canonical + canonical < 2p < 2^256: no carry out");
    for (k, reg) in v.into_iter().enumerate() {
        m.store(Mem::new(Reg::Rsp, dst + 8 * k as i32), reg, what);
    }
}

/// `[rsp + dst] = [a]`, four limbs.
fn g2y_copy<M: Machine>(m: &mut M, dst: i32, a: Mem, what: &str) {
    let v = [R8, R9, R10, R11];
    for (k, reg) in v.into_iter().enumerate() {
        m.load(reg, Mem::new(a.base, a.offset + 8 * k as i32), what);
    }
    for (k, reg) in v.into_iter().enumerate() {
        m.store(Mem::new(Reg::Rsp, dst + 8 * k as i32), reg, what);
    }
}

/// `[rsp + dst] = [a] - [b] mod p`, both operands canonical.
fn g2y_sub_mod<M: Machine>(m: &mut M, dst: i32, a: Mem, b: Mem, what: &str) {
    let v = [R8, R9, R10, R11];
    for (k, reg) in v.into_iter().enumerate() {
        m.load(reg, Mem::new(a.base, a.offset + 8 * k as i32), what);
    }
    m.xor_clear(Rax, "mask seed; also clears flags for the chain");
    for (k, reg) in v.into_iter().enumerate() {
        let sub = Mem::new(b.base, b.offset + 8 * k as i32);
        if k == 0 {
            m.sub_mem(reg, sub, what);
        } else {
            m.sbb_mem(reg, sub, what);
        }
    }
    m.sbb_rr(Rax, Rax, "mask = -borrow");
    fsq_masked_p(m, R12, Rax, 0);
    fsq_masked_p(m, R13, Rax, 1);
    fsq_masked_p(m, R14, Rax, 2);
    fsq_masked_p(m, Rax, Rax, 3);
    m.add(R8, R12, "borrow: += p, limb 0");
    m.adc(R9, R13, "limb 1");
    m.adc(R10, R14, "limb 2");
    m.adc(R11, Rax, "limb 3");
    for (k, reg) in v.into_iter().enumerate() {
        m.store(Mem::new(Reg::Rsp, dst + 8 * k as i32), reg, what);
    }
}

/// One raw 4x4 product row: `[rsp + row.dst] = [rsp + row.x] * [rsp + row.y]`
/// as an exact 512-bit value. Four rounds, the six-word accumulator rotating
/// by renaming, so the round that finalizes a limb costs no data movement.
fn g2y_mulpre_row<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    m.mov(Rsi, rsp, "");
    m.add_mem(Rsi, Mem::new(Rbp, 0), "x = rsp + row.x");
    m.mov(Rcx, rsp, "");
    m.add_mem(Rcx, Mem::new(Rbp, 8), "y = rsp + row.y");
    m.mov(Rdi, rsp, "");
    m.add_mem(Rdi, Mem::new(Rbp, 16), "destination = rsp + row.dst");
    let acc = [R8, R9, R10, R11, R12, R13];
    for (k, reg) in acc.into_iter().enumerate() {
        m.xor_clear(reg, &format!("t{k} = 0"));
    }
    for j in 0..4 {
        let w: [Reg; 6] = core::array::from_fn(|i| acc[(j + i) % 6]);
        m.load(MULTIPLIER, Mem::new(Rsi, 8 * j as i32), &format!("x[{j}]"));
        m.xor_clear(LO, "seed CF = OF = 0 for this round");
        for k in 0..4 {
            mul_mem_into_columns(
                m,
                Rbx,
                Mem::new(Rcx, 8 * k as i32),
                w[k],
                w[k + 1],
                &format!("x[{j}]*y[{k}]"),
                k,
            );
        }
        m.mov_zero(LO, "zero for the chain closes (flags preserved)");
        m.adox(w[4], LO, "close the value chain into t4");
        m.adox(w[5], LO, "ripple the t4 close into t5");
        m.adcx(w[5], LO, "close the carry chain into t5");
        m.claim_flags_clear("window < 2^257 plus a 2^320 product stays under 2^384");
        m.store(
            Mem::new(Rdi, 8 * j as i32),
            w[0],
            &format!("product limb {j} is final"),
        );
        m.xor_clear(w[0], "the emptied word becomes the next round's top");
    }
    let w: [Reg; 6] = core::array::from_fn(|i| acc[(4 + i) % 6]);
    for (k, reg) in w[..4].iter().enumerate() {
        m.store(
            Mem::new(Rdi, 32 + 8 * k as i32),
            *reg,
            &format!("product limb {}", k + 4),
        );
    }
}

/// One combining row: guarded double-width difference of two raw products,
/// then the single Montgomery reduction of that difference. The 512-bit
/// value never leaves R8..R15 between the two halves of the row.
fn g2y_sub_mod_row<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    m.mov(Rsi, rsp, "");
    m.add_mem(Rsi, Mem::new(Rbp, 0), "minuend = rsp + row.a");
    m.mov(Rcx, rsp, "");
    m.add_mem(Rcx, Mem::new(Rbp, 8), "subtrahend = rsp + row.b");
    let v = [R8, R9, R10, R11, R12, R13, R14, R15];
    for (k, reg) in v.into_iter().enumerate() {
        m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("T{k}"));
    }
    m.xor_clear(Rax, "mask seed; also clears flags for the chain");
    for (k, reg) in v.into_iter().enumerate() {
        let sub = Mem::new(Rcx, 8 * k as i32);
        if k == 0 {
            m.sub_mem(reg, sub, "word 0 -= subtrahend");
        } else {
            m.sbb_mem(reg, sub, &format!("word {k} -= subtrahend"));
        }
    }
    m.sbb_rr(Rax, Rax, "mask = -borrow");
    fsq_masked_p(m, Rbx, Rax, 0);
    fsq_masked_p(m, MULTIPLIER, Rax, 1);
    fsq_masked_p(m, Rsi, Rax, 2);
    fsq_masked_p(m, Rax, Rax, 3);
    m.add(R12, Rbx, "borrow: high half += p (mod p*2^256), word 4");
    m.adc(R13, MULTIPLIER, "word 5");
    m.adc(R14, Rsi, "word 6");
    m.adc(R15, Rax, "word 7");
    // The fix-up carries out exactly when it fired, cancelling the borrow.
    // Both branches leave a value below p*2^256, the REDC precondition.
    m.comment("Montgomery reduction of the difference, still in registers");
    m.xor_clear(LO, "clear CF = OF before the dual chains");
    for round in 0..4 {
        if round > 0 {
            m.claim_flags_clear("the previous cancel row rippled out below 2p*2^256");
        }
        let window: [Reg; 5] = core::array::from_fn(|k| v[round + k]);
        cancel_low_word_at(m, window, Rbx, rsp, &round.to_string());
        for (k, &word) in v.iter().enumerate().skip(round + 5) {
            m.adcx(word, LO, &format!("T{k} += carry-chain ripple"));
            m.adox(word, LO, &format!("T{k} += value-chain ripple"));
        }
    }
    m.claim_flags_clear("T < p*2^256 keeps the total below 2p*2^256: no word beyond T7");
    m.load(Rdi, Mem::new(rsp, G2Y_Z), "z");
    m.add_mem(Rdi, Mem::new(Rbp, 16), "+ this lane's byte offset");
    m.comment("result T4..T7 < 2p: one conditional subtraction");
    fsq_csub_store(m, [R12, R13, R14, R15], [Rax, Rbx, MULTIPLIER, Rcx], Rdi, 0);
}

/// `narsil_g2_ysqr_x86`: the Miller doubling step's `y = g^2 - 3*e^2` over
/// Fp2, in mcl's lazy double-width shape -- 4 raw 4x4 products and 2
/// Montgomery reductions where the composed route pays 4 products and 4
/// reductions plus three Fp2 helper round trips.
///
/// # Semantics
///
/// For canonical Fp2 operands `g`, `e` and `f = 3e` (the doubling step
/// already holds `f`), with `R = 2^256` the Montgomery radix:
///
/// * `g^2 = ((g.re + g.im)(g.re - g.im), g.re*(2*g.im)) / R`, the complex
///   square. Both halves are taken as raw 512-bit products.
/// * `3*e^2 = ((e.re + e.im)*(f.re - f.im), e.re*(2*f.im)) / R`: the tripling
///   folds into the second operand of each product, since `f.re - f.im =
///   3(e.re - e.im)` and `2*f.im = 6*e.im`. No double-width tripling exists.
/// * Each output half is ONE guarded 512-bit subtraction followed by ONE
///   Montgomery reduction. The squares themselves are never reduced.
///
/// # Bounds (BN254: p < 2^254, so 2p < 2^255 and 4p < 2^256)
///
/// The whole schedule rests on p being below the top bit of its four-limb
/// representation -- mcl's `isFullBit == false`, which for BN254 holds with
/// two bits to spare (`bitLength(p) = 254 <= 4*64 - 2`). `G2_YSQR_MODULUS_BOUND`
/// pins it, `kernelgen_verify` asserts every stage below on random and
/// adversarial residues.
///
/// * raw sums `gs`, `g2`, `es`, `e6` are `addPre`: two canonical values add
///   to less than `2p < 2^255`, so four limbs hold the sum with no
///   reduction. Only `2p < 2^256` is needed; the spare bit is headroom.
/// * `gd` and `ed` are canonical (a guarded modular subtraction), so every
///   product multiplies a sub-2p value by a sub-p value: all four raw
///   products stay below `2p^2`.
/// * `2p^2 < p*R` because `2p < R`: each product is already a valid REDC
///   input, and so is a guarded difference of two of them. The guarded
///   subtraction adds `p*R` on borrow, which is `0 mod p`, so the reduced
///   value is exact.
/// * `T < p*R` gives `(T + m*p*R)/R < 2p`: one conditional subtraction
///   returns the canonical residue.
///
/// # Structure
///
/// The prologue builds the eight product operands straight from the three
/// argument pointers. Two rolled walks follow -- four product rows, then two
/// combining rows -- so each body is emitted once and the kernel's text stays
/// inside the Miller loop's front-end budget.
///
/// Arguments: `(z: *mut u64x8, g: *const u64x8, e: *const u64x8,
/// f: *const u64x8, consts: *const { p[4], -p^-1 })` in rdi, rsi, rdx, rcx,
/// r8. All inputs canonical, `f = 3e`. Outputs canonical. `z` must not alias
/// `g`, `e` or `f`.
pub fn g2_ysqr_x86<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    m.rodata(G2Y_TAB_LABEL, &g2_ysqr_tables());
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    m.alloc_stack(G2Y_FRAME);
    m.comment(
        "frame: p +0, -p^-1 +32, z +40, table base +48, walk bound +56, product operands +64, raw products +320",
    );
    m.store(Mem::new(rsp, G2Y_Z), Rdi, "spill z");
    for k in 0..4 {
        m.load(Rax, Mem::new(R8, 8 * k), &format!("p{k}"));
        m.store(
            Mem::new(rsp, G2Y_P + 8 * k),
            Rax,
            "cancel and mask rows address the frame as a consts table",
        );
    }
    m.load(Rax, Mem::new(R8, 32), "-p^-1");
    m.store(Mem::new(rsp, G2Y_PINV), Rax, "-p^-1");
    m.lea_rodata(Rax, G2Y_TAB_LABEL, "walk tables");
    m.store(Mem::new(rsp, G2Y_TBL), Rax, "table base");
    m.mov(
        Rbp,
        MULTIPLIER,
        "e pointer: rdx is the implicit mulx multiplicand",
    );

    m.comment("");
    m.comment("product operands. The four sums are uncorrected (mcl addPre):");
    m.comment("p below the top bit is exactly what makes four limbs enough");
    g2y_add_pre(
        m,
        G2Y_GS,
        Mem::new(Rsi, 0),
        Mem::new(Rsi, 32),
        "gs = g.re + g.im",
    );
    g2y_add_pre(
        m,
        G2Y_G2,
        Mem::new(Rsi, 32),
        Mem::new(Rsi, 32),
        "g2 = 2*g.im",
    );
    g2y_add_pre(
        m,
        G2Y_ES,
        Mem::new(Rbp, 0),
        Mem::new(Rbp, 32),
        "es = e.re + e.im",
    );
    g2y_add_pre(
        m,
        G2Y_E6,
        Mem::new(Rcx, 32),
        Mem::new(Rcx, 32),
        "e6 = 2*f.im = 6*e.im",
    );
    g2y_copy(m, G2Y_G0, Mem::new(Rsi, 0), "g.re");
    g2y_copy(m, G2Y_E0, Mem::new(Rbp, 0), "e.re");
    g2y_sub_mod(
        m,
        G2Y_GD,
        Mem::new(Rsi, 0),
        Mem::new(Rsi, 32),
        "gd = g.re - g.im",
    );
    g2y_sub_mod(
        m,
        G2Y_ED,
        Mem::new(Rcx, 0),
        Mem::new(Rcx, 32),
        "ed = f.re - f.im = 3*(e.re - e.im)",
    );

    m.comment("");
    m.comment("four raw 4x4 products, none reduced");
    g2y_walk_setup(m, G2Y_TB_PROD, 4 * 24);
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, G2Y_END)),
        ".Lg2y_prod",
        &mut |m| g2y_mulpre_row(m),
    );

    m.comment("");
    m.comment("two combining rows: one guarded 512-bit difference and one");
    m.comment("Montgomery reduction per output Fp");
    g2y_walk_setup(m, G2Y_TB_MOD, 2 * 24);
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, G2Y_END)),
        ".Lg2y_mod",
        &mut |m| g2y_sub_mod_row(m),
    );

    m.free_stack(G2Y_FRAME);
    for reg in CALLEE_SAVED.iter().rev() {
        m.pop(*reg);
    }
    m.ret();
}

/// Register roles for `narsil_f2mul_x86`. Same two walks as
/// [`g2_ysqr_x86`], so the roles are the same: a six-word raw accumulator
/// and three frame pointers in the product walk, an eight-word double-width
/// value plus the two Montgomery scratch halves in the reduction walk.
pub const F2MUL_REGISTER_MAP: &[(Reg, &str)] = &[
    (
        Rdi,
        "z on entry (spilled); product destination; output pointer of a reduction row",
    ),
    (Rsi, "x on entry; product multiplicand row; mask scratch"),
    (
        Rdx,
        "y on entry (moved out at once: rdx is the implicit mulx multiplicand)",
    ),
    (
        Rcx,
        "consts on entry; product multiplier row; subtrahend row",
    ),
    (R8, "accumulator word 0 / double-width word 0"),
    (R9, "accumulator word 1 / double-width word 1"),
    (R10, "accumulator word 2 / double-width word 2"),
    (R11, "accumulator word 3 / double-width word 3"),
    (
        R12,
        "accumulator word 4 / double-width word 4, result word 0",
    ),
    (
        R13,
        "accumulator word 5 / double-width word 5, result word 1",
    ),
    (R14, "double-width word 6, result word 2; staging scratch"),
    (R15, "double-width word 7, result word 3; staging scratch"),
    (Rbp, "y pointer after entry; then the walk row cursor"),
    (
        Rax,
        "low half of the current product; zero for chain closes; borrow mask",
    ),
    (Rbx, "high half of the current product; prologue scratch"),
];

// f2mul reuses the g2_ysqr frame header verbatim: `cancel_low_word_at`,
// `fsq_masked_p`, `fsq_csub_store` and `g2y_walk_setup` all address these
// offsets, so the two kernels must agree on them.
const F2M_P: i32 = G2Y_P;
const F2M_PINV: i32 = G2Y_PINV;
const F2M_Z: i32 = G2Y_Z;
const F2M_TBL: i32 = G2Y_TBL;
/// `x.re`, `x.im` and their uncorrected sum, then the same for `y`.
const F2M_A0: i32 = 64;
const F2M_A1: i32 = 96;
const F2M_S: i32 = 128;
const F2M_B0: i32 = 160;
const F2M_B1: i32 = 192;
const F2M_T: i32 = 224;
/// Four 64-byte double-width slots: `a0*b0`, `a1*b1`, `s*t`, `a0*b0 + a1*b1`.
const F2M_D: i32 = 256;
const F2M_FRAME: i32 = 512;

/// Byte offsets of the two table regions inside the f2mul rodata blob.
const F2M_TB_PROD: i32 = 0; // 3 rows x 3
const F2M_TB_MOD: i32 = 72; // 2 rows x 3
const F2M_TB_END: i32 = 120;

const F2M_TAB_LABEL: &str = ".Lf2m_tab";

/// The f2mul walk tables, same row shapes as [`g2_ysqr_tables`].
fn f2mul_tables() -> Vec<u64> {
    let mut t: Vec<u64> = Vec::new();
    let row3 = |t: &mut Vec<u64>, a: i32, b: i32, c: i32| {
        t.extend([a as u64, b as u64, c as u64]);
    };
    let d = |k: i32| F2M_D + 64 * k;
    for (x, y, dst) in [
        (F2M_A0, F2M_B0, d(0)),
        (F2M_A1, F2M_B1, d(1)),
        (F2M_S, F2M_T, d(2)),
    ] {
        row3(&mut t, x, y, dst);
    }
    assert_eq!(t.len() * 8, F2M_TB_MOD as usize);
    row3(&mut t, d(0), d(1), 0);
    row3(&mut t, d(2), d(3), 32);
    assert_eq!(t.len() * 8, F2M_TB_END as usize);
    t
}

/// Stage one Fp2 argument: both halves into the frame plus their
/// uncorrected sum. Canonical halves sum below `2p < 2^255`, so four limbs
/// hold the sum with no reduction (mcl's `!isFullBit` `addPre`) and the
/// Karatsuba middle product takes it as an operand unchanged.
fn f2m_stage<M: Machine>(m: &mut M, src: Reg, lo: i32, hi: i32, sum: i32, what: &str) {
    let re = [R8, R9, R10, R11];
    let im = [R12, R13, R14, R15];
    for (k, reg) in re.into_iter().enumerate() {
        m.load(reg, Mem::new(src, 8 * k as i32), &format!("{what}.re[{k}]"));
    }
    for (k, reg) in im.into_iter().enumerate() {
        m.load(
            reg,
            Mem::new(src, 32 + 8 * k as i32),
            &format!("{what}.im[{k}]"),
        );
    }
    for (k, reg) in re.into_iter().enumerate() {
        m.store(
            Mem::new(Reg::Rsp, lo + 8 * k as i32),
            reg,
            &format!("{what}.re[{k}]"),
        );
    }
    for (k, reg) in im.into_iter().enumerate() {
        m.store(
            Mem::new(Reg::Rsp, hi + 8 * k as i32),
            reg,
            &format!("{what}.im[{k}]"),
        );
    }
    for (k, (a, b)) in re.into_iter().zip(im).enumerate() {
        let text = format!("{what}.re + {what}.im, limb {k}");
        if k == 0 {
            m.add(a, b, &text);
        } else {
            m.adc(a, b, &text);
        }
    }
    m.claim_flags_clear("canonical + canonical < 2p < 2^255: no carry, top limb below 2^63");
    for (k, reg) in re.into_iter().enumerate() {
        m.store(
            Mem::new(Reg::Rsp, sum + 8 * k as i32),
            reg,
            &format!("{what}.re + {what}.im [{k}]"),
        );
    }
}

/// `[rsp + dst .. +64] = [rsp + a] + [rsp + b]`, eight limbs, uncorrected.
/// Both addends are raw products of canonical residues, so each is below
/// `p^2 < 2^508`: the sum keeps eight limbs and its top limb stays below
/// `2^62`, so neither CF nor OF survives the chain.
fn f2m_add_pre8<M: Machine>(m: &mut M, dst: i32, a: i32, b: i32) {
    let v = [R8, R9, R10, R11, R12, R13, R14, R15];
    for (k, reg) in v.into_iter().enumerate() {
        m.load(
            reg,
            Mem::new(Reg::Rsp, a + 8 * k as i32),
            &format!("a0*b0 word {k}"),
        );
    }
    for (k, reg) in v.into_iter().enumerate() {
        let addend = Mem::new(Reg::Rsp, b + 8 * k as i32);
        let text = format!("+= a1*b1 word {k}");
        if k == 0 {
            m.add_mem(reg, addend, &text);
        } else {
            m.adc_mem(reg, addend, &text);
        }
    }
    m.claim_flags_clear("a0*b0 + a1*b1 < 2p^2 < 2^509: no carry out of eight words");
    for (k, reg) in v.into_iter().enumerate() {
        m.store(
            Mem::new(Reg::Rsp, dst + 8 * k as i32),
            reg,
            &format!("a0*b0 + a1*b1 word {k}"),
        );
    }
}

/// `narsil_f2mul_x86`: the Fp2 product `(a0 + a1*u)(b0 + b1*u)` over
/// `Fp2 = Fp[u]/(u^2 + 1)` in mcl's lazy double-width Karatsuba shape --
/// three raw 4x4 products and two Montgomery reductions, where the fused
/// sums-of-products route ([`sosd2_x86`]) pays four products and two
/// reductions.
///
/// # Semantics
///
/// With `R = 2^256` and canonical operands, writing `s = a0 + a1`,
/// `t = b0 + b1`:
///
/// * `z.re = (a0*b0 - a1*b1)/R`, one guarded 512-bit difference of two raw
///   products.
/// * `z.im = (s*t - a0*b0 - a1*b1)/R = (a0*b1 + a1*b0)/R`, one 512-bit
///   difference against the raw sum `a0*b0 + a1*b1`.
///
/// Neither raw product is ever reduced, so the Karatsuba identity buys a
/// whole 4x4 product for the price of two four-limb pre-adds and one
/// eight-limb add.
///
/// # Bounds
///
/// Everything below rests on [`G2_YSQR_MODULUS_BOUND`] -- `4p < 2^256`,
/// mcl's `isFullBit == false` for BN254.
///
/// * `s` and `t` are `addPre` sums of canonical residues, so both stay
///   below `2p < 2^255` and keep four limbs.
/// * `a0*b0` and `a1*b1` are below `p^2`, `s*t` below `4p^2`, and
///   `a0*b0 + a1*b1` below `2p^2`. All four fit eight limbs since
///   `4p^2 < p*R`.
/// * The real row's difference is guarded: on borrow it adds `p*R`, which is
///   `0 mod p`, landing in `[p*R - p^2, p*R)`. Without a borrow it is below
///   `p^2`. Either way it is below `p*R`.
/// * The imaginary row cannot borrow -- `s*t = a0*b0 + a0*b1 + a1*b0 +
///   a1*b1` dominates the subtrahend -- and its difference `a0*b1 + a1*b0`
///   is below `2p^2 < p*R`.
/// * `T < p*R` gives `(T + m*p)/R < 2p`: one conditional subtraction per
///   lane returns the canonical residue.
///
/// Arguments: `(z: *mut u64x8, x: *const u64x8, y: *const u64x8,
/// consts: *const { p[4], -p^-1 })` in rdi, rsi, rdx, rcx. Operands are
/// `repr(C)` Fp2 (re then im) and must be canonical. `z` may alias neither
/// `x` nor `y`.
pub fn f2mul_x86<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    m.rodata(F2M_TAB_LABEL, &f2mul_tables());
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    m.alloc_stack(F2M_FRAME);
    m.comment(
        "frame: p +0, -p^-1 +32, z +40, table base +48, walk bound +56, product operands +64, raw products +256",
    );
    m.store(Mem::new(rsp, F2M_Z), Rdi, "spill z");
    for k in 0..4 {
        m.load(Rax, Mem::new(Rcx, 8 * k), &format!("p{k}"));
        m.store(
            Mem::new(rsp, F2M_P + 8 * k),
            Rax,
            "cancel and mask rows address the frame as a consts table",
        );
    }
    m.load(Rax, Mem::new(Rcx, 32), "-p^-1");
    m.store(Mem::new(rsp, F2M_PINV), Rax, "-p^-1");
    m.lea_rodata(Rax, F2M_TAB_LABEL, "walk tables");
    m.store(Mem::new(rsp, F2M_TBL), Rax, "table base");
    m.mov(
        Rbp,
        MULTIPLIER,
        "y pointer: rdx is the implicit mulx multiplicand",
    );

    m.comment("");
    m.comment("product operands. The two sums are uncorrected (mcl addPre):");
    m.comment("p below the top bit is exactly what makes four limbs enough");
    f2m_stage(m, Rsi, F2M_A0, F2M_A1, F2M_S, "x");
    f2m_stage(m, Rbp, F2M_B0, F2M_B1, F2M_T, "y");

    m.comment("");
    m.comment("three raw 4x4 products, none reduced");
    g2y_walk_setup(m, F2M_TB_PROD, 3 * 24);
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, G2Y_END)),
        ".Lf2m_prod",
        &mut |m| g2y_mulpre_row(m),
    );

    m.comment("");
    m.comment("the imaginary row's subtrahend, still double width");
    f2m_add_pre8(m, F2M_D + 3 * 64, F2M_D, F2M_D + 64);

    m.comment("");
    m.comment("two combining rows: one guarded 512-bit difference and one");
    m.comment("Montgomery reduction per output Fp");
    g2y_walk_setup(m, F2M_TB_MOD, 2 * 24);
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, G2Y_END)),
        ".Lf2m_mod",
        &mut |m| g2y_sub_mod_row(m),
    );

    m.free_stack(F2M_FRAME);
    for reg in CALLEE_SAVED.iter().rev() {
        m.pop(*reg);
    }
    m.ret();
}

/// Register roles for `narsil_fp6_mul_x86`.
pub const FP6_REGISTER_MAP: &[(Reg, &str)] = &[
    (
        Rdi,
        "z on entry (spilled as a z+128 cursor); row-base cursor PY in the main loops; z again per component",
    ),
    (
        Rsi,
        "a pointer on entry (spilled); prologue pointer scratch; then PA, the multiplicand cursor a + 8j + 64i",
    ),
    (
        Rdx,
        "b pointer on entry (prologue cursor); the implicit mulx multiplicand",
    ),
    (
        Rcx,
        "consts pointer on entry (prologue only); then the product-walk bound PY + 288",
    ),
    (R8, "active-lane accumulator t0 (xi prologue: value limb 0)"),
    (R9, "active-lane accumulator t1"),
    (R10, "active-lane accumulator t2"),
    (R11, "active-lane accumulator t3"),
    (
        R12,
        "active-lane accumulator t4 (xi prologue: value top limb)",
    ),
    (
        R13,
        "shared top word t5: only the active lane's in-round carries live there",
    ),
    (
        R14,
        "xi outer cursor; then round cursor (byte offset 8j of the source limb)",
    ),
    (
        R15,
        "xi inner cursor; then component cursor (y-window byte offset 0/96/192)",
    ),
    (
        Rbp,
        "lane cursor: y-row byte offset 0 (real lane) / 32 (imag lane)",
    ),
    (
        Rax,
        "low half of the current product; zero for chain closes",
    ),
    (Rbx, "high half of the current product; prologue scratch"),
];

/// fp6 frame layout, all rsp-relative. P at +0 and -p^-1 at +32 mirror the
/// consts-table shape so the cancel rows can address the frame like a table.
const FP6_P: i32 = 0;
const FP6_PINV: i32 = 32;
const FP6_MU: i32 = 40;
const FP6_A_PTR: i32 = 48;
const FP6_Z_CUR: i32 = 56;
/// Dormant lane t0..t4 (t5 is provably zero between lane blocks).
const FP6_DORM: i32 = 64;
/// Five 96-byte Fp2 blocks `[p - im, re, im]` in the order B2 B1 B0 X2 X1.
const FP6_YB: i32 = 104;
const FP6_FRAME: i32 = 584;

/// Two conditional subtractions of p (final value < 2.135p) on t0..t3, then
/// store four limbs at `[rdi + out_off]`. Scratch: rax, rbx, rcx, rdx.
fn fp6_reduce_store<M: Machine>(m: &mut M, out_off: i32, lane: &str) {
    reduce_store(
        m,
        out_off,
        lane,
        2,
        "value < 2.135p, subtract p at most twice",
    );
}

/// `passes` conditional subtractions of p on t0..t3, then store four limbs at
/// `[rdi + out_off]`. `bound` documents the pre-reduction range. Scratch:
/// rax, rbx, rcx, rdx.
fn reduce_store<M: Machine>(m: &mut M, out_off: i32, lane: &str, passes: usize, bound: &str) {
    let value = [T[0], T[1], T[2], T[3]];
    let keep = [Rax, Rbx, Rcx, MULTIPLIER];
    m.comment(&format!("final reduction: {bound}"));
    for pass in 0..passes {
        for (k, (v, s)) in value.iter().zip(keep).enumerate() {
            m.mov(s, *v, &format!("{lane} pass {pass}: keep-copy of word {k}"));
        }
        for (k, v) in value.iter().enumerate() {
            let what = format!("{lane}: word {k} -= p{k}");
            if k == 0 {
                m.sub_mem(*v, Mem::new(Reg::Rsp, FP6_P), &what);
            } else {
                m.sbb_mem(*v, Mem::new(Reg::Rsp, FP6_P + 8 * k as i32), &what);
            }
        }
        for (k, (v, s)) in value.iter().zip(keep).enumerate() {
            m.cmov_carry(*v, s, &format!("borrow: value < p, keep word {k}"));
        }
    }
    for (k, v) in value.iter().enumerate() {
        m.store(
            Mem::new(Rdi, out_off + 8 * k as i32),
            *v,
            &format!("z component {lane} limb {k}"),
        );
    }
}

/// Stage one Fp2 operand into the 96-byte frame block `[p - im, re, im]`
/// (the y-side shape every dual-lane row consumes). Entry: rdx = source Fp2,
/// rdi = destination block, A = p limbs (kept intact), rax/rbx/r12/r13
/// scratch.
fn stage_fp2_block<M: Machine>(m: &mut M, tag: &str) {
    let scratch = [Rax, Rbx, R12, R13];
    for (k, s) in scratch.into_iter().enumerate() {
        m.load(s, Mem::new(Rdx, 8 * k as i32), &format!("{tag}.re[{k}]"));
    }
    for (k, s) in scratch.into_iter().enumerate() {
        m.store(
            Mem::new(Rdi, 32 + 8 * k as i32),
            s,
            &format!("block re[{k}]"),
        );
    }
    for (k, s) in scratch.into_iter().enumerate() {
        m.load(
            s,
            Mem::new(Rdx, 32 + 8 * k as i32),
            &format!("{tag}.im[{k}]"),
        );
    }
    for (k, s) in scratch.into_iter().enumerate() {
        m.store(
            Mem::new(Rdi, 64 + 8 * k as i32),
            s,
            &format!("block im[{k}]"),
        );
    }
    m.comment("negp row: the subtracted imag term enters as p - im");
    for (k, s) in scratch.into_iter().enumerate() {
        m.mov(s, A[k], &format!("p{k}"));
    }
    for (k, s) in scratch.into_iter().enumerate() {
        let what = format!("p{k} - {tag}.im[{k}]");
        if k == 0 {
            m.sub_mem(s, Mem::new(Rdx, 32), &what);
        } else {
            m.sbb_mem(s, Mem::new(Rdx, 32 + 8 * k as i32), &what);
        }
    }
    for (k, s) in scratch.into_iter().enumerate() {
        m.store(Mem::new(Rdi, 8 * k as i32), s, &format!("block negp[{k}]"));
    }
}

/// Stage three Fp2 operands into consecutive 96-byte frame blocks
/// `[p - im, re, im]`. Entry: rdx = first source Fp2 (three contiguous,
/// 192 bytes), A = p limbs (kept intact), rax/rbx/r12/r13 scratch. Rsi/rdi
/// consumed. The destination cursor starts at `rsp + dest_start` and steps
/// `dest_step` per block.
fn fp2_block_stage<M: Machine>(m: &mut M, dest_start: i32, dest_step: i32, label: &str, tag: &str) {
    let rsp = Reg::Rsp;
    m.mov(Rsi, Rdx, "");
    m.add_imm(Rsi, 192, "source end (three Fp2 components)");
    m.mov(Rdi, rsp, "");
    m.add_imm(Rdi, dest_start, "first destination block");
    m.stride_loop(Rdx, 64, LoopEnd::Reg(Rsi), label, &mut |m| {
        stage_fp2_block(m, tag);
        m.add_imm(Rdi, dest_step, "next block");
    });
}

/// xi = 9 + u scaling of two staged Fp2 blocks (`rsp + src_base` and
/// `+ 96`) into their X blocks `block_delta` bytes above: re' = 9re +
/// (p - im), im' = 9im + re, each reduced to canonical via the mu quotient
/// estimate, then the X block's negp row. Consumes r8..r13, rax, rbx, rcx,
/// rbp, rdx, rsi, rdi, r14, r15. Frame table at +0 (p), +32 (-p^-1),
/// +40 (mu).
fn xi_scale_pass<M: Machine>(
    m: &mut M,
    src_base: i32,
    block_delta: i32,
    outer_label: &str,
    inner_label: &str,
) {
    let rsp = Reg::Rsp;
    let v = [R8, R9, R10, R11, R12];
    m.xor_clear(
        R14,
        "outer cursor: first source block (+0) then second (+96)",
    );
    m.stride_loop(R14, 96, LoopEnd::Imm(192), outer_label, &mut |m| {
        m.xor_clear(R15, "inner cursor: real output (+0) then imag (+32)");
        m.stride_loop(R15, 32, LoopEnd::Imm(64), inner_label, &mut |m| {
            m.comment(
                "C row = [PC]: negp(im) for re = 9re - im, re for im = 9im + re; A row = [PC + 32]",
            );
            m.mov(Rsi, rsp, "");
            m.add(Rsi, R14, "+ source block");
            m.add(Rsi, R15, "+ pass");
            m.add_imm(Rsi, src_base, "PC");
            m.mov(Rdi, Rsi, "");
            m.add_imm(
                Rdi,
                block_delta + 32,
                &format!(
                    "output row: each X row sits {} bytes above its C row",
                    block_delta + 32
                ),
            );
            for (k, reg) in v[..4].iter().enumerate() {
                m.load(*reg, Mem::new(Rsi, 32 + 8 * k as i32), &format!("A[{k}]"));
            }
            m.xor_clear(v[4], "top limb; also clears CF/OF for the doubling chains");
            for doubled in ["2A", "4A", "8A"] {
                for (k, reg) in v.iter().enumerate() {
                    let what = format!("{doubled}[{k}]");
                    if k == 0 {
                        m.add(*reg, *reg, &what);
                    } else {
                        m.adc(*reg, *reg, &what);
                    }
                }
            }
            m.comment("9A = 8A + A, then + C: value = 9A + C < 10p < 2^257");
            for (k, reg) in v[..4].iter().enumerate() {
                let what = format!("+= A[{k}]");
                if k == 0 {
                    m.add_mem(*reg, Mem::new(Rsi, 32), &what);
                } else {
                    m.adc_mem(*reg, Mem::new(Rsi, 32 + 8 * k as i32), &what);
                }
            }
            m.adc_zero(v[4], "9A < 9p keeps the top limb below 2^61");
            for (k, reg) in v[..4].iter().enumerate() {
                let what = format!("+= C[{k}]");
                if k == 0 {
                    m.add_mem(*reg, Mem::new(Rsi, 0), &what);
                } else {
                    m.adc_mem(*reg, Mem::new(Rsi, 8 * k as i32), &what);
                }
            }
            m.adc_zero(v[4], "value < 10p < 2^257: top limb is 0 or 1");
            m.comment("estimated quotient: E = floor(value/2^252), q = floor(E*mu/2^58) <= 10");
            m.mov(Rbx, v[4], "E builds from the top limbs");
            m.shld_imm(Rbx, v[3], 4, "E = top five bits of the value");
            m.load(MULTIPLIER, Mem::new(rsp, FP6_MU), "mu");
            m.mulx(Rcx, Rax, Rbx, "E*mu (high half zero: E < 2^5, mu < 2^57)");
            m.shr_imm(Rax, 58, "q");
            m.mov(MULTIPLIER, Rax, "q is the multiplicand");
            m.mulx_mem(Rbx, Rax, Mem::new(rsp, FP6_P), "q*p0 -> (l0, h0)");
            m.mulx_mem(R13, Rcx, Mem::new(rsp, FP6_P + 8), "q*p1 -> (l1, h1)");
            m.add(Rcx, Rbx, "l1 += h0");
            m.mulx_mem(Rbx, Rbp, Mem::new(rsp, FP6_P + 16), "q*p2 -> (l2, h2)");
            m.adc(Rbp, R13, "l2 += h1");
            m.mulx_mem(
                R13,
                MULTIPLIER,
                Mem::new(rsp, FP6_P + 24),
                "q*p3 -> (l3, h3); rdx freed",
            );
            m.adc(MULTIPLIER, Rbx, "l3 += h2");
            m.adc_zero(R13, "h3 += carry; q*p < 11p < 2^260");
            m.sub_rr(v[0], Rax, "value -= q*p, limb 0");
            m.sbb_rr(v[1], Rcx, "limb 1");
            m.sbb_rr(v[2], Rbp, "limb 2");
            m.sbb_rr(v[3], MULTIPLIER, "limb 3");
            m.sbb_rr(v[4], R13, "limb 4");
            m.claim_zero(v[4], "value - q*p < 1.33p < 2^255 fits four limbs");
            m.comment("one conditional subtraction reaches canonical (< 1.33p < 2p)");
            let keep = [Rax, Rbx, Rcx, Rbp];
            for (k, (reg, s)) in v[..4].iter().zip(keep).enumerate() {
                m.mov(s, *reg, &format!("keep-copy of limb {k}"));
            }
            for (k, reg) in v[..4].iter().enumerate() {
                let what = format!("limb {k} -= p{k}");
                if k == 0 {
                    m.sub_mem(*reg, Mem::new(rsp, FP6_P), &what);
                } else {
                    m.sbb_mem(*reg, Mem::new(rsp, FP6_P + 8 * k as i32), &what);
                }
            }
            for (k, (reg, s)) in v[..4].iter().zip(keep).enumerate() {
                m.cmov_carry(*reg, s, &format!("borrow: value < p, keep limb {k}"));
            }
            for (k, reg) in v[..4].iter().enumerate() {
                m.store(
                    Mem::new(Rdi, 8 * k as i32),
                    *reg,
                    &format!("X row limb {k}"),
                );
            }
        });
        m.comment("negp row of the X block just written: p - x.im (x canonical)");
        m.mov(Rsi, rsp, "");
        m.add(Rsi, R14, "+ source block offset");
        m.add_imm(Rsi, src_base + block_delta, "X block of this pass's source");
        for (k, reg) in v[..4].iter().enumerate() {
            m.load(*reg, Mem::new(rsp, FP6_P + 8 * k as i32), &format!("p{k}"));
        }
        for (k, reg) in v[..4].iter().enumerate() {
            let what = format!("p{k} - x.im[{k}]");
            if k == 0 {
                m.sub_mem(*reg, Mem::new(Rsi, 64), &what);
            } else {
                m.sbb_mem(*reg, Mem::new(Rsi, 64 + 8 * k as i32), &what);
            }
        }
        for (k, reg) in v[..4].iter().enumerate() {
            m.store(Mem::new(Rsi, 8 * k as i32), *reg, &format!("X negp[{k}]"));
        }
    });
}

/// Zero both dual-lane accumulators: the six T registers and the five
/// dormant frame words at `rsp + dorm`.
fn zero_lanes<M: Machine>(m: &mut M, dorm: i32) {
    m.comment("both lanes start at zero: registers (imag) and dormant frame (real)");
    for (k, t) in T.into_iter().enumerate() {
        m.xor_clear(t, &format!("t{k} = 0"));
    }
    for (k, t) in T[..5].iter().enumerate() {
        m.store(
            Mem::new(Reg::Rsp, dorm + 8 * k as i32),
            *t,
            &format!("dormant word {k} = 0"),
        );
    }
}

/// Swap the active and dormant lane accumulators through rbx. The shared
/// top word (t5) stays in place: it is provably zero between lane blocks.
fn lane_swap<M: Machine>(m: &mut M, dorm: i32) {
    m.claim_zero(T[5], "the shared top word is clear between lane blocks");
    m.comment("swap the active and dormant lanes through rbx");
    for (k, t) in T[..5].iter().enumerate() {
        let slot = Mem::new(Reg::Rsp, dorm + 8 * k as i32);
        m.load(Rbx, slot, "dormant word");
        m.store(slot, *t, "spill the active word");
        m.mov(*t, Rbx, "activate");
    }
}

/// Montgomery cancel row plus the one-word shift for the six-word T
/// accumulator, reading `-p^-1` and p from the frame table (+32, +0).
/// Register names are loop-iteration-invariant, so the shift is five moves.
fn t6_cancel_shift<M: Machine>(m: &mut M) {
    m.comment("cancel row: m = t0 * -p^-1, then t += m*p zeroes t0");
    m.mov(MULTIPLIER, T[0], "m multiplicand <- t0");
    m.mulx_mem(
        Rbx,
        MULTIPLIER,
        Mem::new(Reg::Rsp, FP6_PINV),
        "m = t0 * -p^-1 mod 2^64 (hi half discarded)",
    );
    m.xor_clear(LO, "re-seed CF = OF = 0 (back edge clobbered flags)");
    sos_row_at(m, Reg::Rsp, FP6_P, "m*p");
    m.claim_zero(T[0], "the Montgomery factor cancels the low word");
    m.comment("shift down one word: the canceled zero word drops");
    for k in 0..5 {
        m.mov(T[k], T[k + 1], &format!("t{k} = t{}", k + 1));
    }
    m.xor_clear(T[5], "t5 = 0 (CF/OF stay clear)");
}

/// `narsil_fp6_mul_x86`: one whole Fp6 = Fp2[v]/(v^3 - xi) product,
/// `z = a * b` with `xi = 9 + u`, in a single leaf call.
///
/// Semantics (exactly `Fp6::mul`'s SoS schoolbook, Longa 2022/367 Eq. 9,
/// with x1 = xi*b1, x2 = xi*b2):
///
/// * c0 = a0*b0 + a1*x2 + a2*x1
/// * c1 = a0*b1 + a1*b0 + a2*x2
/// * c2 = a0*b2 + a1*b1 + a2*b0
///
/// Each output component is one dual-lane sum of three Fp2 products: per
/// lane a T = 6 sum of products with a single interleaved Montgomery
/// reduction (the portable sosd6), subtracted imag terms entering as
/// `p - y` rows. Arguments: `(z, a, b: *mut/const u64x24 in repr(C) Fp6
/// order c0.re, c0.im, .., c2.im. Consts: *const { p[4], -p^-1, mu })` in
/// rdi, rsi, rdx, rcx. `a`, `b` fully canonical (every Fp < p). Outputs
/// canonical. `mu = floor(2^310/p)` drives the xi-scaling reduction.
///
/// # Layout: one cursor for all three components
///
/// The frame holds five Fp2 operand blocks `[p - im, re, im]` (96 bytes
/// each) in the order B2 B1 B0 X2 X1. The three components' operand lists
/// are consecutive windows of that sequence -- c2 = (B2, B1, B0),
/// c1 = (B1, B0, X2), c0 = (B0, X2, X1) -- so a single window cursor
/// (r15 = 0, 96, 192) selects the component and one 96-byte-stride pointer
/// walks its three products. The a side is the same a0, a1, a2 for every
/// component, addressed as a + 8j + 64i. Output components are produced
/// c2, c1, c0. A z cursor in the frame walks down from z + 128.
///
/// # xi scaling in-kernel
///
/// x = xi*w computed once per b component into the X blocks:
/// re = 9*w.re + (p - w.im), im = 9*w.im + w.re, both < 10p over five limbs
/// (9t built as three add-doublings plus t). Reduction to canonical in
/// constant shape: E = floor(value/2^252) (a shld), q = floor(E*mu/2^58)
/// (one mulx by mu), value -= q*p, which lands below 1.33p (worst case over
/// all E buckets), then one conditional subtraction. The interpreter checks
/// the fifth limb dies. The negp rows p - x.im then need x.im < p, which
/// canonical guarantees.
///
/// # Register budget and spill plan
///
/// A dual-lane T = 6 needs two six-word accumulators -- twelve registers,
/// which with rdx/rax/rbx leaves nothing for cursors. Instead the lanes run
/// as blocks (all real rows, cancel, then all imag rows, cancel) and only
/// the active lane keeps registers: the dormant lane's five words live in
/// the frame and swap through rbx at each lane switch. The sixth word (r13)
/// is shared: a lane's in-round overflow dies into t4 at its own cancel
/// shift, so between blocks t5 is provably zero (the interpreter asserts
/// it). That prices the second lane at ten memory ops per round and frees
/// the round, window, and lane cursors plus the two walking pointers --
/// exactly fifteen registers. Cross-lane ILP inside one dual-lane pair is
/// coarser than the row-interleaved sosd2 leaf, but adjacent blocks are
/// data-independent and fit one OoO window together. The point of this leaf
/// is killing per-mul call/marshal overhead (6 calls, 72 pointer stores,
/// 9 negp temps, 2 Rust xi scalings), which is frontend mass, not ILP.
///
/// # Bounds (operands < p after the xi reduction, T = 6)
///
/// Exactly the portable sosd6 bounds: between rounds each lane holds
/// u < 7p < 2^260. The in-round peak before the shift stays below
/// 7p*2^64 < 2^322, so the six-word window absorbs every carry and each
/// row's chain closes are carry-free at t5. The final value is
/// < (1 + 0.1891*6)p < 2.135p: t4 ends exactly zero and two conditional
/// subtractions reach the canonical range.
pub fn fp6_mul_x86<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    m.alloc_stack(FP6_FRAME);
    m.comment(
        "frame: p +0, -p^-1 +32, mu +40, a +48, z cursor +56, dormant lane +64, y window +104",
    );
    m.store(Mem::new(rsp, FP6_A_PTR), Rsi, "spill the a pointer");
    m.mov(Rax, Rdi, "z");
    m.add_imm(
        Rax,
        128,
        "z + 128: components are produced c2 first, cursor walks down",
    );
    m.store(Mem::new(rsp, FP6_Z_CUR), Rax, "z component cursor");
    for (k, reg) in A.iter().enumerate() {
        m.load(
            *reg,
            Mem::new(Rcx, 8 * k as i32),
            &format!("p{k} (kept live through the b copy)"),
        );
    }
    for (k, reg) in A.iter().enumerate() {
        m.store(
            Mem::new(rsp, FP6_P + 8 * k as i32),
            *reg,
            &format!("p{k}: cancel rows address the frame as a consts table"),
        );
    }
    m.load(Rax, Mem::new(Rcx, 32), "-p^-1");
    m.store(Mem::new(rsp, FP6_PINV), Rax, "-p^-1");
    m.load(Rax, Mem::new(Rcx, 40), "mu = floor(2^310/p)");
    m.store(Mem::new(rsp, FP6_MU), Rax, "mu");

    m.comment("");
    m.comment("copy b into the y window: source walks b0 b1 b2, blocks walk B0 B1 B2 down");
    fp2_block_stage(m, FP6_YB + 192, -96, ".Lfp6_bcopy", "b_i");

    m.comment("");
    m.comment("xi scaling: X2 = xi*b2 from B2, then X1 = xi*b1 from B1");
    xi_scale_pass(m, FP6_YB, 288, ".Lfp6_xi", ".Lfp6_xi_val");

    m.comment("");
    m.comment("components c2, c1, c0 = consecutive 3-block windows of [B2, B1, B0, X2, X1]");
    m.xor_clear(R15, "component cursor: window byte offset");
    m.stride_loop(R15, 96, LoopEnd::Imm(288), ".Lfp6_comp", &mut |m| {
        zero_lanes(m, FP6_DORM);
        m.xor_clear(R14, "round cursor: byte offset 8j of the source limb");
        m.stride_loop(R14, 8, LoopEnd::Imm(32), ".Lfp6_round", &mut |m| {
            m.xor_clear(Rbp, "lane cursor: real rows (+0) first, then imag (+32)");
            m.stride_loop(Rbp, 32, LoopEnd::Imm(64), ".Lfp6_lane", &mut |m| {
                lane_swap(m, FP6_DORM);
                m.load(Rsi, Mem::new(rsp, FP6_A_PTR), "a");
                m.add(Rsi, R14, "PA = a + 8j");
                m.mov(Rdi, rsp, "");
                m.add(Rdi, R15, "+ window");
                m.add(Rdi, Rbp, "+ lane row offset");
                m.add_imm(Rdi, FP6_YB, "PY: the first product's block, lane-adjusted");
                m.mov(Rcx, Rdi, "");
                m.add_imm(Rcx, 288, "window end: three products");
                m.stride_loop(Rdi, 96, LoopEnd::Reg(Rcx), ".Lfp6_prod", &mut |m| {
                    m.xor_clear(LO, "re-seed CF = OF = 0 (back edge clobbered flags)");
                    m.load(MULTIPLIER, Mem::new(Rsi, 0), "a_i.re[j]");
                    sos_row_at(m, Rdi, 32, "a_i.re[j]*row0");
                    m.load(MULTIPLIER, Mem::new(Rsi, 32), "a_i.im[j]");
                    sos_row_at(m, Rdi, 0, "a_i.im[j]*row1");
                    m.add_imm(Rsi, 64, "next a component");
                });
                t6_cancel_shift(m);
            });
        });
        m.comment("component epilogue: imag lane in registers, real lane dormant");
        m.load(Rdi, Mem::new(rsp, FP6_Z_CUR), "z component cursor");
        m.claim_zero(T[4], "imag final value < 2.135p < 2^256 fits four words");
        fp6_reduce_store(m, 32, "imag");
        for (k, t) in T[..5].iter().enumerate() {
            m.load(
                *t,
                Mem::new(rsp, FP6_DORM + 8 * k as i32),
                &format!("real lane word {k}"),
            );
        }
        m.claim_zero(T[4], "real final value < 2.135p < 2^256 fits four words");
        fp6_reduce_store(m, 0, "real");
        m.add_imm(Rdi, -64, "");
        m.store(
            Mem::new(rsp, FP6_Z_CUR),
            Rdi,
            "z cursor steps down to the next component",
        );
    });
    m.free_stack(FP6_FRAME);
    for reg in CALLEE_SAVED.iter().rev() {
        m.pop(*reg);
    }
    m.ret();
}

/// Register roles for the compact source-level `xi = 9 + u` helper used by
/// the MCL-shaped Fp12 square wrapper.
pub const FP2_XI_COMPACT_REGISTER_MAP: &[(Reg, &str)] = &[
    (Rdi, "destination on entry; modular-row destination"),
    (Rsi, "source on entry; modular-row source 1"),
    (Rdx, "constants on entry; modular-row source 2"),
    (Rcx, "unused ABI scratch"),
    (R8, "value limb 0"),
    (R9, "value limb 1"),
    (R10, "value limb 2"),
    (R11, "value limb 3"),
    (R12, "conditional-reduction scratch limb 0"),
    (R13, "conditional-reduction scratch limb 1"),
    (R14, "conditional-reduction scratch limb 2"),
    (R15, "conditional-reduction scratch limb 3"),
    (Rbp, "constant-table pointer"),
    (Rbx, "destination base"),
    (Rax, "borrow mask"),
];

/// Register roles for `narsil_fp12_sqr_mcl_x86`.
pub const FP12_SQR_MCL_REGISTER_MAP: &[(Reg, &str)] = &[
    (
        Rdi,
        "in-place Fp12 pointer on entry; callee/row destination",
    ),
    (Rsi, "constant-table pointer on entry; callee/row source 1"),
    (Rdx, "callee/row source 2"),
    (Rcx, "Fp6 callee constants argument; fixed-row loop cursor"),
    (R8, "modular-row value limb 0"),
    (R9, "modular-row value limb 1"),
    (R10, "modular-row value limb 2"),
    (R11, "modular-row value limb 3"),
    (R12, "conditional-reduction scratch limb 0"),
    (R13, "conditional-reduction scratch limb 1"),
    (R14, "conditional-reduction scratch limb 2"),
    (R15, "conditional-reduction scratch limb 3"),
    (Rbp, "constant-table pointer kept across Fp6 calls"),
    (Rbx, "in-place Fp12 pointer kept across Fp6 calls"),
    (Rax, "borrow mask"),
];

/// Store `v mod p`, given `v < 2p`, to `dst`. The source value is in
/// r8..r11. R12..r15 receive the conditional-subtraction copy. `rbp` points
/// at the Fp6 constants table whose first four words are p.
fn compact_csub_store<M: Machine>(m: &mut M, dst: Reg) {
    for (value, scratch) in [R8, R9, R10, R11].into_iter().zip([R12, R13, R14, R15]) {
        m.mov(scratch, value, "keep value before subtracting p");
    }
    for (k, scratch) in [R12, R13, R14, R15].into_iter().enumerate() {
        if k == 0 {
            m.sub_mem(scratch, Mem::new(Rbp, 0), "value -= p");
        } else {
            m.sbb_mem(scratch, Mem::new(Rbp, 8 * k as i32), "value -= p");
        }
    }
    for (value, scratch) in [R8, R9, R10, R11].into_iter().zip([R12, R13, R14, R15]) {
        m.cmov_carry(scratch, value, "borrow: keep the value below p");
    }
    for (k, scratch) in [R12, R13, R14, R15].into_iter().enumerate() {
        m.store(
            Mem::new(dst, 8 * k as i32),
            scratch,
            "canonical output limb",
        );
    }
}

/// Fixed modular Fp row: `[rdi] = [rsi] + [rdx] mod p`.
fn compact_madd_row<M: Machine>(m: &mut M) {
    for (k, value) in [R8, R9, R10, R11].into_iter().enumerate() {
        m.load(value, Mem::new(Rsi, 8 * k as i32), "left limb");
    }
    for (k, value) in [R8, R9, R10, R11].into_iter().enumerate() {
        if k == 0 {
            m.add_mem(value, Mem::new(Rdx, 0), "plus right limb");
        } else {
            m.adc_mem(value, Mem::new(Rdx, 8 * k as i32), "plus right limb");
        }
    }
    m.claim_flags_clear("canonical addends give a sum below 2p < 2^256");
    compact_csub_store(m, Rdi);
}

/// Fixed modular Fp row: `[rdi] = [rsi] - [rdx] mod p`.
fn compact_msub_row<M: Machine>(m: &mut M) {
    for (k, value) in [R8, R9, R10, R11].into_iter().enumerate() {
        m.load(value, Mem::new(Rsi, 8 * k as i32), "left limb");
    }
    m.xor_clear(Rax, "borrow-mask seed");
    for (k, value) in [R8, R9, R10, R11].into_iter().enumerate() {
        if k == 0 {
            m.sub_mem(value, Mem::new(Rdx, 0), "minus right limb");
        } else {
            m.sbb_mem(value, Mem::new(Rdx, 8 * k as i32), "minus right limb");
        }
    }
    m.sbb_rr(Rax, Rax, "mask = -borrow");
    for (k, scratch) in [R12, R13, R14, R15].into_iter().enumerate() {
        m.mov(scratch, Rax, "borrow mask");
        m.and_mem(scratch, Mem::new(Rbp, 8 * k as i32), "p limb when borrowed");
    }
    for (k, (value, addend)) in [R8, R9, R10, R11]
        .into_iter()
        .zip([R12, R13, R14, R15])
        .enumerate()
    {
        if k == 0 {
            m.add(value, addend, "borrow: add p");
        } else {
            m.adc(value, addend, "borrow: add p");
        }
    }
    for (k, value) in [R8, R9, R10, R11].into_iter().enumerate() {
        m.store(Mem::new(Rdi, 8 * k as i32), value, "canonical output limb");
    }
}

/// Compact generated `Fp2 *= xi`, `xi = 9 + u`, used twice by the Fp12
/// wrapper. Form each half as a five-limb value below 10p and reduce it once,
/// instead of canonicalizing every doubling/addition row independently.
/// ABI: `(z: *mut Fp2, x: *const Fp2, consts)`. Z and x distinct.
pub fn fp2_xi_compact_x86<M: Machine>(m: &mut M) {
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    m.mov(Rbx, Rdi, "destination base");
    m.mov(Rbp, Rdx, "constants base");
    m.mov(Rdi, Rsi, "source base stays live across both halves");
    compact_xi_direct(m, Rbx, 0, Rdi, 0, "xi");
    for reg in CALLEE_SAVED.iter().rev() {
        m.pop(*reg);
    }
    m.ret();
}

const F12C_T0: i32 = 0;
const F12C_T1: i32 = 192;
const F12C_AB: i32 = 384;
// Six callee-saved pushes leave the System V entry stack at 8 mod 16. The
// 576-byte live region therefore needs one padding word so rsp is 0 mod 16
// before each nested Fp6/xi call.
const F12C_FRAME: i32 = 584;

/// Reduce a five-limb value below 10p through the same quotient estimate as
/// the Fp6 leaf's xi prologue. Unlike `fsq_mu_reduce5`, this compact-wrapper
/// variant reads p/mu directly from the caller's constants pointer in rbp,
/// so it needs no copied constants table in the wrapper's 576-byte live
/// region (plus eight bytes of System V call-alignment padding).
fn compact_mu_reduce5<M: Machine>(m: &mut M, v: [Reg; 5], s: [Reg; 5]) {
    m.mov(s[0], v[4], "E builds from the top limbs");
    m.shld_imm(s[0], v[3], 4, "E = floor(value/2^252)");
    m.load(MULTIPLIER, Mem::new(Rbp, 40), "mu = floor(2^310/p)");
    m.mulx(s[1], Rax, s[0], "E*mu");
    m.shr_imm(Rax, 58, "q <= 10");
    m.mov(MULTIPLIER, Rax, "q multiplicand");
    m.mulx_mem(s[0], Rax, Mem::new(Rbp, 0), "q*p0 -> (l0,h0)");
    m.mulx_mem(s[2], s[1], Mem::new(Rbp, 8), "q*p1 -> (l1,h1)");
    m.add(s[1], s[0], "l1 += h0");
    m.mulx_mem(s[0], s[3], Mem::new(Rbp, 16), "q*p2 -> (l2,h2)");
    m.adc(s[3], s[2], "l2 += h1");
    m.mulx_mem(s[2], s[4], Mem::new(Rbp, 24), "q*p3 -> (l3,h3)");
    m.adc(s[4], s[0], "l3 += h2");
    m.adc_zero(s[2], "h3 += carry");
    m.sub_rr(v[0], Rax, "value -= q*p limb 0");
    m.sbb_rr(v[1], s[1], "limb 1");
    m.sbb_rr(v[2], s[3], "limb 2");
    m.sbb_rr(v[3], s[4], "limb 3");
    m.sbb_rr(v[4], s[2], "limb 4");
    m.claim_zero(v[4], "estimate leaves value < 1.33p in four limbs");

    for (value, keep) in v[..4].iter().zip(s[..4].iter().copied()) {
        m.mov(keep, *value, "keep before final subtraction");
    }
    for (k, value) in v[..4].iter().enumerate() {
        if k == 0 {
            m.sub_mem(*value, Mem::new(Rbp, 0), "value -= p");
        } else {
            m.sbb_mem(*value, Mem::new(Rbp, 8 * k as i32), "value -= p");
        }
    }
    for (value, keep) in v[..4].iter().zip(s[..4].iter().copied()) {
        m.cmov_carry(*value, keep, "borrow: keep value below p");
    }
}

/// Direct generated xi frontend for the compact Fp12 square. Computes
/// `dst = (9 + u) * src` with one quotient-estimate reduction per Fp half,
/// replacing the old helper's ten separately reduced modular rows.
///
/// Source and destination regions are distinct, which lets the real-half
/// borrow repair use the destination as four words of transient scratch.
/// rbp remains the constants pointer across the complete operation.
fn compact_xi_direct<M: Machine>(
    m: &mut M,
    dst_base: Reg,
    dst_off: i32,
    src_base: Reg,
    src_off: i32,
    tag: &str,
) {
    let v = [R8, R9, R10, R11, R12];
    let s = [R13, R14, R15, Rcx, Rsi];
    for half in 0..2i32 {
        let a_off = src_off + 32 * half;
        let cross_off = src_off + 32 * (1 - half);
        for (k, value) in v[..4].iter().enumerate() {
            m.load(
                *value,
                Mem::new(src_base, a_off + 8 * k as i32),
                &format!("{tag}: A[{k}]"),
            );
        }
        m.xor_clear(v[4], "top limb and carry-chain seed");
        for multiple in ["2A", "4A", "8A"] {
            for (k, value) in v.iter().enumerate() {
                if k == 0 {
                    m.add(*value, *value, multiple);
                } else {
                    m.adc(*value, *value, multiple);
                }
            }
        }
        for (k, value) in v[..4].iter().enumerate() {
            if k == 0 {
                m.add_mem(*value, Mem::new(src_base, a_off), "9A = 8A + A");
            } else {
                m.adc_mem(
                    *value,
                    Mem::new(src_base, a_off + 8 * k as i32),
                    "9A = 8A + A",
                );
            }
        }
        m.adc_zero(v[4], "9A < 9p");

        if half == 0 {
            // 9re-im can be negative only when its five-limb top is zero.
            // Add p under the borrow mask to select its canonical congruent
            // representative before the <10p quotient reduction.
            m.xor_clear(s[0], "zero for the top-limb borrow close");
            for (k, value) in v[..4].iter().enumerate() {
                if k == 0 {
                    m.sub_mem(*value, Mem::new(src_base, cross_off), "9re - im");
                } else {
                    m.sbb_mem(
                        *value,
                        Mem::new(src_base, cross_off + 8 * k as i32),
                        "9re - im",
                    );
                }
            }
            m.sbb_rr(v[4], s[0], "top limb -= borrow");
            // Recreate the borrow from the wrapped top word:
            // after an underflow v4 is all ones. Otherwise it is 0 or 1.
            m.mov(s[0], v[4], "wrapped top limb");
            m.shr_imm(s[0], 63, "borrow bit");
            m.xor_clear(s[1], "mask seed");
            m.sub_rr(s[1], s[0], "mask = -borrow");
            for k in 0..4 {
                m.mov(s[0], s[1], "borrow mask");
                m.and_mem(s[0], Mem::new(Rbp, 8 * k), "masked p limb");
                m.store(
                    Mem::new(dst_base, dst_off + 32 * half + 8 * k),
                    s[0],
                    "stage masked p limb",
                );
            }
            for (k, value) in v[..4].iter().enumerate() {
                if k == 0 {
                    m.add_mem(
                        *value,
                        Mem::new(dst_base, dst_off + 32 * half),
                        "borrow: add p",
                    );
                } else {
                    m.adc_mem(
                        *value,
                        Mem::new(dst_base, dst_off + 32 * half + 8 * k as i32),
                        "borrow: add p",
                    );
                }
            }
            m.adc_zero(v[4], "repair top limb after adding p");
        } else {
            for (k, value) in v[..4].iter().enumerate() {
                if k == 0 {
                    m.add_mem(*value, Mem::new(src_base, cross_off), "9im + re");
                } else {
                    m.adc_mem(
                        *value,
                        Mem::new(src_base, cross_off + 8 * k as i32),
                        "9im + re",
                    );
                }
            }
            m.adc_zero(v[4], "xi half < 10p");
        }

        compact_mu_reduce5(m, v, s);
        for (k, value) in v[..4].iter().enumerate() {
            m.store(
                Mem::new(dst_base, dst_off + 32 * half + 8 * k as i32),
                *value,
                &format!("{tag}: canonical xi output limb {k}"),
            );
        }
    }
}

pub fn fp12_sqr_mcl_x86<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    m.alloc_stack(F12C_FRAME);
    m.mov(Rbx, Rdi, "in-place Fp12");
    m.mov(Rbp, Rsi, "constants");

    m.comment("ab = a*b through the existing generated Fp6 primitive");
    m.mov(Rdi, rsp, "ab destination");
    m.add_imm(Rdi, F12C_AB, "");
    m.mov(Rsi, Rbx, "a");
    m.mov(Rdx, Rbx, "b");
    m.add_imm(Rdx, 192, "");
    m.mov(Rcx, Rbp, "constants");
    m.call("narsil_fp6_mul_x86");

    m.comment("t0 = a+b: six canonical Fp rows");
    m.mov(Rdi, rsp, "t0");
    m.add_imm(Rdi, F12C_T0, "");
    m.mov(Rsi, Rbx, "a");
    m.mov(Rdx, Rbx, "b");
    m.add_imm(Rdx, 192, "");
    m.xor_clear(Rcx, "six-row cursor");
    m.stride_loop(Rcx, 32, LoopEnd::Imm(192), ".Lf12c_t0", &mut |m| {
        compact_madd_row(m);
        for ptr in [Rdi, Rsi, Rdx] {
            m.add_imm(ptr, 32, "next Fp row");
        }
    });

    m.comment("t1.c0 = xi*b.c2 + a.c0");
    m.mov(Rdi, rsp, "t1.c0");
    m.add_imm(Rdi, F12C_T1, "");
    m.mov(Rsi, Rbx, "b.c2");
    m.add_imm(Rsi, 320, "");
    m.mov(Rdx, Rbp, "constants");
    m.call("narsil_fp2_xi_compact_x86");
    m.mov(Rdi, rsp, "t1.c0");
    m.add_imm(Rdi, F12C_T1, "");
    m.mov(Rsi, Rdi, "xi*b.c2");
    m.mov(Rdx, Rbx, "a.c0");
    m.xor_clear(Rcx, "two-row cursor");
    m.stride_loop(Rcx, 32, LoopEnd::Imm(64), ".Lf12c_t1c0", &mut |m| {
        compact_madd_row(m);
        for ptr in [Rdi, Rsi, Rdx] {
            m.add_imm(ptr, 32, "next Fp row");
        }
    });

    m.comment("t1.c1,c2 = b.c0,c1 + a.c1,c2");
    m.mov(Rdi, rsp, "t1.c1");
    m.add_imm(Rdi, F12C_T1 + 64, "");
    m.mov(Rsi, Rbx, "b.c0");
    m.add_imm(Rsi, 192, "");
    m.mov(Rdx, Rbx, "a.c1");
    m.add_imm(Rdx, 64, "");
    m.xor_clear(Rcx, "four-row cursor");
    m.stride_loop(Rcx, 32, LoopEnd::Imm(128), ".Lf12c_t1tail", &mut |m| {
        compact_madd_row(m);
        for ptr in [Rdi, Rsi, Rdx] {
            m.add_imm(ptr, 32, "next Fp row");
        }
    });

    m.comment("st = t0*t1 directly into the c0 half of f");
    m.mov(Rdi, Rbx, "f.c0 destination");
    m.mov(Rsi, rsp, "t0");
    m.add_imm(Rsi, F12C_T0, "");
    m.mov(Rdx, rsp, "t1");
    m.add_imm(Rdx, F12C_T1, "");
    m.mov(Rcx, Rbp, "constants");
    m.call("narsil_fp6_mul_x86");

    m.comment("scratch = xi*ab.c2");
    m.mov(Rdi, rsp, "scratch");
    m.add_imm(Rdi, F12C_T0, "");
    m.mov(Rsi, rsp, "ab.c2");
    m.add_imm(Rsi, F12C_AB + 128, "");
    m.mov(Rdx, Rbp, "constants");
    m.call("narsil_fp2_xi_compact_x86");

    m.comment("c0.c0 = st.c0 - xi*ab.c2 - ab.c0");
    m.mov(Rdi, Rbx, "st.c0");
    m.mov(Rsi, Rdi, "");
    m.mov(Rdx, rsp, "xi*ab.c2");
    m.add_imm(Rdx, F12C_T0, "");
    m.xor_clear(Rcx, "two-row cursor");
    m.stride_loop(Rcx, 32, LoopEnd::Imm(64), ".Lf12c_c00a", &mut |m| {
        compact_msub_row(m);
        for ptr in [Rdi, Rsi, Rdx] {
            m.add_imm(ptr, 32, "next Fp row");
        }
    });
    m.mov(Rdi, Rbx, "c0.c0");
    m.mov(Rsi, Rdi, "");
    m.mov(Rdx, rsp, "ab.c0");
    m.add_imm(Rdx, F12C_AB, "");
    m.xor_clear(Rcx, "two-row cursor");
    m.stride_loop(Rcx, 32, LoopEnd::Imm(64), ".Lf12c_c00b", &mut |m| {
        compact_msub_row(m);
        for ptr in [Rdi, Rsi, Rdx] {
            m.add_imm(ptr, 32, "next Fp row");
        }
    });

    m.comment("c0.c1,c2 = st - (ab.c0,c1) - (ab.c1,c2)");
    m.mov(Rdi, Rbx, "st.c1");
    m.add_imm(Rdi, 64, "");
    m.mov(Rsi, Rdi, "");
    m.mov(Rdx, rsp, "ab.c0");
    m.add_imm(Rdx, F12C_AB, "");
    m.xor_clear(Rcx, "four-row cursor");
    m.stride_loop(Rcx, 32, LoopEnd::Imm(128), ".Lf12c_c0taila", &mut |m| {
        compact_msub_row(m);
        for ptr in [Rdi, Rsi, Rdx] {
            m.add_imm(ptr, 32, "next Fp row");
        }
    });
    m.mov(Rdi, Rbx, "c0.c1");
    m.add_imm(Rdi, 64, "");
    m.mov(Rsi, Rdi, "");
    m.mov(Rdx, rsp, "ab.c1");
    m.add_imm(Rdx, F12C_AB + 64, "");
    m.xor_clear(Rcx, "four-row cursor");
    m.stride_loop(Rcx, 32, LoopEnd::Imm(128), ".Lf12c_c0tailb", &mut |m| {
        compact_msub_row(m);
        for ptr in [Rdi, Rsi, Rdx] {
            m.add_imm(ptr, 32, "next Fp row");
        }
    });

    m.comment("c1 = 2ab: six canonical Fp rows");
    m.mov(Rdi, Rbx, "f.c1");
    m.add_imm(Rdi, 192, "");
    m.mov(Rsi, rsp, "ab");
    m.add_imm(Rsi, F12C_AB, "");
    m.mov(Rdx, Rsi, "double");
    m.xor_clear(Rcx, "six-row cursor");
    m.stride_loop(Rcx, 32, LoopEnd::Imm(192), ".Lf12c_c1", &mut |m| {
        compact_madd_row(m);
        for ptr in [Rdi, Rsi, Rdx] {
            m.add_imm(ptr, 32, "next Fp row");
        }
    });

    m.free_stack(F12C_FRAME);
    for reg in CALLEE_SAVED.iter().rev() {
        m.pop(*reg);
    }
    m.ret();
}

/// Register roles for `narsil_fp12_034_x86`.
pub const FP12_034_REGISTER_MAP: &[(Reg, &str)] = &[
    (
        Rdi,
        "z on entry (spilled); staging cursor; PY per product; z component pointer in the epilogue",
    ),
    (
        Rsi,
        "f pointer on entry (g staging cursor); then PA, the g multiplicand base rsp + G + 8*limb",
    ),
    (
        Rdx,
        "coefficient pointer on entry (staging cursor); the implicit mulx multiplicand",
    ),
    (
        Rcx,
        "consts pointer on entry (prologue only); then the product-walk cursor over the table",
    ),
    (R8, "active-lane accumulator t0 (xi prologue: value limb 0)"),
    (R9, "active-lane accumulator t1"),
    (R10, "active-lane accumulator t2"),
    (R11, "active-lane accumulator t3"),
    (
        R12,
        "active-lane accumulator t4 (xi prologue: value top limb)",
    ),
    (
        R13,
        "shared top word t5: only the active lane's in-round carries live there",
    ),
    (
        R14,
        "xi outer cursor; then round cursor (byte offset 8j of the source limb)",
    ),
    (R15, "component cursor 64*j over the six W-power outputs"),
    (
        Rbp,
        "duplicate-slot cursor in the prologue; then lane cursor: 0 (real) / 32 (imag)",
    ),
    (
        Rax,
        "low half of the current product; zero for chain closes",
    ),
    (
        Rbx,
        "high half of the current product; g-offset and prologue scratch",
    ),
];

/// fp12_034 frame layout. P/-p^-1/mu/dormant sit at the fp6 kernel's offsets
/// so the shared cancel, swap and xi helpers address both frames identically.
const F034_Z_PTR: i32 = 48;
/// Product-walk bound: the absolute address of the table end (the walk
/// cursor is an absolute pointer, so the back edge compares against memory).
const F034_WALK_END: i32 = 56;
/// Product-walk table: three 16-byte entries (y block offset, g offset).
const F034_TAB: i32 = 104;
/// Five 96-byte y blocks `[p - im, re, im]`: C0, C3, C4, X3, X4.
const F034_YB: i32 = 152;
/// Twelve 64-byte g slots: f in W-power order, then slots 0..5 again.
const F034_G: i32 = 632;
const F034_FRAME: i32 = 1400;

/// `narsil_fp12_034_x86`: the whole sparse Fp12 product of the Miller loop,
/// `z = f * (c0 + c3*w + c4*v*w)` with `w^2 = v`, `v^3 = xi = 9 + u`, in a
/// single leaf call (arkworks `mul_by_034`, D-type lines).
///
/// # Semantics (exactly `Fp12::mul_by_034_assign`'s SoS schoolbook)
///
/// In the W-power basis `W = w` (`W^6 = xi`) the element is
/// `sum g_k W^k` with `g = (a0, b0, a1, b1, a2, b2)` for `f = a + b*w`, and
/// the sparse multiplier is `c0 + c3 W + c4 W^3`, so every output is one
/// dual-lane T = 6 sum of three Fp2 products with a single interleaved
/// Montgomery reduction per lane (the portable sosd6):
///
/// * `h_j = g_j*c0 + g_{(j+5) mod 6}*C3' + g_{(j+3) mod 6}*C4'`,
/// * `C3' = xi*c3` exactly at j = 0, `C4' = xi*c4` exactly for j < 3
///   (the wrap terms), both xi values computed in-kernel via the mu
///   quotient-estimate reduction. Subtracted imag terms enter as `p - im`
///   rows staged once per y block.
///
/// Arguments: `(z: *mut u64x48, f: *const u64x48, c: *const u64x24,
/// consts: *const { p[4], -p^-1, mu })` in rdi, rsi, rdx, rcx. `f` is
/// `repr(C)` Fp12 (48 limbs, c0 then c1, each Fp6 as in the fp6 leaf). `c`
/// is the three sparse coefficients c0, c3, c4 as contiguous Fp2s (the
/// caller stages them. They are freshly built per line evaluation, so the
/// staging is free). All inputs canonical (< p). Outputs canonical.
///
/// # In-place update
///
/// `z == f` is the production shape (the Miller accumulator updates in
/// place) and is safe by construction: the prologue stages all of `f` into
/// the frame's g array and `f` is never read again, so output stores cannot
/// alias a live operand and the wrapper needs no copy.
///
/// # Layout: one walk table for all six components
///
/// The g array holds `f` in W-power order with slots 0..5 duplicated to
/// 6..11, so the wrap operands `g_{j+3}`, `g_{j+5}` are plain offsets
/// +192/+320 from slot j -- no modular indexing. The three products of a
/// component are walked through a 3-entry frame table of (y block, g
/// offset) pairs. Only the two wrap y fields change per component (one
/// cmov each: X4-vs-C4 at j < 3, X3-vs-C3 at j = 0). Outputs are produced
/// h_0..h_5. The epilogue maps W order back to `repr(C)` as
/// `z + 192*(j&1) + 64*(j>>1)`.
///
/// # Register budget, spill plan, bounds
///
/// Exactly the fp6 leaf's design: lanes run as blocks sharing one six-word
/// accumulator set (dormant lane in the frame, shared top word provably
/// zero between blocks), the walk cursor is rcx with its bound in the frame
/// (all fifteen GPRs are otherwise live), and the sosd6 bounds apply
/// unchanged -- operands at most p, in-round peak < 7p*2^64 < 2^322, final
/// value < 2.135p, two conditional subtractions per lane.
pub fn fp12_034_x86<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    m.alloc_stack(F034_FRAME);
    m.comment(
        "frame: p +0, -p^-1 +32, mu +40, z +48, walk bound +56, dormant lane +64, product table +104, y blocks +152, g array +632",
    );
    m.store(Mem::new(rsp, F034_Z_PTR), Rdi, "spill z");
    for (k, reg) in A.iter().enumerate() {
        m.load(
            *reg,
            Mem::new(Rcx, 8 * k as i32),
            &format!("p{k} (kept live through the coefficient copy)"),
        );
    }
    for (k, reg) in A.iter().enumerate() {
        m.store(
            Mem::new(rsp, FP6_P + 8 * k as i32),
            *reg,
            &format!("p{k}: cancel rows address the frame as a consts table"),
        );
    }
    m.load(Rax, Mem::new(Rcx, 32), "-p^-1");
    m.store(Mem::new(rsp, FP6_PINV), Rax, "-p^-1");
    m.load(Rax, Mem::new(Rcx, 40), "mu = floor(2^310/p)");
    m.store(Mem::new(rsp, FP6_MU), Rax, "mu");

    m.comment("");
    m.comment("product-walk table: g fields and the C0 entry are fixed, the");
    m.comment("two wrap y fields (e1, e2) are rewritten per component");
    m.xor_clear(Rax, "");
    m.store(Mem::new(rsp, F034_TAB), Rax, "e0.y: the C0 block (+0)");
    m.store(Mem::new(rsp, F034_TAB + 8), Rax, "e0.g: g_j (+0)");
    m.add_imm(Rax, 192, "");
    m.store(Mem::new(rsp, F034_TAB + 24), Rax, "e1.g: g_{j+3} (+192)");
    m.add_imm(Rax, 128, "");
    m.store(Mem::new(rsp, F034_TAB + 40), Rax, "e2.g: g_{j+5} (+320)");
    m.mov(Rax, rsp, "");
    m.add_imm(Rax, F034_TAB + 48, "");
    m.store(
        Mem::new(rsp, F034_WALK_END),
        Rax,
        "product-walk bound: the table end address",
    );

    m.comment("");
    m.comment("g array: f in W-power order g = a0, b0, a1, b1, a2, b2, slots");
    m.comment("duplicated so the wrap products index without mod; f is fully");
    m.comment("staged before any z store, which is what makes z == f safe");
    m.mov(Rax, Rsi, "");
    m.add_imm(Rax, 192, "a-half end");
    m.mov(R14, Rsi, "");
    m.add_imm(R14, 192, "b-half source cursor");
    m.mov(Rdi, rsp, "");
    m.add_imm(
        Rdi,
        F034_G,
        "g slot cursor (one a/b slot pair per iteration)",
    );
    m.mov(Rbp, Rdi, "");
    m.add_imm(Rbp, 384, "duplicate cursor: slot k + 6");
    let scratch = [Rbx, Rcx, R12, R13];
    m.stride_loop(Rsi, 64, LoopEnd::Reg(Rax), ".Lf034_g", &mut |m| {
        for (name, src, dst) in [("a_t", Rsi, 0), ("b_t", R14, 64)] {
            for half in 0..2i32 {
                for (k, s) in scratch.into_iter().enumerate() {
                    m.load(
                        s,
                        Mem::new(src, 32 * half + 8 * k as i32),
                        &format!("{name}[{}]", 4 * half + k as i32),
                    );
                }
                for (k, s) in scratch.into_iter().enumerate() {
                    m.store(
                        Mem::new(Rdi, dst + 32 * half + 8 * k as i32),
                        s,
                        &format!("g slot limb {}", 4 * half + k as i32),
                    );
                }
                for (k, s) in scratch.into_iter().enumerate() {
                    m.store(
                        Mem::new(Rbp, dst + 32 * half + 8 * k as i32),
                        s,
                        "duplicate slot",
                    );
                }
            }
        }
        m.add_imm(R14, 64, "");
        m.add_imm(Rdi, 128, "next slot pair");
        m.add_imm(Rbp, 128, "");
    });

    m.comment("");
    m.comment("stage the coefficient blocks: c0 -> C0, c3 -> C3, c4 -> C4");
    fp2_block_stage(m, F034_YB, 96, ".Lf034_c", "c_i");

    m.comment("");
    m.comment("xi scaling: X3 = xi*c3 from C3, then X4 = xi*c4 from C4");
    xi_scale_pass(m, F034_YB + 96, 192, ".Lf034_xi", ".Lf034_xi_val");

    m.comment("");
    m.comment("components h_j, j = 0..5 in the W-power basis:");
    m.comment("h_j = g_j*C0 + g_{j+5}*C3' + g_{j+3}*C4' (wrap xi via X blocks)");
    m.xor_clear(R15, "component cursor: 64*j");
    m.stride_loop(R15, 64, LoopEnd::Imm(384), ".Lf034_comp", &mut |m| {
        m.comment("wrap selection: e1.y = X4 exactly for j < 3, e2.y = X3 exactly at j = 0");
        m.xor_clear(Rax, "");
        m.add_imm(Rax, 192, "the C4 block offset, also the C -> X block delta");
        m.mov(Rbx, Rax, "");
        m.add(Rbx, Rax, "X4 block offset (+384)");
        m.mov(Rcx, R15, "");
        m.add_imm(Rcx, -192, "CF set exactly when 64j >= 192 and j >= 3");
        m.cmov_carry(Rbx, Rax, "j >= 3: plain C4");
        m.store(Mem::new(rsp, F034_TAB + 16), Rbx, "e1.y");
        m.xor_clear(Rbx, "");
        m.add_imm(Rbx, 96, "C3 block offset");
        m.mov(Rcx, Rbx, "");
        m.add(Rcx, Rax, "X3 block offset (+288)");
        m.xor_clear(Rax, "");
        m.sub_rr(Rax, R15, "CF set exactly when j > 0");
        m.cmov_carry(Rcx, Rbx, "j > 0: plain C3");
        m.store(Mem::new(rsp, F034_TAB + 32), Rcx, "e2.y");
        zero_lanes(m, FP6_DORM);
        m.xor_clear(R14, "round cursor: byte offset 8j of the source limb");
        m.stride_loop(R14, 8, LoopEnd::Imm(32), ".Lf034_round", &mut |m| {
            m.xor_clear(Rbp, "lane cursor: real rows (+0) first, then imag (+32)");
            m.stride_loop(Rbp, 32, LoopEnd::Imm(64), ".Lf034_lane", &mut |m| {
                lane_swap(m, FP6_DORM);
                m.mov(Rsi, rsp, "");
                m.add(Rsi, R15, "+ 64j: the g_j slot");
                m.add(Rsi, R14, "+ 8*limb");
                m.add_imm(Rsi, F034_G, "PA: the g multiplicand base");
                m.mov(Rcx, rsp, "");
                m.add_imm(Rcx, F034_TAB, "product-walk cursor");
                m.stride_loop(
                    Rcx,
                    16,
                    LoopEnd::Mem(Mem::new(rsp, F034_WALK_END)),
                    ".Lf034_prod",
                    &mut |m| {
                        m.mov(Rdi, rsp, "");
                        m.add(Rdi, Rbp, "+ lane row offset");
                        m.add_imm(Rdi, F034_YB, "");
                        m.add_mem(Rdi, Mem::new(Rcx, 0), "PY: this product's y block");
                        m.load(Rbx, Mem::new(Rcx, 8), "g offset of this product");
                        m.load_indexed(MULTIPLIER, Rsi, Rbx, "g.re[limb]");
                        m.xor_clear(LO, "re-seed CF = OF = 0 (pointer math clobbered flags)");
                        sos_row_at(m, Rdi, 32, "g.re*row0");
                        m.load(
                            Rbx,
                            Mem::new(Rcx, 8),
                            "g offset again (rbx was the row's hi scratch)",
                        );
                        m.add_imm(Rbx, 32, "the imag limbs sit 32 bytes up");
                        m.load_indexed(MULTIPLIER, Rsi, Rbx, "g.im[limb]");
                        m.xor_clear(LO, "re-seed CF = OF = 0 (add clobbered flags)");
                        sos_row_at(m, Rdi, 0, "g.im*row1");
                    },
                );
                t6_cancel_shift(m);
            });
        });
        m.comment("component epilogue: W order back to repr(C), z + 192*(j&1) + 64*(j>>1)");
        m.xor_clear(Rcx, "zero source for the shift");
        m.mov(Rax, R15, "");
        m.shr_imm(Rax, 7, "j >> 1");
        m.shld_imm(Rax, Rcx, 6, "A = 64*(j >> 1)");
        m.mov(Rbx, R15, "");
        m.sub_rr(Rbx, Rax, "");
        m.sub_rr(Rbx, Rax, "64j - 128*(j >> 1) = 64*(j&1)");
        m.mov(Rcx, Rbx, "");
        m.add(Rbx, Rbx, "");
        m.add(Rbx, Rcx, "192*(j&1)");
        m.add(Rbx, Rax, "the component's z offset");
        m.load(Rdi, Mem::new(rsp, F034_Z_PTR), "z");
        m.add(Rdi, Rbx, "z component pointer");
        m.add_imm(Rdi, 32, "imag half first: the store cursor walks down");
        m.comment("imag lane from the registers, then the dormant real lane");
        m.xor_clear(Rbp, "output-lane counter");
        m.stride_loop(Rbp, 32, LoopEnd::Imm(64), ".Lf034_out", &mut |m| {
            m.claim_zero(T[4], "final value < 2.135p < 2^256 fits four words");
            fp6_reduce_store(m, 0, "lane");
            m.comment("reload the dormant (real) lane; the last pass reloads dead words");
            for (k, t) in T[..5].iter().enumerate() {
                m.load(
                    *t,
                    Mem::new(rsp, FP6_DORM + 8 * k as i32),
                    &format!("real lane word {k}"),
                );
            }
            m.add_imm(Rdi, -32, "step down to the real half");
        });
    });
    m.free_stack(F034_FRAME);
    for reg in CALLEE_SAVED.iter().rev() {
        m.pop(*reg);
    }
    m.ret();
}

/// Register roles for `narsil_fp4_sqr_x86`.
pub const FP4_SQR_REGISTER_MAP: &[(Reg, &str)] = &[
    (
        Rdi,
        "z on entry (spilled as a cursor); staging/PY pointer; z again in the epilogues",
    ),
    (
        Rsi,
        "r0 pointer on entry (spilled); then the mult cursor of each product",
    ),
    (
        Rdx,
        "r1 pointer on entry (spilled); the implicit mulx multiplicand",
    ),
    (
        Rcx,
        "consts pointer on entry (prologue only); then the product-walk cursor",
    ),
    (R8, "active-lane accumulator t0 (xi prologue: value limb 0)"),
    (R9, "active-lane accumulator t1"),
    (R10, "active-lane accumulator t2"),
    (R11, "active-lane accumulator t3"),
    (
        R12,
        "active-lane accumulator t4 (xi prologue: value top limb)",
    ),
    (
        R13,
        "shared top word t5: only the active lane's in-round carries live there",
    ),
    (
        R14,
        "staging pointer-list cursor; r0 pointer through the scale loop; then round cursor",
    ),
    (R15, "scale-loop half cursor; then component cursor 16*c"),
    (
        Rbp,
        "lane cursor: real rows (+0) / imag rows (+32); output-lane counter",
    ),
    (
        Rax,
        "low half of the current product; zero for chain closes",
    ),
    (Rbx, "high half of the current product; prologue scratch"),
];

/// fp4_sqr frame layout. P/-p^-1/mu/dormant sit at the fp6 kernel's offsets
/// so the shared cancel, swap and xi helpers address both frames identically.
const FP4_Z_CUR: i32 = 48;
/// Product-walk bound: the absolute address of the table end.
const FP4_WALK_END: i32 = 56;
/// Product-walk table: three 16-byte entries (mult base, y block), absolute.
const FP4_TAB: i32 = 104;
/// The r0 and r1 argument pointers, contiguous for the staging walk.
const FP4_SRC: i32 = 152;
/// Two 96-byte y blocks `[p - im, re, im]`: r0 then r1.
const FP4_R0B: i32 = 168;
const FP4_R1B: i32 = 264;
/// xi*r1 rows (re, im). Multiplicand-only, so no negp row.
const FP4_X: i32 = 360;
/// d = 2*r0 rows (re, im). Multiplicand-only.
const FP4_D: i32 = 424;
const FP4_FRAME: i32 = 488;

/// `narsil_fp4_sqr_x86`: the whole Fp4 square of the Granger-Scott
/// cyclotomic square, `(r0 + r1*y)^2` with `y^2 = xi = 9 + u`, in a single
/// leaf call -- the hottest final-exponentiation kernel (three calls per
/// cyclotomic square, 576 per final exp).
///
/// # Semantics (exactly `Fp12::fp4_square_sos`'s SoS row lists)
///
/// With `x = xi*r1` and `d = 2*r0`:
///
/// * `t0 = r0^2 + xi*r1^2`: a dual-lane T = 4 sum of the two Fp2 products
///   `r0*r0 + x*r1` with a single interleaved Montgomery reduction per lane,
/// * `t1 = 2*r0*r1 = d*r1`: a dual-lane T = 2 sum of one Fp2 product,
///
/// subtracted imag terms entering as `p - im` rows staged once per y block.
/// `x` is computed in-kernel via the mu quotient-estimate reduction, `d` as
/// one doubling chain. Both are multiplicand-only -- rows read their limbs
/// into rdx, and the SoS bound constrains only the y side -- so neither
/// needs a negp row, and neither is reduced to canonical: x stays at the
/// q-estimate bound (< 1.33p) and d at 2*r0 (< 2p), both four limbs. The
/// extra multiples of p vanish in the fully-reduced SoS output, so the
/// result is still bit-identical to the composed path.
///
/// Arguments: `(z: *mut u64x16, r0, r1: *const u64x8, consts: *const
/// { p[4], -p^-1, mu })` in rdi, rsi, rdx, rcx. `r0`/`r1` are `repr(C)` Fp2
/// (re then im, canonical). `z` receives t0 then t1 as `repr(C)` Fp2 pairs,
/// all canonical, and must not alias `r0` or `r1` (the wrapper builds a
/// fresh output. Both operands are in fact fully staged before the first
/// store, but the contract keeps the stronger requirement).
///
/// # Layout: one walk table for both components
///
/// The frame stages r0 and r1 as the standard `[p - im, re, im]` y blocks
/// plus the x and d row pairs. A three-entry table of (mult base, y block)
/// absolute address pairs drives one product walk: t0 walks (r0, R0), (x,
/// R1). T1 walks (d, R1) alone -- the component cursor (16*c) selects the
/// window and its end bound, so both components share every loop body.
/// Latency note: the pow_x chain runs through t0 (cyclotomic_square feeds
/// t0/t1 back through z0..z5), and the lane-block shape keeps each lane's
/// four cancels serial exactly as two `narsil_sos_x86` calls would be --
/// this leaf deletes the call/marshal overhead (four calls, 24 pointer
/// stores, two negp temps, a Rust xi scaling and an Fp2 double per
/// fp4_square_sos), not the dependent-chain structure.
///
/// # Register budget, spill plan, bounds
///
/// Exactly the fp6 leaf's design: lanes run as blocks sharing one six-word
/// accumulator set (dormant lane in the frame, shared top word provably
/// zero between blocks), and the sosd6 bound argument applies with T = 4:
/// operands at most p, in-round peak < 5p*2^64 < 2^322, final value
/// < (1 + 0.1891*4)p < 1.76p, so ONE conditional subtraction per lane
/// reaches canonical (T = 2 ends lower still, < 1.38p).
pub fn fp4_sqr_x86<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    m.alloc_stack(FP4_FRAME);
    m.comment(
        "frame: p +0, -p^-1 +32, mu +40, z cursor +48, walk bound +56, dormant lane +64, product table +104, argument pointers +152, y blocks +168, xi rows +360, d rows +424",
    );
    m.store(
        Mem::new(rsp, FP4_Z_CUR),
        Rdi,
        "z cursor: t0 first, t1 at +64",
    );
    m.store(Mem::new(rsp, FP4_SRC), Rsi, "spill the r0 pointer");
    m.store(Mem::new(rsp, FP4_SRC + 8), Rdx, "spill the r1 pointer");
    for (k, reg) in A.iter().enumerate() {
        m.load(
            *reg,
            Mem::new(Rcx, 8 * k as i32),
            &format!("p{k} (kept live through the block staging)"),
        );
    }
    for (k, reg) in A.iter().enumerate() {
        m.store(
            Mem::new(rsp, FP6_P + 8 * k as i32),
            *reg,
            &format!("p{k}: cancel rows address the frame as a consts table"),
        );
    }
    m.load(Rax, Mem::new(Rcx, 32), "-p^-1");
    m.store(Mem::new(rsp, FP6_PINV), Rax, "-p^-1");
    m.load(Rax, Mem::new(Rcx, 40), "mu = floor(2^310/p)");
    m.store(Mem::new(rsp, FP6_MU), Rax, "mu");

    m.comment("");
    m.comment("product-walk table: t0 walks e0, e1; t1 walks e2 alone. The");
    m.comment("address chain climbs the frame so every step is one add");
    m.mov(Rax, rsp, "");
    m.add_imm(Rax, FP4_R0B, "");
    m.store(Mem::new(rsp, FP4_TAB + 8), Rax, "e0.y: the r0 block");
    m.add_imm(Rax, 32, "");
    m.store(
        Mem::new(rsp, FP4_TAB),
        Rax,
        "e0.mult: the r0 block's re row",
    );
    m.add_imm(Rax, FP4_R1B - FP4_R0B - 32, "");
    m.store(Mem::new(rsp, FP4_TAB + 24), Rax, "e1.y: the r1 block");
    m.store(Mem::new(rsp, FP4_TAB + 40), Rax, "e2.y: the r1 block again");
    m.add_imm(Rax, FP4_X - FP4_R1B, "");
    m.store(Mem::new(rsp, FP4_TAB + 16), Rax, "e1.mult: the xi*r1 rows");
    m.add_imm(Rax, FP4_D - FP4_X, "");
    m.store(Mem::new(rsp, FP4_TAB + 32), Rax, "e2.mult: the d rows");

    m.comment("");
    m.comment("stage the operand blocks: r0 then r1 as [p - im, re, im]");
    m.mov(R14, rsp, "");
    m.add_imm(R14, FP4_SRC, "cursor over the two argument pointers");
    m.mov(Rcx, R14, "");
    m.add_imm(Rcx, 16, "pointer-list end");
    m.mov(Rdi, rsp, "");
    m.add_imm(Rdi, FP4_R0B, "first destination block");
    m.stride_loop(R14, 8, LoopEnd::Reg(Rcx), ".Lfp4_stage", &mut |m| {
        m.load(Rdx, Mem::new(R14, 0), "source Fp2");
        stage_fp2_block(m, "r_i");
        m.add_imm(Rdi, 96, "next block");
    });

    m.comment("");
    m.comment("x = xi*r1 (from the staged r1 block) and d = 2*r0, one loop pass");
    m.comment("per half. Both are multiplicand-only: rows read their limbs into");
    m.comment("rdx and the SoS bound constrains only the y side, so x stays at");
    m.comment("its q-estimate bound (< 1.33p) and d at 2r0 (< 2p) -- four limbs");
    m.comment("each, no conditional subtraction, and no negp rows");
    let v = [R8, R9, R10, R11, R12];
    m.load(R14, Mem::new(rsp, FP4_SRC), "r0 (the doubling source)");
    m.xor_clear(R15, "half cursor: real (+0) then imag (+32)");
    m.stride_loop(R15, 32, LoopEnd::Imm(64), ".Lfp4_scale", &mut |m| {
        m.comment(
            "C row = [PC]: negp(im) for re = 9re - im, re for im = 9im + re; A row = [PC + 32]",
        );
        m.mov(Rsi, rsp, "");
        m.add(Rsi, R15, "+ pass");
        m.add_imm(Rsi, FP4_R1B, "PC");
        m.mov(Rdi, Rsi, "");
        m.add_imm(Rdi, FP4_X - FP4_R1B, "the x output row");
        m.xor_clear(MULTIPLIER, "");
        m.add_imm(MULTIPLIER, 9, "xi = 9 + u: the real scale is one mulx row");
        m.mulx_mem(Rbx, v[0], Mem::new(Rsi, 32), "9*A[0] -> (v0, hi)");
        m.mulx_mem(Rcx, v[1], Mem::new(Rsi, 40), "9*A[1] -> (lo, hi)");
        m.add(v[1], Rbx, "v1 += hi(9*A[0])");
        m.mulx_mem(Rbx, v[2], Mem::new(Rsi, 48), "9*A[2] -> (lo, hi)");
        m.adc(v[2], Rcx, "v2 += hi(9*A[1])");
        m.mulx_mem(v[4], v[3], Mem::new(Rsi, 56), "9*A[3] -> (lo, v4)");
        m.adc(v[3], Rbx, "v3 += hi(9*A[2])");
        m.adc_zero(v[4], "9A < 9p: the chain closes into the top limb");
        for (k, reg) in v[..4].iter().enumerate() {
            let what = format!("+= C[{k}]");
            if k == 0 {
                m.add_mem(*reg, Mem::new(Rsi, 0), &what);
            } else {
                m.adc_mem(*reg, Mem::new(Rsi, 8 * k as i32), &what);
            }
        }
        m.adc_zero(v[4], "value < 10p < 2^257: top limb is 0 or 1");
        m.comment("estimated quotient: E = floor(value/2^252), q = floor(E*mu/2^58) <= 10");
        m.mov(Rbx, v[4], "E builds from the top limbs");
        m.shld_imm(Rbx, v[3], 4, "E = top five bits of the value");
        m.load(MULTIPLIER, Mem::new(rsp, FP6_MU), "mu");
        m.mulx(Rcx, Rax, Rbx, "E*mu (high half zero: E < 2^5, mu < 2^57)");
        m.shr_imm(Rax, 58, "q");
        m.mov(MULTIPLIER, Rax, "q is the multiplicand");
        m.mulx_mem(Rbx, Rax, Mem::new(rsp, FP6_P), "q*p0 -> (l0, h0)");
        m.mulx_mem(R13, Rcx, Mem::new(rsp, FP6_P + 8), "q*p1 -> (l1, h1)");
        m.add(Rcx, Rbx, "l1 += h0");
        m.mulx_mem(Rbx, Rbp, Mem::new(rsp, FP6_P + 16), "q*p2 -> (l2, h2)");
        m.adc(Rbp, R13, "l2 += h1");
        m.mulx_mem(
            R13,
            MULTIPLIER,
            Mem::new(rsp, FP6_P + 24),
            "q*p3 -> (l3, h3); rdx freed",
        );
        m.adc(MULTIPLIER, Rbx, "l3 += h2");
        m.adc_zero(R13, "h3 += carry; q*p < 11p < 2^260");
        m.sub_rr(v[0], Rax, "value -= q*p, limb 0");
        m.sbb_rr(v[1], Rcx, "limb 1");
        m.sbb_rr(v[2], Rbp, "limb 2");
        m.sbb_rr(v[3], MULTIPLIER, "limb 3");
        m.sbb_rr(v[4], R13, "limb 4");
        m.claim_zero(v[4], "value - q*p < 1.33p < 2^255 fits four limbs");
        for (k, reg) in v[..4].iter().enumerate() {
            m.store(
                Mem::new(Rdi, 8 * k as i32),
                *reg,
                &format!("x row limb {k}"),
            );
        }
        m.comment("d half: one doubling chain, no reduction (2r0 < 2p < 2^255)");
        m.mov(MULTIPLIER, R14, "");
        m.add(MULTIPLIER, R15, "the r0 half");
        for (k, reg) in v[..4].iter().enumerate() {
            m.load(
                *reg,
                Mem::new(MULTIPLIER, 8 * k as i32),
                &format!("r0 limb {k}"),
            );
        }
        for (k, reg) in v[..4].iter().enumerate() {
            let what = format!("2x limb {k}");
            if k == 0 {
                m.add(*reg, *reg, &what);
            } else {
                m.adc(*reg, *reg, &what);
            }
        }
        m.claim_flags_clear("2r0 < 2^255: the doubling chain closes carry-free");
        m.add_imm(Rdi, FP4_D - FP4_X, "the d output row");
        for (k, reg) in v[..4].iter().enumerate() {
            m.store(Mem::new(Rdi, 8 * k as i32), *reg, &format!("d limb {k}"));
        }
    });

    m.comment("");
    m.comment("components: t0 = r0*r0 + x*r1 (T = 4), then t1 = d*r1 (T = 2)");
    m.xor_clear(R15, "component cursor: 16*c");
    m.stride_loop(R15, 16, LoopEnd::Imm(32), ".Lfp4_comp", &mut |m| {
        m.mov(Rax, rsp, "");
        m.add(Rax, R15, "");
        m.add_imm(
            Rax,
            FP4_TAB + 32,
            "walk bound: e2 for t0, the table end for t1",
        );
        m.store(Mem::new(rsp, FP4_WALK_END), Rax, "");
        zero_lanes(m, FP6_DORM);
        m.xor_clear(R14, "round cursor: byte offset 8j of the source limb");
        m.stride_loop(R14, 8, LoopEnd::Imm(32), ".Lfp4_round", &mut |m| {
            m.xor_clear(Rbp, "lane cursor: real rows (+0) first, then imag (+32)");
            m.stride_loop(Rbp, 32, LoopEnd::Imm(64), ".Lfp4_lane", &mut |m| {
                lane_swap(m, FP6_DORM);
                m.mov(Rcx, rsp, "");
                m.add(Rcx, R15, "");
                m.add(Rcx, R15, "+ 32*c: this component's first entry");
                m.add_imm(Rcx, FP4_TAB, "product-walk cursor");
                m.stride_loop(
                    Rcx,
                    16,
                    LoopEnd::Mem(Mem::new(rsp, FP4_WALK_END)),
                    ".Lfp4_prod",
                    &mut |m| {
                        m.load(Rsi, Mem::new(Rcx, 0), "mult base of this product");
                        m.add(Rsi, R14, "+ 8*limb");
                        m.load(Rdi, Mem::new(Rcx, 8), "y block of this product");
                        m.add(Rdi, Rbp, "PY: the y block, lane-adjusted");
                        m.load(MULTIPLIER, Mem::new(Rsi, 0), "mult.re[limb]");
                        m.xor_clear(LO, "re-seed CF = OF = 0 (pointer math clobbered flags)");
                        sos_row_at(m, Rdi, 32, "mult.re*row0");
                        m.load(MULTIPLIER, Mem::new(Rsi, 32), "mult.im[limb]");
                        sos_row_at(m, Rdi, 0, "mult.im*row1");
                    },
                );
                t6_cancel_shift(m);
            });
        });
        m.comment("component epilogue: imag lane in registers, real lane dormant");
        m.load(Rdi, Mem::new(rsp, FP4_Z_CUR), "z component cursor");
        m.add_imm(Rdi, 32, "imag half first: the store cursor walks down");
        m.xor_clear(Rbp, "output-lane counter");
        m.stride_loop(Rbp, 32, LoopEnd::Imm(64), ".Lfp4_out", &mut |m| {
            m.claim_zero(T[4], "final value < 1.76p < 2^256 fits four words");
            reduce_store(m, 0, "lane", 1, "value < 1.76p, subtract p at most once");
            m.comment("reload the dormant (real) lane; the last pass reloads dead words");
            for (k, t) in T[..5].iter().enumerate() {
                m.load(
                    *t,
                    Mem::new(rsp, FP6_DORM + 8 * k as i32),
                    &format!("real lane word {k}"),
                );
            }
            m.add_imm(Rdi, -32, "step down to the real half");
        });
        m.add_imm(Rdi, 96, "");
        m.store(
            Mem::new(rsp, FP4_Z_CUR),
            Rdi,
            "z cursor steps up to the t1 half",
        );
    });
    m.free_stack(FP4_FRAME);
    for reg in CALLEE_SAVED.iter().rev() {
        m.pop(*reg);
    }
    m.ret();
}

/// Register roles for `narsil_fp12_sqr_x86`.
pub const FP12_SQR_REGISTER_MAP: &[(Reg, &str)] = &[
    (
        Rdi,
        "z on entry (spilled); walk destination pointer in every table-driven pass",
    ),
    (
        Rsi,
        "f pointer on entry (staging source); walk source-1 pointer; mask scratch",
    ),
    (
        Rdx,
        "consts pointer on entry; the implicit mulx multiplicand; walk scratch",
    ),
    (
        Rcx,
        "walk source-2 pointer; product m-walk cursor; staging cursor",
    ),
    (R8, "accumulator/value word 0"),
    (R9, "accumulator/value word 1"),
    (R10, "accumulator/value word 2"),
    (R11, "accumulator/value word 3"),
    (R12, "accumulator/value word 4"),
    (R13, "accumulator/value word 5; mu-reduction scratch"),
    (
        R14,
        "product round cursor (byte offset 8j); 8-limb value word 6",
    ),
    (R15, "product block cursor 96k; 8-limb value word 7"),
    (
        Rbp,
        "outer iteration cursor (ctx row address, spilled per phase); every walk's row cursor",
    ),
    (
        Rax,
        "low half of the current product; zero for chain closes; borrow mask",
    ),
    (
        Rbx,
        "high half of the current product; walk bound and half cursors",
    ),
];

// fp12_sqr frame layout. P/-p^-1/mu sit at the fp6 kernel's offsets so the
// shared cancel-row helper addresses this frame as a consts table.
const FSQ_Z: i32 = 48;
/// Outer two-iteration loop bound (ctx table end address).
const FSQ_OUTER_END: i32 = 56;
/// Current walk bound (one walk at a time), plus a second slot for the
/// sums walk nested inside the side loop.
const FSQ_WALK_END: i32 = 64;
const FSQ_WALK_END2: i32 = 72;
/// Outer cursor spill: every phase uses all fifteen GPRs.
const FSQ_CTX_SPILL: i32 = 80;
/// The rodata table base (lea once, reloaded per walk).
const FSQ_TBL: i32 = 88;
/// Product m-walk bound (constant address, set once).
const FSQ_MEND: i32 = 96;
// Per-iteration ctx fields, unpacked from the ctx row into frame slots.
const FSQ_MUXI_SRC: i32 = 104;
const FSQ_MUXI_DST: i32 = 112;
const FSQ_STAGE_X: i32 = 120;
const FSQ_STAGE_Y: i32 = 128;
const FSQ_MOD_DST: i32 = 136;
const FSQ_ADD_SEG: i32 = 144;
/// p - src.im of the current xi site.
const FSQ_NEGIM: i32 = 152;
/// Staging destination cursor (XSTG then YSTG).
const FSQ_STAGE_CUR: i32 = 184;
/// An 8-limb zero: source of the double-width negation rows.
const FSQ_ZERO8: i32 = 192;
/// Per-iteration xi output (one Fp2): xi*b.c2, then xi*V.c2.
const FSQ_XI: i32 = 256;
/// t0 = a + b and t1 = b*v + a (canonical Fp6 each).
const FSQ_T0: i32 = 320;
const FSQ_T1: i32 = 512;
/// V = a*b mod p and U = t0*t1 mod p (canonical Fp6 each).
const FSQ_V: i32 = 704;
const FSQ_U: i32 = 896;
/// W = t1*v + t1 built here, then y.a = U - W in place. Y.b = 2V follows
/// contiguously so one loop copies both halves out to z.
const FSQ_YA: i32 = 1088;
const FSQ_YB: i32 = 1280;
/// f staged once (48 limbs): all later reads are frame-relative, and z == f
/// becomes trivially safe (no f read after any z store).
const FSQ_FST: i32 = 1472;
/// Karatsuba operand sides: 6 blocks x 96 bytes `[re, im, re+im]` in the
/// order c0, c1, c2, c1+c2, c0+c1, c0+c2.
const FSQ_XSTG: i32 = 1856;
const FSQ_YSTG: i32 = 2432;
/// Product regions: 6 x 192 bytes (d0 +0, d1 +64, d2 +128).
const FSQ_PROD: i32 = 3008;
/// Four 8-limb xi outputs S1, S2, S3, S4 of the nine-fold walk.
const FSQ_SCR: i32 = 4160;
/// Two negated .b lanes (p*2^256 - x) feeding the nine-fold walk.
const FSQ_NB: i32 = 4416;
const FSQ_FRAME: i32 = 4544;

/// Byte offsets of the table regions inside the rodata blob.
const FSQ_TB_CTX: i32 = 0; // 2 rows x 8 u64
const FSQ_TB_ADD: i32 = 128; // 24 rows x 3 (12 per iteration)
const FSQ_TB_SUBS: i32 = 704; // 6 rows x 3
const FSQ_TB_GSUB: i32 = 848; // 32 rows x 3
const FSQ_TB_NINE: i32 = 1616; // 4 rows x 3
const FSQ_TB_GADD: i32 = 1712; // 6 rows x 3
const FSQ_TB_MOD: i32 = 1856; // 6 rows x 2
const FSQ_TB_MSUB: i32 = 1952; // 3 rows x 2
const FSQ_TB_SUMS: i32 = 2000; // 3 rows x 3

const FSQ_TAB_LABEL: &str = ".Lfsq_tab";

/// The read-only walk tables. Offsets are rsp-relative (operands) or
/// blob-relative (the ctx rows' modadd segment field).
fn fp12_sqr_tables() -> Vec<u64> {
    let mut t: Vec<u64> = Vec::new();
    let row3 = |t: &mut Vec<u64>, dst: i32, s1: i32, s2: i32| {
        t.extend([dst as u64, s1 as u64, s2 as u64]);
    };
    // Product regions and their lanes: .a = d0 (+0), .b = d1 (+64).
    let pk = |k: i32| FSQ_PROD + 192 * k;
    let (ad, be, cf, za, zb, zc) = (pk(0), pk(1), pk(2), pk(3), pk(4), pk(5));

    // ctx rows: muxi src/dst, stage x/y, mod dst, modadd segment.
    // Iteration 0 computes V = a*b and prebuilds t0/t1 (t1 needs xi*b.c2).
    // iteration 1 computes U = t0*t1 and prebuilds W and y.b from V.
    t.extend([
        (FSQ_FST + 320) as u64, // b.c2
        FSQ_XI as u64,
        FSQ_FST as u64,
        (FSQ_FST + 192) as u64,
        FSQ_V as u64,
        FSQ_TB_ADD as u64,
        0,
        0,
    ]);
    t.extend([
        (FSQ_V + 128) as u64, // V.c2
        FSQ_XI as u64,
        FSQ_T0 as u64,
        FSQ_T1 as u64,
        FSQ_U as u64,
        (FSQ_TB_ADD + 288) as u64,
        0,
        0,
    ]);

    // Modular single-width adds, iteration 0: t0 = a + b, then
    // t1 = b*v + a = (xi*b.c2 + a.c0, b.c0 + a.c1, b.c1 + a.c2).
    assert_eq!(t.len() * 8, FSQ_TB_ADD as usize);
    for j in 0..6 {
        row3(
            &mut t,
            FSQ_T0 + 32 * j,
            FSQ_FST + 32 * j,
            FSQ_FST + 192 + 32 * j,
        );
    }
    for half in 0..2 {
        row3(
            &mut t,
            FSQ_T1 + 32 * half,
            FSQ_XI + 32 * half,
            FSQ_FST + 32 * half,
        );
    }
    for j in 0..4 {
        row3(
            &mut t,
            FSQ_T1 + 64 + 32 * j,
            FSQ_FST + 192 + 32 * j,
            FSQ_FST + 64 + 32 * j,
        );
    }
    // Iteration 1: W = V*v + V = (xi*V.c2 + V.c0, V.c0 + V.c1, V.c1 + V.c2)
    // into YA, and y.b = 2V into YB.
    for half in 0..2 {
        row3(
            &mut t,
            FSQ_YA + 32 * half,
            FSQ_XI + 32 * half,
            FSQ_V + 32 * half,
        );
    }
    for j in 0..4 {
        row3(
            &mut t,
            FSQ_YA + 64 + 32 * j,
            FSQ_V + 32 * j,
            FSQ_V + 64 + 32 * j,
        );
    }
    for j in 0..6 {
        row3(&mut t, FSQ_YB + 32 * j, FSQ_V + 32 * j, FSQ_V + 32 * j);
    }

    // Modular single-width subs (epilogue): y.a = U - W, in place over YA.
    assert_eq!(t.len() * 8, FSQ_TB_SUBS as usize);
    for j in 0..6 {
        row3(&mut t, FSQ_YA + 32 * j, FSQ_U + 32 * j, FSQ_YA + 32 * j);
    }

    // Double-width subs. First the per-product Karatsuba assembly
    // (d1 -= d0, d1 -= d2 exact. D0 -= d2 guarded), then the cross-term
    // subtractions (a-lane guarded, b-lane exact), then the two negations
    // feeding the nine-fold walk.
    assert_eq!(t.len() * 8, FSQ_TB_GSUB as usize);
    for k in 0..6 {
        let p = pk(k);
        row3(&mut t, p + 64, p + 64, p);
        row3(&mut t, p + 64, p + 64, p + 128);
        row3(&mut t, p, p, p + 128);
    }
    for (dst, src) in [(za, be), (za, cf), (zb, ad), (zb, be), (zc, ad), (zc, cf)] {
        row3(&mut t, dst, dst, src);
        row3(&mut t, dst + 64, dst + 64, src + 64);
    }
    row3(&mut t, FSQ_NB, FSQ_ZERO8, za + 64);
    row3(&mut t, FSQ_NB + 64, FSQ_ZERO8, cf + 64);

    // Nine-fold rows (dst, x, y): dst = 9x + y mod p*2^256, canonical high.
    // S1, S2 = xi*ZA. S3, S4 = xi*CF.
    assert_eq!(t.len() * 8, FSQ_TB_NINE as usize);
    row3(&mut t, FSQ_SCR, za, FSQ_NB);
    row3(&mut t, FSQ_SCR + 64, za + 64, za);
    row3(&mut t, FSQ_SCR + 128, cf, FSQ_NB + 64);
    row3(&mut t, FSQ_SCR + 192, cf + 64, cf);

    // Double-width adds: z.a = xi(ZA) + AD (into S1/S2), z.b = ZB + xi(CF),
    // z.c = ZC + BE.
    assert_eq!(t.len() * 8, FSQ_TB_GADD as usize);
    row3(&mut t, FSQ_SCR, FSQ_SCR, ad);
    row3(&mut t, FSQ_SCR + 64, FSQ_SCR + 64, ad + 64);
    row3(&mut t, zb, zb, FSQ_SCR + 128);
    row3(&mut t, zb + 64, zb + 64, FSQ_SCR + 192);
    row3(&mut t, zc, zc, be);
    row3(&mut t, zc + 64, zc + 64, be + 64);

    // Montgomery reduction rows (src, dst offset relative to the ctx dst).
    assert_eq!(t.len() * 8, FSQ_TB_MOD as usize);
    for (row, src) in [FSQ_SCR, FSQ_SCR + 64, zb, zb + 64, zc, zc + 64]
        .into_iter()
        .enumerate()
    {
        t.extend([src as u64, 32 * row as u64]);
    }

    // Product sub-rows (operand sub-offset, destination sub-offset):
    // d1 = s*s first, then d0 = re*re, then d2 = im*im.
    assert_eq!(t.len() * 8, FSQ_TB_MSUB as usize);
    t.extend([64, 64, 0, 0, 32, 128]);

    // Staging sum rows relative to the side base: block3 = c1 + c2,
    // block4 = c0 + c1, block5 = c0 + c2 (all 12 limbs, s-lanes add).
    assert_eq!(t.len() * 8, FSQ_TB_SUMS as usize);
    row3(&mut t, 288, 96, 192);
    row3(&mut t, 384, 0, 96);
    row3(&mut t, 480, 0, 192);
    assert_eq!(t.len(), 259);
    t
}

/// Load the walk table base and set a walk's cursor and bound:
/// cursor (returned in `cursor`) = table base + `offset`, bound slot
/// `end_slot` = cursor + `bytes`.
fn fsq_walk_setup<M: Machine>(m: &mut M, cursor: Reg, offset: i32, bytes: i32, end_slot: i32) {
    m.load(cursor, Mem::new(Reg::Rsp, FSQ_TBL), "table base");
    m.add_imm(cursor, offset, "walk start");
    m.mov(Rax, cursor, "");
    m.add_imm(Rax, bytes, "walk end");
    m.store(Mem::new(Reg::Rsp, end_slot), Rax, "walk bound");
}

/// Decode a 3-slot walk row at `[rbp]` into rsp-relative pointers:
/// rdi = dst, rsi = source 1, rcx = source 2.
fn fsq_decode3<M: Machine>(m: &mut M) {
    fsq_decode3_at(m, None);
}

/// [`fsq_decode3`] with an optional extra destination base: with
/// `dst_ctx = Some(slot)` the row's dst field is relative to the frame
/// offset held in that ctx slot (the fp12_mul park decode). Sources stay
/// rsp-relative.
fn fsq_decode3_at<M: Machine>(m: &mut M, dst_ctx: Option<i32>) {
    let rsp = Reg::Rsp;
    m.mov(Rdi, rsp, "");
    if let Some(slot) = dst_ctx {
        m.add_mem(Rdi, Mem::new(rsp, slot), "+ ctx destination base");
    }
    m.add_mem(Rdi, Mem::new(Rbp, 0), "dst = rsp + row.dst");
    m.mov(Rsi, rsp, "");
    m.add_mem(Rsi, Mem::new(Rbp, 8), "s1 = rsp + row.s1");
    m.mov(Rcx, rsp, "");
    m.add_mem(Rcx, Mem::new(Rbp, 16), "s2 = rsp + row.s2");
}

/// Load p & mask into `dst` (borrow-mask route of the guarded subtraction).
/// `mask` holds 0 or all-ones.
fn fsq_masked_p<M: Machine>(m: &mut M, dst: Reg, mask: Reg, limb: usize) {
    if dst != mask {
        m.mov(dst, mask, "");
    }
    m.and_mem(
        dst,
        Mem::new(Reg::Rsp, FP6_P + 8 * limb as i32),
        &format!("p{limb} & borrow mask"),
    );
}

/// One dual-chain product row of the 4x4 mulpre: `t[k] += lo_k` on the value
/// chain, `t[k+1] += hi_k` on the carry chain, source limbs from `[y + 8k]`,
/// both chains closed into the top words. Entry/exit: CF = OF = 0.
fn fsq_mulpre_row<M: Machine>(m: &mut M, y: Reg) {
    for k in 0..4 {
        mul_mem_into_columns(
            m,
            Rbx,
            Mem::new(y, 8 * k as i32),
            T[k],
            T[k + 1],
            &format!("x[j]*y[{k}]"),
            k,
        );
    }
    m.mov_zero(LO, "zero for the chain closes (flags preserved)");
    m.adox(T[4], LO, "close the value chain into t4");
    m.adox(T[5], LO, "ripple the t4 close into t5");
    m.adcx(T[5], LO, "close the carry chain into t5");
    m.claim_flags_clear(
        "row peak < 2^257 window + 2^320 product < 2^321, far below the six-word 2^384",
    );
}

/// Copy-subtract-cmov canonicalization of the four-limb value in `v`,
/// through `scratch`, then store at `[base + off .. Off+24]`.
fn fsq_csub_store<M: Machine>(m: &mut M, v: [Reg; 4], scratch: [Reg; 4], base: Reg, off: i32) {
    for (k, (val, s)) in v.iter().zip(scratch).enumerate() {
        m.mov(s, *val, &format!("keep-copy of limb {k}"));
    }
    for (k, s) in scratch.into_iter().enumerate() {
        let what = format!("limb {k} -= p{k}");
        if k == 0 {
            m.sub_mem(s, Mem::new(Reg::Rsp, FP6_P), &what);
        } else {
            m.sbb_mem(s, Mem::new(Reg::Rsp, FP6_P + 8 * k as i32), &what);
        }
    }
    for (k, (val, s)) in v.iter().zip(scratch).enumerate() {
        m.cmov_carry(s, *val, &format!("borrow: value < p, keep limb {k}"));
    }
    for (k, s) in scratch.into_iter().enumerate() {
        m.store(
            Mem::new(base, off + 8 * k as i32),
            s,
            &format!("out limb {k}"),
        );
    }
}

/// mu quotient-estimate reduction of the five-limb value `v` (< 10p) to a
/// canonical residue in `v[0..4]`: E = floor(value/2^252), q = floor(E*mu/
/// 2^58) <= 10, value -= q*p lands below 1.33p, one conditional subtraction.
/// Exactly the fp6 xi bound argument. Clobbers rax, rbx?, no: clobbers
/// `s` (four scratch) plus rdx. Asserts the fifth limb dies.
fn fsq_mu_reduce5<M: Machine>(m: &mut M, v: [Reg; 5], s: [Reg; 5]) {
    let rsp = Reg::Rsp;
    m.comment("estimated quotient: E = floor(value/2^252), q = floor(E*mu/2^58) <= 10");
    m.mov(s[0], v[4], "E builds from the top limbs");
    m.shld_imm(s[0], v[3], 4, "E = top five bits of the value");
    m.load(MULTIPLIER, Mem::new(rsp, FP6_MU), "mu");
    m.mulx(s[1], Rax, s[0], "E*mu (high half zero: E < 2^5, mu < 2^57)");
    m.shr_imm(Rax, 58, "q");
    m.mov(MULTIPLIER, Rax, "q is the multiplicand");
    m.mulx_mem(s[0], Rax, Mem::new(rsp, FP6_P), "q*p0 -> (l0, h0)");
    m.mulx_mem(s[2], s[1], Mem::new(rsp, FP6_P + 8), "q*p1 -> (l1, h1)");
    m.add(s[1], s[0], "l1 += h0");
    m.mulx_mem(s[0], s[3], Mem::new(rsp, FP6_P + 16), "q*p2 -> (l2, h2)");
    m.adc(s[3], s[2], "l2 += h1");
    m.mulx_mem(
        s[2],
        s[4],
        Mem::new(rsp, FP6_P + 24),
        "q*p3 -> (l3, h3); rdx freed",
    );
    m.adc(s[4], s[0], "l3 += h2");
    m.adc_zero(s[2], "h3 += carry; q*p < 11p < 2^260");
    m.sub_rr(v[0], Rax, "value -= q*p, limb 0");
    m.sbb_rr(v[1], s[1], "limb 1");
    m.sbb_rr(v[2], s[3], "limb 2");
    m.sbb_rr(v[3], s[4], "limb 3");
    m.sbb_rr(v[4], s[2], "limb 4");
    m.claim_zero(v[4], "value - q*p < 1.33p < 2^255 fits four limbs");
}

/// `narsil_fp12_sqr_x86`: the whole Fp12 square in mcl's lazy double-width
/// shape -- 36 raw 4x4 products and 12 Montgomery reductions where the
/// composed SoS path pays 84 products and 12 interleaved reductions.
///
/// # Semantics (mcl `Fp12::sqr`, fp_tower.hpp)
///
/// For `f = a + b*w` (`w^2 = v`, `v^3 = xi = 9 + u`):
///
/// * `t0 = a + b`, `t1 = b*v + a = (xi*b.c2 + a.c0, b.c0 + a.c1, b.c1 +
///   a.c2)`, both canonical Fp6.
/// * `V = a*b`, `U = t0*t1`, each one lazy Fp6 product: Karatsuba at both
///   tower levels (6 Fp2Dbl products of 3 raw 4x4 mulpre each = 18 products),
///   all cross-terms held as 512-bit values, ONE Montgomery reduction per
///   output Fp (6 per product).
/// * `z.c1 = 2V`, `z.c0 = U - (V*v + V)`, single-width modular ops.
///
/// # Laziness and the p*2^256 guard
///
/// Unreduced 512-bit values are exact where provably nonnegative (Karatsuba
/// middle terms, the imaginary cross-term lanes) and otherwise guarded
/// mod `p*2^256`: a subtraction that borrows adds p to the HIGH four limbs
/// (one masked add), an addition whose high half reaches p subtracts it
/// (one conditional subtraction). `T = T' mod p*2^256` keeps Montgomery
/// congruence: `T/2^256 = T'/2^256 mod p`.
///
/// # Bounds (BN254: p < 2^253.61, so 4p < 2^256 and K = 2^256, pK < 2^510)
///
/// * staged sums: Fp6-level `b+c` < 2p, in-block `re+im` < 4p < 2^256 --
///   four limbs, no reduction.
/// * raw products: <= (4p)^2 = 16p^2 < 2^512 (needs p < 2^254: two
///   headroom bits, mcl's isLtQuad argument).
/// * Karatsuba middle `d1 - d0 - d2 = ad+bc` <= 8p^2, exact and nonnegative.
///   real lanes `d0 - d2` guarded < pK (valid: subtrahend <= 4p^2 < pK).
/// * cross-term a-lanes stay guarded < pK. B-lanes stay exact <= 4p^2 < pK
///   after their subtractions (the intermediate 8p^2 > pK only ever passes
///   through exact subtractions).
/// * xi on doubles: `9x + y mod pK` splits at the limb-4 boundary. The high
///   part `9*xH + yH + carry < 10p` reduces by the mu quotient estimate
///   (q <= 10, remainder < 1.33p, one conditional subtraction) so the
///   result's high half is canonical -- strictly below every guard bound.
/// * z.c additions peak at 6p^2 < 1.14*pK: one high-half conditional
///   subtraction returns below pK.
/// * every reduced value satisfies T < pK, the Montgomery precondition:
///   `(T + m*p*2^256)/2^256 < 2p`, one conditional subtraction, canonical.
///
/// The interpreter asserts the flag claims on every path. The u512 reference
/// in `kernelgen_verify` asserts each stage bound on random and adversarial
/// inputs.
///
/// # In-place update
///
/// `z == f` is allowed and is the production shape: the prologue stages all
/// of `f` into the frame and `f` is never read again, so no output store can
/// alias a live operand.
///
/// # Structure
///
/// One outer two-iteration loop (iteration 0: V = a*b plus the t0/t1
/// prebuild. Iteration 1: U = t0*t1 plus the W/2V prebuild from V) whose
/// per-iteration pointers come from a ctx row. All linear double-width work
/// runs as table-driven walks over rodata (dst, src1, src2) rows -- guarded
/// sub, nine-fold xi, guarded add, Montgomery reduction, modular single-width
/// add/sub -- so each op kind is emitted once. The same walks are the
/// building blocks for the planned Fp12 full mul (3 Fp6Dbl products),
/// mul_by_034 lazy variant, and the lazy cyclotomic square.
///
/// Arguments: `(z: *mut u64x48, f: *const u64x48, consts: *const { p[4],
/// -p^-1, mu })` in rdi, rsi, rdx. `f` is repr(C) Fp12, canonical. Outputs
/// canonical.
pub fn fp12_sqr_x86<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    let tables = fp12_sqr_tables();
    m.rodata(FSQ_TAB_LABEL, &tables);
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    m.alloc_stack(FSQ_FRAME);
    m.comment(
        "frame: p +0, -p^-1 +32, mu +40, z +48, loop bounds +56..96, ctx slots +104..152, xi scratch +152..320, t0/t1/V/U/y +320..1472, staged f +1472, operand sides +1856, products +3008, xi/neg scratch +4160",
    );
    m.store(Mem::new(rsp, FSQ_Z), Rdi, "spill z");
    for k in 0..4 {
        m.load(Rax, Mem::new(Rdx, 8 * k), &format!("p{k}"));
        m.store(
            Mem::new(rsp, FP6_P + 8 * k),
            Rax,
            "cancel rows address the frame as a consts table",
        );
    }
    m.load(Rax, Mem::new(Rdx, 32), "-p^-1");
    m.store(Mem::new(rsp, FP6_PINV), Rax, "-p^-1");
    m.load(Rax, Mem::new(Rdx, 40), "mu = floor(2^310/p)");
    m.store(Mem::new(rsp, FP6_MU), Rax, "mu");
    m.lea_rodata(Rax, FSQ_TAB_LABEL, "walk tables");
    m.store(Mem::new(rsp, FSQ_TBL), Rax, "table base");
    m.mov(Rbx, Rax, "");
    m.add_imm(Rbx, FSQ_TB_MSUB + 48, "product m-walk end");
    m.store(Mem::new(rsp, FSQ_MEND), Rbx, "");
    m.mov(Rbx, Rax, "");
    m.add_imm(Rbx, 128, "ctx table end (two 64-byte rows)");
    m.store(Mem::new(rsp, FSQ_OUTER_END), Rbx, "");
    m.xor_clear(Rax, "");
    for k in 0..8 {
        m.store(
            Mem::new(rsp, FSQ_ZERO8 + 8 * k),
            Rax,
            "zero word (negation rows subtract from it)",
        );
    }

    m.comment("");
    m.comment("stage f: all later reads are frame-relative, which is what");
    m.comment("makes z == f safe (no f read after any z store)");
    m.mov(Rdi, rsp, "");
    m.add_imm(Rdi, FSQ_FST, "staging cursor");
    m.xor_clear(Rcx, "48 limbs, 4 per iteration");
    m.stride_loop(Rcx, 32, LoopEnd::Imm(384), ".Lfsq_fst", &mut |m| {
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("f limb {k}"));
        }
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.store(Mem::new(Rdi, 8 * k as i32), reg, "staged");
        }
        m.add_imm(Rsi, 32, "");
        m.add_imm(Rdi, 32, "");
    });

    m.comment("");
    m.comment("outer loop: iteration 0 computes V = a*b (and prebuilds t0, t1),");
    m.comment("iteration 1 computes U = t0*t1 (and prebuilds W = V*v + V, 2V)");
    m.load(Rbp, Mem::new(rsp, FSQ_TBL), "ctx cursor = first ctx row");
    m.stride_loop(
        Rbp,
        64,
        LoopEnd::Mem(Mem::new(rsp, FSQ_OUTER_END)),
        ".Lfsq_iter",
        &mut |m| {
            m.store(Mem::new(rsp, FSQ_CTX_SPILL), Rbp, "spill the outer cursor");
            for (field, slot, what) in [
                (0, FSQ_MUXI_SRC, "ctx: xi site source"),
                (8, FSQ_MUXI_DST, "ctx: xi site destination"),
                (16, FSQ_STAGE_X, "ctx: x-side source"),
                (24, FSQ_STAGE_Y, "ctx: y-side source"),
                (32, FSQ_MOD_DST, "ctx: reduction destination"),
                (40, FSQ_ADD_SEG, "ctx: modadd segment"),
            ] {
                m.load(Rax, Mem::new(Rbp, field), what);
                m.store(Mem::new(rsp, slot), Rax, "");
            }
            fsq_muxi(m);
            fsq_modadd_walk(m);
            fsq_stage_sides(m);
            fsq_products(m);
            fsq_gsub_walk(m);
            fsq_nine_walk(m);
            fsq_gadd_walk(m);
            fsq_mod_walk(m);
            m.load(Rbp, Mem::new(rsp, FSQ_CTX_SPILL), "reload the outer cursor");
        },
    );

    m.comment("");
    m.comment("epilogue: y.a = U - W (modular, in place over the W area)");
    fsq_walk_setup(m, Rbp, FSQ_TB_SUBS, 6 * 24, FSQ_WALK_END);
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lfsq_msub",
        &mut |m| dbl_msub_row(m, "y.a"),
    );

    m.comment("");
    m.comment("copy out: y.a then y.b are contiguous, 48 limbs to z");
    m.load(Rdi, Mem::new(rsp, FSQ_Z), "z");
    m.mov(Rsi, rsp, "");
    m.add_imm(Rsi, FSQ_YA, "y.a base");
    m.xor_clear(Rcx, "");
    m.stride_loop(Rcx, 32, LoopEnd::Imm(384), ".Lfsq_out", &mut |m| {
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("y limb {k}"));
        }
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.store(Mem::new(Rdi, 8 * k as i32), reg, "z");
        }
        m.add_imm(Rsi, 32, "");
        m.add_imm(Rdi, 32, "");
    });
    m.free_stack(FSQ_FRAME);
    for reg in CALLEE_SAVED.iter().rev() {
        m.pop(*reg);
    }
    m.ret();
}

/// Single-width xi = 9 + u of the ctx site: dst = (9*re + (p - im),
/// 9*im + re), both halves canonicalized by the mu estimate. The negp
/// subtraction's clean flags also assert the site is canonical.
fn fsq_muxi<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    m.comment("xi site: (9*re - im, 9*im + re), subtraction via p - im");
    m.mov(Rsi, rsp, "");
    m.add_mem(Rsi, Mem::new(rsp, FSQ_MUXI_SRC), "site source Fp2");
    m.mov(Rdi, rsp, "");
    m.add_mem(Rdi, Mem::new(rsp, FSQ_MUXI_DST), "site destination Fp2");
    for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
        m.load(reg, Mem::new(rsp, FP6_P + 8 * k as i32), &format!("p{k}"));
    }
    for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
        let what = format!("p{k} - im[{k}]");
        if k == 0 {
            m.sub_mem(reg, Mem::new(Rsi, 32), &what);
        } else {
            m.sbb_mem(reg, Mem::new(Rsi, 32 + 8 * k as i32), &what);
        }
    }
    m.claim_flags_clear("im < p (canonical site): p - im cannot borrow");
    for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
        m.store(Mem::new(rsp, FSQ_NEGIM + 8 * k as i32), reg, "negp(im)");
    }
    m.xor_clear(Rbx, "half cursor: re output (+0) then im (+32)");
    m.stride_loop(Rbx, 32, LoopEnd::Imm(64), ".Lfsq_muxi", &mut |m| {
        m.comment("value = 9*X + Y: X = site half, Y = negp(im) for re, re for im");
        m.mov(Rcx, Rsi, "");
        m.add(Rcx, Rbx, "X = site + half");
        m.mov(Rbp, rsp, "");
        m.add_imm(Rbp, FSQ_NEGIM, "Y value is negp(im)");
        m.xor_clear(Rax, "");
        m.sub_rr(Rax, Rbx, "CF set exactly on the im half");
        m.cmov_carry(Rbp, Rsi, "im half: Y = site.re");
        m.xor_clear(MULTIPLIER, "");
        m.add_imm(MULTIPLIER, 9, "xi = 9 + u: the scale is one mulx row");
        let v = [R8, R9, R10, R11, R12];
        m.mulx_mem(R13, v[0], Mem::new(Rcx, 0), "9*X[0] -> (v0, hi)");
        m.mulx_mem(R14, v[1], Mem::new(Rcx, 8), "9*X[1] -> (v1, hi)");
        m.add(v[1], R13, "v1 += hi(9*X[0])");
        m.mulx_mem(R13, v[2], Mem::new(Rcx, 16), "9*X[2] -> (v2, hi)");
        m.adc(v[2], R14, "v2 += hi(9*X[1])");
        m.mulx_mem(v[4], v[3], Mem::new(Rcx, 24), "9*X[3] -> (v3, v4)");
        m.adc(v[3], R13, "v3 += hi(9*X[2])");
        m.adc_zero(v[4], "9X < 9p: the chain closes into the top limb");
        for k in 0..4 {
            let what = format!("+= Y[{k}]");
            if k == 0 {
                m.add_mem(v[0], Mem::new(Rbp, 0), &what);
            } else {
                m.adc_mem(v[k], Mem::new(Rbp, 8 * k as i32), &what);
            }
        }
        m.adc_zero(v[4], "value < 10p < 2^257: top limb is at most 2");
        fsq_mu_reduce5(m, v, [R13, R14, R15, Rbp, Rcx]);
        m.comment("one conditional subtraction reaches canonical (< 1.33p)");
        m.mov(R13, Rdi, "");
        m.add(R13, Rbx, "output half");
        fsq_csub_store(
            m,
            [v[0], v[1], v[2], v[3]],
            [Rax, Rcx, Rbp, MULTIPLIER],
            R13,
            0,
        );
    });
}

/// One modular single-width add row: dst = s1 + s2 mod p (all three
/// canonical). Shared by the fp12_sqr and fp12_mul walks.
fn dbl_modadd_row<M: Machine>(m: &mut M) {
    fsq_decode3(m);
    for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
        m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("s1 limb {k}"));
    }
    for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
        let what = format!("+= s2 limb {k}");
        if k == 0 {
            m.add_mem(reg, Mem::new(Rcx, 0), &what);
        } else {
            m.adc_mem(reg, Mem::new(Rcx, 8 * k as i32), &what);
        }
    }
    m.claim_flags_clear("s1 + s2 < 2p < 2^256: no carry out");
    fsq_csub_store(m, [R8, R9, R10, R11], [R12, R13, R14, R15], Rdi, 0);
}

/// One modular single-width subtraction row: dst = s1 - s2 mod p (operands
/// canonical. P returns on borrow). Shared by the fp12_sqr epilogue and the
/// cyc_sqr z-combines. `out` names the destination in the store comments.
fn dbl_msub_row<M: Machine>(m: &mut M, out: &str) {
    fsq_decode3(m);
    for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
        m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("s1 limb {k}"));
    }
    m.xor_clear(Rax, "mask seed; also clears flags for the chain");
    for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
        let what = format!("limb {k} -= s2");
        if k == 0 {
            m.sub_mem(reg, Mem::new(Rcx, 0), &what);
        } else {
            m.sbb_mem(reg, Mem::new(Rcx, 8 * k as i32), &what);
        }
    }
    m.sbb_rr(Rax, Rax, "mask = -borrow");
    fsq_masked_p(m, Rbx, Rax, 0);
    fsq_masked_p(m, Rdx, Rax, 1);
    fsq_masked_p(m, Rsi, Rax, 2);
    fsq_masked_p(m, Rax, Rax, 3);
    m.add(R8, Rbx, "borrow: += p, limb 0");
    m.adc(R9, Rdx, "limb 1");
    m.adc(R10, Rsi, "limb 2");
    m.adc(R11, Rax, "limb 3");
    // The fix-up carries out exactly when it fired (it cancels the borrow).
    // the next row re-seeds its flags, so nothing relies on CF here.
    for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
        m.store(Mem::new(Rdi, 8 * k as i32), reg, &format!("{out} limb {k}"));
    }
}

/// Modular single-width add walk over the iteration's 12-row segment:
/// dst = s1 + s2 mod p (all three canonical).
fn fsq_modadd_walk<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    m.comment("modular add walk: t0/t1 build (iteration 0), W and 2V (iteration 1)");
    m.load(Rbp, Mem::new(rsp, FSQ_TBL), "table base");
    m.add_mem(
        Rbp,
        Mem::new(rsp, FSQ_ADD_SEG),
        "+ this iteration's segment",
    );
    m.mov(Rbx, Rbp, "");
    m.add_imm(Rbx, 12 * 24, "segment end");
    m.store(Mem::new(rsp, FSQ_WALK_END), Rbx, "walk bound");
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lfsq_madd",
        &mut |m| dbl_modadd_row(m),
    );
}

/// Stage both operand sides for the iteration's Fp6 product: per side six
/// 96-byte blocks `[re, im, re+im]` for c0, c1, c2 and the three pairwise
/// Karatsuba sums c1+c2, c0+c1, c0+c2.
fn fsq_stage_sides<M: Machine>(m: &mut M) {
    stage_sides_walk(m, "fsq", FSQ_TB_SUMS);
}

/// [`fsq_stage_sides`] body, shared by the fp12_sqr and fp12_mul kernels:
/// `tag` names the loop labels, `tb_sums` locates the kernel's staging sum
/// rows in its own rodata blob. Frame contract: side sources in the adjacent
/// STAGE_X/STAGE_Y ctx slots, destinations at XSTG then YSTG.
fn stage_sides_walk<M: Machine>(m: &mut M, tag: &str, tb_sums: i32) {
    let rsp = Reg::Rsp;
    m.comment("stage the two operand sides (six blocks each: singles + sums)");
    m.xor_clear(Rax, "");
    m.add_imm(Rax, FSQ_XSTG, "");
    m.store(
        Mem::new(rsp, FSQ_STAGE_CUR),
        Rax,
        "destination: x side first",
    );
    m.mov(Rbx, rsp, "");
    m.add_imm(Rbx, FSQ_STAGE_X, "side cursor walks the two source slots");
    m.mov(Rax, Rbx, "");
    m.add_imm(Rax, 16, "");
    m.store(Mem::new(rsp, FSQ_WALK_END), Rax, "side bound");
    m.stride_loop(
        Rbx,
        8,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        &format!(".L{tag}_side"),
        &mut |m| {
            m.mov(Rsi, rsp, "");
            m.add_mem(Rsi, Mem::new(Rbx, 0), "side source (one Fp6)");
            m.mov(Rdi, rsp, "");
            m.add_mem(Rdi, Mem::new(rsp, FSQ_STAGE_CUR), "side destination");
            m.comment("singles: copy each Fp2 and add its in-block s = re + im");
            m.xor_clear(Rcx, "");
            m.stride_loop(
                Rcx,
                64,
                LoopEnd::Imm(192),
                &format!(".L{tag}_single"),
                &mut |m| {
                    for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
                        m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("re[{k}]"));
                    }
                    for (k, reg) in [R12, R13, R14, R15].into_iter().enumerate() {
                        m.load(reg, Mem::new(Rsi, 32 + 8 * k as i32), &format!("im[{k}]"));
                    }
                    for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
                        m.store(Mem::new(Rdi, 8 * k as i32), reg, &format!("block re[{k}]"));
                    }
                    for (k, reg) in [R12, R13, R14, R15].into_iter().enumerate() {
                        m.store(
                            Mem::new(Rdi, 32 + 8 * k as i32),
                            reg,
                            &format!("block im[{k}]"),
                        );
                    }
                    m.add(R8, R12, "s = re + im, limb 0");
                    m.adc(R9, R13, "limb 1");
                    m.adc(R10, R14, "limb 2");
                    m.adc(R11, R15, "limb 3");
                    m.claim_flags_clear("re + im < 2p < 2^256: s fits four limbs");
                    for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
                        m.store(
                            Mem::new(Rdi, 64 + 8 * k as i32),
                            reg,
                            &format!("block s[{k}]"),
                        );
                    }
                    m.add_imm(Rsi, 64, "next source Fp2");
                    m.add_imm(Rdi, 96, "next block");
                },
            );
            m.comment("sums: whole-block adds (s-lanes add to the sums' s)");
            fsq_walk_setup(m, MULTIPLIER, tb_sums, 3 * 24, FSQ_WALK_END2);
            m.stride_loop(
                MULTIPLIER,
                24,
                LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END2)),
                &format!(".L{tag}_sums"),
                &mut |m| {
                    m.mov(Rbp, rsp, "");
                    m.add_mem(Rbp, Mem::new(rsp, FSQ_STAGE_CUR), "side base");
                    m.mov(Rdi, Rbp, "");
                    m.add_mem(Rdi, Mem::new(MULTIPLIER, 0), "sum block");
                    m.mov(Rsi, Rbp, "");
                    m.add_mem(Rsi, Mem::new(MULTIPLIER, 8), "addend block A");
                    m.mov(Rcx, Rbp, "");
                    m.add_mem(Rcx, Mem::new(MULTIPLIER, 16), "addend block B");
                    m.xor_clear(Rbp, "row cursor: re, im, s");
                    m.stride_loop(
                        Rbp,
                        32,
                        LoopEnd::Imm(96),
                        &format!(".L{tag}_sumrow"),
                        &mut |m| {
                            for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
                                m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("A limb {k}"));
                            }
                            for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
                                let what = format!("+= B limb {k}");
                                if k == 0 {
                                    m.add_mem(reg, Mem::new(Rcx, 0), &what);
                                } else {
                                    m.adc_mem(reg, Mem::new(Rcx, 8 * k as i32), &what);
                                }
                            }
                            // No flags claim: s-lane sums reach 4p, whose top
                            // limb crosses the sign bit (OF may set legally).
                            // CF stays clear since 4p < 2^256, and the u512
                            // reference asserts the value bound.
                            for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
                                m.store(Mem::new(Rdi, 8 * k as i32), reg, "sum limb");
                            }
                            m.add_imm(Rsi, 32, "");
                            m.add_imm(Rcx, 32, "");
                            m.add_imm(Rdi, 32, "");
                        },
                    );
                },
            );
            m.load(Rax, Mem::new(rsp, FSQ_STAGE_CUR), "");
            m.add_imm(Rax, 576, "");
            m.store(
                Mem::new(rsp, FSQ_STAGE_CUR),
                Rax,
                "destination: y side next",
            );
        },
    );
}

/// The 18 raw 4x4 products of one Fp6Dbl product: block k of the x side
/// times block k of the y side, three sub-products each (s*s, re*re, im*im),
/// every result a full 512-bit value in the product region.
fn fsq_products<M: Machine>(m: &mut M) {
    products_walk(m, "fsq", FSQ_TB_MSUB);
}

/// [`fsq_products`] body, shared by the fp12_sqr and fp12_mul kernels:
/// `tag` names the loop labels, `tb_msub` locates the kernel's sub-product
/// rows in its own rodata blob (the FSQ_MEND slot must hold their end).
fn products_walk<M: Machine>(m: &mut M, tag: &str, tb_msub: i32) {
    let rsp = Reg::Rsp;
    m.comment("products: 6 blocks x 3 sub-products, rolled 4x4 mulpre rounds");
    m.xor_clear(R15, "block cursor 96k");
    m.stride_loop(
        R15,
        96,
        LoopEnd::Imm(576),
        &format!(".L{tag}_prod_k"),
        &mut |m| {
            m.load(Rcx, Mem::new(rsp, FSQ_TBL), "");
            m.add_imm(Rcx, tb_msub, "sub-product walk");
            m.stride_loop(
                Rcx,
                16,
                LoopEnd::Mem(Mem::new(rsp, FSQ_MEND)),
                &format!(".L{tag}_prod_m"),
                &mut |m| {
                    m.load(Rax, Mem::new(Rcx, 0), "operand sub-offset");
                    m.load(Rbx, Mem::new(Rcx, 8), "destination sub-offset");
                    m.mov(Rsi, rsp, "");
                    m.add(Rsi, R15, "");
                    m.add(Rsi, Rax, "");
                    m.add_imm(Rsi, FSQ_XSTG, "PA: x sub-row (the multiplicand)");
                    m.mov(Rdi, rsp, "");
                    m.add(Rdi, R15, "");
                    m.add(Rdi, Rax, "");
                    m.add_imm(Rdi, FSQ_YSTG, "PY: y sub-row");
                    m.mov(Rbp, rsp, "");
                    m.add(Rbp, R15, "");
                    m.add(Rbp, R15, "product regions stride 192 = 2*96k");
                    m.add(Rbp, Rbx, "");
                    m.add_imm(Rbp, FSQ_PROD, "PZ");
                    for (k, t) in T.into_iter().enumerate() {
                        m.xor_clear(t, &format!("t{k} = 0"));
                    }
                    m.xor_clear(R14, "round cursor: byte offset 8j of the x limb");
                    m.stride_loop(
                        R14,
                        8,
                        LoopEnd::Imm(32),
                        &format!(".L{tag}_prod_j"),
                        &mut |m| {
                            m.load_indexed(MULTIPLIER, Rsi, R14, "x[j], the row multiplicand");
                            m.xor_clear(LO, "re-seed CF = OF = 0 (back edge clobbered flags)");
                            fsq_mulpre_row(m, Rdi);
                            m.store(Mem::new(Rbp, 0), T[0], "product limb j is final");
                            m.add_imm(Rbp, 8, "next output limb");
                            m.comment("shift down one word");
                            for k in 0..5 {
                                m.mov(T[k], T[k + 1], &format!("t{k} = t{}", k + 1));
                            }
                            m.xor_clear(T[5], "t5 = 0 (CF/OF stay clear)");
                        },
                    );
                    for (k, t) in T[..4].iter().enumerate() {
                        m.store(
                            Mem::new(Rbp, 8 * k as i32),
                            *t,
                            &format!("product limb {}", k + 4),
                        );
                    }
                },
            );
        },
    );
}

/// One guarded double-width subtraction row: dst = s1 - s2, plus p on the
/// HIGH four limbs when the subtraction borrows (mod p*2^256. A no-op mask
/// for the provably nonnegative rows). Shared by fp12_sqr and fp12_mul.
fn dbl_gsub_row<M: Machine>(m: &mut M) {
    fsq_decode3(m);
    let v = [R8, R9, R10, R11, R12, R13, R14, R15];
    for (k, reg) in v.into_iter().enumerate() {
        m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("s1 word {k}"));
    }
    m.xor_clear(Rax, "mask seed; also clears flags for the chain");
    for (k, reg) in v.into_iter().enumerate() {
        let what = format!("word {k} -= s2");
        if k == 0 {
            m.sub_mem(reg, Mem::new(Rcx, 0), &what);
        } else {
            m.sbb_mem(reg, Mem::new(Rcx, 8 * k as i32), &what);
        }
    }
    m.sbb_rr(Rax, Rax, "mask = -borrow");
    fsq_masked_p(m, Rbx, Rax, 0);
    fsq_masked_p(m, MULTIPLIER, Rax, 1);
    fsq_masked_p(m, Rsi, Rax, 2);
    fsq_masked_p(m, Rax, Rax, 3);
    m.add(R12, Rbx, "borrow: high half += p, word 4");
    m.adc(R13, MULTIPLIER, "word 5");
    m.adc(R14, Rsi, "word 6");
    m.adc(R15, Rax, "word 7");
    // The fix-up carries out exactly when it fired (it cancels the
    // borrow). The next row re-seeds its flags. Result: s1 - s2, or
    // s1 - s2 + p*2^256 on borrow, both below 2^512.
    for (k, reg) in v.into_iter().enumerate() {
        m.store(Mem::new(Rdi, 8 * k as i32), reg, &format!("dst word {k}"));
    }
}

/// Guarded double-width subtraction walk: dst = s1 - s2, plus p on the HIGH
/// four limbs when the subtraction borrows (mod p*2^256. A no-op mask for
/// the provably nonnegative rows).
fn fsq_gsub_walk<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    m.comment("double-width sub walk: Karatsuba assembly, cross terms, negations");
    fsq_walk_setup(m, Rbp, FSQ_TB_GSUB, 32 * 24, FSQ_WALK_END);
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lfsq_gsub",
        &mut |m| dbl_gsub_row(m),
    );
}

/// One nine-fold xi row: dst = 9*x + y mod p*2^256 for 512-bit x, y. Splits
/// at the limb-4 boundary: the low half is exact (its carry joins the high),
/// the high half 9*xH + yH + carry < 10p reduces by the mu estimate, so the
/// stored high half is canonical. Shared by fp12_sqr and fp12_mul.
fn dbl_nine_row<M: Machine>(m: &mut M) {
    fsq_decode3(m);
    m.xor_clear(MULTIPLIER, "");
    m.add_imm(MULTIPLIER, 9, "9 is the mulx multiplicand");
    m.comment("low half: l = 9*xL + yL, carry limb l4 <= 10");
    m.mulx_mem(R13, R8, Mem::new(Rsi, 0), "9*x0 -> (l0, hi)");
    m.mulx_mem(R14, R9, Mem::new(Rsi, 8), "9*x1 -> (l1, hi)");
    m.add(R9, R13, "l1 += hi(9*x0)");
    m.mulx_mem(R13, R10, Mem::new(Rsi, 16), "9*x2 -> (l2, hi)");
    m.adc(R10, R14, "l2 += hi(9*x1)");
    m.mulx_mem(R12, R11, Mem::new(Rsi, 24), "9*x3 -> (l3, l4)");
    m.adc(R11, R13, "l3 += hi(9*x2)");
    m.adc_zero(R12, "9*xL < 9*2^256: l4 closes the chain");
    for k in 0..4 {
        let what = format!("+= y{k}");
        if k == 0 {
            m.add_mem(R8, Mem::new(Rcx, 0), &what);
        } else {
            m.adc_mem([R8, R9, R10, R11][k], Mem::new(Rcx, 8 * k as i32), &what);
        }
    }
    m.adc_zero(R12, "l4 <= 10");
    for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
        m.store(Mem::new(Rdi, 8 * k as i32), reg, &format!("dst word {k}"));
    }
    m.comment("high half: v = 9*xH + yH + l4 < 10p (xH, yH < p)");
    m.mulx_mem(R13, R8, Mem::new(Rsi, 32), "9*x4 -> (v0, hi)");
    m.mulx_mem(R14, R9, Mem::new(Rsi, 40), "9*x5 -> (v1, hi)");
    m.add(R9, R13, "v1 += hi(9*x4)");
    m.mulx_mem(R13, R10, Mem::new(Rsi, 48), "9*x6 -> (v2, hi)");
    m.adc(R10, R14, "v2 += hi(9*x5)");
    m.mulx_mem(R15, R11, Mem::new(Rsi, 56), "9*x7 -> (v3, v4)");
    m.adc(R11, R13, "v3 += hi(9*x6)");
    m.adc_zero(R15, "9*xH < 9p closes into v4");
    m.add(R8, R12, "+= l4");
    for reg in [R9, R10, R11, R15] {
        m.adc_zero(reg, "ripple the l4 carry");
    }
    for k in 0..4 {
        let what = format!("+= y{}", k + 4);
        if k == 0 {
            m.add_mem(R8, Mem::new(Rcx, 32), &what);
        } else {
            m.adc_mem(
                [R8, R9, R10, R11][k],
                Mem::new(Rcx, 32 + 8 * k as i32),
                &what,
            );
        }
    }
    m.adc_zero(R15, "v < 10p < 2^257");
    fsq_mu_reduce5(m, [R8, R9, R10, R11, R15], [R13, R14, R12, Rbx, Rsi]);
    m.comment("one conditional subtraction: the stored high half is canonical");
    fsq_csub_store(m, [R8, R9, R10, R11], [Rax, Rbx, Rcx, Rsi], Rdi, 32);
}

/// Nine-fold xi walk: dst = 9*x + y mod p*2^256 for 512-bit x, y (see
/// [`dbl_nine_row`]).
fn fsq_nine_walk<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    m.comment("nine-fold walk: xi = 9 + u on 512-bit values, mu-canonical high");
    fsq_walk_setup(m, Rbp, FSQ_TB_NINE, 4 * 24, FSQ_WALK_END);
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lfsq_nine",
        &mut |m| dbl_nine_row(m),
    );
}

/// One guarded double-width addition row: dst = s1 + s2, minus p on the
/// HIGH four limbs when they reach p (mod p*2^256). Shared by fp12_sqr and
/// fp12_mul. `dst_ctx` is the fp12_mul park decode (see
/// [`fsq_decode3_at`]).
fn dbl_gadd_row<M: Machine>(m: &mut M, dst_ctx: Option<i32>) {
    let rsp = Reg::Rsp;
    fsq_decode3_at(m, dst_ctx);
    let v = [R8, R9, R10, R11, R12, R13, R14, R15];
    for (k, reg) in v.into_iter().enumerate() {
        m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("s1 word {k}"));
    }
    for (k, reg) in v.into_iter().enumerate() {
        let what = format!("word {k} += s2");
        if k == 0 {
            m.add_mem(reg, Mem::new(Rcx, 0), &what);
        } else {
            m.adc_mem(reg, Mem::new(Rcx, 8 * k as i32), &what);
        }
    }
    m.claim_flags_clear("sum of two sub-2^510 values < 2^511: no carry out");
    m.comment("high half >= p: subtract p once (sum < 2p*2^256)");
    m.mov(Rax, R12, "");
    m.mov(Rbx, R13, "");
    m.mov(MULTIPLIER, R14, "");
    m.mov(Rcx, R15, "");
    m.sub_mem(Rax, Mem::new(rsp, FP6_P), "high word 0 - p0");
    m.sbb_mem(Rbx, Mem::new(rsp, FP6_P + 8), "high word 1 - p1");
    m.sbb_mem(MULTIPLIER, Mem::new(rsp, FP6_P + 16), "high word 2 - p2");
    m.sbb_mem(Rcx, Mem::new(rsp, FP6_P + 24), "high word 3 - p3");
    m.cmov_carry(Rax, R12, "borrow: high < p, keep");
    m.cmov_carry(Rbx, R13, "");
    m.cmov_carry(MULTIPLIER, R14, "");
    m.cmov_carry(Rcx, R15, "");
    for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
        m.store(Mem::new(Rdi, 8 * k as i32), reg, &format!("dst word {k}"));
    }
    for (k, reg) in [Rax, Rbx, MULTIPLIER, Rcx].into_iter().enumerate() {
        m.store(
            Mem::new(Rdi, 32 + 8 * k as i32),
            reg,
            &format!("dst word {}", k + 4),
        );
    }
}

/// Guarded double-width addition walk: dst = s1 + s2, minus p on the HIGH
/// four limbs when they reach p (mod p*2^256).
fn fsq_gadd_walk<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    m.comment("double-width add walk: xi'd cross terms into the output lanes");
    fsq_walk_setup(m, Rbp, FSQ_TB_GADD, 6 * 24, FSQ_WALK_END);
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lfsq_gadd",
        &mut |m| dbl_gadd_row(m, None),
    );
}

/// One Montgomery reduction row: reduces one 512-bit value T < p*2^256 to
/// the canonical residue T/2^256 mod p at [MOD_DST ctx slot] + row offset.
/// Shared by fp12_sqr and fp12_mul.
fn dbl_mod_row<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    m.mov(Rsi, rsp, "");
    m.add_mem(Rsi, Mem::new(Rbp, 0), "source T");
    m.mov(Rdi, rsp, "");
    m.add_mem(Rdi, Mem::new(rsp, FSQ_MOD_DST), "V or U base");
    m.add_mem(Rdi, Mem::new(Rbp, 8), "+ coefficient offset");
    let t = [R8, R9, R10, R11, R12, R13, R14, R15];
    for (k, reg) in t.into_iter().enumerate() {
        m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("T{k}"));
    }
    m.xor_clear(LO, "clear CF = OF before the dual chains");
    for round in 0..4 {
        if round > 0 {
            m.claim_flags_clear("previous row rippled both chains out under the 2pK bound");
        }
        let window: [Reg; 5] = core::array::from_fn(|k| t[round + k]);
        cancel_low_word_at(m, window, Rbx, rsp, &round.to_string());
        for (k, &word) in t.iter().enumerate().skip(round + 5) {
            m.adcx(word, LO, &format!("T{k} += carry-chain ripple"));
            m.adox(word, LO, &format!("T{k} += value-chain ripple"));
        }
    }
    m.claim_flags_clear("T < p*2^256 keeps the total below 2p*2^256: no word beyond T7");
    m.comment("result T4..T7 < 2p: one conditional subtraction");
    fsq_csub_store(m, [R12, R13, R14, R15], [Rax, Rbx, MULTIPLIER, Rsi], Rdi, 0);
}

/// Montgomery reduction walk: each row reduces one 512-bit value T < p*2^256
/// to the canonical residue T/2^256 mod p at the ctx destination.
fn fsq_mod_walk<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    m.comment("Montgomery reduction walk: 6 coefficients, 4 cancel rows each");
    fsq_walk_setup(m, Rbp, FSQ_TB_MOD, 6 * 16, FSQ_WALK_END);
    m.stride_loop(
        Rbp,
        16,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lfsq_mod",
        &mut |m| dbl_mod_row(m),
    );
}

/// Register roles for `narsil_fp12_mul_x86` (the fp12_sqr roles verbatim:
/// every phase is a shared walk body).
pub const FP12_MUL_REGISTER_MAP: &[(Reg, &str)] = &[
    (
        Rdi,
        "z on entry (spilled); walk destination pointer in every table-driven pass",
    ),
    (
        Rsi,
        "a pointer on entry (staging source); walk source-1 pointer; mask scratch",
    ),
    (
        Rdx,
        "b pointer on entry (staging source); the implicit mulx multiplicand; walk scratch",
    ),
    (
        Rcx,
        "consts pointer on entry; walk source-2 pointer; product m-walk cursor; staging cursor",
    ),
    (R8, "accumulator/value word 0"),
    (R9, "accumulator/value word 1"),
    (R10, "accumulator/value word 2"),
    (R11, "accumulator/value word 3"),
    (R12, "accumulator/value word 4"),
    (R13, "accumulator/value word 5; mu-reduction scratch"),
    (
        R14,
        "product round cursor (byte offset 8j); 8-limb value word 6",
    ),
    (R15, "product block cursor 96k; 8-limb value word 7"),
    (
        Rbp,
        "outer iteration cursor (ctx row address, spilled per phase); every walk's row cursor",
    ),
    (
        Rax,
        "low half of the current product; zero for chain closes; borrow mask",
    ),
    (
        Rbx,
        "high half of the current product; walk bound and half cursors",
    ),
];

// fp12_mul frame layout. Every slot and region a shared walk body addresses
// (p/-p^-1/mu, z, loop bounds, ctx spill, table base, product m-walk end,
// STAGE_X/STAGE_Y, MOD_DST, STAGE_CUR, ZERO8, XSTG/YSTG, PROD, SCR, NB)
// sits at its fp12_sqr offset, so the bodies emit unchanged. Only the
// mul-specific regions differ.
/// Ctx: the iteration's Fp6Dbl park base (frame offset).
const FMU_PARK: i32 = 104;
/// Ctx: per-iteration gsub/nine/gadd walk end offsets (table-relative).
const FMU_GSUB_END: i32 = 112;
const FMU_NINE_END: i32 = 144;
const FMU_GADD_END: i32 = 152;
/// p*2^256 - BD.c.im, the negation feeding the mulVadd xi rows.
const FMU_NB2: i32 = 256;
/// a staged (48 limbs) then b staged (48 limbs), contiguous.
const FMU_AST: i32 = 320;
const FMU_BST: i32 = 704;
/// t1 = a0 + a1 and t2 = b0 + b1 (canonical Fp6 each).
const FMU_T1: i32 = 1088;
const FMU_T2: i32 = 1280;
/// The twelve reduced output coefficients (z.a then z.b), copied to z last.
const FMU_YOUT: i32 = 1472;
/// Fp6Dbl parks: AC = a0*b0, BD = a1*b1, CR = t1*t2, TA = the assembled z.a.
/// each six 64-byte lanes in order a.re, a.im, b.re, b.im, c.re, c.im.
const FMU_AC: i32 = 4544;
const FMU_BD: i32 = 4928;
const FMU_CR: i32 = 5312;
const FMU_TA: i32 = 5696;
const FMU_FRAME: i32 = 6080;

/// Byte offsets of the table regions inside the fp12_mul rodata blob.
const FMU_TB_CTX: i32 = 0; // 3 rows x 6
const FMU_TB_MADD: i32 = 144; // 12 rows x 3
const FMU_TB_GSUB: i32 = 432; // 33 rows x 3 (row 32: iteration 2 only)
const FMU_TB_NINE: i32 = 1224; // 6 rows x 3 (rows 4..6: iteration 2 only)
const FMU_TB_GADD: i32 = 1368; // 12 rows x 3 (rows 6..12: iteration 2 only)
const FMU_TB_GSUB2: i32 = 1656; // 12 rows x 3
const FMU_TB_MOD: i32 = 1944; // 12 rows x 2
const FMU_TB_MSUB: i32 = 2136; // 3 rows x 2
const FMU_TB_SUMS: i32 = 2184; // 3 rows x 3

const FMU_TAB_LABEL: &str = ".Lfmu_tab";

/// The fp12_mul read-only walk tables. Offsets are rsp-relative (operands),
/// blob-relative (the ctx rows' walk ends), or PARK-relative (the gadd
/// destinations, decoded through the ctx park slot).
fn fp12_mul_tables() -> Vec<u64> {
    let mut t: Vec<u64> = Vec::new();
    let row3 = |t: &mut Vec<u64>, dst: i32, s1: i32, s2: i32| {
        t.extend([dst as u64, s1 as u64, s2 as u64]);
    };
    let pk = |k: i32| FSQ_PROD + 192 * k;
    let (ad, be, cf, za, zb, zc) = (pk(0), pk(1), pk(2), pk(3), pk(4), pk(5));

    // Ctx rows: stage x/y sources, Fp6Dbl park, then the gsub/nine/gadd walk
    // end offsets. Iteration 2 extends each walk with the z.a = mulVadd(BD,
    // AC) rows -- exactly the rows whose operands (AC, BD) are parked by
    // then. Z.b needs CR too and runs in the epilogue.
    let ends = |t: &mut Vec<u64>, last: bool| {
        let (g, n, a) = if last { (33, 6, 12) } else { (32, 4, 6) };
        t.extend([
            (FMU_TB_GSUB + 24 * g) as u64,
            (FMU_TB_NINE + 24 * n) as u64,
            (FMU_TB_GADD + 24 * a) as u64,
        ]);
    };
    t.extend([FMU_AST as u64, FMU_BST as u64, FMU_AC as u64]);
    ends(&mut t, false);
    t.extend([
        (FMU_AST + 192) as u64,
        (FMU_BST + 192) as u64,
        FMU_BD as u64,
    ]);
    ends(&mut t, false);
    t.extend([FMU_T1 as u64, FMU_T2 as u64, FMU_CR as u64]);
    ends(&mut t, true);

    // Modular single-width adds: t1 = a0 + a1, t2 = b0 + b1.
    assert_eq!(t.len() * 8, FMU_TB_MADD as usize);
    for j in 0..6 {
        row3(
            &mut t,
            FMU_T1 + 32 * j,
            FMU_AST + 32 * j,
            FMU_AST + 192 + 32 * j,
        );
    }
    for j in 0..6 {
        row3(
            &mut t,
            FMU_T2 + 32 * j,
            FMU_BST + 32 * j,
            FMU_BST + 192 + 32 * j,
        );
    }

    // Double-width subs: the fp12_sqr mulPre rows verbatim (Karatsuba
    // assembly, cross terms, NB negations), plus the iteration-2-only
    // negation of the parked BD.c.im feeding the mulVadd xi.
    assert_eq!(t.len() * 8, FMU_TB_GSUB as usize);
    for k in 0..6 {
        let p = pk(k);
        row3(&mut t, p + 64, p + 64, p);
        row3(&mut t, p + 64, p + 64, p + 128);
        row3(&mut t, p, p, p + 128);
    }
    for (dst, src) in [(za, be), (za, cf), (zb, ad), (zb, be), (zc, ad), (zc, cf)] {
        row3(&mut t, dst, dst, src);
        row3(&mut t, dst + 64, dst + 64, src + 64);
    }
    row3(&mut t, FSQ_NB, FSQ_ZERO8, za + 64);
    row3(&mut t, FSQ_NB + 64, FSQ_ZERO8, cf + 64);
    row3(&mut t, FMU_NB2, FSQ_ZERO8, FMU_BD + 320);

    // Nine-fold rows: S1..S4 as in fp12_sqr, then (iteration 2 only) the
    // mulVadd xi of the parked BD.c into TA's first lane pair.
    assert_eq!(t.len() * 8, FMU_TB_NINE as usize);
    row3(&mut t, FSQ_SCR, za, FSQ_NB);
    row3(&mut t, FSQ_SCR + 64, za + 64, za);
    row3(&mut t, FSQ_SCR + 128, cf, FSQ_NB + 64);
    row3(&mut t, FSQ_SCR + 192, cf + 64, cf);
    row3(&mut t, FMU_TA, FMU_BD + 256, FMU_NB2);
    row3(&mut t, FMU_TA + 64, FMU_BD + 320, FMU_BD + 256);

    // Double-width adds, destinations PARK-relative (ctx dst decode).
    // Rows 0..6: assemble and park the Fp6Dbl product (fp12_sqr's adds, but
    // parked instead of in place). Rows 6..12 (iteration 2, PARK = CR):
    // z.a = mulVadd(BD, AC) into TA -- xi(BD.c) + AC.a in place over the
    // nine outputs, BD.a + AC.b and BD.b + AC.c into TA's b/c lanes (BD
    // itself stays intact for the epilogue z.b subtractions).
    assert_eq!(t.len() * 8, FMU_TB_GADD as usize);
    row3(&mut t, 0, FSQ_SCR, ad);
    row3(&mut t, 64, FSQ_SCR + 64, ad + 64);
    row3(&mut t, 128, zb, FSQ_SCR + 128);
    row3(&mut t, 192, zb + 64, FSQ_SCR + 192);
    row3(&mut t, 256, zc, be);
    row3(&mut t, 320, zc + 64, be + 64);
    for (i, (s1, s2)) in [
        (FMU_TA, FMU_AC),
        (FMU_TA + 64, FMU_AC + 64),
        (FMU_BD, FMU_AC + 128),
        (FMU_BD + 64, FMU_AC + 192),
        (FMU_BD + 128, FMU_AC + 256),
        (FMU_BD + 192, FMU_AC + 320),
    ]
    .into_iter()
    .enumerate()
    {
        row3(&mut t, FMU_TA - FMU_CR + 64 * i as i32, s1, s2);
    }

    // Epilogue double-width subs: z.b = CR - AC - BD, every lane guarded
    // (the parked lanes are congruences mod p*2^256, not exact values).
    assert_eq!(t.len() * 8, FMU_TB_GSUB2 as usize);
    for lane in 0..6 {
        let off = 64 * lane;
        row3(&mut t, FMU_CR + off, FMU_CR + off, FMU_AC + off);
        row3(&mut t, FMU_CR + off, FMU_CR + off, FMU_BD + off);
    }

    // MOD_DST is zero. The destination offsets are absolute.
    assert_eq!(t.len() * 8, FMU_TB_MOD as usize);
    for i in 0..6 {
        t.extend([(FMU_TA + 64 * i) as u64, (FMU_YOUT + 32 * i) as u64]);
    }
    for i in 0..6 {
        t.extend([(FMU_CR + 64 * i) as u64, (FMU_YOUT + 192 + 32 * i) as u64]);
    }

    // Product sub-rows and staging sum rows: identical to fp12_sqr's.
    assert_eq!(t.len() * 8, FMU_TB_MSUB as usize);
    t.extend([64, 64, 0, 0, 32, 128]);
    assert_eq!(t.len() * 8, FMU_TB_SUMS as usize);
    row3(&mut t, 288, 96, 192);
    row3(&mut t, 384, 0, 96);
    row3(&mut t, 480, 0, 192);
    assert_eq!(t.len(), 282);
    t
}

/// Walk setup whose end offset (table-relative) comes from a ctx slot:
/// cursor = table base + `start`, bound slot `end_slot` = base + [ctx_end].
fn fmu_walk_setup_ctx<M: Machine>(m: &mut M, cursor: Reg, start: i32, ctx_end: i32, end_slot: i32) {
    let rsp = Reg::Rsp;
    m.load(cursor, Mem::new(rsp, FSQ_TBL), "table base");
    m.mov(Rax, cursor, "");
    m.add_mem(Rax, Mem::new(rsp, ctx_end), "+ ctx walk end offset");
    m.store(Mem::new(rsp, end_slot), Rax, "walk bound");
    m.add_imm(cursor, start, "walk start");
}

/// `narsil_fp12_mul_x86`: the whole Fp12 product in mcl's lazy double-width
/// shape -- 54 raw 4x4 products and 12 Montgomery reductions where the
/// composed path (Karatsuba over three Fp6 products) pays 108 products and
/// 18 reductions.
///
/// # Semantics (mcl `Fp12::mul`, fp_tower.hpp)
///
/// For `x = a0 + a1*w`, `y = b0 + b1*w` (`w^2 = v`, `v^3 = xi = 9 + u`):
///
/// * `t1 = a0 + a1`, `t2 = b0 + b1`, canonical Fp6 (modular adds).
/// * `AC = a0*b0`, `BD = a1*b1`, `CR = t1*t2`: three Fp6Dbl::mulPre, each
///   the fp12_sqr iteration verbatim (Karatsuba at both tower levels, 18 raw
///   products, cross terms held as 512-bit values) but WITHOUT the per-
///   iteration reduction -- the six coefficient lanes park as 512-bit
///   values < p*2^256 with canonical high halves.
/// * `z.a = mod(mulVadd(BD, AC))`: `(xi*BD.c + AC.a, BD.a + AC.b,
///   BD.b + AC.c)` on doubles (one xi via the nine walk, six guarded adds),
///   then six reductions.
/// * `z.b = mod(CR - AC - BD)`: twelve guarded subtractions, six reductions.
///
/// # Bounds (BN254: p < 2^253.61, K = 2^256, pK < 2^510)
///
/// Inside each mulPre iteration the fp12_sqr bounds hold verbatim (staged
/// sums < 4p < 2^256, raw products <= 16p^2 < 2^512, Karatsuba middles
/// exact <= 8p^2, guarded lanes < pK, nine-walk highs < 10p reduced
/// canonical). The mul-specific stages:
///
/// * parked lanes: every gadd output is < pK with a canonical (< p) high
///   half -- the guard subtracts p from the high half exactly when it
///   reaches p.
/// * `NB2 = 0 - BD.c.im mod pK < pK`, high half < p, so both mulVadd nine
///   rows meet the nine-walk precondition (operand highs < p).
/// * mulVadd adds: two sub-pK addends < 2pK < 2^511. The high-half guard
///   returns below pK.
/// * z.b subtractions: minuend and subtrahend < pK, guarded difference
///   < pK.
/// * every reduced value satisfies T < pK, the Montgomery precondition:
///   `(T + m*p*2^256)/2^256 < 2p`, one conditional subtraction, canonical.
///
/// The interpreter asserts the flag claims on every path. The u512 reference
/// in `kernelgen_verify` asserts each stage bound on random and adversarial
/// inputs.
///
/// # Aliasing
///
/// `z == a`, `z == b` and `a == b` are all allowed (`z == a` is the
/// production MulAssign shape): the prologue stages both operands into the
/// frame and neither is read again, so no output store can alias a live
/// operand.
///
/// # Structure
///
/// One outer three-iteration loop (AC, BD, CR) whose per-iteration pointers
/// and walk bounds come from a ctx row. Every phase is a walk body shared
/// with fp12_sqr (stage sides, products, guarded sub, nine-fold xi, guarded
/// add with the park destination decode, Montgomery reduction, modular
/// add). Iteration 2's gsub/nine/gadd segments are extended with the
/// mulVadd rows for z.a (their operands AC and BD are parked by then), so
/// the epilogue needs only the z.b subtractions, the twelve reductions and
/// the copy-out -- no walk body is emitted twice except gsub.
///
/// Arguments: `(z: *mut u64x48, a: *const u64x48, b: *const u64x48,
/// consts: *const { p[4], -p^-1, mu })` in rdi, rsi, rdx, rcx. `a` and `b`
/// are repr(C) Fp12, canonical. Outputs canonical.
pub fn fp12_mul_x86<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    let tables = fp12_mul_tables();
    m.rodata(FMU_TAB_LABEL, &tables);
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    m.alloc_stack(FMU_FRAME);
    m.comment(
        "frame: p +0, -p^-1 +32, mu +40, z +48, loop bounds +56..96, ctx slots +104..160, NB2 +256, staged a/b +320..1088, t1/t2 +1088..1472, outputs +1472, operand sides +1856, products +3008, xi/neg scratch +4160, AC/BD/CR/TA parks +4544..6080",
    );
    m.store(Mem::new(rsp, FSQ_Z), Rdi, "spill z");
    for k in 0..4 {
        m.load(Rax, Mem::new(Rcx, 8 * k), &format!("p{k}"));
        m.store(
            Mem::new(rsp, FP6_P + 8 * k),
            Rax,
            "cancel rows address the frame as a consts table",
        );
    }
    m.load(Rax, Mem::new(Rcx, 32), "-p^-1");
    m.store(Mem::new(rsp, FP6_PINV), Rax, "-p^-1");
    m.load(Rax, Mem::new(Rcx, 40), "mu = floor(2^310/p)");
    m.store(Mem::new(rsp, FP6_MU), Rax, "mu");
    m.lea_rodata(Rax, FMU_TAB_LABEL, "walk tables");
    m.store(Mem::new(rsp, FSQ_TBL), Rax, "table base");
    m.mov(Rbx, Rax, "");
    m.add_imm(Rbx, FMU_TB_MSUB + 48, "product m-walk end");
    m.store(Mem::new(rsp, FSQ_MEND), Rbx, "");
    m.mov(Rbx, Rax, "");
    m.add_imm(Rbx, FMU_TB_MADD, "ctx table end (three 48-byte rows)");
    m.store(Mem::new(rsp, FSQ_OUTER_END), Rbx, "");
    m.xor_clear(Rax, "");
    for k in 0..8 {
        m.store(
            Mem::new(rsp, FSQ_ZERO8 + 8 * k),
            Rax,
            "zero word (negation rows subtract from it)",
        );
    }
    m.store(
        Mem::new(rsp, FSQ_MOD_DST),
        Rax,
        "mod rows carry absolute destination offsets",
    );

    m.comment("");
    m.comment("stage a then b: all later reads are frame-relative, which is");
    m.comment("what makes z == a, z == b and a == b safe (no operand read");
    m.comment("after any z store)");
    m.mov(Rdi, rsp, "");
    m.add_imm(Rdi, FMU_AST, "staging cursor (b's area follows a's)");
    m.xor_clear(Rcx, "48 limbs, 4 per iteration");
    m.stride_loop(Rcx, 32, LoopEnd::Imm(384), ".Lfmu_sta", &mut |m| {
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("a limb {k}"));
        }
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.store(Mem::new(Rdi, 8 * k as i32), reg, "staged");
        }
        m.add_imm(Rsi, 32, "");
        m.add_imm(Rdi, 32, "");
    });
    m.mov(Rsi, MULTIPLIER, "b pointer (rdi has walked to b's area)");
    m.xor_clear(Rcx, "");
    m.stride_loop(Rcx, 32, LoopEnd::Imm(384), ".Lfmu_stb", &mut |m| {
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("b limb {k}"));
        }
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.store(Mem::new(Rdi, 8 * k as i32), reg, "staged");
        }
        m.add_imm(Rsi, 32, "");
        m.add_imm(Rdi, 32, "");
    });

    m.comment("");
    m.comment("t1 = a0 + a1, t2 = b0 + b1 (modular, canonical)");
    fsq_walk_setup(m, Rbp, FMU_TB_MADD, 12 * 24, FSQ_WALK_END);
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lfmu_madd",
        &mut |m| dbl_modadd_row(m),
    );

    m.comment("");
    m.comment("outer loop: iteration 0 parks AC = a0*b0, 1 parks BD = a1*b1,");
    m.comment("2 parks CR = t1*t2 and rides the mulVadd rows for z.a");
    m.load(Rbp, Mem::new(rsp, FSQ_TBL), "ctx cursor = first ctx row");
    m.stride_loop(
        Rbp,
        48,
        LoopEnd::Mem(Mem::new(rsp, FSQ_OUTER_END)),
        ".Lfmu_iter",
        &mut |m| {
            m.store(Mem::new(rsp, FSQ_CTX_SPILL), Rbp, "spill the outer cursor");
            for (field, slot, what) in [
                (0, FSQ_STAGE_X, "ctx: x-side source"),
                (8, FSQ_STAGE_Y, "ctx: y-side source"),
                (16, FMU_PARK, "ctx: Fp6Dbl park base"),
                (24, FMU_GSUB_END, "ctx: gsub walk end"),
                (32, FMU_NINE_END, "ctx: nine walk end"),
                (40, FMU_GADD_END, "ctx: gadd walk end"),
            ] {
                m.load(Rax, Mem::new(Rbp, field), what);
                m.store(Mem::new(rsp, slot), Rax, "");
            }
            stage_sides_walk(m, "fmu", FMU_TB_SUMS);
            products_walk(m, "fmu", FMU_TB_MSUB);
            m.comment("double-width sub walk: Karatsuba assembly, cross terms, negations");
            fmu_walk_setup_ctx(m, Rbp, FMU_TB_GSUB, FMU_GSUB_END, FSQ_WALK_END);
            m.stride_loop(
                Rbp,
                24,
                LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
                ".Lfmu_gsub",
                &mut |m| dbl_gsub_row(m),
            );
            m.comment("nine-fold walk: xi = 9 + u on 512-bit values, mu-canonical high");
            fmu_walk_setup_ctx(m, Rbp, FMU_TB_NINE, FMU_NINE_END, FSQ_WALK_END);
            m.stride_loop(
                Rbp,
                24,
                LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
                ".Lfmu_nine",
                &mut |m| dbl_nine_row(m),
            );
            m.comment("double-width add walk: assemble and park the Fp6Dbl (+ z.a rows)");
            fmu_walk_setup_ctx(m, Rbp, FMU_TB_GADD, FMU_GADD_END, FSQ_WALK_END);
            m.stride_loop(
                Rbp,
                24,
                LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
                ".Lfmu_gadd",
                &mut |m| dbl_gadd_row(m, Some(FMU_PARK)),
            );
            m.load(Rbp, Mem::new(rsp, FSQ_CTX_SPILL), "reload the outer cursor");
        },
    );

    m.comment("");
    m.comment("z.b assembly: CR -= AC, CR -= BD (all lanes guarded mod p*2^256)");
    fsq_walk_setup(m, Rbp, FMU_TB_GSUB2, 12 * 24, FSQ_WALK_END);
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lfmu_gsub2",
        &mut |m| dbl_gsub_row(m),
    );
    m.comment("Montgomery reduction walk: 12 output coefficients into YOUT");
    fsq_walk_setup(m, Rbp, FMU_TB_MOD, 12 * 16, FSQ_WALK_END);
    m.stride_loop(
        Rbp,
        16,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lfmu_mod",
        &mut |m| dbl_mod_row(m),
    );

    m.comment("");
    m.comment("copy out: z.a then z.b are contiguous, 48 limbs to z");
    m.load(Rdi, Mem::new(rsp, FSQ_Z), "z");
    m.mov(Rsi, rsp, "");
    m.add_imm(Rsi, FMU_YOUT, "output base");
    m.xor_clear(Rcx, "");
    m.stride_loop(Rcx, 32, LoopEnd::Imm(384), ".Lfmu_out", &mut |m| {
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("z limb {k}"));
        }
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.store(Mem::new(Rdi, 8 * k as i32), reg, "z");
        }
        m.add_imm(Rsi, 32, "");
        m.add_imm(Rdi, 32, "");
    });
    m.free_stack(FMU_FRAME);
    for reg in CALLEE_SAVED.iter().rev() {
        m.pop(*reg);
    }
    m.ret();
}

/// Register roles for `narsil_cyc_sqr_x86` (walk bodies shared with
/// fp12_sqr/fp12_mul, so the roles mirror theirs).
pub const CYC_SQR_REGISTER_MAP: &[(Reg, &str)] = &[
    (
        Rdi,
        "z on entry (spilled); walk destination pointer in every table-driven pass",
    ),
    (
        Rsi,
        "f pointer on entry (staging source); walk source-1 pointer; mask scratch",
    ),
    (
        Rdx,
        "consts pointer on entry; the implicit mulx multiplicand; walk scratch",
    ),
    (
        Rcx,
        "walk source-2 pointer; staging cursor; the product output cursor",
    ),
    (R8, "accumulator/value word 0"),
    (R9, "accumulator/value word 1"),
    (R10, "accumulator/value word 2"),
    (R11, "accumulator/value word 3"),
    (R12, "accumulator/value word 4"),
    (R13, "accumulator/value word 5; mu-reduction scratch"),
    (
        R14,
        "product round cursor (byte offset 8j); 8-limb value word 6",
    ),
    (
        R15,
        "product cursor (64-byte operand-pair steps); 8-limb value word 7",
    ),
    (
        Rbp,
        "the walk row cursor, marching once through the whole table",
    ),
    (
        Rax,
        "low half of the current product; zero for chain closes; borrow mask",
    ),
    (Rbx, "high half of the current product; prologue scratch"),
];

// cyc_sqr frame layout. Every slot a shared walk body addresses (p/-p^-1/mu,
// z, walk bound, table base, MOD_DST) sits at its fp12_sqr offset, so the
// bodies emit unchanged.
/// An 8-limb zero window (+96..152): negation rows subtract from it, copy
/// rows add its low half. Deliberately spans the FSQ_MOD_DST slot (+136),
/// whose required value here is exactly zero (mod rows carry absolute
/// destination offsets), so one store pass initializes both.
const CYC_ZERO8: i32 = 96;
/// f staged once (48 limbs): all later reads are frame-relative, and z == f
/// becomes trivially safe (no f read after any z store).
const CYC_FST: i32 = 208;
/// The three Fp4 cross operands s_k = x0_k + x1_k (canonical Fp2 each).
const CYC_SSUM: i32 = 592;
/// negp images p - r of the subtractive z-combine operands r0, r4, r3: the
/// three subtraction openers run as modular adds of these.
const CYC_NP: i32 = 784;
/// Nine 128-byte square-operand blocks `[a - b, a + b, 2b, a]`, all four
/// rows canonical, three blocks per Fp4 (sqrPre of x0, x1, s).
const CYC_SQB: i32 = 976;
/// Eighteen 8-limb raw products: block q's a-lane at 128q, b-lane at
/// 128q + 64 (the 64-byte operand-pair stride maps 1:1 onto output lanes).
const CYC_PROD: i32 = 2128;
/// Negations: nbb_k = -sqr(x0_k).b for the three yb rows, then -U_2.b for
/// the xi*t5 fold.
const CYC_NB: i32 = 3280;
/// The nine-fold y operands: ya_k = T0.a - T1.b, yb_k = T1.a + T0.b.
const CYC_Y: i32 = 3536;
/// Nine-fold outputs: the complete T2_k = xi*T1_k + T0_k (three Fp2Dbl),
/// then XT = xi*U_2.
const CYC_SCR: i32 = 3920;
/// The six reduced Fp4 outputs t0, t1, t2, t3, t4, xi*t5 (canonical Fp2s).
const CYC_TT: i32 = 4432;
/// The z-combine area, laid out as the final repr(C) Fp12:
/// z0, z4, z3, z2, z1, z5 at +0, +64, +128, +192, +256, +320.
const CYC_OUT: i32 = 4816;
const CYC_FRAME: i32 = 5200;

/// Byte offsets of the table regions inside the cyc_sqr rodata blob. The
/// regions are contiguous in walk order: one cursor marches through the
/// whole table, each phase only storing the next bound.
const CYC_TB_MADD1: i32 = 0; // 33 rows x 3
const CYC_TB_MSUB1: i32 = 792; // 15 rows x 3
const CYC_TB_GSUB: i32 = 1152; // 22 rows x 3
const CYC_TB_NINE: i32 = 1680; // 8 rows x 3
const CYC_TB_MOD: i32 = 1872; // 12 rows x 2
const CYC_TB_MADD2: i32 = 2064; // 36 rows x 3
const CYC_TB_END: i32 = 2928;

const CYC_TAB_LABEL: &str = ".Lcyc_tab";

/// The cyc_sqr read-only walk tables (rsp-relative offsets).
fn cyc_sqr_tables() -> Vec<u64> {
    let mut t: Vec<u64> = Vec::new();
    let row3 = |t: &mut Vec<u64>, dst: i32, s1: i32, s2: i32| {
        t.extend([dst as u64, s1 as u64, s2 as u64]);
    };
    // The Granger-Scott operand pairs in repr(C) f order (r0 = c0.c0,
    // r4 = c0.c1, r3 = c0.c2, r2 = c1.c0, r1 = c1.c1, r5 = c1.c2):
    // Fp4 #0 squares (r0, r1), #1 (r2, r3), #2 (r4, r5).
    let (r0, r4, r3, r2, r1, r5) = (
        CYC_FST,
        CYC_FST + 64,
        CYC_FST + 128,
        CYC_FST + 192,
        CYC_FST + 256,
        CYC_FST + 320,
    );
    let pairs = [(r0, r1), (r2, r3), (r4, r5)];
    // Square q's staged block and raw product lanes. Squares 3k, 3k+1,
    // 3k+2 are Fp4 k's x0, x1, s. Per square, T.a = (a-b)(a+b) and
    // T.b = 2b*a (all four operand rows canonical, so both lanes < p^2).
    let blk = |q: i32| CYC_SQB + 128 * q;
    let pa = |q: i32| CYC_PROD + 128 * q;
    let pb = |q: i32| CYC_PROD + 128 * q + 64;
    // Square q's source Fp2.
    let src = |q: i32| {
        let (x0, x1) = pairs[(q / 3) as usize];
        [x0, x1, CYC_SSUM + 64 * (q / 3)][(q % 3) as usize]
    };

    // Modular add walk 1. First the three cross operands s_k = x0_k + x1_k
    // (per Fp half), then per square the additive block rows: a + b, 2b,
    // and the a copy (a + 0), every output canonical.
    for (k, (x0, x1)) in pairs.into_iter().enumerate() {
        for half in 0..2 {
            row3(
                &mut t,
                CYC_SSUM + 64 * k as i32 + 32 * half,
                x0 + 32 * half,
                x1 + 32 * half,
            );
        }
    }
    for q in 0..9 {
        row3(&mut t, blk(q) + 32, src(q), src(q) + 32);
        row3(&mut t, blk(q) + 64, src(q) + 32, src(q) + 32);
        row3(&mut t, blk(q) + 96, src(q), CYC_ZERO8);
    }

    // Modular sub walk: the a - b block rows, then the negp images
    // p - r = 0 - r mod p of the subtractive z-combine operands.
    assert_eq!(t.len() * 8, CYC_TB_MSUB1 as usize);
    for q in 0..9 {
        row3(&mut t, blk(q), src(q), src(q) + 32);
    }
    for (j, r) in [r0, r4, r3].into_iter().enumerate() {
        for half in 0..2 {
            row3(
                &mut t,
                CYC_NP + 64 * j as i32 + 32 * half,
                CYC_ZERO8,
                r + 32 * half,
            );
        }
    }

    // Double-width subs. Per Fp4 the nine-fold y operands
    // (ya = T0.a - T1.b, yb = T1.a + T0.b, the addition via the negated
    // nbb = 0 - T0.b), then the twelve U_k = TS_k - T0_k - T1_k rows (in
    // place over the s-square lanes), then NB = 0 - U_2.b feeding the
    // xi*t5 fold (its operand is final only after the U rows).
    assert_eq!(t.len() * 8, CYC_TB_GSUB as usize);
    for k in 0..3 {
        row3(&mut t, CYC_Y + 128 * k, pa(3 * k), pb(3 * k + 1));
        row3(&mut t, CYC_NB + 64 * k, CYC_ZERO8, pb(3 * k));
        row3(&mut t, CYC_Y + 128 * k + 64, pa(3 * k + 1), CYC_NB + 64 * k);
    }
    for k in 0..3 {
        let u = 3 * k + 2;
        row3(&mut t, pa(u), pa(u), pa(3 * k));
        row3(&mut t, pa(u), pa(u), pa(3 * k + 1));
        row3(&mut t, pb(u), pb(u), pb(3 * k));
        row3(&mut t, pb(u), pb(u), pb(3 * k + 1));
    }
    row3(&mut t, CYC_NB + 192, CYC_ZERO8, pb(8));

    // Nine-fold rows completing each T2 = xi*T1 + T0 in one step
    // (dst = 9x + y with the T0 term folded into y), then XT = xi*U_2
    // (t5 is consumed only as xi*t5, so the xi folds into the double-width
    // value and t5 itself never materializes).
    assert_eq!(t.len() * 8, CYC_TB_NINE as usize);
    for k in 0..3 {
        row3(&mut t, CYC_SCR + 128 * k, pa(3 * k + 1), CYC_Y + 128 * k);
        row3(
            &mut t,
            CYC_SCR + 128 * k + 64,
            pb(3 * k + 1),
            CYC_Y + 128 * k + 64,
        );
    }
    row3(&mut t, CYC_SCR + 384, pa(8), CYC_NB + 192);
    row3(&mut t, CYC_SCR + 448, pb(8), pa(8));

    // MOD_DST is zero. The destination offsets are absolute.
    // t0 = mod(T2_0), t1 = mod(U_0), t2 = mod(T2_1),
    // t3 = mod(U_1), t4 = mod(T2_2), xt5 = mod(XT).
    assert_eq!(t.len() * 8, CYC_TB_MOD as usize);
    let tt = |i: i32| CYC_TT + 64 * i;
    for (i, s) in [
        (0, CYC_SCR),
        (1, pa(2)),
        (2, CYC_SCR + 128),
        (3, pa(5)),
        (4, CYC_SCR + 256),
        (5, CYC_SCR + 384),
    ] {
        t.extend([s as u64, tt(i) as u64]);
        t.extend([(s + 64) as u64, (tt(i) + 32) as u64]);
    }

    // Modular add walk 2: the z-combines, in dependency order. Openers
    // (the subtractive ones as adds of the negp images), then all six
    // in-place doublings, then the final += t of every z -- exactly the
    // composed 3t +- 2r shape (z = 2*(t -+ r) + t).
    assert_eq!(t.len() * 8, CYC_TB_MADD2 as usize);
    let (z0, z4, z3, z2, z1, z5) = (
        CYC_OUT,
        CYC_OUT + 64,
        CYC_OUT + 128,
        CYC_OUT + 192,
        CYC_OUT + 256,
        CYC_OUT + 320,
    );
    let openers = [
        (z0, tt(0), CYC_NP),
        (z4, tt(2), CYC_NP + 64),
        (z3, tt(4), CYC_NP + 128),
        (z2, tt(5), r2),
        (z1, tt(1), r1),
        (z5, r5, tt(3)),
    ];
    for (dst, s1, s2) in openers {
        for half in 0..2 {
            row3(&mut t, dst + 32 * half, s1 + 32 * half, s2 + 32 * half);
        }
    }
    for j in 0..12 {
        row3(&mut t, CYC_OUT + 32 * j, CYC_OUT + 32 * j, CYC_OUT + 32 * j);
    }
    for (dst, t_src) in [
        (z0, tt(0)),
        (z4, tt(2)),
        (z3, tt(4)),
        (z2, tt(5)),
        (z1, tt(1)),
        (z5, tt(3)),
    ] {
        for half in 0..2 {
            row3(&mut t, dst + 32 * half, dst + 32 * half, t_src + 32 * half);
        }
    }
    assert_eq!(t.len() * 8, CYC_TB_END as usize);
    t
}

/// Advance the walk chain: the row cursor already sits at this region's
/// start (the previous walk's exact exit value), so only the new bound is
/// stored.
fn cyc_walk_bound<M: Machine>(m: &mut M, end_off: i32) {
    m.load(Rax, Mem::new(Reg::Rsp, FSQ_TBL), "table base");
    m.add_imm(Rax, end_off, "next walk's bound");
    m.store(Mem::new(Reg::Rsp, FSQ_WALK_END), Rax, "walk bound");
}

/// `narsil_cyc_sqr_x86`: the Granger-Scott cyclotomic square in mcl's lazy
/// double-width shape -- 18 raw 4x4 products and 12 Montgomery reductions
/// where the composed SoS path pays 36 products and 12 interleaved
/// reductions. The single hottest final-exponentiation shape: 192 calls per
/// final exp, all on the latency-critical pow_x dependent chain.
///
/// # Semantics (exactly `Fp12::cyclotomic_square`'s composed path. Mcl
/// `fasterSqr`/`sqrFp4`, pairing_impl.hpp)
///
/// With the arkworks mapping r0 = c0.c0, r4 = c0.c1, r3 = c0.c2,
/// r2 = c1.c0, r1 = c1.c1, r5 = c1.c2 and three Fp4 squares
/// `(t0, t1) = (r0 + r1*y)^2` with `y^2 = xi = 9 + u`.
///
/// * each Fp4 square is three lazy Fp2Dbl::sqrPre products
///   (`T0 = sqr(x0)`, `T1 = sqr(x1)`, `TS = sqr(x0 + x1)`), each TWO raw
///   4x4 products by the complex method (`(a - b)*(a + b)` and `2b*a`, all
///   four operand rows staged canonical), with `t0 = mod(xi*T1 + T0)` and
///   `t1 = mod(TS - T0 - T1)` -- one Montgomery reduction per output Fp
///   instead of the composed path's interleaved sos4/sos2 dispatches.
/// * the T0 addition folds into the nine-fold's y operand
///   (`T2.a = 9*T1.a + (T0.a - T1.b)`, `T2.b = 9*T1.b + (T1.a + T0.b)`),
///   so each T2 finishes inside the nine walk and no double-width add walk
///   exists.
/// * t5 is consumed only as xi*t5 (the z2 combine), so its xi folds into
///   the double-width value: `xt5 = mod(xi*(TS_2 - T0_2 - T1_2))` -- same
///   reduction count, one single-width xi saved.
/// * z-combines on the REDUCED values, single-width modular (mcl's
///   fasterSqr shape, bit-identical to the composed path):
///   z0 = 2(t0 - r0) + t0, z1 = 2(t1 + r1) + t1, z2 = 2(xt5 + r2) + xt5,
///   z3 = 2(t4 - r3) + t4, z4 = 2(t2 - r4) + t2, z5 = 2(r5 + t3) + t3,
///   the three subtractions entering as adds of staged negp images
///   (t - r = t + (p - r) mod p, canonical either way), output
///   (z0, z4, z3, z2, z1, z5) in repr(C) order.
///
/// # Bounds (BN254: p < 2^253.61, K = 2^256, pK < 2^510)
///
/// * staged rows all canonical: a - b, a + b, 2b, a, s = x0 + x1 and the
///   negp images are modular single-width outputs < p.
/// * raw products < p^2 < pK: every sqrPre lane is nonnegative and already
///   below the guard modulus.
/// * nine-fold y operands: ya = T0.a - T1.b and yb = T1.a + T0.b (via
///   nbb = 0 - T0.b) guarded mod pK, high halves < p. U = TS - T0 - T1
///   guarded mod pK (subtrahends < p^2 < pK).
/// * nine rows: operand highs < p (lanes < p^2 or guarded < pK), output
///   < pK with a mu-canonical high half.
/// * every reduced value satisfies T < pK, the Montgomery precondition:
///   result < 2p, one conditional subtraction, canonical.
/// * z-combines use canonical single-width adds with sums below `2p`.
///
/// The interpreter asserts the flag claims on every path. The u512
/// reference in `kernelgen_verify` asserts each stage bound on random and
/// adversarial inputs.
///
/// # In-place update
///
/// `z == f` is allowed and is the production shape (pow_x squares its
/// accumulator in place): the prologue stages all of `f` into the frame and
/// `f` is never read again, so no output store can alias a live operand.
///
/// # Structure and latency
///
/// No outer loop: the three Fp4 squares are fully independent, so every
/// phase runs once as a flat walk over all three -- maximal cross-Fp4 ILP
/// for the OoO window. All staging and z-combine work runs as rows of the
/// two single-width bodies (modular add, modular sub), so besides them the
/// kernel emits only the product loop and the three shared double-width
/// bodies (guarded sub, nine-fold, reduction), each exactly once. One row
/// cursor marches through the whole rodata table. Phase boundaries store
/// the next bound. The critical path per call is one product (4 rolled
/// rounds) + one guarded sub + nine + one reduction + three single-width
/// combine rows. The composed path has the same reduction depth but 2x the
/// product mass.
///
/// Arguments: `(z: *mut u64x48, f: *const u64x48, consts: *const { p[4],
/// -p^-1, mu })` in rdi, rsi, rdx. `f` is repr(C) Fp12, canonical. Outputs
/// canonical. The Granger-Scott identity requires a cyclotomic-subgroup
/// input for z to equal f^2, but the leaf computes the composed formula
/// bit-identically on ANY canonical input.
pub fn cyc_sqr_x86<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    let tables = cyc_sqr_tables();
    m.rodata(CYC_TAB_LABEL, &tables);
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    m.alloc_stack(CYC_FRAME);
    m.comment(
        "frame: p +0, -p^-1 +32, mu +40, z +48, walk bound +64, table base +88, zero8 +96 (spans mod dst +136), staged f +208, s sums +592, negp images +784, square blocks +976, products +2128, negations +3280, nine-fold y +3536, xi scratch +3920, t values +4432, z combines +4816",
    );
    m.store(Mem::new(rsp, FSQ_Z), Rdi, "spill z");
    for k in 0..4 {
        m.load(Rax, Mem::new(Rdx, 8 * k), &format!("p{k}"));
        m.store(
            Mem::new(rsp, FP6_P + 8 * k),
            Rax,
            "cancel rows address the frame as a consts table",
        );
    }
    m.load(Rax, Mem::new(Rdx, 32), "-p^-1");
    m.store(Mem::new(rsp, FP6_PINV), Rax, "-p^-1");
    m.load(Rax, Mem::new(Rdx, 40), "mu = floor(2^310/p)");
    m.store(Mem::new(rsp, FP6_MU), Rax, "mu");
    m.lea_rodata(Rax, CYC_TAB_LABEL, "walk tables");
    m.store(Mem::new(rsp, FSQ_TBL), Rax, "table base");
    m.xor_clear(Rax, "");
    for k in 0..8 {
        m.store(
            Mem::new(rsp, CYC_ZERO8 + 8 * k),
            Rax,
            "zero word (negation/copy rows and the spanned MOD_DST slot)",
        );
    }

    m.comment("");
    m.comment("stage f: all later reads are frame-relative, which is what");
    m.comment("makes z == f safe (no f read after any z store)");
    m.mov(Rdi, rsp, "");
    m.add_imm(Rdi, CYC_FST, "staging cursor");
    m.xor_clear(Rcx, "48 limbs, 4 per iteration");
    m.stride_loop(Rcx, 32, LoopEnd::Imm(384), ".Lcyc_fst", &mut |m| {
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("f limb {k}"));
        }
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.store(Mem::new(Rdi, 8 * k as i32), reg, "staged");
        }
        m.add_imm(Rsi, 32, "");
        m.add_imm(Rdi, 32, "");
    });

    m.comment("");
    m.comment("modular add walk 1: the s_k = x0_k + x1_k cross operands, then");
    m.comment("the additive block rows a + b, 2b and the a copy, all canonical");
    fsq_walk_setup(
        m,
        Rbp,
        CYC_TB_MADD1,
        CYC_TB_MSUB1 - CYC_TB_MADD1,
        FSQ_WALK_END,
    );
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lcyc_madd1",
        &mut |m| dbl_modadd_row(m),
    );

    m.comment("");
    m.comment("modular sub walk: the a - b block rows, then the negp images");
    m.comment("p - r of the subtractive z-combine operands");
    cyc_walk_bound(m, CYC_TB_GSUB);
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lcyc_msub",
        &mut |m| dbl_msub_row(m, "dst"),
    );

    m.comment("");
    m.comment("products: 18 raw 4x4 mulpre, two per square (the complex method:");
    m.comment("a-lane (a-b)(a+b), b-lane 2b*a); the 64-byte operand-pair stride");
    m.comment("maps 1:1 onto the 64-byte output lanes");
    m.xor_clear(R15, "product cursor: 64-byte operand-pair steps");
    m.stride_loop(R15, 64, LoopEnd::Imm(18 * 64), ".Lcyc_prod", &mut |m| {
        m.mov(Rsi, rsp, "");
        m.add(Rsi, R15, "");
        m.add_imm(Rsi, CYC_SQB, "PA: the multiplicand row");
        m.mov(Rdi, Rsi, "");
        m.add_imm(Rdi, 32, "PY: the y row");
        m.mov(Rcx, Rsi, "");
        m.add_imm(Rcx, CYC_PROD - CYC_SQB, "PZ: the 512-bit product lane");
        for (k, t) in T.into_iter().enumerate() {
            m.xor_clear(t, &format!("t{k} = 0"));
        }
        m.xor_clear(R14, "round cursor: byte offset 8j of the multiplicand limb");
        m.stride_loop(R14, 8, LoopEnd::Imm(32), ".Lcyc_prod_j", &mut |m| {
            m.load_indexed(MULTIPLIER, Rsi, R14, "x[j], the row multiplicand");
            m.xor_clear(LO, "re-seed CF = OF = 0 (back edge clobbered flags)");
            fsq_mulpre_row(m, Rdi);
            m.store(Mem::new(Rcx, 0), T[0], "product limb j is final");
            m.add_imm(Rcx, 8, "next output limb");
            m.comment("shift down one word");
            for k in 0..5 {
                m.mov(T[k], T[k + 1], &format!("t{k} = t{}", k + 1));
            }
            m.xor_clear(T[5], "t5 = 0 (CF/OF stay clear)");
        });
        for (k, t) in T[..4].iter().enumerate() {
            m.store(
                Mem::new(Rcx, 8 * k as i32),
                *t,
                &format!("product limb {}", k + 4),
            );
        }
    });

    m.comment("");
    m.comment("double-width sub walk: the nine-fold y operands ya = T0.a - T1.b");
    m.comment("and yb = T1.a + T0.b (via nbb = 0 - T0.b), U = TS - T0 - T1, then");
    m.comment("the xi*t5 negation (its operand U_2.b is final only after the U rows)");
    cyc_walk_bound(m, CYC_TB_NINE);
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lcyc_gsub",
        &mut |m| dbl_gsub_row(m),
    );

    m.comment("");
    m.comment("nine-fold walk: each T2 = xi*T1 + T0 completes as 9x + y, then");
    m.comment("XT = xi*U_2 (the t5 site), mu-canonical high halves");
    cyc_walk_bound(m, CYC_TB_MOD);
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lcyc_nine",
        &mut |m| dbl_nine_row(m),
    );

    m.comment("");
    m.comment("Montgomery reduction walk: t0, t1, t2, t3, t4, xi*t5");
    cyc_walk_bound(m, CYC_TB_MADD2);
    m.stride_loop(
        Rbp,
        16,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lcyc_mod",
        &mut |m| dbl_mod_row(m),
    );

    m.comment("");
    m.comment("modular add walk 2: the z-combines -- openers (subtractions as");
    m.comment("adds of the negp images), six in-place doublings, final += t");
    cyc_walk_bound(m, CYC_TB_END);
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lcyc_madd2",
        &mut |m| dbl_modadd_row(m),
    );

    m.comment("");
    m.comment("copy out: the z area is already repr(C) Fp12, 48 limbs to z");
    m.load(Rdi, Mem::new(rsp, FSQ_Z), "z");
    m.mov(Rsi, rsp, "");
    m.add_imm(Rsi, CYC_OUT, "z-combine base");
    m.xor_clear(Rcx, "");
    m.stride_loop(Rcx, 32, LoopEnd::Imm(384), ".Lcyc_out", &mut |m| {
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("z limb {k}"));
        }
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.store(Mem::new(Rdi, 8 * k as i32), reg, "z");
        }
        m.add_imm(Rsi, 32, "");
        m.add_imm(Rdi, 32, "");
    });
    m.free_stack(CYC_FRAME);
    for reg in CALLEE_SAVED.iter().rev() {
        m.pop(*reg);
    }
    m.ret();
}

/// Register roles for `narsil_fp12_034l_x86` (walk bodies shared with
/// fp12_sqr/fp12_mul/cyc_sqr, so the roles mirror theirs).
pub const FP12_034L_REGISTER_MAP: &[(Reg, &str)] = &[
    (
        Rdi,
        "z on entry (spilled); block staging cursor; walk destination pointer",
    ),
    (
        Rsi,
        "f pointer on entry (staging source, then repointed at c); walk source-1 pointer; mask scratch",
    ),
    (
        Rdx,
        "c pointer on entry; the implicit mulx multiplicand; walk scratch",
    ),
    (
        Rcx,
        "consts pointer on entry; staging block counter; product m-walk cursor; walk source-2 pointer",
    ),
    (R8, "accumulator/value word 0"),
    (R9, "accumulator/value word 1"),
    (R10, "accumulator/value word 2"),
    (R11, "accumulator/value word 3"),
    (R12, "accumulator/value word 4"),
    (R13, "accumulator/value word 5; mu-reduction scratch"),
    (
        R14,
        "product round cursor (byte offset 8j); 8-limb value word 6",
    ),
    (
        R15,
        "product pair-row cursor (24-byte steps); 8-limb value word 7",
    ),
    (
        Rbp,
        "product region pointer; the row cursor of every other walk",
    ),
    (
        Rax,
        "low half of the current product; zero for chain closes; borrow mask",
    ),
    (
        Rbx,
        "high half of the current product; walk bound and prologue scratch",
    ),
];

// fp12_034l frame layout. Every slot a shared walk body addresses
// (p/-p^-1/mu, z, walk bound, table base, product m-walk end, MOD_DST,
// ZERO8) sits at its fp12_sqr offset, so the bodies emit unchanged.
/// Six staged f blocks `[re, im, s = re + im]` (a0.a .. A0.c, a1.a .. A1.c),
/// then the three staged c blocks C0, C3, C4, contiguous so one staging
/// cursor walks both.
const F34L_XB: i32 = 256;
const F34L_YB: i32 = 832;
/// t1 = a0 + a1 blocks (three), then Y3 = C0 + C3, then the Karatsuba sum
/// blocks XS1 = a1.a + a1.b, XS2 = t1.a + t1.b, YS1 = C3 + C4,
/// YS2 = Y3 + C4. All rows of these six blocks canonical (madd rows).
const F34L_T1: i32 = 1120;
const F34L_Y3: i32 = 1408;
const F34L_XS1: i32 = 1504;
const F34L_XS2: i32 = 1600;
const F34L_YS1: i32 = 1696;
const F34L_YS2: i32 = 1792;
/// Thirteen 192-byte product regions (d0 +0, d1 +64, d2 +128). After the
/// Karatsuba assembly the d2 slots are dead and repark the negations and
/// nine-fold y operands (see the table builder's slot audit).
const F34L_PROD: i32 = 1888;
/// The twelve reduced output coefficients (z.a then z.b), copied to z last.
const F34L_YOUT: i32 = 4384;
const F34L_FRAME: i32 = 4768;

/// Byte offsets of the table regions inside the fp12_034l rodata blob.
/// gsub1 through mod are contiguous in walk order (one cursor marches,
/// phase boundaries store the next bound). Madd and the product pair rows
/// have their own setups.
const F34L_TB_MADD: i32 = 0; // 24 rows x 3
const F34L_TB_GSUB1: i32 = 576; // 68 rows x 3
const F34L_TB_NINE: i32 = 2208; // 6 rows x 3
const F34L_TB_GSUB2: i32 = 2352; // 16 rows x 3
const F34L_TB_MOD: i32 = 2736; // 12 rows x 2
const F34L_TB_PROD: i32 = 2928; // 13 rows x 3
const F34L_TB_MSUB: i32 = 3240; // 3 rows x 2
const F34L_TB_END: i32 = 3288;

const F34L_TAB_LABEL: &str = ".Lf34l_tab";

/// The fp12_034l read-only walk tables (rsp-relative offsets).
///
/// Product regions and Fp6Dbl lanes: with `f = a0 + a1*w` and the sparse
/// multiplier `X0 + X1*w = C0 + (C3 + C4*v)*w`:
///
/// * P0..P2:  AC = a0 * (C0, 0, 0) coefficient-wise (a0.a*C0, a0.b*C0,
///   a0.c*C0).
/// * P3..P7:  BD = a1 * (C3 + C4*v) by mul_01 Karatsuba: AD = a1.a*C3,
///   BE = a1.b*C4, CD = a1.c*C3, CE = a1.c*C4, TT = (a1.a + a1.b)(C3 + C4).
/// * P8..P12: CR = t1 * ((C0 + C3) + C4*v), the same five products over
///   t1 = a0 + a1 and Y3 = C0 + C3.
///
/// Assembled in place: BD.a = AD + xi*CE (nine rows over P6, y operands
/// prebuilt in P3), BD.b = TT - AD - BE (over P7), BD.c = BE + CD (over P4,
/// the addition as a subtraction of the negated CD). CR likewise over
/// P11/P12/P9. Then z.a.c0 = xi*BD.c + AC.a (nine rows into P5, the dead CD
/// region), z.a.c1 = BD.a + AC.b (over P6), z.a.c2 = BD.b + AC.c (over P7),
/// z.b = CR - AC - BD (over P11/P12/P9), twelve reductions into YOUT.
fn fp12_034l_tables() -> Vec<u64> {
    let mut t: Vec<u64> = Vec::new();
    let row3 = |t: &mut Vec<u64>, dst: i32, s1: i32, s2: i32| {
        t.extend([dst as u64, s1 as u64, s2 as u64]);
    };
    let xb = |k: i32| F34L_XB + 96 * k;
    let yb = |k: i32| F34L_YB + 96 * k;
    let t1 = |k: i32| F34L_T1 + 96 * k;
    let p = |k: i32| F34L_PROD + 192 * k;
    // Lane shorthands: a-lane +0, b-lane +64, d2 scratch +128.
    let (ac_a, ac_b, ac_c) = (p(0), p(1), p(2));
    let (ad, be, cd, ce, tt) = (p(3), p(4), p(5), p(6), p(7));
    let (ad2, be2, cd2, ce2, tt2) = (p(8), p(9), p(10), p(11), p(12));

    // Modular add walk: one (dst, s1, s2) row per Fp. Blocks t1.a..c and Y3
    // build re then im from the staged singles, then s = re + im. The four
    // Karatsuba sum blocks likewise (XS2 needs t1, YS2 needs Y3: row order
    // is dependency order). Every output canonical, which is what keeps all
    // thirteen raw products below 4p^2 < p*2^256.
    for k in 0..3 {
        row3(&mut t, t1(k), xb(k), xb(3 + k));
        row3(&mut t, t1(k) + 32, xb(k) + 32, xb(3 + k) + 32);
        row3(&mut t, t1(k) + 64, t1(k), t1(k) + 32);
    }
    for (dst, s1, s2) in [
        (F34L_Y3, yb(0), yb(1)),
        (F34L_XS1, xb(3), xb(4)),
        (F34L_XS2, t1(0), t1(1)),
        (F34L_YS1, yb(1), yb(2)),
        (F34L_YS2, F34L_Y3, yb(2)),
    ] {
        row3(&mut t, dst, s1, s2);
        row3(&mut t, dst + 32, s1 + 32, s2 + 32);
        row3(&mut t, dst + 64, dst, dst + 32);
    }

    // Double-width sub walk 1. First the per-product Karatsuba assembly
    // (all three rows guarded: modular s lanes make the middle a congruence,
    // not an exact value), then the in-place BD.b/CR.b subtractions, the
    // BD.c/CR.c additions as subtractions of negations, the xi y-operand
    // prebuilds for BD.a/CR.a, the z.a.c0 y operands (BD.c is final by
    // then), and the AC.b/AC.c negations for the z.a.c1/c2 rows of walk 2.
    // Dead d2 slots repark every 64-byte scratch value: CD's negation in
    // cd+128/ce+128, CD"s in cd2+128/ce2+128, -CE.a in ad+128, -CE'.a in
    // ad2+128, the z.a.c0 y pair in ac_a+128/ac_c+128 with -BD.c.a in
    // ac_b+128, -AC.b in be+128/tt+128, -AC.c in be2+128/tt2+128.
    assert_eq!(t.len() * 8, F34L_TB_GSUB1 as usize);
    for k in 0..13 {
        let q = p(k);
        row3(&mut t, q + 64, q + 64, q);
        row3(&mut t, q + 64, q + 64, q + 128);
        row3(&mut t, q, q, q + 128);
    }
    for (minuend, s1, s2) in [(tt, ad, be), (tt2, ad2, be2)] {
        row3(&mut t, minuend, minuend, s1);
        row3(&mut t, minuend + 64, minuend + 64, s1 + 64);
        row3(&mut t, minuend, minuend, s2);
        row3(&mut t, minuend + 64, minuend + 64, s2 + 64);
    }
    for (dst, addend, spill_im) in [(be, cd, ce + 128), (be2, cd2, ce2 + 128)] {
        row3(&mut t, addend + 128, FSQ_ZERO8, addend);
        row3(&mut t, spill_im, FSQ_ZERO8, addend + 64);
        row3(&mut t, dst, dst, addend + 128);
        row3(&mut t, dst + 64, dst + 64, spill_im);
    }
    for (y_site, x_site) in [(ad, ce), (ad2, ce2)] {
        row3(&mut t, y_site, y_site, x_site + 64);
        row3(&mut t, y_site + 128, FSQ_ZERO8, x_site);
        row3(&mut t, y_site + 64, y_site + 64, y_site + 128);
    }
    row3(&mut t, ac_a + 128, ac_a, be + 64);
    row3(&mut t, ac_b + 128, FSQ_ZERO8, be);
    row3(&mut t, ac_c + 128, ac_a + 64, ac_b + 128);
    row3(&mut t, be + 128, FSQ_ZERO8, ac_b);
    row3(&mut t, tt + 128, FSQ_ZERO8, ac_b + 64);
    row3(&mut t, be2 + 128, FSQ_ZERO8, ac_c);
    row3(&mut t, tt2 + 128, FSQ_ZERO8, ac_c + 64);

    // Nine-fold rows (dst, x, y): BD.a = xi*CE + AD and CR.a = xi*CE' + AD'
    // complete in place over CE/CE' (the additions folded into the y
    // operands), z.a.c0 = xi*BD.c + AC.a into the dead CD region.
    assert_eq!(t.len() * 8, F34L_TB_NINE as usize);
    for (x_site, y_site) in [(ce, ad), (ce2, ad2)] {
        row3(&mut t, x_site, x_site, y_site);
        row3(&mut t, x_site + 64, x_site + 64, y_site + 64);
    }
    row3(&mut t, cd, be, ac_a + 128);
    row3(&mut t, cd + 64, be + 64, ac_c + 128);

    // Double-width sub walk 2: z.b = CR - AC - BD lane for lane (the z.b.c0
    // minuend CR.a sits in ce2 after the nine rows, BD.a in ce), then
    // z.a.c1 = BD.a - (-AC.b) over ce and z.a.c2 = BD.b - (-AC.c) over tt
    // (after the z.b rows consumed both as subtrahends).
    assert_eq!(t.len() * 8, F34L_TB_GSUB2 as usize);
    for (minuend, s1, s2) in [(ce2, ac_a, ce), (tt2, ac_b, tt), (be2, ac_c, be)] {
        row3(&mut t, minuend, minuend, s1);
        row3(&mut t, minuend, minuend, s2);
        row3(&mut t, minuend + 64, minuend + 64, s1 + 64);
        row3(&mut t, minuend + 64, minuend + 64, s2 + 64);
    }
    for (dst, neg_re, neg_im) in [(ce, be + 128, tt + 128), (tt, be2 + 128, tt2 + 128)] {
        row3(&mut t, dst, dst, neg_re);
        row3(&mut t, dst + 64, dst + 64, neg_im);
    }

    // MOD_DST is zero. The offsets use the repr(C) Fp12 order.
    assert_eq!(t.len() * 8, F34L_TB_MOD as usize);
    for (i, src) in [cd, ce, tt, ce2, tt2, be2].into_iter().enumerate() {
        t.extend([src as u64, (F34L_YOUT + 64 * i as i32) as u64]);
        t.extend([(src + 64) as u64, (F34L_YOUT + 64 * i as i32 + 32) as u64]);
    }

    // Product pair rows (x block, y block, product region).
    assert_eq!(t.len() * 8, F34L_TB_PROD as usize);
    for (x, y, dst) in [
        (xb(0), yb(0), ac_a),
        (xb(1), yb(0), ac_b),
        (xb(2), yb(0), ac_c),
        (xb(3), yb(1), ad),
        (xb(4), yb(2), be),
        (xb(5), yb(1), cd),
        (xb(5), yb(2), ce),
        (F34L_XS1, F34L_YS1, tt),
        (t1(0), F34L_Y3, ad2),
        (t1(1), yb(2), be2),
        (t1(2), F34L_Y3, cd2),
        (t1(2), yb(2), ce2),
        (F34L_XS2, F34L_YS2, tt2),
    ] {
        row3(&mut t, x, y, dst);
    }

    // Product sub-rows (operand sub-offset, destination sub-offset):
    // d1 = s*s first, then d0 = re*re, then d2 = im*im.
    assert_eq!(t.len() * 8, F34L_TB_MSUB as usize);
    t.extend([64, 64, 0, 0, 32, 128]);
    assert_eq!(t.len() * 8, F34L_TB_END as usize);
    t
}

/// Stage `count` canonical Fp2 operands from `[rsi]` as consecutive 96-byte
/// frame blocks `[re, im, s = re + im]` at `[rdi]` (raw s < 2p), advancing
/// both pointers. The block shape every product sub-row consumes.
fn f34l_stage_blocks<M: Machine>(m: &mut M, count: i32, label: &str) {
    m.xor_clear(Rcx, "block counter");
    m.stride_loop(Rcx, 96, LoopEnd::Imm(96 * count), label, &mut |m| {
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("re[{k}]"));
        }
        for (k, reg) in [R12, R13, R14, R15].into_iter().enumerate() {
            m.load(reg, Mem::new(Rsi, 32 + 8 * k as i32), &format!("im[{k}]"));
        }
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.store(Mem::new(Rdi, 8 * k as i32), reg, &format!("block re[{k}]"));
        }
        for (k, reg) in [R12, R13, R14, R15].into_iter().enumerate() {
            m.store(
                Mem::new(Rdi, 32 + 8 * k as i32),
                reg,
                &format!("block im[{k}]"),
            );
        }
        m.add(R8, R12, "s = re + im, limb 0");
        m.adc(R9, R13, "limb 1");
        m.adc(R10, R14, "limb 2");
        m.adc(R11, R15, "limb 3");
        m.claim_flags_clear("re + im < 2p < 2^256: s fits four limbs");
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.store(
                Mem::new(Rdi, 64 + 8 * k as i32),
                reg,
                &format!("block s[{k}]"),
            );
        }
        m.add_imm(Rsi, 64, "next source Fp2");
        m.add_imm(Rdi, 96, "next block");
    });
}

/// `narsil_fp12_034l_x86`: the Miller loop's sparse Fp12 line update
/// `z = f * (c0 + c3*w + c4*v*w)` in mcl's lazy double-width shape (mcl
/// `mul_403`, pairing_impl.hpp) -- 39 raw 4x4 products and 12 Montgomery
/// reductions where the v1 leaf (`narsil_fp12_034_x86`) pays 72 products
/// and 12 interleaved reductions and mcl itself pays 39 + 18 (this kernel
/// defers mcl's per-stage `Fp2Dbl::mod` to one reduction per output Fp).
///
/// # Semantics (exactly `Fp12::mul_by_034_assign_sosd6`'s value)
///
/// For `f = a0 + a1*w` (`w^2 = v`, `v^3 = xi = 9 + u`) and the sparse
/// multiplier `X0 + X1*w` with `X0 = (c0, 0, 0)`, `X1 = (c3, c4, 0)`
/// (mcl's `(a, b, c) -> (b, 0, 0, c, a, 0)` slot map reads b = c0, c = c3,
/// a = c4 in our convention):
///
/// * `AC = a0*X0`: three Fp2Dbl products, coefficient-wise (9 raw products).
/// * `BD = a1*X1` and `CR = (a0 + a1)*(X0 + X1)`: mcl `Fp6mul_01` with
///   Karatsuba on the v-coefficient -- AD, BE, CD, CE and
///   `TT = (A + B)(d + e)` (5 Fp2Dbl products each, 30 raw products), then
///   `.a = AD + xi*CE`, `.b = TT - AD - BE`, `.c = BE + CD`, all on
///   512-bit values.
/// * `z.a = AC + v*BD = (AC.a + xi*BD.c, AC.b + BD.a, AC.c + BD.b)`,
///   `z.b = CR - AC - BD`, still double-width. ONE Montgomery reduction per
///   output Fp (12 total).
///
/// # Bounds (BN254: p < 2^253.61, K = 2^256, pK < 2^510)
///
/// Simpler than fp12_sqr's: every double-width value in this kernel stays
/// below pK at every step, so no exact-passage argument is needed.
///
/// * block rows: staged singles are canonical with raw s = re + im < 2p.
///   the six built blocks (t1, Y3, XS1, XS2, YS1, YS2) are fully canonical
///   via modular add rows. Reducing a Karatsuba s operand mod p is sound
///   here because every later stage is a congruence mod pK and the final
///   Montgomery reduction only needs T < pK and T = z*2^256 mod p.
/// * raw products: re/im rows < p everywhere, so d0, d2 < p^2 and
///   d1 = s*s < 4p^2 < pK (needs 4p < 2^256: BN254 yes).
/// * every double-width subtraction is guarded mod pK (the modular s lanes
///   make the Karatsuba middles congruences that may wrap). Guarded
///   subtraction is closed on values < pK, additions enter as subtractions
///   of guarded negations, so no double-width add body exists at all.
/// * nine rows: operands < pK have high halves < p, output < pK with a
///   mu-canonical high half.
/// * every reduced value satisfies T < pK, the Montgomery precondition:
///   `(T + m*p*2^256)/2^256 < 2p`, one conditional subtraction, canonical.
///
/// The interpreter asserts the flag claims on every path. The u512
/// reference in `kernelgen_verify` asserts each stage bound on random and
/// adversarial inputs.
///
/// # In-place update
///
/// `z == f` is allowed and is the production shape (`ell()` updates the
/// Miller accumulator in place): the prologue stages all of `f` into the
/// frame and `f` is never read again, so no output store can alias a live
/// operand.
///
/// # Structure
///
/// No outer loop (the cyc_sqr shape): staging, one modular-add walk, the
/// 13-pair product walk, then guarded-sub / nine / guarded-sub / reduction
/// walks in one pass -- each shared body emitted once except the guarded
/// sub, which runs again after the nine rows because z.b subtracts the
/// nine outputs BD.a and CR.a. All Fp6Dbl assembly happens in place over
/// the product regions. The 13 dead d2 slots park every negation and
/// nine-fold y operand, so the frame carries no separate scratch area.
///
/// Arguments: `(z: *mut u64x48, f: *const u64x48, c: *const u64x24,
/// consts: *const { p[4], -p^-1, mu })` in rdi, rsi, rdx, rcx. `f` is
/// repr(C) Fp12. `c` is the three sparse coefficients c0, c3, c4 as
/// contiguous Fp2s (the v1 leaf's ABI exactly). All inputs canonical.
/// outputs canonical.
pub fn fp12_034l_x86<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    let tables = fp12_034l_tables();
    m.rodata(F34L_TAB_LABEL, &tables);
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    m.alloc_stack(F34L_FRAME);
    m.comment(
        "frame: p +0, -p^-1 +32, mu +40, z +48, walk bound +64, table base +88, m-walk end +96, mod dst +136, zero8 +192, staged f/c blocks +256, t1/Y3/sum blocks +1120, products +1888, outputs +4384",
    );
    m.store(Mem::new(rsp, FSQ_Z), Rdi, "spill z");
    for k in 0..4 {
        m.load(Rax, Mem::new(Rcx, 8 * k), &format!("p{k}"));
        m.store(
            Mem::new(rsp, FP6_P + 8 * k),
            Rax,
            "cancel rows address the frame as a consts table",
        );
    }
    m.load(Rax, Mem::new(Rcx, 32), "-p^-1");
    m.store(Mem::new(rsp, FP6_PINV), Rax, "-p^-1");
    m.load(Rax, Mem::new(Rcx, 40), "mu = floor(2^310/p)");
    m.store(Mem::new(rsp, FP6_MU), Rax, "mu");
    m.lea_rodata(Rax, F34L_TAB_LABEL, "walk tables");
    m.store(Mem::new(rsp, FSQ_TBL), Rax, "table base");
    m.mov(Rbx, Rax, "");
    m.add_imm(Rbx, F34L_TB_MSUB + 48, "product m-walk end");
    m.store(Mem::new(rsp, FSQ_MEND), Rbx, "");
    m.xor_clear(Rax, "");
    for k in 0..8 {
        m.store(
            Mem::new(rsp, FSQ_ZERO8 + 8 * k),
            Rax,
            "zero word (negation rows subtract from it)",
        );
    }
    m.store(
        Mem::new(rsp, FSQ_MOD_DST),
        Rax,
        "mod rows carry absolute destination offsets",
    );

    m.comment("");
    m.comment("stage f then c as [re, im, s] blocks: all later reads are");
    m.comment("frame-relative, which is what makes z == f safe (no operand");
    m.comment("read after any z store)");
    m.mov(Rdi, rsp, "");
    m.add_imm(Rdi, F34L_XB, "staging cursor (c's blocks follow f's)");
    f34l_stage_blocks(m, 6, ".Lf34l_stf");
    m.mov(Rsi, MULTIPLIER, "c pointer");
    f34l_stage_blocks(m, 3, ".Lf34l_stc");

    m.comment("");
    m.comment("modular add walk: t1 = a0 + a1, Y3 = C0 + C3, then the four");
    m.comment("Karatsuba sum blocks, every row canonical");
    fsq_walk_setup(m, Rbp, F34L_TB_MADD, 24 * 24, FSQ_WALK_END);
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lf34l_madd",
        &mut |m| dbl_modadd_row(m),
    );

    m.comment("");
    m.comment("products: 13 sparse block pairs x 3 sub-products, rolled 4x4");
    m.comment("mulpre rounds; the pair walk reads (x block, y block, region)");
    fsq_walk_setup(m, R15, F34L_TB_PROD, 13 * 24, FSQ_WALK_END);
    m.stride_loop(
        R15,
        24,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lf34l_prod_k",
        &mut |m| {
            m.load(Rcx, Mem::new(rsp, FSQ_TBL), "");
            m.add_imm(Rcx, F34L_TB_MSUB, "sub-product walk");
            m.stride_loop(
                Rcx,
                16,
                LoopEnd::Mem(Mem::new(rsp, FSQ_MEND)),
                ".Lf34l_prod_m",
                &mut |m| {
                    m.load(Rax, Mem::new(Rcx, 0), "operand sub-offset");
                    m.load(Rbx, Mem::new(Rcx, 8), "destination sub-offset");
                    m.mov(Rsi, rsp, "");
                    m.add_mem(Rsi, Mem::new(R15, 0), "x block of this pair");
                    m.add(Rsi, Rax, "PA: x sub-row (the multiplicand)");
                    m.mov(Rdi, rsp, "");
                    m.add_mem(Rdi, Mem::new(R15, 8), "y block of this pair");
                    m.add(Rdi, Rax, "PY: y sub-row");
                    m.mov(Rbp, rsp, "");
                    m.add_mem(Rbp, Mem::new(R15, 16), "product region");
                    m.add(Rbp, Rbx, "PZ");
                    for (k, t) in T.into_iter().enumerate() {
                        m.xor_clear(t, &format!("t{k} = 0"));
                    }
                    m.xor_clear(R14, "round cursor: byte offset 8j of the x limb");
                    m.stride_loop(R14, 8, LoopEnd::Imm(32), ".Lf34l_prod_j", &mut |m| {
                        m.load_indexed(MULTIPLIER, Rsi, R14, "x[j], the row multiplicand");
                        m.xor_clear(LO, "re-seed CF = OF = 0 (back edge clobbered flags)");
                        fsq_mulpre_row(m, Rdi);
                        m.store(Mem::new(Rbp, 0), T[0], "product limb j is final");
                        m.add_imm(Rbp, 8, "next output limb");
                        m.comment("shift down one word");
                        for k in 0..5 {
                            m.mov(T[k], T[k + 1], &format!("t{k} = t{}", k + 1));
                        }
                        m.xor_clear(T[5], "t5 = 0 (CF/OF stay clear)");
                    });
                    for (k, t) in T[..4].iter().enumerate() {
                        m.store(
                            Mem::new(Rbp, 8 * k as i32),
                            *t,
                            &format!("product limb {}", k + 4),
                        );
                    }
                },
            );
        },
    );

    m.comment("");
    m.comment("double-width sub walk 1: Karatsuba assembly, BD.b/CR.b, the");
    m.comment("BD.c/CR.c combines, xi y operands, z.a negations");
    fsq_walk_setup(
        m,
        Rbp,
        F34L_TB_GSUB1,
        F34L_TB_NINE - F34L_TB_GSUB1,
        FSQ_WALK_END,
    );
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lf34l_gsub1",
        &mut |m| dbl_gsub_row(m),
    );

    m.comment("");
    m.comment("nine-fold walk: BD.a, CR.a and z.a.c0 complete as 9x + y with");
    m.comment("the additions folded into the y operands, mu-canonical highs");
    cyc_walk_bound(m, F34L_TB_GSUB2);
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lf34l_nine",
        &mut |m| dbl_nine_row(m),
    );

    m.comment("");
    m.comment("double-width sub walk 2: z.b = CR - AC - BD (BD.a and CR.a are");
    m.comment("nine outputs, hence the second pass), then z.a.c1/c2");
    cyc_walk_bound(m, F34L_TB_MOD);
    m.stride_loop(
        Rbp,
        24,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lf34l_gsub2",
        &mut |m| dbl_gsub_row(m),
    );

    m.comment("");
    m.comment("Montgomery reduction walk: 12 output coefficients into YOUT");
    cyc_walk_bound(m, F34L_TB_PROD);
    m.stride_loop(
        Rbp,
        16,
        LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
        ".Lf34l_mod",
        &mut |m| dbl_mod_row(m),
    );

    m.comment("");
    m.comment("copy out: z.a then z.b are contiguous, 48 limbs to z");
    m.load(Rdi, Mem::new(rsp, FSQ_Z), "z");
    m.mov(Rsi, rsp, "");
    m.add_imm(Rsi, F34L_YOUT, "output base");
    m.xor_clear(Rcx, "");
    m.stride_loop(Rcx, 32, LoopEnd::Imm(384), ".Lf34l_out", &mut |m| {
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("z limb {k}"));
        }
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.store(Mem::new(Rdi, 8 * k as i32), reg, "z");
        }
        m.add_imm(Rsi, 32, "");
        m.add_imm(Rdi, 32, "");
    });
    m.free_stack(F34L_FRAME);
    for reg in CALLEE_SAVED.iter().rev() {
        m.pop(*reg);
    }
    m.ret();
}

/// Register roles for `narsil_fp12_034k_x86` (walk bodies shared with
/// fp12_sqr/fp12_mul/cyc_sqr/fp12_034l, so the roles mirror theirs).
pub const FP12_034K_REGISTER_MAP: &[(Reg, &str)] = &[
    (
        Rdi,
        "z on entry (spilled); staging/xi destination pointer; walk destination pointer",
    ),
    (
        Rsi,
        "f pointer on entry (g staging source); PA / walk source-1 pointer; mask and mu scratch",
    ),
    (
        Rdx,
        "c pointer on entry; the implicit mulx multiplicand; walk scratch",
    ),
    (
        Rcx,
        "consts pointer on entry; staging half cursor; product-entry cursor; walk source-2 pointer",
    ),
    (R8, "accumulator/value word 0"),
    (R9, "accumulator/value word 1"),
    (R10, "accumulator/value word 2"),
    (R11, "accumulator/value word 3"),
    (R12, "accumulator/value word 4"),
    (R13, "accumulator/value word 5; mu-reduction scratch"),
    (
        R14,
        "staging scratch; xi site cursor; product round cursor (byte offset 8j)",
    ),
    (
        R15,
        "staging scratch; xi half cursor; coefficient cursor (spilled around the product walk, where it is the sub-product cursor)",
    ),
    (
        Rbp,
        "product region pointer; the row cursor of the assembly walks",
    ),
    (
        Rax,
        "low half of the current product; zero for chain closes; borrow mask",
    ),
    (
        Rbx,
        "high half of the current product; staging component counter; prologue scratch",
    ),
];

// fp12_034k frame layout. Every slot a shared walk body addresses
// (p/-p^-1/mu, z, walk bound, table base, m-walk end, MOD_DST) sits at its
// fp12_sqr offset, so the bodies emit unchanged.
/// Coefficient-walk bound: the absolute address of the coefficient table end.
const F34K_CEND: i32 = 56;
/// Product-entry walk bound, rewritten per coefficient.
const F34K_EEND: i32 = 72;
/// Coefficient-cursor spill: the product walk reuses its register.
const F34K_CJ: i32 = 104;
/// Five 128-byte y blocks `[p - im, re, im, s]`: C0, C3, C4, X3, X4. The
/// negp row exists only as the xi pass's subtrahend (C3, C4). The X blocks
/// leave it unwritten.
const F34K_YB: i32 = 144;
/// Six 96-byte g blocks `[re, im, s]`: f in W-power order a0, b0, a1, b1,
/// a2, b2 (no duplicate slots: the wrap offsets are precomputed per row).
const F34K_G: i32 = 784;
/// Three 192-byte product regions (d0 +0, d1 +64, d2 +128), one per entry
/// of the current coefficient.
const F34K_PROD: i32 = 1360;
/// The twelve reduced output coefficients in repr(C) order, copied to z last.
const F34K_YOUT: i32 = 1936;
const F34K_FRAME: i32 = 2320;

/// Byte offsets of the table regions inside the fp12_034k rodata blob.
/// gsub through mod are contiguous in walk order (one cursor marches, phase
/// boundaries store the next bound). The sub-product and coefficient rows
/// have their own setups.
const F34K_TB_GSUB: i32 = 0; // 9 rows x 3
const F34K_TB_GADD: i32 = 216; // 4 rows x 3
const F34K_TB_MOD: i32 = 312; // 2 rows x 2
const F34K_TB_MSUB: i32 = 344; // 3 rows x 2
const F34K_TB_COEFF: i32 = 392; // 6 rows x 10
const F34K_TB_END: i32 = 872;

const F34K_TAB_LABEL: &str = ".Lf34k_tab";

/// The fp12_034k read-only walk tables (rsp-relative offsets).
///
/// The assembly rows cover ONE coefficient (the six coefficients share the
/// three product regions and rerun the same rows). The coefficient rows bake
/// the v1 leaf's per-component state -- the W-to-repr(C) output offset and
/// the wrap selection e1.y = X4 exactly for j < 3, e2.y = X3 exactly at
/// j = 0, g offsets mod 6 -- into constants, so no runtime cmov or offset
/// arithmetic exists at all.
fn fp12_034k_tables() -> Vec<u64> {
    let mut t: Vec<u64> = Vec::new();
    let row3 = |t: &mut Vec<u64>, dst: i32, s1: i32, s2: i32| {
        t.extend([dst as u64, s1 as u64, s2 as u64]);
    };
    let p = |k: i32| F34K_PROD + 192 * k;

    // Guarded-sub rows: per region the Karatsuba assembly d1 -= d0,
    // d1 -= d2 (the middle term a0*b1 + a1*b0, a congruence where an s lane
    // was reduced mod p), then d0 -= d2 (the real lane a0*b0 - a1*b1).
    for k in 0..3 {
        let q = p(k);
        row3(&mut t, q + 64, q + 64, q);
        row3(&mut t, q + 64, q + 64, q + 128);
        row3(&mut t, q, q, q + 128);
    }

    // Guarded-add rows: both lanes of regions 1 and 2 fold into region 0.
    assert_eq!(t.len() * 8, F34K_TB_GADD as usize);
    for lane in [0, 64] {
        for k in 1..3 {
            row3(&mut t, p(0) + lane, p(0) + lane, p(k) + lane);
        }
    }

    // Montgomery reduction rows (src, dst offset relative to the MOD_DST
    // slot, which holds this coefficient's YOUT offset).
    assert_eq!(t.len() * 8, F34K_TB_MOD as usize);
    t.extend([p(0) as u64, 0, (p(0) + 64) as u64, 32]);

    // Product sub-rows (operand sub-offset, destination sub-offset):
    // d1 = s*s first, then d0 = re*re, then d2 = im*im.
    assert_eq!(t.len() * 8, F34K_TB_MSUB as usize);
    t.extend([64, 64, 0, 0, 32, 128]);

    // Coefficient rows: the YOUT offset (z + 192*(j&1) + 64*(j>>1) in W
    // order h_0..h_5), then three product entries (y re-row, g block,
    // region). G blocks have no negp row, so their re-row IS the block. The
    // y fields are pre-biased +32 past the negp row so one sub-offset
    // addresses both sides.
    assert_eq!(t.len() * 8, F34K_TB_COEFF as usize);
    let yb = |b: i32| F34K_YB + 128 * b + 32;
    let g = |slot: i32| F34K_G + 96 * slot;
    for j in 0..6i32 {
        t.push((F34K_YOUT + 192 * (j & 1) + 64 * (j >> 1)) as u64);
        let entries = [
            (yb(0), g(j)),
            (yb(if j < 3 { 4 } else { 2 }), g((j + 3) % 6)),
            (yb(if j == 0 { 3 } else { 1 }), g((j + 5) % 6)),
        ];
        for (k, (y, g_off)) in entries.into_iter().enumerate() {
            t.extend([y as u64, g_off as u64, (p(k as i32)) as u64]);
        }
    }
    assert_eq!(t.len() * 8, F34K_TB_END as usize);
    t
}

/// `narsil_fp12_034k_x86`: the Miller loop's sparse Fp12 line update
/// `z = f * (c0 + c3*w + c4*v*w)` in the v1 leaf's uniform W-basis walk
/// shape with Karatsuba inside every Fp2 product -- 54 raw 4x4 products and
/// 12 Montgomery reductions where the v1 leaf (`narsil_fp12_034_x86`) pays
/// 72 products and 12 interleaved reductions. The lazy 034l leaf reaches 39
/// products but through a serial staged DAG. This kernel keeps v1's
/// execution shape (six identical, independent coefficient bodies) and cuts
/// only the product mass.
///
/// # Semantics (exactly `Fp12::mul_by_034_assign_sosd6`'s value)
///
/// In the W-power basis `W = w` (`W^6 = xi = 9 + u`) the element is
/// `sum g_k W^k` with `g = (a0, b0, a1, b1, a2, b2)` for `f = a + b*w`, and
/// the sparse multiplier is `c0 + c3 W + c4 W^3`, so every output
/// coefficient is one Fp2 sum of three Fp2 products:
///
/// * `h_j = g_j*c0 + g_{(j+5) mod 6}*C3' + g_{(j+3) mod 6}*C4'`,
/// * `C3' = xi*c3` exactly at j = 0, `C4' = xi*c4` exactly for j < 3 (the
///   wrap terms), both xi values computed in-kernel via the mu
///   quotient-estimate reduction.
///
/// Each Fp2 product runs as Karatsuba over raw 4x4 products (d0 = re*re,
/// d2 = im*im, d1 = s*s for the staged pre-sums s = re + im), assembled
/// double-width: real lane d0 - d2, imag lane d1 - d0 - d2, all guarded mod
/// p*2^256. The three products of a coefficient then fold lane-wise into
/// region 0 with guarded adds, and ONE Montgomery reduction per output Fp
/// (12 total) lands the canonical result.
///
/// # Bounds (BN254: p < 2^253.61, K = 2^256, pK < 2^510)
///
/// * staged singles canonical. Pre-sums s = re + im raw < 2p, EXCEPT the X
///   blocks' s, whose re/im are the mu-canonical xi outputs (< p), so
///   s < 2p there too. No pre-sum canonicalization is needed: with both
///   sums < 2p every raw product obeys d1 = s*s < 4p^2 < pK (needs
///   4p < 2^256: BN254 yes, mcl's isLtQuad argument), and d0, d2 < p^2.
/// * Karatsuba assembly: all three rows guarded mod pK. D1 - d0 - d2 is
///   exact and nonnegative when both s lanes are raw, but the X blocks'
///   xi-canonical rows make d1 a congruence that may wrap, so the guard is
///   load-bearing there (subtrahends < p^2 < pK keep it closed). D0 - d2
///   may always wrap.
/// * lane folds: three guarded values < pK each, pairwise guarded adds stay
///   < pK (sums < 2pK < 2^511, one high-half conditional subtraction).
/// * every reduced value satisfies T < pK, the Montgomery precondition:
///   `(T + m*p*2^256)/2^256 < 2p`, one conditional subtraction, canonical.
///
/// The interpreter asserts the flag claims on every path. The u512
/// reference in `kernelgen_verify` asserts each stage bound on random and
/// adversarial inputs.
///
/// # In-place update
///
/// `z == f` is allowed and is the production shape (`ell()` updates the
/// Miller accumulator in place): the prologue stages all of `f` into the
/// frame and `f` is never read again. Z is written only by the final
/// copy-out.
///
/// # Structure
///
/// One outer six-iteration coefficient walk over rodata rows that bake the
/// v1 leaf's per-component state -- output offset, wrap y selection, wrapped
/// g offsets -- into constants (v1 recomputes them with cmovs every
/// iteration). Per coefficient: a 3-entry product walk (each entry one
/// block pair, three rolled 4x4 mulpre sub-products into its region), then
/// guarded-sub / guarded-add / reduction walks over one shared row table.
/// All double-width bodies are the shared fp12_sqr walk bodies, emitted
/// once each.
///
/// Arguments: `(z: *mut u64x48, f: *const u64x48, c: *const u64x24,
/// consts: *const { p[4], -p^-1, mu })` in rdi, rsi, rdx, rcx. `f` is
/// repr(C) Fp12. `c` is the three sparse coefficients c0, c3, c4 as
/// contiguous Fp2s (the v1 leaf's ABI exactly). All inputs canonical.
/// outputs canonical.
pub fn fp12_034k_x86<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    let tables = fp12_034k_tables();
    m.rodata(F34K_TAB_LABEL, &tables);
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    m.alloc_stack(F34K_FRAME);
    m.comment(
        "frame: p +0, -p^-1 +32, mu +40, z +48, coefficient bound +56, walk bound +64, entry bound +72, table base +88, m-walk end +96, cursor spill +104, mod dst +136, y blocks +144, g blocks +784, products +1360, outputs +1936",
    );
    m.store(Mem::new(rsp, FSQ_Z), Rdi, "spill z");
    for k in 0..4 {
        m.load(Rax, Mem::new(Rcx, 8 * k), &format!("p{k}"));
        m.store(
            Mem::new(rsp, FP6_P + 8 * k),
            Rax,
            "cancel rows address the frame as a consts table",
        );
    }
    m.load(Rax, Mem::new(Rcx, 32), "-p^-1");
    m.store(Mem::new(rsp, FP6_PINV), Rax, "-p^-1");
    m.load(Rax, Mem::new(Rcx, 40), "mu = floor(2^310/p)");
    m.store(Mem::new(rsp, FP6_MU), Rax, "mu");
    m.lea_rodata(Rax, F34K_TAB_LABEL, "walk tables");
    m.store(Mem::new(rsp, FSQ_TBL), Rax, "table base");
    m.mov(Rbx, Rax, "");
    m.add_imm(Rbx, F34K_TB_MSUB + 48, "product m-walk end");
    m.store(Mem::new(rsp, FSQ_MEND), Rbx, "");
    m.mov(Rbx, Rax, "");
    m.add_imm(Rbx, F34K_TB_END, "coefficient walk end");
    m.store(Mem::new(rsp, F34K_CEND), Rbx, "");

    m.comment("");
    m.comment("g blocks: f staged in W-power order as [re, im, s] with raw");
    m.comment("s = re + im. The source walks f linearly (a0..a2 then b0..b2);");
    m.comment("the destination interleaves W order as half bias + 192 per");
    m.comment("component. f is fully staged before any z store, which is");
    m.comment("what makes z == f safe");
    m.xor_clear(Rcx, "half cursor: a blocks (+0) then b blocks (+96)");
    m.stride_loop(Rcx, 96, LoopEnd::Imm(192), ".Lf34k_gh", &mut |m| {
        m.mov(Rdi, rsp, "");
        m.add(Rdi, Rcx, "+ half bias");
        m.add_imm(Rdi, F34K_G, "this half's first g block");
        m.xor_clear(Rbx, "component counter");
        m.stride_loop(Rbx, 64, LoopEnd::Imm(192), ".Lf34k_g", &mut |m| {
            for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
                m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("f.re[{k}]"));
            }
            for (k, reg) in [R12, R13, R14, R15].into_iter().enumerate() {
                m.load(reg, Mem::new(Rsi, 32 + 8 * k as i32), &format!("f.im[{k}]"));
            }
            for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
                m.store(Mem::new(Rdi, 8 * k as i32), reg, &format!("block re[{k}]"));
            }
            for (k, reg) in [R12, R13, R14, R15].into_iter().enumerate() {
                m.store(
                    Mem::new(Rdi, 32 + 8 * k as i32),
                    reg,
                    &format!("block im[{k}]"),
                );
            }
            m.add(R8, R12, "s = re + im, limb 0");
            m.adc(R9, R13, "limb 1");
            m.adc(R10, R14, "limb 2");
            m.adc(R11, R15, "limb 3");
            m.claim_flags_clear("re + im < 2p < 2^256: s fits four limbs");
            for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
                m.store(
                    Mem::new(Rdi, 64 + 8 * k as i32),
                    reg,
                    &format!("block s[{k}]"),
                );
            }
            m.add_imm(Rsi, 64, "next source Fp2");
            m.add_imm(Rdi, 192, "next block of this half");
        });
    });

    m.comment("");
    m.comment("y blocks: c0, c3, c4 staged as [p - im, re, im, s]; the negp");
    m.comment("row exists only as the xi pass's subtrahend");
    m.mov(Rsi, MULTIPLIER, "c pointer");
    m.mov(Rdi, rsp, "");
    m.add_imm(Rdi, F34K_YB, "block cursor");
    m.xor_clear(Rcx, "three coefficient blocks");
    m.stride_loop(Rcx, 64, LoopEnd::Imm(192), ".Lf34k_c", &mut |m| {
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("c_i.re[{k}]"));
        }
        for (k, reg) in [R12, R13, R14, R15].into_iter().enumerate() {
            m.load(
                reg,
                Mem::new(Rsi, 32 + 8 * k as i32),
                &format!("c_i.im[{k}]"),
            );
        }
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.store(
                Mem::new(Rdi, 32 + 8 * k as i32),
                reg,
                &format!("block re[{k}]"),
            );
        }
        for (k, reg) in [R12, R13, R14, R15].into_iter().enumerate() {
            m.store(
                Mem::new(Rdi, 64 + 8 * k as i32),
                reg,
                &format!("block im[{k}]"),
            );
        }
        m.add(R8, R12, "s = re + im, limb 0");
        m.adc(R9, R13, "limb 1");
        m.adc(R10, R14, "limb 2");
        m.adc(R11, R15, "limb 3");
        m.claim_flags_clear("re + im < 2p < 2^256: s fits four limbs");
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.store(
                Mem::new(Rdi, 96 + 8 * k as i32),
                reg,
                &format!("block s[{k}]"),
            );
        }
        m.comment("negp row: the xi subtrahend enters as p - im");
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.load(reg, Mem::new(rsp, FP6_P + 8 * k as i32), &format!("p{k}"));
        }
        m.sub_rr(R8, R12, "p0 - c_i.im[0]");
        m.sbb_rr(R9, R13, "p1 - c_i.im[1]");
        m.sbb_rr(R10, R14, "p2 - c_i.im[2]");
        m.sbb_rr(R11, R15, "p3 - c_i.im[3]");
        m.claim_flags_clear("im < p (canonical): p - im cannot borrow");
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.store(
                Mem::new(Rdi, 8 * k as i32),
                reg,
                &format!("block negp[{k}]"),
            );
        }
        m.add_imm(Rsi, 64, "next source Fp2");
        m.add_imm(Rdi, 128, "next block");
    });

    m.comment("");
    m.comment("xi scaling: X3 = xi*c3 from C3, then X4 = xi*c4 from C4, each");
    m.comment("half 9*A + C reduced canonical by the mu quotient estimate;");
    m.comment("the site tail adds the X block's raw s row");
    m.xor_clear(R14, "site cursor: the C3 block (+0) then C4 (+128)");
    m.stride_loop(R14, 128, LoopEnd::Imm(256), ".Lf34k_xi", &mut |m| {
        m.xor_clear(R15, "half cursor: real output (+0) then imag (+32)");
        m.stride_loop(R15, 32, LoopEnd::Imm(64), ".Lf34k_xi_val", &mut |m| {
            m.comment(
                "C row = [PC]: negp(im) for re = 9re - im, re for im = 9im + re; A row = [PC + 32]",
            );
            m.mov(Rsi, rsp, "");
            m.add(Rsi, R14, "+ site block");
            m.add(Rsi, R15, "+ pass");
            m.add_imm(Rsi, F34K_YB + 128, "PC");
            m.mov(Rdi, Rsi, "");
            m.add_imm(
                Rdi,
                256 + 32,
                "output row: each X row sits 288 bytes above its C row",
            );
            let v = [R8, R9, R10, R11, R12];
            m.xor_clear(MULTIPLIER, "");
            m.add_imm(MULTIPLIER, 9, "xi = 9 + u: the real scale is one mulx row");
            m.mulx_mem(Rbx, v[0], Mem::new(Rsi, 32), "9*A[0] -> (v0, hi)");
            m.mulx_mem(Rcx, v[1], Mem::new(Rsi, 40), "9*A[1] -> (lo, hi)");
            m.add(v[1], Rbx, "v1 += hi(9*A[0])");
            m.mulx_mem(Rbx, v[2], Mem::new(Rsi, 48), "9*A[2] -> (lo, hi)");
            m.adc(v[2], Rcx, "v2 += hi(9*A[1])");
            m.mulx_mem(v[4], v[3], Mem::new(Rsi, 56), "9*A[3] -> (lo, v4)");
            m.adc(v[3], Rbx, "v3 += hi(9*A[2])");
            m.adc_zero(v[4], "9A < 9p: the chain closes into the top limb");
            for (k, reg) in v[..4].iter().enumerate() {
                let what = format!("+= C[{k}]");
                if k == 0 {
                    m.add_mem(*reg, Mem::new(Rsi, 0), &what);
                } else {
                    m.adc_mem(*reg, Mem::new(Rsi, 8 * k as i32), &what);
                }
            }
            m.adc_zero(v[4], "value < 10p < 2^257: top limb is 0 or 1");
            fsq_mu_reduce5(m, v, [R13, Rbx, Rcx, Rbp, Rsi]);
            m.comment("one conditional subtraction reaches canonical (< 1.33p)");
            fsq_csub_store(
                m,
                [v[0], v[1], v[2], v[3]],
                [Rax, Rbx, Rcx, MULTIPLIER],
                Rdi,
                0,
            );
        });
        m.comment("s row of the X block just written: raw re + im (< 2p)");
        m.mov(Rsi, rsp, "");
        m.add(Rsi, R14, "+ site block offset");
        m.add_imm(Rsi, F34K_YB + 128 + 256, "the X block of this site");
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.load(reg, Mem::new(Rsi, 32 + 8 * k as i32), &format!("X.re[{k}]"));
        }
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            let what = format!("+= X.im[{k}]");
            if k == 0 {
                m.add_mem(reg, Mem::new(Rsi, 64), &what);
            } else {
                m.adc_mem(reg, Mem::new(Rsi, 64 + 8 * k as i32), &what);
            }
        }
        m.claim_flags_clear("re + im < 2p < 2^256: s fits four limbs");
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.store(
                Mem::new(Rsi, 96 + 8 * k as i32),
                reg,
                &format!("block s[{k}]"),
            );
        }
    });

    m.comment("");
    m.comment("coefficient walk: per rodata row the YOUT offset and three");
    m.comment("product entries (y re-row, g block, region) with the wrap");
    m.comment("selection and W-to-repr(C) mapping baked in");
    m.load(R15, Mem::new(rsp, FSQ_TBL), "table base");
    m.add_imm(R15, F34K_TB_COEFF, "coefficient cursor");
    m.stride_loop(
        R15,
        80,
        LoopEnd::Mem(Mem::new(rsp, F34K_CEND)),
        ".Lf34k_comp",
        &mut |m| {
            m.store(Mem::new(rsp, F34K_CJ), R15, "spill the coefficient cursor");
            m.load(Rax, Mem::new(R15, 0), "this coefficient's output offset");
            m.store(Mem::new(rsp, FSQ_MOD_DST), Rax, "reduction destination");
            m.mov(Rcx, R15, "");
            m.add_imm(Rcx, 8, "product-entry cursor");
            m.mov(Rax, R15, "");
            m.add_imm(Rax, 80, "");
            m.store(Mem::new(rsp, F34K_EEND), Rax, "entry-walk bound");
            m.stride_loop(
                Rcx,
                24,
                LoopEnd::Mem(Mem::new(rsp, F34K_EEND)),
                ".Lf34k_ent",
                &mut |m| {
                    m.load(R15, Mem::new(rsp, FSQ_TBL), "");
                    m.add_imm(R15, F34K_TB_MSUB, "sub-product walk");
                    m.stride_loop(
                        R15,
                        16,
                        LoopEnd::Mem(Mem::new(rsp, FSQ_MEND)),
                        ".Lf34k_prod_m",
                        &mut |m| {
                            m.load(Rax, Mem::new(R15, 0), "operand sub-offset");
                            m.load(Rbx, Mem::new(R15, 8), "destination sub-offset");
                            m.mov(Rsi, rsp, "");
                            m.add_mem(Rsi, Mem::new(Rcx, 8), "g block of this entry");
                            m.add(Rsi, Rax, "PA: the multiplicand sub-row");
                            m.mov(Rdi, rsp, "");
                            m.add_mem(Rdi, Mem::new(Rcx, 0), "y re-row of this entry");
                            m.add(Rdi, Rax, "PY: the y sub-row");
                            m.mov(Rbp, rsp, "");
                            m.add_mem(Rbp, Mem::new(Rcx, 16), "product region");
                            m.add(Rbp, Rbx, "PZ");
                            for (k, t) in T.into_iter().enumerate() {
                                m.xor_clear(t, &format!("t{k} = 0"));
                            }
                            m.xor_clear(R14, "round cursor: byte offset 8j of the g limb");
                            m.stride_loop(R14, 8, LoopEnd::Imm(32), ".Lf34k_prod_j", &mut |m| {
                                m.load_indexed(MULTIPLIER, Rsi, R14, "g[j], the row multiplicand");
                                m.xor_clear(LO, "re-seed CF = OF = 0 (back edge clobbered flags)");
                                fsq_mulpre_row(m, Rdi);
                                m.store(Mem::new(Rbp, 0), T[0], "product limb j is final");
                                m.add_imm(Rbp, 8, "next output limb");
                                m.comment("shift down one word");
                                for k in 0..5 {
                                    m.mov(T[k], T[k + 1], &format!("t{k} = t{}", k + 1));
                                }
                                m.xor_clear(T[5], "t5 = 0 (CF/OF stay clear)");
                            });
                            for (k, t) in T[..4].iter().enumerate() {
                                m.store(
                                    Mem::new(Rbp, 8 * k as i32),
                                    *t,
                                    &format!("product limb {}", k + 4),
                                );
                            }
                        },
                    );
                },
            );
            m.comment("Karatsuba assembly: per region d1 -= d0, d1 -= d2, d0 -= d2");
            fsq_walk_setup(
                m,
                Rbp,
                F34K_TB_GSUB,
                F34K_TB_GADD - F34K_TB_GSUB,
                FSQ_WALK_END,
            );
            m.stride_loop(
                Rbp,
                24,
                LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
                ".Lf34k_gsub",
                &mut |m| dbl_gsub_row(m),
            );
            m.comment("lane folds: regions 1 and 2 accumulate into region 0");
            cyc_walk_bound(m, F34K_TB_MOD);
            m.stride_loop(
                Rbp,
                24,
                LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
                ".Lf34k_gadd",
                &mut |m| dbl_gadd_row(m, None),
            );
            m.comment("Montgomery reduction: the coefficient's two lanes into YOUT");
            cyc_walk_bound(m, F34K_TB_MSUB);
            m.stride_loop(
                Rbp,
                16,
                LoopEnd::Mem(Mem::new(rsp, FSQ_WALK_END)),
                ".Lf34k_mod",
                &mut |m| dbl_mod_row(m),
            );
            m.load(R15, Mem::new(rsp, F34K_CJ), "reload the coefficient cursor");
        },
    );

    m.comment("");
    m.comment("copy out: z.a then z.b are contiguous, 48 limbs to z");
    m.load(Rdi, Mem::new(rsp, FSQ_Z), "z");
    m.mov(Rsi, rsp, "");
    m.add_imm(Rsi, F34K_YOUT, "output base");
    m.xor_clear(Rcx, "");
    m.stride_loop(Rcx, 32, LoopEnd::Imm(384), ".Lf34k_out", &mut |m| {
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.load(reg, Mem::new(Rsi, 8 * k as i32), &format!("z limb {k}"));
        }
        for (k, reg) in [R8, R9, R10, R11].into_iter().enumerate() {
            m.store(Mem::new(Rdi, 8 * k as i32), reg, "z");
        }
        m.add_imm(Rsi, 32, "");
        m.add_imm(Rdi, 32, "");
    });
    m.free_stack(F34K_FRAME);
    for reg in CALLEE_SAVED.iter().rev() {
        m.pop(*reg);
    }
    m.ret();
}

/// `narsil_sos_x86`: rolled sum-of-products Montgomery reduction
/// (Longa, ePrint 2022/367, Alg. 2, B = 1) with runtime product count.
///
/// Computes `(sum_{i<T} a_i * b_i) * R^{-1} mod p` for `T` operand pairs
/// (`T` even, production shapes 2/4/6/8) passed as a table of `2T` pointers.
/// every Fp2/Fp6/Fp12 tower product routes through this one loop body, so
/// the whole tower's hot code is a few hundred bytes and stays op-cache
/// resident (the unrolled portable bodies miss L1I on every pairing round).
///
/// # Shape
///
/// Outer counted round: `j` walks the four source limbs (cursor = byte
/// offset `8j`). Inner counted walk: the pair table, one dual-chain product
/// row per pair, two pairs per iteration (the walk stride is 32 bytes, which
/// is also what forces `T` even: the trip proof rejects odd counts), then
/// one Montgomery cancel row and the one-word shift. The back edges'
/// `add`/`cmp` clobber CF/OF, so every row closes both chains and the first
/// row of each iteration re-seeds with a flag-cutting `xor` -- exactly the
/// discipline the dual-chain serialization finding demands.
///
/// # Bounds (operands <= p, T <= 10)
///
/// Between rounds the accumulator is `u_j < (T+1)p < 11p < 2^260`. The
/// in-round peak before the shift stays below `(T+1)*p*2^64 < 2^325`, so a
/// six-word accumulator absorbs every carry and each row's chain closes are
/// provably carry-free at t5 (the interpreter asserts the claims). The final
/// value is `< (1 + 0.1891*T)p < 3p < 2^256`: t4 ends exactly zero and two
/// conditional subtractions reach the canonical range.
pub fn sos_rolled<M: Machine>(m: &mut M) {
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    m.push(Rdi);
    m.comment("table end = pairs + 16T; T arrives in rdx (2 pointers/pair)");
    m.mov(TABLE_END, Rdx, "T");
    for doubled in [2, 4, 8, 16] {
        m.add(TABLE_END, TABLE_END, &format!("{doubled}T"));
    }
    m.add(TABLE_END, Rsi, "pair-table end");
    for (k, t) in T.into_iter().enumerate() {
        m.xor_clear(t, &format!("t{k} = 0"));
    }
    m.xor_clear(JOFF, "byte offset of the round's source limb: 8j = 0");

    m.comment("");
    m.stride_loop(JOFF, 8, LoopEnd::Imm(32), ".Lsos_round", &mut |m| {
        m.comment("product rows: t += a_i[j] * b_i, two pairs per iteration");
        m.mov(CURSOR, Rsi, "rewind the pair-table cursor");
        m.stride_loop(
            CURSOR,
            32,
            LoopEnd::Reg(TABLE_END),
            ".Lsos_pair",
            &mut |m| {
                m.load(Rdi, Mem::new(CURSOR, 0), "a_i pointer");
                m.load_indexed(MULTIPLIER, Rdi, JOFF, "a_i[j], the row multiplicand");
                m.load(Rdi, Mem::new(CURSOR, 8), "b_i pointer");
                m.xor_clear(LO, "re-seed CF = OF = 0 (back edge clobbered flags)");
                sos_row(m, Rdi, "a_i[j]*b_i");
                m.comment("second pair of the iteration");
                m.load(Rdi, Mem::new(CURSOR, 16), "a_{i+1} pointer");
                m.load_indexed(MULTIPLIER, Rdi, JOFF, "a_{i+1}[j]");
                m.load(Rdi, Mem::new(CURSOR, 24), "b_{i+1} pointer");
                m.xor_clear(
                    LO,
                    "flag-cutting re-seed: without it the row serializes on the previous row's closes",
                );
                sos_row(m, Rdi, "a_{i+1}[j]*b_{i+1}");
            },
        );
        m.comment("cancel row: m = t0 * -p^-1, then t += m*p zeroes t0");
        m.mov(MULTIPLIER, T[0], "m multiplicand <- t0");
        m.mulx_mem(
            Rbx,
            MULTIPLIER,
            p_inv(),
            "m = t0 * -p^-1 mod 2^64 (hi half discarded)",
        );
        m.xor_clear(LO, "re-seed CF = OF = 0 (back edge clobbered flags)");
        sos_row(m, Rcx, "m*p");
        m.claim_zero(T[0], "the Montgomery factor cancels the low word");
        m.comment("shift down one word: the canceled zero word drops");
        for k in 0..5 {
            m.mov(T[k], T[k + 1], &format!("t{k} = t{}", k + 1));
        }
        m.xor_clear(T[5], "t5 = 0 (CF/OF stay clear)");
    });

    m.comment("");
    m.claim_zero(T[4], "final value < 3p < 2^256 fits four words");
    m.pop(Rdi);
    m.comment("final reduction: value < 3p, subtract p at most twice");
    let value: [Reg; 4] = [T[0], T[1], T[2], T[3]];
    let keep: [Reg; 4] = [Rdx, Rbx, R14, R15];
    for pass in 0..2 {
        for (k, (v, s)) in value.iter().zip(keep).enumerate() {
            m.mov(s, *v, &format!("pass {pass}: keep-copy of word {k}"));
        }
        for (k, v) in value.iter().enumerate() {
            let what = format!("word {k} -= p{k}");
            if k == 0 {
                m.sub_mem(*v, p_limb(k), &what);
            } else {
                m.sbb_mem(*v, p_limb(k), &what);
            }
        }
        for (k, (v, s)) in value.iter().zip(keep).enumerate() {
            m.cmov_carry(*v, s, &format!("borrow: value < p, keep word {k}"));
        }
    }
    for (k, v) in value.iter().enumerate() {
        m.store(Mem::new(OUT_PTR, 8 * k as i32), *v, &format!("z{k}"));
    }
    for reg in CALLEE_SAVED.iter().rev() {
        m.pop(*reg);
    }
    m.ret();
}

/// Register roles for `narsil_sosd6_x86`.
pub const SOSD6_REGISTER_MAP: &[(Reg, &str)] = &[
    (Rdi, "z on entry (spilled at once); then lane1 word 4"),
    (
        Rsi,
        "stage pointer: walks the 24 transposed x limbs linearly, 8 bytes per pair (also the round cursor)",
    ),
    (
        Rdx,
        "consts pointer on entry (copied to the frame); the implicit mulx multiplicand",
    ),
    (
        Rcx,
        "y pair-block cursor over the five rolled pairs; the shared top word in the round tail",
    ),
    (R8, "lane0 word 0 (prologue: p limb 0 for the negp rows)"),
    (R9, "lane0 word 1 (prologue: p limb 1)"),
    (R10, "lane0 word 2 (prologue: p limb 2)"),
    (R11, "lane0 word 3 (prologue: p limb 3)"),
    (R12, "lane0 word 4 (prologue: negp scratch)"),
    (R13, "lane1 word 0 (prologue: negp scratch)"),
    (R14, "lane1 word 1"),
    (R15, "lane1 word 2"),
    (Rbp, "lane1 word 3"),
    (
        Rax,
        "low half of the current product; zero for chain closes",
    ),
    (Rbx, "high half of the current product; prologue scratch"),
];

/// Lane 0 accumulator words. [`SOSD6_TOP`] is the sixth word.
const SOSD6_L0: [Reg; 5] = [R8, R9, R10, R11, R12];
/// Lane1 accumulator words 0..4.
const SOSD6_L1: [Reg; 5] = [R13, R14, R15, Rbp, Rdi];
/// The shared sixth accumulator word. At T = 6 a lane's word 4 can only
/// overflow on its sixth product row and its cancel row (five rows peak
/// below 2^320), and each lane's overflow dies at its own shift, so one
/// register serves both lanes back to back inside the round tail. Doubles
/// as the y pair-block cursor while the five rolled pairs run.
const SOSD6_TOP: Reg = Rcx;

// sosd6 frame layout, all rsp + disp8. The consts mirror the table shape
// (p at +0, -p^-1 at +32) so cancel rows and reductions address rsp like a
// consts pointer. Ny21 and the y20 copy are the round tail's row sources.
const SOSD6_P: i32 = 0;
const SOSD6_PINV: i32 = 32;
const SOSD6_NY21: i32 = 40;
const SOSD6_Y20: i32 = 72;
const SOSD6_Z_PTR: i32 = 104;
const SOSD6_YB: i32 = 112;
const SOSD6_YEND: i32 = 120;
const SOSD6_FRAME: i32 = 128;

// Stage-block byte offsets (see the `sosd6_x86` ABI): the transposed x limbs
// at +0, then five 64-byte y pair blocks. Blocks 1 and 3 arrive holding
// [y01, y00] and [y11, y10]. The prologue negates their low vectors in
// place. Pair 5's sources (ny21, a y20 copy) live in the frame instead: the
// tail has no free base register for the stage.
const SOSD6_STAGE_Y: i32 = 192;
const SOSD6_STAGE_NY01: i32 = 256;
const SOSD6_STAGE_NY11: i32 = 384;
const SOSD6_STAGE_Y20: i32 = 448;
const SOSD6_STAGE_Y21: i32 = 480;

/// Five-word dual-chain product row: `t[k] += lo_k` on the value chain,
/// `t[k+1] += hi_k` on the carry chain, sources at `[base + off + 8k]`.
/// Valid for each lane's first five rows of a round, where the running
/// value stays below 2^320 and word 4 absorbs both closes. Entry and exit
/// invariant: CF = OF = 0.
fn sosd6_row5<M: Machine>(m: &mut M, t: [Reg; 5], base: Reg, off: i32, product: &str) {
    for k in 0..4 {
        mul_mem_into_columns(
            m,
            Rbx,
            Mem::new(base, off + 8 * k as i32),
            t[k],
            t[k + 1],
            &format!("{product}[{k}]"),
            k,
        );
    }
    m.mov_zero(LO, "zero for the chain closes (flags preserved)");
    m.adox(t[4], LO, "close the value chain into word 4");
    m.claim_flags_clear("value through five rows < (5*2^64 + 7)p < 2^320: word 4 cannot wrap");
}

/// Sixth-row variant: the running value may pass 2^320, so both chain
/// closes ripple into the shared top word. Sources at `[rsp + off]` (the
/// frame-held ny21 or y20 copy). Entry invariant: CF = OF = 0 and
/// [`SOSD6_TOP`] = 0 for the first of the two tail rows, or the other
/// lane's freed zero for the second.
fn sosd6_row6<M: Machine>(m: &mut M, t: [Reg; 5], off: i32, product: &str) {
    for k in 0..4 {
        mul_mem_into_columns(
            m,
            Rbx,
            Mem::new(Reg::Rsp, off + 8 * k as i32),
            t[k],
            t[k + 1],
            &format!("{product}[{k}]"),
            k,
        );
    }
    m.mov_zero(LO, "zero for the chain closes (flags preserved)");
    m.adox(t[4], LO, "close the value chain into word 4");
    m.adox(
        SOSD6_TOP,
        LO,
        "ripple the word-4 close into the shared top word",
    );
    m.adcx(
        SOSD6_TOP,
        LO,
        "close the carry chain into the shared top word",
    );
    m.claim_flags_clear("row-6 peak < 7p + 6p*2^64 < 2^321: the top word cannot wrap");
}

/// Montgomery cancel row over the six-word window `[t, SOSD6_TOP]`, then the
/// one-word shift that drops the canceled zero and frees the top word for
/// the other lane. Register names are loop-iteration-invariant, so the
/// shift is five moves plus a flag-safe xor.
fn sosd6_cancel_shift<M: Machine>(m: &mut M, t: [Reg; 5], lane: &str) {
    m.comment(&format!(
        "{lane} cancel row: m = w0 * -p^-1, then += m*p zeroes w0"
    ));
    m.mov(MULTIPLIER, t[0], "m multiplicand <- w0");
    m.mulx_mem(
        Rbx,
        MULTIPLIER,
        Mem::new(Reg::Rsp, SOSD6_PINV),
        "m = w0 * -p^-1 mod 2^64 (hi half discarded)",
    );
    for k in 0..4 {
        mul_mem_into_columns(
            m,
            Rbx,
            Mem::new(Reg::Rsp, SOSD6_P + 8 * k as i32),
            t[k],
            t[k + 1],
            &format!("m*p{k}"),
            k,
        );
    }
    m.mov_zero(LO, "zero for the chain closes (flags preserved)");
    m.adox(t[4], LO, "close the value chain into word 4");
    m.adox(
        SOSD6_TOP,
        LO,
        "ripple the word-4 close into the shared top word",
    );
    m.adcx(
        SOSD6_TOP,
        LO,
        "close the carry chain into the shared top word",
    );
    m.claim_flags_clear("cancel row closed both chains under the 2^321 bound");
    m.claim_zero(t[0], "the Montgomery factor cancels the low word");
    m.comment(&format!(
        "{lane} shift down one word: the canceled zero drops, the top word empties"
    ));
    for k in 0..4 {
        m.mov(t[k], t[k + 1], &format!("w{k} = w{}", k + 1));
    }
    m.mov(t[4], SOSD6_TOP, "word 4 <- the shared top word");
    m.xor_clear(
        SOSD6_TOP,
        "top word frees for the other lane (CF/OF stay clear)",
    );
}

/// `narsil_sosd6_x86`: dedicated dual-lane sum of products, fixed T = 6 per
/// lane -- both Fp components of `sum_{i<3} x_i * y_i` over Fp2 in one leaf:
///
/// * lane0 = `(sum x_i0*y_i0 + x_i1*(p - y_i1)) * R^-1 mod p`,
/// * lane1 = `(sum x_i0*y_i1 + x_i1*y_i0) * R^-1 mod p`,
///
/// operands at most p, both lanes canonical on return -- exactly the
/// portable `sosd6`, with the three `p - y_i1` images computed in-kernel.
/// This is the tower's hottest dual-lane shape (each composed Fp12 square
/// dispatches it six times, each mul_by_034 six, each Fp6 mul three), and
/// the composed route pays it as two serial `narsil_sos_x86` calls whose
/// ~700-instruction single-lane carry chains cannot overlap.
///
/// # ABI: one caller-staged block
///
/// Arguments: `(z: *mut u64x8 (lane0 then lane1), stage: *mut u64x64,
/// consts: *const { p[4], -p^-1 })` in rdi, rsi, rdx. The stage block is
/// caller-built scratch, 512 bytes:
///
/// * +0..191: the 24 x limbs transposed -- `x_i[j]` at byte `8*(6j + i)`,
///   operand order x00 x01 x10 x11 x20 x21 -- so one pointer (rsi, already
///   the argument register) walks all four rounds' multiplicands linearly.
/// * +192..511: five 64-byte y pair blocks, `[y00, y01] [y01, y00]
///   [y10, y11] [y11, y10] [y20, y21]`: block i holds pair i's lane0 row
///   source, then its lane1 row source. The kernel overwrites the low
///   vectors of blocks 1 and 3 with `p - y01`, `p - y11` in place (`stage`
///   is therefore `*mut`). `p - y21` and the pair-5 y20 copy go to the
///   kernel frame instead.
///
/// Twelve SysV pointer arguments do not exist, so the ABI choice is between
/// a 12-pointer table (the `narsil_sos_x86` shape) and this staged block.
/// The block wins on total overhead: a table costs the kernel twelve
/// pointer loads plus 48 double-indirected limb loads and ~60 stores of
/// prologue staging on every call (the rounds must read y via rsp and the
/// multiplicands via one linear pointer regardless -- fifteen registers are
/// spoken for, see below), while the caller builds the block with plain
/// vector copies the compiler schedules freely, replacing the pointer-table
/// stores plus three `negp` temporaries the composed route already paid.
///
/// # Register budget and spill plan
///
/// Two six-word lane accumulators would take twelve registers, and with
/// rdx/rax/rbx there would be none left for any cursor. The T = 6 bound
/// rescues one word: from a round-boundary value below 7p, five product
/// rows peak below `(5*2^64 + 7)p < 2^320`, so each lane's first five rows
/// close inside a five-word window, and only the sixth row and the cancel
/// row can spill into a sixth word. Those run in the round tail, one lane
/// at a time, so a single shared top word (rcx) serves both lanes -- eleven
/// accumulator registers, and rcx doubles as the y pair-block cursor while
/// the five rolled pairs run. Rsi walks the transposed x limbs (multiplicand
/// pointer and round cursor at once: five inner steps of 8 plus the back
/// edge's 8 = 48 bytes per round, and the y blocks begin exactly where the
/// x limbs end, so one frame slot is both the outer loop bound and the y
/// rewind value). No accumulator word ever touches memory. The frame holds
/// only the consts copy, the two pair-5 row sources, z, and the two walk
/// bounds.
///
/// # Shape and interleaving
///
/// Per source limb j, the five rolled pairs each load one multiplicand
/// `x_i[j]` and run a lane0 row then a lane1 row against the pair block --
/// the sosd2 alternation at T = 6, so both lanes' dual carry chains stay in
/// flight across the whole round body. The tail keeps the alternation as
/// far as the shared top word allows: lane0's sixth row, cancel and shift
/// (freeing the top word at zero), then lane1's sixth row, cancel and
/// shift. The tail's lane1 rows depend on the top word only through the
/// flag-cutting xor that frees it, so out-of-order execution overlaps them
/// with lane0's cancel. Program order alone is serial there.
///
/// # Bounds (operands <= p, T = 6)
///
/// Exactly the portable sosd6 bounds: between rounds each lane holds
/// `u < 7p < 2^260`. Five product rows peak below 2^320 (five-word rows),
/// the sixth row below `7p + 6p*2^64 < 2^321` and the cancel row below
/// `7p(1 + 2^64) < 2^321`. The interpreter checks each close. After the
/// shift the value is again below 7p, so the top word is
/// zero and hands over cleanly. The final value is `< (1 + 0.1891*6)p <
/// 2.135p < 2^256`: word 4 ends exactly zero and two conditional
/// subtractions per lane reach the canonical range.
pub fn sosd6_x86<M: Machine>(m: &mut M) {
    let rsp = Reg::Rsp;
    for reg in CALLEE_SAVED {
        m.push(reg);
    }
    m.alloc_stack(SOSD6_FRAME);
    m.comment("frame: p +0, -p^-1 +32, ny21 +40, y20 copy +72, z +104, y base +112, y end +120");
    m.store(Mem::new(rsp, SOSD6_Z_PTR), Rdi, "spill z");
    m.comment("consts into the frame: cancel rows and reductions address rsp as a table");
    for (k, reg) in A.iter().enumerate() {
        m.load(
            *reg,
            Mem::new(Rdx, 8 * k as i32),
            &format!("p{k} (kept live for the negp rows)"),
        );
    }
    for (k, reg) in A.iter().enumerate() {
        m.store(
            Mem::new(rsp, SOSD6_P + 8 * k as i32),
            *reg,
            &format!("p{k}"),
        );
    }
    m.load(Rax, Mem::new(Rdx, 32), "-p^-1");
    m.store(Mem::new(rsp, SOSD6_PINV), Rax, "-p^-1");

    let scratch = [Rax, Rbx, R12, R13];
    for (name, src, dst_base, dst) in [
        ("ny01", SOSD6_STAGE_NY01, Rsi, SOSD6_STAGE_NY01),
        ("ny11", SOSD6_STAGE_NY11, Rsi, SOSD6_STAGE_NY11),
        ("ny21", SOSD6_STAGE_Y21, rsp, SOSD6_NY21),
    ] {
        m.comment(&format!(
            "{name} = p - {}: lane0's subtracted term enters as the negp image",
            &name[1..]
        ));
        for (k, s) in scratch.into_iter().enumerate() {
            m.mov(s, A[k], &format!("p{k}"));
        }
        for (k, s) in scratch.into_iter().enumerate() {
            let what = format!("p{k} - {}[{k}]", &name[1..]);
            if k == 0 {
                m.sub_mem(s, Mem::new(Rsi, src), &what);
            } else {
                m.sbb_mem(s, Mem::new(Rsi, src + 8 * k as i32), &what);
            }
        }
        for (k, s) in scratch.into_iter().enumerate() {
            m.store(
                Mem::new(dst_base, dst + 8 * k as i32),
                s,
                &format!("{name}[{k}]"),
            );
        }
    }
    m.comment("copy y20 beside ny21: the tail's lane1 row reads pair 5 off rsp");
    for (k, s) in scratch.into_iter().enumerate() {
        m.load(
            s,
            Mem::new(Rsi, SOSD6_STAGE_Y20 + 8 * k as i32),
            &format!("y20[{k}]"),
        );
    }
    for (k, s) in scratch.into_iter().enumerate() {
        m.store(
            Mem::new(rsp, SOSD6_Y20 + 8 * k as i32),
            s,
            &format!("y20[{k}]"),
        );
    }
    m.mov(Rax, Rsi, "");
    m.add_imm(
        Rax,
        SOSD6_STAGE_Y,
        "y pair blocks start where the x limbs end",
    );
    m.store(
        Mem::new(rsp, SOSD6_YB),
        Rax,
        "y rewind value = outer loop bound",
    );
    m.add_imm(Rax, 320, "past the five rolled pair blocks");
    m.store(Mem::new(rsp, SOSD6_YEND), Rax, "inner walk bound");
    m.comment("both lanes start at zero (the first round adds into zeros)");
    for (k, t) in SOSD6_L0.into_iter().enumerate() {
        m.xor_clear(t, &format!("lane0 w{k} = 0"));
    }
    for (k, u) in SOSD6_L1.into_iter().enumerate() {
        m.xor_clear(u, &format!("lane1 w{k} = 0"));
    }

    m.comment("");
    m.comment("rounds: rsi walks the transposed x limbs, 48 bytes per round");
    m.stride_loop(
        Rsi,
        8,
        LoopEnd::Mem(Mem::new(rsp, SOSD6_YB)),
        ".Lsosd6_round",
        &mut |m| {
            m.load(SOSD6_TOP, Mem::new(rsp, SOSD6_YB), "y pair-block cursor rewinds");
            m.comment("five rolled pairs: adjacent lane rows share each multiplicand");
            m.stride_loop(
                SOSD6_TOP,
                64,
                LoopEnd::Mem(Mem::new(rsp, SOSD6_YEND)),
                ".Lsosd6_pair",
                &mut |m| {
                    m.load(MULTIPLIER, Mem::new(Rsi, 0), "x_i[j], the pair's multiplicand");
                    m.xor_clear(LO, "re-seed CF = OF = 0 (back edge clobbered flags)");
                    sosd6_row5(m, SOSD6_L0, SOSD6_TOP, 0, "x_i[j]*row0");
                    sosd6_row5(m, SOSD6_L1, SOSD6_TOP, 32, "x_i[j]*row1");
                    m.add_imm(Rsi, 8, "next x limb (clobbers CF/OF; chains are closed)");
                },
            );
            m.comment("pair 5: the only rows that can overflow word 4; the top word serves one lane at a time");
            m.xor_clear(
                SOSD6_TOP,
                "top word = 0; re-seeds CF = OF = 0 after the back edge",
            );
            m.load(MULTIPLIER, Mem::new(Rsi, 0), "x21[j]");
            sosd6_row6(m, SOSD6_L0, SOSD6_NY21, "x21[j]*ny21");
            sosd6_cancel_shift(m, SOSD6_L0, "lane0");
            m.load(MULTIPLIER, Mem::new(Rsi, 0), "x21[j] again (the cancel row owned rdx)");
            sosd6_row6(m, SOSD6_L1, SOSD6_Y20, "x21[j]*y20");
            sosd6_cancel_shift(m, SOSD6_L1, "lane1");
        },
    );

    m.comment("");
    m.claim_zero(
        SOSD6_L0[4],
        "lane0 final value < 2.135p < 2^256 fits four words",
    );
    m.claim_zero(
        SOSD6_L1[4],
        "lane1 final value < 2.135p < 2^256 fits four words",
    );
    m.load(
        Rdi,
        Mem::new(rsp, SOSD6_Z_PTR),
        "reload z into lane1's freed word 4",
    );
    m.comment("final reduction per lane: value < 2.135p, subtract p at most twice");
    let keep = [Rax, Rbx, Rcx, MULTIPLIER];
    let lanes: [([Reg; 4], i32); 2] = [([R8, R9, R10, R11], 0), ([R13, R14, R15, Rbp], 32)];
    for (lane, (value, out_off)) in lanes.into_iter().enumerate() {
        for pass in 0..2 {
            for (k, (v, s)) in value.iter().zip(keep).enumerate() {
                m.mov(
                    s,
                    *v,
                    &format!("lane{lane} pass {pass}: keep-copy of word {k}"),
                );
            }
            for (k, v) in value.iter().enumerate() {
                let what = format!("lane{lane}: word {k} -= p{k}");
                if k == 0 {
                    m.sub_mem(*v, Mem::new(rsp, SOSD6_P), &what);
                } else {
                    m.sbb_mem(*v, Mem::new(rsp, SOSD6_P + 8 * k as i32), &what);
                }
            }
            for (k, (v, s)) in value.iter().zip(keep).enumerate() {
                m.cmov_carry(*v, s, &format!("borrow: lane{lane} < p, keep word {k}"));
            }
        }
        for (k, v) in value.iter().enumerate() {
            m.store(
                Mem::new(Rdi, out_off + 8 * k as i32),
                *v,
                &format!("z[{}]", lane * 4 + k),
            );
        }
    }
    m.free_stack(SOSD6_FRAME);
    for reg in CALLEE_SAVED.iter().rev() {
        m.pop(*reg);
    }
    m.ret();
}
