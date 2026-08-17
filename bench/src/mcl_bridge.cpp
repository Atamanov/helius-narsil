#include <mcl/bn.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

using mcl::bn::Fp;
using mcl::bn::Fp2;
using mcl::bn::Fp6;
using mcl::bn::Fp12;
using mcl::bn::Fr;
using mcl::bn::G1;
using mcl::bn::G2;

// Fp12 is twelve 4x64 limbs only in the 256-bit build. A wider MCL would fold
// different words than the Rust lanes and every cross-lane digest would drift.
static_assert(sizeof(Fp12) == 48 * sizeof(std::uint64_t),
              "the bridge requires MCL's 256-bit Fp/Fr build");

namespace {

struct Groth16Context {
    Fp12 alpha_beta;
    G2 gamma_g2;
    G2 delta_g2;
    // Fixed verifier-key G2 line schedules. The other lanes replay prepared
    // schedules, so the timed verify must not rebuild them.
    std::vector<Fp6> gamma_coeff;
    std::vector<Fp6> delta_coeff;
    std::vector<G1> gamma_abc_g1;
    G1 proof_a;
    G2 proof_b;
    G1 proof_c;
    std::vector<Fr> public_inputs;
};

// One verifying key of a multi-proof batch. A batch may span several keys,
// which is what a validator sees when one block carries several circuits.
struct Groth16BatchKey {
    G1 alpha;
    G1 k0;
    G1 k1;
    // The three key-owned G2 line schedules, built when the context is
    // created. A verifying key is fixed, so the timed verify replays them.
    // Each member's B is the one live G2 of a batch.
    std::vector<Fp6> beta_coeff;
    std::vector<Fp6> gamma_coeff;
    std::vector<Fp6> delta_coeff;
};

struct Groth16BatchMember {
    G1 a;
    G1 c;
    G2 b;
    Fr x;
    Fr r;
    std::size_t key;
};

struct Groth16BatchContext {
    std::vector<Groth16BatchKey> keys;
    std::vector<Groth16BatchMember> members;
};

struct GnarkCommittedGroth16Context {
    Fp12 alpha_beta;
    G2 gamma_neg;
    G2 delta_neg;
    G1 k[3];
    G2 commitment_key_g;
    G2 commitment_key_g_sigma_neg;
    // The four fixed verifying-key G2 line schedules. The other lanes replay
    // prepared schedules, so the timed verify must not rebuild them. proof_b
    // is the one live G2 in the timed loop.
    std::vector<Fp6> gamma_neg_coeff;
    std::vector<Fp6> delta_neg_coeff;
    std::vector<Fp6> commitment_key_g_coeff;
    std::vector<Fp6> commitment_key_g_sigma_neg_coeff;
    G1 proof_a;
    G2 proof_b;
    G1 proof_c;
    G1 commitment;
    G1 commitment_pok;
    Fr public_input;
};

struct MillerContext {
    G1 g1;
    G2 g2;
    std::vector<Fp6> g2_precomputed;
    Fp12 output;
    Fp12 prepared_output;
};

struct DirectContext {
    Fp fp[2];
    Fp2 fp2[2];
    Fp12 fp12;
    Fp2 sparse[3];
    Fp scale;
    Fp12 fp12b;
    Fp12 fp12cyclo;
};

struct MsmContext {
    std::vector<G1> bases;
    std::vector<Fr> scalars;
};

volatile std::uint64_t operation_sink = 0;

constexpr std::uint8_t FP_MODULUS[32] = {
    0x30, 0x64, 0x4e, 0x72, 0xe1, 0x31, 0xa0, 0x29,
    0xb8, 0x50, 0x45, 0xb6, 0x81, 0x81, 0x58, 0x5d,
    0x97, 0x81, 0x6a, 0x91, 0x68, 0x71, 0xca, 0x8d,
    0x3c, 0x20, 0x8c, 0x16, 0xd8, 0x7c, 0xfd, 0x47,
};

constexpr std::uint8_t FR_MODULUS[32] = {
    0x30, 0x64, 0x4e, 0x72, 0xe1, 0x31, 0xa0, 0x29,
    0xb8, 0x50, 0x45, 0xb6, 0x81, 0x81, 0x58, 0x5d,
    0x28, 0x33, 0xe8, 0x48, 0x79, 0xb9, 0x70, 0x91,
    0x43, 0xe1, 0xf5, 0x93, 0xf0, 0x00, 0x00, 0x01,
};

bool canonical(const std::uint8_t* input, const std::uint8_t* modulus) {
    return std::lexicographical_compare(input, input + 32, modulus, modulus + 32);
}

bool all_zero(const std::uint8_t* input, std::size_t size) {
    std::uint8_t aggregate = 0;
    for (std::size_t i = 0; i < size; ++i) aggregate |= input[i];
    return aggregate == 0;
}

bool decode_fp(Fp& output, const std::uint8_t* input) {
    if (!canonical(input, FP_MODULUS)) return false;
    bool ok = false;
    output.setBigEndianMod(&ok, input, 32);
    return ok;
}

bool decode_fr(Fr& output, const std::uint8_t* input) {
    if (!canonical(input, FR_MODULUS)) return false;
    bool ok = false;
    output.setBigEndianMod(&ok, input, 32);
    return ok;
}

bool encode_fp(std::uint8_t* output, const Fp& input) {
    std::uint8_t little[32] = {};
    const std::size_t size = input.getLittleEndian(little, sizeof(little));
    if (size == 0 || size > sizeof(little)) return false;
    std::memset(output, 0, 32);
    for (std::size_t i = 0; i < size; ++i) output[31 - i] = little[i];
    return true;
}

bool decode_g1(G1& output, const std::uint8_t* input) {
    if (all_zero(input, 64)) {
        output.clear();
        return true;
    }
    Fp x, y;
    if (!decode_fp(x, input) || !decode_fp(y, input + 32)) return false;
    output.x = x;
    output.y = y;
    output.z = 1;
    return output.isValid();
}

bool decode_g2(G2& output, const std::uint8_t* input) {
    if (all_zero(input, 128)) {
        output.clear();
        return true;
    }
    Fp x1, x0, y1, y0;
    if (!decode_fp(x1, input) || !decode_fp(x0, input + 32) ||
        !decode_fp(y1, input + 64) || !decode_fp(y0, input + 96)) {
        return false;
    }
    output.x = Fp2(x0, x1);
    output.y = Fp2(y0, y1);
    output.z = 1;
    // verifyOrderG2(true) makes isValid perform both curve and subgroup checks.
    return output.isValid();
}

bool encode_g1(std::uint8_t* output, G1 input) {
    if (input.isZero()) {
        std::memset(output, 0, 64);
        return true;
    }
    input.normalize();
    return encode_fp(output, input.x) && encode_fp(output + 32, input.y);
}

// The two folds below mirror the Rust side of the four-lane protocol bit for
// bit. Word folds run over raw Montgomery limbs, which all four lanes share
// (4x64 little-endian limbs, R = 2^256), so equal values fold equally.
inline std::uint64_t rotl64(std::uint64_t value, unsigned by) {
    return (value << by) | (value >> (64 - by));
}

// One digest chain step, bit for bit the Rust fold_word. The odd multiplier
// keeps every step bijective, so repeated inputs never cancel.
const std::uint64_t FOLD_MULTIPLIER = 0x9e3779b97f4a7c15ULL;

inline std::uint64_t fold_step(std::uint64_t acc, std::uint64_t word) {
    return (rotl64(acc, 7) ^ word) * FOLD_MULTIPLIER;
}

template<class T>
std::uint64_t fold_limbs(std::uint64_t acc, const T& value) {
    static_assert(sizeof(T) % 8 == 0, "limb folding requires whole words");
    std::uint64_t words[sizeof(T) / 8];
    std::memcpy(words, &value, sizeof(T));
    for (std::uint64_t word : words) acc = fold_step(acc, word);
    return acc;
}

bool parse_direct(DirectContext& context, const std::uint8_t* blob) {
    const std::uint8_t* cursor = blob;
    for (Fp& value : context.fp) {
        if (!decode_fp(value, cursor)) return false;
        cursor += 32;
    }
    for (Fp2& value : context.fp2) {
        if (!decode_fp(value.a, cursor) || !decode_fp(value.b, cursor + 32)) return false;
        cursor += 64;
    }
    Fp* fp12 = context.fp12.getFp0();
    for (std::size_t i = 0; i < 12; ++i) {
        if (!decode_fp(fp12[i], cursor)) return false;
        cursor += 32;
    }
    for (Fp2& value : context.sparse) {
        if (!decode_fp(value.a, cursor) || !decode_fp(value.b, cursor + 32)) return false;
        cursor += 64;
    }
    if (!decode_fp(context.scale, cursor)) return false;
    return true;
}

// `result` receives the online target-group product every lane digests, the
// value the equation tests against e(alpha, beta).
bool groth16_verify_prepared(Groth16Context& context, Fp12& result, G1& accumulator) {
    G1 public_input_accumulator = context.gamma_abc_g1[0];
    if (!context.public_inputs.empty()) {
        G1 variable_part;
        G1::mulVec(variable_part,
                   context.gamma_abc_g1.data() + 1,
                   context.public_inputs.data(),
                   context.public_inputs.size());
        G1::add(public_input_accumulator, public_input_accumulator, variable_part);
    }
    accumulator = public_input_accumulator;
    G1 neg_input, neg_c;
    G1::neg(neg_input, public_input_accumulator);
    G1::neg(neg_c, context.proof_c);
    // mcl's widest shared-accumulator entry points cover two pairs, so the
    // three-pair product runs as one mixed loop plus one prepared loop.
    Fp12 online, rest;
    mcl::bn::precomputedMillerLoop2mixed(online,
                                         context.proof_a, context.proof_b,
                                         neg_input, context.gamma_coeff.data());
    mcl::bn::precomputedMillerLoop(rest, neg_c, context.delta_coeff.data());
    online *= rest;
    mcl::bn::finalExp(result, online);
    return result == context.alpha_beta;
}

// Random-linear-combination batch verification of several Groth16 proofs.
//
// Every scalar multiplication is inside this call, so the row times a batch
// verification and not a replay of precomputed points. Per-key aggregation is
// what makes a same-key batch cheaper than a mixed one: gamma, delta, and
// alpha-beta collapse to one pair each per key, not per proof.
//
// Under one key `sum_i r_i (k0 + x_i k1)` is `(sum_i r_i) k0 + (sum_i r_i x_i)
// k1`, so the public inputs of a whole key cost two weighted points, not two
// per member. Every remaining weighting that shares a G2 partner goes through
// one mulVec. What stays per member is `r_i A_i`, which pairs against that
// member's own B.
//
// `fold` receives each key's public-input accumulator in key order and then
// the Miller product, which together are the value-bearing part of the digest.
// The combined GT of an accepting batch is one whatever the batch held, so the
// Miller product is what ties the digest to the pairs that were run.
// `result` receives the combined GT.
bool groth16_batch_verify(const Groth16BatchContext& context,
                          Fp12& result,
                          std::uint64_t& fold) {
    const std::size_t members = context.members.size();
    const std::size_t keys = context.keys.size();
    std::vector<G1> g1(members);
    std::vector<G2> g2(members);
    std::vector<Fr> r_sum(keys), rx_sum(keys);
    std::vector<std::vector<G1>> c_bases(keys);
    std::vector<std::vector<Fr>> c_scalars(keys);
    for (std::size_t j = 0; j < keys; ++j) {
        r_sum[j].clear();
        rx_sum[j].clear();
    }
    for (std::size_t i = 0; i < members; ++i) {
        const Groth16BatchMember& member = context.members[i];
        Fr::add(r_sum[member.key], r_sum[member.key], member.r);
        Fr weighted_x;
        Fr::mul(weighted_x, member.r, member.x);
        Fr::add(rx_sum[member.key], rx_sum[member.key], weighted_x);
        c_bases[member.key].push_back(member.c);
        c_scalars[member.key].push_back(member.r);
        G1::mul(g1[i], member.a, member.r);
        g2[i] = member.b;
    }
    std::vector<G1> prepared_g1;
    std::vector<const Fp6*> prepared_coeff;
    prepared_g1.reserve(3 * keys);
    prepared_coeff.reserve(3 * keys);
    G1 total;
    total.clear();
    for (std::size_t j = 0; j < keys; ++j) {
        const Groth16BatchKey& key = context.keys[j];
        G1 k_bases[2] = {key.k0, key.k1};
        const Fr k_scalars[2] = {r_sum[j], rx_sum[j]};
        G1 input_sum;
        G1::mulVec(input_sum, k_bases, k_scalars, 2);
        G1::add(total, total, input_sum);
        G1 neg_input, c_sum, neg_c, weighted_alpha, neg_alpha;
        G1::neg(neg_input, input_sum);
        G1::mulVec(c_sum, c_bases[j].data(), c_scalars[j].data(), c_bases[j].size());
        G1::neg(neg_c, c_sum);
        G1::mul(weighted_alpha, key.alpha, r_sum[j]);
        G1::neg(neg_alpha, weighted_alpha);
        prepared_g1.push_back(neg_input);
        prepared_coeff.push_back(key.gamma_coeff.data());
        prepared_g1.push_back(neg_c);
        prepared_coeff.push_back(key.delta_coeff.data());
        prepared_g1.push_back(neg_alpha);
        prepared_coeff.push_back(key.beta_coeff.data());
    }
    // mcl's widest prepared entry point covers two pairs, and each group pays
    // its own squaring chain, so the schedules replay two at a time whatever
    // key they belong to. A Miller product over a set is the product over any
    // partition of it, so the grouping cannot move the result.
    Fp12 miller;
    mcl::bn::millerLoopVec(miller, g1.data(), g2.data(), g1.size());
    std::size_t at = 0;
    for (; at + 1 < prepared_g1.size(); at += 2) {
        Fp12 pair;
        mcl::bn::precomputedMillerLoop2(pair,
                                        prepared_g1[at], prepared_coeff[at],
                                        prepared_g1[at + 1], prepared_coeff[at + 1]);
        miller *= pair;
    }
    if (at < prepared_g1.size()) {
        Fp12 last;
        mcl::bn::precomputedMillerLoop(last, prepared_g1[at], prepared_coeff[at]);
        miller *= last;
    }
    if (!total.isZero()) {
        total.normalize();
        fold = fold_limbs(fold, total.x);
        fold = fold_limbs(fold, total.y);
    }
    fold = fold_limbs(fold, miller);
    mcl::bn::finalExp(result, miller);
    return result.isOne();
}

// `result` receives the last computed GT, the proof-of-knowledge product on
// rejection and the main-equation product otherwise, like the Rust lanes.
bool gnark_committed_verify_prehashed(GnarkCommittedGroth16Context& context,
                                      const Fr& hash, Fp12& result, G1& accumulator) {
    G1 variable_part;
    const Fr scalars[2] = {context.public_input, hash};
    G1::mulVec(variable_part, context.k + 1, scalars, 2);
    G1 k_sum;
    G1::add(k_sum, context.k[0], variable_part);
    G1::add(k_sum, k_sum, context.commitment);
    accumulator = k_sum;

    Fp12 miller;
    mcl::bn::precomputedMillerLoop2(miller,
                                    context.commitment,
                                    context.commitment_key_g_sigma_neg_coeff.data(),
                                    context.commitment_pok,
                                    context.commitment_key_g_coeff.data());
    mcl::bn::finalExp(result, miller);
    if (!result.isOne()) return false;

    // mcl's widest shared-accumulator entry points cover two pairs, so the
    // three-pair product runs as one mixed loop (live proof_b plus prepared
    // delta) plus one prepared loop and one Fp12 mul.
    Fp12 rest;
    mcl::bn::precomputedMillerLoop2mixed(miller,
                                         context.proof_a, context.proof_b,
                                         context.proof_c,
                                         context.delta_neg_coeff.data());
    mcl::bn::precomputedMillerLoop(rest, k_sum, context.gamma_neg_coeff.data());
    miller *= rest;
    mcl::bn::finalExp(result, miller);
    return result == context.alpha_beta;
}

}  // namespace

extern "C" int mcl_init() {
    try {
        mcl::bn::initPairing(mcl::BN_SNARK1);
        // BN254 G1 has cofactor one; the syscall validates on-curve but performs no
        // redundant scalar-order multiplication. G2 order remains mandatory.
        mcl::bn::verifyOrderG1(false);
        mcl::bn::verifyOrderG2(true);
        // NDEBUG removes assert, so the `Fp6 lines[128]` stack schedules used
        // by the operation probes need this explicit bound.
        if (mcl::bn::getPrecomputedQcoeffSize() > 128) return -2;
        return 0;
    } catch (...) {
        return -1;
    }
}

// First isEnableJIT call is not thread safe. The Rust side probes once right
// after init, before any worker threads exist.
extern "C" int mcl_runtime_jit() {
    try {
        return mcl::fp::isEnableJIT() ? 1 : 0;
    } catch (...) {
        return -1;
    }
}

extern "C" int mcl_g1_create(const std::uint8_t* encoded, void** output) {
    if (encoded == nullptr || output == nullptr) return -1;
    *output = nullptr;
    try {
        std::unique_ptr<G1> point(new G1);
        if (!decode_g1(*point, encoded)) {
            return -2;
        }
        *output = point.release();
        return 0;
    } catch (...) {
        return -3;
    }
}

extern "C" void mcl_g1_destroy(void* handle) {
    delete static_cast<G1*>(handle);
}

extern "C" int mcl_g2_create(const std::uint8_t* encoded, void** output) {
    if (encoded == nullptr || output == nullptr) return -1;
    *output = nullptr;
    try {
        std::unique_ptr<G2> point(new G2);
        if (!decode_g2(*point, encoded)) {
            return -2;
        }
        *output = point.release();
        return 0;
    } catch (...) {
        return -3;
    }
}

extern "C" void mcl_g2_destroy(void* handle) {
    delete static_cast<G2*>(handle);
}

extern "C" int mcl_g2_is_valid_order(const void* handle, std::uint8_t* output) {
    if (handle == nullptr || output == nullptr) return -1;
    try {
        *output = static_cast<const G2*>(handle)->isValidOrder() ? 1 : 0;
        return 0;
    } catch (...) {
        return -2;
    }
}

// Four-lane direct context. The blob appends one full Fp12 multiplicand and
// one cyclotomic-subgroup Fp12 (12 coefficients each, 384 bytes each) after
// the 800-byte direct layout.
extern "C" int mcl_direct_create2(const std::uint8_t* blob, void** output) {
    if (blob == nullptr || output == nullptr) return -1;
    *output = nullptr;
    try {
        std::unique_ptr<DirectContext> context(new DirectContext);
        if (!parse_direct(*context, blob)) return -2;
        const std::uint8_t* cursor = blob + 800;
        Fp* fp12b = context->fp12b.getFp0();
        for (std::size_t i = 0; i < 12; ++i) {
            if (!decode_fp(fp12b[i], cursor)) return -2;
            cursor += 32;
        }
        Fp* fp12cyclo = context->fp12cyclo.getFp0();
        for (std::size_t i = 0; i < 12; ++i) {
            if (!decode_fp(fp12cyclo[i], cursor)) return -2;
            cursor += 32;
        }
        *output = context.release();
        return 0;
    } catch (...) {
        return -3;
    }
}

extern "C" void mcl_direct_destroy(void* handle) {
    delete static_cast<DirectContext*>(handle);
}

extern "C" int mcl_direct_check(void* handle,
                                        std::uint32_t operation,
                                        std::uint8_t* output) {
    if (handle == nullptr || output == nullptr) return -1;
    try {
        DirectContext& c = *static_cast<DirectContext*>(handle);
        if (operation == 0 || operation == 1) {
            Fp value;
            if (operation == 0) Fp::mul(value, c.fp[0], c.fp[1]);
            else Fp::sqr(value, c.fp[0]);
            return encode_fp(output, value) ? 0 : -2;
        }
        if (operation == 4 || operation == 5 || operation == 14) {
            Fp2 value;
            if (operation == 4) Fp2::mul(value, c.fp2[0], c.fp2[1]);
            else if (operation == 5) Fp2::sqr(value, c.fp2[0]);
            else Fp2::mulFp(value, c.fp2[0], c.scale);
            return encode_fp(output, value.a) && encode_fp(output + 32, value.b) ? 0 : -2;
        }
        if (operation == 15 || operation == 16 || operation == 17) {
            Fp12 value;
            if (operation == 15) {
                Fp12::sqr(value, c.fp12);
            } else if (operation == 17) {
                Fp12::mul(value, c.fp12, c.fp12b);
            } else {
                value = c.fp12;
                Fp6 line;
                line.a = c.sparse[2];
                line.b = c.sparse[0];
                line.c = c.sparse[1];
                mcl::bn::mulSparseLine(value, line);
            }
            const Fp* coefficients = value.getFp0();
            for (std::size_t i = 0; i < 12; ++i) {
                if (!encode_fp(output + i * 32, coefficients[i])) return -2;
            }
            return 0;
        }
        return -3;
    } catch (...) {
        return -4;
    }
}

extern "C" int mcl_msm_create(const std::uint8_t* points,
                                      const std::uint8_t* scalars,
                                      std::size_t count,
                                      void** output) {
    if (points == nullptr || scalars == nullptr || output == nullptr) return -1;
    *output = nullptr;
    try {
        if (count == 0 || count > 2048) return -2;
        std::unique_ptr<MsmContext> context(new MsmContext);
        context->bases.resize(count);
        context->scalars.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            if (!decode_g1(context->bases[i], points + i * 64)) return -3;
            if (!decode_fr(context->scalars[i], scalars + i * 32)) return -3;
        }
        *output = context.release();
        return 0;
    } catch (...) {
        return -4;
    }
}

extern "C" void mcl_msm_destroy(void* handle) {
    delete static_cast<MsmContext*>(handle);
}

extern "C" int mcl_msm_check(void* handle, std::uint8_t* output) {
    if (handle == nullptr || output == nullptr) return -1;
    try {
        MsmContext& context = *static_cast<MsmContext*>(handle);
        G1 result;
        G1::mulVec(result, context.bases.data(), context.scalars.data(), context.bases.size());
        return encode_g1(output, result) ? 0 : -2;
    } catch (...) {
        return -3;
    }
}

extern "C" int mcl_miller_create(const std::uint8_t* g1,
                                         const std::uint8_t* g2,
                                         void** output) {
    if (g1 == nullptr || g2 == nullptr || output == nullptr) return -1;
    *output = nullptr;
    try {
        std::unique_ptr<MillerContext> context(new MillerContext);
        if (!decode_g1(context->g1, g1) || !decode_g2(context->g2, g2)) return -2;
        context->g2_precomputed.resize(mcl::bn::getPrecomputedQcoeffSize());
        mcl::bn::precomputeG2(context->g2_precomputed.data(), context->g2);
        mcl::bn::millerLoop(context->output, context->g1, context->g2);
        *output = context.release();
        return 0;
    } catch (...) {
        return -3;
    }
}

extern "C" void mcl_miller_destroy(void* handle) {
    delete static_cast<MillerContext*>(handle);
}

extern "C" int mcl_miller_raw(void* handle, std::uint8_t* output) {
    if (handle == nullptr || output == nullptr) return -1;
    try {
        MillerContext& context = *static_cast<MillerContext*>(handle);
        mcl::bn::millerLoop(context.output, context.g1, context.g2);
        const Fp* coefficients = context.output.getFp0();
        for (std::size_t i = 0; i < 12; ++i) {
            if (!encode_fp(output + i * 32, coefficients[i])) return -2;
        }
        return 0;
    } catch (...) {
        return -3;
    }
}

extern "C" int mcl_miller_final(void* handle, std::uint8_t* output) {
    if (handle == nullptr || output == nullptr) return -1;
    try {
        MillerContext& context = *static_cast<MillerContext*>(handle);
        mcl::bn::millerLoop(context.output, context.g1, context.g2);
        Fp12 result;
        mcl::bn::finalExp(result, context.output);
        const Fp* coefficients = result.getFp0();
        for (std::size_t i = 0; i < 12; ++i) {
            if (!encode_fp(output + i * 32, coefficients[i])) return -2;
        }
        return 0;
    } catch (...) {
        return -3;
    }
}

extern "C" int mcl_miller_prepared_final(void* handle, std::uint8_t* output) {
    if (handle == nullptr || output == nullptr) return -1;
    try {
        MillerContext& context = *static_cast<MillerContext*>(handle);
        mcl::bn::precomputedMillerLoop(
            context.prepared_output,
            context.g1,
            context.g2_precomputed.data());
        Fp12 result;
        mcl::bn::finalExp(result, context.prepared_output);
        const Fp* coefficients = result.getFp0();
        for (std::size_t i = 0; i < 12; ++i) {
            if (!encode_fp(output + i * 32, coefficients[i])) return -2;
        }
        return 0;
    } catch (...) {
        return -3;
    }
}

// Prepared-pair shape probe.
//
// mcl's widest prepared Miller entry covers two pairs, so a product over three
// prepared pairs runs as one two-pair loop plus one one-pair loop and one Fp12
// multiply. The extra one-pair loop pays a second Miller squaring chain that a
// single wider accumulator would not pay. `pairs` selects the shape and the
// caller times both, which is what turns that structural claim into a number.
extern "C" int mcl_prepared_shape_run(void* handle,
                                            std::size_t pairs,
                                            std::size_t iterations,
                                            std::uint64_t* digest) {
    if (handle == nullptr || digest == nullptr) return -1;
    if (pairs != 1 && pairs != 2) return -2;
    try {
        MillerContext& context = *static_cast<MillerContext*>(handle);
        const Fp6* coefficients = context.g2_precomputed.data();
        std::uint64_t acc = *digest;
        Fp12 value;
        for (std::size_t i = 0; i < iterations; ++i) {
            if (pairs == 1) {
                mcl::bn::precomputedMillerLoop(value, context.g1, coefficients);
            } else {
                mcl::bn::precomputedMillerLoop2(value,
                                                context.g1, coefficients,
                                                context.g1, coefficients);
            }
            acc = fold_limbs(acc, value);
        }
        *digest = acc;
        operation_sink = acc;
        return 0;
    } catch (...) {
        return -3;
    }
}

// Build a schedule exactly the way the timed g2_prepare row builds one, then
// replay it. The row's digest binds an mcl-local layout, so this replay is the
// only statement that mcl prepared a schedule for the same G2 point.
extern "C" int mcl_g2_prepare_replay_raw(void* handle, std::uint8_t* output) {
    if (handle == nullptr || output == nullptr) return -1;
    try {
        MillerContext& context = *static_cast<MillerContext*>(handle);
        std::vector<Fp6> lines(mcl::bn::getPrecomputedQcoeffSize());
        mcl::bn::precomputeG2(lines.data(), context.g2);
        Fp12 value;
        mcl::bn::precomputedMillerLoop(value, context.g1, lines.data());
        const Fp* coefficients = value.getFp0();
        for (std::size_t i = 0; i < 12; ++i) {
            if (!encode_fp(output + i * 32, coefficients[i])) return -2;
        }
        return 0;
    } catch (...) {
        return -3;
    }
}

extern "C" int mcl_groth16_create(const std::uint8_t* blob,
                                          std::size_t gamma_abc_count,
                                          std::size_t public_input_count,
                                          void** output) {
    if (blob == nullptr || output == nullptr || gamma_abc_count != public_input_count + 1) {
        return -1;
    }
    *output = nullptr;
    try {
        std::unique_ptr<Groth16Context> context(new Groth16Context);
        const std::uint8_t* cursor = blob;
        G1 alpha_g1;
        G2 beta_g2;
        if (!decode_g1(alpha_g1, cursor)) return -2;
        cursor += 64;
        if (!decode_g2(beta_g2, cursor)) return -2;
        cursor += 128;
        if (!decode_g2(context->gamma_g2, cursor)) return -2;
        cursor += 128;
        if (!decode_g2(context->delta_g2, cursor)) return -2;
        cursor += 128;
        context->gamma_abc_g1.resize(gamma_abc_count);
        for (G1& point : context->gamma_abc_g1) {
            if (!decode_g1(point, cursor)) return -2;
            cursor += 64;
        }
        if (!decode_g1(context->proof_a, cursor)) return -2;
        cursor += 64;
        if (!decode_g2(context->proof_b, cursor)) return -2;
        cursor += 128;
        if (!decode_g1(context->proof_c, cursor)) return -2;
        cursor += 64;
        context->public_inputs.resize(public_input_count);
        for (Fr& scalar : context->public_inputs) {
            if (!decode_fr(scalar, cursor)) return -2;
            cursor += 32;
        }
        // Groth16's alpha/beta verifier-key term is proof-independent. Match
        // Arkworks' PreparedVerifyingKey and Helius' FixedPair by moving it
        // completely out of the timed online verification path.
        mcl::bn::pairing(context->alpha_beta, alpha_g1, beta_g2);
        // The gamma/delta G2 line schedules are verifier-key material too.
        context->gamma_coeff.resize(mcl::bn::getPrecomputedQcoeffSize());
        context->delta_coeff.resize(mcl::bn::getPrecomputedQcoeffSize());
        mcl::bn::precomputeG2(context->gamma_coeff.data(), context->gamma_g2);
        mcl::bn::precomputeG2(context->delta_coeff.data(), context->delta_g2);
        *output = context.release();
        return 0;
    } catch (...) {
        return -3;
    }
}

extern "C" void mcl_groth16_destroy(void* handle) {
    delete static_cast<Groth16Context*>(handle);
}

extern "C" int mcl_groth16_verify(void* handle, std::uint8_t* output) {
    if (handle == nullptr || output == nullptr) return -1;
    try {
        Fp12 result;
        G1 accumulator;
        *output =
            groth16_verify_prepared(*static_cast<Groth16Context*>(handle), result, accumulator) ? 1 : 0;
        return 0;
    } catch (...) {
        return -2;
    }
}

// The blob is 1312 bytes:
// alpha || beta || gamma || delta || k[0..2] || commitment_key_g ||
// commitment_key_g_sigma_neg || proof_a || proof_b || proof_c ||
// commitment || commitment_pok || public_input.
extern "C" int mcl_gnark_committed_groth16_create(
    const std::uint8_t* blob,
    void** output) {
    if (blob == nullptr || output == nullptr) return -1;
    *output = nullptr;
    try {
        std::unique_ptr<GnarkCommittedGroth16Context> context(
            new GnarkCommittedGroth16Context);
        const std::uint8_t* cursor = blob;
        G1 alpha;
        G2 beta;
        if (!decode_g1(alpha, cursor)) return -2;
        cursor += 64;
        if (!decode_g2(beta, cursor)) return -2;
        cursor += 128;
        if (!decode_g2(context->gamma_neg, cursor)) return -2;
        cursor += 128;
        G2::neg(context->gamma_neg, context->gamma_neg);
        if (!decode_g2(context->delta_neg, cursor)) return -2;
        cursor += 128;
        G2::neg(context->delta_neg, context->delta_neg);
        for (G1& point : context->k) {
            if (!decode_g1(point, cursor)) return -2;
            cursor += 64;
        }
        if (!decode_g2(context->commitment_key_g, cursor)) return -2;
        cursor += 128;
        if (!decode_g2(context->commitment_key_g_sigma_neg, cursor)) return -2;
        cursor += 128;
        if (!decode_g1(context->proof_a, cursor)) return -2;
        cursor += 64;
        if (!decode_g2(context->proof_b, cursor)) return -2;
        cursor += 128;
        if (!decode_g1(context->proof_c, cursor)) return -2;
        cursor += 64;
        if (!decode_g1(context->commitment, cursor)) return -2;
        cursor += 64;
        if (!decode_g1(context->commitment_pok, cursor)) return -2;
        cursor += 64;
        if (!decode_fr(context->public_input, cursor)) return -2;
        mcl::bn::pairing(context->alpha_beta, alpha, beta);
        const std::size_t coeff_size = mcl::bn::getPrecomputedQcoeffSize();
        context->gamma_neg_coeff.resize(coeff_size);
        context->delta_neg_coeff.resize(coeff_size);
        context->commitment_key_g_coeff.resize(coeff_size);
        context->commitment_key_g_sigma_neg_coeff.resize(coeff_size);
        mcl::bn::precomputeG2(context->gamma_neg_coeff.data(), context->gamma_neg);
        mcl::bn::precomputeG2(context->delta_neg_coeff.data(), context->delta_neg);
        mcl::bn::precomputeG2(context->commitment_key_g_coeff.data(),
                              context->commitment_key_g);
        mcl::bn::precomputeG2(context->commitment_key_g_sigma_neg_coeff.data(),
                              context->commitment_key_g_sigma_neg);
        *output = context.release();
        return 0;
    } catch (...) {
        return -3;
    }
}

extern "C" void mcl_gnark_committed_groth16_destroy(void* handle) {
    delete static_cast<GnarkCommittedGroth16Context*>(handle);
}

// Equivalence path only. It runs live millerLoopVec products rather than the
// prepared schedules the timed loop replays, so it is an independent check of
// the same statement.
extern "C" int mcl_gnark_committed_groth16_verify(
    void* handle,
    const std::uint8_t* commitment_hash,
    std::uint8_t* output) {
    if (handle == nullptr || commitment_hash == nullptr || output == nullptr) {
        return -1;
    }
    try {
        GnarkCommittedGroth16Context& context =
            *static_cast<GnarkCommittedGroth16Context*>(handle);
        Fr hash;
        if (!decode_fr(hash, commitment_hash)) return -2;

        G1 variable_part;
        const Fr scalars[2] = {context.public_input, hash};
        G1::mulVec(variable_part, context.k + 1, scalars, 2);
        G1 k_sum;
        G1::add(k_sum, context.k[0], variable_part);
        G1::add(k_sum, k_sum, context.commitment);

        const G1 pok_g1[2] = {context.commitment, context.commitment_pok};
        const G2 pok_g2[2] = {
            context.commitment_key_g_sigma_neg,
            context.commitment_key_g,
        };
        Fp12 miller;
        Fp12 result;
        mcl::bn::millerLoopVec(miller, pok_g1, pok_g2, 2);
        mcl::bn::finalExp(result, miller);
        if (!result.isOne()) {
            *output = 0;
            return 0;
        }

        const G1 groth_g1[3] = {context.proof_a, context.proof_c, k_sum};
        const G2 groth_g2[3] = {
            context.proof_b,
            context.delta_neg,
            context.gamma_neg,
        };
        mcl::bn::millerLoopVec(miller, groth_g1, groth_g2, 3);
        mcl::bn::finalExp(result, miller);
        *output = result == context.alpha_beta ? 1 : 0;
        return 0;
    } catch (...) {
        return -3;
    }
}

// Pooled committed-wires loop. Each context carries its own transcript hash,
// so the loop rotates over distinct proofs instead of replaying one.
extern "C" int mcl_gnark_committed_pool_run(void* const* handles,
                                                    const std::uint8_t* hashes,
                                                    std::size_t handle_count,
                                                    std::size_t iterations,
                                                    std::size_t rotation,
                                                    std::uint64_t* digest) {
    if (handles == nullptr || hashes == nullptr || handle_count == 0 ||
        digest == nullptr || iterations == 0) {
        return -1;
    }
    try {
        std::vector<Fr> decoded(handle_count);
        for (std::size_t i = 0; i < handle_count; ++i) {
            if (!decode_fr(decoded[i], hashes + i * 32)) return -2;
        }
        std::uint64_t acc = 0;
        Fp12 result;
        G1 accumulator;
        for (std::size_t i = 0; i < iterations; ++i) {
            const std::size_t slot = (i + rotation) % handle_count;
            GnarkCommittedGroth16Context& context =
                *static_cast<GnarkCommittedGroth16Context*>(handles[slot]);
            const std::uint64_t verdict =
                gnark_committed_verify_prehashed(context, decoded[slot], result, accumulator)
                    ? 1
                    : 0;
            if (!accumulator.isZero()) {
                accumulator.normalize();
                acc = fold_limbs(acc, accumulator.x);
                acc = fold_limbs(acc, accumulator.y);
            }
            acc = fold_limbs(acc, result);
            acc = fold_step(acc, verdict);
        }
        *digest = acc;
        operation_sink = acc;
        return 0;
    } catch (...) {
        return -3;
    }
}

// keys holds `key_count` records of alpha||beta||gamma||delta||k0||k1.
// members holds `member_count` records of a||b||c||public_input||challenge.
extern "C" int mcl_groth16_batch_create(const std::uint8_t* keys,
                                                std::size_t key_count,
                                                const std::uint8_t* members,
                                                const std::uint32_t* member_keys,
                                                std::size_t member_count,
                                                void** output) {
    if (keys == nullptr || members == nullptr || member_keys == nullptr ||
        output == nullptr || key_count == 0 || member_count == 0) {
        return -1;
    }
    *output = nullptr;
    try {
        std::unique_ptr<Groth16BatchContext> context(new Groth16BatchContext);
        context->keys.resize(key_count);
        const std::uint8_t* cursor = keys;
        const std::size_t coeff_size = mcl::bn::getPrecomputedQcoeffSize();
        for (std::size_t j = 0; j < key_count; ++j) {
            Groth16BatchKey& key = context->keys[j];
            G2 beta, gamma, delta;
            if (!decode_g1(key.alpha, cursor)) return -2;
            cursor += 64;
            if (!decode_g2(beta, cursor)) return -2;
            cursor += 128;
            if (!decode_g2(gamma, cursor)) return -2;
            cursor += 128;
            if (!decode_g2(delta, cursor)) return -2;
            cursor += 128;
            if (!decode_g1(key.k0, cursor)) return -2;
            cursor += 64;
            if (!decode_g1(key.k1, cursor)) return -2;
            cursor += 64;
            key.beta_coeff.resize(coeff_size);
            key.gamma_coeff.resize(coeff_size);
            key.delta_coeff.resize(coeff_size);
            mcl::bn::precomputeG2(key.beta_coeff.data(), beta);
            mcl::bn::precomputeG2(key.gamma_coeff.data(), gamma);
            mcl::bn::precomputeG2(key.delta_coeff.data(), delta);
        }
        context->members.resize(member_count);
        cursor = members;
        for (std::size_t i = 0; i < member_count; ++i) {
            Groth16BatchMember& member = context->members[i];
            if (member_keys[i] >= key_count) return -2;
            member.key = member_keys[i];
            if (!decode_g1(member.a, cursor)) return -2;
            cursor += 64;
            if (!decode_g2(member.b, cursor)) return -2;
            cursor += 128;
            if (!decode_g1(member.c, cursor)) return -2;
            cursor += 64;
            if (!decode_fr(member.x, cursor)) return -2;
            cursor += 32;
            if (!decode_fr(member.r, cursor)) return -2;
            cursor += 32;
        }
        *output = context.release();
        return 0;
    } catch (...) {
        return -3;
    }
}

extern "C" void mcl_groth16_batch_destroy(void* handle) {
    delete static_cast<Groth16BatchContext*>(handle);
}

extern "C" int mcl_groth16_batch_verify(void* handle, std::uint8_t* output) {
    if (handle == nullptr || output == nullptr) return -1;
    try {
        Fp12 result;
        std::uint64_t fold = 0;
        *output = groth16_batch_verify(*static_cast<Groth16BatchContext*>(handle), result, fold)
                      ? 1
                      : 0;
        return 0;
    } catch (...) {
        return -2;
    }
}

// Four-lane operation codes. This switch is the mcl half of the timed
// protocol: every case performs exactly the per-iteration work of the
// corresponding Rust lane and folds the same output words into the digest
// every iteration, so the digests must agree bit for bit across lanes.
extern "C" int mcl_four_lane_run(void* const* handles,
                                         std::size_t handle_count,
                                         std::uint32_t operation,
                                         std::size_t iterations,
                                         std::size_t rotation,
                                         std::uint64_t* digest) {
    if (handles == nullptr || handle_count == 0 || digest == nullptr || iterations == 0) {
        return -1;
    }
    try {
        std::uint64_t acc = 0;
        switch (operation) {
        case 0: {
            Fp value;
            for (std::size_t i = 0; i < iterations; ++i) {
                DirectContext& c =
                    *static_cast<DirectContext*>(handles[(i + rotation) % handle_count]);
                Fp::mul(value, c.fp[0], c.fp[1]);
                acc = fold_limbs(acc, value);
            }
            break;
        }
        case 1: {
            Fp value;
            for (std::size_t i = 0; i < iterations; ++i) {
                DirectContext& c =
                    *static_cast<DirectContext*>(handles[(i + rotation) % handle_count]);
                Fp::sqr(value, c.fp[0]);
                acc = fold_limbs(acc, value);
            }
            break;
        }
        case 2: {
            Fp2 value;
            for (std::size_t i = 0; i < iterations; ++i) {
                DirectContext& c =
                    *static_cast<DirectContext*>(handles[(i + rotation) % handle_count]);
                Fp2::mul(value, c.fp2[0], c.fp2[1]);
                acc = fold_limbs(acc, value);
            }
            break;
        }
        case 3: {
            Fp2 value;
            for (std::size_t i = 0; i < iterations; ++i) {
                DirectContext& c =
                    *static_cast<DirectContext*>(handles[(i + rotation) % handle_count]);
                Fp2::sqr(value, c.fp2[0]);
                acc = fold_limbs(acc, value);
            }
            break;
        }
        case 4: {
            Fp12 value;
            for (std::size_t i = 0; i < iterations; ++i) {
                DirectContext& c =
                    *static_cast<DirectContext*>(handles[(i + rotation) % handle_count]);
                Fp12::mul(value, c.fp12, c.fp12b);
                acc = fold_limbs(acc, value);
            }
            break;
        }
        case 5: {
            Fp12 value;
            for (std::size_t i = 0; i < iterations; ++i) {
                DirectContext& c =
                    *static_cast<DirectContext*>(handles[(i + rotation) % handle_count]);
                Fp12::sqr(value, c.fp12);
                acc = fold_limbs(acc, value);
            }
            break;
        }
        case 6: {
            Fp12 value;
            for (std::size_t i = 0; i < iterations; ++i) {
                DirectContext& c =
                    *static_cast<DirectContext*>(handles[(i + rotation) % handle_count]);
                value = c.fp12;
                Fp6 line;
                line.a = c.sparse[2];
                line.b = c.sparse[0];
                line.c = c.sparse[1];
                mcl::bn::mulSparseLine(value, line);
                acc = fold_limbs(acc, value);
            }
            break;
        }
        case 7: {
            Fp2 value;
            for (std::size_t i = 0; i < iterations; ++i) {
                DirectContext& c =
                    *static_cast<DirectContext*>(handles[(i + rotation) % handle_count]);
                Fp2::mulFp(value, c.fp2[0], c.scale);
                acc = fold_limbs(acc, value);
            }
            break;
        }
        case 8: {
            // Allocation policy differs per lane and stays disclosed in the
            // report: ark allocates internally per iteration, this loop
            // allocates a vector per iteration, helius writes a stack array.
            // Line schedules are lane-local layouts, so each lane folds its
            // own last line and the runner never compares digests across
            // lanes for this operation.
            for (std::size_t i = 0; i < iterations; ++i) {
                MillerContext& c =
                    *static_cast<MillerContext*>(handles[(i + rotation) % handle_count]);
                std::vector<Fp6> lines(mcl::bn::getPrecomputedQcoeffSize());
                mcl::bn::precomputeG2(lines.data(), c.g2);
                acc = fold_limbs(acc, lines.back());
            }
            break;
        }
        case 9: {
            Fp12 value;
            for (std::size_t i = 0; i < iterations; ++i) {
                MillerContext& c =
                    *static_cast<MillerContext*>(handles[(i + rotation) % handle_count]);
                mcl::bn::millerLoop(value, c.g1, c.g2);
                acc = fold_limbs(acc, value);
            }
            break;
        }
        case 10: {
            Fp12 value;
            for (std::size_t i = 0; i < iterations; ++i) {
                MillerContext& c =
                    *static_cast<MillerContext*>(handles[(i + rotation) % handle_count]);
                mcl::bn::precomputedMillerLoop(value, c.g1, c.g2_precomputed.data());
                acc = fold_limbs(acc, value);
            }
            break;
        }
        case 11: {
            Fp12 value;
            for (std::size_t i = 0; i < iterations; ++i) {
                MillerContext& c =
                    *static_cast<MillerContext*>(handles[(i + rotation) % handle_count]);
                mcl::bn::finalExp(value, c.output);
                acc = fold_limbs(acc, value);
            }
            break;
        }
        case 12: {
            Fp12 value;
            for (std::size_t i = 0; i < iterations; ++i) {
                MillerContext& c =
                    *static_cast<MillerContext*>(handles[(i + rotation) % handle_count]);
                mcl::bn::pairing(value, c.g1, c.g2);
                acc = fold_limbs(acc, value);
            }
            break;
        }
        case 13: {
            // Handles are raw 64-byte G1 encodings. Decode per iteration so
            // this lane pays the same canonical plus on-curve work as Rust.
            // The decoded coordinates fold before the verdict, nonidentity
            // only, identically in every lane. decode_g1 leaves z == 1.
            G1 point;
            for (std::size_t i = 0; i < iterations; ++i) {
                const std::uint8_t* bytes =
                    static_cast<const std::uint8_t*>(handles[(i + rotation) % handle_count]);
                const bool decoded = decode_g1(point, bytes);
                if (decoded && !point.isZero()) {
                    acc = fold_limbs(acc, point.x);
                    acc = fold_limbs(acc, point.y);
                }
                acc = fold_step(acc, decoded ? 1 : 0);
            }
            break;
        }
        case 14: {
            // Typed handles come from decode_g2, so z == 1 and the affine
            // coordinates fold directly, then the subgroup verdict.
            for (std::size_t i = 0; i < iterations; ++i) {
                G2& point = *static_cast<G2*>(handles[(i + rotation) % handle_count]);
                const std::uint64_t verdict = point.isValidOrder() ? 1 : 0;
                if (!point.isZero()) {
                    acc = fold_limbs(acc, point.x);
                    acc = fold_limbs(acc, point.y);
                }
                acc = fold_step(acc, verdict);
            }
            break;
        }
        case 15: {
            G1 result;
            for (std::size_t i = 0; i < iterations; ++i) {
                MsmContext& c =
                    *static_cast<MsmContext*>(handles[(i + rotation) % handle_count]);
                G1::mulVec(result, c.bases.data(), c.scalars.data(), c.bases.size());
                if (!result.isZero()) {
                    result.normalize();
                    acc = fold_limbs(acc, result.x);
                    acc = fold_limbs(acc, result.y);
                }
            }
            break;
        }
        case 16: {
            Fp12 result;
            for (std::size_t i = 0; i < iterations; ++i) {
                Groth16Context& c =
                    *static_cast<Groth16Context*>(handles[(i + rotation) % handle_count]);
                G1 accumulator;
                const std::uint64_t verdict = groth16_verify_prepared(c, result, accumulator) ? 1 : 0;
                acc = fold_limbs(acc, result);
                acc = fold_step(acc, verdict);
            }
            break;
        }
        case 18: {
            Fp12 value;
            for (std::size_t i = 0; i < iterations; ++i) {
                DirectContext& c =
                    *static_cast<DirectContext*>(handles[(i + rotation) % handle_count]);
                mcl::bn::cyclotomicSqr(value, c.fp12cyclo);
                acc = fold_limbs(acc, value);
            }
            break;
        }
        case 19: {
            Fp12 value;
            for (std::size_t i = 0; i < iterations; ++i) {
                MillerContext& c =
                    *static_cast<MillerContext*>(handles[(i + rotation) % handle_count]);
                mcl::bn::precomputedMillerLoop(value, c.g1, c.g2_precomputed.data());
                mcl::bn::finalExp(value, value);
                acc = fold_limbs(acc, value);
            }
            break;
        }
        case 20: {
            Fp12 result;
            G1 accumulator;
            for (std::size_t i = 0; i < iterations; ++i) {
                Groth16Context& c =
                    *static_cast<Groth16Context*>(handles[(i + rotation) % handle_count]);
                const std::uint64_t verdict =
                    groth16_verify_prepared(c, result, accumulator) ? 1 : 0;
                if (!accumulator.isZero()) {
                    accumulator.normalize();
                    acc = fold_limbs(acc, accumulator.x);
                    acc = fold_limbs(acc, accumulator.y);
                }
                acc = fold_limbs(acc, result);
                acc = fold_step(acc, verdict);
            }
            break;
        }
        case 21: {
            Fp12 result;
            for (std::size_t i = 0; i < iterations; ++i) {
                Groth16BatchContext& c =
                    *static_cast<Groth16BatchContext*>(handles[(i + rotation) % handle_count]);
                const std::uint64_t verdict = groth16_batch_verify(c, result, acc) ? 1 : 0;
                acc = fold_limbs(acc, result);
                acc = fold_step(acc, verdict);
            }
            break;
        }
        default: return -2;
        }
        *digest = acc;
        operation_sink = acc;
        return 0;
    } catch (...) {
        return -4;
    }
}
