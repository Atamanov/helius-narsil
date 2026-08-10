#include <mcl/bn.h>

#include <cstddef>
#include <cstdint>

namespace {

int set_fp(mclBnFp *out, const std::uint8_t *bytes) {
  static constexpr char digits[] = "0123456789abcdef";
  char text[64];
  for (std::size_t i = 0; i < 32; ++i) {
    text[2 * i] = digits[bytes[i] >> 4];
    text[2 * i + 1] = digits[bytes[i] & 15];
  }
  return mclBnFp_setStr(out, text, sizeof(text), 16);
}

int set_fr(mclBnFr *out, const std::uint8_t *bytes) {
  static constexpr char digits[] = "0123456789abcdef";
  char text[64];
  for (std::size_t i = 0; i < 32; ++i) {
    text[2 * i] = digits[bytes[i] >> 4];
    text[2 * i + 1] = digits[bytes[i] & 15];
  }
  return mclBnFr_setStr(out, text, sizeof(text), 16);
}

int set_g1(mclBnG1 *out, const std::uint8_t *bytes) {
  if (set_fp(&out->x, bytes) != 0 || set_fp(&out->y, bytes + 32) != 0) {
    return -1;
  }
  mclBnFp_setInt(&out->z, 1);
  return mclBnG1_isValid(out) ? 0 : -2;
}

int set_g2(mclBnG2 *out, const std::uint8_t *bytes) {
  if (set_fp(&out->x.d[1], bytes) != 0 ||
      set_fp(&out->x.d[0], bytes + 32) != 0 ||
      set_fp(&out->y.d[1], bytes + 64) != 0 ||
      set_fp(&out->y.d[0], bytes + 96) != 0) {
    return -1;
  }
  mclBnFp_setInt(&out->z.d[0], 1);
  mclBnFp_setInt(&out->z.d[1], 0);
  return mclBnG2_isValid(out) && mclBnG2_isValidOrder(out) ? 0 : -2;
}

int set_gt(mclBnGT *out, const std::uint8_t *bytes) {
  for (std::size_t i = 0; i < 12; ++i) {
    if (set_fp(&out->d[i], bytes + 32 * i) != 0) {
      return -1;
    }
  }
  return 0;
}

} // namespace

extern "C" int narsil_mcl_init() {
  return mclBn_init(MCL_BN_SNARK1, MCLBN_COMPILED_TIME_VAR);
}

extern "C" int narsil_mcl_pairing_matches(const std::uint8_t *g1,
                                           const std::uint8_t *g2,
                                           const std::uint8_t *expected) {
  mclBnG1 p;
  mclBnG2 q;
  mclBnGT got;
  mclBnGT want;
  if (set_g1(&p, g1) != 0 || set_g2(&q, g2) != 0 ||
      set_gt(&want, expected) != 0) {
    return -1;
  }
  mclBn_pairing(&got, &p, &q);
  return mclBnGT_isEqual(&got, &want);
}

extern "C" int narsil_mcl_miller_matches(const std::uint8_t *g1,
                                          const std::uint8_t *g2,
                                          const std::uint8_t *expected) {
  mclBnG1 p;
  mclBnG2 q;
  mclBnGT got;
  mclBnGT want;
  if (set_g1(&p, g1) != 0 || set_g2(&q, g2) != 0 ||
      set_gt(&want, expected) != 0) {
    return -1;
  }
  mclBn_millerLoop(&got, &p, &q);
  return mclBnGT_isEqual(&got, &want);
}

extern "C" int narsil_mcl_final_exp_matches(const std::uint8_t *input,
                                             const std::uint8_t *expected) {
  mclBnGT value;
  mclBnGT got;
  mclBnGT want;
  if (set_gt(&value, input) != 0 || set_gt(&want, expected) != 0) {
    return -1;
  }
  mclBn_finalExp(&got, &value);
  return mclBnGT_isEqual(&got, &want);
}

extern "C" int narsil_mcl_msm_matches(const std::uint8_t *points,
                                      const std::uint8_t *scalars,
                                      std::size_t count,
                                      const std::uint8_t *expected) {
  if (count == 0 || count > 8) {
    return -1;
  }
  mclBnG1 p[8];
  mclBnFr s[8];
  mclBnG1 got;
  mclBnG1 want;
  for (std::size_t i = 0; i < count; ++i) {
    if (set_g1(&p[i], points + 64 * i) != 0 ||
        set_fr(&s[i], scalars + 32 * i) != 0) {
      return -2;
    }
  }
  if (set_g1(&want, expected) != 0) {
    return -3;
  }
  mclBnG1_mulVec(&got, p, s, count);
  return mclBnG1_isEqual(&got, &want);
}
