//! Bit-accurate A64 interpreter: the executable semantics of the schedule.
//!
//! Registers are u64s. The C flag is modeled exactly per instruction (the
//! only NZCV bit the schedule consumes). Memory is a sparse map that panics
//! on wild or uninitialized access. `ret` verifies the AAPCS64 callee-saved
//! contract plus stack balance. ADC asserts its discarded carry-out is zero
//! and CINC asserts it cannot wrap, so the schedule's carry-bound reasoning
//! is checked rather than trusted.

use std::collections::BTreeMap;

use super::machine::{MachineA64, Reg};

/// Synthetic, well-separated buffer addresses for the kernel arguments.
pub const OUT_ADDR: u64 = 0x1_0000;
pub const X_ADDR: u64 = 0x2_0000;
pub const Y_ADDR: u64 = 0x3_0000;
pub const CONSTS_ADDR: u64 = 0x4_0000;
pub const X1_ADDR: u64 = 0x5_0000;
pub const Y1_ADDR: u64 = 0x6_0000;
pub const TABLE_ADDR: u64 = 0x7_0000;
const STACK_TOP: u64 = 0x8_0000;
/// Operand `k` of the table ABI. Well above the stack, one page apart, so a
/// stray offset lands on uninitialized memory instead of another operand.
const fn table_operand(index: usize) -> u64 {
    0x10_0000 + 0x1000 * index as u64
}
/// Recognizable poison for registers the ABI leaves undefined.
const POISON: u64 = 0xdead_0000_dead_0000;

const CALLEE_SAVED: [Reg; 10] = [
    Reg::X19,
    Reg::X20,
    Reg::X21,
    Reg::X22,
    Reg::X23,
    Reg::X24,
    Reg::X25,
    Reg::X26,
    Reg::X27,
    Reg::X28,
];

/// One slot per [`Reg`] variant.
const REGS: usize = 31;

pub struct InterpA64 {
    regs: [u64; REGS],
    /// Registers holding a schedule-defined value. Callee-saved registers
    /// start undefined: spills may copy them, arithmetic may not read them.
    defined: [bool; REGS],
    carry: bool,
    mem: BTreeMap<u64, u64>,
    entry_callee_saved: [(Reg, u64); CALLEE_SAVED.len()],
    returned: bool,
}

impl InterpA64 {
    /// The part of the frame every kernel shares: poisoned registers, the
    /// stack, `x0 = out`, and the constant table.
    fn base(p: [u64; 4], p_inv: u64) -> Self {
        let mut regs = [0u64; REGS];
        let mut defined = [false; REGS];
        for (index, value) in regs.iter_mut().enumerate() {
            *value = POISON | (index as u64) << 16;
        }
        for (reg, value) in [(Reg::X0, OUT_ADDR), (Reg::Sp, STACK_TOP), (Reg::Xzr, 0)] {
            regs[reg.index()] = value;
            defined[reg.index()] = true;
        }

        let mut mem = BTreeMap::new();
        for (limb, word) in p.into_iter().enumerate() {
            mem.insert(CONSTS_ADDR + 8 * limb as u64, word);
        }
        mem.insert(CONSTS_ADDR + 32, p_inv);

        InterpA64 {
            entry_callee_saved: CALLEE_SAVED.map(|r| (r, regs[r.index()])),
            regs,
            defined,
            // The ABI does not define entry flags. Poison C so a schedule
            // that consumes it before producing it fails deterministically.
            carry: true,
            mem,
            returned: false,
        }
    }

    /// An argument register holding a synthetic buffer address.
    fn pointer(&mut self, reg: Reg, address: u64) {
        self.regs[reg.index()] = address;
        self.defined[reg.index()] = true;
    }

    fn operand(&mut self, address: u64, value: [u64; 4]) {
        for (limb, word) in value.into_iter().enumerate() {
            self.mem.insert(address + 8 * limb as u64, word);
        }
    }

    /// Set up an AAPCS64 call frame:
    /// `x0 = out`, `x1 = x`, `x2 = y`, `x3 = { p[4], -p^-1 }`.
    pub fn call_frame(x: [u64; 4], y: [u64; 4], p: [u64; 4], p_inv: u64) -> Self {
        let mut frame = Self::base(p, p_inv);
        frame.pointer(Reg::X1, X_ADDR);
        frame.pointer(Reg::X2, Y_ADDR);
        frame.pointer(Reg::X3, CONSTS_ADDR);
        frame.operand(X_ADDR, x);
        frame.operand(Y_ADDR, y);
        frame
    }

    /// Set up an AAPCS64 call frame for the twelve-pointer table ABI:
    /// `x0 = z` (8 limbs, lane0 then lane1), `x1 = table`,
    /// `x2 = { p[4], -p^-1 }`. Operands are in table order,
    /// `x_i0 x_i1 y_i0 y_i1` per product.
    pub fn sosd6_frame(operands: [[u64; 4]; 12], p: [u64; 4], p_inv: u64) -> Self {
        let mut frame = Self::base(p, p_inv);
        frame.pointer(Reg::X1, TABLE_ADDR);
        frame.pointer(Reg::X2, CONSTS_ADDR);
        for (index, operand) in operands.into_iter().enumerate() {
            let address = table_operand(index);
            frame.operand(address, operand);
            frame.mem.insert(TABLE_ADDR + 8 * index as u64, address);
        }
        frame
    }

    /// Set up an AAPCS64 call frame for the dual-lane sosd2 kernel:
    /// `x0 = z` (8 limbs, lane0 then lane1), `x1 = x0`, `x2 = x1`, `x3 = y0`,
    /// `x4 = y1`, `x5 = { p[4], -p^-1 }`.
    pub fn sosd2_frame(
        x0: [u64; 4],
        x1: [u64; 4],
        y0: [u64; 4],
        y1: [u64; 4],
        p: [u64; 4],
        p_inv: u64,
    ) -> Self {
        let mut frame = Self::call_frame(x0, y0, p, p_inv);
        frame.pointer(Reg::X2, X1_ADDR);
        frame.pointer(Reg::X3, Y_ADDR);
        frame.pointer(Reg::X4, Y1_ADDR);
        frame.pointer(Reg::X5, CONSTS_ADDR);
        frame.operand(X1_ADDR, x1);
        frame.operand(Y1_ADDR, y1);
        frame
    }

    pub fn output(&self) -> [u64; 4] {
        assert!(self.returned, "kernel did not return");
        core::array::from_fn(|limb| self.read_mem(OUT_ADDR + 8 * limb as u64))
    }

    /// Both output lanes of the sosd2 kernel: `z[0..4]` and `z[4..8]`.
    pub fn output_lanes(&self) -> ([u64; 4], [u64; 4]) {
        assert!(self.returned, "kernel did not return");
        (
            core::array::from_fn(|limb| self.read_mem(OUT_ADDR + 8 * limb as u64)),
            core::array::from_fn(|limb| self.read_mem(OUT_ADDR + 32 + 8 * limb as u64)),
        )
    }

    fn read_reg(&self, r: Reg) -> u64 {
        assert!(self.defined[r.index()], "read of undefined register {r}");
        self.regs[r.index()]
    }

    fn write_reg(&mut self, r: Reg, value: u64) {
        assert_ne!(r, Reg::Xzr, "xzr is not writable");
        assert_ne!(r, Reg::Sp, "sp only moves through pre/post-index forms");
        self.regs[r.index()] = value;
        self.defined[r.index()] = true;
    }

    fn read_mem(&self, addr: u64) -> u64 {
        assert_eq!(addr % 8, 0, "unaligned load at {addr:#x}");
        *self
            .mem
            .get(&addr)
            .unwrap_or_else(|| panic!("load of uninitialized memory at {addr:#x}"))
    }

    fn write_mem(&mut self, addr: u64, value: u64) {
        assert_eq!(addr % 8, 0, "unaligned store at {addr:#x}");
        self.mem.insert(addr, value);
    }

    fn offset(&self, base: Reg, imm: i32) -> u64 {
        self.read_reg(base).wrapping_add(imm as u64)
    }

    /// `rn + rm + carry_in`, returning (result, carry_out).
    fn carry_add(&self, rn: Reg, rm: Reg, carry_in: bool) -> (u64, bool) {
        let (mid, c1) = self.read_reg(rn).overflowing_add(self.read_reg(rm));
        let (out, c2) = mid.overflowing_add(carry_in as u64);
        (out, c1 | c2)
    }

    /// `rn - rm - borrow_in`, returning (result, C) with C = NOT borrow.
    fn borrow_sub(&self, rn: Reg, rm: Reg, carry_in: bool) -> (u64, bool) {
        let (mid, b1) = self.read_reg(rn).overflowing_sub(self.read_reg(rm));
        let (out, b2) = mid.overflowing_sub(!carry_in as u64);
        (out, !(b1 | b2))
    }
}

impl MachineA64 for InterpA64 {
    fn comment(&mut self, _text: &str) {}

    fn claim_zero(&mut self, r: Reg, why: &str) {
        assert_eq!(self.read_reg(r), 0, "claim broken, {r} is not zero: {why}");
    }

    fn stp_pre(&mut self, r1: Reg, r2: Reg, imm: i32) {
        // AAPCS64 keeps sp 16-byte aligned at every instruction boundary.
        assert!(imm < 0 && imm % 16 == 0, "pre-index spill only");
        // Reads the raw registers: spilling caller-owned callee-saved values
        // is exactly what the prologue does before they are "defined".
        let sp = self.regs[Reg::Sp.index()].wrapping_add(imm as u64);
        self.regs[Reg::Sp.index()] = sp;
        self.write_mem(sp, self.regs[r1.index()]);
        self.write_mem(sp + 8, self.regs[r2.index()]);
    }

    fn stp(&mut self, r1: Reg, r2: Reg, base: Reg, imm: i32, _what: &str) {
        let addr = self.offset(base, imm);
        // Raw reads for the same prologue-spill reason as `stp_pre`.
        self.write_mem(addr, self.regs[r1.index()]);
        self.write_mem(addr + 8, self.regs[r2.index()]);
    }

    fn ldp(&mut self, r1: Reg, r2: Reg, base: Reg, imm: i32, _what: &str) {
        assert_ne!(r1, r2, "ldp destinations must differ");
        let addr = self.offset(base, imm);
        let (low, high) = (self.read_mem(addr), self.read_mem(addr + 8));
        self.write_reg(r1, low);
        self.write_reg(r2, high);
    }

    fn ldp_post(&mut self, r1: Reg, r2: Reg, base: Reg, imm: i32) {
        assert_ne!(r1, r2, "ldp destinations must differ");
        let addr = self.read_reg(base);
        let (low, high) = (self.read_mem(addr), self.read_mem(addr + 8));
        self.write_reg(r1, low);
        self.write_reg(r2, high);
        self.regs[base.index()] = addr.wrapping_add(imm as u64);
    }

    fn str_off(&mut self, r: Reg, base: Reg, imm: i32, _what: &str) {
        let addr = self.offset(base, imm);
        // Raw read: the prologue spills x23 before it is "defined".
        self.write_mem(addr, self.regs[r.index()]);
    }

    fn ldr(&mut self, r: Reg, base: Reg, imm: i32, _what: &str) {
        let value = self.read_mem(self.offset(base, imm));
        self.write_reg(r, value);
    }

    fn ldr_post(&mut self, r: Reg, base: Reg, imm: i32, _what: &str) {
        assert_ne!(r, base, "post-index load must not also write the base");
        let addr = self.read_reg(base);
        let value = self.read_mem(addr);
        self.write_reg(r, value);
        self.regs[base.index()] = addr.wrapping_add(imm as u64);
    }

    fn mov_zero(&mut self, r: Reg, _what: &str) {
        self.write_reg(r, 0);
    }

    fn mul(&mut self, rd: Reg, rn: Reg, rm: Reg, _what: &str) {
        let product = self.read_reg(rn) as u128 * self.read_reg(rm) as u128;
        self.write_reg(rd, product as u64);
    }

    fn umulh(&mut self, rd: Reg, rn: Reg, rm: Reg, _what: &str) {
        let product = self.read_reg(rn) as u128 * self.read_reg(rm) as u128;
        self.write_reg(rd, (product >> 64) as u64);
    }

    fn adds(&mut self, rd: Reg, rn: Reg, rm: Reg, _what: &str) {
        let (out, carry) = self.carry_add(rn, rm, false);
        self.write_reg(rd, out);
        self.carry = carry;
    }

    fn adcs(&mut self, rd: Reg, rn: Reg, rm: Reg, _what: &str) {
        let (out, carry) = self.carry_add(rn, rm, self.carry);
        self.write_reg(rd, out);
        self.carry = carry;
    }

    fn adc(&mut self, rd: Reg, rn: Reg, rm: Reg, what: &str) {
        let (out, carry) = self.carry_add(rn, rm, self.carry);
        assert!(!carry, "adc discarded a nonzero carry-out: {what}");
        self.write_reg(rd, out);
    }

    fn cinc_hs(&mut self, rd: Reg, rn: Reg, what: &str) {
        let value = self.read_reg(rn);
        if self.carry {
            assert_ne!(value, u64::MAX, "cinc wrapped: {what}");
            self.write_reg(rd, value + 1);
        } else {
            self.write_reg(rd, value);
        }
    }

    fn cmn(&mut self, rn: Reg, rm: Reg, _what: &str) {
        let (_, carry) = self.carry_add(rn, rm, false);
        self.carry = carry;
    }

    fn cset_hs(&mut self, rd: Reg, _what: &str) {
        self.write_reg(rd, self.carry as u64);
    }

    fn subs(&mut self, rd: Reg, rn: Reg, rm: Reg, _what: &str) {
        let (out, carry) = self.borrow_sub(rn, rm, true);
        self.write_reg(rd, out);
        self.carry = carry;
    }

    fn sbcs(&mut self, rd: Reg, rn: Reg, rm: Reg, _what: &str) {
        let (out, carry) = self.borrow_sub(rn, rm, self.carry);
        self.write_reg(rd, out);
        self.carry = carry;
    }

    fn csel_hs(&mut self, rd: Reg, rn: Reg, rm: Reg, _what: &str) {
        let value = if self.carry {
            self.read_reg(rn)
        } else {
            self.read_reg(rm)
        };
        self.write_reg(rd, value);
    }

    fn counted_loop(
        &mut self,
        counter: Reg,
        count: u32,
        _label: &str,
        body: &mut dyn FnMut(&mut Self),
    ) {
        assert!(count > 0, "counted loop must run at least once");
        // `mov w<counter>, #count`: a 32-bit write zeroes the upper half.
        self.write_reg(counter, count as u64);
        loop {
            body(self);
            // `subs w<counter>, w<counter>, #1` then `b.ne`: 32-bit forms.
            let previous =
                u32::try_from(self.read_reg(counter)).expect("loop counter stayed 32-bit");
            let next = previous.checked_sub(1).expect("loop counter underflow");
            self.write_reg(counter, next as u64);
            // SUBS sets C = NOT borrow. Previous >= 1 here, so C = 1. The
            // body must open its own chains and not read this stale flag.
            self.carry = true;
            if next == 0 {
                break;
            }
        }
    }

    fn ret(&mut self) {
        assert_eq!(
            self.regs[Reg::Sp.index()],
            STACK_TOP,
            "stack unbalanced at ret",
        );
        for (reg, entry_value) in self.entry_callee_saved {
            assert_eq!(
                self.regs[reg.index()],
                entry_value,
                "callee-saved {reg} not restored",
            );
        }
        self.returned = true;
    }
}
