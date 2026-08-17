//! Rust `asm!` template emitter for the register-shape schedules.
//!
//! The schedule names concrete registers, so an inline body must rename them
//! to template placeholders and let the register allocator choose. Every
//! register a register-shape schedule touches must appear in its operand
//! table: an unlisted one panics here instead of reaching the template as a
//! fixed register, which would defeat the allocation the inline form exists
//! for.
//!
//! Memory forms panic as well. The register shape emits no load, no store and
//! no `ret`, so `nomem`, `nostack` and `pure` hold by construction, not by
//! inspection of the emitted text.

use super::machine::{MachineA64, Reg};

/// What an inline operand carries into and out of the template.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InlineRole {
    /// A limb of `a` on entry, the matching result limb on exit.
    Result,
    /// A limb of `b` on entry, schedule scratch afterwards.
    Operand,
    /// A modulus limb. Never written.
    Modulus,
    /// Written before it is read.
    Scratch,
}

/// One placeholder of an `asm!` template and the value it binds.
pub struct InlineOperand {
    pub reg: Reg,
    /// Placeholder name, also the Rust operand name. Unique per template.
    pub name: &'static str,
    pub role: InlineRole,
}

impl InlineOperand {
    /// The `asm!` operand declaration. `index` counts within the role, so
    /// `Result` 0 binds `a[0]` and returns `z[0]`.
    fn declaration(&self, index: usize) -> String {
        let name = self.name;
        match self.role {
            InlineRole::Result => format!("{name} = inout(reg) {name}"),
            InlineRole::Operand => format!("{name} = inout(reg) b[{index}] => _"),
            InlineRole::Modulus => format!("{name} = in(reg) p[{index}]"),
            InlineRole::Scratch => format!("{name} = out(reg) _"),
        }
    }
}

/// Registers of one role, in table order.
pub fn role_registers(operands: &[InlineOperand], role: InlineRole) -> Vec<Reg> {
    operands
        .iter()
        .filter(|operand| operand.role == role)
        .map(|operand| operand.reg)
        .collect()
}

pub struct EmitterInlineA64 {
    operands: &'static [InlineOperand],
    lines: Vec<String>,
}

impl EmitterInlineA64 {
    pub fn new(operands: &'static [InlineOperand]) -> Self {
        Self {
            operands,
            lines: Vec::new(),
        }
    }

    /// The template lines, one instruction each.
    pub fn into_lines(self) -> Vec<String> {
        self.lines
    }

    /// The `asm!` operand declarations, in table order.
    pub fn declarations(&self) -> Vec<String> {
        let mut seen = [0usize; 4];
        self.operands
            .iter()
            .map(|operand| {
                let slot = &mut seen[operand.role as usize];
                let index = *slot;
                *slot += 1;
                operand.declaration(index)
            })
            .collect()
    }

    fn name(&self, r: Reg) -> String {
        if r == Reg::Xzr {
            // Not an allocatable register, so it needs no operand.
            return "xzr".to_string();
        }
        let operand = self
            .operands
            .iter()
            .find(|operand| operand.reg == r)
            .unwrap_or_else(|| panic!("{r} has no inline operand; the template cannot name it"));
        format!("{{{}}}", operand.name)
    }

    fn instr(&mut self, text: String) {
        self.lines.push(text);
    }

    fn rrr(&mut self, mnemonic: &str, rd: Reg, rn: Reg, rm: Reg) {
        let (rd, rn, rm) = (self.name(rd), self.name(rn), self.name(rm));
        self.instr(format!("{mnemonic} {rd}, {rn}, {rm}"));
    }

    fn no_memory(form: &str) -> ! {
        panic!("the register shape emits no {form}; the inline body must stay register only")
    }
}

impl MachineA64 for EmitterInlineA64 {
    fn comment(&mut self, _text: &str) {}

    fn claim_zero(&mut self, _r: Reg, _why: &str) {}

    fn stp_pre(&mut self, _r1: Reg, _r2: Reg, _imm: i32) {
        Self::no_memory("stack spill")
    }

    fn stp(&mut self, _r1: Reg, _r2: Reg, _base: Reg, _imm: i32, _what: &str) {
        Self::no_memory("store")
    }

    fn ldp(&mut self, _r1: Reg, _r2: Reg, _base: Reg, _imm: i32, _what: &str) {
        Self::no_memory("load")
    }

    fn ldp_post(&mut self, _r1: Reg, _r2: Reg, _base: Reg, _imm: i32) {
        Self::no_memory("load")
    }

    fn str_off(&mut self, _r: Reg, _base: Reg, _imm: i32, _what: &str) {
        Self::no_memory("store")
    }

    fn ldr(&mut self, _r: Reg, _base: Reg, _imm: i32, _what: &str) {
        Self::no_memory("load")
    }

    fn ldr_post(&mut self, _r: Reg, _base: Reg, _imm: i32, _what: &str) {
        Self::no_memory("load")
    }

    fn mov_zero(&mut self, r: Reg, _what: &str) {
        let rd = self.name(r);
        self.instr(format!("mov {rd}, xzr"));
    }

    fn mul(&mut self, rd: Reg, rn: Reg, rm: Reg, _what: &str) {
        self.rrr("mul", rd, rn, rm);
    }

    fn umulh(&mut self, rd: Reg, rn: Reg, rm: Reg, _what: &str) {
        self.rrr("umulh", rd, rn, rm);
    }

    fn adds(&mut self, rd: Reg, rn: Reg, rm: Reg, _what: &str) {
        self.rrr("adds", rd, rn, rm);
    }

    fn adcs(&mut self, rd: Reg, rn: Reg, rm: Reg, _what: &str) {
        self.rrr("adcs", rd, rn, rm);
    }

    fn adc(&mut self, rd: Reg, rn: Reg, rm: Reg, _what: &str) {
        self.rrr("adc", rd, rn, rm);
    }

    fn cinc_hs(&mut self, rd: Reg, rn: Reg, _what: &str) {
        let (rd, rn) = (self.name(rd), self.name(rn));
        self.instr(format!("cinc {rd}, {rn}, hs"));
    }

    fn cmn(&mut self, rn: Reg, rm: Reg, _what: &str) {
        let (rn, rm) = (self.name(rn), self.name(rm));
        self.instr(format!("cmn {rn}, {rm}"));
    }

    fn cset_hs(&mut self, rd: Reg, _what: &str) {
        let rd = self.name(rd);
        self.instr(format!("cset {rd}, hs"));
    }

    fn subs(&mut self, rd: Reg, rn: Reg, rm: Reg, _what: &str) {
        self.rrr("subs", rd, rn, rm);
    }

    fn sbcs(&mut self, rd: Reg, rn: Reg, rm: Reg, _what: &str) {
        self.rrr("sbcs", rd, rn, rm);
    }

    fn csel_hs(&mut self, rd: Reg, rn: Reg, rm: Reg, _what: &str) {
        let (rd, rn, rm) = (self.name(rd), self.name(rn), self.name(rm));
        self.instr(format!("csel {rd}, {rn}, {rm}, hs"));
    }

    fn counted_loop(
        &mut self,
        _counter: Reg,
        _count: u32,
        _label: &str,
        _body: &mut dyn FnMut(&mut Self),
    ) {
        panic!("an inline body carries no label; a loop would need a unique one per expansion")
    }

    fn ret(&mut self) {
        panic!("an inline body returns by falling out of the template")
    }
}
