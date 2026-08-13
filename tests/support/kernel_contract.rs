//! Text-level footprint and System V admission hooks for generated kernels.
//!
//! Algebra, poisoned CF/OF, initialized-register use, callee-saved registers,
//! and stack restoration remain the schedule interpreter's job. This module
//! adds deliberately boring limits that are easy to wire to a new symbol and
//! difficult for a giant generated leaf to bypass unnoticed.

#[derive(Clone, Copy, Debug)]
pub struct KernelBudget {
    pub symbol: &'static str,
    pub max_instructions: usize,
    pub max_text_bytes: usize,
    pub max_direct_calls: usize,
    pub max_stack_frame: usize,
}

pub const EXISTING_WIDE_PRIMITIVES: &[KernelBudget] = &[
    KernelBudget {
        symbol: "narsil_mont4_mulpre_x86",
        max_instructions: 512,
        max_text_bytes: 4 * 1024,
        max_direct_calls: 0,
        max_stack_frame: 128,
    },
    KernelBudget {
        symbol: "narsil_mont4_redc_x86",
        max_instructions: 512,
        max_text_bytes: 4 * 1024,
        max_direct_calls: 0,
        max_stack_frame: 128,
    },
    KernelBudget {
        symbol: "narsil_f2sqr_x86",
        max_instructions: 512,
        max_text_bytes: 2 * 1024,
        max_direct_calls: 0,
        max_stack_frame: 192,
    },
    KernelBudget {
        symbol: "narsil_f2sqr_small_x86",
        max_instructions: 256,
        max_text_bytes: 1024,
        max_direct_calls: 0,
        max_stack_frame: 192,
    },
    KernelBudget {
        symbol: "narsil_g2_ysqr_x86",
        max_instructions: 512,
        max_text_bytes: 2 * 1024,
        max_direct_calls: 0,
        max_stack_frame: 640,
    },
];

pub const FUTURE_MILLER_KERNELS: &[KernelBudget] = &[
    KernelBudget {
        symbol: "narsil_g2_double_wide_x86",
        max_instructions: 1_500,
        max_text_bytes: 12 * 1024,
        max_direct_calls: 24,
        max_stack_frame: 8 * 1024,
    },
    KernelBudget {
        symbol: "narsil_g2_add_wide_x86",
        max_instructions: 1_500,
        max_text_bytes: 12 * 1024,
        max_direct_calls: 24,
        max_stack_frame: 8 * 1024,
    },
];

fn kernel_body<'a>(rendered: &'a str, symbol: &str) -> Option<&'a str> {
    let marker = format!("\n{symbol}:\n");
    let body = rendered.split(&marker).nth(1)?;
    body.split(&format!("    .size {symbol},")).next()
}

fn declared_sizes(rendered: &str, symbol: &str) -> (usize, usize) {
    let prefix = format!("   {symbol}: schedule ");
    let line = rendered
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("{symbol}: generated provenance header missing"));
    let (_, counts) = line
        .split_once(", ")
        .unwrap_or_else(|| panic!("{symbol}: malformed provenance header"));
    let mut counts = counts.split(", ");
    let instructions = counts
        .next()
        .and_then(|part| part.strip_suffix(" instructions"))
        .and_then(|part| part.parse().ok())
        .unwrap_or_else(|| panic!("{symbol}: malformed instruction count"));
    let bytes = counts
        .next()
        .and_then(|part| part.strip_suffix(" bytes"))
        .and_then(|part| part.parse().ok())
        .unwrap_or_else(|| panic!("{symbol}: malformed byte count"));
    (instructions, bytes)
}

fn immediate_after(line: &str, opcode: &str) -> Option<usize> {
    line.trim()
        .strip_prefix(opcode)?
        .split(|ch: char| ch == ',' || ch.is_whitespace())
        .next()?
        .parse()
        .ok()
}

/// True when the instruction writes rsp outside the push/pop/sub/add shapes
/// the depth model tracks. Reading rsp -- a `[rsp + disp]` frame operand, or
/// materializing a frame pointer with `mov reg, rsp` -- leaves the stack
/// pointer alone and stays admitted.
fn writes_rsp(instruction: &str) -> bool {
    let code = instruction
        .split_once("/*")
        .map_or(instruction, |(code, _)| code);
    let Some((_, operands)) = code.trim_end().split_once(char::is_whitespace) else {
        return false;
    };
    let destination = operands.split(',').next().unwrap_or_default().trim();
    destination == "rsp"
}

fn assert_stack_and_calls(body: &str, budget: &KernelBudget) {
    // System V function entry is rsp = 8 (mod 16). Track only the generated
    // DSL's fixed push/pop/sub/add shapes. Reject dynamic rsp manipulation.
    let mut depth = 0usize;
    let mut maximum = 0usize;
    let mut calls = 0usize;
    for line in body.lines() {
        let instruction = line.trim();
        if instruction.starts_with("push ") {
            depth = depth.checked_add(8).expect("stack-depth overflow");
        } else if instruction.starts_with("pop ") {
            depth = depth.checked_sub(8).expect("stack pop before push");
        } else if let Some(bytes) = immediate_after(instruction, "sub rsp, ") {
            depth = depth.checked_add(bytes).expect("stack-depth overflow");
        } else if let Some(bytes) = immediate_after(instruction, "add rsp, ") {
            depth = depth.checked_sub(bytes).expect("stack free exceeds frame");
        } else if instruction.starts_with("call ") {
            calls += 1;
            assert_eq!(
                (8 + 16 - depth % 16) % 16,
                0,
                "{}: rsp is not 16-byte aligned at `{instruction}`",
                budget.symbol,
            );
            let target = instruction
                .strip_prefix("call ")
                .expect("checked prefix")
                .split_whitespace()
                .next()
                .expect("call target");
            assert!(
                target.starts_with("narsil_") && !target.contains(['[', ']', '*']),
                "{}: only direct generated-kernel calls are admitted: {target}",
                budget.symbol,
            );
        } else if writes_rsp(instruction) {
            panic!(
                "{}: unmodelled stack instruction `{instruction}`",
                budget.symbol
            );
        }
        maximum = maximum.max(depth);
    }
    assert_eq!(depth, 0, "{}: text-level stack imbalance", budget.symbol);
    assert!(
        calls <= budget.max_direct_calls,
        "{}: {calls} calls exceed budget {}",
        budget.symbol,
        budget.max_direct_calls,
    );
    assert!(
        maximum <= budget.max_stack_frame,
        "{}: {maximum}-byte frame exceeds budget {}",
        budget.symbol,
        budget.max_stack_frame,
    );
}

pub fn assert_kernel_contract(rendered: &str, budget: &KernelBudget) {
    let body = kernel_body(rendered, budget.symbol)
        .unwrap_or_else(|| panic!("{}: kernel missing", budget.symbol));
    let (instructions, bytes) = declared_sizes(rendered, budget.symbol);
    assert!(
        instructions <= budget.max_instructions,
        "{}: {instructions} instructions exceed budget {}",
        budget.symbol,
        budget.max_instructions,
    );
    assert!(
        bytes <= budget.max_text_bytes,
        "{}: {bytes} text bytes exceed budget {}",
        budget.symbol,
        budget.max_text_bytes,
    );
    assert!(
        body.trim_end().ends_with("ret"),
        "{}: missing ret",
        budget.symbol
    );
    assert_stack_and_calls(body, budget);
}

pub fn assert_kernel_contract_if_present(rendered: &str, budget: &KernelBudget) {
    if kernel_body(rendered, budget.symbol).is_some() {
        assert_kernel_contract(rendered, budget);
    }
}
