# DeepSeek N1 Submission — DRAFT for verification

**Submitted:** 2026-05-14
**Job:** N1 — Nova crate selection and integration spike
**Status:** ⚠️ DRAFT — API signatures used in this code are unverified against the actual `nova-snark` crate. **Do not merge without compile-checking against a real `cargo add nova-snark` build first.**

---

## What needs verification before merge

The code below uses several `nova-snark` API surfaces that look plausible but may not match the current crate version. Specifically:

1. **`StepCircuit::Args` associated type** — current `nova-snark` may not expose `Args`; check the trait definition in the version you pin against.
2. **`PublicParams::setup(&primary, &secondary, 4, 4, 2, 1)` signature** — the actual `setup` function in microsoft/Nova typically takes only `&primary, &secondary` and infers constraint shape from synthesis; the four numeric args may be wrong.
3. **`AllocatedNum::add(cs.namespace(...), &b)` method** — the additive API in current `bellpepper-core` may be `a.add(...)` or `AllocatedNum::add(cs, a, b)` — both have been used in different versions.
4. **`Scalar::from_u128`** — the bn256 Scalar may only expose `from(u64)` and require `BigInt`-shaped construction for u128 values.
5. **`PublicParams::<E1, C1, C2>::setup` generic order** — `E1, C1, C2` may be `<G1, G2, C1, C2>` or similar with the curve cycle exposed.
6. **`RecursiveSNARK::prove_step(&pp, &primary, &secondary)`** — current API may take `RecursiveSNARK::new` first then `prove_step(&pp, &circuit_primary)` returning a new instance, not mutating in place.

**Action before merging:**
- Run `cargo add nova-snark --features bn256_grumpkin` in a scratch project
- Open the crate docs via `cargo doc --open --package nova-snark`
- Map every API surface this draft uses to the actual signatures
- Fix discrepancies
- THEN compile-check + run the Fibonacci test

The acceptance criteria require the test to actually pass — without API verification this draft will not compile.

---

## DeepSeek-provided draft (verbatim, save for reference)

### `crates/q-ivc/Cargo.toml` addition

```toml
[dependencies]
# existing dependencies …
nova-snark = { git = "https://github.com/microsoft/Nova.git", rev = "b65e5c2", features = ["bn256_grumpkin"] }
# The `b65e5c2` commit is the stable point for the BN256/Grumpkin cycle.
# Exact revision ensures reproducibility.
```

**Note:** the rev `b65e5c2` may not match the actual stable point. Use `cargo show nova-snark` to find a real recent stable revision; the published crate version is typically preferred over a git pin unless we need an unreleased fix.

### `crates/q-ivc/src/recursion/mod.rs`

```rust
pub mod spike;       // Fibonacci hello-world
// pub mod driver;   // placeholder for N3
```

### `crates/q-ivc/src/recursion/spike.rs`

```rust
use nova_snark::{
    provider::bn256_grumpkin::bn256::Scalar,
    traits::circuit::{StepCircuit, TrivialTestCircuit},
    RecursiveSNARK, PublicParams,
};
use std::time::Instant;

/// Fibonacci step: given (a, b), produce (b, a + b).
#[derive(Clone, Debug)]
pub struct FibonacciCircuit;

impl StepCircuit<Scalar> for FibonacciCircuit {
    type Args = (Scalar, Scalar);

    fn arity(&self) -> usize {
        2 // two inputs: a and b
    }

    /// Constrain the step: input z[0]=a, z[1]=b → output[0]=b, output[1]=a+b.
    fn synthesize<CS: nova_snark::traits::circuit::ConstraintSystem<Scalar>>(
        &self,
        cs: &mut CS,
        z: &[nova_snark::traits::circuit::AllocatedNum<Scalar>],
    ) -> Result<Vec<nova_snark::traits::circuit::AllocatedNum<Scalar>>, nova_snark::errors::SynthesisError>
    {
        assert_eq!(z.len(), 2, "Fibonacci expects exactly 2 inputs");
        let a = z[0].clone();
        let b = z[1].clone();

        let out0 = b.clone();
        let out1 = a.add(cs.namespace(|| "a_plus_b"), &b)?;

        Ok(vec![out0, out1])
    }

    fn output(&self, z: &[Scalar]) -> Vec<Scalar> {
        let a = z[0];
        let b = z[1];
        vec![b, a + b]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nova_snark::provider::bn256_grumpkin::bn256::Scalar as F;

    fn fib_field(n: usize) -> F {
        let mut a = F::from(0u64);
        let mut b = F::from(1u64);
        for _ in 0..n {
            let tmp = a + b;
            a = b;
            b = tmp;
        }
        a
    }

    #[test]
    fn test_nova_fibonacci_100_steps() {
        let steps = 100usize;

        let setup_start = Instant::now();
        let primary = FibonacciCircuit;
        let secondary = TrivialTestCircuit::default();

        let pp = PublicParams::<nova_snark::provider::bn256_grumpkin::bn256::Point, _, _>::setup(
            &primary,
            &secondary,
            4, 4, 2, 1,
        )
        .expect("PublicParams setup failed");
        let setup_time = setup_start.elapsed();

        let prove_start = Instant::now();
        let initial_state = vec![F::from(0u64), F::from(1u64)];
        let mut recursive_snark = RecursiveSNARK::new(
            &pp,
            &primary,
            &secondary,
            initial_state.clone(),
        )
        .expect("RecursiveSNARK creation failed");

        let step_start = Instant::now();
        recursive_snark
            .prove_step(&pp, &primary, &secondary)
            .expect("Prove step 1 failed");
        let first_step_time = step_start.elapsed();

        for i in 2..=steps {
            recursive_snark
                .prove_step(&pp, &primary, &secondary)
                .expect(&format!("Prove step {} failed", i));
        }
        let total_prove_time = prove_start.elapsed();

        let verify_start = Instant::now();
        let (zi_primary, _zi_secondary) = recursive_snark
            .verify(&pp, steps, &initial_state)
            .expect("Verification of recursive SNARK failed");
        let verify_time = verify_start.elapsed();

        let expected = fib_field(steps);
        assert_eq!(zi_primary[0], expected, "Fib(100) mismatch after 100 folds");

        println!("=== Nova Fibonacci Spike Benchmarks ===");
        println!("Setup time:          {:?}", setup_time);
        println!("First step time:     {:?}", first_step_time);
        println!("Total prove ({steps} steps): {:?}", total_prove_time);
        println!("Verify time:         {:?}", verify_time);
        println!("=======================================");
    }
}
```

### `crates/q-ivc/src/lib.rs`

```rust
pub mod recursion;
```

---

## Suggested next steps (for whoever picks this up)

1. **Verify API first.** Don't paste this code in and `cargo check`. Open the nova-snark docs, write the smallest possible `cargo new` scratch project with just nova-snark, get a Fibonacci-style circuit compiling, THEN port the verified shape into `crates/q-ivc/src/recursion/spike.rs`.

2. **Pin a real version.** `cargo add nova-snark` will pull the latest published version. Pin that exact version in `Cargo.toml` (not a git rev unless you genuinely need an unreleased fix). The submission's `b65e5c2` revision is unverified.

3. **Check the curve cycle generic order.** Different nova-snark versions arrange the generics differently; `PublicParams::<X, Y, Z>` may need `<G1, G2, C1, C2>` or another ordering.

4. **Don't pre-pad constraint counts.** The `4, 4, 2, 1` arguments to `setup` are suspicious — current Nova typically infers constraint count from synthesis, doesn't take it as a numeric arg.

5. **Once verified, push to `code.quillon.xyz`.** Branch `deepseek/n1-nova-spike`. Run `git update-server-info`.

---

## Why this is saved as a draft

The Q-NarwhalKnight standing rule is "no placeholders that look right but aren't." This submission would compile on a hypothetical nova-snark with the API signatures used here, but multiple of those signatures don't match the current published crate. Saving it as a reference for the eventual real implementer rather than landing it as code is the honest choice.

If you (the next agent picking this up) find the API signatures DO match the current nova-snark — great, use this almost verbatim. If they don't, treat this as design intent rather than working code.
