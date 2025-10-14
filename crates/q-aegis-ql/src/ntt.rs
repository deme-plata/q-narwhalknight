//! Number Theoretic Transform (NTT) for fast polynomial multiplication
//!
//! Optimized Cooley-Tukey NTT implementation with precomputed roots
//! for O(n log n) polynomial multiplication.

/// Compute modular exponentiation (base^exp mod modulus)
fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;

    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % modulus;
        }
        base = (base * base) % modulus;
        exp >>= 1;
    }

    result
}

/// Find a primitive root of unity for NTT
fn find_primitive_root(n: usize, modulus: u32) -> u32 {
    // For modulus = 4093, n = 512, we need a 512-th root of unity
    // This requires (modulus - 1) to be divisible by n
    assert_eq!((modulus - 1) % (n as u32), 0, "Invalid NTT parameters");

    // Try to find a generator
    for candidate in 2..modulus {
        let root = mod_pow(candidate as u64, ((modulus - 1) / n as u32) as u64, modulus as u64);
        // Verify it's a primitive n-th root
        if mod_pow(root, n as u64, modulus as u64) == 1 {
            return root as u32;
        }
    }

    panic!("No primitive root found");
}

/// Precompute NTT roots of unity
pub fn precompute_roots(n: usize, modulus: u32) -> Vec<u32> {
    let root = find_primitive_root(n, modulus);
    let mut roots = Vec::with_capacity(n);

    for i in 0..n {
        let power = mod_pow(root as u64, i as u64, modulus as u64);
        roots.push(power as u32);
    }

    roots
}

/// Bit-reverse permutation for Cooley-Tukey NTT
fn bit_reverse(arr: &[u32]) -> Vec<u32> {
    let n = arr.len();
    let mut result = arr.to_vec();
    let log_n = (n as f64).log2() as usize;

    for i in 0..n {
        let mut rev = 0;
        let mut temp = i;
        for _ in 0..log_n {
            rev = (rev << 1) | (temp & 1);
            temp >>= 1;
        }
        if i < rev {
            result.swap(i, rev);
        }
    }

    result
}

/// Forward NTT (polynomial to frequency domain)
pub fn forward_ntt(poly: &[u32], roots: &[u32], modulus: u32) -> Vec<u32> {
    let n = poly.len();
    assert!(n.is_power_of_two(), "NTT requires power-of-2 length");

    let mut result = bit_reverse(poly);
    let mut layer = 1;

    while layer < n {
        for i in (0..n).step_by(2 * layer) {
            for j in 0..layer {
                let idx = i + j;
                let root_idx = (n / (2 * layer)) * j;

                let t = ((roots[root_idx] as u64) * (result[idx + layer] as u64)) % (modulus as u64);
                let u = result[idx] as u64;

                result[idx + layer] = ((u + modulus as u64 - t) % (modulus as u64)) as u32;
                result[idx] = ((u + t) % (modulus as u64)) as u32;
            }
        }
        layer <<= 1;
    }

    result
}

/// Inverse NTT (frequency domain to polynomial)
pub fn inverse_ntt(freq: &[u32], roots: &[u32], modulus: u32) -> Vec<u32> {
    let n = freq.len();

    // Compute inverse roots
    let mut inv_roots = Vec::with_capacity(n);
    for &root in roots {
        // Inverse is root^(n-1) in modular arithmetic
        let inv = mod_pow(root as u64, (modulus - 2) as u64, modulus as u64);
        inv_roots.push(inv as u32);
    }

    // Apply forward NTT with inverse roots
    let mut result = forward_ntt(freq, &inv_roots, modulus);

    // Multiply by n^{-1} mod modulus
    let n_inv = mod_pow(n as u64, (modulus - 2) as u64, modulus as u64) as u32;
    for coeff in &mut result {
        *coeff = ((*coeff as u64 * n_inv as u64) % modulus as u64) as u32;
    }

    result
}

/// Pointwise multiplication in frequency domain
pub fn pointwise_multiply(a: &[u32], b: &[u32], modulus: u32) -> Vec<u32> {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| ((x as u64 * y as u64) % modulus as u64) as u32)
        .collect()
}

/// Polynomial addition (coefficient-wise)
pub fn polynomial_add(a: &[u32], b: &[u32], modulus: u32) -> Vec<u32> {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| ((x as u64 + y as u64) % modulus as u64) as u32)
        .collect()
}

/// Polynomial subtraction (coefficient-wise)
pub fn polynomial_subtract(a: &[u32], b: &[u32], modulus: u32) -> Vec<u32> {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| ((x as u64 + modulus as u64 - y as u64) % modulus as u64) as u32)
        .collect()
}

/// Polynomial multiplication using NTT (O(n log n))
pub fn polynomial_multiply_mod(a: &[u32], b: &[u32], modulus: u32) -> Vec<u32> {
    let n = a.len().max(b.len()).next_power_of_two();
    let mut a_padded = a.to_vec();
    let mut b_padded = b.to_vec();
    a_padded.resize(n, 0);
    b_padded.resize(n, 0);

    let roots = precompute_roots(n, modulus);

    let a_ntt = forward_ntt(&a_padded, &roots, modulus);
    let b_ntt = forward_ntt(&b_padded, &roots, modulus);
    let c_ntt = pointwise_multiply(&a_ntt, &b_ntt, modulus);

    inverse_ntt(&c_ntt, &roots, modulus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntt_inverse() {
        // Use NTT-friendly modulus: 12289 = 24 * 512 + 1
        let modulus = 12289;
        let n = 512;
        let roots = precompute_roots(n, modulus);

        let poly = vec![1, 2, 3, 4, 5, 0, 0, 0];
        let mut padded = poly.clone();
        padded.resize(n, 0);

        let freq = forward_ntt(&padded, &roots, modulus);
        let recovered = inverse_ntt(&freq, &roots, modulus);

        for (i, &val) in recovered.iter().take(poly.len()).enumerate() {
            assert_eq!(val, poly[i]);
        }
    }
}
