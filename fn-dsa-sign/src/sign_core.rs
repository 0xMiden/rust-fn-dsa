//! Core signing operation for external hash-to-point implementations.
//!
//! This module provides `sign_poly` which computes the signature polynomial
//! from a hash-to-point result. This allows callers to use their own hash
//! function (e.g., RPO256) while reusing fn-dsa's Gaussian sampling.

use fn_dsa_comm::{mq, PRNG};

use crate::{
    flr::FLR,
    poly::{self, FFT, iFFT},
    sampler::Sampler,
};

/// 1/12289 as FLR constant for normalization.
const INV_Q: FLR = FLR::scaled(6004310871091074, -66);

/// Computes the signature polynomial using Gaussian sampling.
///
/// This function takes a hash-to-point result and produces the signature
/// polynomial s2. It handles the sampling retry loop internally.
///
/// # Parameters
/// - `logn`: log2 of the polynomial degree (e.g., 9 for Falcon-512)
/// - `hm`: hash-to-point result as u16 coefficients (n elements)
/// - `seed`: 56-byte seed for the PRNG used in Gaussian sampling
/// - `basis`: precomputed FFT basis [b00, b01, b10, b11] (4*n FLR elements)
/// - `tmp`: temporary buffer (must have at least 9*n FLR elements)
///
/// # Returns
/// The signature polynomial s2 as i16 coefficients (n elements).
///
/// # Type Parameter
/// - `P`: PRNG type to use for Gaussian sampling (e.g., SHAKE256_PRNG or ChaCha20PRNG)
pub fn sign_poly<P: PRNG>(
    logn: u32,
    hm: &[u16],
    seed: &[u8; 56],
    basis: &[FLR],
    tmp: &mut [FLR],
) -> alloc::vec::Vec<i16> {
    let n = 1usize << logn;
    assert!(hm.len() >= n);
    assert!(basis.len() >= 4 * n);
    assert!(tmp.len() >= 9 * n);

    // Create sampler - its state evolves across retries for fresh randomness
    let mut samp = Sampler::<P>::new(logn, seed);

    // Extract basis components (immutable references)
    let b00 = &basis[0..n];
    let b01 = &basis[n..2 * n];
    let b10 = &basis[2 * n..3 * n];
    let b11 = &basis[3 * n..4 * n];

    // Compute Gram matrix G = B * adj(B) once (doesn't change between retries)
    // g00 = b00*adj(b00) + b01*adj(b01)
    // g01 = b00*adj(b10) + b01*adj(b11)
    // g11 = b10*adj(b10) + b11*adj(b11)
    let mut g00 = alloc::vec![FLR::ZERO; n];
    let mut g01 = alloc::vec![FLR::ZERO; n];
    let mut g11 = alloc::vec![FLR::ZERO; n];
    let mut temp = alloc::vec![FLR::ZERO; n];

    g00.copy_from_slice(b00);
    poly::poly_mulownadj_fft(logn, &mut g00);
    temp.copy_from_slice(b01);
    poly::poly_mulownadj_fft(logn, &mut temp);
    poly::poly_add(logn, &mut g00, &temp);

    g01.copy_from_slice(b00);
    poly::poly_muladj_fft(logn, &mut g01, b10);
    temp.copy_from_slice(b01);
    poly::poly_muladj_fft(logn, &mut temp, b11);
    poly::poly_add(logn, &mut g01, &temp);

    g11.copy_from_slice(b10);
    poly::poly_mulownadj_fft(logn, &mut g11);
    temp.copy_from_slice(b11);
    poly::poly_mulownadj_fft(logn, &mut temp);
    poly::poly_add(logn, &mut g11, &temp);

    // Compute target vectors once (doesn't change between retries)
    // t0 = (hm/q) * b11 = (hm/q) * (-F)
    // t1 = -(hm/q) * b01 = (hm/q) * f
    let mut c_fft = alloc::vec![FLR::ZERO; n];
    for i in 0..n {
        c_fft[i] = FLR::from_i32(hm[i] as i32);
    }
    FFT(logn, &mut c_fft);

    let mut t0_orig = c_fft.clone();
    poly::poly_mul_fft(logn, &mut t0_orig, b11);
    poly::poly_mulconst(logn, &mut t0_orig, INV_Q);

    let mut t1_orig = c_fft;
    poly::poly_mul_fft(logn, &mut t1_orig, b01);
    poly::poly_mulconst(logn, &mut t1_orig, -INV_Q);

    loop {
        // Make working copies for this iteration
        let mut g00_work = g00.clone();
        let mut g01_work = g01.clone();
        let mut g11_work = g11.clone();
        let mut t0_work = t0_orig.clone();
        let mut t1_work = t1_orig.clone();

        // Sample z from discrete Gaussian using Gram matrix
        // After this, t0_work and t1_work contain sampled z0 and z1
        samp.ffsamp_fft(
            &mut t0_work,
            &mut t1_work,
            &mut g00_work,
            &mut g01_work,
            &mut g11_work,
            tmp,
        );

        // Compute t - z (the difference from target)
        let mut t0_min_z0 = t0_orig.clone();
        let mut t1_min_z1 = t1_orig.clone();
        for i in 0..n {
            t0_min_z0[i] -= t0_work[i];
            t1_min_z1[i] -= t1_work[i];
        }

        // Compute s = (t - z) * B where B = [[g, -f], [G, -F]]
        // s0 = (t0-z0) * g + (t1-z1) * G = (t0-z0) * b00 + (t1-z1) * b10
        let mut s0 = t0_min_z0.clone();
        poly::poly_mul_fft(logn, &mut s0, b00);
        temp.copy_from_slice(&t1_min_z1);
        poly::poly_mul_fft(logn, &mut temp, b10);
        poly::poly_add(logn, &mut s0, &temp);

        // s1 = (t0-z0) * (-f) + (t1-z1) * (-F) = (t0-z0) * b01 + (t1-z1) * b11
        let mut s1 = t0_min_z0;
        poly::poly_mul_fft(logn, &mut s1, b01);
        temp.copy_from_slice(&t1_min_z1);
        poly::poly_mul_fft(logn, &mut temp, b11);
        poly::poly_add(logn, &mut s1, &temp);

        // Check norm ||s||² in FFT domain
        // In FFT domain, ||s||² = (1/n) * sum of |s[i]|²
        let mut length_squared = 0.0_f64;
        let hn = n / 2;
        for i in 0..hn {
            let s0_re = s0[i].to_f64();
            let s0_im = s0[i + hn].to_f64();
            let s1_re = s1[i].to_f64();
            let s1_im = s1[i + hn].to_f64();
            length_squared += s0_re * s0_re + s0_im * s0_im;
            length_squared += s1_re * s1_re + s1_im * s1_im;
        }
        length_squared /= n as f64;

        // Check against squared norm bound
        if length_squared > mq::SQBETA[logn as usize] as f64 {
            // Norm too large, retry with evolved sampler state
            continue;
        }

        // Transform s1 back to coefficient domain via inverse FFT
        iFFT(logn, &mut s1);

        // Convert FLR values to i16 coefficients for signature
        let mut s2 = alloc::vec![0i16; n];
        for i in 0..n {
            s2[i] = s1[i].rint() as i16;
        }

        return s2;
    }
}
