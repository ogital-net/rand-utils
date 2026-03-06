/// Generates a random string of specified length using a secure random number generator.
///
/// The generated string contains characters from the set:
/// `-`, `_`, `0-9`, `A-Z`, and `a-z`.
///
/// This function uses SIMD instructions for performance when available:
/// - SSSE3 on x86_64
/// - NEON on AArch64/ARM64
///
/// The implementation processes data in 16-byte chunks for optimal SIMD performance.
///
/// # Arguments
/// * `len` - The length of the random string to generate
///
/// # Returns
/// A String of the specified length containing random characters.
///
/// # Panics
/// Panics if random byte generation fails.
pub fn rand_string<R: rand_core::Rng>(rng: &mut R, len: usize) -> String {
    if len == 0 {
        return "".to_string();
    }

    // Use SIMD optimization if available - round up to 16-byte boundary
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("ssse3") {
            // Round up to nearest 16-byte boundary for optimal SIMD processing
            let simd_len = (len + 15) & !15;
            let mut buf: Vec<u8> = Vec::with_capacity(simd_len);

            rng.fill_bytes(unsafe {
                // SAFETY: We're transmuting the spare capacity of the vector to a mutable slice of u8.
                // spare_capacity_mut() returns uninitialized space, and we immediately fill it with
                // rand_bytes, which initializes it before we use it.
                std::mem::transmute::<&mut [std::mem::MaybeUninit<u8>], &mut [u8]>(
                    &mut buf.spare_capacity_mut()[..simd_len],
                )
            });
            // SAFETY: We just initialized `simd_len` bytes in the buffer via rand_bytes call above.
            unsafe { buf.set_len(simd_len) };

            // Process all bytes with SIMD (no scalar fallback needed)
            unsafe { rand_string_simd_ssse3(&mut buf) };

            // Truncate to requested length
            buf.truncate(len);

            return unsafe { String::from_utf8_unchecked(buf) };
        }
    }

    // Use NEON optimization on ARM if available
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // Round up to nearest 16-byte boundary for optimal SIMD processing
            let simd_len = (len + 15) & !15;
            let mut buf: Vec<u8> = Vec::with_capacity(simd_len);

            rng.fill_bytes(unsafe {
                // SAFETY: We're transmuting the spare capacity of the vector to a mutable slice of u8.
                // spare_capacity_mut() returns uninitialized space, and we immediately fill it with
                // rand_bytes, which initializes it before we use it.
                std::mem::transmute::<&mut [std::mem::MaybeUninit<u8>], &mut [u8]>(
                    &mut buf.spare_capacity_mut()[..simd_len],
                )
            });
            // SAFETY: We just initialized `simd_len` bytes in the buffer via rand_bytes call above.
            unsafe { buf.set_len(simd_len) };

            // Process all bytes with NEON SIMD (no scalar fallback needed)
            unsafe { rand_string_simd_neon(&mut buf) };

            // Truncate to requested length
            buf.truncate(len);

            return unsafe { String::from_utf8_unchecked(buf) };
        }
    }

    // Fallback to scalar implementation for other architectures or when SIMD is unavailable
    let mut buf: Vec<u8> = Vec::with_capacity(len);
    rng.fill_bytes(unsafe {
        // SAFETY: We're transmuting the spare capacity of the vector to a mutable slice of u8.
        // spare_capacity_mut() returns uninitialized space, and we immediately fill it with
        // rand_bytes, which initializes it before we use it.
        std::mem::transmute::<&mut [std::mem::MaybeUninit<u8>], &mut [u8]>(buf.spare_capacity_mut())
    });
    // SAFETY: We just initialized `len` bytes in the buffer via rand_bytes call above.
    unsafe { buf.set_len(len) };

    rand_string_scalar(&mut buf);
    // SAFETY: the charset provided is valid UTF-8
    unsafe { String::from_utf8_unchecked(buf) }
}

/// Scalar implementation of charset lookup for random string generation.
#[inline]
fn rand_string_scalar(buf: &mut [u8]) {
    const CHARS: [u8; 64] = [
        b'A', b'B', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'J', b'K', b'L', b'M', b'N', b'O',
        b'P', b'Q', b'R', b'S', b'T', b'U', b'V', b'W', b'X', b'Y', b'Z', b'a', b'b', b'c', b'd',
        b'e', b'f', b'g', b'h', b'i', b'j', b'k', b'l', b'm', b'n', b'o', b'p', b'q', b'r', b's',
        b't', b'u', b'v', b'w', b'x', b'y', b'z', b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7',
        b'8', b'9', b'-', b'_',
    ];

    for b in buf.iter_mut() {
        *b = CHARS[(*b >> 2) as usize];
    }
}

/// SIMD implementation using SSSE3 instructions for charset lookup.
/// Processes 16 bytes at a time using shuffle instructions.
///
/// # Safety
/// The buffer length must be a multiple of 16 bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[inline]
unsafe fn rand_string_simd_ssse3(buf: &mut [u8]) {
    // SAFETY: This entire function is marked unsafe and requires SSSE3.
    // All SIMD operations are safe when the target feature is enabled.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        // Our charset: A-Z (indices 0-25), a-z (26-51), 0-9 (52-61), - (62), _ (63)
        // We need to map 6-bit values (0-63) to these characters

        // Split the 64-character table into 4 lookup tables of 16 characters each
        // LUT 0: indices 0-15 => A-P
        let lut0 = _mm_setr_epi8(
            b'A' as i8, b'B' as i8, b'C' as i8, b'D' as i8, b'E' as i8, b'F' as i8, b'G' as i8,
            b'H' as i8, b'I' as i8, b'J' as i8, b'K' as i8, b'L' as i8, b'M' as i8, b'N' as i8,
            b'O' as i8, b'P' as i8,
        );

        // LUT 1: indices 16-31 => Q-Z, a-e
        let lut1 = _mm_setr_epi8(
            b'Q' as i8, b'R' as i8, b'S' as i8, b'T' as i8, b'U' as i8, b'V' as i8, b'W' as i8,
            b'X' as i8, b'Y' as i8, b'Z' as i8, b'a' as i8, b'b' as i8, b'c' as i8, b'd' as i8,
            b'e' as i8, b'f' as i8,
        );

        // LUT 2: indices 32-47 => g-v
        let lut2 = _mm_setr_epi8(
            b'g' as i8, b'h' as i8, b'i' as i8, b'j' as i8, b'k' as i8, b'l' as i8, b'm' as i8,
            b'n' as i8, b'o' as i8, b'p' as i8, b'q' as i8, b'r' as i8, b's' as i8, b't' as i8,
            b'u' as i8, b'v' as i8,
        );

        // LUT 3: indices 48-63 => w-z, 0-9, -, _
        let lut3 = _mm_setr_epi8(
            b'w' as i8, b'x' as i8, b'y' as i8, b'z' as i8, b'0' as i8, b'1' as i8, b'2' as i8,
            b'3' as i8, b'4' as i8, b'5' as i8, b'6' as i8, b'7' as i8, b'8' as i8, b'9' as i8,
            b'-' as i8, b'_' as i8,
        );

        let len = buf.len();
        let mut i = 0;

        debug_assert!(
            buf.as_ptr().addr() & 15 == 0,
            "pointer must be 16 byte aligned"
        );

        // Process all bytes in 16-byte chunks (buffer length is guaranteed to be a multiple of 16)
        while i < len {
            // Load 16 random bytes
            let input = _mm_load_si128(buf.as_ptr().add(i) as *const __m128i);

            // Extract 6 bits per byte (shift right by 2), giving us values 0-63
            let indices = _mm_srli_epi16(input, 2);
            let indices = _mm_and_si128(indices, _mm_set1_epi8(0x3f));

            // Extract the high 2 bits (bits 4-5) to determine which LUT to use
            // and the low 4 bits (bits 0-3) to index within the LUT
            let hi_bits = _mm_srli_epi16(indices, 4);
            let hi_bits = _mm_and_si128(hi_bits, _mm_set1_epi8(0x03));
            let lo_bits = _mm_and_si128(indices, _mm_set1_epi8(0x0f));

            // Perform lookups in all 4 tables
            let result0 = _mm_shuffle_epi8(lut0, lo_bits);
            let result1 = _mm_shuffle_epi8(lut1, lo_bits);
            let result2 = _mm_shuffle_epi8(lut2, lo_bits);
            let result3 = _mm_shuffle_epi8(lut3, lo_bits);

            // Create masks for each LUT based on hi_bits value
            let mask0 = _mm_cmpeq_epi8(hi_bits, _mm_setzero_si128());
            let mask1 = _mm_cmpeq_epi8(hi_bits, _mm_set1_epi8(1));
            let mask2 = _mm_cmpeq_epi8(hi_bits, _mm_set1_epi8(2));
            let mask3 = _mm_cmpeq_epi8(hi_bits, _mm_set1_epi8(3));

            // Blend results based on masks
            let mut result = _mm_and_si128(mask0, result0);
            result = _mm_or_si128(result, _mm_and_si128(mask1, result1));
            result = _mm_or_si128(result, _mm_and_si128(mask2, result2));
            result = _mm_or_si128(result, _mm_and_si128(mask3, result3));

            // Store result
            _mm_storeu_si128(buf.as_mut_ptr().add(i) as *mut __m128i, result);

            i += 16;
        }
    }
}

/// SIMD implementation using ARM NEON instructions for charset lookup.
/// Processes 16 bytes at a time using table lookup instructions.
///
/// # Safety
/// The buffer length must be a multiple of 16 bytes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn rand_string_simd_neon(buf: &mut [u8]) {
    // SAFETY: This entire function is marked unsafe and requires NEON.
    // All SIMD operations are safe when the target feature is enabled.
    unsafe {
        #[cfg(target_arch = "aarch64")]
        use std::arch::aarch64::*;

        // Our charset: A-Z (indices 0-25), a-z (26-51), 0-9 (52-61), - (62), _ (63)
        // We need to map 6-bit values (0-63) to these characters

        // Split the 64-character table into 4 lookup tables of 16 characters each
        // LUT 0: indices 0-15 => A-P
        let lut0 = [
            b'A', b'B', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'J', b'K', b'L', b'M', b'N',
            b'O', b'P',
        ];

        // LUT 1: indices 16-31 => Q-Z, a-e
        let lut1 = [
            b'Q', b'R', b'S', b'T', b'U', b'V', b'W', b'X', b'Y', b'Z', b'a', b'b', b'c', b'd',
            b'e', b'f',
        ];

        // LUT 2: indices 32-47 => g-v
        let lut2 = [
            b'g', b'h', b'i', b'j', b'k', b'l', b'm', b'n', b'o', b'p', b'q', b'r', b's', b't',
            b'u', b'v',
        ];

        // LUT 3: indices 48-63 => w-z, 0-9, -, _
        let lut3 = [
            b'w', b'x', b'y', b'z', b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9',
            b'-', b'_',
        ];

        let lut0_vec = vld1q_u8(lut0.as_ptr());
        let lut1_vec = vld1q_u8(lut1.as_ptr());
        let lut2_vec = vld1q_u8(lut2.as_ptr());
        let lut3_vec = vld1q_u8(lut3.as_ptr());

        let len = buf.len();
        let mut i = 0;

        // Process all bytes in 16-byte chunks (buffer length is guaranteed to be a multiple of 16)
        while i < len {
            // Load 16 random bytes
            let input = vld1q_u8(buf.as_ptr().add(i));

            // Extract 6 bits per byte (shift right by 2), giving us values 0-63
            let indices = vshrq_n_u8(input, 2);
            let indices = vandq_u8(indices, vdupq_n_u8(0x3f));

            // Extract the high 2 bits (bits 4-5) to determine which LUT to use
            // and the low 4 bits (bits 0-3) to index within the LUT
            let hi_bits = vshrq_n_u8(indices, 4);
            let hi_bits = vandq_u8(hi_bits, vdupq_n_u8(0x03));
            let lo_bits = vandq_u8(indices, vdupq_n_u8(0x0f));

            // Perform lookups in all 4 tables using NEON table lookup
            let result0 = vqtbl1q_u8(lut0_vec, lo_bits);
            let result1 = vqtbl1q_u8(lut1_vec, lo_bits);
            let result2 = vqtbl1q_u8(lut2_vec, lo_bits);
            let result3 = vqtbl1q_u8(lut3_vec, lo_bits);

            // Create masks for each LUT based on hi_bits value
            let zero_vec = vdupq_n_u8(0);
            let mask0 = vceqq_u8(hi_bits, zero_vec);
            let mask1 = vceqq_u8(hi_bits, vdupq_n_u8(1));
            let mask2 = vceqq_u8(hi_bits, vdupq_n_u8(2));
            let mask3 = vceqq_u8(hi_bits, vdupq_n_u8(3));

            // Blend results based on masks using bitwise select
            // vbslq_u8(mask, a, b) returns: (mask & a) | (!mask & b)
            let mut result = vbslq_u8(mask0, result0, zero_vec);
            result = vorrq_u8(result, vandq_u8(mask1, result1));
            result = vorrq_u8(result, vandq_u8(mask2, result2));
            result = vorrq_u8(result, vandq_u8(mask3, result3));

            // Store result
            vst1q_u8(buf.as_mut_ptr().add(i), result);

            i += 16;
        }
    }
}

/// A cryptographically secure random number generator backed by AWS-LC.
///
/// This RNG uses AWS-LC's `RAND_bytes` function, which provides cryptographically
/// strong random numbers suitable for security-sensitive applications. AWS-LC is
/// a general-purpose cryptographic library maintained by Amazon Web Services.
///
/// # Features
///
/// This type is only available when the `aws-lc` feature is enabled.
///
/// # Implementation Details
///
/// - Implements `rand_core::TryRng` and `rand_core::TryCryptoRng`
/// - All operations are infallible in practice (AWS-LC aborts on failure)
/// - Zero-cost abstraction with no state
/// - Thread-safe and can be safely cloned
///
/// # Examples
///
/// ```ignore
/// use rand_utils::AwsLcRng;
/// use rand_core::TryRng;
///
/// let mut rng = AwsLcRng;
/// let random_u64 = rng.try_next_u64().unwrap();
///
/// let mut buffer = [0u8; 32];
/// rng.try_fill_bytes(&mut buffer).unwrap();
/// ```
///
/// # Performance
///
/// This RNG is optimized for cryptographic security rather than raw speed.
/// For non-cryptographic use cases, consider using a faster PRNG.
#[cfg(feature = "aws-lc")]
#[derive(Debug, Clone, Copy, Default)]
pub struct AwsLcRng;

#[cfg(feature = "aws-lc")]
impl rand_core::TryRng for AwsLcRng {
    type Error = core::convert::Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        let mut buf = core::mem::MaybeUninit::<[u8; 4]>::uninit();
        unsafe { aws_lc_sys::RAND_bytes(buf.as_mut_ptr() as *mut u8, 4) };
        Ok(u32::from_ne_bytes(unsafe { buf.assume_init() }))
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let mut buf = core::mem::MaybeUninit::<[u8; 8]>::uninit();
        unsafe { aws_lc_sys::RAND_bytes(buf.as_mut_ptr() as *mut u8, 8) };
        Ok(u64::from_ne_bytes(unsafe { buf.assume_init() }))
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        // This is infallible, AWS-LC aborts if RAND_bytes fails
        unsafe { aws_lc_sys::RAND_bytes(dst.as_mut_ptr(), dst.len()) };
        Ok(())
    }
}

#[cfg(feature = "aws-lc")]
impl rand_core::TryCryptoRng for AwsLcRng {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rand_string() {
        let mut rng = rand::rng();
        // Test empty string
        assert_eq!(rand_string(&mut rng, 0).len(), 0);

        // Test string of specific length
        let s = rand_string(&mut rng, 16);
        assert_eq!(s.len(), 16);

        // Test that two strings are different
        let s1 = rand_string(&mut rng, 32);
        let s2 = rand_string(&mut rng, 32);
        assert_ne!(s1, s2);

        // Test string contents are from valid charset
        const VALID_CHARS: &[u8] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
        for c in s1.bytes() {
            assert!(VALID_CHARS.contains(&c));
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_rand_string_simd_consistency() {
        // Test that SIMD and scalar implementations produce the same output
        // for the same input random bytes

        const TEST_SIZES: &[usize] = &[16, 32, 48, 64, 80, 128, 256];

        for &size in TEST_SIZES {
            use rand::Rng;

            let mut rng = rand::rng();
            // Generate random bytes (size is always a multiple of 16)
            let mut random_bytes = vec![0u8; size];
            rng.fill_bytes(&mut random_bytes);

            // Process with scalar
            let mut scalar_result = random_bytes.clone();
            rand_string_scalar(&mut scalar_result);

            // Process with SIMD (if available)
            let mut simd_result = random_bytes.clone();
            if is_x86_feature_detected!("ssse3") {
                unsafe { rand_string_simd_ssse3(&mut simd_result) };
            }

            // Compare results
            if is_x86_feature_detected!("ssse3") {
                assert_eq!(
                    scalar_result, simd_result,
                    "SIMD and scalar results differ for size {size}"
                );
            }

            // Verify all bytes are valid charset characters
            const VALID_CHARS: &[u8] =
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
            for &byte in &scalar_result {
                assert!(
                    VALID_CHARS.contains(&byte),
                    "Invalid character: {}",
                    byte as char
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_rand_string_simd_neon_consistency() {
        // Test that SIMD and scalar implementations produce the same output
        // for the same input random bytes

        const TEST_SIZES: &[usize] = &[16, 32, 48, 64, 80, 128, 256];

        for &size in TEST_SIZES {
            use rand::Rng;

            let mut rng = rand::rng();
            // Generate random bytes (size is always a multiple of 16)
            let mut random_bytes = vec![0u8; size];
            rng.fill_bytes(&mut random_bytes);

            // Process with scalar
            let mut scalar_result = random_bytes.clone();
            rand_string_scalar(&mut scalar_result);

            // Process with SIMD (if available)
            let mut simd_result = random_bytes.clone();
            if std::arch::is_aarch64_feature_detected!("neon") {
                unsafe { rand_string_simd_neon(&mut simd_result) };
            }

            // Compare results
            if std::arch::is_aarch64_feature_detected!("neon") {
                assert_eq!(
                    scalar_result, simd_result,
                    "SIMD and scalar results differ for size {}",
                    size
                );
            }

            // Verify all bytes are valid charset characters
            const VALID_CHARS: &[u8] =
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
            for &byte in &scalar_result {
                assert!(
                    VALID_CHARS.contains(&byte),
                    "Invalid character: {}",
                    byte as char
                );
            }
        }
    }

    #[test]
    fn test_rand_string_lengths() {
        let mut rng = rand::rng();
        // Test that rand_string works correctly for various lengths including
        // those that aren't multiples of 16
        const TEST_LENGTHS: &[usize] =
            &[1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 250];

        for &len in TEST_LENGTHS {
            let s = rand_string(&mut rng, len);
            assert_eq!(s.len(), len, "String length should match requested length");

            // Verify all characters are from valid charset
            const VALID_CHARS: &[u8] =
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
            for c in s.bytes() {
                assert!(VALID_CHARS.contains(&c), "Invalid character: {}", c as char);
            }
        }
    }

    #[test]
    #[cfg(feature = "aws-lc")]
    fn test_aws_lc_rng_basic() {
        use rand_core::TryRng;

        let mut rng = AwsLcRng;

        // Test try_next_u32
        let val1 = rng.try_next_u32().unwrap();
        let val2 = rng.try_next_u32().unwrap();
        // Extremely unlikely to be equal with cryptographic RNG
        assert_ne!(
            val1, val2,
            "Two consecutive random u32s should not be equal"
        );

        // Test try_next_u64
        let val1 = rng.try_next_u64().unwrap();
        let val2 = rng.try_next_u64().unwrap();
        assert_ne!(
            val1, val2,
            "Two consecutive random u64s should not be equal"
        );
    }

    #[test]
    #[cfg(feature = "aws-lc")]
    fn test_aws_lc_rng_fill_bytes() {
        use rand_core::TryRng;

        let mut rng = AwsLcRng;
        let mut buf1 = [0u8; 32];
        let mut buf2 = [0u8; 32];

        rng.try_fill_bytes(&mut buf1).unwrap();
        rng.try_fill_bytes(&mut buf2).unwrap();

        // Verify buffers were filled (not all zeros)
        assert!(
            buf1.iter().any(|&x| x != 0),
            "Buffer should contain non-zero bytes"
        );
        assert!(
            buf2.iter().any(|&x| x != 0),
            "Buffer should contain non-zero bytes"
        );

        // Verify buffers are different
        assert_ne!(
            buf1, buf2,
            "Two consecutive fills should produce different results"
        );
    }

    #[test]
    #[cfg(feature = "aws-lc")]
    fn test_aws_lc_rng_various_sizes() {
        use rand_core::TryRng;

        let mut rng = AwsLcRng;

        // Test various buffer sizes
        for size in [1, 7, 16, 31, 64, 100, 256, 1024] {
            let mut buf = vec![0u8; size];
            rng.try_fill_bytes(&mut buf).unwrap();

            // Verify buffer was filled with some randomness
            // For buffers > 1 byte, at least one byte should be non-zero
            if size > 1 {
                assert!(
                    buf.iter().any(|&x| x != 0),
                    "Buffer of size {} should contain non-zero bytes",
                    size
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "aws-lc")]
    fn test_aws_lc_rng_copy() {
        use rand_core::TryRng;

        let mut rng1 = AwsLcRng;
        let mut rng2 = rng1; // Copy, not clone

        // Both should produce random values (though they won't be the same
        // since there's no shared state)
        let val1 = rng1.try_next_u64().unwrap();
        let val2 = rng2.try_next_u64().unwrap();

        // Just verify both work - they're independent
        assert_ne!(val1, 0);
        assert_ne!(val2, 0);
    }

    #[test]
    #[cfg(feature = "aws-lc")]
    fn test_aws_lc_rng_with_rand_string() {
        // Test that AwsLcRng works with our rand_string function
        let mut rng = AwsLcRng;

        let s1 = rand_string(&mut rng, 32);
        let s2 = rand_string(&mut rng, 32);

        assert_eq!(s1.len(), 32);
        assert_eq!(s2.len(), 32);
        assert_ne!(s1, s2, "Two random strings should be different");

        // Verify all characters are from valid charset
        const VALID_CHARS: &[u8] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
        for c in s1.bytes() {
            assert!(VALID_CHARS.contains(&c));
        }
    }

    #[test]
    #[cfg(feature = "aws-lc")]
    fn test_aws_lc_rng_empty_buffer() {
        use rand_core::TryRng;

        let mut rng = AwsLcRng;
        let mut buf = [];

        // Should handle empty buffer without error
        rng.try_fill_bytes(&mut buf).unwrap();
    }
}
