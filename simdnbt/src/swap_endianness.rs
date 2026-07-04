use std::mem;

// Plain loops — LLVM on Linux auto-vectorizes these better than manual SIMD
// swizzles. For swap_endianness specifically, chunks_exact_mut gives the
// compiler a clean loop with known iteration bounds, which the loop
// vectorizer handles well on both x86-64 and AArch64.

mod private {
    pub trait Sealed {}

    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}
pub trait SwappableNumber: private::Sealed {}
impl<T: private::Sealed> SwappableNumber for T {}

#[inline]
fn swap_endianness_16bit(bytes: &mut [u8], num: usize) {
    for chunk in bytes[..num * 2].chunks_exact_mut(2) {
        chunk.swap(0, 1);
    }
}

#[inline]
fn swap_endianness_32bit(bytes: &mut [u8], num: usize) {
    for chunk in bytes[..num * 4].chunks_exact_mut(4) {
        chunk.swap(0, 3);
        chunk.swap(1, 2);
    }
}

#[inline]
fn swap_endianness_64bit(bytes: &mut [u8], num: usize) {
    for chunk in bytes[..num * 8].chunks_exact_mut(8) {
        chunk.swap(0, 7);
        chunk.swap(1, 6);
        chunk.swap(2, 5);
        chunk.swap(3, 4);
    }
}

/// Swap the endianness of the given array (unless we're on a big-endian system)
/// in-place depending on the width of the given type.
fn swap_endianness_from_type<T: SwappableNumber>(items: &mut [u8]) {
    let item_width = mem::size_of::<T>();
    let length = items.len() / item_width;

    if cfg!(target_endian = "little") {
        match item_width {
            2 => swap_endianness_16bit(items, length),
            4 => swap_endianness_32bit(items, length),
            8 => swap_endianness_64bit(items, length),
            _ => panic!("unsupported size of type"),
        }
    }
}

/// Swaps the endianness of the given data and return it as a `Vec<u8>`.
#[inline]
pub fn swap_endianness_as_u8<T: SwappableNumber>(data: &[u8]) -> Vec<u8> {
    let mut items = data.to_vec();
    swap_endianness_from_type::<T>(&mut items);

    items
}

#[inline]
pub fn swap_endianness<T: SwappableNumber>(data: &[u8]) -> Vec<T> {
    let width_of_t = mem::size_of::<T>();
    let length_of_vec_t = data.len() / width_of_t;

    // the data must be a multiple of the item width, otherwise it's UB
    assert_eq!(data.len() % width_of_t, 0);

    // have the vec be of T initially so it's aligned
    let mut vec_t = Vec::<T>::with_capacity(length_of_vec_t);
    let mut vec_u8: Vec<u8> = {
        let ptr = vec_t.as_mut_ptr() as *mut u8;
        mem::forget(vec_t);
        // SAFETY: the new capacity is correct since we checked that data.len() is a
        // multiple of width_of_t
        unsafe { Vec::from_raw_parts(ptr, 0, data.len()) }
    };
    vec_u8.extend_from_slice(data);

    swap_endianness_from_type::<T>(&mut vec_u8);

    // now convert our Vec<u8> back to Vec<T>

    let ptr = vec_u8.as_mut_ptr() as *mut T;
    mem::forget(vec_u8);
    // SAFETY: The length won't be greater than the length of the original data
    unsafe { Vec::from_raw_parts(ptr, length_of_vec_t, length_of_vec_t) }
}

#[cfg(test)]
// swap_endianness_as_u8 only does anything on LE systems, so otherwise it'll
// error
#[cfg(target_endian = "little")]
mod tests {
    use super::*;

    #[test]
    fn test_swap_endianness_u16() {
        assert_eq!(
            swap_endianness_as_u8::<u16>(&[1, 2, 3, 4, 5, 6, 7, 8]),
            [2, 1, 4, 3, 6, 5, 8, 7]
        );
    }
    #[test]
    fn test_swap_endianness_u32() {
        assert_eq!(
            swap_endianness_as_u8::<u32>(&[1, 2, 3, 4, 5, 6, 7, 8]),
            [4, 3, 2, 1, 8, 7, 6, 5]
        );
    }
    #[test]
    fn test_swap_endianness_u64() {
        assert_eq!(
            swap_endianness_as_u8::<u64>(&[1, 2, 3, 4, 5, 6, 7, 8]),
            [8, 7, 6, 5, 4, 3, 2, 1]
        );
    }

    #[test]
    fn test_swap_endianness_u64_vec() {
        assert_eq!(
            swap_endianness::<u64>(&[1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1]),
            vec![
                u64::from_le_bytes([8, 7, 6, 5, 4, 3, 2, 1]),
                u64::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 8])
            ]
        );
    }
}
