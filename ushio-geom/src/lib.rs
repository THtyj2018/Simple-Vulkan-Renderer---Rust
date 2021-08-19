//! Geometric Math

mod mat2;
mod mat3;
mod mat4;
mod quat;
mod vec2;
mod vec3;
mod vec4;

use std::{intrinsics::copy_nonoverlapping, mem::size_of};

pub use mat2::*;
pub use mat3::*;
pub use mat4::*;
pub use quat::*;
pub use vec2::*;
pub use vec3::*;
pub use vec4::*;

pub trait ScalarArray: Sized {
    type Scalar;

    fn scalar_count() -> usize;

    fn as_ptr(&self) -> *const Self::Scalar {
        self as *const Self as *const Self::Scalar
    }

    fn as_mut_ptr(&mut self) -> *mut Self::Scalar {
        self as *mut Self as *mut Self::Scalar
    }
}

pub trait STD140 {}

pub trait IntoSTD140: Sized + std::fmt::Debug + Clone + Copy + PartialEq + Default {
    type Output: Sized + STD140 + std::fmt::Debug + Clone + PartialEq + Default;

    fn into_std140(&self) -> Self::Output;

    fn into_std140_bytes(&self) -> Vec<u8> {
        let std140 = self.into_std140();
        let p = &std140 as *const Self::Output as *const u8;
        let mut v = vec![];
        v.resize(size_of::<Self::Output>(), 0);
        unsafe {
            copy_nonoverlapping(p, v.as_mut_ptr(), size_of::<Self::Output>());
        }
        v
    }

    const SIZE: usize = size_of::<Self::Output>();
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct STD140ArrayElem<T>(pub T);

pub trait IntoSTD140ArrayElem: IntoSTD140 {
    fn into_std140_array_elem(&self) -> STD140ArrayElem<Self>;
}

impl<T: IntoSTD140> IntoSTD140ArrayElem for T {
    fn into_std140_array_elem(&self) -> STD140ArrayElem<Self> {
        STD140ArrayElem(self.clone())
    }
}

#[repr(C, align(4))]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct STD140Float(pub f32);

impl STD140 for STD140Float {}

impl IntoSTD140 for f32 {
    type Output = STD140Float;

    fn into_std140(&self) -> Self::Output {
        STD140Float(*self)
    }
}

#[repr(C, align(4))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct STD140UInt(pub u32);

impl STD140 for STD140UInt {}

impl IntoSTD140 for u32 {
    type Output = STD140UInt;

    fn into_std140(&self) -> Self::Output {
        STD140UInt(*self)
    }
}

#[repr(C, align(4))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct STD140Int(pub i32);

impl STD140 for STD140Int {}

impl IntoSTD140 for i32 {
    type Output = STD140Int;

    fn into_std140(&self) -> Self::Output {
        STD140Int(*self)
    }
}

#[repr(C, align(4))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct STD140Bool(pub bool);

impl STD140 for STD140Bool {}

impl IntoSTD140 for bool {
    type Output = STD140Bool;

    fn into_std140(&self) -> Self::Output {
        STD140Bool(*self)
    }
}

#[macro_export]
macro_rules! std140_structs {
    (
        $(
            $v:vis struct $name:ident uniform $std140:ident {
                $(
                    $fv:vis $f:ident: $t:tt,
                )*
            }
        )*
    ) => {
        $(
            #[derive(Debug, Clone, Copy, PartialEq, Default)]
            $v struct $name {
                   $(
                    $fv $f: $t,
                )*
            }

            #[repr(C, packed(4))]
            #[derive(Debug, Clone, Copy, PartialEq, Default)]
            $v struct $std140 {
                $(
                    $fv $f: std140_structs!($t),
                )*
            }

            impl STD140 for $std140 {}

            impl IntoSTD140 for $name {
                type Output = $std140;

                fn into_std140(&self) -> Self::Output {
                    $std140 {
                        $(
                            $f: std140_structs!($t, self.$f),
                        )*
                    }
                }
            }
        )*
    };

    ([$at:ty; $len:literal]) => {
        [STD140ArrayElem<$at>; $len]
    };

    ($t:ty) => {
        <$t as IntoSTD140>::Output
    };

    ([$at:ty; $len:literal], $val:expr) => {
        unsafe {
            let mut _tmp_a: [STD140ArrayElem<$at>; $len] = std::mem::MaybeUninit::uninit().assume_init();
            for _tmp_i in 0..$len {
                _tmp_a[_tmp_i] = $val[_tmp_i].into_std140_array_elem();
            }
            _tmp_a
        }
    };

    ($t:ty, $val:expr) => {
        $val.into_std140()
    };
}
