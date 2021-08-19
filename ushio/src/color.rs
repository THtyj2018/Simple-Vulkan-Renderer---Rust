//! Color

use std::ops::{Index, IndexMut};

use ushio_geom::{IntoSTD140, STD140Vec4, ScalarArray, Vec4};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const fn rgb(r: f32, g: f32, b: f32) -> Color {
        Color { r, g, b, a: 1.0 }
    }

    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Color {
        Color { r, g, b, a }
    }

    pub const fn black() -> Color {
        Color::rgb(0.0, 0.0, 0.0)
    }

    pub const fn white() -> Color {
        Color::rgb(1.0, 1.0, 1.0)
    }

    pub const fn red() -> Color {
        Color::rgb(1.0, 0.0, 0.0)
    }

    pub const fn green() -> Color {
        Color::rgb(0.0, 1.0, 0.0)
    }

    pub const fn blue() -> Color {
        Color::rgb(0.0, 0.0, 1.0)
    }

    pub const fn yellow() -> Color {
        Color::rgb(1.0, 1.0, 0.0)
    }

    pub const fn cyan() -> Color {
        Color::rgb(0.0, 1.0, 1.0)
    }

    pub const fn magenta() -> Color {
        Color::rgb(1.0, 0.0, 1.0)
    }
}

impl ScalarArray for Color {
    type Scalar = f32;

    fn scalar_count() -> usize {
        4
    }
}

impl Default for Color {
    fn default() -> Self {
        Color::black()
    }
}

impl Index<usize> for Color {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < Self::scalar_count());
        unsafe { self.as_ptr().offset(index as isize).as_ref().unwrap() }
    }
}

impl IndexMut<usize> for Color {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < Self::scalar_count());
        unsafe { self.as_mut_ptr().offset(index as isize).as_mut().unwrap() }
    }
}

impl Into<(f32, f32, f32, f32)> for Color {
    fn into(self) -> (f32, f32, f32, f32) {
        (self.r, self.g, self.b, self.a)
    }
}

impl Into<[f32; 4]> for Color {
    fn into(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

impl Into<Vec4> for Color {
    fn into(self) -> Vec4 {
        Vec4::new(self.r, self.g, self.b, self.a)
    }
}

impl From<[f32; 4]> for Color {
    fn from(a: [f32; 4]) -> Self {
        Color::rgba(a[0], a[1], a[2], a[3])
    }
}

impl From<Vec4> for Color {
    fn from(v: Vec4) -> Self {
        Color::rgba(v.x, v.w, v.z, v.w)
    }
}

impl IntoSTD140 for Color {
    type Output = STD140Vec4;

    fn into_std140(&self) -> Self::Output {
        let v: Vec4 = self.clone().into();
        v.into_std140()
    }
}
