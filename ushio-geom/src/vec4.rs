//! 4D Vector

use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::{IntoSTD140, ScalarArray, Vec2, Vec3, STD140};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct STD140Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
        Vec4 { x, y, z, w }
    }

    pub const fn from_xyz(xyz: Vec3, w: f32) -> Vec4 {
        Self::new(xyz.x, xyz.y, xyz.z, w)
    }

    pub const fn one() -> Vec4 {
        Self::new(1.0, 1.0, 1.0, 1.0)
    }

    pub const fn zero() -> Vec4 {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    pub const fn identity() -> Vec4 {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    pub fn dot(self, rhs: Vec4) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    pub const fn xyz(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    pub const fn xyw(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.w)
    }

    pub const fn xzw(self) -> Vec3 {
        Vec3::new(self.x, self.z, self.w)
    }

    pub const fn yzw(self) -> Vec3 {
        Vec3::new(self.y, self.z, self.w)
    }

    pub const fn zwx(self) -> Vec3 {
        Vec3::new(self.z, self.w, self.x)
    }

    pub const fn wxy(self) -> Vec3 {
        Vec3::new(self.w, self.x, self.y)
    }

    pub const fn xy(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }
}

impl Default for Vec4 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Default for STD140Vec4 {
    fn default() -> Self {
        Vec4::default().into_std140()
    }
}

impl Into<(f32, f32, f32, f32)> for Vec4 {
    fn into(self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.z, self.w)
    }
}

impl Into<[f32; 4]> for Vec4 {
    fn into(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }
}

impl From<[f32; 4]> for Vec4 {
    fn from(a: [f32; 4]) -> Self {
        Vec4::new(a[0], a[1], a[2], a[3])
    }
}

impl STD140 for STD140Vec4 {}

impl IntoSTD140 for Vec4 {
    type Output = STD140Vec4;

    fn into_std140(&self) -> Self::Output {
        STD140Vec4 {
            x: self.x,
            y: self.y,
            z: self.z,
            w: self.w,
        }
    }
}

impl ScalarArray for Vec4 {
    type Scalar = f32;

    fn scalar_count() -> usize {
        4
    }
}

impl Index<usize> for Vec4 {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < Self::scalar_count());
        unsafe { self.as_ptr().offset(index as isize).as_ref().unwrap() }
    }
}

impl IndexMut<usize> for Vec4 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < Self::scalar_count());
        unsafe { self.as_mut_ptr().offset(index as isize).as_mut().unwrap() }
    }
}

impl Add<f32> for Vec4 {
    type Output = Vec4;

    fn add(self, rhs: f32) -> Self::Output {
        Self::new(self.x + rhs, self.y + rhs, self.z + rhs, self.w + rhs)
    }
}

impl Add<Vec4> for Vec4 {
    type Output = Vec4;

    fn add(self, rhs: Vec4) -> Self::Output {
        Self::new(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
            self.w + rhs.w,
        )
    }
}

impl Add<Vec4> for f32 {
    type Output = Vec4;

    fn add(self, rhs: Vec4) -> Self::Output {
        Vec4::new(self + rhs.x, self + rhs.y, self + rhs.z, self + rhs.w)
    }
}

impl AddAssign<f32> for Vec4 {
    fn add_assign(&mut self, rhs: f32) {
        self.x += rhs;
        self.y += rhs;
        self.z += rhs;
        self.w += rhs;
    }
}

impl AddAssign<Vec4> for Vec4 {
    fn add_assign(&mut self, rhs: Vec4) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
        self.w += rhs.w;
    }
}

impl Sub<f32> for Vec4 {
    type Output = Vec4;

    fn sub(self, rhs: f32) -> Self::Output {
        Self::new(self.x - rhs, self.y - rhs, self.z - rhs, self.w - rhs)
    }
}

impl Sub<Vec4> for Vec4 {
    type Output = Vec4;

    fn sub(self, rhs: Vec4) -> Self::Output {
        Self::new(
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
            self.w - rhs.w,
        )
    }
}

impl Sub<Vec4> for f32 {
    type Output = Vec4;

    fn sub(self, rhs: Vec4) -> Self::Output {
        Vec4::new(self - rhs.x, self - rhs.y, self - rhs.z, self - rhs.w)
    }
}

impl SubAssign<f32> for Vec4 {
    fn sub_assign(&mut self, rhs: f32) {
        self.x -= rhs;
        self.y -= rhs;
        self.z -= rhs;
        self.w -= rhs;
    }
}

impl SubAssign<Vec4> for Vec4 {
    fn sub_assign(&mut self, rhs: Vec4) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
        self.w -= rhs.w;
    }
}

impl Neg for Vec4 {
    type Output = Vec4;

    fn neg(self) -> Self::Output {
        Vec4::new(-self.x, -self.y, -self.z, -self.w)
    }
}

impl Mul<f32> for Vec4 {
    type Output = Vec4;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
    }
}

impl Mul<Vec4> for Vec4 {
    type Output = Vec4;

    fn mul(self, rhs: Vec4) -> Self::Output {
        Self::new(
            self.x * rhs.x,
            self.y * rhs.y,
            self.z * rhs.z,
            self.w * rhs.w,
        )
    }
}

impl Mul<Vec4> for f32 {
    type Output = Vec4;

    fn mul(self, rhs: Vec4) -> Self::Output {
        Vec4::new(self * rhs.x, self * rhs.y, self * rhs.z, self * rhs.w)
    }
}

impl MulAssign<f32> for Vec4 {
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
        self.w *= rhs;
    }
}

impl MulAssign<Vec4> for Vec4 {
    fn mul_assign(&mut self, rhs: Vec4) {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self.z *= rhs.z;
        self.w *= rhs.w;
    }
}

impl Div<f32> for Vec4 {
    type Output = Vec4;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs, self.w / rhs)
    }
}

impl Div<Vec4> for Vec4 {
    type Output = Vec4;

    fn div(self, rhs: Vec4) -> Self::Output {
        Self::new(
            self.x / rhs.x,
            self.y / rhs.y,
            self.z / rhs.z,
            self.w / rhs.w,
        )
    }
}

impl Div<Vec4> for f32 {
    type Output = Vec4;

    fn div(self, rhs: Vec4) -> Self::Output {
        Vec4::new(self / rhs.x, self / rhs.y, self / rhs.z, self / rhs.w)
    }
}

impl DivAssign<f32> for Vec4 {
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
        self.w /= rhs;
    }
}

impl DivAssign<Vec4> for Vec4 {
    fn div_assign(&mut self, rhs: Vec4) {
        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
        self.w /= rhs.w;
    }
}
