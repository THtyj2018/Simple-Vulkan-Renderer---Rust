//! 3D Vector

use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::{IntoSTD140, ScalarArray, Vec2, STD140};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct STD140Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }

    pub const fn from_xy(xy: Vec2, z: f32) -> Vec3 {
        Self::new(xy.x, xy.y, z)
    }

    pub const fn one() -> Vec3 {
        Self::new(1.0, 1.0, 1.0)
    }

    pub const fn zero() -> Vec3 {
        Self::new(0.0, 0.0, 0.0)
    }

    pub const fn pos_x() -> Vec3 {
        Self::new(1.0, 0.0, 0.0)
    }

    pub const fn neg_x() -> Vec3 {
        Self::new(-0.0, 0.0, 0.0)
    }

    pub const fn pos_y() -> Vec3 {
        Self::new(0.0, 1.0, 0.0)
    }

    pub const fn neg_y() -> Vec3 {
        Self::new(0.0, -1.0, 0.0)
    }

    pub const fn pos_z() -> Vec3 {
        Self::new(0.0, 0.0, 1.0)
    }

    pub const fn neg_z() -> Vec3 {
        Self::new(0.0, 0.0, -1.0)
    }

    pub fn dot(self, rhs: Vec3) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross(self, rhs: Vec3) -> Vec3 {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    pub fn magnitude_sqr(self) -> f32 {
        self.dot(self)
    }

    pub fn magnitude(self) -> f32 {
        self.magnitude_sqr().sqrt()
    }

    pub fn magnitude_inv(self) -> f32 {
        1.0 / self.magnitude_sqr().sqrt()
    }

    pub fn normalize(&mut self) {
        *self *= self.magnitude_inv()
    }

    pub fn normalized(self) -> Self {
        self * self.magnitude_inv()
    }

    pub const fn xy(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    pub const fn yz(self) -> Vec2 {
        Vec2::new(self.y, self.z)
    }

    pub const fn zx(self) -> Vec2 {
        Vec2::new(self.z, self.x)
    }
}

impl Default for Vec3 {
    fn default() -> Self {
        Self::zero()
    }
}

impl Default for STD140Vec3 {
    fn default() -> Self {
        Vec3::default().into_std140()
    }
}

impl Into<(f32, f32, f32)> for Vec3 {
    fn into(self) -> (f32, f32, f32) {
        (self.x, self.y, self.z)
    }
}

impl Into<[f32; 3]> for Vec3 {
    fn into(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

impl From<[f32; 3]> for Vec3 {
    fn from(a: [f32; 3]) -> Self {
        Vec3::new(a[0], a[1], a[2])
    }
}

impl STD140 for STD140Vec3 {}

impl IntoSTD140 for Vec3 {
    type Output = STD140Vec3;

    fn into_std140(&self) -> Self::Output {
        STD140Vec3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

impl ScalarArray for Vec3 {
    type Scalar = f32;

    fn scalar_count() -> usize {
        3
    }
}

impl Index<usize> for Vec3 {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < Self::scalar_count());
        unsafe { self.as_ptr().offset(index as isize).as_ref().unwrap() }
    }
}

impl IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < Self::scalar_count());
        unsafe { self.as_mut_ptr().offset(index as isize).as_mut().unwrap() }
    }
}

impl Add<f32> for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: f32) -> Self::Output {
        Self::new(self.x + rhs, self.y + rhs, self.z + rhs)
    }
}

impl Add<Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Vec3) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Add<Vec3> for f32 {
    type Output = Vec3;

    fn add(self, rhs: Vec3) -> Self::Output {
        Vec3::new(self + rhs.x, self + rhs.y, self + rhs.z)
    }
}

impl AddAssign<f32> for Vec3 {
    fn add_assign(&mut self, rhs: f32) {
        self.x += rhs;
        self.y += rhs;
        self.z += rhs;
    }
}

impl AddAssign<Vec3> for Vec3 {
    fn add_assign(&mut self, rhs: Vec3) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub<f32> for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: f32) -> Self::Output {
        Self::new(self.x - rhs, self.y - rhs, self.z - rhs)
    }
}

impl Sub<Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Vec3) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Sub<Vec3> for f32 {
    type Output = Vec3;

    fn sub(self, rhs: Vec3) -> Self::Output {
        Vec3::new(self - rhs.x, self - rhs.y, self - rhs.z)
    }
}

impl SubAssign<f32> for Vec3 {
    fn sub_assign(&mut self, rhs: f32) {
        self.x -= rhs;
        self.y -= rhs;
        self.z -= rhs;
    }
}

impl SubAssign<Vec3> for Vec3 {
    fn sub_assign(&mut self, rhs: Vec3) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self::Output {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3::new(self * rhs.x, self * rhs.y, self * rhs.z)
    }
}

impl MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl MulAssign<Vec3> for Vec3 {
    fn mul_assign(&mut self, rhs: Vec3) {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self.z *= rhs.z;
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f32) -> Self::Output {
        Vec3::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Div<Vec3> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: Vec3) -> Self::Output {
        Self::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl Div<Vec3> for f32 {
    type Output = Vec3;

    fn div(self, rhs: Vec3) -> Self::Output {
        Vec3::new(self / rhs.x, self / rhs.y, self / rhs.z)
    }
}

impl DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

impl DivAssign<Vec3> for Vec3 {
    fn div_assign(&mut self, rhs: Vec3) {
        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
    }
}
