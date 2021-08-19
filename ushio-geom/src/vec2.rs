//! 2D Vector

use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::{IntoSTD140, ScalarArray, STD140};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct STD140Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const fn new(x: f32, y: f32) -> Vec2 {
        Vec2 { x, y }
    }

    pub const fn one() -> Vec2 {
        Self::new(1.0, 1.0)
    }

    pub const fn zero() -> Vec2 {
        Self::new(0.0, 0.0)
    }

    pub const fn pos_x() -> Vec2 {
        Self::new(1.0, 0.0)
    }

    pub const fn neg_x() -> Vec2 {
        Self::new(-1.0, 0.0)
    }

    pub const fn pos_y() -> Vec2 {
        Self::new(0.0, 1.0)
    }

    pub const fn neg_y() -> Vec2 {
        Self::new(0.0, -1.0)
    }

    pub fn dot(self, rhs: Vec2) -> f32 {
        self.x * rhs.x + self.y * rhs.y
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
}

impl Default for Vec2 {
    fn default() -> Self {
        Self::zero()
    }
}

impl Default for STD140Vec2 {
    fn default() -> Self {
        Vec2::default().into_std140()
    }
}

impl Into<(f32, f32)> for Vec2 {
    fn into(self) -> (f32, f32) {
        (self.x, self.y)
    }
}

impl STD140 for STD140Vec2 {}

impl IntoSTD140 for Vec2 {
    type Output = STD140Vec2;

    fn into_std140(&self) -> Self::Output {
        STD140Vec2 {
            x: self.x,
            y: self.y,
        }
    }
}

impl ScalarArray for Vec2 {
    type Scalar = f32;

    fn scalar_count() -> usize {
        2
    }
}

impl Index<usize> for Vec2 {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < Self::scalar_count());
        unsafe { self.as_ptr().offset(index as isize).as_ref().unwrap() }
    }
}

impl IndexMut<usize> for Vec2 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < Self::scalar_count());
        unsafe { self.as_mut_ptr().offset(index as isize).as_mut().unwrap() }
    }
}

impl Add<f32> for Vec2 {
    type Output = Vec2;

    fn add(self, rhs: f32) -> Self::Output {
        Self::new(self.x + rhs, self.y + rhs)
    }
}

impl Add<Vec2> for Vec2 {
    type Output = Vec2;

    fn add(self, rhs: Vec2) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Add<Vec2> for f32 {
    type Output = Vec2;

    fn add(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self + rhs.x, self + rhs.y)
    }
}

impl AddAssign<f32> for Vec2 {
    fn add_assign(&mut self, rhs: f32) {
        self.x += rhs;
        self.y += rhs;
    }
}

impl AddAssign<Vec2> for Vec2 {
    fn add_assign(&mut self, rhs: Vec2) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Sub<f32> for Vec2 {
    type Output = Vec2;

    fn sub(self, rhs: f32) -> Self::Output {
        Self::new(self.x - rhs, self.y - rhs)
    }
}

impl Sub<Vec2> for Vec2 {
    type Output = Vec2;

    fn sub(self, rhs: Vec2) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Sub<Vec2> for f32 {
    type Output = Vec2;

    fn sub(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self - rhs.x, self - rhs.y)
    }
}

impl SubAssign<f32> for Vec2 {
    fn sub_assign(&mut self, rhs: f32) {
        self.x -= rhs;
        self.y -= rhs;
    }
}

impl SubAssign<Vec2> for Vec2 {
    fn sub_assign(&mut self, rhs: Vec2) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Neg for Vec2 {
    type Output = Vec2;

    fn neg(self) -> Self::Output {
        Vec2::new(-self.x, -self.y)
    }
}

impl Mul<f32> for Vec2 {
    type Output = Vec2;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl Mul<Vec2> for Vec2 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        Self::new(self.x * rhs.x, self.y * rhs.y)
    }
}

impl Mul<Vec2> for f32 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self * rhs.x, self * rhs.y)
    }
}

impl MulAssign<f32> for Vec2 {
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl MulAssign<Vec2> for Vec2 {
    fn mul_assign(&mut self, rhs: Vec2) {
        self.x *= rhs.x;
        self.y *= rhs.y;
    }
}

impl Div<f32> for Vec2 {
    type Output = Vec2;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

impl Div<Vec2> for Vec2 {
    type Output = Vec2;

    fn div(self, rhs: Vec2) -> Self::Output {
        Self::new(self.x / rhs.x, self.y / rhs.y)
    }
}

impl Div<Vec2> for f32 {
    type Output = Vec2;

    fn div(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self / rhs.x, self / rhs.y)
    }
}

impl DivAssign<f32> for Vec2 {
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

impl DivAssign<Vec2> for Vec2 {
    fn div_assign(&mut self, rhs: Vec2) {
        self.x /= rhs.x;
        self.y /= rhs.y;
    }
}
