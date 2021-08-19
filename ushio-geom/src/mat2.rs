//! Matrix2x2

use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{IntoSTD140, STD140Vec2, Vec2, STD140};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat2 {
    pub cols: [Vec2; 2],
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct STD140Mat2 {
    pub cols: [STD140Vec2; 2],
}

macro_rules! swap {
    ($lhs:expr, $rhs:expr) => {{
        let tmp = $lhs;
        $lhs = $rhs;
        $rhs = tmp;
    }};
}

impl Mat2 {
    pub const fn new(col0: Vec2, col1: Vec2) -> Mat2 {
        Mat2 { cols: [col0, col1] }
    }

    pub const fn zero() -> Mat2 {
        Self::eye(0.0)
    }

    pub const fn identity() -> Mat2 {
        Self::eye(1.0)
    }

    pub const fn eye(val: f32) -> Mat2 {
        Self::new(Vec2::new(val, 0.0), Vec2::new(0.0, val))
    }

    pub fn transpose(&mut self) {
        swap!(self[0][1], self[1][0]);
    }

    pub fn transposed(&self) -> Mat2 {
        Mat2 {
            cols: [
                Vec2::new(self[0][0], self[1][0]),
                Vec2::new(self[0][1], self[1][1]),
            ],
        }
    }

    pub fn det(&self) -> f32 {
        self[0][0] * self[1][1] - self[0][1] * self[1][0]
    }

    pub fn inv(&self) -> Mat2 {
        let d = self.det();
        if d == 0.0 {
            todo!()
        }
        Mat2 {
            cols: [
                Vec2::new(self[1][1] / d, -self[0][1] / d),
                Vec2::new(-self[1][0] / d, self[0][0] / d),
            ],
        }
    }
}

impl Default for Mat2 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Default for STD140Mat2 {
    fn default() -> Self {
        Mat2::default().into_std140()
    }
}

impl STD140 for STD140Mat2 {}

impl IntoSTD140 for Mat2 {
    type Output = STD140Mat2;

    fn into_std140(&self) -> Self::Output {
        STD140Mat2 {
            cols: [self[0].into_std140(), self[1].into_std140()],
        }
    }
}

impl Index<usize> for Mat2 {
    type Output = Vec2;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < 2);
        &self.cols[index]
    }
}

impl IndexMut<usize> for Mat2 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < 2);
        &mut self.cols[index]
    }
}

impl Add<f32> for Mat2 {
    type Output = Mat2;

    fn add(self, rhs: f32) -> Self::Output {
        Self::new(self[0] + rhs, self[1] + rhs)
    }
}

impl Add<Mat2> for Mat2 {
    type Output = Mat2;

    fn add(self, rhs: Mat2) -> Self::Output {
        Self::new(self[0] + rhs[0], self[1] + rhs[1])
    }
}

impl Add<Mat2> for f32 {
    type Output = Mat2;

    fn add(self, rhs: Mat2) -> Self::Output {
        rhs + self
    }
}

impl AddAssign<f32> for Mat2 {
    fn add_assign(&mut self, rhs: f32) {
        self[0] += rhs;
        self[1] += rhs;
    }
}

impl AddAssign<Mat2> for Mat2 {
    fn add_assign(&mut self, rhs: Mat2) {
        self[0] += rhs[0];
        self[1] += rhs[1];
    }
}

impl Sub<f32> for Mat2 {
    type Output = Mat2;

    fn sub(self, rhs: f32) -> Self::Output {
        Self::new(self[0] - rhs, self[1] - rhs)
    }
}

impl Sub<Mat2> for Mat2 {
    type Output = Mat2;

    fn sub(self, rhs: Mat2) -> Self::Output {
        Self::new(self[0] - rhs[0], self[1] - rhs[1])
    }
}

impl Sub<Mat2> for f32 {
    type Output = Mat2;

    fn sub(self, rhs: Mat2) -> Self::Output {
        Mat2::new(self - rhs[0], self - rhs[1])
    }
}

impl SubAssign<f32> for Mat2 {
    fn sub_assign(&mut self, rhs: f32) {
        self[0] -= rhs;
        self[1] -= rhs;
    }
}

impl SubAssign<Mat2> for Mat2 {
    fn sub_assign(&mut self, rhs: Mat2) {
        self[0] -= rhs[0];
        self[1] -= rhs[1];
    }
}

impl Neg for Mat2 {
    type Output = Mat2;

    fn neg(self) -> Self::Output {
        Self::new(-self[0], -self[1])
    }
}

impl Mul<f32> for Mat2 {
    type Output = Mat2;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self[0] * rhs, self[1] * rhs)
    }
}

impl Mul<Vec2> for Mat2 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2::new(
            self[0][0] * rhs[0] + self[1][0] * rhs[1],
            self[0][1] * rhs[0] + self[1][1] * rhs[1],
        )
    }
}

impl Mul<Mat2> for f32 {
    type Output = Mat2;

    fn mul(self, rhs: Mat2) -> Self::Output {
        rhs * self
    }
}

impl Mul<Mat2> for Vec2 {
    type Output = Vec2;

    fn mul(self, rhs: Mat2) -> Self::Output {
        Vec2::new(
            self[0] * rhs[0][0] + self[1] * rhs[0][1],
            self[0] * rhs[1][0] + self[1] * rhs[1][1],
        )
    }
}

impl Mul<Mat2> for Mat2 {
    type Output = Mat2;

    fn mul(self, rhs: Mat2) -> Self::Output {
        Self::new(self * rhs[0], self * rhs[1])
    }
}

impl MulAssign<f32> for Mat2 {
    fn mul_assign(&mut self, rhs: f32) {
        self[0] *= rhs;
        self[1] *= rhs;
    }
}

impl Div<f32> for Mat2 {
    type Output = Mat2;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self[0] / rhs, self[1] / rhs)
    }
}

impl DivAssign<f32> for Mat2 {
    fn div_assign(&mut self, rhs: f32) {
        self[0] /= rhs;
        self[1] /= rhs;
    }
}
