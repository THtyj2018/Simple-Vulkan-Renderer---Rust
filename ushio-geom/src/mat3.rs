//! Matrix3x3

use std::ops::{Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg};

use crate::{IntoSTD140, Mat2, STD140Vec3, Vec3, STD140};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub cols: [Vec3; 3],
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct STD140Mat3 {
    pub cols: [STD140Vec3; 3],
}

macro_rules! swap {
    ($lhs:expr, $rhs:expr) => {{
        let tmp = $lhs;
        $lhs = $rhs;
        $rhs = tmp;
    }};
}

impl Mat3 {
    pub const fn new(col0: Vec3, col1: Vec3, col2: Vec3) -> Mat3 {
        Mat3 {
            cols: [col0, col1, col2],
        }
    }

    pub const fn zero() -> Mat3 {
        Self::eye(0.0)
    }

    pub const fn identity() -> Mat3 {
        Self::eye(1.0)
    }

    pub const fn eye(val: f32) -> Mat3 {
        Self::new(
            Vec3::new(val, 0.0, 0.0),
            Vec3::new(0.0, val, 0.0),
            Vec3::new(0.0, 0.0, val),
        )
    }

    pub fn from_mat2(m: Mat2, col2: Vec3) -> Mat3 {
        Self::new(Vec3::from_xy(m[0], 0.0), Vec3::from_xy(m[1], 0.0), col2)
    }

    pub fn transpose(&mut self) {
        swap!(self[0][1], self[1][0]);
        swap!(self[0][2], self[2][0]);
        swap!(self[1][2], self[2][1]);
    }

    pub fn transposed(&self) -> Mat3 {
        Self::new(
            Vec3::new(self[0][0], self[1][0], self[2][0]),
            Vec3::new(self[0][1], self[1][1], self[2][1]),
            Vec3::new(self[0][2], self[1][2], self[2][2]),
        )
    }

    pub fn det(&self) -> f32 {
        self[0][0] * (self[1][1] * self[2][2] - self[2][1] * self[1][2])
            - self[1][0] * (self[0][1] * self[2][2] - self[2][1] * self[0][2])
            + self[2][0] * (self[0][1] * self[1][2] - self[1][1] * self[0][2])
    }

    pub fn inv(&self) -> Mat3 {
        let m = Mat3::new(
            Vec3::new(
                self[1][1] * self[2][2] - self[2][1] * self[1][2],
                self[2][1] * self[0][2] - self[0][1] * self[2][2],
                self[0][1] * self[1][2] - self[1][1] * self[0][2],
            ),
            Vec3::new(
                self[1][2] * self[2][0] - self[2][2] * self[1][0],
                self[2][2] * self[0][0] - self[0][2] * self[2][0],
                self[0][2] * self[1][0] - self[1][2] * self[0][0],
            ),
            Vec3::new(
                self[1][0] * self[2][1] - self[2][0] * self[1][1],
                self[2][0] * self[0][1] - self[0][0] * self[2][1],
                self[0][0] * self[1][1] - self[1][0] * self[0][1],
            ),
        );
        m / m[0].dot(Vec3::new(self[0][0], self[1][0], self[2][0]))
    }
}

impl Default for Mat3 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Default for STD140Mat3 {
    fn default() -> Self {
        Mat3::default().into_std140()
    }
}

impl STD140 for STD140Mat3 {}

impl IntoSTD140 for Mat3 {
    type Output = STD140Mat3;

    fn into_std140(&self) -> Self::Output {
        STD140Mat3 {
            cols: [
                self[0].into_std140(),
                self[1].into_std140(),
                self[2].into_std140(),
            ],
        }
    }
}

impl Index<usize> for Mat3 {
    type Output = Vec3;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < 3);
        &self.cols[index]
    }
}

impl IndexMut<usize> for Mat3 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < 3);
        &mut self.cols[index]
    }
}

impl Neg for Mat3 {
    type Output = Mat3;

    fn neg(self) -> Self::Output {
        Self::new(-self[0], -self[1], -self[2])
    }
}

impl Mul<f32> for Mat3 {
    type Output = Mat3;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self[0] * rhs, self[1] * rhs, self[2] * rhs)
    }
}

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3::new(
            self[0][0] * rhs[0] + self[1][0] * rhs[1] + self[2][0] * rhs[2],
            self[0][1] * rhs[0] + self[1][1] * rhs[1] + self[2][1] * rhs[2],
            self[0][2] * rhs[0] + self[1][2] * rhs[1] + self[2][2] * rhs[2],
        )
    }
}

impl Mul<Mat3> for f32 {
    type Output = Mat3;

    fn mul(self, rhs: Mat3) -> Self::Output {
        rhs * self
    }
}

impl Mul<Mat3> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: Mat3) -> Self::Output {
        Vec3::new(
            self[0] * rhs[0][0] + self[1] * rhs[0][1] + self[2] * rhs[0][2],
            self[0] * rhs[1][0] + self[1] * rhs[1][1] + self[2] * rhs[1][2],
            self[0] * rhs[2][0] + self[1] * rhs[2][1] + self[2] * rhs[2][2],
        )
    }
}

impl Mul<Mat3> for Mat3 {
    type Output = Mat3;

    fn mul(self, rhs: Mat3) -> Self::Output {
        Self::new(self * rhs[0], self * rhs[1], self * rhs[2])
    }
}

impl MulAssign<f32> for Mat3 {
    fn mul_assign(&mut self, rhs: f32) {
        self[0] *= rhs;
        self[1] *= rhs;
        self[2] *= rhs;
    }
}

impl Div<f32> for Mat3 {
    type Output = Mat3;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self[0] / rhs, self[1] / rhs, self[2] / rhs)
    }
}

impl DivAssign<f32> for Mat3 {
    fn div_assign(&mut self, rhs: f32) {
        self[0] /= rhs;
        self[1] /= rhs;
        self[2] /= rhs;
    }
}
