//! Matrix4x4

use std::{
    ops::{Div, Index, IndexMut, Mul, MulAssign, Neg},
    usize,
};

use crate::{IntoSTD140, Mat3, STD140Vec4, Vec4, STD140};

#[repr(C)]
/// Column-main-sequence 4x4 Matrix
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub cols: [Vec4; 4],
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct STD140Mat4 {
    pub cols: [STD140Vec4; 4],
}

macro_rules! swap {
    ($lhs:expr, $rhs:expr) => {{
        let tmp = $lhs;
        $lhs = $rhs;
        $rhs = tmp;
    }};
}

impl Mat4 {
    pub const fn new(col0: Vec4, col1: Vec4, col2: Vec4, col3: Vec4) -> Mat4 {
        Mat4 {
            cols: [col0, col1, col2, col3],
        }
    }

    pub const fn zero() -> Mat4 {
        Self::eye(0.0)
    }

    pub const fn identity() -> Mat4 {
        Self::eye(1.0)
    }

    pub const fn eye(val: f32) -> Mat4 {
        Self::new(
            Vec4::new(val, 0.0, 0.0, 0.0),
            Vec4::new(0.0, val, 0.0, 0.0),
            Vec4::new(0.0, 0.0, val, 0.0),
            Vec4::new(0.0, 0.0, 0.0, val),
        )
    }

    pub fn from_mat3(m: Mat3, col3: Vec4) -> Mat4 {
        Self::new(
            Vec4::from_xyz(m[0], 0.0),
            Vec4::from_xyz(m[1], 0.0),
            Vec4::from_xyz(m[2], 0.0),
            col3,
        )
    }

    pub fn orthographic(width: f32, height: f32, near_clip: f32, far_clip: f32) -> Mat4 {
        assert!(width > 0.0);
        assert!(height > 0.0);
        assert!(far_clip > near_clip);

        Mat4::new(
            Vec4::new(2.0 / width, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0 / height, 0.0, 0.0),
            Vec4::new(0.0, 0.0, -1.0 / (near_clip - far_clip), 0.0),
            Vec4::new(0.0, 0.0, near_clip / (near_clip - far_clip), 1.0),
        )
    }

    pub fn mat3(&self) -> Mat3 {
        Mat3 {
            cols: [self[0].xyz(), self[1].xyz(), self[2].xyz()],
        }
    }

    pub fn transpose(&mut self) {
        swap!(self[0][1], self[1][0]);
        swap!(self[0][2], self[2][0]);
        swap!(self[0][3], self[3][0]);
        swap!(self[1][2], self[2][1]);
        swap!(self[1][3], self[3][1]);
        swap!(self[2][3], self[3][2]);
    }

    pub fn transposed(&self) -> Mat4 {
        Self::new(
            Vec4::new(self[0][0], self[1][0], self[2][0], self[3][0]),
            Vec4::new(self[0][1], self[1][1], self[2][1], self[3][1]),
            Vec4::new(self[0][2], self[1][2], self[2][2], self[3][2]),
            Vec4::new(self[0][3], self[1][3], self[2][3], self[3][3]),
        )
    }

    pub fn det(&self) -> f32 {
        Vec4::new(
            Mat3::new(self[1].yzw(), self[2].yzw(), self[3].yzw()).det(),
            Mat3::new(self[2].yzw(), self[3].yzw(), self[0].yzw()).det(),
            Mat3::new(self[3].yzw(), self[0].yzw(), self[1].yzw()).det(),
            Mat3::new(self[0].yzw(), self[1].yzw(), self[2].yzw()).det(),
        )
        .dot(Vec4::new(self[0][0], self[1][0], self[2][0], self[3][0]))
    }

    pub fn inv(&self) -> Mat4 {
        let m = Mat4::new(
            Vec4::new(
                Mat3::new(self[1].yzw(), self[2].yzw(), self[3].yzw()).det(),
                -Mat3::new(self[0].yzw(), self[2].yzw(), self[3].yzw()).det(),
                Mat3::new(self[0].yzw(), self[1].yzw(), self[3].yzw()).det(),
                -Mat3::new(self[0].yzw(), self[1].yzw(), self[2].yzw()).det(),
            ),
            Vec4::new(
                -Mat3::new(self[1].xzw(), self[2].xzw(), self[3].xzw()).det(),
                Mat3::new(self[0].xzw(), self[2].xzw(), self[3].xzw()).det(),
                -Mat3::new(self[0].xzw(), self[1].xzw(), self[3].xzw()).det(),
                Mat3::new(self[0].xzw(), self[1].xzw(), self[2].xzw()).det(),
            ),
            Vec4::new(
                Mat3::new(self[1].xyw(), self[2].xyw(), self[3].xyw()).det(),
                -Mat3::new(self[0].xyw(), self[2].xyw(), self[3].xyw()).det(),
                Mat3::new(self[0].xyw(), self[1].xyw(), self[3].xyw()).det(),
                -Mat3::new(self[0].xyw(), self[1].xyw(), self[2].xyw()).det(),
            ),
            Vec4::new(
                -Mat3::new(self[1].xyz(), self[2].xyz(), self[3].xyz()).det(),
                Mat3::new(self[0].xyz(), self[2].xyz(), self[3].xyz()).det(),
                -Mat3::new(self[0].xyz(), self[1].xyz(), self[3].xyz()).det(),
                Mat3::new(self[0].xyz(), self[1].xyz(), self[2].xyz()).det(),
            ),
        );
        m / m[0].dot(Vec4::new(self[0][0], self[1][0], self[2][0], self[3][0]))
    }
}

impl Default for Mat4 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Default for STD140Mat4 {
    fn default() -> Self {
        Mat4::default().into_std140()
    }
}

impl STD140 for STD140Mat4 {}

impl IntoSTD140 for Mat4 {
    type Output = STD140Mat4;

    fn into_std140(&self) -> Self::Output {
        STD140Mat4 {
            cols: [
                self[0].into_std140(),
                self[1].into_std140(),
                self[2].into_std140(),
                self[3].into_std140(),
            ],
        }
    }
}

impl Index<usize> for Mat4 {
    type Output = Vec4;

    fn index(&self, index: usize) -> &Self::Output {
        &self.cols[index]
    }
}

impl IndexMut<usize> for Mat4 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.cols[index]
    }
}

impl Neg for Mat4 {
    type Output = Mat4;

    fn neg(self) -> Self::Output {
        Self::new(-self[0], -self[1], -self[2], -self[3])
    }
}

impl Mul<f32> for Mat4 {
    type Output = Mat4;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self[0] * rhs, self[1] * rhs, self[2] * rhs, self[3] * rhs)
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;

    fn mul(self, rhs: Vec4) -> Self::Output {
        Vec4::new(
            self[0][0] * rhs[0] + self[1][0] * rhs[1] + self[2][0] * rhs[2] + self[3][0] * rhs[3],
            self[0][1] * rhs[0] + self[1][1] * rhs[1] + self[2][1] * rhs[2] + self[3][1] * rhs[3],
            self[0][2] * rhs[0] + self[1][2] * rhs[1] + self[2][2] * rhs[2] + self[3][2] * rhs[3],
            self[0][3] * rhs[0] + self[1][3] * rhs[1] + self[2][3] * rhs[2] + self[3][3] * rhs[3],
        )
    }
}

impl Mul<Mat4> for f32 {
    type Output = Mat4;

    fn mul(self, rhs: Mat4) -> Self::Output {
        rhs * self
    }
}

impl Mul<Mat4> for Vec4 {
    type Output = Vec4;

    fn mul(self, rhs: Mat4) -> Self::Output {
        Vec4::new(
            self[0] * rhs[0][0] + self[1] * rhs[0][1] + self[2] * rhs[0][2] + self[3] * rhs[0][3],
            self[0] * rhs[1][0] + self[1] * rhs[1][1] + self[2] * rhs[1][2] + self[3] * rhs[1][3],
            self[0] * rhs[2][0] + self[1] * rhs[2][1] + self[2] * rhs[2][2] + self[3] * rhs[2][3],
            self[0] * rhs[3][0] + self[1] * rhs[3][1] + self[2] * rhs[3][2] + self[3] * rhs[3][3],
        )
    }
}

impl Mul<Mat4> for Mat4 {
    type Output = Mat4;

    fn mul(self, rhs: Mat4) -> Self::Output {
        Self::new(self * rhs[0], self * rhs[1], self * rhs[2], self * rhs[3])
    }
}

impl MulAssign<f32> for Mat4 {
    fn mul_assign(&mut self, rhs: f32) {
        self[0] *= rhs;
        self[1] *= rhs;
        self[2] *= rhs;
        self[3] *= rhs;
    }
}

impl Div<f32> for Mat4 {
    type Output = Mat4;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self[0] / rhs, self[1] / rhs, self[2] / rhs, self[3] / rhs)
    }
}
