//! Quaternion

use std::ops::{Mul, Neg};

use crate::{IntoSTD140, Mat3, Mat4, Vec3, Vec4, STD140};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    v: Vec3,
    w: f32,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct STD140Quaternion {
    v: Vec3,
    w: f32,
}

impl Quaternion {
    pub fn new(axis: Vec3, theta: f32) -> Quaternion {
        // TODO
        let (s, c) = (theta * 0.5).sin_cos();
        Self::new_impl_v(axis.normalized() * s, c)
    }

    pub const fn identity() -> Quaternion {
        Self::new_impl(0.0, 0.0, 0.0, 1.0)
    }

    pub(crate) const fn new_impl(x: f32, y: f32, z: f32, w: f32) -> Quaternion {
        Quaternion {
            v: Vec3::new(x, y, z),
            w,
        }
    }

    const fn new_impl_v(v: Vec3, w: f32) -> Quaternion {
        Quaternion { v, w }
    }

    pub fn look_forward(forward: Vec3, up: Vec3) -> Quaternion {
        let r = -forward.cross(up).normalized();
        let u = -r.cross(forward).normalized();
        let b = r.cross(u);
        let w = (r.x + u.y + b.z + 1.0).sqrt() / 2.0;
        let v = if w.abs() < f32::EPSILON {
            let x = ((r.x + 1.0) / 2.0).sqrt();
            if x.abs() < f32::EPSILON {
                let y = ((u.y + 1.0) / 2.0).sqrt();
                if y.abs() < f32::EPSILON {
                    Vec3::new(0.0, 0.0, 1.0)
                } else {
                    Vec3::new(x, y, (b.y + u.z) / (4.0 * y))
                }
            } else {
                Vec3::new(x, (u.x + r.y) / (4.0 * x), (r.z + b.x) / (4.0 * x))
            }
        } else {
            Vec3::new(
                (b.y - u.z) / (4.0 * w),
                (r.z - b.x) / (4.0 * w),
                (u.x - r.y) / (4.0 * w),
            )
        };
        Self::new_impl_v(v, w)
    }

    pub fn conj(self) -> Quaternion {
        Self::new_impl_v(-self.v, self.w)
    }

    pub fn inv(self) -> Quaternion {
        self.conj()
    }

    pub fn apply_to(self, p: Vec3) -> Vec3 {
        // 2.0 * self.v.dot(p) * self.v
        //     + (2.0 * self.w * self.w - 1.0) * p
        //     + 2.0 * self.w * self.v.cross(p)
        // ????
        let m: Mat3 = self.into();
        m * p
    }

    pub fn pow(self, index: f32) -> Quaternion {
        // TODO
        let (s, c) = (self.w.acos() * index).sin_cos();
        Quaternion::new_impl_v(self.v.normalized() * s, c)
    }

    pub fn slerp(self, rhs: Quaternion, t: f32) -> Quaternion {
        self * (self.conj() * rhs).pow(t)
    }
}

impl Default for Quaternion {
    fn default() -> Self {
        Self::identity()
    }
}

impl Default for STD140Quaternion {
    fn default() -> Self {
        Quaternion::default().into_std140()
    }
}

impl Into<[f32; 4]> for Quaternion {
    fn into(self) -> [f32; 4] {
        [self.v.x, self.v.y, self.v.z, self.w]
    }
}

impl From<[f32; 4]> for Quaternion {
    fn from(a: [f32; 4]) -> Self {
        Vec4::from(a).into()
    }
}

impl Into<Vec4> for Quaternion {
    fn into(self) -> Vec4 {
        Vec4::new(self.v.x, self.v.y, self.v.z, self.w)
    }
}

impl From<Vec4> for Quaternion {
    fn from(v: Vec4) -> Self {
        let v = v / v.dot(v).sqrt();
        Quaternion::new_impl_v(v.xyz(), v.w)
    }
}

impl Into<Mat3> for Quaternion {
    fn into(self) -> Mat3 {
        let xx = self.v.x * self.v.x;
        let yy = self.v.y * self.v.y;
        let zz = self.v.z * self.v.z;
        let ww = self.w * self.w;
        let xy = self.v.x * self.v.y;
        let yz = self.v.y * self.v.z;
        let zx = self.v.z * self.v.x;
        let xw = self.v.x * self.w;
        let yw = self.v.y * self.w;
        let zw = self.v.z * self.w;

        Mat3 {
            cols: [
                Vec3::new(2.0 * (xx + ww) - 1.0, 2.0 * (xy - zw), 2.0 * (zx + yw)),
                Vec3::new(2.0 * (xy + zw), 2.0 * (yy + ww) - 1.0, 2.0 * (yz - xw)),
                Vec3::new(2.0 * (zx - yw), 2.0 * (yz + xw), 2.0 * (zz + ww) - 1.0),
            ],
        }
    }
}

impl Into<Mat4> for Quaternion {
    fn into(self) -> Mat4 {
        Mat4::from_mat3(self.into(), Vec4::identity())
    }
}

impl STD140 for STD140Quaternion {}

impl IntoSTD140 for Quaternion {
    type Output = STD140Quaternion;

    fn into_std140(&self) -> Self::Output {
        STD140Quaternion {
            v: self.v,
            w: self.w,
        }
    }
}

impl Neg for Quaternion {
    type Output = Quaternion;

    fn neg(self) -> Self::Output {
        Quaternion::new_impl_v(-self.v, -self.w)
    }
}

impl Mul<Quaternion> for Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: Quaternion) -> Self::Output {
        Self::new_impl_v(
            self.w * rhs.v + self.v * rhs.w + self.v.cross(rhs.v),
            self.w * rhs.w - self.v.dot(rhs.v),
        )
    }
}
