//! Transform

use ushio_geom::{Mat3, Mat4, Quaternion, Vec3, Vec4};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub(crate) translation: Vec3,
    pub(crate) rotation: Quaternion,
    pub(crate) scale: Vec3,
}

impl Transform {
    pub const fn new(translation: Vec3, rotation: Quaternion, scale: Vec3) -> Transform {
        Transform {
            translation,
            rotation,
            scale,
        }
    }

    pub const fn identity() -> Transform {
        Self::new(Vec3::zero(), Quaternion::identity(), Vec3::one())
    }

    pub fn lookat(eye: Vec3, dst: Vec3, up: Vec3) -> Transform {
        Transform {
            translation: eye,
            rotation: Quaternion::look_forward(dst - eye, up),
            scale: Vec3::one(),
        }
    }

    pub fn local_to_parent(&self, position: Vec3) -> Vec3 {
        self.translation + self.rotation.apply_to(self.scale * position)
    }

    pub fn translation(&self) -> Vec3 {
        self.translation
    }

    pub fn rotation(&self) -> Quaternion {
        self.rotation
    }

    pub fn scale(&self) -> Vec3 {
        self.scale
    }

    pub fn rotatation_scale(&self) -> Mat3 {
        let mut rot: Mat3 = self.rotation.into();
        rot.cols[0] *= self.scale.x;
        rot.cols[1] *= self.scale.y;
        rot.cols[2] *= self.scale.z;
        rot
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::identity()
    }
}

impl Into<Mat4> for Transform {
    fn into(self) -> Mat4 {
        let mut r: Mat3 = self.rotation.into();
        r[0] *= self.scale.x;
        r[1] *= self.scale.y;
        r[2] *= self.scale.z;
        Mat4::from_mat3(r, Vec4::from_xyz(self.translation, 1.0))
    }
}
