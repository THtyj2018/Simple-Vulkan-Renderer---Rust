//! Camera

use ushio_geom::{Mat4, Vec4};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMode {
    Perspective,
    Orthographic,
}

#[derive(Debug, Clone)]
pub struct Camera {
    mode: CameraMode,
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
    near_clip: f32,
    far_clip: f32,
}

impl Camera {
    pub fn perspective(aspect: f32, fov: f32, near_clip: f32, far_clip: f32) -> Camera {
        assert!(aspect > 0.0);
        assert!((fov > 0.0) && (fov < std::f32::consts::PI));
        assert!(near_clip > 0.0);
        assert!(far_clip > near_clip);
        let hh = near_clip * (fov * 0.5).tan();
        let hw = aspect * hh;
        Camera {
            mode: CameraMode::Perspective,
            left: -hw,
            top: hh,
            right: hw,
            bottom: -hh,
            near_clip,
            far_clip,
        }
    }

    pub fn orthographic(width: f32, height: f32, near_clip: f32, far_clip: f32) -> Camera {
        assert!(width > 0.0);
        assert!(height > 0.0);
        assert!(far_clip > near_clip);
        let hh = height / 2.0;
        let hw = width / 2.0;
        Camera {
            mode: CameraMode::Orthographic,
            left: -hw,
            top: hh,
            right: hw,
            bottom: -hh,
            near_clip,
            far_clip,
        }
    }

    pub fn projection(&self) -> Mat4 {
        // TODO
        match self.mode {
            CameraMode::Perspective => Mat4::new(
                Vec4::new(
                    2.0 * self.near_clip / (self.right - self.left),
                    0.0,
                    0.0,
                    0.0,
                ),
                Vec4::new(
                    0.0,
                    2.0 * self.near_clip / (self.top - self.bottom),
                    0.0,
                    0.0,
                ),
                Vec4::new(
                    (self.right + self.left) / (self.right - self.left),
                    (self.top + self.bottom) / (self.top - self.bottom),
                    -self.far_clip / (self.near_clip - self.far_clip),
                    1.0,
                ),
                Vec4::new(
                    0.0,
                    0.0,
                    self.near_clip * self.far_clip / (self.near_clip - self.far_clip),
                    0.0,
                ),
            ),
            // TODO
            CameraMode::Orthographic => Mat4::new(
                Vec4::new(2.0 / (self.right - self.left), 0.0, 0.0, 0.0),
                Vec4::new(0.0, 2.0 / (self.top - self.bottom), 0.0, 0.0),
                Vec4::new(0.0, 0.0, 1.0 / (self.near_clip - self.far_clip), 0.0),
                Vec4::new(
                    (self.left + self.right) / (self.left - self.right),
                    (self.top + self.bottom) / (self.bottom - self.top),
                    self.near_clip / (self.near_clip - self.far_clip),
                    1.0,
                ),
            ),
        }
    }
}
