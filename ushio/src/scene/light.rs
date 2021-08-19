//! Lights

use crate::Color;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightMode {
    Directional,
    Point,
    Spot,
}

#[derive(Debug, Clone)]
pub struct Light {
    pub mode: LightMode,
    pub color: Color,
    pub intensity: f32,
    pub attenuations: [f32; 3],
    pub cone: f32,
}

impl Light {
    pub fn directional(color: Color, intensity: f32) -> Light {
        Light {
            mode: LightMode::Directional,
            color,
            intensity,
            attenuations: [0.0, 0.0, 0.0],
            cone: f32::default(),
        }
    }
}