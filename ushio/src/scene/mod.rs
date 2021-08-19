//! Scene manager module

mod camera;
#[allow(non_snake_case, unused)]
mod gltf;
mod light;
pub mod material;
pub(crate) mod mesh;
mod node;
mod skybox;
mod trs;

use std::{cell::RefCell, rc::Rc, sync::Arc};

pub use camera::*;
pub use light::*;
pub use node::*;
pub use skybox::*;
pub use trs::Transform;

#[derive(Clone)]
pub struct SceneParams {
    pub ambient: f32,
}

impl Default for SceneParams {
    fn default() -> Self {
        SceneParams { ambient: 0.0 }
    }
}

#[derive(Clone)]
pub struct Scene {
    pub nodes: Vec<Rc<RefCell<SceneNode>>>,
    pub camera: Rc<RefCell<SceneNode>>,
    pub skybox: Option<Arc<Skybox>>,
    pub params: SceneParams,
}
