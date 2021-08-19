//! 3D Mesh definitions

use std::sync::Arc;

use lazy_static::lazy_static;

use crate::gfx::{IndexBuffer, VertexBuffer};

use super::material::Material;

#[derive(Default)]
pub(crate) struct Attributes {
    pub(crate) position: Option<VertexBuffer>,
    pub(crate) normal: Option<VertexBuffer>,
    pub(crate) tangent: Option<VertexBuffer>,
    pub(crate) texcoord0: Option<VertexBuffer>,
    pub(crate) texcoord1: Option<VertexBuffer>,
    pub(crate) color0: Option<VertexBuffer>,
    pub(crate) joints0: Option<VertexBuffer>,
    pub(crate) weights0: Option<VertexBuffer>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrimitiveMode {
    Points,
    Lines,
    LineLoop,
    LineStrip,
    Triangles,
    TriangleStrip,
    TriangleFan,
}

impl Default for PrimitiveMode {
    fn default() -> Self {
        PrimitiveMode::Triangles
    }
}

pub(crate) struct Primitive {
    pub(crate) attributes: Attributes,
    pub(crate) indices: Option<IndexBuffer>,
    pub(crate) mode: PrimitiveMode,
    // TODO: morph targets
    pub(crate) material: Arc<Material>,
}

impl Default for Primitive {
    fn default() -> Self {
        Primitive {
            attributes: Attributes::default(),
            indices: None,
            mode: PrimitiveMode::default(),
            material: Material::get_default(),
        }
    }
}

lazy_static! {
    static ref DEFAULT_PRIMITIVE: Arc<Primitive> = Arc::new(Primitive::default());
}

impl Primitive {
    pub(crate) fn get_default() -> Arc<Primitive> {
        DEFAULT_PRIMITIVE.clone()
    }
}

pub(crate) struct Mesh {
    pub(crate) name: Option<String>,
    pub(crate) primitives: Vec<Arc<Primitive>>,
    // TODO: morph targets weights
}