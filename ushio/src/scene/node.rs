//! Scene node

use lazy_static::lazy_static;
use thiserror::Error;
use ushio_geom::{Vec2, Vec3, Vec4};

use crate::gfx::{
    AttributeType, ComponentType, GraphicsError, TransferDstBufferInfo, VertexBuffer,
};

use super::{
    gltf::{load_gltf, GLTFError},
    material::{Material, TextureError},
    mesh::{Attributes, Mesh, Primitive},
    Camera, Light, Transform,
};
use std::{
    cell::RefCell,
    path::Path,
    rc::{Rc, Weak},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

// pub(crate) struct Skin {
//     name: Option<String>,
//     inverse_bind_matrices: Vec<Mat4>, // TODO
//     joints: Vec<Weak<SceneNode>>, // TODO
//     skeleton: Weak<SceneNode>,
// }

#[derive(Debug, Error)]
pub enum NodeError {
    #[error("Graphics Error: {0}")]
    Graphics(#[from] GraphicsError),
    #[error("glTF2.0 load and parse error: {0}")]
    GLTF(#[from] GLTFError),
    #[error("Texture load error: {0}")]
    Texture(#[from] TextureError),
}

pub struct SceneNode {
    pub(crate) uid: usize,
    pub(crate) name: Option<String>,
    pub(crate) children: Vec<Rc<RefCell<SceneNode>>>,
    pub(crate) parent: Option<Weak<RefCell<SceneNode>>>,
    pub(crate) transform: Transform,
    // TODO: skin: Option<Skin>,
    pub(crate) mesh: Option<Rc<Mesh>>,
    pub(crate) camera: Option<Camera>,
    pub(crate) light: Option<Light>,
}

lazy_static! {
    static ref NEXT_UID: AtomicUsize = AtomicUsize::new(0);
}

impl SceneNode {
    pub(super) fn gen_uid() -> usize {
        NEXT_UID.fetch_add(1, Ordering::SeqCst)
    }

    pub fn load<P: AsRef<Path>>(filepath: P) -> Result<Vec<Rc<RefCell<SceneNode>>>, NodeError> {
        let filepath = filepath.as_ref();
        let nodes = if let Some(ext) = filepath.extension() {
            match ext.to_str().unwrap() {
                "gltf" => load_gltf(filepath)?,
                _ => todo!(),
            }
        } else {
            todo!()
        };

        Ok(nodes)
    }

    pub fn new_empty() -> Rc<RefCell<SceneNode>> {
        Rc::new(RefCell::new(SceneNode {
            uid: Self::gen_uid(),
            name: None,
            children: vec![],
            parent: None,
            transform: Transform::identity(),
            mesh: None,
            camera: None,
            light: None,
        }))
    }

    pub fn new_camera(camera: Camera) -> Rc<RefCell<SceneNode>> {
        Rc::new(RefCell::new(SceneNode {
            uid: Self::gen_uid(),
            name: None,
            children: vec![],
            parent: None,
            transform: Transform::identity(),
            mesh: None,
            camera: Some(camera),
            light: None,
        }))
    }

    pub fn new_light(light: Light) -> Rc<RefCell<SceneNode>> {
        Rc::new(RefCell::new(SceneNode {
            uid: Self::gen_uid(),
            name: None,
            children: vec![],
            parent: None,
            transform: Transform::identity(),
            mesh: None,
            camera: None,
            light: Some(light),
        }))
    }

    pub fn new_plane(
        xsize: f32,
        zsize: f32,
        tex_scale_x: f32,
        tex_scale_y: f32,
        material: Arc<Material>,
    ) -> Result<Rc<RefCell<SceneNode>>, NodeError> {
        let positions = [
            Vec3::new(-0.5 * xsize, 0.0, 0.5 * zsize),
            Vec3::new(0.5 * xsize, 0.0, -0.5 * zsize),
            Vec3::new(-0.5 * xsize, 0.0, -0.5 * zsize),
            Vec3::new(-0.5 * xsize, 0.0, 0.5 * zsize),
            Vec3::new(0.5 * xsize, 0.0, 0.5 * zsize),
            Vec3::new(0.5 * xsize, 0.0, -0.5 * zsize),
        ];
        let normals = [
            Vec3::pos_y(),
            Vec3::pos_y(),
            Vec3::pos_y(),
            Vec3::pos_y(),
            Vec3::pos_y(),
            Vec3::pos_y(),
        ];
        let tangents = [
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            Vec4::new(1.0, 0.0, 0.0, 1.0),
        ];
        let uf = xsize / tex_scale_x;
        let vf = zsize / tex_scale_y;
        let texcoords = [
            Vec2::new(0.5 * (1.0 - uf), 0.5 * (1.0 + vf)),
            Vec2::new(0.5 * (1.0 + uf), 0.5 * (1.0 - vf)),
            Vec2::new(0.5 * (1.0 - uf), 0.5 * (1.0 - vf)),
            Vec2::new(0.5 * (1.0 - uf), 0.5 * (1.0 + vf)),
            Vec2::new(0.5 * (1.0 + uf), 0.5 * (1.0 + vf)),
            Vec2::new(0.5 * (1.0 + uf), 0.5 * (1.0 - vf)),
        ];
        let mut buf_info = TransferDstBufferInfo {
            component_type: ComponentType::Float,
            attribute_type: AttributeType::Vec3,
            attribute_count: 6,
        };
        let pos_buf = VertexBuffer::new_direct(&positions, buf_info, false)?;
        let n_buf = VertexBuffer::new_direct(&normals, buf_info, false)?;
        buf_info.attribute_type = AttributeType::Vec4;
        let tan_buf = VertexBuffer::new_direct(&tangents, buf_info, false)?;
        buf_info.attribute_type = AttributeType::Vec2;
        let tc_buf = VertexBuffer::new_direct(&texcoords, buf_info, false)?;

        Ok(Rc::new(RefCell::new(SceneNode {
            uid: Self::gen_uid(),
            name: None,
            children: vec![],
            parent: None,
            transform: Default::default(),
            mesh: Some(Rc::new(Mesh {
                name: None,
                primitives: vec![Arc::new(Primitive {
                    attributes: Attributes {
                        position: Some(pos_buf),
                        normal: Some(n_buf),
                        tangent: Some(tan_buf),
                        texcoord0: Some(tc_buf),
                        ..Default::default()
                    },
                    material,
                    ..Default::default()
                })],
            })),
            camera: None,
            light: None,
        })))
    }

    pub fn set_name<S: Into<String>>(&mut self, name: S) {
        self.name = Some(name.into());
    }

    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }

    pub fn transform(&self) -> &Transform {
        &self.transform
    }

    pub fn set_transform(&mut self, transform: Transform) {
        self.transform = transform;
    }

    pub fn set_scale(&mut self, scale: Vec3) {
        self.transform.scale = scale;
    }

    pub fn set_camera(&mut self, camera: Camera) {
        self.camera.replace(camera);
    }

    pub fn attach(parent: &mut Rc<RefCell<SceneNode>>, child: Rc<RefCell<SceneNode>>) {
        if let Some(p) = child.borrow().parent.as_ref() {
            if let Some(p) = p.upgrade() {
                if *parent == p {
                    return;
                }
                let index = p
                    .borrow()
                    .children
                    .iter()
                    .position(|n| *n == child)
                    .unwrap();
                p.borrow_mut().children.swap_remove(index);
            }
        }
        child.borrow_mut().parent.replace(Rc::downgrade(parent));
        parent.borrow_mut().children.push(child);
    }
}

impl PartialEq for SceneNode {
    fn eq(&self, other: &SceneNode) -> bool {
        self.uid == other.uid
    }
}
