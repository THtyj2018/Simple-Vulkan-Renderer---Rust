//! glTF2.0 Loader

use base64;
use image;
use log;
use serde::Deserialize;
use serde_json;
use std::{
    borrow::BorrowMut,
    cell::RefCell,
    fs::File,
    io::{Error as IOError, Read},
    path::Path,
    rc::{Rc, Weak},
    sync::Arc,
};
use thiserror::Error;
use ushio_geom::{Quaternion, Vec3};

use crate::{
    gfx::{
        self, AttributeType, ComponentType, GraphicsError, IndexBuffer,
        StagingBuffer, TextureImage, TransferDstBufferInfo, VertexBuffer,
    },
    Color,
};

use super::{
    material::{
        self, AlphaMode, CommonTexture, Filter, MipmapMode, NormalTexture, OcclusionTexture,
        SamplerInfo, WrapMode,
    },
    mesh::{self, PrimitiveMode},
    SceneNode, Transform,
};

type JValue = serde_json::Value;
type JObject = serde_json::Map<String, JValue>;

#[derive(Deserialize)]
struct Asset {
    #[serde(default)]
    copyright: Option<String>,
    #[serde(default)]
    generator: Option<String>,
    version: String,
    #[serde(default)]
    minVersion: Option<String>,

    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

#[derive(Deserialize)]
struct Scene {
    #[serde(default)]
    nodes: Vec<usize>,

    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

#[derive(Deserialize)]
struct Node {
    #[serde(default)]
    camera: Option<usize>,
    #[serde(default)]
    children: Vec<usize>,
    #[serde(default)]
    skin: Option<usize>,
    #[serde(default)]
    matrix: Option<[f32; 16]>,
    #[serde(default)]
    mesh: Option<usize>,
    #[serde(default = "Node::rotation")]
    rotation: [f32; 4],
    #[serde(default = "Node::scale")]
    scale: [f32; 3],
    #[serde(default = "Node::translation")]
    translation: [f32; 3],
    #[serde(default)]
    weights: Option<Vec<f32>>,

    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

impl Node {
    fn translation() -> [f32; 3] {
        Vec3::zero().into()
    }

    fn rotation() -> [f32; 4] {
        Quaternion::identity().into()
    }

    fn scale() -> [f32; 3] {
        Vec3::one().into()
    }
}

#[derive(Deserialize)]
struct Perspective {
    #[serde(default = "Perspective::aspect_ratio")]
    aspectRatio: f32,
    yfov: f32,
    #[serde(default)]
    zfar: Option<f32>,
    znear: f32,

    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

impl Perspective {
    fn aspect_ratio() -> f32 {
        1.0
    }
}

#[derive(Deserialize)]
struct Orthographic {
    xmag: f32,
    ymag: f32,
    zfar: f32,
    znear: f32,

    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

#[derive(Deserialize)]
struct Camera {
    r#type: String,
    perspevtice: Option<Perspective>,
    orthographic: Option<Orthographic>,

    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

#[derive(Deserialize)]
struct Primitive {
    /// A dictionary object, where each key corresponds to mesh attribute semantic and
    /// each value is the index of the accessor containing attribute's data.
    attributes: JObject,
    #[serde(default)]
    indices: Option<usize>,
    #[serde(default)]
    material: Option<usize>,
    #[serde(default = "Primitive::mode")]
    mode: u32,
    /// An array of Morph Targets, each Morph Target is a dictionary mapping attributes
    /// (only `POSITION`, `NORMAL`, and `TANGENT` supported) to their deviations in the Morph Target.
    targets: Option<Vec<JObject>>,

    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

impl Primitive {
    fn mode() -> u32 {
        4
    }
}

#[derive(Deserialize)]
struct Mesh {
    primitives: Vec<Primitive>,
    /// Array of weights to be applied to the Morph Targets.
    #[serde(default)]
    weights: Option<Vec<f32>>,

    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

#[derive(Deserialize)]
struct Skin {
    joints: Vec<usize>,
    #[serde(default)]
    inverseBindMatrices: Option<usize>,
    /// The index of the node used as a skeleton root. The node must be
    /// the closest common root of the joints hierarchy or a direct or indirect parent node of the closest common root.
    #[serde(default)]
    skeleton: Option<usize>,

    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

#[derive(Deserialize)]
struct Target {
    path: String,
    #[serde(default)]
    node: Option<usize>,

    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

#[derive(Deserialize)]
struct Channel {
    sampler: usize,
    target: Target,

    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

#[derive(Deserialize)]
struct AnimationSampler {
    input: usize,
    output: usize,
    #[serde(default = "AnimationSampler::interpolation")]
    interpolation: String,

    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

impl AnimationSampler {
    fn interpolation() -> String {
        "LINEAR".into()
    }
}

#[derive(Deserialize)]
struct Animation {
    channels: Vec<Channel>,
    samplers: Vec<AnimationSampler>,

    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

#[derive(Deserialize)]
struct TextureInfo {
    index: usize,
    #[serde(default = "TextureInfo::tex_coord")]
    texCoord: u32,

    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

impl TextureInfo {
    fn tex_coord() -> u32 {
        0
    }
}

#[derive(Deserialize)]
struct NormalTextureInfo {
    index: usize,
    #[serde(default = "TextureInfo::tex_coord")]
    texCoord: u32,
    #[serde(default = "NormalTextureInfo::scale")]
    scale: f32,

    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

impl NormalTextureInfo {
    fn scale() -> f32 {
        1.0
    }
}

#[derive(Deserialize)]
struct OcclusionTextureInfo {
    index: usize,
    #[serde(default = "TextureInfo::tex_coord")]
    texCoord: u32,
    #[serde(default = "OcclusionTextureInfo::strength")]
    strength: f32,

    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

impl OcclusionTextureInfo {
    fn tex_coord() -> u32 {
        0
    }

    fn strength() -> f32 {
        1.0
    }
}

#[derive(Deserialize)]
struct PbrMetallicRoughness {
    #[serde(default = "PbrMetallicRoughness::base_color_factor")]
    baseColorFactor: [f32; 4],
    #[serde(default)]
    baseColorTexture: Option<TextureInfo>,
    #[serde(default = "PbrMetallicRoughness::metallic_factor")]
    metallicFactor: f32,
    #[serde(default = "PbrMetallicRoughness::roughness_factor")]
    roughnessFactor: f32,
    #[serde(default)]
    metallicRoughnessTexture: Option<TextureInfo>,

    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

impl PbrMetallicRoughness {
    fn base_color_factor() -> [f32; 4] {
        Color::white().into()
    }

    fn metallic_factor() -> f32 {
        1.0
    }

    fn roughness_factor() -> f32 {
        1.0
    }
}

impl Default for PbrMetallicRoughness {
    fn default() -> Self {
        PbrMetallicRoughness {
            baseColorFactor: Self::base_color_factor(),
            baseColorTexture: None,
            metallicFactor: Self::metallic_factor(),
            roughnessFactor: Self::roughness_factor(),
            metallicRoughnessTexture: None,

            extensions: JObject::default(),
            extras: JValue::default(),
        }
    }
}

#[derive(Deserialize)]
struct Material {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,

    #[serde(default)]
    pbrMetallicRoughness: PbrMetallicRoughness,
    #[serde(default)]
    normalTexture: Option<NormalTextureInfo>,
    #[serde(default)]
    occlusionTexture: Option<OcclusionTextureInfo>,
    #[serde(default = "Material::emissive_factor")]
    emissiveFactor: [f32; 3],
    #[serde(default)]
    emissiveTexture: Option<TextureInfo>,
    #[serde(default = "Material::alpha_mode")]
    alphaMode: String,
    #[serde(default = "Material::alpha_cutoff")]
    alphaCutOff: f32,
    #[serde(default = "Material::double_sided")]
    doubleSided: bool,
}

impl Material {
    fn alpha_mode() -> String {
        "OPAQUE".to_string()
    }

    fn alpha_cutoff() -> f32 {
        0.5
    }

    fn double_sided() -> bool {
        false
    }

    fn emissive_factor() -> [f32; 3] {
        Vec3::zero().into()
    }
}

#[derive(Deserialize)]
struct Texture {
    #[serde(default)]
    sampler: Option<usize>,
    #[serde(default)]
    source: Option<usize>,

    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

#[derive(Deserialize)]
struct Sampler {
    #[serde(default = "Sampler::filter")]
    magFilter: u32,
    #[serde(default = "Sampler::filter")]
    minFilter: u32,
    #[serde(default = "Sampler::wrap")]
    wrapS: u32,
    #[serde(default = "Sampler::wrap")]
    wrapT: u32,

    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

impl Sampler {
    fn filter() -> u32 {
        // TODO
        9728 // NEAREST
    }

    fn wrap() -> u32 {
        10497 // REPEAT
    }
}

#[derive(Deserialize)]
struct Image {
    #[serde(default)]
    uri: Option<String>,
    #[serde(default)]
    bufferView: Option<usize>,
    #[serde(default)]
    mimeType: Option<String>,

    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

#[derive(Deserialize)]
struct SparseIndices {
    bufferView: usize,
    #[serde(default = "Accessor::byte_offset")]
    byteOffset: usize,
    componentType: u32,

    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

#[derive(Deserialize)]
struct SparseValues {
    bufferView: usize,
    #[serde(default = "Accessor::byte_offset")]
    byteOffset: usize,

    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

#[derive(Deserialize)]
struct Sparse {
    count: usize,
    indices: SparseIndices,
    values: SparseValues,

    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

#[derive(Deserialize)]
struct Accessor {
    #[serde(default)]
    bufferView: Option<usize>,
    #[serde(default = "Accessor::byte_offset")]
    byteOffset: usize,
    componentType: u32,
    #[serde(default = "Accessor::normalized")]
    normalized: bool,
    count: usize,
    r#type: String,
    #[serde(default)]
    max: Option<[f32; 3]>,
    #[serde(default)]
    min: Option<[f32; 3]>,
    #[serde(default)]
    sparse: Option<Sparse>,

    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

impl Accessor {
    fn byte_offset() -> usize {
        0
    }

    fn normalized() -> bool {
        false
    }
}

#[derive(Deserialize)]
struct BufferView {
    buffer: usize,
    #[serde(default = "BufferView::byte_offset")]
    byteOffset: usize,
    byteLength: usize,
    #[serde(default)]
    byteStride: Option<usize>,
    #[serde(default)]
    target: Option<u32>,

    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

impl BufferView {
    fn byte_offset() -> usize {
        0
    }
}

#[derive(Deserialize)]
struct Buffer {
    byteLength: usize,
    #[serde(default)]
    uri: Option<String>,

    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,
}

#[derive(Deserialize)]
struct GLTF {
    #[serde(default)]
    extensions: JObject,
    #[serde(default)]
    extras: JValue,

    asset: Asset,
    #[serde(default)]
    scene: Option<usize>,
    #[serde(default)]
    scenes: Vec<Scene>,
    #[serde(default)]
    nodes: Vec<Node>,
    #[serde(default)]
    cameras: Vec<Camera>,
    #[serde(default)]
    meshes: Vec<Mesh>,
    #[serde(default)]
    skins: Vec<Skin>,
    #[serde(default)]
    animations: Vec<Animation>,
    #[serde(default)]
    materials: Vec<Material>,
    #[serde(default)]
    textures: Vec<Texture>,
    #[serde(default)]
    samplers: Vec<Sampler>,
    #[serde(default)]
    images: Vec<Image>,
    #[serde(default)]
    accessors: Vec<Accessor>,
    #[serde(default)]
    bufferViews: Vec<BufferView>,
    #[serde(default)]
    buffers: Vec<Buffer>,
}

#[derive(Debug, Error)]
pub enum GLTFError {
    #[error("glTF2.0 load error: {0}")]
    IO(#[from] IOError),
    #[error("glTF2.0 json or format parse error: {0}")]
    JSON(#[from] serde_json::Error),
    #[error("glTF need version 2.0")]
    Version,
    #[error("glTF2.0 file descripts empty scenes")]
    Empty,
    #[error("glTF2.0 content error: {0}")]
    Content(String),
    #[error("glTF2.0 buffer transfer error: {0}")]
    Transfer(#[from] GraphicsError),
    #[error("glTF2.0 base64 buffer decode error: {0}")]
    Base64(#[from] base64::DecodeError),
    #[error("from utf8 error: {0}")]
    UTF8(#[from] std::string::FromUtf8Error),
    #[error("glTF2.0 image parse error: {0}")]
    Image(#[from] image::ImageError),
}

impl GLTFError {
    fn content<T, S: Into<String>>(s: S) -> Result<T, GLTFError> {
        Err(GLTFError::Content(s.into()))
    }
}

type GLTFResult<T> = Result<T, GLTFError>;

struct GLTFContext<'a> {
    path: &'a Path,
    gltf: &'a GLTF,
    nodes: Vec<Option<Rc<RefCell<SceneNode>>>>,
    meshes: Vec<Option<Rc<mesh::Mesh>>>,
    materials: Vec<Option<Arc<material::Material>>>,
    textures: Vec<Option<Arc<material::Texture>>>,
    images: Vec<Option<Arc<TextureImage>>>,
    staging_buffers: Vec<Option<Rc<StagingBuffer>>>,
}

impl<'a> GLTFContext<'a> {
    fn new(path: &'a Path, gltf: &'a GLTF) -> GLTFResult<GLTFContext<'a>> {
        if gltf.asset.version != "2.0" {
            return Err(GLTFError::Version);
        }

        if gltf.scenes.is_empty() {
            return Err(GLTFError::Empty);
        }

        let mut nodes = vec![];
        let mut meshes = vec![];
        let mut materials = vec![];
        let mut textures = vec![];
        let mut images = vec![];
        let mut staging_buffers = vec![];
        nodes.resize(gltf.nodes.len(), None);
        meshes.resize(gltf.meshes.len(), None);
        materials.resize(gltf.materials.len(), None);
        textures.resize(gltf.textures.len(), None);
        images.resize(gltf.images.len(), None);
        staging_buffers.resize(gltf.buffers.len(), None);

        Ok(GLTFContext {
            path,
            gltf,
            nodes,
            meshes,
            materials,
            textures,
            images,
            staging_buffers,
        })
    }

    fn parse(&mut self) -> GLTFResult<Vec<Rc<RefCell<SceneNode>>>> {
        let gltf = self.gltf;

        let scene = &gltf.scenes[gltf.scene.unwrap_or(0)];
        if scene.nodes.is_empty() {
            return Ok(vec![]);
        }

        let mut nodes = vec![];
        for node_id in &scene.nodes {
            nodes.push(self.parse_node(*node_id, None)?);
        }
        Ok(nodes)
    }

    fn parse_node(
        &mut self,
        node_id: usize,
        parent: Option<Weak<RefCell<SceneNode>>>,
    ) -> GLTFResult<Rc<RefCell<SceneNode>>> {
        if node_id >= self.nodes.len() {
            return GLTFError::content(format!(
                "node index out of range ({} >= {})",
                node_id,
                self.nodes.len()
            ));
        }

        if let Some(node) = self.nodes[node_id].clone() {
            return Ok(node);
        }

        let node = &self.gltf.nodes[node_id];

        let transform = if let Some(matrix) = node.matrix.as_ref() {
            Transform::identity() // TODO
        } else {
            Transform {
                translation: node.translation.into(),
                rotation: node.rotation.into(),
                scale: node.scale.into(),
            }
        };

        let mesh = if let Some(mesh_id) = node.mesh {
            Some(self.parse_mesh(mesh_id)?)
        } else {
            None
        };

        let mut scene_node = Rc::new(RefCell::new(SceneNode {
            uid: SceneNode::gen_uid(),
            name: node.name.clone(),
            children: vec![],
            parent,
            transform,
            mesh,
            camera: None, // TODO
            light: None,
        }));

        for _node_id in node.children.iter() {
            unsafe {
                let p = scene_node.as_ptr();
                (*p).children
                    .push(self.parse_node(*_node_id, Some(Rc::downgrade(&scene_node)))?);
            }
        }

        self.nodes[node_id].replace(scene_node.clone());

        Ok(scene_node)
    }

    fn parse_mesh(&mut self, mesh_id: usize) -> GLTFResult<Rc<mesh::Mesh>> {
        if mesh_id >= self.meshes.len() {
            return GLTFError::content(format!(
                "mesh index out of range ({} >= {})",
                mesh_id,
                self.meshes.len()
            ));
        }

        if let Some(mesh) = self.meshes[mesh_id].clone() {
            return Ok(mesh);
        }

        let mesh = &self.gltf.meshes[mesh_id];

        let mut primitives = vec![];
        for p in mesh.primitives.iter() {
            let mut attributes = mesh::Attributes::default();
            if let Some(attribute) = p.attributes.get("POSITION") {
                let accessor_id = self.parse_attr_accessor_id(attribute)?;
                attributes
                    .position
                    .replace(self.gen_vertex_buffer(accessor_id)?);
            } else {
                primitives.push(mesh::Primitive::get_default());
                continue;
            };
            if let Some(attribute) = p.attributes.get("NORMAL") {
                let accessor_id = self.parse_attr_accessor_id(attribute)?;
                attributes
                    .normal
                    .replace(self.gen_vertex_buffer(accessor_id)?);
            }
            if let Some(attribute) = p.attributes.get("TANGENT") {
                let accessor_id = self.parse_attr_accessor_id(attribute)?;
                attributes
                    .tangent
                    .replace(self.gen_vertex_buffer(accessor_id)?);
            }
            if let Some(attribute) = p.attributes.get("TEXCOORD_0") {
                let accessor_id = self.parse_attr_accessor_id(attribute)?;
                attributes
                    .texcoord0
                    .replace(self.gen_vertex_buffer(accessor_id)?);
            }
            if let Some(attribute) = p.attributes.get("TEXCOORD_1") {
                let accessor_id = self.parse_attr_accessor_id(attribute)?;
                attributes
                    .texcoord1
                    .replace(self.gen_vertex_buffer(accessor_id)?);
            }
            if let Some(attribute) = p.attributes.get("COLOR_0") {
                let accessor_id = self.parse_attr_accessor_id(attribute)?;
                attributes
                    .color0
                    .replace(self.gen_vertex_buffer(accessor_id)?);
            }
            if let Some(attribute) = p.attributes.get("JOINTS_0") {
                let accessor_id = self.parse_attr_accessor_id(attribute)?;
                attributes
                    .joints0
                    .replace(self.gen_vertex_buffer(accessor_id)?);
            }
            if let Some(attribute) = p.attributes.get("WEIGHTS_0") {
                let accessor_id = self.parse_attr_accessor_id(attribute)?;
                attributes
                    .weights0
                    .replace(self.gen_vertex_buffer(accessor_id)?);
            }

            let indices = if let Some(accessor_id) = p.indices {
                Some(self.gen_index_buffer(accessor_id)?)
            } else {
                None
            };

            primitives.push(Arc::new(mesh::Primitive {
                attributes,
                indices,
                mode: Self::convert_primitive_mode(p.mode)?,
                material: match p.material {
                    Some(material_id) => self.parse_material(material_id)?,
                    None => material::Material::get_default(),
                },
            }))
        }

        let scene_mesh = Rc::new(mesh::Mesh {
            name: mesh.name.clone(),
            primitives,
        });

        self.meshes[mesh_id].replace(scene_mesh.clone());

        Ok(scene_mesh)
    }

    fn parse_material(&mut self, material_id: usize) -> GLTFResult<Arc<material::Material>> {
        if material_id >= self.materials.len() {
            return GLTFError::content(format!(
                "material index out of range ({} >= {})",
                material_id,
                self.materials.len()
            ));
        }

        if let Some(material) = self.materials[material_id].clone() {
            return Ok(material);
        }

        let mat = &self.gltf.materials[material_id];

        let ret = Arc::new(material::Material {
            name: mat.name.clone(),
            base_color_factor: mat.pbrMetallicRoughness.baseColorFactor.into(),
            base_color_texture: self
                .parse_common_texture(&mat.pbrMetallicRoughness.baseColorTexture)?,
            metallic_factor: mat.pbrMetallicRoughness.metallicFactor,
            roughness_factor: mat.pbrMetallicRoughness.roughnessFactor,
            metallic_roughness_texture: self
                .parse_common_texture(&mat.pbrMetallicRoughness.metallicRoughnessTexture)?,
            normal_texture: self.parse_normal_texture(&mat.normalTexture)?,
            occlusion_texture: self.parse_occlusion_texture(&mat.occlusionTexture)?,
            emissive_factor: mat.emissiveFactor.into(),
            emissive_texture: self.parse_common_texture(&mat.emissiveTexture)?,
            alpha_mode: Self::convert_alpha_mode(&mat.alphaMode)?,
            alpha_cutoff: mat.alphaCutOff,
            double_sided: mat.doubleSided,
        });

        self.materials[material_id].replace(ret.clone());
        Ok(ret)
    }

    fn parse_common_texture(
        &mut self,
        texture_info: &Option<TextureInfo>,
    ) -> GLTFResult<Option<CommonTexture>> {
        Ok(match texture_info {
            Some(info) => Some(CommonTexture {
                texture: self.parse_texture(info.index)?,
                texcoord_set: info.texCoord,
            }),
            None => None,
        })
    }

    fn parse_normal_texture(
        &mut self,
        texture_info: &Option<NormalTextureInfo>,
    ) -> GLTFResult<Option<NormalTexture>> {
        Ok(match texture_info {
            Some(info) => Some(NormalTexture {
                texture: self.parse_texture(info.index)?,
                texcoord_set: info.texCoord,
                scale: info.scale,
            }),
            None => None,
        })
    }

    fn parse_occlusion_texture(
        &mut self,
        texture_info: &Option<OcclusionTextureInfo>,
    ) -> GLTFResult<Option<OcclusionTexture>> {
        Ok(match texture_info {
            Some(info) => Some(OcclusionTexture {
                texture: self.parse_texture(info.index)?,
                texcoord_set: info.texCoord,
                strength: info.strength,
            }),
            None => None,
        })
    }

    fn parse_texture(&mut self, texture_id: usize) -> GLTFResult<Arc<material::Texture>> {
        if texture_id >= self.textures.len() {
            return GLTFError::content(format!(
                "texture index out of range ({} >= {})",
                texture_id,
                self.textures.len()
            ));
        }

        if let Some(texture) = self.textures[texture_id].clone() {
            return Ok(texture);
        }

        let tex = &self.gltf.textures[texture_id];

        let (sampler_info, _) = match tex.sampler {
            Some(id) => self.parse_sampler(id)?,
            None => (SamplerInfo::default(), true),
        };
        let image = match tex.source {
            Some(id) => self.parse_image(id, true)?,
            None => todo!(),
        };
        let sampler = Arc::new(gfx::Sampler::new(&sampler_info, image.mip_levels())?);
        let texture = Arc::new(material::Texture {
            name: tex.name.clone(),
            image,
            sampler,
        });

        self.textures[texture_id].replace(texture.clone());
        Ok(texture)
    }

    fn parse_sampler(&mut self, sampler_id: usize) -> GLTFResult<(SamplerInfo, bool)> {
        if sampler_id >= self.gltf.samplers.len() {
            return GLTFError::content(format!(
                "texture index out of range ({} >= {})",
                sampler_id,
                self.gltf.samplers.len()
            ));
        }
        let sampler = &self.gltf.samplers[sampler_id];
        let (min_filter, mipmap_mode, gen_mipmap) = Self::convert_min_filter(sampler.minFilter)?;
        Ok((
            SamplerInfo {
                mag_filter: Self::convert_mag_filter(sampler.magFilter)?,
                min_filter,
                mipmap_mode,
                wrap_u: Self::convert_wrap_mode(sampler.wrapS)?,
                wrap_v: Self::convert_wrap_mode(sampler.wrapT)?,
                ..Default::default()
            },
            gen_mipmap,
        ))
    }

    fn parse_image(&mut self, image_id: usize, gen_mipmap: bool) -> GLTFResult<Arc<TextureImage>> {
        if image_id >= self.images.len() {
            return GLTFError::content(format!(
                "image index out of range ({} >= {})",
                image_id,
                self.images.len()
            ));
        }

        if let Some(image) = self.images[image_id].clone() {
            return Ok(image);
        }

        let img = &self.gltf.images[image_id];

        let image_data = if let Some(buffer_view_id) = img.bufferView {
            if let Some(mime_type) = &img.mimeType {
                let buffer_view = &self.gltf.bufferViews[buffer_view_id];
                let buffer = &self.gltf.buffers[buffer_view.buffer];
                todo!()
            } else {
                return GLTFError::content(format!(
                    "image {} mimeType undefined; required when bufferView is defined",
                    image_id
                ));
            }
        } else if let Some(uri) = &img.uri {
            self.parse_uri(uri)?
        } else {
            return GLTFError::content(format!(
                "image {} uri and bufferView undefined; requires when no extended data source",
                image_id
            ));
        };
        let any_image = image::load_from_memory(&image_data)?;
        let image = Arc::new(TextureImage::new(&any_image, gen_mipmap)?);

        self.images[image_id].replace(image.clone());
        Ok(image)
    }

    fn parse_attr_accessor_id(&self, attribute: &JValue) -> GLTFResult<usize> {
        if let JValue::Number(n) = attribute {
            if let Some(accessor_id) = n.as_u64() {
                let id = accessor_id as usize;
                if id >= self.gltf.accessors.len() {
                    return GLTFError::content(format!(
                        "accessor index out of range ({} >= {})",
                        id,
                        self.gltf.accessors.len()
                    ));
                }
                Ok(id)
            } else {
                GLTFError::content("primitive attributes must be an index refer to an accessor")
            }
        } else {
            GLTFError::content("primitive attributes must be an index refer to an accessor")
        }
    }

    fn gen_vertex_buffer(&mut self, accessor_id: usize) -> GLTFResult<VertexBuffer> {
        let accessor = &self.gltf.accessors[accessor_id];
        if let Some(buffer_view_id) = accessor.bufferView {
            let buffer_view = &self.gltf.bufferViews[buffer_view_id];
            let buffer = self.parse_staging_buffer(buffer_view.buffer)?;
            if let Some(byte_stride) = buffer_view.byteStride {
                todo!()
            } else {
                Ok(VertexBuffer::new(
                    buffer.as_ref(),
                    buffer_view.byteOffset + accessor.byteOffset,
                    TransferDstBufferInfo {
                        component_type: Self::convert_component_type(accessor.componentType)?,
                        attribute_type: Self::convert_attribute_type(&accessor.r#type)?,
                        attribute_count: accessor.count,
                    },
                    accessor.normalized,
                )?)
            }
        } else {
            todo!()
        }
    }

    fn gen_index_buffer(&mut self, accessor_id: usize) -> GLTFResult<IndexBuffer> {
        let accessor = &self.gltf.accessors[accessor_id];
        if let Some(buffer_view_id) = accessor.bufferView {
            let buffer_view = &self.gltf.bufferViews[buffer_view_id];
            let buffer = self.parse_staging_buffer(buffer_view.buffer)?;
            if let Some(byte_stride) = buffer_view.byteStride {
                todo!()
            } else {
                Ok(IndexBuffer::new(
                    buffer.as_ref(),
                    buffer_view.byteOffset + accessor.byteOffset,
                    TransferDstBufferInfo {
                        component_type: Self::convert_component_type(accessor.componentType)?,
                        attribute_type: Self::convert_attribute_type(&accessor.r#type)?,
                        attribute_count: accessor.count,
                    },
                )?)
            }
        } else {
            todo!()
        }
    }

    fn parse_staging_buffer(&mut self, buffer_id: usize) -> GLTFResult<Rc<StagingBuffer>> {
        if buffer_id >= self.meshes.len() {
            return GLTFError::content(format!(
                "buffer index out of range ({} >= {})",
                buffer_id,
                self.meshes.len()
            ));
        }

        if let Some(buffer) = self.staging_buffers[buffer_id].clone() {
            return Ok(buffer);
        }

        let buffer = &self.gltf.buffers[buffer_id];
        let data = match buffer.uri.as_ref() {
            Some(uri) => self.parse_uri(uri)?,
            None => todo!(),
        };

        let staging_buffer = Rc::new(StagingBuffer::new(&data)?);
        self.staging_buffers[buffer_id].replace(staging_buffer.clone());
        Ok(staging_buffer)
    }

    fn parse_uri(&self, uri: &String) -> GLTFResult<Vec<u8>> {
        if uri.starts_with("data:") {
            let uri = uri.as_bytes();
            if let Some(idx) = uri.iter().position(|x| *x == b',') {
                Ok(base64::decode(uri.get((idx + 1)..uri.len()).unwrap())?)
            } else {
                GLTFError::content("Base64 header error")
            }
        } else {
            let path = self.path.parent().unwrap().join(uri);
            let mut f = File::open(path.clone())?;
            let mut buf = vec![];
            f.read_to_end(&mut buf)?;
            log::info!("\"{}\" loaded", path.to_str().unwrap_or("?file?"));
            Ok(buf)
        }
    }

    fn convert_primitive_mode(mode: u32) -> GLTFResult<PrimitiveMode> {
        Ok(match mode {
            0 => PrimitiveMode::Points,
            1 => PrimitiveMode::Lines,
            2 => PrimitiveMode::LineLoop,
            3 => PrimitiveMode::LineStrip,
            4 => PrimitiveMode::Triangles,
            5 => PrimitiveMode::TriangleStrip,
            6 => PrimitiveMode::TriangleFan,
            _ => return GLTFError::content(format!("Invalid primitive mode {}", mode)),
        })
    }

    fn convert_component_type(component_type: u32) -> GLTFResult<ComponentType> {
        Ok(match component_type {
            5120 => ComponentType::Byte,
            5121 => ComponentType::UnsignedByte,
            5122 => ComponentType::Short,
            5123 => ComponentType::UnsignedShort,
            5125 => ComponentType::UnsignedInt,
            5126 => ComponentType::Float,
            _ => return GLTFError::content(format!("Invalid component type {}", component_type)),
        })
    }

    fn convert_attribute_type(ty: &String) -> GLTFResult<AttributeType> {
        Ok(match ty.as_str() {
            "SCALAR" => AttributeType::Scalar,
            "VEC2" => AttributeType::Vec2,
            "VEC3" => AttributeType::Vec3,
            "VEC4" => AttributeType::Vec4,
            "MAT2" => AttributeType::Mat2,
            "MAT3" => AttributeType::Mat3,
            "MAT4" => AttributeType::Mat4,
            _ => return GLTFError::content(format!("Invalid attribute type {}", *ty)),
        })
    }

    fn convert_alpha_mode(mode: &String) -> GLTFResult<AlphaMode> {
        Ok(match mode.as_str() {
            "OPAQUE" => AlphaMode::Opaque,
            "MASK" => AlphaMode::Mask,
            "BLEND" => AlphaMode::Blend,
            _ => return GLTFError::content(format!("Invalid material alpha mode {}", *mode)),
        })
    }

    fn convert_mag_filter(filter: u32) -> GLTFResult<Filter> {
        Ok(match filter {
            9728 => Filter::Nearest,
            9729 => Filter::Linear,
            _ => return GLTFError::content(format!("Invalid mag filter {}", filter)),
        })
    }

    fn convert_min_filter(filter: u32) -> GLTFResult<(Filter, MipmapMode, bool)> {
        Ok(match filter {
            9728 => (Filter::Nearest, MipmapMode::default(), false),
            9729 => (Filter::Linear, MipmapMode::default(), false),
            9984 => (Filter::Nearest, MipmapMode::Nearest, true),
            9985 => (Filter::Linear, MipmapMode::Nearest, true),
            9986 => (Filter::Nearest, MipmapMode::Linear, true),
            9987 => (Filter::Linear, MipmapMode::Linear, true),
            _ => return GLTFError::content(format!("Invalid min filter {}", filter)),
        })
    }

    fn convert_wrap_mode(mode: u32) -> GLTFResult<WrapMode> {
        Ok(match mode {
            10497 => WrapMode::Repeat,
            33648 => WrapMode::MirroredRepeat,
            33071 => WrapMode::ClampToEdge,
            _ => return GLTFError::content(format!("Invalid wrap mode {}", mode)),
        })
    }
}

pub(super) fn load_gltf(path: &Path) -> GLTFResult<Vec<Rc<RefCell<SceneNode>>>> {
    let mut f = File::open(path)?;
    let mut buf = String::new();
    let f_size = f.read_to_string(&mut buf)?;
    let gltf: GLTF = serde_json::from_str(&buf)?;

    GLTFContext::new(path, &gltf)?.parse()
}
