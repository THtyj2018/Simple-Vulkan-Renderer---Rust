//! Material definitions

use std::{fs::File, io::Read, path::Path, sync::Arc};

use image;
use lazy_static::lazy_static;
use log;
use thiserror::Error;
use ushio_geom::Vec3;

use crate::{
    gfx::{GraphicsError, Sampler, TextureImage},
    Color,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    Nearest,
    Linear,
}

impl Default for Filter {
    fn default() -> Self {
        Filter::Nearest
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MipmapMode {
    Nearest,
    Linear,
}

impl Default for MipmapMode {
    fn default() -> Self {
        MipmapMode::Nearest
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrapMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
}

impl Default for WrapMode {
    fn default() -> Self {
        WrapMode::Repeat
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerInfo {
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub mipmap_mode: MipmapMode,
    pub wrap_u: WrapMode,
    pub wrap_v: WrapMode,
    pub wrap_w: WrapMode,
}

impl Default for SamplerInfo {
    fn default() -> Self {
        SamplerInfo {
            mag_filter: Filter::default(),
            min_filter: Filter::default(),
            mipmap_mode: MipmapMode::default(),
            wrap_u: WrapMode::default(),
            wrap_v: WrapMode::default(),
            wrap_w: WrapMode::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

pub struct Texture {
    pub(crate) name: Option<String>,
    pub(crate) image: Arc<TextureImage>,
    pub(crate) sampler: Arc<Sampler>,
}

#[derive(Debug, Error)]
pub enum TextureError {
    #[error("Texture loading error: {0}")]
    IO(#[from] std::io::Error),
    #[error("Texture data transfer error: {0}")]
    Transfer(#[from] GraphicsError),
    #[error("Texture image parse error: {0}")]
    Image(#[from] image::ImageError),
}

lazy_static! {
    static ref DEFAULT_TEXTURE: Arc<Texture> = {
        Arc::new(Texture {
            name: None,
            image: Arc::new(
                TextureImage::new_from_data(
                    std::iter::repeat(Color::white())
                        .take(256)
                        .collect::<Vec<_>>()
                        .as_slice(),
                    16,
                    16,
                    false,
                )
                .unwrap(),
            ),
            sampler: Arc::new(Sampler::new(&SamplerInfo::default(), 1).unwrap()),
        })
    };
}

impl Texture {
    pub fn load<P: AsRef<Path>>(
        filepath: P,
        sampler_info: &SamplerInfo,
        gen_mipmap: bool,
    ) -> Result<Arc<Texture>, TextureError> {
        let mut f = File::open(filepath.as_ref().clone())?;
        let mut buffer = vec![];
        f.read_to_end(&mut buffer)?;
        log::info!(
            "\"{}\" loaded",
            filepath.as_ref().to_str().unwrap_or("?file?")
        );
        let any_image = image::load_from_memory(&buffer)?;
        let image = Arc::new(TextureImage::new(&any_image, gen_mipmap)?);
        let sampler = Arc::new(Sampler::new(&sampler_info, image.mip_levels())?);
        Ok(Arc::new(Texture {
            name: None,
            image,
            sampler,
        }))
    }
}

#[derive(Clone)]
pub struct CommonTexture {
    pub texture: Arc<Texture>,
    pub texcoord_set: u32,
}

impl Default for CommonTexture {
    fn default() -> Self {
        CommonTexture {
            texture: DEFAULT_TEXTURE.clone(),
            texcoord_set: 0,
        }
    }
}

#[derive(Clone)]
pub struct NormalTexture {
    pub texture: Arc<Texture>,
    pub texcoord_set: u32,
    pub scale: f32,
}

impl Default for NormalTexture {
    fn default() -> Self {
        NormalTexture {
            texture: DEFAULT_TEXTURE.clone(),
            texcoord_set: 0,
            scale: 1.0,
        }
    }
}

#[derive(Clone)]
pub struct OcclusionTexture {
    pub texture: Arc<Texture>,
    pub texcoord_set: u32,
    pub strength: f32,
}

impl Default for OcclusionTexture {
    fn default() -> Self {
        OcclusionTexture {
            texture: DEFAULT_TEXTURE.clone(),
            texcoord_set: 0,
            strength: 1.0,
        }
    }
}

pub struct Material {
    pub name: Option<String>,
    pub base_color_factor: Color,
    pub base_color_texture: Option<CommonTexture>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_roughness_texture: Option<CommonTexture>,
    pub normal_texture: Option<NormalTexture>,
    pub occlusion_texture: Option<OcclusionTexture>,
    pub emissive_factor: Vec3,
    pub emissive_texture: Option<CommonTexture>,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
}

impl Default for Material {
    fn default() -> Self {
        Material {
            name: None,
            base_color_factor: Color::white(),
            base_color_texture: None,
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            metallic_roughness_texture: None,
            normal_texture: None,
            occlusion_texture: None,
            emissive_factor: Vec3::zero(),
            emissive_texture: None,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
        }
    }
}

lazy_static! {
    static ref DEFAULT_MATERIAL: Arc<Material> = Arc::new(Material::default());
}

impl Material {
    pub fn get_default() -> Arc<Material> {
        DEFAULT_MATERIAL.clone()
    }
}
