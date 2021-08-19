//! Skybox

use std::{fs::File, io::Read, path::Path, sync::Arc};

use super::material::{SamplerInfo, WrapMode};
use crate::gfx::{CubeMap, GraphicsError, Sampler};

use image;
use log;
use thiserror::Error;

pub struct Skybox {
    pub(crate) cubemap: Arc<CubeMap>,
    pub(crate) sampler: Arc<Sampler>,
}

pub struct SkyboxImagePaths<P: AsRef<Path>> {
    pub right: P,
    pub left: P,
    pub top: P,
    pub bottom: P,
    pub front: P,
    pub back: P,
}

#[derive(Debug, Error)]
pub enum SkyboxError {
    #[error("Skybox loading error: {0}")]
    IO(#[from] std::io::Error),
    #[error("Skybox transfer error: {0}")]
    Transfer(#[from] GraphicsError),
    #[error("Skybox image parse error: {0}")]
    Image(#[from] image::ImageError),
}

impl Skybox {
    pub fn load<P: AsRef<Path>>(paths: SkyboxImagePaths<P>) -> Result<Arc<Skybox>, SkyboxError> {
        let mut imgs = vec![];
        let mut f = File::open(paths.right.as_ref().clone())?;
        let mut buffer = vec![];
        f.read_to_end(&mut buffer)?;
        imgs.push(image::load_from_memory(&buffer)?);
        log::info!(
            "\"{}\" loaded",
            paths.right.as_ref().to_str().unwrap_or("?file?")
        );
        f = File::open(paths.left.as_ref().clone())?;
        buffer.clear();
        f.read_to_end(&mut buffer)?;
        imgs.push(image::load_from_memory(&buffer)?);
        log::info!(
            "\"{}\" loaded",
            paths.left.as_ref().to_str().unwrap_or("?file?")
        );
        f = File::open(paths.top.as_ref().clone())?;
        buffer.clear();
        f.read_to_end(&mut buffer)?;
        imgs.push(image::load_from_memory(&buffer)?);
        log::info!(
            "\"{}\" loaded",
            paths.top.as_ref().to_str().unwrap_or("?file?")
        );
        f = File::open(paths.bottom.as_ref().clone())?;
        buffer.clear();
        f.read_to_end(&mut buffer)?;
        imgs.push(image::load_from_memory(&buffer)?);
        log::info!(
            "\"{}\" loaded",
            paths.bottom.as_ref().to_str().unwrap_or("?file?")
        );
        f = File::open(paths.front.as_ref().clone())?;
        buffer.clear();
        f.read_to_end(&mut buffer)?;
        imgs.push(image::load_from_memory(&buffer)?);
        log::info!(
            "\"{}\" loaded",
            paths.front.as_ref().to_str().unwrap_or("?file?")
        );
        f = File::open(paths.back.as_ref().clone())?;
        buffer.clear();
        f.read_to_end(&mut buffer)?;
        imgs.push(image::load_from_memory(&buffer)?);
        log::info!(
            "\"{}\" loaded",
            paths.back.as_ref().to_str().unwrap_or("?file?")
        );
        
        let cubemap = Arc::new(CubeMap::new(&imgs, false)?);
        let sampler = Arc::new(Sampler::new(
            &SamplerInfo {
                wrap_u: WrapMode::ClampToEdge,
                wrap_v: WrapMode::ClampToEdge,
                wrap_w: WrapMode::ClampToEdge,
                ..Default::default()
            },
            1,
        )?);

        Ok(Arc::new(Skybox { cubemap, sampler }))
    }
}
