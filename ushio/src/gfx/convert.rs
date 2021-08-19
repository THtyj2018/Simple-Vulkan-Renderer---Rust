//! Type convert

use ash::vk;
use image::DynamicImage;

use crate::scene::material::{Filter, MipmapMode, WrapMode};

pub(super) fn convert_msaa_samples(samples: u32) -> vk::SampleCountFlags {
    match samples {
        0 | 1 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_2,
        3 | 4 => vk::SampleCountFlags::TYPE_4,
        _ => vk::SampleCountFlags::TYPE_8,
    }
}

pub(super) fn convert_image_format(
    image: &DynamicImage,
) -> (vk::Format, usize, vk::ComponentMapping) {
    match *image {
        DynamicImage::ImageLuma8(_) => (
            vk::Format::R8_UNORM,
            1,
            vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::R,
                b: vk::ComponentSwizzle::R,
                a: vk::ComponentSwizzle::ONE,
            },
        ),
        DynamicImage::ImageLumaA8(_) => (
            vk::Format::R8G8_UNORM,
            2,
            vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::R,
                b: vk::ComponentSwizzle::R,
                a: vk::ComponentSwizzle::G,
            },
        ),
        DynamicImage::ImageRgb8(_) => (
            vk::Format::R8G8B8_UNORM,
            3,
            vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::ONE,
            },
        ),
        DynamicImage::ImageRgba8(_) => (
            vk::Format::R8G8B8A8_UNORM,
            4,
            vk::ComponentMapping::default(),
        ),
        DynamicImage::ImageBgr8(_) => (
            vk::Format::B8G8R8_UNORM,
            3,
            vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::ONE,
            },
        ),
        DynamicImage::ImageBgra8(_) => (
            vk::Format::B8G8R8A8_UNORM,
            4,
            vk::ComponentMapping::default(),
        ),
        DynamicImage::ImageLuma16(_) => (
            vk::Format::R16_UNORM,
            2,
            vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::R,
                b: vk::ComponentSwizzle::R,
                a: vk::ComponentSwizzle::ONE,
            },
        ),
        DynamicImage::ImageLumaA16(_) => (
            vk::Format::R16G16_UNORM,
            4,
            vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::R,
                b: vk::ComponentSwizzle::R,
                a: vk::ComponentSwizzle::G,
            },
        ),
        DynamicImage::ImageRgb16(_) => (
            vk::Format::R16G16B16_UNORM,
            6,
            vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::ONE,
            },
        ),
        DynamicImage::ImageRgba16(_) => (
            vk::Format::R16G16B16A16_UNORM,
            8,
            vk::ComponentMapping::default(),
        ),
    }
}

pub(super) fn convert_filter(filter: Filter) -> vk::Filter {
    match filter {
        Filter::Nearest => vk::Filter::NEAREST,
        Filter::Linear => vk::Filter::LINEAR,
    }
}

pub(super) fn convert_mipmap_mode(mode: MipmapMode) -> vk::SamplerMipmapMode {
    match mode {
        MipmapMode::Nearest => vk::SamplerMipmapMode::NEAREST,
        MipmapMode::Linear => vk::SamplerMipmapMode::LINEAR,
    }
}

pub(super) fn convert_wrap_mode(mode: WrapMode) -> vk::SamplerAddressMode {
    match mode {
        WrapMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        WrapMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        WrapMode::Repeat => vk::SamplerAddressMode::REPEAT,
    }
}
