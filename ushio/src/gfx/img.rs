//! Gpu Images

use std::sync::Arc;

use ash::vk;
use image::{DynamicImage, EncodableLayout, GenericImageView};

use crate::{scene::material::SamplerInfo, Color};

use super::{convert::*, GpuImage, Graphics, GraphicsResult, ImageCreateInfo, StagingBuffer};

pub(super) struct DepthImage {
    _image: GpuImage,
    pub(super) view: vk::ImageView,
}

impl DepthImage {
    pub(super) fn new(
        format: vk::Format,
        extent: vk::Extent2D,
        samples: vk::SampleCountFlags,
    ) -> GraphicsResult<DepthImage> {
        let create_info = ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            },
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            samples,
            ..Default::default()
        };

        let image = GpuImage::new(&create_info)?;

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        Graphics::get_ref().transition_image_layout(
            image.raw,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            subresource_range,
        )?;

        let create_info = vk::ImageViewCreateInfo {
            image: image.raw,
            view_type: vk::ImageViewType::TYPE_2D,
            format: image.info.format,
            components: vk::ComponentMapping::default(),
            subresource_range,
            ..Default::default()
        };
        let view = unsafe { Graphics::get_ref().create_image_view(&create_info)? };

        Ok(DepthImage {
            _image: image,
            view,
        })
    }
}

impl Drop for DepthImage {
    fn drop(&mut self) {
        unsafe {
            Graphics::get_ref().destroy_image_view(self.view);
        }
    }
}

pub(super) struct ShadowMap {
    image: GpuImage,
    pub(super) view: vk::ImageView,
    pub(super) sampler: vk::Sampler,
}

impl ShadowMap {
    pub(super) fn new(format: vk::Format, width: u32, height: u32) -> GraphicsResult<ShadowMap> {
        let gfx = Graphics::get_ref();

        let create_info = ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        let image = GpuImage::new(&create_info)?;

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        gfx.transition_image_layout(
            image.raw,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            subresource_range,
        )?;

        let create_info = vk::ImageViewCreateInfo {
            image: image.raw,
            view_type: vk::ImageViewType::TYPE_2D,
            format: image.info.format,
            components: vk::ComponentMapping::default(),
            subresource_range,
            ..Default::default()
        };
        let view = unsafe { gfx.create_image_view(&create_info)? };

        let create_info = vk::SamplerCreateInfo {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_BORDER,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_BORDER,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_BORDER,
            mip_lod_bias: 0.0,
            min_lod: 0.0,
            max_lod: 1.0,
            border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
            ..Default::default()
        };

        let sampler = unsafe {
            match gfx.create_sampler(&create_info) {
                Ok(s) => s,
                Err(e) => {
                    gfx.destroy_image_view(view);
                    return Err(e);
                }
            }
        };

        Ok(ShadowMap {
            image,
            view,
            sampler,
        })
    }

    pub(super) fn extent(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.image.info.extent.width,
            height: self.image.info.extent.height,
        }
    }
}

impl Drop for ShadowMap {
    fn drop(&mut self) {
        unsafe {
            Graphics::get_ref().destroy_sampler(self.sampler);
            Graphics::get_ref().destroy_image_view(self.view);
        }
    }
}

pub(super) struct MSAAImage {
    _image: GpuImage,
    pub(super) view: vk::ImageView,
}

impl MSAAImage {
    pub(super) fn new(
        format: vk::Format,
        extent: vk::Extent2D,
        samples: vk::SampleCountFlags,
    ) -> GraphicsResult<MSAAImage> {
        let create_info = ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            },
            usage: vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            samples,
            ..Default::default()
        };

        let image = GpuImage::new(&create_info)?;

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        Graphics::get_ref().transition_image_layout(
            image.raw,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            subresource_range,
        )?;

        let create_info = vk::ImageViewCreateInfo {
            image: image.raw,
            view_type: vk::ImageViewType::TYPE_2D,
            format: image.info.format,
            components: vk::ComponentMapping::default(),
            subresource_range,
            ..Default::default()
        };
        let view = unsafe { Graphics::get_ref().create_image_view(&create_info)? };

        Ok(MSAAImage {
            _image: image,
            view,
        })
    }
}

impl Drop for MSAAImage {
    fn drop(&mut self) {
        unsafe {
            Graphics::get_ref().destroy_image_view(self.view);
        }
    }
}

pub(crate) struct TextureImage {
    image: Arc<GpuImage>,
    pub(super) view: vk::ImageView,
}

impl TextureImage {
    pub(crate) fn new(src: &DynamicImage, gen_mipmap: bool) -> GraphicsResult<TextureImage> {
        // TODO
        let src = &DynamicImage::ImageRgba8(src.to_rgba8());
        let (format, _, components) = convert_image_format(src);
        let (width, height) = src.dimensions();
        let src = StagingBuffer::new(src.as_bytes())?;

        Self::new_impl(&src, format, width, height, gen_mipmap, components)
    }

    pub(crate) fn new_from_data(
        pixels: &[Color],
        width: u32,
        height: u32,
        gen_mipmap: bool,
    ) -> GraphicsResult<TextureImage> {
        let format = vk::Format::R8G8B8A8_UNORM;
        let src = StagingBuffer::new(pixels)?;
        Self::new_impl(&src, format, width, height, gen_mipmap, Default::default())
    }

    fn new_impl(
        src: &StagingBuffer,
        format: vk::Format,
        width: u32,
        height: u32,
        gen_mipmap: bool,
        components: vk::ComponentMapping,
    ) -> GraphicsResult<TextureImage> {
        let mip_levels = match gen_mipmap {
            true => 32 - width.leading_zeros().max(height.leading_zeros()),
            false => 1,
        };
        let mut image_create_info = ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            mip_levels,
            ..Default::default()
        };
        if gen_mipmap {
            image_create_info.usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        let image = GpuImage::new(&image_create_info)?;

        Graphics::get_ref().transfer_buffer_to_image(
            &src,
            &image,
            &[vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
            }],
        )?;

        let create_info = vk::ImageViewCreateInfo {
            image: image.raw,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            components,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };

        unsafe {
            let view = Graphics::get_ref().create_image_view(&create_info)?;
            Ok(TextureImage {
                image: Arc::new(image),
                view,
            })
        }
    }

    pub(crate) fn mip_levels(&self) -> u32 {
        self.image.info.mip_levels
    }
}

impl Drop for TextureImage {
    fn drop(&mut self) {
        unsafe {
            Graphics::get_ref().destroy_image_view(self.view);
        }
    }
}

pub(crate) struct Sampler {
    pub(super) raw: vk::Sampler,
}

impl Sampler {
    pub(crate) fn new(sampler_info: &SamplerInfo, mip_levels: u32) -> GraphicsResult<Sampler> {
        let create_info = vk::SamplerCreateInfo {
            mag_filter: convert_filter(sampler_info.mag_filter),
            min_filter: convert_filter(sampler_info.min_filter),
            mipmap_mode: convert_mipmap_mode(sampler_info.mipmap_mode),
            address_mode_u: convert_wrap_mode(sampler_info.wrap_u),
            address_mode_v: convert_wrap_mode(sampler_info.wrap_v),
            mip_lod_bias: 0.0,
            anisotropy_enable: vk::TRUE,
            max_anisotropy: 16.0,
            compare_enable: vk::FALSE,
            min_lod: 0.0,
            max_lod: mip_levels as _,
            unnormalized_coordinates: vk::FALSE,
            ..Default::default()
        };
        unsafe {
            let raw = Graphics::get_ref().create_sampler(&create_info)?;
            Ok(Sampler { raw })
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            Graphics::get_ref().destroy_sampler(self.raw);
        }
    }
}

pub(crate) struct CubeMap {
    image: Arc<GpuImage>,
    pub(super) view: vk::ImageView,
}

impl CubeMap {
    pub(crate) fn new(src: &[DynamicImage], gen_mipmap: bool) -> GraphicsResult<CubeMap> {
        assert!(src.len() == 6);

        let (width, height) = src[0].dimensions();
        let format = vk::Format::R8G8B8A8_UNORM;
        // TODO
        let src = src.iter().map(|img| img.to_rgba8()).collect::<Vec<_>>();
        let src_buf = StagingBuffer::new_multi(
            src.iter()
                .map(|img| img.as_bytes())
                .collect::<Vec<_>>()
                .as_slice(),
        )?;

        let mip_levels = match gen_mipmap {
            true => 32 - width.leading_zeros().max(height.leading_zeros()),
            false => 1,
        };
        let mut image_create_info = ImageCreateInfo {
            flags: vk::ImageCreateFlags::CUBE_COMPATIBLE,
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            array_layers: 6,
            mip_levels,
            ..Default::default()
        };
        if gen_mipmap {
            image_create_info.usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        let image = GpuImage::new(&image_create_info)?;

        Graphics::get_ref().transfer_buffer_to_image(
            &src_buf,
            &image,
            &[
                vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 6,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    },
                }
            ],
        )?;

        let create_info = vk::ImageViewCreateInfo {
            image: image.raw,
            view_type: vk::ImageViewType::CUBE,
            format,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 6,
            },
            ..Default::default()
        };

        unsafe {
            let view = Graphics::get_ref().create_image_view(&create_info)?;
            Ok(CubeMap {
                image: Arc::new(image),
                view,
            })
        }
    }
}
