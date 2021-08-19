//! Cubemaps

use std::{ffi::CString, mem::size_of};

use ash::vk;
use ushio_geom::{std140_structs, IntoSTD140, Mat4, STD140};

use crate::{gfx::trace::ResourceTracer, scene::Skybox};

use super::{Graphics, GraphicsPipelineCreateInfo, GraphicsResult, ShaderModule, UniformBuffer};

std140_structs! {
    pub(super) struct SkyboxPerFrame uniform STD140SkyboxPerFrame {
        pub(super) camera: Mat4,
    }
}

pub(super) struct SkyboxPipeline {
    frame_ubs: Vec<UniformBuffer<SkyboxPerFrame>>,
    set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl SkyboxPipeline {
    pub(super) fn null() -> SkyboxPipeline {
        SkyboxPipeline {
            frame_ubs: vec![],
            set_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_sets: vec![],
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
        }
    }

    pub(super) unsafe fn recreate(
        &mut self,
        frames_in_flight: usize,
        render_pass: vk::RenderPass,
        subpass: u32,
        width: u32,
        height: u32,
        samples: vk::SampleCountFlags,
        attachments: &[vk::PipelineColorBlendAttachmentState],
    ) -> GraphicsResult<()> {
        self.destroy();
        let gfx = Graphics::get_ref();

        for _ in 0..frames_in_flight {
            self.frame_ubs.push(UniformBuffer::new()?);
        }

        self.set_layout = gfx.create_descriptor_set_layout(
            &[
                vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    ..Default::default()
                },
                vk::DescriptorSetLayoutBinding {
                    binding: 1,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    ..Default::default()
                },
            ],
            &[],
        )?;

        self.descriptor_pool = gfx.create_descriptor_pool(
            &[
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: frames_in_flight as _,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: frames_in_flight as _,
                },
            ],
            frames_in_flight,
            false,
        )?;

        self.descriptor_sets =
            gfx.allocate_descriptor_sets(self.descriptor_pool, self.set_layout, frames_in_flight)?;

        self.layout = gfx.create_pipeline_layout(&[self.set_layout], &[])?;

        self.pipeline = {
            let vert = ShaderModule::new(include_bytes!("../../../shaders/skybox.vert.spv"))?;
            let frag = ShaderModule::new(include_bytes!("../../../shaders/skybox.frag.spv"))?;
            let main_cstr = CString::new("main").unwrap();

            let shader_stages = &[
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::VERTEX,
                    module: vert.raw(),
                    p_name: main_cstr.as_ptr(),
                    p_specialization_info: std::ptr::null(),
                    ..Default::default()
                },
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::FRAGMENT,
                    module: frag.raw(),
                    p_name: main_cstr.as_ptr(),
                    p_specialization_info: std::ptr::null(),
                    ..Default::default()
                },
            ];

            let create_info = GraphicsPipelineCreateInfo {
                layout: self.layout,
                render_pass,
                subpass,
                shader_stages,
                vertex_binding_descs: &[],
                vertex_attributes_descs: &[],
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                viewport: vk::Viewport {
                    x: 0.0,
                    y: height as _,
                    width: width as _,
                    height: -(height as f32),
                    min_depth: 0.0,
                    max_depth: 1.0,
                },
                scissor: vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: vk::Extent2D { width, height },
                },
                cull_mode: vk::CullModeFlags::BACK,
                front_face: vk::FrontFace::CLOCKWISE,
                rasterization_samples: samples,
                depth_test_enable: true,
                depth_write_enable: false,
                attachments,
                dynamic_states: &[],
                ..Default::default()
            };

            gfx.create_graphics_pipeline(&create_info)?
        };

        Ok(())
    }

    pub(super) unsafe fn destroy(&mut self) {
        let gfx = Graphics::get_ref();

        gfx.destroy_pipeline(self.pipeline);
        self.pipeline = vk::Pipeline::null();
        gfx.destroy_pipeline_layout(self.layout);
        self.layout = vk::PipelineLayout::null();

        self.descriptor_sets.clear();
        gfx.destroy_descriptor_pool(self.descriptor_pool);
        self.descriptor_pool = vk::DescriptorPool::null();
        gfx.destroy_descriptor_set_layout(self.set_layout);
        self.set_layout = vk::DescriptorSetLayout::null();

        self.frame_ubs.clear();
    }

    pub(super) unsafe fn prepare_draw(
        &self,
        current_frame: usize,
        per_frame: &SkyboxPerFrame,
        skybox: &Skybox,
    ) -> GraphicsResult<()> {
        self.frame_ubs[current_frame].update(per_frame)?;

        let frame_ub_info = vk::DescriptorBufferInfo {
            buffer: self.frame_ubs[current_frame].buf.raw,
            offset: 0,
            range: size_of::<<SkyboxPerFrame as IntoSTD140>::Output>() as _,
        };

        let skybox_info = vk::DescriptorImageInfo {
            sampler: skybox.sampler.raw,
            image_view: skybox.cubemap.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };

        Graphics::get_ref().write_descriptor_sets(&[
            vk::WriteDescriptorSet {
                dst_set: self.descriptor_sets[current_frame],
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                p_buffer_info: &frame_ub_info,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: self.descriptor_sets[current_frame],
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: &skybox_info,
                ..Default::default()
            },
        ]);

        ResourceTracer::get_ref().touch_cubemap(skybox.cubemap.clone());
        ResourceTracer::get_ref().touch_sampler(skybox.sampler.clone());
        Ok(())
    }

    pub(super) unsafe fn draw(&self, command_buffer: vk::CommandBuffer, current_frame: usize) {
        let gfx = Graphics::get_ref();

        gfx.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline,
        );

        gfx.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.layout,
            0,
            &[self.descriptor_sets[current_frame]],
            &[],
        );

        gfx.cmd_draw(command_buffer, 36, 1, 0, 0);
    }
}
