//! Shadow pass

use std::{ffi::CString, mem::size_of, sync::Arc};

use ash::vk;

use ushio_geom::{std140_structs, IntoSTD140, Mat4, Vec3, STD140};

use crate::{
    gfx::{Graphics, GraphicsPipelineCreateInfo, ShaderModule},
    scene::mesh::Primitive,
};

use super::{trace::ResourceTracer, GraphicsResult, ShadowMap, StorageBuffer, UniformBuffer};

std140_structs! {
    pub(super) struct ShadowPerFrame uniform STD140ShadowPerFrame {
        pub(super) light: Mat4,
    }

    pub(super) struct ShadowPerMesh uniform STD140ShadowPerMesh {
        pub(super) transform: Mat4,
    }
}

pub(super) struct ShadowPass {
    frame_ubs: Vec<UniformBuffer<ShadowPerFrame>>,
    mesh_ssbos: Vec<StorageBuffer<ShadowPerMesh>>,
    set0_layout: vk::DescriptorSetLayout,
    set1_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets0: Vec<vk::DescriptorSet>,
    descriptor_sets1: Vec<vk::DescriptorSet>,
    render_pass: vk::RenderPass,
    pub(super) shadow_maps: Vec<ShadowMap>,
    framebuffers: Vec<vk::Framebuffer>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl ShadowPass {
    pub(super) fn null() -> ShadowPass {
        ShadowPass {
            frame_ubs: vec![],
            mesh_ssbos: vec![],
            set0_layout: vk::DescriptorSetLayout::null(),
            set1_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_sets0: vec![],
            descriptor_sets1: vec![],
            render_pass: vk::RenderPass::null(),
            shadow_maps: vec![],
            framebuffers: vec![],
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
        }
    }

    pub(super) unsafe fn recreate(
        &mut self,
        frames_in_flight: usize,
        format: vk::Format,
        width: u32,
        height: u32,
    ) -> GraphicsResult<()> {
        self.destroy();
        let gfx = Graphics::get_ref();

        for _ in 0..frames_in_flight {
            self.frame_ubs.push(UniformBuffer::new()?);
            self.mesh_ssbos.push(StorageBuffer::new(16384)?);
        }

        self.set0_layout = gfx.create_descriptor_set_layout(
            &[vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::VERTEX,
                ..Default::default()
            }],
            &[],
        )?;
        self.set1_layout = gfx.create_descriptor_set_layout(
            &[vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::VERTEX,
                ..Default::default()
            }],
            &[],
        )?;

        self.descriptor_pool = gfx.create_descriptor_pool(
            &[
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: frames_in_flight as _,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    descriptor_count: frames_in_flight as _,
                },
            ],
            frames_in_flight * 2,
            false,
        )?;

        self.descriptor_sets0 =
            gfx.allocate_descriptor_sets(self.descriptor_pool, self.set0_layout, frames_in_flight)?;
        self.descriptor_sets1 =
            gfx.allocate_descriptor_sets(self.descriptor_pool, self.set1_layout, frames_in_flight)?;

        for (i, set) in self.descriptor_sets0.iter().enumerate() {
            let buffer_info = vk::DescriptorBufferInfo {
                buffer: self.frame_ubs[i].buf.raw,
                offset: 0,
                range: self.frame_ubs[i].buf.size as _,
            };
            gfx.write_descriptor_sets(&[vk::WriteDescriptorSet {
                dst_set: *set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                p_image_info: std::ptr::null(),
                p_buffer_info: &buffer_info,
                p_texel_buffer_view: std::ptr::null(),
                ..Default::default()
            }]);
        }

        self.render_pass = {
            let depth_attachment = vk::AttachmentDescription {
                format,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                ..Default::default()
            };

            let depth_attachment_ref = vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            };

            let subpass = vk::SubpassDescription {
                flags: vk::SubpassDescriptionFlags::empty(),
                input_attachment_count: 0,
                p_input_attachments: 0 as _,
                pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
                color_attachment_count: 0,
                p_color_attachments: 0 as _,
                p_resolve_attachments: 0 as _,
                p_depth_stencil_attachment: &depth_attachment_ref as _,
                ..Default::default()
            };

            let subpass_dependencies = [
                vk::SubpassDependency {
                    src_subpass: vk::SUBPASS_EXTERNAL,
                    dst_subpass: 0,
                    src_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                    dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    src_access_mask: vk::AccessFlags::SHADER_READ,
                    dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    dependency_flags: vk::DependencyFlags::BY_REGION,
                },
                vk::SubpassDependency {
                    src_subpass: 0,
                    dst_subpass: vk::SUBPASS_EXTERNAL,
                    src_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                    src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    dependency_flags: vk::DependencyFlags::BY_REGION,
                },
            ];

            gfx.create_render_pass(&[depth_attachment], &[subpass], &subpass_dependencies)?
        };

        for _ in 0..frames_in_flight {
            self.shadow_maps
                .push(ShadowMap::new(format, width, height)?);
        }

        for i in 0..frames_in_flight {
            self.framebuffers.push(gfx.create_framebuffer(
                self.render_pass,
                &[self.shadow_maps[i].view],
                width,
                height,
            )?);
        }

        self.pipeline_layout =
            gfx.create_pipeline_layout(&[self.set0_layout, self.set1_layout], &[])?;

        self.pipeline = {
            let vert = ShaderModule::new(include_bytes!("../../../shaders/shadow_map.vert.spv"))?;
            let frag = ShaderModule::new(include_bytes!("../../../shaders/empty.frag.spv"))?;
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

            let vertex_binding_descs = &[vk::VertexInputBindingDescription {
                binding: 0,
                stride: size_of::<Vec3>() as _,
                input_rate: vk::VertexInputRate::VERTEX,
            }];

            let vertex_attributes_descs = &[vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            }];

            let create_info = GraphicsPipelineCreateInfo {
                layout: self.pipeline_layout,
                render_pass: self.render_pass,
                subpass: 0,
                shader_stages,
                vertex_binding_descs,
                vertex_attributes_descs,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                viewport: vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: width as _,
                    height: height as _,
                    min_depth: 0.0,
                    max_depth: 1.0,
                },
                scissor: vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: vk::Extent2D { width, height },
                },
                cull_mode: vk::CullModeFlags::BACK,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                depth_bias_enable: true,
                depth_bias_constant_factor: 1.25,
                depth_bias_slope_factor: 1.75,
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                depth_test_enable: true,
                depth_write_enable: true,
                attachments: &[],
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
        gfx.destroy_pipeline_layout(self.pipeline_layout);
        self.pipeline_layout = vk::PipelineLayout::null();
        while let Some(framebuffer) = self.framebuffers.pop() {
            gfx.destroy_framebuffer(framebuffer);
        }
        self.shadow_maps.clear();
        gfx.destroy_render_pass(self.render_pass);
        self.render_pass = vk::RenderPass::null();

        self.descriptor_sets1.clear();
        self.descriptor_sets0.clear();
        gfx.destroy_descriptor_pool(self.descriptor_pool);
        self.descriptor_pool = vk::DescriptorPool::null();
        gfx.destroy_descriptor_set_layout(self.set1_layout);
        self.set1_layout = vk::DescriptorSetLayout::null();
        gfx.destroy_descriptor_set_layout(self.set0_layout);
        self.set0_layout = vk::DescriptorSetLayout::null();

        self.mesh_ssbos.clear();
        self.frame_ubs.clear();
    }

    pub(super) unsafe fn prepare_draw(
        &self,
        current_frame: usize,
        per_frame: &ShadowPerFrame,
        per_meshes: &[ShadowPerMesh],
    ) -> GraphicsResult<()> {
        self.frame_ubs[current_frame].update(per_frame)?;
        self.mesh_ssbos[current_frame].update(per_meshes)?;

        let mesh_ssbo_info = vk::DescriptorBufferInfo {
            buffer: self.mesh_ssbos[current_frame].buf.raw,
            offset: 0,
            range: (size_of::<<ShadowPerFrame as IntoSTD140>::Output>() * per_meshes.len()) as _,
        };

        if !per_meshes.is_empty() {
            Graphics::get_ref().write_descriptor_sets(&[vk::WriteDescriptorSet {
                dst_set: self.descriptor_sets1[current_frame],
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &mesh_ssbo_info,
                ..Default::default()
            }]);
        }

        Ok(())
    }

    pub(super) unsafe fn draw<'a, T>(
        &self,
        command_buffer: vk::CommandBuffer,
        current_frame: usize,
        primitives: T,
    ) where
        T: Iterator<Item = (usize, &'a Arc<Primitive>)>,
    {
        let gfx = Graphics::get_ref();
        let tracer = ResourceTracer::get_ref();

        gfx.cmd_begin_render_pass(
            command_buffer,
            self.render_pass,
            self.framebuffers[current_frame],
            vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.shadow_maps[current_frame].extent(),
            },
            &[vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            }],
            vk::SubpassContents::INLINE,
        );

        gfx.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline,
        );

        gfx.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout,
            0,
            &[
                self.descriptor_sets0[current_frame],
                self.descriptor_sets1[current_frame],
            ],
            &[],
        );

        for (mesh_id, primitive) in primitives {
            let attributes = &primitive.attributes;

            let vertex_count = if let Some(position) = attributes.position.as_ref() {
                gfx.cmd_bind_vertex_buffer(command_buffer, 0, position);
                tracer.touch_buffer(position.buf.clone());
                position.vertex_count()
            } else {
                continue;
            };

            if let Some(indices) = primitive.indices.as_ref() {
                gfx.cmd_bind_index_buffer(command_buffer, indices);
                tracer.touch_buffer(indices.buf.clone());
                gfx.cmd_draw_indexed(command_buffer, indices.index_count(), 1, 0, mesh_id);
            } else {
                gfx.cmd_draw(command_buffer, vertex_count, 1, 0, mesh_id);
            }
        }

        gfx.cmd_end_render_pass(command_buffer);
    }
}
