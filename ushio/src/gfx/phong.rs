//! Test renderer

use std::{
    cell::RefCell,
    ffi::CString,
    mem::size_of,
    rc::Rc,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, RwLock,
    },
};

use ash::vk;
use lazy_static::lazy_static;
use ushio_geom::{std140_structs, IntoSTD140, Mat3, Mat4, Vec2, Vec3, Vec4, STD140};

use crate::{
    gfx::{GraphicsPipelineCreateInfo, ShaderModule, VertexBuffer},
    scene::{material::Texture, mesh::Primitive, Scene, SceneNode, SceneParams, Skybox},
    Color,
};

use super::{
    convert::convert_msaa_samples, DepthImage, Display, DisplayConfig, DynamicUniformBuffer,
    Graphics, GraphicsResult, MSAAImage, RendererCreateInfo, RendererDataTrait,
    RendererRequirements, RendererTrait, RenderingDataTrait, ResourceTracer, ShadowPass,
    ShadowPerFrame, ShadowPerMesh, SkyboxPerFrame, SkyboxPipeline, UniformBuffer,
};

std140_structs! {
    struct PerFrame uniform STD140PerFrame {
        camera: Mat4,
        eye: Vec3,
    }

    struct DirLight uniform STD140DirLight {
        matrix: Mat4,
        direction: Vec3,
        color: Color,
        ambient: f32,
        intensity: f32,
    }

    struct MaterialInfo uniform STD140MaterialInfo {
        base_color: Color,
        base_color_map_index: i32,
        normal_map_index: i32,
        normal_scale: f32,
        specular: f32,
    }

    struct PerMesh uniform STD140PerMesh {
        transform: Mat4,
    }
}

static CIS_SIZED_ARRAY_COUNT: u32 = 1000;

pub struct RendererData {
    clear_color: RwLock<Color>,
}

lazy_static! {
    static ref S_RENDERER_DATA: RendererData = RendererData {
        clear_color: RwLock::new(Color::rgb(0.2, 0.2, 0.2)),
    };
}

impl Drop for RendererData {
    fn drop(&mut self) {}
}

impl RendererDataTrait for RendererData {
    fn get_ref() -> &'static Self {
        &S_RENDERER_DATA
    }

    fn set_clear_color(&self, color: Color) {
        *self.clear_color.write().unwrap() = color;
    }

    fn clear_color(&self) -> Color {
        *self.clear_color.read().unwrap()
    }
}

struct RenderInfo {
    mesh_id: usize,
    material_id: usize,
    primitive: Arc<Primitive>,
}

pub(crate) struct RenderingData {
    per_frame: Option<PerFrame>,
    dir_light: Option<DirLight>,
    shadow_per_frame: Option<ShadowPerFrame>,
    skybox_per_frame: Option<SkyboxPerFrame>,
    skybox: Option<Arc<Skybox>>,
    info_list: Vec<RenderInfo>,
    per_meshes: Vec<PerMesh>,
    shadow_per_meshes: Vec<ShadowPerMesh>,
    per_materials: Vec<MaterialInfo>,
    textures: Vec<Arc<Texture>>,
}

impl RenderingDataTrait for RenderingData {
    fn parse_scene(scene: Scene) -> Self {
        let mut data = RenderingData {
            per_frame: None,
            dir_light: None,
            shadow_per_frame: None,
            skybox_per_frame: None,
            skybox: None,
            info_list: vec![],
            per_meshes: vec![],
            shadow_per_meshes: vec![],
            per_materials: vec![],
            textures: vec![],
        };

        data.record_render_info(
            &scene.nodes,
            &*scene.camera.borrow(),
            &scene.params,
            Default::default(),
            Default::default(),
        );

        if data.skybox_per_frame.is_some() {
            if let Some(skybox) = scene.skybox {
                data.skybox = Some(skybox);
            }
        }
        data
    }
}

impl RenderingData {
    fn record_render_info(
        &mut self,
        nodes: &Vec<Rc<RefCell<SceneNode>>>,
        main_camera: &SceneNode,
        scene_params: &SceneParams,
        translation: Vec3,
        rs: Mat3,
    ) {
        for scene_node in nodes {
            let sn = scene_node.borrow();
            let translation = translation + rs * sn.transform.translation;
            let rs = rs * sn.transform.rotatation_scale();
            let transform = Mat4::from_mat3(rs, Vec4::from_xyz(translation, 1.0));
            self.record_render_info(&sn.children, main_camera, scene_params, translation, rs);
            if let Some(camera) = sn.camera.clone() {
                if *sn == *main_camera {
                    self.per_frame.replace(PerFrame {
                        camera: camera.projection() * transform.inv(),
                        eye: translation,
                    });
                    let without_translation = transform.mat3();
                    self.skybox_per_frame.replace(SkyboxPerFrame {
                        camera: camera.projection()
                            * Mat4::from_mat3(without_translation.inv(), Vec4::identity()),
                    });
                }
            }
            if let Some(light) = sn.light.clone() {
                let projection = Mat4::orthographic(10.0, 10.0, -100.0, 100.0); // TODO
                self.dir_light.replace(DirLight {
                    matrix: projection * transform.inv(),
                    direction: -rs[2], // TODO
                    color: light.color,
                    ambient: scene_params.ambient,
                    intensity: light.intensity,
                });
                self.shadow_per_frame.replace(ShadowPerFrame {
                    light: projection * transform.inv(),
                });
            }
            if let Some(mesh) = sn.mesh.clone() {
                for primitive in &mesh.primitives {
                    self.info_list.push(RenderInfo {
                        mesh_id: self.per_meshes.len(),
                        material_id: self.per_materials.len(),
                        primitive: primitive.clone(),
                    });
                    let mut material_info = MaterialInfo {
                        base_color: primitive.material.base_color_factor,
                        base_color_map_index: -1,
                        normal_map_index: -1,
                        normal_scale: 1.0,
                        specular: primitive.material.metallic_factor,
                    };
                    if let Some(texture) = &primitive.material.base_color_texture {
                        material_info.base_color_map_index = self.textures.len() as _;
                        self.textures.push(texture.texture.clone());
                        if let Some(normal_map) = &primitive.material.normal_texture {
                            material_info.normal_map_index = self.textures.len() as _;
                            self.textures.push(normal_map.texture.clone());
                            material_info.normal_scale = normal_map.scale;
                        }
                    }
                    self.per_materials.push(material_info);
                }
                self.per_meshes.push(PerMesh { transform });
                self.shadow_per_meshes.push(ShadowPerMesh { transform });
            }
        }
    }
}

pub(crate) struct Renderer {
    next_frame: AtomicUsize,
    frames_in_flight: usize,
    queue: vk::Queue,
    availables: Vec<vk::Semaphore>,
    presentables: Vec<vk::Semaphore>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    fences: Vec<vk::Fence>,

    frame_ubs: Vec<UniformBuffer<PerFrame>>,
    light_ubs: Vec<UniformBuffer<DirLight>>,
    mesh_dubs: Vec<DynamicUniformBuffer<PerMesh>>,
    material_dubs: Vec<DynamicUniformBuffer<MaterialInfo>>,

    set0_layout: vk::DescriptorSetLayout,
    set1_layout: vk::DescriptorSetLayout,
    set2_layout: vk::DescriptorSetLayout,

    descriptor_pool: vk::DescriptorPool,
    descriptor_sets0: Vec<vk::DescriptorSet>,
    descriptor_sets1: Vec<vk::DescriptorSet>,
    descriptor_sets2: Vec<vk::DescriptorSet>,

    shadow_pass: ShadowPass,
    skybox_pipeline: SkyboxPipeline,

    display: Display,
    depth_images: Vec<DepthImage>,
    msaa_images: Vec<MSAAImage>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl RendererTrait for Renderer {
    fn get_requirements() -> RendererRequirements {
        RendererRequirements {
            descriptor_binding_partially_bound: true,
            ..Default::default()
        }
    }

    fn new(create_info: &RendererCreateInfo) -> GraphicsResult<Self> {
        unsafe {
            let gfx = Graphics::get_ref();
            let mut renderer = Self::null();
            let frames_in_flight = renderer.frames_in_flight;

            renderer.queue = gfx.get_device_queue(2);

            for _ in 0..frames_in_flight {
                renderer.availables.push(gfx.create_semaphore()?);
                renderer.presentables.push(gfx.create_semaphore()?);
            }

            renderer.command_pool =
                gfx.create_command_pool(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)?;
            renderer.command_buffers = gfx.allocate_command_buffers(
                renderer.command_pool,
                vk::CommandBufferLevel::PRIMARY,
                frames_in_flight,
            )?;
            for _ in 0..frames_in_flight {
                renderer.fences.push(gfx.create_fence(true)?);
            }

            for _ in 0..frames_in_flight {
                renderer.frame_ubs.push(UniformBuffer::new()?);
                renderer.light_ubs.push(UniformBuffer::new()?);
                renderer.mesh_dubs.push(DynamicUniformBuffer::new(64)?);
                renderer.material_dubs.push(DynamicUniformBuffer::new(64)?);
            }

            renderer.set0_layout = gfx.create_descriptor_set_layout(
                &[
                    vk::DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        ..Default::default()
                    },
                    vk::DescriptorSetLayoutBinding {
                        binding: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        ..Default::default()
                    },
                    vk::DescriptorSetLayoutBinding {
                        binding: 2,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                        ..Default::default()
                    },
                    vk::DescriptorSetLayoutBinding {
                        binding: 3,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                        ..Default::default()
                    },
                ],
                &[],
            )?;
            renderer.set1_layout = gfx.create_descriptor_set_layout(
                &[vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    ..Default::default()
                }],
                &[],
            )?;
            renderer.set2_layout = gfx.create_descriptor_set_layout(
                &[
                    vk::DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        ..Default::default()
                    },
                    vk::DescriptorSetLayoutBinding {
                        binding: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: CIS_SIZED_ARRAY_COUNT,
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                        ..Default::default()
                    },
                ],
                &[
                    vk::DescriptorBindingFlags::empty(),
                    vk::DescriptorBindingFlags::PARTIALLY_BOUND,
                ],
            )?;

            renderer.descriptor_pool = gfx.create_descriptor_pool(
                &[
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: (frames_in_flight * 2) as _,
                    },
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                        descriptor_count: (frames_in_flight * 2) as _,
                    },
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: (frames_in_flight * 3) as _,
                    },
                ],
                frames_in_flight * 3,
                false,
            )?;

            renderer.descriptor_sets0 = gfx.allocate_descriptor_sets(
                renderer.descriptor_pool,
                renderer.set0_layout,
                frames_in_flight,
            )?;
            renderer.descriptor_sets1 = gfx.allocate_descriptor_sets(
                renderer.descriptor_pool,
                renderer.set1_layout,
                frames_in_flight,
            )?;
            renderer.descriptor_sets2 = gfx.allocate_descriptor_sets(
                renderer.descriptor_pool,
                renderer.set2_layout,
                frames_in_flight,
            )?;

            renderer
                .shadow_pass
                .recreate(frames_in_flight, vk::Format::D16_UNORM, 4096, 4096)?;

            for (i, set) in renderer.descriptor_sets0.iter().enumerate() {
                let buffer_info0 = vk::DescriptorBufferInfo {
                    buffer: renderer.frame_ubs[i].buf.raw,
                    offset: 0,
                    range: renderer.frame_ubs[i].buf.size as _,
                };
                let buffer_info1 = vk::DescriptorBufferInfo {
                    buffer: renderer.light_ubs[i].buf.raw,
                    offset: 0,
                    range: renderer.light_ubs[i].buf.size as _,
                };
                let shadow_map_info = vk::DescriptorImageInfo {
                    sampler: renderer.shadow_pass.shadow_maps[i].sampler,
                    image_view: renderer.shadow_pass.shadow_maps[i].view,
                    image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                };
                gfx.write_descriptor_sets(&[
                    vk::WriteDescriptorSet {
                        dst_set: *set,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        p_buffer_info: &buffer_info0,
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: *set,
                        dst_binding: 1,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        p_buffer_info: &buffer_info1,
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: *set,
                        dst_binding: 2,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &shadow_map_info,
                        ..Default::default()
                    },
                ]);
            }

            renderer.create_impl(create_info)?;
            Ok(renderer)
        }
    }

    fn recreate(&mut self, create_info: &RendererCreateInfo) -> GraphicsResult<()> {
        unsafe {
            self.destroy_impl();
            self.create_impl(create_info)?;
        }
        Ok(())
    }

    type RenderingDataType = RenderingData;

    fn render(&self, rendering_data: Self::RenderingDataType) -> GraphicsResult<()> {
        unsafe { self.render_impl(rendering_data) }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        let gfx = Graphics::get_ref();

        unsafe {
            self.destroy_impl();
            self.shadow_pass.destroy();

            self.descriptor_sets2.clear();
            self.descriptor_sets1.clear();
            self.descriptor_sets0.clear();
            gfx.destroy_descriptor_pool(self.descriptor_pool);

            gfx.destroy_descriptor_set_layout(self.set2_layout);
            gfx.destroy_descriptor_set_layout(self.set1_layout);
            gfx.destroy_descriptor_set_layout(self.set0_layout);

            self.material_dubs.clear();
            self.mesh_dubs.clear();
            self.light_ubs.clear();
            self.frame_ubs.clear();

            while let Some(fence) = self.fences.pop() {
                gfx.destroy_fence(fence);
            }
            if self.command_pool != vk::CommandPool::null() {
                gfx.free_command_buffers(self.command_pool, self.command_buffers.as_slice());
                self.command_buffers.clear();
                gfx.destroy_command_pool(self.command_pool);
                self.command_pool = vk::CommandPool::null();
            }

            while let Some(semaphore) = self.presentables.pop() {
                gfx.destroy_semaphore(semaphore);
            }
            while let Some(semaphore) = self.availables.pop() {
                gfx.destroy_semaphore(semaphore);
            }

            self.display.destroy();
        }
    }
}

impl Renderer {
    fn null() -> Self {
        Renderer {
            next_frame: AtomicUsize::new(0),
            frames_in_flight: 3,
            queue: vk::Queue::null(),
            availables: vec![],
            presentables: vec![],
            command_pool: vk::CommandPool::null(),
            command_buffers: vec![],
            fences: vec![],

            frame_ubs: vec![],
            light_ubs: vec![],
            mesh_dubs: vec![],
            material_dubs: vec![],

            set0_layout: vk::DescriptorSetLayout::null(),
            set1_layout: vk::DescriptorSetLayout::null(),
            set2_layout: vk::DescriptorSetLayout::null(),

            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_sets0: vec![],
            descriptor_sets1: vec![],
            descriptor_sets2: vec![],

            shadow_pass: ShadowPass::null(),
            skybox_pipeline: SkyboxPipeline::null(),

            display: Display::null(),
            depth_images: vec![],
            msaa_images: vec![],
            render_pass: vk::RenderPass::null(),
            framebuffers: vec![],
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
        }
    }

    unsafe fn create_impl(&mut self, create_info: &RendererCreateInfo) -> GraphicsResult<()> {
        let gfx = Graphics::get_ref();

        self.display.recreate(
            DisplayConfig {
                extent: vk::Extent2D {
                    width: create_info.width,
                    height: create_info.height,
                },
                ..Default::default()
            }
            .constraint()?,
        )?;

        let swapchain_image_count = self.display.images.len();
        let swapchain_extent = self.display.config.extent;

        let depth_format = vk::Format::D32_SFLOAT; // TODO
        let msaa_samples = convert_msaa_samples(create_info.samples);
        for _ in 0..swapchain_image_count {
            self.depth_images.push(DepthImage::new(
                depth_format,
                swapchain_extent,
                msaa_samples,
            )?);
            self.msaa_images.push(MSAAImage::new(
                self.display.config.format,
                swapchain_extent,
                msaa_samples,
            )?);
        }

        self.render_pass = {
            let color_attachment = vk::AttachmentDescription {
                format: self.display.config.format,
                samples: msaa_samples,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, // vk::ImageLayout::PRESENT_SRC_KHR,
                ..Default::default()
            };

            let color_attachment_ref = vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            };

            let depth_attachment = vk::AttachmentDescription {
                format: depth_format,
                samples: msaa_samples,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::DONT_CARE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                ..Default::default()
            };

            let depth_attachment_ref = vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            };

            let msaa_attachment = vk::AttachmentDescription {
                format: self.display.config.format,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::DONT_CARE,
                store_op: vk::AttachmentStoreOp::STORE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                ..Default::default()
            };

            let msaa_attachment_ref = vk::AttachmentReference {
                attachment: 2,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            };

            let subpass = vk::SubpassDescription {
                flags: vk::SubpassDescriptionFlags::empty(),
                input_attachment_count: 0,
                p_input_attachments: 0 as _,
                pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
                color_attachment_count: 1,
                p_color_attachments: &color_attachment_ref as _,
                p_resolve_attachments: &msaa_attachment_ref as _,
                p_depth_stencil_attachment: &depth_attachment_ref as _,
                ..Default::default()
            };

            let subpass_dependency = vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dependency_flags: vk::DependencyFlags::BY_REGION,
            };

            gfx.create_render_pass(
                &[color_attachment, depth_attachment, msaa_attachment],
                &[subpass],
                &[subpass_dependency],
            )?
        };

        for i in 0..swapchain_image_count {
            self.framebuffers.push(gfx.create_framebuffer(
                self.render_pass,
                &[
                    self.msaa_images[i].view,
                    self.depth_images[i].view,
                    self.display.image_views[i],
                ],
                swapchain_extent.width,
                swapchain_extent.height,
            )?);
        }

        self.pipeline_layout = gfx
            .create_pipeline_layout(&[self.set0_layout, self.set1_layout, self.set2_layout], &[])?;

        let attachments = &[vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::TRUE,
            src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::all(), // RGBA
        }];

        self.pipeline = {
            let vert = ShaderModule::new(include_bytes!("../../../shaders/phong.vert.spv"))?;
            let frag = ShaderModule::new(include_bytes!("../../../shaders/phong.frag.spv"))?;
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

            let vertex_binding_descs = &[
                vk::VertexInputBindingDescription {
                    binding: 0,
                    stride: size_of::<Vec3>() as _,
                    input_rate: vk::VertexInputRate::VERTEX,
                },
                vk::VertexInputBindingDescription {
                    binding: 1,
                    stride: size_of::<Vec3>() as _,
                    input_rate: vk::VertexInputRate::VERTEX,
                },
                vk::VertexInputBindingDescription {
                    binding: 2,
                    stride: size_of::<Vec4>() as _,
                    input_rate: vk::VertexInputRate::VERTEX,
                },
                vk::VertexInputBindingDescription {
                    binding: 3,
                    stride: size_of::<Vec2>() as _,
                    input_rate: vk::VertexInputRate::VERTEX,
                },
            ];

            let vertex_attributes_descs = &[
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: 0,
                },
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 1,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: 0,
                },
                vk::VertexInputAttributeDescription {
                    location: 2,
                    binding: 2,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: 0,
                },
                vk::VertexInputAttributeDescription {
                    location: 3,
                    binding: 3,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: 0,
                },
            ];

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
                    y: swapchain_extent.height as _,
                    width: swapchain_extent.width as _,
                    height: -(swapchain_extent.height as f32),
                    min_depth: 0.0,
                    max_depth: 1.0,
                },
                scissor: vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: swapchain_extent,
                },
                cull_mode: vk::CullModeFlags::BACK,
                front_face: vk::FrontFace::CLOCKWISE,
                rasterization_samples: msaa_samples,
                depth_test_enable: true,
                depth_write_enable: true,
                attachments,
                dynamic_states: &[],
                ..Default::default()
            };

            Graphics::get_ref().create_graphics_pipeline(&create_info)?
        };

        self.skybox_pipeline.recreate(
            self.frames_in_flight,
            self.render_pass,
            0,
            swapchain_extent.width,
            swapchain_extent.height,
            msaa_samples,
            attachments,
        )?;

        Ok(())
    }

    unsafe fn destroy_impl(&mut self) {
        let gfx = Graphics::get_ref();

        self.skybox_pipeline.destroy();

        gfx.destroy_pipeline(self.pipeline);
        self.pipeline = vk::Pipeline::null();
        gfx.destroy_pipeline_layout(self.pipeline_layout);
        self.pipeline_layout = vk::PipelineLayout::null();
        while let Some(framebuffer) = self.framebuffers.pop() {
            gfx.destroy_framebuffer(framebuffer);
        }
        gfx.destroy_render_pass(self.render_pass);
        self.render_pass = vk::RenderPass::null();
        self.depth_images.clear();
        self.msaa_images.clear();
    }

    fn acquire_next_frame(&self) -> usize {
        let current_frame = self.next_frame.load(Ordering::SeqCst);
        self.next_frame.store(
            (current_frame + 1) % self.frames_in_flight,
            Ordering::SeqCst,
        );
        current_frame
    }

    unsafe fn render_impl(&self, rendering_data: RenderingData) -> GraphicsResult<()> {
        let gfx = Graphics::get_ref();
        let tracer = ResourceTracer::get_ref();

        let current_frame = self.acquire_next_frame();
        let available = self.availables[current_frame];
        let presentable = self.presentables[current_frame];
        let command_buffer = self.command_buffers[current_frame];
        let fence = self.fences[current_frame];
        let frame_ub = &self.frame_ubs[current_frame];
        let light_ub = &self.light_ubs[current_frame];
        let material_dub = &self.material_dubs[current_frame];
        let mesh_dub = &self.mesh_dubs[current_frame];
        let descriptor_set0 = self.descriptor_sets0[current_frame];
        let descriptor_set1 = self.descriptor_sets1[current_frame];
        let descriptor_set2 = self.descriptor_sets2[current_frame];

        gfx.wait_for_fence(fence, u64::MAX)?;
        gfx.reset_fence(fence)?;

        let display = &self.display;
        let image_index = display.next_image(u64::MAX, available, vk::Fence::null())?;
        tracer.set_current_image_index(image_index);
        let framebuffer = self.framebuffers[image_index];

        // data updating

        let per_mesh_range = ((size_of::<<PerMesh as IntoSTD140>::Output>() + 255) >> 8) << 8;
        let per_material_range =
            ((size_of::<<MaterialInfo as IntoSTD140>::Output>() + 255) >> 8) << 8;

        if let Some(per_frame) = rendering_data.per_frame {
            frame_ub.update(&per_frame)?;
            mesh_dub.update(rendering_data.per_meshes.as_slice())?;
            material_dub.update(rendering_data.per_materials.as_slice())?;

            if let Some(skybox) = rendering_data.skybox.as_ref() {
                let skybox_info = vk::DescriptorImageInfo {
                    sampler: skybox.sampler.raw,
                    image_view: skybox.cubemap.view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                };

                gfx.write_descriptor_sets(&[vk::WriteDescriptorSet {
                    dst_set: descriptor_set0,
                    dst_binding: 3,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    p_image_info: &skybox_info,
                    ..Default::default()
                }]);

                tracer.touch_cubemap(skybox.cubemap.clone());
                tracer.touch_sampler(skybox.sampler.clone());
            }

            let mesh_dub_info = vk::DescriptorBufferInfo {
                buffer: mesh_dub.buf.raw,
                offset: 0,
                range: per_mesh_range as _,
            };
            let material_dub_info = vk::DescriptorBufferInfo {
                buffer: material_dub.buf.raw,
                offset: 0,
                range: per_material_range as _,
            };

            gfx.write_descriptor_sets(&[
                vk::WriteDescriptorSet {
                    dst_set: descriptor_set1,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    p_buffer_info: &mesh_dub_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: descriptor_set2,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    p_buffer_info: &material_dub_info,
                    ..Default::default()
                },
            ]);

            let image_infos = rendering_data
                .textures
                .iter()
                .map(|texture| {
                    tracer.touch_sampler(texture.sampler.clone());
                    tracer.touch_tex_image(texture.image.clone());
                    vk::DescriptorImageInfo {
                        sampler: texture.sampler.raw,
                        image_view: texture.image.view,
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    }
                })
                .collect::<Vec<_>>();

            if image_infos.len() > 0 {
                gfx.write_descriptor_sets(&[vk::WriteDescriptorSet {
                    dst_set: descriptor_set2,
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: image_infos.len() as _,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    p_image_info: image_infos.as_slice().as_ptr() as _,
                    ..Default::default()
                }]);
            }
        }

        if let Some(dir_light) = rendering_data.dir_light {
            light_ub.update(&dir_light)?;
            self.shadow_pass.prepare_draw(
                current_frame,
                rendering_data.shadow_per_frame.as_ref().unwrap(),
                rendering_data.shadow_per_meshes.as_slice(),
            )?;
        }

        if let Some(skybox) = rendering_data.skybox.as_ref() {
            let per_frame = rendering_data.skybox_per_frame.unwrap();
            self.skybox_pipeline
                .prepare_draw(current_frame, &per_frame, skybox.as_ref())?;
        }

        // rendering

        gfx.reset_command_buffer(command_buffer, false)?;
        gfx.begin_command_buffer(
            command_buffer,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
        )?;

        self.shadow_pass.draw(
            command_buffer,
            current_frame,
            rendering_data
                .info_list
                .iter()
                .map(|info| (info.mesh_id, &info.primitive)),
        );

        // Explicit synchronization is not required between the render pass, as this is done implicit via sub pass dependencies
        // how?

        gfx.cmd_begin_render_pass(
            command_buffer,
            self.render_pass,
            framebuffer,
            vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.display.config.extent,
            },
            &[
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: RendererData::get_ref().clear_color().into(),
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ],
            vk::SubpassContents::INLINE,
        );

        gfx.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline,
        );

        if rendering_data.per_frame.is_some() {
            gfx.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[descriptor_set0],
                &[],
            );

            for RenderInfo {
                mesh_id,
                material_id,
                primitive,
            } in rendering_data.info_list
            {
                gfx.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    1,
                    &[descriptor_set1, descriptor_set2],
                    &[
                        (per_mesh_range * mesh_id) as _,
                        (per_material_range * material_id) as _,
                    ],
                );

                let attributes = &primitive.attributes;

                let vertex_count = if let Some(position) = attributes.position.as_ref() {
                    gfx.cmd_bind_vertex_buffer(command_buffer, 0, position);
                    tracer.touch_buffer(position.buf.clone());
                    position.vertex_count()
                } else {
                    continue;
                };

                if let Some(normal) = attributes.normal.as_ref() {
                    gfx.cmd_bind_vertex_buffer(command_buffer, 1, normal);
                    tracer.touch_buffer(normal.buf.clone());
                } else {
                    continue;
                }

                if let Some(tangent) = attributes.tangent.as_ref() {
                    gfx.cmd_bind_vertex_buffer(command_buffer, 2, tangent);
                    tracer.touch_buffer(tangent.buf.clone());
                } else {
                    gfx.cmd_bind_vertex_buffer(command_buffer, 2, &VertexBuffer::get_default());
                }

                if let Some(texcoord) = attributes.texcoord0.as_ref() {
                    gfx.cmd_bind_vertex_buffer(command_buffer, 3, texcoord);
                    tracer.touch_buffer(texcoord.buf.clone());
                } else {
                    gfx.cmd_bind_vertex_buffer(command_buffer, 3, &VertexBuffer::get_default());
                }

                if let Some(indices) = primitive.indices.as_ref() {
                    gfx.cmd_bind_index_buffer(command_buffer, indices);
                    tracer.touch_buffer(indices.buf.clone());
                    gfx.cmd_draw_indexed(command_buffer, indices.index_count(), 1, 0, 0);
                } else {
                    gfx.cmd_draw(command_buffer, vertex_count, 1, 0, 0);
                }
            }
        }

        if rendering_data.skybox.is_some() {
            self.skybox_pipeline.draw(command_buffer, current_frame);
        }

        gfx.cmd_end_render_pass(command_buffer);

        gfx.end_command_buffer(command_buffer)?;

        gfx.queue_submit(
            self.queue,
            &[available],
            &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
            &[command_buffer],
            &[presentable],
            fence,
        )?;

        display.present(self.queue, &[presentable], image_index)?;
        Ok(())
    }
}
