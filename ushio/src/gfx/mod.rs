//! Graphics Module

use std::{
    intrinsics::copy_nonoverlapping,
    mem::size_of,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    usize,
};

use crate::{scene::Scene, Color, ConfigParams, Window};
use ash::{
    extensions::{ext as vkext, khr as vkkhr},
    vk,
};
use ash_window;
mod convert;
use lazy_static::lazy_static;
use log;
#[allow(dead_code)]
mod buf;
mod cubemap;
mod img;
mod shadow;
mod trace;
pub(crate) use buf::*;
use cubemap::*;
pub(crate) use img::*;
use shadow::*;
use thiserror::Error;
use trace::*;
use vk_mem as vma;

#[cfg(feature = "validation")]
unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        std::borrow::Cow::from("")
    } else {
        std::ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        std::borrow::Cow::from("")
    } else {
        std::ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    let level = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::Level::Debug,
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::Level::Info,
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::Level::Warn,
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::Level::Error,
        _ => panic!("Value of message severity crashed."),
    };

    log::log!(
        level,
        "{:?} [{} ({})] : {}\n",
        message_type,
        message_id_name,
        message_id_number,
        message,
    );

    vk::FALSE
}

#[derive(Debug, Error)]
pub enum GraphicsError {
    #[error("Load vulkan library error: {0}")]
    LoadLibrary(#[from] ash::LoadingError),
    #[error("Load instance api error: {0}")]
    LoadInstance(#[from] ash::InstanceError),
    #[error("Vulkan error from '{func}': {result}")]
    VkError {
        func: &'static str,
        result: vk::Result,
    },
    #[error("Vulkan memory allocator error: {0}")]
    VmaError(#[from] vma::Error),
    #[error("Other graphics init error: {0}")]
    Other(String),
}

trait VkErrorConvert<T> {
    fn gfx_vk_result(self, api: &'static str) -> Result<T, GraphicsError>;
}

impl<T> VkErrorConvert<T> for Result<T, vk::Result> {
    fn gfx_vk_result(self, func: &'static str) -> Result<T, GraphicsError> {
        match self {
            Ok(t) => Ok(t),
            Err(result) => Err(GraphicsError::VkError { func, result }),
        }
    }
}

type GraphicsResult<T> = Result<T, GraphicsError>;

macro_rules! gfx_error_clean {
    ($result: expr, $proc: expr) => {
        match $result {
            Ok(v) => Ok(v),
            Err(e) => {
                $proc;
                Err(e)
            }
        }
    };
}

fn make_version(major: u32, minor: u32, patch: u32) -> u32 {
    (major << 22) + (minor << 12) + patch
}

#[derive(Debug)]
pub struct GraphicsParams {
    pub app_name: String,
    pub app_version: u32,
}

impl Default for GraphicsParams {
    fn default() -> Self {
        GraphicsParams {
            app_name: "Rushio".to_string(),
            app_version: make_version(1, 0, 0),
        }
    }
}

struct VulkanContext {
    entry: ash::Entry,
    allocator: Option<vk::AllocationCallbacks>,
    instance: ash::Instance,
    surface_khr: vkkhr::Surface,
    #[cfg(feature = "validation")]
    debug_utils_ext: vkext::DebugUtils,
    #[cfg(feature = "validation")]
    debug_messenger: vk::DebugUtilsMessengerEXT,
    enabled_layer_names: Vec<*const i8>,
}

unsafe impl Sync for VulkanContext {}

lazy_static! {
    static ref S_VULKAN_CONTEXT: VulkanContext = unsafe {
        let params = &ConfigParams::read().graphics;
        let entry = match ash::Entry::new() {
            Ok(entry) => {
                log::info!("Loading vulkan library success!");
                entry
            }
            Err(e) => {
                log::error!("Failed to load vulkan library: {:?}", e);
                panic!();
            }
        };
        let allocator = None;

        let mut ext_names = ash_window::enumerate_required_extensions(&Window::get_ref().raw)
            .unwrap()
            .iter()
            .map(|s| s.as_ptr())
            .collect::<Vec<_>>();

        let mut enabled_layer_names = vec![];
        #[cfg(feature = "validation")]
        {
            enabled_layer_names.push("VK_LAYER_KHRONOS_validation".as_ptr() as *const _);
            ext_names.push(vkext::DebugUtils::name().as_ptr());
        }

        let app_info = vk::ApplicationInfo {
            p_application_name: params.app_name.as_ptr() as *const _,
            application_version: params.app_version,
            api_version: vk::API_VERSION_1_2,
            ..Default::default()
        };

        let instance_ci = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&ext_names)
            .enabled_layer_names(&enabled_layer_names);

        let instance = entry
            .create_instance(&instance_ci, allocator.as_ref())
            .unwrap();

        let surface_khr = vkkhr::Surface::new(&entry, &instance);
        #[cfg(feature = "validation")]
        let debug_utils_ext = vkext::DebugUtils::new(&entry, &instance);

        #[cfg(feature = "validation")]
        let debug_messenger = {
            let create_info = vk::DebugUtilsMessengerCreateInfoEXT {
                message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                message_type: vk::DebugUtilsMessageTypeFlagsEXT::all(),
                pfn_user_callback: Some(vulkan_debug_callback),
                ..Default::default()
            };
            match debug_utils_ext
                .create_debug_utils_messenger(&create_info, allocator.as_ref())
                .gfx_vk_result("VkCreateDebugUtilsMessengerEXT")
            {
                Ok(messenger) => messenger,
                Err(e) => {
                    log::warn!("Successfully load vulkan validation layer but failed to create debug messenger: {}", e);
                    vk::DebugUtilsMessengerEXT::null()
                }
            }
        };

        VulkanContext {
            entry,
            allocator,
            instance,
            surface_khr,
            #[cfg(feature = "validation")]
            debug_utils_ext,
            #[cfg(feature = "validation")]
            debug_messenger,
            enabled_layer_names,
        }
    };
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            #[cfg(feature = "validation")]
            self.debug_utils_ext
                .destroy_debug_utils_messenger(self.debug_messenger, self.allocation_callbacks());
            self.instance.destroy_instance(self.allocation_callbacks());
        }
    }
}

impl VulkanContext {
    fn get_ref() -> &'static Self {
        &S_VULKAN_CONTEXT
    }

    fn allocation_callbacks(&self) -> Option<&vk::AllocationCallbacks> {
        self.allocator.as_ref()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RendererRequirements {
    descriptor_binding_partially_bound: bool,
}

impl Default for RendererRequirements {
    fn default() -> Self {
        RendererRequirements {
            descriptor_binding_partially_bound: false,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RendererCreateInfo {
    pub width: u32,
    pub height: u32,
    pub samples: u32,
}

impl Default for RendererCreateInfo {
    fn default() -> Self {
        RendererCreateInfo {
            width: u32::default(),
            height: u32::default(),
            samples: u32::default(),
        }
    }
}

pub trait RendererDataTrait: Sized {
    fn get_ref() -> &'static Self;

    fn set_clear_color(&self, color: Color);

    fn clear_color(&self) -> Color;
}

pub(crate) trait RenderingDataTrait: Sized + Send {
    fn parse_scene(scene: Scene) -> Self;
}

pub(crate) trait RendererTrait: Sized {
    type RenderingDataType: RenderingDataTrait;

    fn get_requirements() -> RendererRequirements;

    fn new(create_info: &RendererCreateInfo) -> GraphicsResult<Self>;

    fn recreate(&mut self, create_info: &RendererCreateInfo) -> GraphicsResult<()>;

    fn render(&self, rendering_data: Self::RenderingDataType) -> GraphicsResult<()>;
}

#[cfg(feature = "renderer-phong")]
mod phong;
#[cfg(feature = "renderer-phong")]
pub use phong::RendererData;
#[cfg(feature = "renderer-phong")]
pub(crate) use phong::{Renderer, RenderingData};

pub(crate) struct Graphics {
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    device: ash::Device,
    swapchain_khr: vkkhr::Swapchain,
    allocator: vma::Allocator,
    transfer_queue: vk::Queue,
    transfer_command_pool: Mutex<vk::CommandPool>,
}

unsafe impl Sync for Graphics {}

lazy_static! {
    static ref S_GRAPHICS: Graphics = unsafe {
        let vctx = VulkanContext::get_ref();
        let renderer_reqs = Renderer::get_requirements();

        let surface = ash_window::create_surface(
            &vctx.entry,
            &vctx.instance,
            &Window::get_ref().raw,
            vctx.allocation_callbacks(),
        )
        .unwrap();

        let physical_devices = gfx_error_clean!(
            vctx.instance.enumerate_physical_devices(),
            vctx.surface_khr
                .destroy_surface(surface, vctx.allocation_callbacks())
        )
        .unwrap();

        let (physical_device, queue_family_index) = gfx_error_clean!(
            match physical_devices
                .iter()
                .map(|physical_device| {
                    vctx.instance
                        .get_physical_device_queue_family_properties(*physical_device)
                        .iter()
                        .enumerate()
                        .find_map(|(index, ref info)| {
                            if info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                && info.queue_flags.contains(vk::QueueFlags::COMPUTE)
                                && vctx
                                    .surface_khr
                                    .get_physical_device_surface_support(
                                        *physical_device,
                                        index as u32,
                                        surface,
                                    )
                                    // TODO
                                    .unwrap_or(false)
                            {
                                Some((*physical_device, index as u32))
                            } else {
                                None
                            }
                        })
                })
                .find_map(|v| v)
            {
                Some(v) => Ok(v),
                None => Err(GraphicsError::Other(
                    "Failed to pick physical device".into(),
                )),
            },
            vctx.surface_khr
                .destroy_surface(surface, vctx.allocation_callbacks())
        )
        .unwrap();

        let physical_device_properties = vctx
            .instance
            .get_physical_device_properties(physical_device);

        log::info!(
            "Pick physical device {}: {}",
            physical_device_properties.device_id,
            std::ffi::CStr::from_ptr(&physical_device_properties.device_name as *const _)
                .to_string_lossy()
        );

        let device = {
            let device_extension_names = [vkkhr::Swapchain::name().as_ptr()];
            let features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: vk::TRUE,
                sampler_anisotropy: vk::TRUE,
                ..Default::default()
            };
            // TODO
            let priorities = [0.0, 0.5, 1.0];
            let queue_cis = [vk::DeviceQueueCreateInfo {
                queue_family_index,
                queue_count: priorities.len() as _,
                p_queue_priorities: priorities.as_ptr(),
                ..Default::default()
            }];

            let descriptor_indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures {
                descriptor_binding_partially_bound: renderer_reqs.descriptor_binding_partially_bound as _,
                ..Default::default()
            };

            let p_next = &descriptor_indexing_features as *const _ as _;

            let create_info = vk::DeviceCreateInfo {
                p_next,
                queue_create_info_count: queue_cis.len() as _,
                p_queue_create_infos: queue_cis.as_ptr(),
                enabled_extension_count: device_extension_names.len() as _,
                pp_enabled_extension_names: device_extension_names.as_ptr(),
                enabled_layer_count: vctx.enabled_layer_names.len() as _,
                pp_enabled_layer_names: vctx.enabled_layer_names.as_ptr(),
                p_enabled_features: &features as _,
                ..Default::default()
            };

            gfx_error_clean!(
                vctx.instance.create_device(
                    physical_device,
                    &create_info,
                    vctx.allocation_callbacks()
                ),
                vctx.surface_khr
                    .destroy_surface(surface, vctx.allocation_callbacks())
            )
            .unwrap()
        };

        // let vertex_input_dynamic_fn = vk::ExtVertexInputDynamicStateFn::load(|name| unsafe {
        //     std::mem::transmute(vctx.entry.get_instance_proc_addr(vctx.instance.handle(), name.as_ptr()))
        // });

        let swapchain_khr = vkkhr::Swapchain::new(&vctx.instance, &device);

        let allocator = {
            let create_info = vk_mem::AllocatorCreateInfo {
                physical_device,
                device: device.clone(),
                instance: vctx.instance.clone(),
                flags: vk_mem::AllocatorCreateFlags::NONE,
                preferred_large_heap_block_size: 0,
                frame_in_use_count: 0,
                heap_size_limits: None,
            };
            gfx_error_clean!(vk_mem::Allocator::new(&create_info), {
                device.destroy_device(vctx.allocation_callbacks());
                vctx.surface_khr
                    .destroy_surface(surface, vctx.allocation_callbacks());
            })
            .unwrap()
        };

        let transfer_queue = device.get_device_queue(queue_family_index, 0);

        let gfx = Graphics {
            surface,
            physical_device,
            queue_family_index,
            device,
            swapchain_khr,
            allocator,
            transfer_queue,
            transfer_command_pool: Mutex::new(vk::CommandPool::null()),
        };
        *gfx.transfer_command_pool.lock().unwrap() =
            gfx.create_command_pool(vk::CommandPoolCreateFlags::TRANSIENT).unwrap();

        gfx
    };
}

impl Drop for Graphics {
    fn drop(&mut self) {
        unsafe {
            let vctx = VulkanContext::get_ref();
            self.destroy_command_pool(*self.transfer_command_pool.lock().unwrap());
            self.allocator.destroy();
            self.device.destroy_device(vctx.allocation_callbacks());
            vctx.surface_khr
                .destroy_surface(self.surface, vctx.allocation_callbacks());
        }
    }
}

impl Graphics {
    pub(crate) fn get_ref() -> &'static Graphics {
        &S_GRAPHICS
    }

    fn allocation_callbacks(&self) -> Option<&vk::AllocationCallbacks> {
        VulkanContext::get_ref().allocator.as_ref()
    }

    pub(crate) unsafe fn wait_idle(&self) -> GraphicsResult<()> {
        self.device
            .device_wait_idle()
            .gfx_vk_result("vkDeviceWaitIdle")
    }

    unsafe fn get_device_queue(&self, idx: u32) -> vk::Queue {
        self.device.get_device_queue(self.queue_family_index, idx)
    }

    unsafe fn create_semaphore(&self) -> GraphicsResult<vk::Semaphore> {
        let create_info = vk::SemaphoreCreateInfo::default();
        self.device
            .create_semaphore(&create_info, self.allocation_callbacks())
            .gfx_vk_result("VkCreateSemaphore")
    }

    unsafe fn destroy_semaphore(&self, semaphore: vk::Semaphore) {
        self.device
            .destroy_semaphore(semaphore, self.allocation_callbacks());
    }

    unsafe fn create_fence(&self, signaled: bool) -> GraphicsResult<vk::Fence> {
        let create_info = vk::FenceCreateInfo {
            flags: match signaled {
                true => vk::FenceCreateFlags::SIGNALED,
                false => vk::FenceCreateFlags::empty(),
            },
            ..Default::default()
        };
        self.device
            .create_fence(&create_info, self.allocation_callbacks())
            .gfx_vk_result("VkCreateFence")
    }

    unsafe fn destroy_fence(&self, fence: vk::Fence) {
        self.device
            .destroy_fence(fence, self.allocation_callbacks());
    }

    unsafe fn wait_for_fence(&self, fence: vk::Fence, timeout: u64) -> GraphicsResult<()> {
        self.device
            .wait_for_fences(&[fence], true, timeout)
            .gfx_vk_result("vkWaitForFences")
    }

    unsafe fn reset_fence(&self, fence: vk::Fence) -> GraphicsResult<()> {
        self.device
            .reset_fences(&[fence])
            .gfx_vk_result("vkResetFences")
    }

    unsafe fn create_command_pool(
        &self,
        flags: vk::CommandPoolCreateFlags,
    ) -> GraphicsResult<vk::CommandPool> {
        let create_info = vk::CommandPoolCreateInfo {
            flags,
            queue_family_index: self.queue_family_index,
            ..Default::default()
        };
        self.device
            .create_command_pool(&create_info, self.allocation_callbacks())
            .gfx_vk_result("VkCreateCommandPool")
    }

    unsafe fn destroy_command_pool(&self, pool: vk::CommandPool) {
        self.device
            .destroy_command_pool(pool, self.allocation_callbacks());
    }

    unsafe fn allocate_command_buffers(
        &self,
        command_pool: vk::CommandPool,
        level: vk::CommandBufferLevel,
        count: usize,
    ) -> GraphicsResult<Vec<vk::CommandBuffer>> {
        let create_info = vk::CommandBufferAllocateInfo {
            command_pool,
            level,
            command_buffer_count: count as _,
            ..Default::default()
        };
        self.device
            .allocate_command_buffers(&create_info)
            .gfx_vk_result("VkAllocateCommandBuffer")
    }

    unsafe fn free_command_buffers(
        &self,
        command_pool: vk::CommandPool,
        command_buffers: &[vk::CommandBuffer],
    ) {
        self.device
            .free_command_buffers(command_pool, command_buffers);
    }

    unsafe fn allocate_command_buffer(
        &self,
        command_pool: vk::CommandPool,
        level: vk::CommandBufferLevel,
    ) -> GraphicsResult<vk::CommandBuffer> {
        let create_info = vk::CommandBufferAllocateInfo {
            command_pool,
            level,
            command_buffer_count: 1,
            ..Default::default()
        };

        Ok(self
            .device
            .allocate_command_buffers(&create_info)
            .gfx_vk_result("VkAllocateCommandBuffer")?[0])
    }

    unsafe fn free_command_buffer(
        &self,
        command_pool: vk::CommandPool,
        command_buffer: vk::CommandBuffer,
    ) {
        self.device
            .free_command_buffers(command_pool, &[command_buffer]);
    }

    unsafe fn reset_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        release_resources: bool,
    ) -> GraphicsResult<()> {
        self.device
            .reset_command_buffer(
                command_buffer,
                match release_resources {
                    true => vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                    false => vk::CommandBufferResetFlags::empty(),
                },
            )
            .gfx_vk_result("vkResetCommandBuffer")
    }

    unsafe fn begin_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        flags: vk::CommandBufferUsageFlags,
    ) -> GraphicsResult<()> {
        let begin_info = vk::CommandBufferBeginInfo {
            flags,
            p_inheritance_info: 0 as _,
            ..Default::default()
        };

        self.device
            .begin_command_buffer(command_buffer, &begin_info)
            .gfx_vk_result("vkBeginCommandBuffer")
    }

    unsafe fn end_command_buffer(&self, command_buffer: vk::CommandBuffer) -> GraphicsResult<()> {
        Graphics::get_ref()
            .device
            .end_command_buffer(command_buffer)
            .gfx_vk_result("vkEndCommandBuffer")
    }

    unsafe fn queue_submit(
        &self,
        queue: vk::Queue,
        wait_semaphores: &[vk::Semaphore],
        wait_dst_stage_masks: &[vk::PipelineStageFlags],
        command_buffers: &[vk::CommandBuffer],
        signal_semaphores: &[vk::Semaphore],
        fence: vk::Fence,
    ) -> GraphicsResult<()> {
        let submit_info = vk::SubmitInfo {
            wait_semaphore_count: wait_semaphores.len() as _,
            p_wait_semaphores: wait_semaphores.as_ptr() as _,
            p_wait_dst_stage_mask: wait_dst_stage_masks.as_ptr() as _,
            command_buffer_count: command_buffers.len() as _,
            p_command_buffers: command_buffers.as_ptr() as _,
            signal_semaphore_count: signal_semaphores.len() as _,
            p_signal_semaphores: signal_semaphores.as_ptr() as _,
            ..Default::default()
        };

        self.device
            .queue_submit(queue, &[submit_info], fence)
            .gfx_vk_result("vkQueueSubmit")
    }

    unsafe fn create_descriptor_set_layout(
        &self,
        bindings: &[vk::DescriptorSetLayoutBinding],
        binding_flags: &[vk::DescriptorBindingFlags],
    ) -> GraphicsResult<vk::DescriptorSetLayout> {
        assert!(binding_flags.is_empty() || binding_flags.len() == bindings.len());

        let binding_create_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo {
            binding_count: bindings.len() as _,
            p_binding_flags: binding_flags.as_ptr(),
            ..Default::default()
        };

        let p_next = match binding_flags.is_empty() {
            true => std::ptr::null(),
            false => &binding_create_info as *const _ as _,
        };

        let create_info = vk::DescriptorSetLayoutCreateInfo {
            p_next,
            binding_count: bindings.len() as _,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        self.device
            .create_descriptor_set_layout(&create_info, self.allocation_callbacks())
            .gfx_vk_result("VkCreateDescriptorSetLayout")
    }

    unsafe fn destroy_descriptor_set_layout(&self, layout: vk::DescriptorSetLayout) {
        self.device
            .destroy_descriptor_set_layout(layout, self.allocation_callbacks());
    }

    unsafe fn create_descriptor_pool(
        &self,
        pool_sizes: &[vk::DescriptorPoolSize],
        max_sets: usize,
        individually_free: bool,
    ) -> GraphicsResult<vk::DescriptorPool> {
        let flags = match individually_free {
            true => vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            false => vk::DescriptorPoolCreateFlags::empty(),
        };
        let create_info = vk::DescriptorPoolCreateInfo {
            flags,
            max_sets: max_sets as _,
            pool_size_count: pool_sizes.len() as _,
            p_pool_sizes: pool_sizes.as_ptr(),
            ..Default::default()
        };

        self.device
            .create_descriptor_pool(&create_info, self.allocation_callbacks())
            .gfx_vk_result("VkCreateDescriptorPool")
    }

    unsafe fn destroy_descriptor_pool(&self, pool: vk::DescriptorPool) {
        self.device
            .destroy_descriptor_pool(pool, self.allocation_callbacks());
    }

    unsafe fn allocate_descriptor_sets(
        &self,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
        count: usize,
    ) -> GraphicsResult<Vec<vk::DescriptorSet>> {
        let layouts = std::iter::repeat(layout).take(count).collect::<Vec<_>>();

        let create_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: pool,
            descriptor_set_count: layouts.len() as _,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        };

        self.device
            .allocate_descriptor_sets(&create_info)
            .gfx_vk_result("VkAllocateDescriptorSet")
    }

    unsafe fn write_descriptor_sets(&self, descriptor_writes: &[vk::WriteDescriptorSet]) {
        self.device.update_descriptor_sets(descriptor_writes, &[]);
    }

    unsafe fn create_pipeline_layout(
        &self,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> GraphicsResult<vk::PipelineLayout> {
        let create_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: descriptor_set_layouts.len() as _,
            p_set_layouts: descriptor_set_layouts.as_ptr(),
            push_constant_range_count: push_constant_ranges.len() as _,
            p_push_constant_ranges: push_constant_ranges.as_ptr(),
            ..Default::default()
        };

        self.device
            .create_pipeline_layout(&create_info, self.allocation_callbacks())
            .gfx_vk_result("VkCreatePipelineLayout")
    }

    unsafe fn destroy_pipeline_layout(&self, pipeline_layout: vk::PipelineLayout) {
        self.device
            .destroy_pipeline_layout(pipeline_layout, self.allocation_callbacks());
    }

    unsafe fn create_render_pass(
        &self,
        attachment_descs: &[vk::AttachmentDescription],
        subpass_descs: &[vk::SubpassDescription],
        dependencies: &[vk::SubpassDependency],
    ) -> GraphicsResult<vk::RenderPass> {
        let create_info = vk::RenderPassCreateInfo {
            attachment_count: attachment_descs.len() as _,
            p_attachments: attachment_descs.as_ptr(),
            subpass_count: subpass_descs.len() as _,
            p_subpasses: subpass_descs.as_ptr(),
            dependency_count: dependencies.len() as _,
            p_dependencies: dependencies.as_ptr(),
            ..Default::default()
        };

        self.device
            .create_render_pass(&create_info, self.allocation_callbacks())
            .gfx_vk_result("vkCreateRenderPass")
    }

    unsafe fn destroy_render_pass(&self, render_pass: vk::RenderPass) {
        self.device
            .destroy_render_pass(render_pass, self.allocation_callbacks());
    }

    unsafe fn cmd_begin_render_pass(
        &self,
        command_buffer: vk::CommandBuffer,
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
        render_area: vk::Rect2D,
        clear_values: &[vk::ClearValue],
        subpass_contents: vk::SubpassContents,
    ) {
        let create_info = vk::RenderPassBeginInfo {
            render_pass,
            framebuffer,
            render_area,
            clear_value_count: clear_values.len() as _,
            p_clear_values: clear_values.as_ptr() as _,
            ..Default::default()
        };

        self.device
            .cmd_begin_render_pass(command_buffer, &create_info, subpass_contents);
    }

    unsafe fn cmd_end_render_pass(&self, command_buffer: vk::CommandBuffer) {
        self.device.cmd_end_render_pass(command_buffer);
    }

    unsafe fn cmd_bind_descriptor_sets(
        &self,
        command_buffer: vk::CommandBuffer,
        bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first_set_id: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            bind_point,
            layout,
            first_set_id,
            descriptor_sets,
            dynamic_offsets,
        );
    }

    unsafe fn cmd_bind_pipeline(
        &self,
        command_buffer: vk::CommandBuffer,
        bind_point: vk::PipelineBindPoint,
        pipeline: vk::Pipeline,
    ) {
        self.device
            .cmd_bind_pipeline(command_buffer, bind_point, pipeline);
    }

    unsafe fn cmd_bind_vertex_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        binding_id: u32,
        buffer: &VertexBuffer,
    ) {
        self.device
            .cmd_bind_vertex_buffers(command_buffer, binding_id, &[buffer.buf.raw], &[0]);
    }

    unsafe fn cmd_bind_index_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        buffer: &IndexBuffer,
    ) {
        let index_type = match buffer.info.component_type {
            ComponentType::Byte => vk::IndexType::UINT8_EXT,
            ComponentType::UnsignedShort => vk::IndexType::UINT16,
            ComponentType::UnsignedInt => vk::IndexType::UINT32,
            ty => panic!("Unexpected index type: {:?}", ty),
        };
        self.device
            .cmd_bind_index_buffer(command_buffer, buffer.buf.raw, 0, index_type);
    }

    unsafe fn cmd_draw(
        &self,
        command_buffer: vk::CommandBuffer,
        vertex_count: usize,
        instance_count: usize,
        first_vertex: usize,
        first_instance: usize,
    ) {
        self.device.cmd_draw(
            command_buffer,
            vertex_count as _,
            instance_count as _,
            first_vertex as _,
            first_instance as _,
        );
    }

    unsafe fn cmd_draw_indexed(
        &self,
        command_buffer: vk::CommandBuffer,
        index_count: usize,
        instance_count: usize,
        first_index: usize,
        first_instance: usize,
    ) {
        self.device.cmd_draw_indexed(
            command_buffer,
            index_count as _,
            instance_count as _,
            first_index as _,
            0,
            first_instance as _,
        );
    }

    // unsafe fn cmd_set_vertex_input_ext(
    //     &self,
    //     command_buffer: vk::CommandBuffer,
    //     vertex_binding_descs: &[vk::VertexInputBindingDescription],
    //     vertex_attributes_descs: &[vk::VertexInputAttributeDescription],
    // ) {
    //     let vertex_binding_descs = vertex_binding_descs
    //         .iter()
    //         .map(|desc| vk::VertexInputBindingDescription2EXT {
    //             binding: desc.binding,
    //             stride: desc.stride,
    //             input_rate: desc.input_rate,
    //             divisor: 1,
    //             ..Default::default()
    //         })
    //         .collect::<Vec<_>>();

    //     let vertex_attributes_descs = vertex_attributes_descs
    //         .iter()
    //         .map(|desc| vk::VertexInputAttributeDescription2EXT {
    //             location: desc.location,
    //             binding: desc.binding,
    //             format: desc.format,
    //             offset: desc.offset,
    //             ..Default::default()
    //         })
    //         .collect::<Vec<_>>();

    //     self.vertex_input_dynamic_fn.cmd_set_vertex_input_ext(
    //         command_buffer,
    //         vertex_binding_descs.len() as _,
    //         vertex_binding_descs.as_ptr(),
    //         vertex_attributes_descs.len() as _,
    //         vertex_attributes_descs.as_ptr(),
    //     );
    // }

    unsafe fn create_framebuffer(
        &self,
        render_pass: vk::RenderPass,
        attachments: &[vk::ImageView],
        width: u32,
        height: u32,
    ) -> GraphicsResult<vk::Framebuffer> {
        let create_info = vk::FramebufferCreateInfo {
            render_pass,
            attachment_count: attachments.len() as _,
            p_attachments: attachments.as_ptr(),
            width,
            height,
            layers: 1,
            ..Default::default()
        };

        self.device
            .create_framebuffer(&create_info, self.allocation_callbacks())
            .gfx_vk_result("VkCreateFramebuffer")
    }

    unsafe fn destroy_framebuffer(&self, framebuffer: vk::Framebuffer) {
        self.device
            .destroy_framebuffer(framebuffer, self.allocation_callbacks());
    }

    unsafe fn create_graphics_pipeline<'a>(
        &self,
        create_info: &GraphicsPipelineCreateInfo<'a>,
    ) -> GraphicsResult<vk::Pipeline> {
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: create_info.vertex_binding_descs.len() as _,
            p_vertex_binding_descriptions: create_info.vertex_binding_descs.as_ptr(),
            vertex_attribute_description_count: create_info.vertex_attributes_descs.len() as _,
            p_vertex_attribute_descriptions: create_info.vertex_attributes_descs.as_ptr(),
            ..Default::default()
        };

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
            topology: create_info.topology,
            primitive_restart_enable: create_info.restart_enable as _,
            ..Default::default()
        };

        let tessellation_state = vk::PipelineTessellationStateCreateInfo {
            patch_control_points: create_info.patch_control_points,
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            p_viewports: &create_info.viewport as _,
            scissor_count: 1,
            p_scissors: &create_info.scissor as _,
            ..Default::default()
        };

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: create_info.depth_clamp_enable as _,
            rasterizer_discard_enable: create_info.rasterization_discard_enable as _,
            polygon_mode: create_info.polygon_mode,
            cull_mode: create_info.cull_mode,
            front_face: create_info.front_face,
            depth_bias_enable: create_info.depth_bias_enable as _,
            depth_bias_constant_factor: create_info.depth_bias_constant_factor,
            depth_bias_clamp: create_info.depth_bias_clamp,
            depth_bias_slope_factor: create_info.depth_bias_slope_factor,
            line_width: create_info.line_width,
            ..Default::default()
        };

        let multisample_state = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: create_info.rasterization_samples,
            sample_shading_enable: create_info.sample_shading_enable as _,
            min_sample_shading: create_info.min_sample_shading,
            p_sample_mask: 0 as _,
            alpha_to_coverage_enable: create_info.alpha_to_coverage_enable as _,
            alpha_to_one_enable: create_info.alpha_to_one_enable as _,
            ..Default::default()
        };

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: create_info.depth_test_enable as _,
            depth_write_enable: create_info.depth_write_enable as _,
            depth_compare_op: create_info.depth_compare_op,
            depth_bounds_test_enable: create_info.depth_bounds_test_enable as _,
            stencil_test_enable: vk::FALSE,
            front: vk::StencilOpState::default(),
            back: vk::StencilOpState::default(),
            min_depth_bounds: create_info.min_depth_bounds,
            max_depth_bounds: create_info.max_depth_bounds,
            ..Default::default()
        };

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            logic_op_enable: create_info.logic_blend_op_enable as _,
            logic_op: create_info.logic_blend_op,
            attachment_count: create_info.attachments.len() as _,
            p_attachments: create_info.attachments.as_ptr(),
            blend_constants: create_info.blend_constants,
            ..Default::default()
        };

        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: create_info.dynamic_states.len() as _,
            p_dynamic_states: create_info.dynamic_states.as_ptr(),
            ..Default::default()
        };

        let raw_create_info = vk::GraphicsPipelineCreateInfo {
            stage_count: create_info.shader_stages.len() as _,
            p_stages: create_info.shader_stages.as_ptr(),
            p_vertex_input_state: &vertex_input_state as _,
            p_input_assembly_state: &input_assembly_state as _,
            p_tessellation_state: &tessellation_state as _,
            p_viewport_state: &viewport_state as _,
            p_rasterization_state: &rasterization_state as _,
            p_multisample_state: &multisample_state as _,
            p_depth_stencil_state: &depth_stencil_state as _,
            p_color_blend_state: &color_blend_state as _,
            p_dynamic_state: &dynamic_state as _,
            layout: create_info.layout,
            render_pass: create_info.render_pass,
            subpass: create_info.subpass,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: 0,
            ..Default::default()
        };

        match self.device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            &[raw_create_info],
            self.allocation_callbacks(),
        ) {
            Ok(v) => Ok(v[0]),
            Err((_, result)) => Err(GraphicsError::VkError {
                func: "vkCreateGraphicsPipelines",
                result,
            }),
        }
    }

    unsafe fn destroy_pipeline(&self, pipeline: vk::Pipeline) {
        self.device
            .destroy_pipeline(pipeline, self.allocation_callbacks());
    }

    unsafe fn create_buffer(
        &self,
        create_info: &BufferCreateInfo,
    ) -> GraphicsResult<(vk::Buffer, vma::Allocation, vma::AllocationInfo)> {
        let buffer_info = vk::BufferCreateInfo {
            size: create_info.size as _,
            usage: create_info.buffer_usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 1,
            p_queue_family_indices: &self.queue_family_index as _,
            ..Default::default()
        };

        let allocation_info = vma::AllocationCreateInfo {
            usage: create_info.memory_usage,
            flags: vma::AllocationCreateFlags::empty(),
            required_flags: create_info.memory_properties,
            preferred_flags: vk::MemoryPropertyFlags::empty(),
            memory_type_bits: u32::MAX,
            pool: None,
            user_data: None,
        };

        Ok(self
            .allocator
            .create_buffer(&buffer_info, &allocation_info)?)
    }

    unsafe fn destroy_buffer(&self, buffer: vk::Buffer, allocation: &vma::Allocation) {
        self.allocator.destroy_buffer(buffer, allocation).unwrap();
    }

    unsafe fn map_memory(&self, allocation: &vma::Allocation) -> GraphicsResult<*mut u8> {
        Ok(self.allocator.map_memory(allocation)?)
    }

    unsafe fn unmap_memory(&self, allocation: &vma::Allocation) -> GraphicsResult<()> {
        self.allocator.unmap_memory(allocation)?;
        Ok(())
    }

    unsafe fn map_copy_memory(
        &self,
        data: &[u8],
        allocation: &vma::Allocation,
        offset: isize,
    ) -> GraphicsResult<()> {
        let ptr = self.map_memory(allocation)?;
        copy_nonoverlapping(data.as_ptr(), ptr.offset(offset), data.len());
        self.unmap_memory(allocation)?;
        Ok(())
    }

    fn transfer_buffer(
        &self,
        src: &StagingBuffer,
        dst: &GpuBuffer,
        regions: &[vk::BufferCopy],
    ) -> GraphicsResult<()> {
        let commands = TransferCommands::new(&self.transfer_command_pool)?;
        let cb = commands.command_buffer;
        let fence = commands.fence;

        unsafe {
            self.begin_command_buffer(cb, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;
            self.device.cmd_copy_buffer(cb, src.raw, dst.raw, regions);
            self.end_command_buffer(cb)?;
            self.queue_submit(self.transfer_queue, &[], &[], &[cb], &[], fence)?;
            self.wait_for_fence(fence, u64::MAX)?;
        }

        Ok(())
    }

    unsafe fn create_image(
        &self,
        create_info: &ImageCreateInfo,
    ) -> GraphicsResult<(vk::Image, vma::Allocation, vma::AllocationInfo)> {
        let image_info = vk::ImageCreateInfo {
            flags: create_info.flags,
            image_type: create_info.image_type,
            format: create_info.format,
            extent: create_info.extent,
            mip_levels: create_info.mip_levels,
            array_layers: create_info.array_layers,
            samples: create_info.samples,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: create_info.usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 1,
            p_queue_family_indices: &self.queue_family_index as _,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let allocation_info = vma::AllocationCreateInfo {
            usage: vma::MemoryUsage::GpuOnly,
            flags: vma::AllocationCreateFlags::empty(),
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            preferred_flags: vk::MemoryPropertyFlags::empty(),
            memory_type_bits: u32::MAX,
            pool: None,
            user_data: None,
        };

        match self.allocator.create_image(&image_info, &allocation_info) {
            Ok(tuple) => Ok(tuple),
            Err(result) => Err(GraphicsError::VmaError(result)),
        }
    }

    unsafe fn destroy_image(&self, image: vk::Image, allocation: &vma::Allocation) {
        self.allocator.destroy_image(image, allocation).unwrap();
    }

    unsafe fn cmd_transition_image_layout(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        subresource_range: vk::ImageSubresourceRange,
    ) {
        let mut barrier = vk::ImageMemoryBarrier {
            old_layout,
            new_layout,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            subresource_range,
            ..Default::default()
        };

        let src_stage_mask = match old_layout {
            vk::ImageLayout::UNDEFINED => {
                barrier.src_access_mask = vk::AccessFlags::empty();
                vk::PipelineStageFlags::TOP_OF_PIPE
            }
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => {
                barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                vk::PipelineStageFlags::TRANSFER
            }
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => {
                barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
                vk::PipelineStageFlags::TRANSFER
            }
            _ => {
                log::error!(
                    "Unsupported image layout transition: from {:?} to {:?}",
                    old_layout,
                    new_layout
                );
                panic!();
            }
        };

        let dst_stage_mask = match new_layout {
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => {
                barrier.dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                vk::PipelineStageFlags::TRANSFER
            }
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => {
                barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
                vk::PipelineStageFlags::TRANSFER
            }
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
                barrier.dst_access_mask = vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
            }
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => {
                barrier.dst_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            }
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => {
                barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
                vk::PipelineStageFlags::FRAGMENT_SHADER
            }
            _ => {
                log::error!(
                    "Unsupported image layout transition: from {:?} to {:?}",
                    old_layout,
                    new_layout
                );
                panic!();
            }
        };

        self.device.cmd_pipeline_barrier(
            command_buffer,
            src_stage_mask,
            dst_stage_mask,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    fn transition_image_layout(
        &self,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        subresource_range: vk::ImageSubresourceRange,
    ) -> GraphicsResult<()> {
        let commands = TransferCommands::new(&self.transfer_command_pool)?;
        let cb = commands.command_buffer;
        let fence = commands.fence;
        unsafe {
            self.begin_command_buffer(
                commands.command_buffer,
                vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            )?;
            self.cmd_transition_image_layout(cb, image, old_layout, new_layout, subresource_range);
            self.end_command_buffer(cb)?;
            self.queue_submit(self.transfer_queue, &[], &[], &[cb], &[], fence)?;
            self.wait_for_fence(fence, u64::MAX)?;
        }
        Ok(())
    }

    fn transfer_buffer_to_image(
        &self,
        src: &StagingBuffer,
        dst: &GpuImage,
        regions: &[vk::BufferImageCopy],
    ) -> GraphicsResult<()> {
        let commands = TransferCommands::new(&self.transfer_command_pool)?;
        let cb = commands.command_buffer;
        let fence = commands.fence;

        let mut subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: dst.info.mip_levels,
            base_array_layer: 0,
            layer_count: dst.info.array_layers,
        };

        let mut src_subresource = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: dst.info.array_layers,
        };

        let mut width = dst.info.extent.width as _;
        let mut height = dst.info.extent.height as _;

        unsafe {
            self.begin_command_buffer(cb, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;

            self.cmd_transition_image_layout(
                cb,
                dst.raw,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                subresource_range,
            );
            self.device.cmd_copy_buffer_to_image(
                cb,
                src.raw,
                dst.raw,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                regions,
            );

            subresource_range.level_count = 1;
            for i in regions.len() as _..dst.info.mip_levels {
                subresource_range.base_mip_level = i - 1;
                self.cmd_transition_image_layout(
                    cb,
                    dst.raw,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    subresource_range,
                );

                let mut dst_subresource = src_subresource;
                dst_subresource.mip_level = i;
                let next_w = width / 2;
                let next_h = height / 2;
                let blit = vk::ImageBlit {
                    src_subresource,
                    src_offsets: [
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: width,
                            y: height,
                            z: 1,
                        },
                    ],
                    dst_subresource,
                    dst_offsets: [
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: next_w,
                            y: next_h,
                            z: 1,
                        },
                    ],
                };

                src_subresource = dst_subresource;
                width = next_w;
                height = next_h;

                self.device.cmd_blit_image(
                    cb,
                    dst.raw,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst.raw,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::LINEAR,
                );

                self.cmd_transition_image_layout(
                    cb,
                    dst.raw,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    subresource_range,
                );
            }

            subresource_range.base_mip_level = dst.info.mip_levels - 1;
            self.cmd_transition_image_layout(
                cb,
                dst.raw,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                subresource_range,
            );

            self.end_command_buffer(cb)?;
            self.queue_submit(self.transfer_queue, &[], &[], &[cb], &[], fence)?;
            self.wait_for_fence(fence, u64::MAX)?;
        }

        Ok(())
    }

    unsafe fn create_image_view(
        &self,
        create_info: &vk::ImageViewCreateInfo,
    ) -> GraphicsResult<vk::ImageView> {
        self.device
            .create_image_view(create_info, self.allocation_callbacks())
            .gfx_vk_result("vkCreateImageView")
    }

    unsafe fn destroy_image_view(&self, view: vk::ImageView) {
        self.device
            .destroy_image_view(view, self.allocation_callbacks());
    }

    unsafe fn create_sampler(
        &self,
        create_info: &vk::SamplerCreateInfo,
    ) -> GraphicsResult<vk::Sampler> {
        self.device
            .create_sampler(create_info, self.allocation_callbacks())
            .gfx_vk_result("vkCreateSampler")
    }

    unsafe fn destroy_sampler(&self, sampler: vk::Sampler) {
        self.device
            .destroy_sampler(sampler, self.allocation_callbacks());
    }
}

struct ShaderModule(vk::ShaderModule);

impl ShaderModule {
    fn new(code: &[u8]) -> GraphicsResult<ShaderModule> {
        let create_info = vk::ShaderModuleCreateInfo {
            code_size: code.len(),
            p_code: code.as_ptr() as _,
            ..Default::default()
        };

        unsafe {
            let raw = Graphics::get_ref()
                .device
                .create_shader_module(&create_info, Graphics::get_ref().allocation_callbacks())
                .gfx_vk_result("vkCreateShaderModule")?;
            Ok(ShaderModule(raw))
        }
    }

    fn raw(&self) -> vk::ShaderModule {
        self.0
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            Graphics::get_ref()
                .device
                .destroy_shader_module(self.raw(), Graphics::get_ref().allocation_callbacks());
        }
    }
}

#[derive(Debug, Clone)]
struct GraphicsPipelineCreateInfo<'a> {
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    subpass: u32,
    shader_stages: &'a [vk::PipelineShaderStageCreateInfo],
    // ---- vertex input state ----
    vertex_binding_descs: &'a [vk::VertexInputBindingDescription],
    vertex_attributes_descs: &'a [vk::VertexInputAttributeDescription],
    // ---- input assembly state ----
    topology: vk::PrimitiveTopology,
    /// Controls whether a special vertex index value is treated as restarting the assembly of primitives.
    /// This enable only applies to indexed draws (vkCmdDrawIndexed, vkCmdDrawMultiIndexedEXT, and vkCmdDrawIndexedIndirect),
    /// and the special index value is either 0xFFFFFFFF when the indexType parameter of vkCmdBindIndexBuffer
    /// is equal to VK_INDEX_TYPE_UINT32, 0xFF when indexType is equal to VK_INDEX_TYPE_UINT8_EXT, or 0xFFFF
    /// when indexType is equal to VK_INDEX_TYPE_UINT16. Primitive restart is not allowed for 'list' topologies.
    restart_enable: bool,
    // ---- tessellation state ----
    /// The number of control points per patch.
    /// Ignored if the pipeline does not include a tessellation control shader stage and tessellation evaluation shader stage.
    patch_control_points: u32,
    // ---- viewport state ----
    viewport: vk::Viewport,
    scissor: vk::Rect2D,
    // ---- rasterization state ----
    depth_clamp_enable: bool,
    rasterization_discard_enable: bool,
    polygon_mode: vk::PolygonMode,
    cull_mode: vk::CullModeFlags,
    front_face: vk::FrontFace,
    depth_bias_enable: bool,
    depth_bias_constant_factor: f32,
    depth_bias_clamp: f32,
    depth_bias_slope_factor: f32,
    line_width: f32,
    // ---- multisample state ----
    rasterization_samples: vk::SampleCountFlags,
    sample_shading_enable: bool,
    min_sample_shading: f32,
    /// Controls whether a temporary coverage value is generated based on the alpha component
    /// of the fragment’s first color output as specified in the 'Multisample Coverage' section.
    alpha_to_coverage_enable: bool,
    /// Controls whether the alpha component of the fragment’s first color output is replaced
    /// with one as described in 'Multisample Coverage'.
    alpha_to_one_enable: bool,
    // ---- depth stencil state ----
    depth_test_enable: bool,
    depth_write_enable: bool,
    depth_compare_op: vk::CompareOp,
    depth_bounds_test_enable: bool,
    min_depth_bounds: f32,
    max_depth_bounds: f32,
    // ---- color blend state ----
    logic_blend_op_enable: bool,
    logic_blend_op: vk::LogicOp,
    attachments: &'a [vk::PipelineColorBlendAttachmentState],
    /// An array of four values used as the R, G, B, and A components of the blend constant that are used
    /// in blending, depending on the blend factor.
    blend_constants: [f32; 4],
    // ---- dynamic states ----
    dynamic_states: &'a [vk::DynamicState],
}

impl<'a> Default for GraphicsPipelineCreateInfo<'a> {
    fn default() -> Self {
        GraphicsPipelineCreateInfo {
            layout: vk::PipelineLayout::null(),
            render_pass: vk::RenderPass::null(),
            subpass: u32::default(),
            shader_stages: &[],
            vertex_binding_descs: &[],
            vertex_attributes_descs: &[],
            topology: vk::PrimitiveTopology::default(),
            restart_enable: false,
            patch_control_points: u32::default(),
            viewport: vk::Viewport::default(),
            scissor: vk::Rect2D::default(),
            depth_clamp_enable: false,
            rasterization_discard_enable: false,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::default(),
            front_face: vk::FrontFace::default(),
            depth_bias_enable: false,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: false,
            min_sample_shading: 1.0,
            alpha_to_coverage_enable: false,
            alpha_to_one_enable: false,
            depth_test_enable: bool::default(),
            depth_write_enable: bool::default(),
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
            depth_bounds_test_enable: false,
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
            logic_blend_op_enable: false,
            logic_blend_op: vk::LogicOp::default(),
            attachments: &[],
            blend_constants: [0.0, 0.0, 0.0, 0.0],
            dynamic_states: &[],
        }
    }
}

#[derive(Debug, Clone)]
struct BufferCreateInfo {
    size: usize,
    buffer_usage: vk::BufferUsageFlags,
    memory_usage: vma::MemoryUsage,
    memory_properties: vk::MemoryPropertyFlags,
}

impl Default for BufferCreateInfo {
    fn default() -> Self {
        BufferCreateInfo {
            size: usize::default(),
            buffer_usage: vk::BufferUsageFlags::default(),
            memory_usage: vma::MemoryUsage::Unknown,
            memory_properties: vk::MemoryPropertyFlags::default(),
        }
    }
}

#[derive(Debug, Clone)]
struct ImageCreateInfo {
    flags: vk::ImageCreateFlags,
    image_type: vk::ImageType,
    format: vk::Format,
    extent: vk::Extent3D,
    usage: vk::ImageUsageFlags,
    mip_levels: u32,
    array_layers: u32,
    samples: vk::SampleCountFlags,
}

impl Default for ImageCreateInfo {
    fn default() -> Self {
        ImageCreateInfo {
            flags: vk::ImageCreateFlags::default(),
            image_type: vk::ImageType::default(),
            format: vk::Format::default(),
            extent: vk::Extent3D::default(),
            usage: vk::ImageUsageFlags::default(),
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
        }
    }
}
pub(crate) struct StagingBuffer {
    raw: vk::Buffer,
    allocation: vma::Allocation,
}

impl StagingBuffer {
    pub(crate) fn new<T: Sized>(data: &[T]) -> GraphicsResult<StagingBuffer> {
        unsafe {
            let create_info = BufferCreateInfo {
                size: data.len() * size_of::<T>(),
                buffer_usage: vk::BufferUsageFlags::TRANSFER_SRC,
                memory_usage: vma::MemoryUsage::CpuOnly,
                memory_properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            };

            let (buffer, allocation, _) = Graphics::get_ref().create_buffer(&create_info)?;

            let staging_buffer = StagingBuffer {
                raw: buffer,
                allocation,
            };

            let p = Graphics::get_ref().map_memory(&staging_buffer.allocation)?;
            copy_nonoverlapping(data.as_ptr() as _, p, create_info.size);
            Graphics::get_ref().unmap_memory(&staging_buffer.allocation)?;
            Ok(staging_buffer)
        }
    }

    pub(crate) fn new_multi<T: Sized>(data: &[&[T]]) -> GraphicsResult<StagingBuffer> {
        unsafe {
            let mut size = 0;
            for s in data {
                size += s.len() * size_of::<T>();
            }
            let create_info = BufferCreateInfo {
                size,
                buffer_usage: vk::BufferUsageFlags::TRANSFER_SRC,
                memory_usage: vma::MemoryUsage::CpuOnly,
                memory_properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            };

            let (buffer, allocation, _) = Graphics::get_ref().create_buffer(&create_info)?;

            let staging_buffer = StagingBuffer {
                raw: buffer,
                allocation,
            };

            let p = Graphics::get_ref().map_memory(&staging_buffer.allocation)?;
            let mut offset = 0;
            for s in data {
                let bytes_len = s.len() * size_of::<T>();
                copy_nonoverlapping(s.as_ptr() as _, p.offset(offset), bytes_len);
                offset += bytes_len as isize;
            }
            Graphics::get_ref().unmap_memory(&staging_buffer.allocation)?;
            Ok(staging_buffer)
        }
    }
}

impl Drop for StagingBuffer {
    fn drop(&mut self) {
        unsafe {
            Graphics::get_ref().destroy_buffer(self.raw, &self.allocation);
        }
    }
}

struct GpuBuffer {
    raw: vk::Buffer,
    allocation: vma::Allocation,
    size: usize,
}

impl GpuBuffer {
    fn new(create_info: &BufferCreateInfo) -> GraphicsResult<GpuBuffer> {
        unsafe {
            let (buffer, allocation, _) = Graphics::get_ref().create_buffer(create_info)?;

            Ok(GpuBuffer {
                raw: buffer,
                allocation,
                size: create_info.size,
            })
        }
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        unsafe {
            Graphics::get_ref().destroy_buffer(self.raw, &self.allocation);
        }
    }
}

struct GpuImage {
    raw: vk::Image,
    allocation: vma::Allocation,
    info: ImageCreateInfo,
}

impl GpuImage {
    fn new(create_info: &ImageCreateInfo) -> GraphicsResult<GpuImage> {
        unsafe {
            let (image, allocation, _) = Graphics::get_ref().create_image(create_info)?;

            Ok(GpuImage {
                raw: image,
                allocation,
                info: create_info.clone(),
            })
        }
    }
}

impl Drop for GpuImage {
    fn drop(&mut self) {
        unsafe {
            Graphics::get_ref().destroy_image(self.raw, &self.allocation);
        }
    }
}

struct TransferCommands<'a> {
    command_pool: std::sync::MutexGuard<'a, vk::CommandPool>,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl<'a> TransferCommands<'a> {
    fn new(command_pool: &'a Mutex<vk::CommandPool>) -> GraphicsResult<TransferCommands> {
        unsafe {
            let command_pool = command_pool.lock().unwrap();
            let gfx = Graphics::get_ref();
            let command_buffer =
                gfx.allocate_command_buffer(*command_pool, vk::CommandBufferLevel::PRIMARY)?;
            let fence = match gfx.create_fence(false) {
                Ok(f) => f,
                Err(e) => {
                    Graphics::get_ref().free_command_buffer(*command_pool, command_buffer);
                    return Err(e);
                }
            };

            Ok(TransferCommands {
                command_pool,
                command_buffer,
                fence,
            })
        }
    }
}

impl<'a> Drop for TransferCommands<'a> {
    fn drop(&mut self) {
        unsafe {
            let gfx = Graphics::get_ref();
            gfx.destroy_fence(self.fence);
            gfx.free_command_buffer(*self.command_pool, self.command_buffer);
        }
    }
}

#[derive(Debug, Clone)]
struct DisplayConfig {
    extent: vk::Extent2D,
    format: vk::Format,
    color_space: vk::ColorSpaceKHR,
    pre_transform: vk::SurfaceTransformFlagsKHR,
    composite_alpha: vk::CompositeAlphaFlagsKHR,
    present_mode: vk::PresentModeKHR,
}

impl Default for DisplayConfig {
    fn default() -> Self {
        DisplayConfig {
            extent: vk::Extent2D::default(),
            format: vk::Format::R8G8B8A8_UNORM,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: vk::PresentModeKHR::MAILBOX,
        }
    }
}

impl DisplayConfig {
    fn constraint(&self) -> Result<DisplayConfig, GraphicsError> {
        let vctx = VulkanContext::get_ref();
        let gfx = Graphics::get_ref();
        unsafe {
            let surface_formats = vctx
                .surface_khr
                .get_physical_device_surface_formats(gfx.physical_device, gfx.surface)
                .gfx_vk_result("vkGetPhysicalDeviceSurfaceFormatKHR")?;

            let surface_caps = vctx
                .surface_khr
                .get_physical_device_surface_capabilities(gfx.physical_device, gfx.surface)
                .gfx_vk_result("vkGetPhysicalDeviceSurfaceCapabilitiesKHR")?;

            let present_modes = vctx
                .surface_khr
                .get_physical_device_surface_present_modes(gfx.physical_device, gfx.surface)
                .gfx_vk_result("vkGetPhysicalDeviceSurfacePresentModesKHR")?;

            let extent = if surface_caps.current_extent.width != u32::MAX {
                surface_caps.current_extent
            } else {
                vk::Extent2D {
                    width: self
                        .extent
                        .width
                        .min(surface_caps.max_image_extent.width)
                        .max(surface_caps.min_image_extent.width),
                    height: self
                        .extent
                        .height
                        .min(surface_caps.max_image_extent.height)
                        .max(surface_caps.min_image_extent.height),
                }
            };

            let (format, color_space) = {
                match surface_formats.contains(&vk::SurfaceFormatKHR {
                    format: self.format,
                    color_space: self.color_space,
                }) {
                    true => (self.format, self.color_space),
                    false => match surface_formats.iter().next() {
                        Some(f) => (f.format, f.color_space),
                        None => {
                            return Err(GraphicsError::Other(
                                "Vulkan surface format not found".to_string(),
                            ))
                        }
                    },
                }
            };

            let pre_transform = if surface_caps
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_caps.current_transform
            };

            let present_mode = present_modes
                .iter()
                .cloned()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            Ok(DisplayConfig {
                extent,
                format,
                color_space,
                pre_transform,
                composite_alpha: self.composite_alpha,
                present_mode,
            })
        }
    }
}

struct Display {
    swapchain: vk::SwapchainKHR,
    config: DisplayConfig,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    current_frame: AtomicUsize,
}

impl Display {
    fn null() -> Self {
        Display {
            swapchain: vk::SwapchainKHR::null(),
            config: DisplayConfig::default(),
            images: vec![],
            image_views: vec![],
            current_frame: AtomicUsize::new(0),
        }
    }

    unsafe fn recreate(&mut self, config: DisplayConfig) -> GraphicsResult<()> {
        let gfx = Graphics::get_ref();

        while let Some(image_view) = self.image_views.pop() {
            gfx.destroy_image_view(image_view);
        }
        self.images.clear();

        let create_info = vk::SwapchainCreateInfoKHR {
            surface: gfx.surface,
            min_image_count: 3,
            image_format: config.format,
            image_color_space: config.color_space,
            image_extent: config.extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: config.pre_transform,
            composite_alpha: config.composite_alpha,
            present_mode: config.present_mode,
            clipped: vk::TRUE,
            queue_family_index_count: 1,
            p_queue_family_indices: &gfx.queue_family_index as _,
            old_swapchain: self.swapchain,
            ..Default::default()
        };

        let swapchain = gfx
            .swapchain_khr
            .create_swapchain(&create_info, gfx.allocation_callbacks())
            .gfx_vk_result("vkCreateSwapchainKHR")?;

        gfx.swapchain_khr
            .destroy_swapchain(self.swapchain, gfx.allocation_callbacks());
        self.swapchain = vk::SwapchainKHR::null();

        let images = gfx_error_clean!(
            gfx.swapchain_khr
                .get_swapchain_images(swapchain)
                .gfx_vk_result("vkGetSwapchainImageKHR"),
            gfx.swapchain_khr
                .destroy_swapchain(swapchain, gfx.allocation_callbacks())
        )?;

        self.destroy();

        self.swapchain = swapchain;
        self.config = config;
        self.images = images;

        for image in self.images.iter() {
            let create_info = vk::ImageViewCreateInfo {
                view_type: vk::ImageViewType::TYPE_2D,
                format: self.config.format,
                components: vk::ComponentMapping::default(),
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image: *image,
                ..Default::default()
            };

            self.image_views.push(gfx.create_image_view(&create_info)?);
        }

        self.current_frame.store(0, Ordering::SeqCst);

        Ok(())
    }

    unsafe fn destroy(&mut self) {
        let gfx = Graphics::get_ref();
        while let Some(image_view) = self.image_views.pop() {
            gfx.destroy_image_view(image_view);
        }
        self.images.clear();
        gfx.swapchain_khr
            .destroy_swapchain(self.swapchain, gfx.allocation_callbacks());
        self.swapchain = vk::SwapchainKHR::null();
    }

    unsafe fn next_image(
        &self,
        timeout: u64,
        semaphore: vk::Semaphore,
        fence: vk::Fence,
    ) -> GraphicsResult<usize> {
        let (index, _) = Graphics::get_ref()
            .swapchain_khr
            .acquire_next_image(self.swapchain, timeout, semaphore, fence)
            .gfx_vk_result("vkAcquireNextImage")?;
        Ok(index as _)
    }

    unsafe fn present(
        &self,
        queue: vk::Queue,
        wait_semaphores: &[vk::Semaphore],
        image_index: usize,
    ) -> GraphicsResult<()> {
        let image_index = image_index as u32;
        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: wait_semaphores.len() as _,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: &self.swapchain as _,
            p_image_indices: &image_index as _,
            ..Default::default()
        };

        Graphics::get_ref()
            .swapchain_khr
            .queue_present(queue, &present_info)
            .gfx_vk_result("vkQueuePresent")?;

        Ok(())
    }
}
