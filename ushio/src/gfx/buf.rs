//! Gpu buffers

use ash::vk;
use lazy_static::lazy_static;
use std::{intrinsics::copy_nonoverlapping, marker::PhantomData, mem::size_of, sync::Arc};
use ushio_geom::{IntoSTD140, STD140};
use vk_mem as vma;

use super::{BufferCreateInfo, GpuBuffer, Graphics, GraphicsResult, StagingBuffer};

pub(crate) struct UniformBuffer<T: IntoSTD140> {
    pub(super) buf: GpuBuffer,
    _marker: PhantomData<<T as IntoSTD140>::Output>,
}

impl<T: IntoSTD140> UniformBuffer<T> {
    pub(crate) fn new() -> GraphicsResult<Self> {
        let create_info = BufferCreateInfo {
            size: <T as IntoSTD140>::SIZE,
            buffer_usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            memory_usage: vma::MemoryUsage::CpuToGpu,
            memory_properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        };

        let buf = GpuBuffer::new(&create_info)?;
        Ok(UniformBuffer {
            buf,
            _marker: PhantomData {},
        })
    }

    pub(crate) fn update(&self, val: &T) -> GraphicsResult<()> {
        unsafe {
            Graphics::get_ref().map_copy_memory(&val.into_std140_bytes(), &self.buf.allocation, 0)
        }
    }
}

pub(crate) struct DynamicUniformBuffer<T: IntoSTD140> {
    pub(super) buf: GpuBuffer,
    count: usize,
    _marker: PhantomData<[<T as IntoSTD140>::Output]>,
}

#[repr(C, align(256))]
struct DynamicUniformElem<T: STD140>(T);

impl<T: IntoSTD140> DynamicUniformBuffer<T> {
    pub(crate) const BLOCK_SIZE: usize = size_of::<DynamicUniformElem<<T as IntoSTD140>::Output>>();

    pub(crate) fn new(count: usize) -> GraphicsResult<Self> {
        let create_info = BufferCreateInfo {
            size: Self::BLOCK_SIZE * count,
            buffer_usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            memory_usage: vma::MemoryUsage::CpuToGpu,
            memory_properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        };

        Ok(DynamicUniformBuffer {
            buf: GpuBuffer::new(&create_info)?,
            count,
            _marker: PhantomData {},
        })
    }

    pub(crate) fn update(&self, data: &[T]) -> GraphicsResult<()> {
        let aligned = data
            .iter()
            .map(|val| DynamicUniformElem(val.into_std140()))
            .collect::<Vec<_>>();
        unsafe {
            let allocation = &self.buf.allocation;
            let dst = Graphics::get_ref().map_memory(allocation)?;
            copy_nonoverlapping(aligned.as_ptr() as _, dst, aligned.len() * Self::BLOCK_SIZE);
            Graphics::get_ref().unmap_memory(allocation)?;
        }
        Ok(())
    }

    pub(crate) fn max_count(&self) -> usize {
        self.count
    }
}

pub(crate) struct StorageBuffer<T: IntoSTD140> {
    pub(super) buf: GpuBuffer,
    count: usize,
    _marker: PhantomData<[<T as IntoSTD140>::Output]>,
}

impl<T: IntoSTD140> StorageBuffer<T> {
    pub(crate) const BLOCK_SIZE: usize = size_of::<<T as IntoSTD140>::Output>();

    pub(crate) fn new(count: usize) -> GraphicsResult<Self> {
        let create_info = BufferCreateInfo {
            size: Self::BLOCK_SIZE * count,
            buffer_usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_usage: vma::MemoryUsage::CpuToGpu,
            memory_properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        };

        Ok(StorageBuffer {
            buf: GpuBuffer::new(&create_info)?,
            count,
            _marker: PhantomData {},
        })
    }

    pub(crate) fn update(&self, data: &[T]) -> GraphicsResult<()> {
        unsafe {
            let allocation = &self.buf.allocation;
            let dst = Graphics::get_ref().map_memory(allocation)?;
            copy_nonoverlapping(data.as_ptr() as _, dst, data.len() * Self::BLOCK_SIZE);
            Graphics::get_ref().unmap_memory(allocation)?;
        }
        Ok(())
    }

    pub(crate) fn max_count(&self) -> usize {
        self.count
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ComponentType {
    Byte,
    UnsignedByte,
    Short,
    UnsignedShort,
    UnsignedInt,
    Float,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttributeType {
    Scalar,
    Vec2,
    Vec3,
    Vec4,
    Mat2,
    Mat3,
    Mat4,
}

fn calc_attribute_size(ctype: ComponentType, atype: AttributeType) -> usize {
    let csize = match ctype {
        ComponentType::Byte => 1,
        ComponentType::UnsignedByte => 1,
        ComponentType::Short => 2,
        ComponentType::UnsignedShort => 2,
        ComponentType::UnsignedInt => 4,
        ComponentType::Float => 4,
    };

    let acount = match atype {
        AttributeType::Scalar => 1,
        AttributeType::Vec2 => 2,
        AttributeType::Vec3 => 3,
        AttributeType::Vec4 => 4,
        AttributeType::Mat2 => 4,
        AttributeType::Mat3 => 9,
        AttributeType::Mat4 => 16,
    };

    csize * acount
}

#[derive(Clone, Copy)]
pub(crate) struct TransferDstBufferInfo {
    pub(crate) component_type: ComponentType,
    pub(crate) attribute_type: AttributeType,
    pub(crate) attribute_count: usize,
}

#[derive(Clone)]
pub(crate) struct VertexBuffer {
    pub(super) buf: Arc<GpuBuffer>,
    pub(super) info: TransferDstBufferInfo,
    pub(super) normalized: bool,
}

#[derive(Clone)]
pub(crate) struct IndexBuffer {
    pub(super) buf: Arc<GpuBuffer>,
    pub(super) info: TransferDstBufferInfo,
}

lazy_static! {
    static ref DEFAULT_VERTEX_BUFFER: VertexBuffer = VertexBuffer::new_direct(
        &[0.0],
        TransferDstBufferInfo {
            component_type: ComponentType::Float,
            attribute_type: AttributeType::Scalar,
            attribute_count: 1,
        },
        false
    )
    .unwrap();
}

impl VertexBuffer {
    pub(crate) fn new(
        src: &StagingBuffer,
        src_offset: usize,
        info: TransferDstBufferInfo,
        normalized: bool,
    ) -> GraphicsResult<VertexBuffer> {
        let create_info = BufferCreateInfo {
            size: calc_attribute_size(info.component_type, info.attribute_type)
                * info.attribute_count,
            buffer_usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            memory_usage: vma::MemoryUsage::GpuOnly,
            memory_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        };
        let buf = Arc::new(GpuBuffer::new(&create_info)?);
        Graphics::get_ref().transfer_buffer(
            src,
            buf.as_ref(),
            &[vk::BufferCopy {
                src_offset: src_offset as _,
                dst_offset: 0,
                size: create_info.size as _,
            }],
        )?;
        Ok(VertexBuffer {
            buf,
            info,
            normalized,
        })
    }

    pub(crate) fn new_direct<T: Sized + Clone + Copy>(
        data: &[T],
        info: TransferDstBufferInfo,
        normalized: bool,
    ) -> GraphicsResult<VertexBuffer> {
        let create_info = BufferCreateInfo {
            size: calc_attribute_size(info.component_type, info.attribute_type)
                * info.attribute_count,
            buffer_usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            memory_usage: vma::MemoryUsage::GpuOnly,
            memory_properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        };
        assert!(data.len() * size_of::<T>() >= create_info.size);
        let buf = GpuBuffer::new(&create_info)?;
        unsafe {
            let src = data.as_ptr() as *const u8;
            let dst = Graphics::get_ref().map_memory(&buf.allocation)?;
            copy_nonoverlapping(src, dst, create_info.size);
            Graphics::get_ref().unmap_memory(&buf.allocation)?;
        }
        Ok(VertexBuffer {
            buf: Arc::new(buf),
            info,
            normalized,
        })
    }

    pub(crate) fn new_sparse(
        src: &StagingBuffer,
        regions: &[vk::BufferCopy],
        info: TransferDstBufferInfo,
        normalized: bool,
    ) -> GraphicsResult<VertexBuffer> {
        let create_info = BufferCreateInfo {
            size: calc_attribute_size(info.component_type, info.attribute_type)
                * info.attribute_count,
            buffer_usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            memory_usage: vma::MemoryUsage::GpuOnly,
            memory_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        };
        let buf = Arc::new(GpuBuffer::new(&create_info)?);
        Graphics::get_ref().transfer_buffer(src, buf.as_ref(), regions)?;
        Ok(VertexBuffer {
            buf,
            info,
            normalized,
        })
    }

    pub(super) fn get_default() -> VertexBuffer {
        DEFAULT_VERTEX_BUFFER.clone()
    }

    pub(crate) fn vertex_count(&self) -> usize {
        self.info.attribute_count
    }
}

impl IndexBuffer {
    pub(crate) fn new(
        src: &StagingBuffer,
        src_offset: usize,
        info: TransferDstBufferInfo,
    ) -> GraphicsResult<IndexBuffer> {
        let create_info = BufferCreateInfo {
            size: calc_attribute_size(info.component_type, info.attribute_type)
                * info.attribute_count,
            buffer_usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            memory_usage: vma::MemoryUsage::GpuOnly,
            memory_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        };
        let buf = Arc::new(GpuBuffer::new(&create_info)?);
        Graphics::get_ref().transfer_buffer(
            src,
            buf.as_ref(),
            &[vk::BufferCopy {
                src_offset: src_offset as _,
                dst_offset: 0,
                size: create_info.size as _,
            }],
        )?;
        Ok(IndexBuffer { buf, info })
    }

    pub(crate) fn new_sparse(
        src: &StagingBuffer,
        regions: &[vk::BufferCopy],
        info: TransferDstBufferInfo,
    ) -> GraphicsResult<IndexBuffer> {
        let create_info = BufferCreateInfo {
            size: calc_attribute_size(info.component_type, info.attribute_type)
                * info.attribute_count,
            buffer_usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            memory_usage: vma::MemoryUsage::GpuOnly,
            memory_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        };
        let buf = Arc::new(GpuBuffer::new(&create_info)?);
        Graphics::get_ref().transfer_buffer(src, buf.as_ref(), regions)?;
        Ok(IndexBuffer { buf, info })
    }

    pub(crate) fn index_count(&self) -> usize {
        self.info.attribute_count
    }
}
