//! Graphics resource tracing

use crossbeam::queue::SegQueue;
use lazy_static::lazy_static;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

use super::{CubeMap, GpuBuffer, Sampler, TextureImage};

pub(super) struct ResourceTracer {
    current_frame_index: AtomicUsize,
    buffer_cache: Vec<SegQueue<Arc<GpuBuffer>>>,
    tex_image_cache: Vec<SegQueue<Arc<TextureImage>>>,
    cubemap_cache: Vec<SegQueue<Arc<CubeMap>>>,
    sampler_cache: Vec<SegQueue<Arc<Sampler>>>,
}

lazy_static! {
    static ref S_RESOURCE_TRACER: ResourceTracer = {
        let mut buffer_cache = vec![];
        let mut tex_image_cache = vec![];
        let mut cubemap_cache = vec![];
        let mut sampler_cache = vec![];
        for _ in 0..8 {
            buffer_cache.push(SegQueue::new());
            tex_image_cache.push(SegQueue::new());
            cubemap_cache.push(SegQueue::new());
            sampler_cache.push(SegQueue::new());
        }

        ResourceTracer {
            current_frame_index: AtomicUsize::new(0),
            buffer_cache,
            tex_image_cache,
            cubemap_cache,
            sampler_cache,
        }
    };
}

impl ResourceTracer {
    pub(super) fn get_ref() -> &'static ResourceTracer {
        &S_RESOURCE_TRACER
    }

    pub(super) fn set_current_image_index(&self, idx: usize) {
        self.current_frame_index.store(idx, Ordering::Release);
        while let Some(_) = self.buffer_cache[idx].pop() {}
        while let Some(_) = self.tex_image_cache[idx].pop() {}
        while let Some(_) = self.sampler_cache[idx].pop() {}
        // TODO
    }

    fn current_image_index(&self) -> usize {
        self.current_frame_index.load(Ordering::Acquire)
    }

    pub(super) fn touch_buffer(&self, buf: Arc<GpuBuffer>) {
        self.buffer_cache[self.current_image_index()].push(buf);
    }

    pub(super) fn touch_tex_image(&self, image: Arc<TextureImage>) {
        self.tex_image_cache[self.current_image_index()].push(image);
    }

    pub(super) fn touch_cubemap(&self, cubemap: Arc<CubeMap>) {
        self.cubemap_cache[self.current_image_index()].push(cubemap);
    }

    pub(super) fn touch_sampler(&self, sampler: Arc<Sampler>) {
        self.sampler_cache[self.current_image_index()].push(sampler);
    }
}