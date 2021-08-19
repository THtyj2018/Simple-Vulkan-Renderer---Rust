//! Rushio Engine

mod color;
pub use color::Color;
pub mod gfx;
pub mod scene;
use scene::Scene;

use std::{
    mem::ManuallyDrop,
    sync::{
        atomic::{AtomicBool, Ordering},
        RwLock, RwLockReadGuard,
    },
};

use lazy_static::lazy_static;
use thiserror::Error;
use winit::{
    dpi::PhysicalSize,
    error::OsError,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
    window::{Fullscreen, Window as WInitWindow, WindowBuilder},
};

use gfx::{
    Graphics, GraphicsParams, Renderer, RendererCreateInfo, RendererTrait, RenderingData,
    RenderingDataTrait,
};

pub struct Window {
    event_loop: RwLock<EventLoop<()>>,
    raw: WInitWindow,
    minimized: AtomicBool,
}

unsafe impl Sync for Window {}

#[derive(Debug, Error)]
pub enum WindowCreateError {
    #[error("{0}")]
    WInit(#[from] OsError),
}

#[derive(Debug)]
pub struct WindowParams {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub resizeable: bool,
    pub fullscreen: bool,
}

impl Default for WindowParams {
    fn default() -> Self {
        WindowParams {
            title: "Demo window".to_string(),
            width: 960,
            height: 540,
            resizeable: false,
            fullscreen: false,
        }
    }
}

struct RunContext {
    running: AtomicBool,
}

#[derive(Debug, Default)]
pub struct ConfigParams {
    pub window: WindowParams,
    pub graphics: GraphicsParams,
}

lazy_static! {
    static ref S_PARAMS: RwLock<ConfigParams> = RwLock::new(ConfigParams::default());
    static ref S_WINDOW: Window = {
        let params = &ConfigParams::read().window;
        let event_loop = RwLock::new(EventLoop::new());

        let fullscreen = match params.fullscreen {
            true => Some(Fullscreen::Borderless(None)),
            false => None,
        };

        let window = WindowBuilder::new()
            .with_title(params.title.clone())
            .with_inner_size(PhysicalSize::new(params.width, params.height))
            .with_fullscreen(fullscreen)
            .with_resizable(params.resizeable)
            .build(event_loop.read().as_ref().unwrap())
            .unwrap();

        Window {
            event_loop,
            raw: window,
            minimized: AtomicBool::new(false),
        }
    };
    static ref S_RUN_CONTEXT: RunContext = RunContext {
        running: AtomicBool::new(true),
    };
}

impl ConfigParams {
    fn read() -> RwLockReadGuard<'static, Self> {
        S_PARAMS.read().unwrap()
    }
}

impl Window {
    pub fn get_ref() -> &'static Window {
        &S_WINDOW
    }

    pub fn inner_size(&self) -> (u32, u32) {
        let size = self.raw.inner_size();
        (size.width, size.height)
    }

    pub fn inner_aspect(&self) -> f32 {
        let size = self.raw.inner_size();
        match size.height {
            0 => 1.0,
            _ => (size.width as f32) / (size.height as f32),
        }
    }

    fn mark_minimized(&self, val: bool) {
        self.minimized.store(val, Ordering::Release);
    }

    pub fn is_minimized(&self) -> bool {
        self.minimized.load(Ordering::Acquire)
    }
}

impl RunContext {
    fn get_ref() -> &'static RunContext {
        &S_RUN_CONTEXT
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    fn quit(&self) {
        self.running.store(false, Ordering::Release)
    }
}

pub trait GlobalScript: Sized {
    fn new() -> Self;

    fn get_scene(&self) -> Scene;

    fn update(&mut self);
}

pub fn launch<S: GlobalScript>(params: ConfigParams) {
    *S_PARAMS.write().unwrap() = params;
    let mut script = ManuallyDrop::new(S::new());

    let window = Window::get_ref();
    let (width, height) = window.inner_size();
    let mut renderer = ManuallyDrop::new(
        Renderer::new(&RendererCreateInfo {
            width,
            height,
            samples: 4,
        })
        .unwrap(),
    );

    let rctx = RunContext::get_ref();

    while rctx.is_running() {
        let mut event_loop = window.event_loop.write().unwrap();
        event_loop.run_return(|event, _, ctrl_flow| {
            *ctrl_flow = ControlFlow::Poll;
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => {
                        rctx.quit();
                        *ctrl_flow = ControlFlow::Exit;
                    }
                    WindowEvent::Resized(size) => {
                        log::debug!("Window resized to {:?}", size);
                        if size.width == 0 && size.height == 0 {
                            window.mark_minimized(true);
                        } else {
                            window.mark_minimized(false);
                            unsafe {
                                match Graphics::get_ref().wait_idle() {
                                    Ok(_) => (),
                                    Err(e) => {
                                        log::error!("Window resize error: {}", e);
                                        rctx.quit();
                                        *ctrl_flow = ControlFlow::Exit;
                                    }
                                }
                                match renderer.recreate(&RendererCreateInfo {
                                    width: size.width,
                                    height: size.height,
                                    samples: 4,
                                }) {
                                    Ok(_) => (),
                                    Err(e) => {
                                        log::error!("Window resize error: {}", e);
                                        rctx.quit();
                                        *ctrl_flow = ControlFlow::Exit;
                                    }
                                }
                            }
                        }
                    }
                    _ => (),
                },
                Event::MainEventsCleared => {
                    if rctx.is_running()
                        && *ctrl_flow != ControlFlow::Exit
                        && !window.is_minimized()
                    {
                        script.update();
                        let rendering_data = RenderingData::parse_scene(script.get_scene());

                        if let Err(e) = renderer.render(rendering_data) {
                            log::error!("Rendering error: {}", e);
                            rctx.quit();
                            *ctrl_flow = ControlFlow::Exit;
                        };
                    }
                }
                _ => (),
            }
        });
    }

    ManuallyDrop::into_inner(script);
    unsafe {
        Graphics::get_ref().wait_idle().unwrap();
    }
    ManuallyDrop::into_inner(renderer);
}
