use std::{cell::RefCell, env, path::Path, rc::Rc, sync::{mpsc, Arc}, thread::JoinHandle, time::Instant};

use ushio::{
    launch,
    scene::{
        material::{CommonTexture, Material, NormalTexture, Texture},
        Camera, Light, NodeError, Scene, SceneNode, SceneParams, Skybox, SkyboxError,
        SkyboxImagePaths, Transform,
    },
    Color, ConfigParams, GlobalScript, Window, WindowParams,
};

use log;
use log4rs;
use ushio_geom::Vec3;

struct SendSceneNodes(Result<Vec<Rc<RefCell<SceneNode>>>, NodeError>);

unsafe impl Send for SendSceneNodes {}

struct AsyncModel {
    join_handle: Option<JoinHandle<SendSceneNodes>>,
    rx: mpsc::Receiver<()>,
    root: Option<Rc<RefCell<SceneNode>>>,
}

struct AsyncSkybox {
    join_handle: Option<JoinHandle<Result<Arc<Skybox>, SkyboxError>>>,
    rx: mpsc::Receiver<()>,
    skybox: Option<Arc<Skybox>>,
}

impl AsyncModel {
    fn new(filepath: &'static str) -> Self {
        Self::new_impl(move || {
            let result = SceneNode::load(filepath);
            log::info!("{} loaded!", filepath);
            result
        })
    }

    fn new_impl<F>(f: F) -> Self
    where
        F: 'static + Send + FnOnce() -> Result<Vec<Rc<RefCell<SceneNode>>>, NodeError>,
    {
        let (tx, rx) = mpsc::channel();
        AsyncModel {
            join_handle: Some(std::thread::spawn(move || {
                let result = SendSceneNodes(f());
                tx.send(()).unwrap();
                result
            })),
            rx,
            root: None,
        }
    }

    fn update(&mut self) {
        if self.join_handle.is_some() {
            if self.rx.try_recv().is_ok() {
                let nodes = self.join_handle.take().unwrap().join().unwrap();
                self.root = nodes.0.unwrap().pop();
            }
        }
    }
}

impl AsyncSkybox {
    fn new<P: 'static + Send + AsRef<Path>>(paths: SkyboxImagePaths<P>) -> Self {
        let (tx, rx) = mpsc::channel();
        AsyncSkybox {
            join_handle: Some(std::thread::spawn(move || {
                let skybox = Skybox::load(paths);
                tx.send(()).unwrap();
                skybox
            })),
            rx,
            skybox: None,
        }
    }

    fn update(&mut self) {
        if self.join_handle.is_some() {
            if self.rx.try_recv().is_ok() {
                let skybox = self.join_handle.take().unwrap().join().unwrap();
                self.skybox = Some(skybox.unwrap());
            }
        }
    }
}

struct Script {
    barbara: AsyncModel,
    seele: AsyncModel,
    marisa: AsyncModel,
    ground: AsyncModel,
    plaster: AsyncModel,
    torus: AsyncModel,
    skybox: AsyncSkybox,
    camera: Rc<RefCell<SceneNode>>,
    light: Rc<RefCell<SceneNode>>,
    start: Instant,
    last_time: Instant,
}

impl GlobalScript for Script {
    fn new() -> Script {
        let ground = AsyncModel::new_impl(|| {
            let color = Texture::load(
                "assets/textures/Bricks018_1K_Color.png",
                &Default::default(),
                false,
            )?;
            let normal = Texture::load(
                "assets/textures/Bricks018_1K_Normal.png",
                &Default::default(),
                false,
            )?;

            SceneNode::new_plane(
                1000.0,
                1000.0,
                2.0,
                2.0,
                Arc::new(Material {
                    base_color_texture: Some(CommonTexture {
                        texture: color,
                        ..Default::default()
                    }),
                    normal_texture: Some(NormalTexture {
                        texture: normal,
                        ..Default::default()
                    }),
                    metallic_factor: 0.0,
                    roughness_factor: 1.0,
                    ..Default::default()
                }),
            )
            .map(|node| vec![node])
        });

        let skybox = AsyncSkybox::new(SkyboxImagePaths {
            right: "assets/skybox/right.jpg",
            left: "assets/skybox/left.jpg",
            top: "assets/skybox/top.jpg",
            bottom: "assets/skybox/top.jpg", // TODO
            front: "assets/skybox/front.jpg",
            back: "assets/skybox/back.jpg",
        });

        Script {
            barbara: AsyncModel::new(
                "D:/Desktop/Create/mmd/models/Genshin/芭芭拉_闪耀协奏/gltf2/barbara.gltf",
            ),
            seele: AsyncModel::new("assets/seele_natsu/seele.gltf"),
            marisa: AsyncModel::new(
                "D:/Desktop/Create/mmd/models/Touhou/霧雨魔理沙/gltf/marisa.gltf",
            ),
            ground,
            plaster: AsyncModel::new("assets/plaster/plaster.gltf"),
            torus: AsyncModel::new("assets/torus/metal_torus.gltf"),
            skybox,
            camera: SceneNode::new_empty(),
            light: SceneNode::new_light(Light::directional(Color::white(), 0.85)),
            start: Instant::now(),
            last_time: Instant::now(),
        }
    }

    fn get_scene(&self) -> Scene {
        let mut nodes = vec![self.camera.clone(), self.light.clone()];
        if let Some(barbara) = self.barbara.root.clone() {
            nodes.push(barbara);
        }
        if let Some(seele) = self.seele.root.clone() {
            let mut node = SceneNode::new_empty();
            node.borrow_mut().set_transform(Transform::new(
                Vec3::new(0.0, 0.0, -0.4),
                Default::default(),
                Vec3::one() * 1.2,
            ));
            SceneNode::attach(&mut node, seele);
            nodes.push(node);
        }
        if let Some(marisa) = self.marisa.root.clone() {
            nodes.push(marisa);
        }
        if let Some(ground) = self.ground.root.clone() {
            nodes.push(ground);
        }
        if let Some(plaster) = self.plaster.root.clone() {
            nodes.push(plaster);
        }
        if let Some(torus) = self.torus.root.clone() {
            nodes.push(torus);
        }
        Scene {
            nodes,
            camera: self.camera.clone(),
            skybox: self.skybox.skybox.clone(),
            params: SceneParams { ambient: 0.24 },
        }
    }

    fn update(&mut self) {
        let elapsed = self.start.elapsed().as_secs_f32();
        let _ = self.last_time.elapsed().as_secs_f32();
        self.last_time = Instant::now();

        self.barbara.update();
        self.seele.update();
        self.marisa.update();
        self.plaster.update();
        self.torus.update();
        self.ground.update();
        self.skybox.update();

        if let Some(barbara) = self.barbara.root.as_ref() {
            barbara.borrow_mut().set_transform(Transform::lookat(
                Vec3::new(-0.7, 0.0, 0.3),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::pos_y(),
            ));
        }

        if let Some(marisa) = self.marisa.root.as_ref() {
            marisa.borrow_mut().set_transform(Transform::new(
                Vec3::new(1.0, 0.0, 0.0),
                Default::default(),
                Vec3::one(),
            ));
        }

        if let Some(plaster) = self.plaster.root.as_ref() {
            plaster.borrow_mut().set_transform(Transform::new(
                Vec3::new(-2.0, 2.5, -2.0),
                Default::default(),
                Vec3::one(),
            ));
        }

        if let Some(torus) = self.torus.root.as_ref() {
            torus.borrow_mut().set_transform(Transform::lookat(
                Vec3::new(0.5, 1.8, 2.0),
                Vec3::zero(),
                Vec3::pos_y(),
            ));
            torus.borrow_mut().set_scale(Vec3::one() * 0.7);
        }

        let mut camera = self.camera.borrow_mut();
        camera.set_transform(Transform::lookat(
            Vec3::new(elapsed.cos(), 0.8, elapsed.sin()) * 2.0,
            Vec3::new(0.0, 1.2, 0.0),
            Vec3::pos_y(),
        ));
        camera.set_camera(Camera::perspective(
            Window::get_ref().inner_aspect(),
            1.2,
            0.1,
            100.0,
        ));

        self.light.borrow_mut().set_transform(Transform::lookat(
            Vec3::new(3.0, 4.0, 2.8),
            Vec3::zero(),
            Vec3::pos_y(),
        ));
    }
}

fn main() {
    init_log(log::LevelFilter::Debug, "rushio.log");
    let mut params = ConfigParams {
        window: WindowParams {
            width: 1440,
            height: 810,
            title: "3D Graphics Demo".into(),
            ..Default::default()
        },
        ..Default::default()
    };
    for arg in env::args() {
        if arg == "--fullscreen" {
            params.window.fullscreen = true;
        } else if arg == "--resizeable" {
            params.window.resizeable = true;
        }
    }
    launch::<Script>(params);
}

fn init_log(level: log::LevelFilter, log_file_name: &str) {
    use log4rs::{
        append::{console, file},
        config,
        encode::pattern,
        init_config,
    };

    let stdout = console::ConsoleAppender::builder()
        .encoder(Box::new(pattern::PatternEncoder::new(
            "[Console] {d} - {l} - {t} - {m}{n}",
        )))
        .build();

    let file = file::FileAppender::builder()
        .encoder(Box::new(pattern::PatternEncoder::new(
            "[File] {d} - {l} - {t} - {m}{n}",
        )))
        .append(false)
        .build(log_file_name)
        .unwrap();

    let config = config::Config::builder()
        .appender(config::Appender::builder().build("stdout", Box::new(stdout)))
        .appender(config::Appender::builder().build("file", Box::new(file)))
        .logger(
            config::Logger::builder()
                .appender("file")
                .additive(false)
                .build("bog", level),
        )
        .build(
            config::Root::builder()
                .appender("stdout")
                .appender("file")
                .build(level),
        )
        .unwrap();

    let _ = init_config(config).unwrap();
}
