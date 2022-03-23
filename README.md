# Simple Vulkan Renderer - Rust
My simple renderer in rust based on vulkan, aiming at learning vulkan and 3D rendering.

## Crates
+ `ushio-geom` 3D Math.
+ `ushio` Renderer and simple scene graph, including a glTF2.0 loader.
+ `rushio` example main logic.

## Features
+ Y-up right hand coordinates.
+ Transformation represented by translation, rotation (quaternion) and scale.
+ Phong pipeline based on vulkan and glsl.
+ Skybox
+ Simple shadow mapping
+ Normal map (Need to fix)
+ Render frames in flight
+ glTF2.0 loader
+ Node, mesh and material definitions inspired by glTF2.0

## Screenshot
![image](./Screenshot.jpg)
<b>NOTE</b>: Used assets are <b>not</b> included in this repo.