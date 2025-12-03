use std::collections::HashMap;
use std::time::Instant;

use engine::{Engine, Scene};
use engine::math::{Camera, Mat4, Quat, Vec3};
use log::info;
use gltf;
use wgpu::util::DeviceExt;
use wgpu::SurfaceError;
use winit::{dpi::PhysicalSize, event::*, event_loop::EventLoop, window::WindowBuilder};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: (mem::size_of::<[f32; 3]>() * 2) as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    mvp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    light_dir: [f32; 3],
    _pad0: f32,
    ambient_color: [f32; 3],
    _pad1: f32,
    specular_color: [f32; 3],
    shininess: f32,
}

#[derive(Default)]
struct InputState {
    move_forward: bool,
    move_back: bool,
    move_left: bool,
    move_right: bool,
    move_up: bool,
    move_down: bool,
    rotate_button_held: bool,
    last_cursor_pos: Option<(f64, f64)>,
}

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

struct Renderer<'window> {
    surface: wgpu::Surface<'window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    depth_format: wgpu::TextureFormat,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    window: &'window winit::window::Window,
    mesh_cache: HashMap<String, GpuMesh>,
    ground_mesh: GpuMesh,
}

impl<'window> Renderer<'window> {
    async fn new(window: &'window winit::window::Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window)
            .expect("Failed to create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let default_mesh_path = "editor/3D assets/Weapon Pack/Models/GLTF format/pistol.glb";
        let (vertices, indices) = load_mesh_from_any(default_mesh_path)
            .unwrap_or_else(builtin_cube_mesh);

        let default_mesh = GpuMesh {
            vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Default Mesh Vertex Buffer"),
                contents: bytemuck::cast_slice(vertices.as_slice()),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            index_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Default Mesh Index Buffer"),
                contents: bytemuck::cast_slice(indices.as_slice()),
                usage: wgpu::BufferUsages::INDEX,
            }),
            num_indices: indices.len() as u32,
        };

        let depth_format = wgpu::TextureFormat::Depth32Float;
        let (depth_texture, depth_view) =
            Self::create_depth_resources(&device, &config, depth_format);

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniform Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let shader_source = r#"
            struct CameraUniform {
                mvp: mat4x4<f32>,
                model: mat4x4<f32>,
                light_dir: vec3<f32>,
                _pad0: f32,
                ambient_color: vec3<f32>,
                _pad1: f32,
                specular_color: vec3<f32>,
                shininess: f32,
            };

            @group(0) @binding(0)
            var<uniform> camera: CameraUniform;

            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) color: vec3<f32>,
                @location(1) normal: vec3<f32>,
            };

            @vertex
            fn vs_main(
                @location(0) position: vec3<f32>,
                @location(1) normal: vec3<f32>,
                @location(2) color: vec3<f32>,
            ) -> VertexOutput {
                var out: VertexOutput;
                let world_pos = camera.model * vec4(position, 1.0);
                let world_normal = (camera.model * vec4(normal, 0.0)).xyz;
                out.position = camera.mvp * vec4(position, 1.0);
                out.color = color;
                out.normal = world_normal;
                return out;
            }

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                let n = normalize(in.normal);
                let l = normalize(-camera.light_dir);
                let diffuse = max(dot(n, l), 0.0);

                let fill_dir = normalize(vec3<f32>(1.0, 0.5, 0.25));
                let fill = max(dot(n, fill_dir), 0.0) * 0.5;

                let base = in.color;
                let ambient = camera.ambient_color;
                let diffuse_color = base * (diffuse + fill);

                let view_dir = normalize(vec3<f32>(0.0, 0.0, 1.0));
                let reflect_dir = reflect(-l, n);
                let spec_angle = max(dot(view_dir, reflect_dir), 0.0);
                let specular = pow(spec_angle, camera.shininess) * camera.specular_color;

                let lit = ambient + diffuse_color + specular;
                return vec4(lit, 1.0);
            }
        "#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Game Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Game Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Game Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Ground plane.
        let ground_half_size = 1000.0_f32;
        let ground_vertices = [
            Vertex {
                position: [-ground_half_size, 0.0, -ground_half_size],
                normal: [0.0, 1.0, 0.0],
                color: [0.3, 0.3, 0.3],
            },
            Vertex {
                position: [ground_half_size, 0.0, -ground_half_size],
                normal: [0.0, 1.0, 0.0],
                color: [0.3, 0.3, 0.3],
            },
            Vertex {
                position: [ground_half_size, 0.0, ground_half_size],
                normal: [0.0, 1.0, 0.0],
                color: [0.3, 0.3, 0.3],
            },
            Vertex {
                position: [-ground_half_size, 0.0, ground_half_size],
                normal: [0.0, 1.0, 0.0],
                color: [0.3, 0.3, 0.3],
            },
        ];
        let ground_indices: [u32; 6] = [0, 1, 2, 2, 3, 0];
        let ground_mesh = GpuMesh {
            vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Ground Vertex Buffer"),
                contents: bytemuck::cast_slice(&ground_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            index_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Ground Index Buffer"),
                contents: bytemuck::cast_slice(&ground_indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
            num_indices: ground_indices.len() as u32,
        };

        let mut mesh_cache = HashMap::new();
        mesh_cache.insert(default_mesh_path.to_string(), default_mesh);

        Self {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            depth_texture,
            depth_view,
            depth_format,
            camera_buffer,
            camera_bind_group,
            window,
            mesh_cache,
            ground_mesh,
        }
    }

    fn create_depth_resources(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        (depth_texture, depth_view)
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            let (depth_texture, depth_view) =
                Self::create_depth_resources(&self.device, &self.config, self.depth_format);
            self.depth_texture = depth_texture;
            self.depth_view = depth_view;
        }
    }

    fn render(&mut self, scene: &Scene, ambient: f32, specular: f32, shininess: f32) -> Result<(), SurfaceError> {
        let camera = &scene.camera;
        let view_proj = camera.view_projection_matrix();

        let mut light_dir = Vec3::new(-1.0, -1.0, -0.5).normalize();
        let ambient_color = [ambient; 3];
        let specular_color = [specular; 3];

        // Ensure meshes are loaded.
        let default_mesh_path = "editor/3D assets/Weapon Pack/Models/GLTF format/pistol.glb".to_string();
        let mut keys_to_load: Vec<String> = Vec::new();
        for entity in &scene.entities {
            let mesh_key = if entity.mesh_path.is_empty() {
                default_mesh_path.clone()
            } else {
                entity.mesh_path.clone()
            };
            if !self.mesh_cache.contains_key(&mesh_key) {
                keys_to_load.push(mesh_key);
            }
        }
        for mesh_key in keys_to_load {
            if let Some((verts, inds)) = load_mesh_from_any(&mesh_key) {
                let mesh = GpuMesh {
                    vertex_buffer: self.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Entity Vertex Buffer"),
                            contents: bytemuck::cast_slice(verts.as_slice()),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    ),
                    index_buffer: self.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Entity Index Buffer"),
                            contents: bytemuck::cast_slice(inds.as_slice()),
                            usage: wgpu::BufferUsages::INDEX,
                        },
                    ),
                    num_indices: inds.len() as u32,
                };
                self.mesh_cache.insert(mesh_key.clone(), mesh);
            } else {
                log::error!("Failed to load mesh from '{}'", mesh_key);
            }
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Game Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.2,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            // Ground.
            {
                let model = Mat4::IDENTITY;
                let mvp = view_proj * model;
                let camera_uniform = CameraUniform {
                    mvp: mvp.to_cols_array_2d(),
                    model: model.to_cols_array_2d(),
                    light_dir: light_dir.to_array(),
                    _pad0: 0.0,
                    ambient_color,
                    _pad1: 0.0,
                    specular_color,
                    shininess,
                };
                self.queue.write_buffer(
                    &self.camera_buffer,
                    0,
                    bytemuck::bytes_of(&camera_uniform),
                );

                render_pass.set_vertex_buffer(0, self.ground_mesh.vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    self.ground_mesh.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                render_pass.draw_indexed(0..self.ground_mesh.num_indices, 0, 0..1);
            }

            // Entities.
            for entity in &scene.entities {
                let mesh_key = if entity.mesh_path.is_empty() {
                    default_mesh_path.clone()
                } else {
                    entity.mesh_path.clone()
                };

                let mesh = match self.mesh_cache.get(&mesh_key) {
                    Some(m) => m,
                    None => continue,
                };

                let model = entity.transform.matrix();
                let mvp = view_proj * model;
                let camera_uniform = CameraUniform {
                    mvp: mvp.to_cols_array_2d(),
                    model: model.to_cols_array_2d(),
                    light_dir: light_dir.to_array(),
                    _pad0: 0.0,
                    ambient_color,
                    _pad1: 0.0,
                    specular_color,
                    shininess,
                };
                self.queue.write_buffer(
                    &self.camera_buffer,
                    0,
                    bytemuck::bytes_of(&camera_uniform),
                );

                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    mesh.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                render_pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
            }
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();
        Ok(())
    }
}

fn update_camera_from_input(camera: &mut Camera, input: &InputState, dt: f32) {
    let forward = camera.target - camera.position;
    let forward_dir = if forward.length_squared() > 0.0 {
        forward.normalize()
    } else {
        Vec3::Z
    };

    let right = forward_dir.cross(camera.up);
    let right_dir = if right.length_squared() > 0.0 {
        right.normalize()
    } else {
        Vec3::X
    };

    let mut movement = Vec3::ZERO;

    if input.move_forward {
        movement += forward_dir;
    }
    if input.move_back {
        movement -= forward_dir;
    }
    if input.move_right {
        movement += right_dir;
    }
    if input.move_left {
        movement -= right_dir;
    }
    if input.move_up {
        movement += camera.up;
    }
    if input.move_down {
        movement -= camera.up;
    }

    if movement.length_squared() > 0.0 {
        let speed = 3.0;
        let delta = movement.normalize() * speed * dt;
        camera.position += delta;
        camera.target += delta;
    }
}

fn rotate_camera_from_mouse(camera: &mut Camera, dx: f32, dy: f32) {
    let sensitivity = 0.005;
    let yaw = -dx * sensitivity;
    let pitch = -dy * sensitivity;

    let forward = camera.target - camera.position;
    let radius = forward.length();
    if radius <= 0.0001 {
        return;
    }

    let mut dir = forward.normalize();

    let yaw_rot = Quat::from_rotation_y(yaw);
    dir = yaw_rot * dir;

    let right = dir.cross(camera.up).normalize();
    let pitch_rot = Quat::from_axis_angle(right, pitch);
    dir = pitch_rot * dir;

    camera.target = camera.position + dir * radius;
}

fn zoom_camera_along_view(camera: &mut Camera, scroll: f32) {
    let forward = camera.target - camera.position;
    if forward.length_squared() == 0.0 {
        return;
    }
    let dir = forward.normalize();
    let amount = scroll * 0.5;
    camera.position += dir * amount;
}

fn builtin_cube_mesh() -> (Vec<Vertex>, Vec<u32>) {
    let base_positions = [
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(-0.5, 0.5, -0.5),
        Vec3::new(-0.5, -0.5, 0.5),
        Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(-0.5, 0.5, 0.5),
    ];

    let mut vertices = Vec::with_capacity(base_positions.len());
    for (i, p) in base_positions.iter().enumerate() {
        let normal = p.normalize_or_zero();
        let color = match i {
            0 => [1.0, 0.0, 0.0],
            1 => [0.0, 1.0, 0.0],
            2 => [0.0, 0.0, 1.0],
            3 => [1.0, 1.0, 0.0],
            4 => [1.0, 0.0, 1.0],
            5 => [0.0, 1.0, 1.0],
            6 => [1.0, 1.0, 1.0],
            _ => [0.0, 0.0, 0.0],
        };

        vertices.push(Vertex {
            position: p.to_array(),
            normal: normal.to_array(),
            color,
        });
    }

    let indices: Vec<u32> = vec![
        0, 1, 2, 2, 3, 0, // back
        4, 5, 6, 6, 7, 4, // front
        4, 0, 3, 3, 7, 4, // left
        1, 5, 6, 6, 2, 1, // right
        4, 5, 1, 1, 0, 4, // bottom
        3, 2, 6, 6, 7, 3, // top
    ];

    (vertices, indices)
}

fn load_mesh_from_any(path: &str) -> Option<(Vec<Vertex>, Vec<u32>)> {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".glb") {
        load_mesh_from_glb(path)
    } else if lower.ends_with(".gltf") {
        load_mesh_from_gltf(path)
    } else if lower.ends_with(".obj") {
        load_mesh_from_obj(path)
    } else if lower.ends_with(".stl") {
        load_mesh_from_stl(path)
    } else {
        None
    }
}

fn load_mesh_from_glb(path: &str) -> Option<(Vec<Vertex>, Vec<u32>)> {
    use std::fs;
    use std::path::Path;

    let data = fs::read(Path::new(path)).ok()?;
    let (document, buffers, _images) = gltf::import_slice(&data).ok()?;

    let mesh = document.meshes().next()?;
    let primitive = mesh.primitives().next()?;

    let reader = primitive.reader(|buffer| {
        buffers.get(buffer.index()).map(|b| b.0.as_slice())
    });

    let positions: Vec<[f32; 3]> = reader.read_positions()?.collect();

    let normals: Vec<[f32; 3]> = if let Some(normals_iter) = reader.read_normals() {
        normals_iter.collect()
    } else {
        vec![[0.0, 1.0, 0.0]; positions.len()]
    };

    let colors: Vec<[f32; 3]> = if let Some(colors0) = reader.read_colors(0) {
        colors0.into_rgb_f32().collect()
    } else {
        vec![[1.0, 1.0, 1.0]; positions.len()]
    };

    let mut vertices = Vec::with_capacity(positions.len());
    for i in 0..positions.len() {
        vertices.push(Vertex {
            position: positions[i],
            normal: normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
            color: colors.get(i).copied().unwrap_or([1.0, 1.0, 1.0]),
        });
    }

    let indices: Vec<u32> = if let Some(indices) = reader.read_indices() {
        use gltf::mesh::util::ReadIndices;
        match indices {
            ReadIndices::U8(iter) => iter.map(|i| i as u32).collect(),
            ReadIndices::U16(iter) => iter.map(|i| i as u32).collect(),
            ReadIndices::U32(iter) => iter.collect(),
        }
    } else {
        (0..positions.len() as u32).collect()
    };

    Some((vertices, indices))
}

fn load_mesh_from_gltf(path: &str) -> Option<(Vec<Vertex>, Vec<u32>)> {
    let (document, buffers, _images) = gltf::import(path).ok()?;

    let mesh = document.meshes().next()?;
    let primitive = mesh.primitives().next()?;

    let reader = primitive.reader(|buffer| {
        buffers.get(buffer.index()).map(|b| b.0.as_slice())
    });

    let positions: Vec<[f32; 3]> = reader.read_positions()?.collect();

    let normals: Vec<[f32; 3]> = if let Some(normals_iter) = reader.read_normals() {
        normals_iter.collect()
    } else {
        vec![[0.0, 1.0, 0.0]; positions.len()]
    };

    let colors: Vec<[f32; 3]> = if let Some(colors0) = reader.read_colors(0) {
        colors0.into_rgb_f32().collect()
    } else {
        vec![[1.0, 1.0, 1.0]; positions.len()]
    };

    let mut vertices = Vec::with_capacity(positions.len());
    for i in 0..positions.len() {
        vertices.push(Vertex {
            position: positions[i],
            normal: normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
            color: colors.get(i).copied().unwrap_or([1.0, 1.0, 1.0]),
        });
    }

    let indices: Vec<u32> = if let Some(indices) = reader.read_indices() {
        use gltf::mesh::util::ReadIndices;
        match indices {
            ReadIndices::U8(iter) => iter.map(|i| i as u32).collect(),
            ReadIndices::U16(iter) => iter.map(|i| i as u32).collect(),
            ReadIndices::U32(iter) => iter.collect(),
        }
    } else {
        (0..positions.len() as u32).collect()
    };

    Some((vertices, indices))
}

fn load_mesh_from_obj(path: &str) -> Option<(Vec<Vertex>, Vec<u32>)> {
    use std::path::Path;

    let path = Path::new(path);
    let (models, _materials) = tobj::load_obj(
        path,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )
    .ok()?;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for m in models {
        let mesh = m.mesh;
        let positions = mesh.positions;
        let normals = mesh.normals;

        let num_verts = positions.len() / 3;
        for i in 0..num_verts {
            let px = positions[i * 3] as f32;
            let py = positions[i * 3 + 1] as f32;
            let pz = positions[i * 3 + 2] as f32;
            let (nx, ny, nz) = if normals.len() >= (i + 1) * 3 {
                (
                    normals[i * 3] as f32,
                    normals[i * 3 + 1] as f32,
                    normals[i * 3 + 2] as f32,
                )
            } else {
                (0.0, 1.0, 0.0)
            };

            vertices.push(Vertex {
                position: [px, py, pz],
                normal: [nx, ny, nz],
                color: [1.0, 1.0, 1.0],
            });
        }

        indices.extend(mesh.indices.into_iter().map(|i| i as u32));
    }

    if vertices.is_empty() || indices.is_empty() {
        None
    } else {
        Some((vertices, indices))
    }
}

fn load_mesh_from_stl(path: &str) -> Option<(Vec<Vertex>, Vec<u32>)> {
    use std::fs::File;
    use std::path::Path;

    let mut file = File::open(Path::new(path)).ok()?;
    let stl = stl_io::read_stl(&mut file).ok()?;

    let raw_vertices = stl.vertices;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for face in stl.faces {
        let normal = face.normal;
        let base_index = vertices.len() as u32;
        for &vi in &face.vertices {
            let v = raw_vertices[vi];
            vertices.push(Vertex {
                position: [v[0] as f32, v[1] as f32, v[2] as f32],
                normal: [normal[0] as f32, normal[1] as f32, normal[2] as f32],
                color: [1.0, 1.0, 1.0],
            });
        }
        indices.extend_from_slice(&[base_index, base_index + 1, base_index + 2]);
    }

    if vertices.is_empty() || indices.is_empty() {
        None
    } else {
        Some((vertices, indices))
    }
}

fn main() {
    env_logger::init();
    info!("Starting game runner...");

    // Try to load a scene from disk; fall back to a simple default.
    let scene = engine::scene_io::load_scene_from_file("scene.json").unwrap_or_else(|| {
        let aspect = 16.0 / 9.0;
        let camera = Camera::new_perspective(
            Vec3::new(0.0, 2.0, 5.0),
            Vec3::ZERO,
            aspect,
            45_f32.to_radians(),
            0.1,
            100.0,
        );
        Scene {
            entities: vec![engine::Entity {
                name: "Entity 0".to_string(),
                transform: engine::math::Transform::identity(),
                velocity: Vec3::ZERO,
                acceleration: Vec3::ZERO,
                mesh_path: "editor/3D assets/Weapon Pack/Models/GLTF format/pistol.glb"
                    .to_string(),
                is_character: false,
            }],
            camera,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            linear_damping: 0.2,
        }
    });

    let mut engine = Engine::new(scene);

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let window = WindowBuilder::new()
        .with_title("Rust 3D Engine Game")
        .build(&event_loop)
        .expect("Failed to create window");
    let window: &'static winit::window::Window = Box::leak(Box::new(window));

    let mut renderer = pollster::block_on(Renderer::new(window));
    let mut input = InputState::default();
    let mut last_frame = Instant::now();

    let ambient = 0.2;
    let specular = 0.5;
    let shininess = 16.0;

    event_loop
        .run(move |event, elwt| {
            match event {
                Event::WindowEvent { event, window_id } if window_id == window.id() => {
                    match event {
                        WindowEvent::CloseRequested => {
                            engine.shutdown();
                            elwt.exit();
                        }
                        WindowEvent::KeyboardInput { event, .. } => {
                            use winit::event::ElementState;
                            use winit::keyboard::{KeyCode, PhysicalKey};

                            let pressed = event.state == ElementState::Pressed;
                            match event.physical_key {
                                PhysicalKey::Code(KeyCode::KeyW) => input.move_forward = pressed,
                                PhysicalKey::Code(KeyCode::KeyS) => input.move_back = pressed,
                                PhysicalKey::Code(KeyCode::KeyA) => input.move_left = pressed,
                                PhysicalKey::Code(KeyCode::KeyD) => input.move_right = pressed,
                                PhysicalKey::Code(KeyCode::Space) => input.move_up = pressed,
                                PhysicalKey::Code(KeyCode::ShiftLeft) => input.move_down = pressed,
                                PhysicalKey::Code(KeyCode::Escape) if pressed => {
                                    engine.shutdown();
                                    elwt.exit();
                                }
                                _ => {}
                            }
                        }
                        WindowEvent::MouseInput { state, button, .. } => {
                            if button == winit::event::MouseButton::Right {
                                let pressed = state == winit::event::ElementState::Pressed;
                                input.rotate_button_held = pressed;
                                if pressed {
                                    input.last_cursor_pos = None;
                                }
                            }
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            if input.rotate_button_held {
                                if let Some((lx, ly)) = input.last_cursor_pos {
                                    let dx = position.x - lx;
                                    let dy = position.y - ly;
                                    rotate_camera_from_mouse(
                                        &mut engine.scene.camera,
                                        dx as f32,
                                        dy as f32,
                                    );
                                }
                                input.last_cursor_pos = Some((position.x, position.y));
                            } else {
                                input.last_cursor_pos = Some((position.x, position.y));
                            }
                        }
                        WindowEvent::MouseWheel { delta, .. } => {
                            let scroll = match delta {
                                winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                                winit::event::MouseScrollDelta::PixelDelta(pos) => {
                                    pos.y as f32 / 50.0
                                }
                            };
                            if scroll.abs() > 0.0 {
                                zoom_camera_along_view(&mut engine.scene.camera, scroll);
                            }
                        }
                        WindowEvent::Resized(size) => {
                            renderer.resize(size);
                        }
                        WindowEvent::RedrawRequested => {
                            let now = Instant::now();
                            let dt = (now - last_frame).as_secs_f32();
                            last_frame = now;

                            update_camera_from_input(&mut engine.scene.camera, &input, dt);
                            engine.update(dt);

                            match renderer.render(&engine.scene, ambient, specular, shininess) {
                                Ok(()) => {}
                                Err(SurfaceError::Lost | SurfaceError::Outdated) => {
                                    renderer.resize(window.inner_size());
                                }
                                Err(SurfaceError::OutOfMemory) => {
                                    elwt.exit();
                                }
                                Err(SurfaceError::Timeout) => {
                                    // Skip frame.
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Event::AboutToWait => {
                    window.request_redraw();
                }
                _ => {}
            }
        })
        .expect("Event loop error");
}
