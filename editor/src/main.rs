use std::time::Instant;
use std::collections::HashMap;

use engine::{Engine, Scene};
use engine::math::{Camera, Vec3, Quat};
use log::info;
use serde::Deserialize;
use gltf;
use egui::TextureHandle;

mod assets;
use assets::discover_mesh_assets;
use wgpu::SurfaceError;
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, event::*, event_loop::EventLoop, window::WindowBuilder};
use egui::{self, Context as EguiContext};
use egui_wgpu::{Renderer as EguiRenderer, ScreenDescriptor};
use egui_winit::State as EguiWinitState;

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

#[derive(Default)]
struct EditorUiState {
    paused: bool,
    ambient: f32,
    specular: f32,
    shininess: f32,
    invert_light: bool,
    selected_entity: Option<usize>,
    glb_path: String,
    grid_snap: bool,
    grid_size: f32,
}

struct MeshAsset {
    name: String,
    mesh_path: String,
    thumbnail: Option<TextureHandle>,
}

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

#[derive(Deserialize)]
struct MeshFileVertex {
    position: [f32; 3],
    color: [f32; 3],
}

#[derive(Deserialize)]
struct MeshFile {
    vertices: Vec<MeshFileVertex>,
    indices: Vec<u32>,
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
    egui_ctx: EguiContext,
    egui_winit: EguiWinitState,
    egui_renderer: EguiRenderer,
    ui_state: EditorUiState,
    mesh_assets: Vec<MeshAsset>,
    mesh_cache: HashMap<String, GpuMesh>,
    ground_mesh: GpuMesh,
}

impl<'window> Renderer<'window> {
    async fn new(event_loop: &EventLoop<()>, window: &'window winit::window::Window) -> Self {
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

        // Load default mesh into cache (used when an entity has an empty mesh_path).
        let default_mesh_path = "editor/3D assets/Weapon Pack/Models/GLTF format/pistol.glb";
        let (vertices, indices) = load_mesh_from_any(default_mesh_path)
            .or_else(|| load_mesh_from_json("assets/cube.json"))
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
        let (depth_texture, depth_view) = Self::create_depth_resources(&device, &config, depth_format);

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

                // Simple multi-light: main light + a weak fill light.
                let fill_dir = normalize(vec3<f32>(1.0, 0.5, 0.25));
                let fill = max(dot(n, fill_dir), 0.0) * 0.5;

                let base = in.color;
                let ambient = camera.ambient_color;
                let diffuse_color = base * (diffuse + fill);

                // Simple specular using a fixed view direction.
                let view_dir = normalize(vec3<f32>(0.0, 0.0, 1.0));
                let reflect_dir = reflect(-l, n);
                let spec_angle = max(dot(view_dir, reflect_dir), 0.0);
                let specular = pow(spec_angle, camera.shininess) * camera.specular_color;

                let lit = ambient + diffuse_color + specular;
                return vec4(lit, 1.0);
            }
        "#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Triangle Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Renderer Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Triangle Pipeline"),
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

        let egui_ctx = EguiContext::default();
        let egui_winit = EguiWinitState::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            event_loop,
            None,
            None,
        );
        let egui_renderer = EguiRenderer::new(&device, config.format, None, 1);
        let ui_state = EditorUiState {
            paused: false,
            ambient: 0.2,
            specular: 0.5,
            shininess: 16.0,
            invert_light: false,
            selected_entity: None,
            glb_path: default_mesh_path.to_string(),
            grid_snap: false,
            grid_size: 1.0,
        };

        let mesh_assets = discover_mesh_assets(&egui_ctx, "editor/3D assets");

        // Create a large ground plane mesh (endless-looking).
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
            egui_ctx,
            egui_winit,
            egui_renderer,
            ui_state,
            mesh_assets,
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

    fn render(&mut self, scene: &mut Scene, dt: f32) -> Result<(), SurfaceError> {
        // Build editor UI with egui.
        let raw_input = self.egui_winit.take_egui_input(self.window);
        let mut delete_entity: Option<usize> = None;
        let mut duplicate_entity: Option<usize> = None;
        let mut reload_glb: Option<String> = None;
        let full_output = {
            let ui_state = &mut self.ui_state;
            let scene_ref: &mut Scene = scene;

            self.egui_ctx.run(raw_input, |ctx| {
            egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
                let fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
                ui.label(format!("FPS: {:.1}", fps));
                let camera_ref = &scene_ref.camera;
                ui.label(format!(
                    "Camera: ({:.2}, {:.2}, {:.2})",
                    camera_ref.position.x, camera_ref.position.y, camera_ref.position.z
                ));
            });

            egui::SidePanel::left("left_panel")
                .resizable(true)
                .default_width(260.0)
                .show(ctx, |ui| {
                ui.heading("Controls");
                ui.checkbox(&mut ui_state.paused, "Pause simulation");
                ui.checkbox(&mut ui_state.invert_light, "Invert light");
                ui.add(egui::Slider::new(&mut ui_state.ambient, 0.0..=1.0).text("Ambient"));
                ui.add(egui::Slider::new(&mut ui_state.specular, 0.0..=2.0).text("Specular"));
                ui.add(egui::Slider::new(&mut ui_state.shininess, 2.0..=64.0).text("Shininess"));

                ui.separator();
                ui.heading("Mesh");
                ui.label("GLB path (relative to project root)");
                ui.text_edit_singleline(&mut ui_state.glb_path);
                if ui.button("Reload GLB").clicked() {
                    reload_glb = Some(ui_state.glb_path.clone());
                }

                ui.separator();
                ui.label("Assets");
                egui::ScrollArea::vertical()
                    .max_height(200.0)
                    .show(ui, |ui| {
                        for asset in &self.mesh_assets {
                            ui.horizontal(|ui| {
                                if let Some(tex) = &asset.thumbnail {
                                    let size = tex.size_vec2();
                                    let scale = 64.0 / size.y.max(1.0);
                                    let image = egui::Image::new((
                                        tex.id(),
                                        egui::vec2(size.x * scale, size.y * scale),
                                    ));
                                    ui.add(image);
                                }

                                let label = if ui_state.glb_path == asset.mesh_path {
                                    format!("{} (selected)", asset.name)
                                } else {
                                    asset.name.clone()
                                };

                                if ui.button(label).clicked() {
                                    ui_state.glb_path = asset.mesh_path.clone();
                                    reload_glb = Some(asset.mesh_path.clone());
                                    if let Some(sel) = ui_state.selected_entity {
                                        if let Some(ent) = scene_ref.entities.get_mut(sel) {
                                            ent.mesh_path = asset.mesh_path.clone();
                                        }
                                    }
                                }
                            });
                            ui.separator();
                        }
                    });

                ui.separator();
                ui.heading("Transform helpers");
                ui.checkbox(&mut ui_state.grid_snap, "Snap to grid");
                ui.add(
                    egui::DragValue::new(&mut ui_state.grid_size)
                        .speed(0.1)
                        .clamp_range(0.1..=100.0)
                        .prefix("Grid size: "),
                );

                ui.separator();
                ui.heading("Entities");

                if ui.button("Add entity").clicked() {
                    let index = scene_ref.entities.len();
                    let mesh_path = ui_state.glb_path.clone();
                    scene_ref.entities.push(engine::Entity {
                        name: format!("Entity {}", index),
                        transform: engine::math::Transform::identity(),
                        velocity: Vec3::ZERO,
                        mesh_path,
                    });
                }

                for (i, entity) in scene_ref.entities.iter().enumerate() {
                    let label = format!("{} ({})", entity.name, i);
                    let selected = ui_state.selected_entity == Some(i);
                    if ui.selectable_label(selected, label).clicked() {
                        ui_state.selected_entity = Some(i);
                    }
                }

                if let Some(i) = ui_state.selected_entity {
                    if let Some(entity) = scene_ref.entities.get_mut(i) {
                        ui.separator();
                        ui.label("Name");
                        ui.text_edit_singleline(&mut entity.name);

                        ui.separator();
                        ui.label("Selected entity transform");
                        ui.horizontal(|ui| {
                            ui.label("Position");
                            let resp_x = ui.add(
                                egui::DragValue::new(&mut entity.transform.translation.x)
                                    .speed(0.1),
                            );
                            let resp_y = ui.add(
                                egui::DragValue::new(&mut entity.transform.translation.y)
                                    .speed(0.1),
                            );
                            let resp_z = ui.add(
                                egui::DragValue::new(&mut entity.transform.translation.z)
                                    .speed(0.1),
                            );
                            if ui_state.grid_snap && ui_state.grid_size > 0.0 {
                                let snap = |v: &mut f32, step: f32| {
                                    *v = (*v / step).round() * step;
                                };
                                if resp_x.changed() {
                                    snap(&mut entity.transform.translation.x, ui_state.grid_size);
                                }
                                if resp_y.changed() {
                                    snap(&mut entity.transform.translation.y, ui_state.grid_size);
                                }
                                if resp_z.changed() {
                                    snap(&mut entity.transform.translation.z, ui_state.grid_size);
                                }
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Scale");
                            ui.add(
                                egui::DragValue::new(&mut entity.transform.scale.x).speed(0.1),
                            );
                            ui.add(
                                egui::DragValue::new(&mut entity.transform.scale.y).speed(0.1),
                            );
                            ui.add(
                                egui::DragValue::new(&mut entity.transform.scale.z).speed(0.1),
                            );
                        });

                        ui.horizontal(|ui| {
                            if ui.button("Align to ground").clicked() {
                                entity.transform.translation.y = 0.0;
                            }
                            if ui.button("Duplicate").clicked() {
                                duplicate_entity = Some(i);
                            }
                            if ui.button("Delete").clicked() {
                                delete_entity = Some(i);
                            }
                        });
                    } else {
                        ui_state.selected_entity = None;
                    }
                }

                ui.separator();
                if ui.button("Save scene").clicked() {
                    if let Err(err) = save_scene_to_file("scene.json", scene_ref) {
                        log::error!("Failed to save scene: {}", err);
                    }
                }
                if ui.button("Load scene").clicked() {
                    if let Some(new_scene) = load_scene_from_file("scene.json") {
                        *scene_ref = new_scene;
                        ui_state.selected_entity = None;
                    }
                }
            });
            })
        };
        if let Some(i) = delete_entity {
            if i < scene.entities.len() {
                scene.entities.remove(i);
                if scene.entities.is_empty() {
                    self.ui_state.selected_entity = None;
                } else if i >= scene.entities.len() {
                    self.ui_state.selected_entity = Some(scene.entities.len() - 1);
                } else {
                    self.ui_state.selected_entity = Some(i);
                }
            }
        }
        if let Some(i) = duplicate_entity {
            if i < scene.entities.len() {
                let src = &scene.entities[i];
                scene.entities.push(engine::Entity {
                    name: format!("{} Copy", src.name),
                    transform: engine::math::Transform {
                        translation: src.transform.translation,
                        rotation: src.transform.rotation,
                        scale: src.transform.scale,
                    },
                    velocity: Vec3::ZERO,
                    mesh_path: src.mesh_path.clone(),
                });
                self.ui_state.selected_entity = Some(scene.entities.len() - 1);
            }
        }

        if let Some(path) = reload_glb {
            if let Some((vertices, indices)) = load_mesh_from_any(&path) {
                let mesh = GpuMesh {
                    vertex_buffer: self
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Asset Vertex Buffer"),
                            contents: bytemuck::cast_slice(vertices.as_slice()),
                            usage: wgpu::BufferUsages::VERTEX,
                        }),
                    index_buffer: self
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Asset Index Buffer"),
                            contents: bytemuck::cast_slice(indices.as_slice()),
                            usage: wgpu::BufferUsages::INDEX,
                        }),
                    num_indices: indices.len() as u32,
                };
                self.mesh_cache.insert(path.clone(), mesh);
            } else {
                log::error!("Failed to load mesh from '{}'", path);
            }
        }
        self.egui_winit
            .handle_platform_output(self.window, full_output.platform_output);

        let paint_jobs = self.egui_ctx.tessellate(full_output.shapes, 1.0);
        let screen_desc = ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: 1.0,
        };

        // Snapshot UI parameters we need for rendering so we don't hold long-lived borrows.
        let invert_light = self.ui_state.invert_light;
        let ambient = self.ui_state.ambient;
        let specular = self.ui_state.specular;
        let shininess = self.ui_state.shininess;
        let default_mesh_path = self.ui_state.glb_path.clone();

        let camera = &scene.camera;
        let view_proj = camera.view_projection_matrix();

        let mut light_dir = Vec3::new(-1.0, -1.0, -0.5).normalize();
        if invert_light {
            light_dir = -light_dir;
        }
        let ambient_color = [ambient; 3];
        let specular_color = [specular; 3];

        // Ensure all meshes referenced by entities are loaded into the cache
        // before we start the render pass to avoid mutable/immutable borrow conflicts.
        {
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
                    log::error!(
                        "Failed to load mesh for one or more entities from '{}'",
                        mesh_key
                    );
                }
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
                label: Some("Triangle Pass"),
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
            // Draw ground plane.
            {
                let model = engine::math::Mat4::IDENTITY;
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

            // Draw each entity with its mesh.
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

        // Upload egui data and render UI on top.
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.egui_renderer
            .update_buffers(&self.device, &self.queue, &mut encoder, &paint_jobs, &screen_desc);

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            self.egui_renderer
                .render(&mut rpass, &paint_jobs, &screen_desc);
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();
        Ok(())
    }
}

fn main() {
    env_logger::init();
    info!("Starting editor...");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let window = WindowBuilder::new()
        .with_title("Rust 3D Engine Editor")
        .build(&event_loop)
        .expect("Failed to create window");
    let window: &'static winit::window::Window = Box::leak(Box::new(window));

    let aspect = {
        let size = window.inner_size();
        size.width as f32 / size.height.max(1) as f32
    };

    let camera = Camera::new_perspective(
        Vec3::new(0.0, 2.0, 5.0),
        Vec3::ZERO,
        aspect,
        45_f32.to_radians(),
        0.1,
        100.0,
    );

    let scene = Scene {
        entities: vec![engine::Entity {
            name: "Entity 0".to_string(),
            transform: engine::math::Transform::identity(),
            velocity: Vec3::ZERO,
            mesh_path: "editor/3D assets/Weapon Pack/Models/GLTF format/pistol.glb".to_string(),
        }],
        camera,
    };

    let mut engine = Engine::new(scene);
    let mut renderer = pollster::block_on(Renderer::new(&event_loop, window));

    let mut input = InputState::default();
    let mut last_frame = Instant::now();

    event_loop
        .run(move |event, elwt| {
            match event {
                Event::WindowEvent { event, window_id } if window_id == window.id() => {
                    renderer
                        .egui_winit
                        .on_window_event(window, &event);

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
                                    rotate_camera_from_mouse(&mut engine.scene.camera, dx as f32, dy as f32);
                                }
                                input.last_cursor_pos = Some((position.x, position.y));
                            } else {
                                input.last_cursor_pos = Some((position.x, position.y));
                            }
                        }
                        WindowEvent::MouseWheel { delta, .. } => {
                            let scroll = match delta {
                                winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                                winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 50.0,
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

                            if !renderer.ui_state.paused {
                                update_camera_from_input(&mut engine.scene.camera, &input, dt);
                                engine.update(dt);
                            }

                            match renderer.render(&mut engine.scene, dt) {
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

    // Yaw around the global up axis.
    let yaw_rot = Quat::from_rotation_y(yaw);
    dir = yaw_rot * dir;

    // Pitch around the camera's right axis.
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

fn load_mesh_from_json(path: &str) -> Option<(Vec<Vertex>, Vec<u32>)> {
    use std::fs;
    use std::path::Path;

    let path = Path::new(path);
    let data = fs::read_to_string(path).ok()?;
    let mesh_file: MeshFile = serde_json::from_str(&data).ok()?;

    if mesh_file.vertices.is_empty() || mesh_file.indices.is_empty() {
        return None;
    }

    let mut vertices = Vec::with_capacity(mesh_file.vertices.len());
    for v in mesh_file.vertices {
        let position = engine::math::Vec3::from_array(v.position);
        let normal = position.normalize_or_zero();
        vertices.push(Vertex {
            position: v.position,
            normal: normal.to_array(),
            color: v.color,
        });
    }

    Some((
        vertices,
        mesh_file.indices.into_iter().map(|i| i as u32).collect(),
    ))
}

fn builtin_cube_mesh() -> (Vec<Vertex>, Vec<u32>) {
    let base_positions = [
        engine::math::Vec3::new(-0.5, -0.5, -0.5),
        engine::math::Vec3::new(0.5, -0.5, -0.5),
        engine::math::Vec3::new(0.5, 0.5, -0.5),
        engine::math::Vec3::new(-0.5, 0.5, -0.5),
        engine::math::Vec3::new(-0.5, -0.5, 0.5),
        engine::math::Vec3::new(0.5, -0.5, 0.5),
        engine::math::Vec3::new(0.5, 0.5, 0.5),
        engine::math::Vec3::new(-0.5, 0.5, 0.5),
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
        // back
        0, 1, 2, 2, 3, 0,
        // front
        4, 5, 6, 6, 7, 4,
        // left
        4, 0, 3, 3, 7, 4,
        // right
        1, 5, 6, 6, 2, 1,
        // bottom
        4, 5, 1, 1, 0, 4,
        // top
        3, 2, 6, 6, 7, 3,
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
        buffers
            .get(buffer.index())
            .map(|b| b.0.as_slice())
    });

    let positions: Vec<[f32; 3]> = reader.read_positions()?.collect();

    let normals: Vec<[f32; 3]> = if let Some(normals_iter) = reader.read_normals() {
        normals_iter.collect()
    } else {
        vec![[0.0, 1.0, 0.0]; positions.len()]
    };

    let colors: Vec<[f32; 3]> = if let Some(colors0) = reader.read_colors(0) {
        colors0
            .into_rgb_f32()
            .collect()
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
        buffers
            .get(buffer.index())
            .map(|b| b.0.as_slice())
    });

    let positions: Vec<[f32; 3]> = reader.read_positions()?.collect();

    let normals: Vec<[f32; 3]> = if let Some(normals_iter) = reader.read_normals() {
        normals_iter.collect()
    } else {
        vec![[0.0, 1.0, 0.0]; positions.len()]
    };

    let colors: Vec<[f32; 3]> = if let Some(colors0) = reader.read_colors(0) {
        colors0
            .into_rgb_f32()
            .collect()
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

#[derive(serde::Serialize, serde::Deserialize)]
struct SerializableTransform {
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SerializableEntity {
    name: String,
    transform: SerializableTransform,
    velocity: [f32; 3],
    mesh_path: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SerializableCamera {
    position: [f32; 3],
    target: [f32; 3],
    up: [f32; 3],
    fov_y_radians: f32,
    aspect: f32,
    z_near: f32,
    z_far: f32,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SerializableScene {
    entities: Vec<SerializableEntity>,
    camera: SerializableCamera,
}

fn save_scene_to_file(path: &str, scene: &Scene) -> Result<(), String> {
    use std::fs;
    use std::path::Path;

    let serializable = scene_to_serializable(scene);
    let json = serde_json::to_string_pretty(&serializable)
        .map_err(|e| format!("serialize error: {e}"))?;
    fs::write(Path::new(path), json).map_err(|e| format!("write error: {e}"))?;
    Ok(())
}

fn load_scene_from_file(path: &str) -> Option<Scene> {
    use std::fs;
    use std::path::Path;

    let data = fs::read_to_string(Path::new(path)).ok()?;
    let serializable: SerializableScene = serde_json::from_str(&data).ok()?;
    Some(serializable_to_scene(&serializable))
}

fn scene_to_serializable(scene: &Scene) -> SerializableScene {
    let entities = scene
        .entities
        .iter()
        .map(|e| SerializableEntity {
            name: e.name.clone(),
            transform: SerializableTransform {
                translation: e.transform.translation.to_array(),
                rotation: {
                    let q = e.transform.rotation;
                    [q.x, q.y, q.z, q.w]
                },
                scale: e.transform.scale.to_array(),
            },
            velocity: e.velocity.to_array(),
            mesh_path: e.mesh_path.clone(),
        })
        .collect();

    let cam = &scene.camera;
    let camera = SerializableCamera {
        position: cam.position.to_array(),
        target: cam.target.to_array(),
        up: cam.up.to_array(),
        fov_y_radians: cam.fov_y_radians,
        aspect: cam.aspect,
        z_near: cam.z_near,
        z_far: cam.z_far,
    };

    SerializableScene { entities, camera }
}

fn serializable_to_scene(data: &SerializableScene) -> Scene {
    let entities = data
        .entities
        .iter()
        .map(|e| engine::Entity {
            name: e.name.clone(),
            transform: engine::math::Transform {
                translation: Vec3::from_array(e.transform.translation),
                rotation: Quat::from_xyzw(
                    e.transform.rotation[0],
                    e.transform.rotation[1],
                    e.transform.rotation[2],
                    e.transform.rotation[3],
                ),
                scale: Vec3::from_array(e.transform.scale),
            },
            velocity: Vec3::from_array(e.velocity),
            mesh_path: e.mesh_path.clone(),
        })
        .collect();

    let cam = &data.camera;
    let camera = Camera {
        position: Vec3::from_array(cam.position),
        target: Vec3::from_array(cam.target),
        up: Vec3::from_array(cam.up),
        fov_y_radians: cam.fov_y_radians,
        aspect: cam.aspect,
        z_near: cam.z_near,
        z_far: cam.z_far,
    };

    Scene { entities, camera }
}
