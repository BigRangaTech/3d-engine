use std::time::Instant;
use std::collections::HashMap;

use engine::{Engine, Scene};
use engine::math::{Camera, Mat4, Vec3, Quat};
use log::info;
use serde::Deserialize;
use gltf;
use egui::TextureHandle;
use rayon::ThreadPoolBuilder;

mod assets;
use assets::{convert_assets_to_glb, discover_mesh_assets};
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
    show_assets_window: bool,
    filter_tag: String,
    filter_by_layer: bool,
    filter_layer: i32,
    scene_path: String,
    // Undo/redo.
    can_undo: bool,
    can_redo: bool,
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
    assets_loaded: bool,
    undo_stack: Vec<Scene>,
    redo_stack: Vec<Scene>,
    mesh_cache: HashMap<String, GpuMesh>,
    ground_mesh: GpuMesh,
    dragging_entity: Option<usize>,
    drag_plane_y: f32,
    default_mesh_path: String,
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
                @location(2) world_pos: vec3<f32>,
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
                out.world_pos = world_pos.xyz;
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

                var base = in.color;
                // Checkerboard ground if we're on the ground plane.
                let ground_color = vec3<f32>(0.3, 0.3, 0.3);
                if abs(in.world_pos.y) < 0.01 && all(abs(in.color - ground_color) < vec3<f32>(0.001, 0.001, 0.001)) {
                    let tile_size = 2.0;
                    let tx = floor(in.world_pos.x / tile_size);
                    let tz = floor(in.world_pos.z / tile_size);
                    let sum = tx + tz;
                    let pattern = sum - 2.0 * floor(sum * 0.5);
                    let checker = step(1.0, pattern);
                    let dark = ground_color * 0.6;
                    let light = ground_color * 1.4;
                    base = mix(light, dark, checker);
                }
                let ambient = camera.ambient_color;
                let diffuse_color = base * (diffuse + fill);

                // Simple specular using a fixed view direction.
                let view_dir = normalize(vec3<f32>(0.0, 0.0, 1.0));
                let reflect_dir = reflect(-l, n);
                let spec_angle = max(dot(view_dir, reflect_dir), 0.0);
                let specular = pow(spec_angle, camera.shininess) * camera.specular_color;

                var lit = ambient + diffuse_color + specular;

                // Simple distance-based fog (from world origin).
                let dist = length(in.world_pos);
                let fog_start = 20.0;
                let fog_end = 80.0;
                let fog_factor = clamp((fog_end - dist) / (fog_end - fog_start), 0.0, 1.0);
                let fog_color = vec3<f32>(0.05, 0.07, 0.10);
                lit = mix(fog_color, lit, fog_factor);

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
            show_assets_window: false,
            filter_tag: String::new(),
            filter_by_layer: false,
            filter_layer: 0,
            scene_path: "scene.json".to_string(),
            can_undo: false,
            can_redo: false,
        };

        let mesh_assets = Vec::new();

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
            assets_loaded: false,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            mesh_cache,
            ground_mesh,
            dragging_entity: None,
            drag_plane_y: 0.0,
            default_mesh_path: default_mesh_path.to_string(),
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
        // Snapshot scene at the start of the frame so we can push
        // an undo state if a gizmo drag begins this frame.
        let scene_snapshot = scene.clone();

        // Build editor UI with egui.
        let raw_input = self.egui_winit.take_egui_input(self.window);
        let mut delete_entity: Option<usize> = None;
        let mut duplicate_entity: Option<usize> = None;
        let mut set_player_for: Option<usize> = None;
        let mut set_trigger_for: Option<usize> = None;
        let mut align_ground_for: Option<usize> = None;
        let mut reload_glb: Option<String> = None;
        let mut convert_assets: bool = false;
        let mut do_undo: bool = false;
        let mut do_redo: bool = false;
        let can_undo = self.ui_state.can_undo;
        let can_redo = self.ui_state.can_redo;
        let mut gizmo_drag_started = false;
        let mut inspector_changed = false;
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

                let input = ctx.input(|i| i.clone());
                if input.modifiers.ctrl && input.key_pressed(egui::Key::Z) {
                    do_undo = true;
                }
                if input.modifiers.ctrl && input.key_pressed(egui::Key::Y) {
                    do_redo = true;
                }
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
                if ui.button("Assets").clicked() {
                    ui_state.show_assets_window = true;
                }

                ui.separator();
                ui.heading("Physics");
                ui.horizontal(|ui| {
                    ui.label("Gravity");
                    if ui
                        .add(egui::DragValue::new(&mut scene_ref.gravity.x).speed(0.1))
                        .changed()
                    {
                        inspector_changed = true;
                    }
                    if ui
                        .add(egui::DragValue::new(&mut scene_ref.gravity.y).speed(0.1))
                        .changed()
                    {
                        inspector_changed = true;
                    }
                    if ui
                        .add(egui::DragValue::new(&mut scene_ref.gravity.z).speed(0.1))
                        .changed()
                    {
                        inspector_changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Damping");
                    if ui
                        .add(
                            egui::DragValue::new(&mut scene_ref.linear_damping)
                                .speed(0.01)
                                .range(0.0..=5.0),
                        )
                        .changed()
                    {
                        inspector_changed = true;
                    }
                });

                ui.separator();
                ui.heading("Transform helpers");
                ui.checkbox(&mut ui_state.grid_snap, "Snap to grid");
                ui.add(
                    egui::DragValue::new(&mut ui_state.grid_size)
                        .speed(0.1)
                        .range(0.1..=100.0)
                        .prefix("Grid size: "),
                );

                ui.separator();
                ui.heading("Entities");
                ui.horizontal(|ui| {
                    ui.label("Filter tag");
                    ui.text_edit_singleline(&mut ui_state.filter_tag);
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut ui_state.filter_by_layer, "Filter layer");
                    ui.add(egui::DragValue::new(&mut ui_state.filter_layer).speed(1.0));
                });

                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(can_undo, egui::Button::new("Undo"))
                        .clicked()
                    {
                        do_undo = true;
                    }
                    if ui
                        .add_enabled(can_redo, egui::Button::new("Redo"))
                        .clicked()
                    {
                        do_redo = true;
                    }
                });

                if ui.button("Add entity").clicked() {
                    self.undo_stack.push(scene_ref.clone());
                    self.redo_stack.clear();
                    let index = scene_ref.entities.len();
                    let mut mesh_path = if let Some(sel) = ui_state.selected_entity {
                        scene_ref
                            .entities
                            .get(sel)
                            .map(|e| e.mesh_path.clone())
                            .unwrap_or_else(|| ui_state.glb_path.clone())
                    } else {
                        ui_state.glb_path.clone()
                    };
                    if mesh_path.is_empty() {
                        mesh_path = self.default_mesh_path.clone();
                    }
                    // Spawn new entities in front of the camera so they
                    // don't all start exactly on top of each other.
                    let cam = &scene_ref.camera;
                    let mut forward = cam.target - cam.position;
                    if forward.length_squared() == 0.0 {
                        forward = Vec3::new(0.0, 0.0, -1.0);
                    }
                    forward = forward.normalize();
                    let base_pos = cam.position + forward * 5.0;
                    // Slight offset per index so multiple new entities are visible.
                    let offset = Vec3::new(index as f32 * 5.0, 0.0, 0.0);
                    scene_ref.entities.push(engine::Entity {
                        name: format!("Entity {}", index),
                        transform: engine::math::Transform {
                            translation: base_pos + offset,
                            rotation: Quat::IDENTITY,
                            scale: Vec3::ONE,
                        },
                        velocity: Vec3::ZERO,
                        acceleration: Vec3::ZERO,
                        mesh_path,
                        is_character: false,
                        // If a tag filter is active, give the new entity that tag
                        // so it is visible under the current filter.
                        tag: if ui_state.filter_tag.is_empty() {
                            String::new()
                        } else {
                            ui_state.filter_tag.clone()
                        },
                        layer: 0,
                    });
                }

                if ui.button("Spawn 1000 stress entities").clicked() {
                    self.undo_stack.push(scene_ref.clone());
                    self.redo_stack.clear();
                    let base_index = scene_ref.entities.len();
                    let mesh_path = ui_state.glb_path.clone();
                    let mut count = 0;
                    'outer: for x in 0..10 {
                        for y in 0..10 {
                            for z in 0..10 {
                                let idx = base_index + count;
                                let pos = Vec3::new(
                                    x as f32 * 2.0,
                                    y as f32 * 2.0,
                                    z as f32 * 2.0,
                                );
                                scene_ref.entities.push(engine::Entity {
                                    name: format!("Stress {}", idx),
                                    transform: engine::math::Transform {
                                        translation: pos,
                                        rotation: Quat::IDENTITY,
                                        scale: Vec3::ONE,
                                    },
                                    velocity: Vec3::ZERO,
                                    acceleration: Vec3::ZERO,
                                    mesh_path: mesh_path.clone(),
                                    is_character: false,
                                    tag: "stress".to_string(),
                                    layer: 0,
                                });
                                count += 1;
                                if count >= 1000 {
                                    break 'outer;
                                }
                            }
                        }
                    }
                }

                if ui.button("Delete stress entities").clicked() {
                    self.undo_stack.push(scene_ref.clone());
                    self.redo_stack.clear();
                    scene_ref.entities.retain(|e| e.tag != "stress");
                    ui_state.selected_entity = None;
                }

                for (i, entity) in scene_ref.entities.iter().enumerate() {
                    if !ui_state.filter_tag.is_empty()
                        && !entity.tag.contains(&ui_state.filter_tag)
                    {
                        continue;
                    }
                    if ui_state.filter_by_layer && entity.layer != ui_state.filter_layer {
                        continue;
                    }
                    let label = format!(
                        "{} ({}) [tag: {} layer: {}]",
                        entity.name, i, entity.tag, entity.layer
                    );
                    let selected = ui_state.selected_entity == Some(i);
                    if ui.selectable_label(selected, label).clicked() {
                        ui_state.selected_entity = Some(i);
                    }
                }

                if let Some(i) = ui_state.selected_entity {
                    if let Some(entity) = scene_ref.entities.get_mut(i) {
                        ui.separator();
                        ui.label("Name");
                        if ui.text_edit_singleline(&mut entity.name).changed() {
                            inspector_changed = true;
                        }

                        ui.label(format!(
                            "Mesh: {}",
                            if entity.mesh_path.is_empty() {
                                "<default>".to_string()
                            } else {
                                entity.mesh_path.clone()
                            }
                        ));

                        ui.horizontal(|ui| {
                            ui.label("Tag");
                            if ui.text_edit_singleline(&mut entity.tag).changed() {
                                inspector_changed = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Layer");
                            if ui
                                .add(egui::DragValue::new(&mut entity.layer).speed(1.0))
                                .changed()
                            {
                                inspector_changed = true;
                            }
                            if ui.button("Show this layer").clicked() {
                                ui_state.filter_tag.clear();
                                ui_state.filter_by_layer = true;
                                ui_state.filter_layer = entity.layer;
                            }
                        });

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
                            if resp_x.changed() || resp_y.changed() || resp_z.changed() {
                                inspector_changed = true;
                            }
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
                            let s_x = ui.add(
                                egui::DragValue::new(&mut entity.transform.scale.x).speed(0.1),
                            );
                            let s_y = ui.add(
                                egui::DragValue::new(&mut entity.transform.scale.y).speed(0.1),
                            );
                            let s_z = ui.add(
                                egui::DragValue::new(&mut entity.transform.scale.z).speed(0.1),
                            );
                            if s_x.changed() || s_y.changed() || s_z.changed() {
                                inspector_changed = true;
                            }
                        });

                        ui.separator();
                        ui.label("Selected entity physics");
                        ui.horizontal(|ui| {
                            ui.label("Acceleration");
                            let a_x = ui.add(
                                egui::DragValue::new(&mut entity.acceleration.x).speed(0.1),
                            );
                            let a_y = ui.add(
                                egui::DragValue::new(&mut entity.acceleration.y).speed(0.1),
                            );
                            let a_z = ui.add(
                                egui::DragValue::new(&mut entity.acceleration.z).speed(0.1),
                            );
                            if a_x.changed() || a_y.changed() || a_z.changed() {
                                inspector_changed = true;
                            }
                        });
                        if ui
                            .checkbox(
                                &mut entity.is_character,
                                "Character (no auto-spin)",
                            )
                            .changed()
                        {
                            inspector_changed = true;
                        }
                        ui.horizontal(|ui| {
                            if ui.button("Set as player").clicked() {
                                set_player_for = Some(i);
                            }
                            if ui.button("Set as trigger").clicked() {
                                set_trigger_for = Some(i);
                            }
                        });

                        ui.horizontal(|ui| {
                        if ui.button("Align to ground").clicked() {
                            align_ground_for = Some(i);
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
                ui.heading("Scene");
                ui.horizontal(|ui| {
                    ui.label("File");
                    ui.text_edit_singleline(&mut ui_state.scene_path);
                });
                if ui.button("Save").clicked() {
                    if let Err(err) =
                        engine::scene_io::save_scene_to_file(&ui_state.scene_path, scene_ref)
                    {
                        log::error!("Failed to save scene '{}': {}", ui_state.scene_path, err);
                    }
                }
                if ui.button("Load").clicked() {
                    if let Some(new_scene) =
                        engine::scene_io::load_scene_from_file(&ui_state.scene_path)
                    {
                        self.undo_stack.push(scene_ref.clone());
                        self.redo_stack.clear();
                        *scene_ref = new_scene;
                        ui_state.selected_entity = None;
                    } else {
                        log::error!("Failed to load scene '{}'", ui_state.scene_path);
                    }
                }

                ui.separator();
            });

            // Viewport gizmo for moving/rotating the selected entity.
            if let Some(sel) = ui_state.selected_entity {
                if let Some(entity) = scene_ref.entities.get_mut(sel) {
                    let view_proj = scene_ref.camera.view_projection_matrix();
                    let world_pos = entity.transform.translation;
                    let pos4 = view_proj
                        * engine::math::Mat4::from_translation(world_pos)
                        * engine::math::Vec3::new(0.0, 0.0, 0.0).extend(1.0);
                    let w = pos4.w;
                    if w > 0.0 {
                        let ndc_x = pos4.x / w;
                        let ndc_y = pos4.y / w;
                        let screen_x =
                            (ndc_x * 0.5 + 0.5) * self.config.width as f32;
                        let screen_y =
                            (-ndc_y * 0.5 + 0.5) * self.config.height as f32;

                        let screen_pos = egui::pos2(screen_x, screen_y);
                        egui::Area::new(egui::Id::new("entity_gizmo"))
                            .order(egui::Order::Foreground)
                            .fixed_pos(egui::pos2(0.0, 0.0))
                            .show(ctx, |ui| {
                                let painter = ui.painter();
                                let axis_len = 30.0;
                                let thickness = 6.0;

                                // Axis X handle (red), horizontal.
                                let rect_x = egui::Rect::from_center_size(
                                    screen_pos + egui::vec2(axis_len * 0.5, 0.0),
                                    egui::vec2(axis_len, thickness),
                                );
                                // Axis Z handle (green), vertical (screen space).
                                let rect_z = egui::Rect::from_center_size(
                                    screen_pos + egui::vec2(0.0, -axis_len * 0.5),
                                    egui::vec2(thickness, axis_len),
                                );
                                // Center handle for rotation/plane move.
                                let rect_center = egui::Rect::from_center_size(
                                    screen_pos,
                                    egui::vec2(18.0, 18.0),
                                );

                                let id_x = egui::Id::new("gizmo_translate_x");
                                let id_z = egui::Id::new("gizmo_translate_z");
                                let id_c = egui::Id::new("gizmo_center");

                                let response_x =
                                    ui.interact(rect_x, id_x, egui::Sense::drag());
                                let response_z =
                                    ui.interact(rect_z, id_z, egui::Sense::drag());
                                let response_c =
                                    ui.interact(rect_center, id_c, egui::Sense::drag());

                                // Draw handles.
                                painter.rect_stroke(
                                    rect_center,
                                    4.0,
                                    egui::Stroke::new(1.0, egui::Color32::YELLOW),
                                );
                                painter.line_segment(
                                    [
                                        screen_pos,
                                        screen_pos + egui::vec2(axis_len, 0.0),
                                    ],
                                    egui::Stroke::new(2.0, egui::Color32::RED),
                                );
                                painter.line_segment(
                                    [
                                        screen_pos,
                                        screen_pos - egui::vec2(0.0, axis_len),
                                    ],
                                    egui::Stroke::new(2.0, egui::Color32::GREEN),
                                );

                                if response_x.drag_started()
                                    || response_z.drag_started()
                                    || response_c.drag_started()
                                {
                                    gizmo_drag_started = true;
                                }

                                if self.dragging_entity.is_none() {
                                    let (pointer_delta, modifiers) =
                                        ui.ctx().input(|i| (i.pointer.delta(), i.modifiers));

                                    // Axis-constrained translation.
                                    if response_x.dragged() {
                                        let scale = 0.01;
                                        entity.transform.translation.x +=
                                            pointer_delta.x as f32 * scale;
                                    } else if response_z.dragged() {
                                        let scale = 0.01;
                                        entity.transform.translation.z -=
                                            pointer_delta.y as f32 * scale;
                                    } else if response_c.dragged() {
                                        // Center: Shift-drag rotates, otherwise free XZ move.
                                        if modifiers.shift {
                                            let angle = -pointer_delta.x as f32 * 0.01;
                                            let rot = Quat::from_rotation_y(angle);
                                            entity.transform.rotation =
                                                rot * entity.transform.rotation;
                                        } else {
                                            let scale = 0.01;
                                            entity.transform.translation.x +=
                                                pointer_delta.x as f32 * scale;
                                            entity.transform.translation.z -=
                                                pointer_delta.y as f32 * scale;
                                        }
                                    }
                                }
                            });
                    }
                }
            }

            if ui_state.show_assets_window {
                egui::Window::new("Assets")
                    .open(&mut ui_state.show_assets_window)
                    .resizable(true)
                    .default_width(320.0)
                    .show(ctx, |ui| {
                        if !self.assets_loaded {
                            ui.label("Please wait, loading assets...");
                            // Request a one‑time load of assets after this frame.
                            convert_assets = false;
                        } else {
                            if ui
                                .button("Convert assets (copy + FBX -> GLB)")
                                .clicked()
                            {
                                convert_assets = true;
                            }
                            ui.separator();
                            egui::ScrollArea::vertical().show(ui, |ui| {
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
                                                if let Some(ent) = scene_ref
                                                    .entities
                                                    .get_mut(sel)
                                                {
                                                    ent.mesh_path =
                                                        asset.mesh_path.clone();
                                                    inspector_changed = true;
                                                }
                                            }
                                        }
                                    });
                                    ui.separator();
                                }
                            });
                        }
                    });
            }
            })
        };

        // If a gizmo drag or inspector edit started this frame, push an undo snapshot
        // captured at the beginning of render.
        if gizmo_drag_started || inspector_changed {
            self.undo_stack.push(scene_snapshot);
            self.redo_stack.clear();
        }

        // If the primary mouse button is no longer held, stop any drag in progress.
        let pointer_down = self.egui_ctx.input(|i| i.pointer.primary_down());
        if !pointer_down {
            self.dragging_entity = None;
        }
        // Apply deferred entity actions that require whole-scene access.
        if set_player_for.is_some() || set_trigger_for.is_some() || align_ground_for.is_some() {
            self.undo_stack.push(scene.clone());
            self.redo_stack.clear();
        }
        if let Some(i) = set_player_for {
            if let Some(entity) = scene.entities.get_mut(i) {
                entity.tag = "player".to_string();
                entity.is_character = true;
            }
        }
        if let Some(i) = set_trigger_for {
            if let Some(entity) = scene.entities.get_mut(i) {
                entity.tag = "trigger".to_string();
                entity.is_character = false;
            }
        }
        if let Some(i) = align_ground_for {
            if let Some(entity) = scene.entities.get_mut(i) {
                entity.transform.translation.y = 0.0;
            }
        }
        if let Some(i) = delete_entity {
            self.undo_stack.push(scene.clone());
            self.redo_stack.clear();
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
        if !self.assets_loaded && self.ui_state.show_assets_window {
            let mut assets = discover_mesh_assets(&self.egui_ctx, "editor/3D assets");
            assets.extend(discover_mesh_assets(
                &self.egui_ctx,
                "editor/Converted assets",
            ));
            self.mesh_assets = assets;
            self.assets_loaded = true;
        }
        if do_undo {
            if let Some(prev) = self.undo_stack.pop() {
                self.redo_stack.push(scene.clone());
                *scene = prev;
            }
        }
        if do_redo {
            if let Some(next) = self.redo_stack.pop() {
                self.undo_stack.push(scene.clone());
                *scene = next;
            }
        }
        self.ui_state.can_undo = !self.undo_stack.is_empty();
        self.ui_state.can_redo = !self.redo_stack.is_empty();

        if convert_assets {
            convert_assets_to_glb("editor/3D assets", "editor/Converted assets");
            let mut assets = discover_mesh_assets(&self.egui_ctx, "editor/3D assets");
            assets.extend(discover_mesh_assets(
                &self.egui_ctx,
                "editor/Converted assets",
            ));
            self.mesh_assets = assets;
            self.assets_loaded = true;
        }

        if let Some(i) = duplicate_entity {
            if i < scene.entities.len() {
                let src = &scene.entities[i];
                // Offset the duplicate slightly so it doesn't sit exactly
                // on top of the original.
                let mut new_transform = engine::math::Transform {
                    translation: src.transform.translation,
                    rotation: src.transform.rotation,
                    scale: src.transform.scale,
                };
                new_transform.translation.x += 5.0;
                scene.entities.push(engine::Entity {
                    name: format!("{} Copy", src.name),
                    transform: new_transform,
                    velocity: Vec3::ZERO,
                    acceleration: Vec3::ZERO,
                    mesh_path: src.mesh_path.clone(),
                    is_character: src.is_character,
                    tag: src.tag.clone(),
                    layer: src.layer,
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
        let default_mesh_path = self.default_mesh_path.clone();
        let filter_tag = self.ui_state.filter_tag.clone();
        let filter_by_layer = self.ui_state.filter_by_layer;
        let filter_layer = self.ui_state.filter_layer;

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
                if !filter_tag.is_empty() && !entity.tag.contains(&filter_tag) {
                    continue;
                }
                if filter_by_layer && entity.layer != filter_layer {
                    continue;
                }
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

            // Draw each entity with its mesh (respecting tag/layer filters).
            for entity in &scene.entities {
                if !filter_tag.is_empty() && !entity.tag.contains(&filter_tag) {
                    continue;
                }
                if filter_by_layer && entity.layer != filter_layer {
                    continue;
                }
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
    // Configure Rayon to use up to 16 threads (or however many are available, if fewer).
    if let Ok(threads) = std::thread::available_parallelism() {
        let max_threads = 16usize;
        let n = threads.get().min(max_threads).max(1);
        let _ = ThreadPoolBuilder::new().num_threads(n).build_global();
    }

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
            acceleration: Vec3::ZERO,
            mesh_path: "editor/3D assets/Weapon Pack/Models/GLTF format/pistol.glb".to_string(),
            is_character: false,
            tag: String::new(),
            layer: 0,
        }],
        camera,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        linear_damping: 0.2,
    };

    let mut engine = Engine::new(scene);
    let mut renderer = pollster::block_on(Renderer::new(&event_loop, window));

    let mut input = InputState::default();
    let mut last_frame = Instant::now();

    event_loop
        .run(move |event, elwt| {
            match event {
                Event::WindowEvent { event, window_id } if window_id == window.id() => {
                    let response = renderer
                        .egui_winit
                        .on_window_event(window, &event);

                    // If egui wants to capture this event, don't also feed it to our input.
                    if response.consumed {
                        return;
                    }

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
                            } else if button == winit::event::MouseButton::Left {
                                let pressed = state == winit::event::ElementState::Pressed;
                                if pressed {
                                    if let Some((x, y)) = input.last_cursor_pos {
                                        if let Some(index) = pick_entity_at_cursor(
                                            &engine.scene,
                                            &engine.scene.camera,
                                            renderer.config.width,
                                            renderer.config.height,
                                            x,
                                            y,
                                        ) {
                                            renderer.undo_stack.push(engine.scene.clone());
                                            renderer.redo_stack.clear();
                                            renderer.ui_state.selected_entity = Some(index);
                                            renderer.drag_plane_y = engine.scene.entities[index]
                                                .transform
                                                .translation
                                                .y;
                                            renderer.dragging_entity = Some(index);
                                        } else {
                                            renderer.dragging_entity = None;
                                            renderer.ui_state.selected_entity = None;
                                        }
                                    }
                                } else {
                                    renderer.dragging_entity = None;
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

                            if let Some(index) = renderer.dragging_entity {
                                if let Some((origin, dir)) = screen_ray(
                                    &engine.scene.camera,
                                    renderer.config.width,
                                    renderer.config.height,
                                    position.x,
                                    position.y,
                                ) {
                                    if let Some(hit) = ray_plane_y(origin, dir, renderer.drag_plane_y) {
                                        if let Some(entity) = engine.scene.entities.get_mut(index) {
                                            let mut pos = hit;
                                            if renderer.ui_state.grid_snap
                                                && renderer.ui_state.grid_size > 0.0
                                            {
                                                let step = renderer.ui_state.grid_size;
                                                let snap = |v: f32| (v / step).round() * step;
                                                pos.x = snap(pos.x);
                                                pos.y = snap(pos.y);
                                                pos.z = snap(pos.z);
                                            }
                                            entity.transform.translation = pos;
                                        }
                                    }
                                }
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

fn screen_ray(
    camera: &Camera,
    width: u32,
    height: u32,
    x: f64,
    y: f64,
) -> Option<(Vec3, Vec3)> {
    if width == 0 || height == 0 {
        return None;
    }

    let view_proj: Mat4 = camera.view_projection_matrix();
    let inv = view_proj.inverse();

    let sx = x as f32 / width as f32;
    let sy = y as f32 / height as f32;
    let ndc_x = sx * 2.0 - 1.0;
    let ndc_y = 1.0 - sy * 2.0;

    let near_clip = Vec3::new(ndc_x, ndc_y, 0.0).extend(1.0);
    let far_clip = Vec3::new(ndc_x, ndc_y, 1.0).extend(1.0);

    let near_world4 = inv * near_clip;
    let far_world4 = inv * far_clip;

    let near_world = Vec3::new(
        near_world4.x / near_world4.w,
        near_world4.y / near_world4.w,
        near_world4.z / near_world4.w,
    );
    let far_world = Vec3::new(
        far_world4.x / far_world4.w,
        far_world4.y / far_world4.w,
        far_world4.z / far_world4.w,
    );

    let origin = camera.position;
    let dir = (far_world - origin).normalize();
    Some((origin, dir))
}

fn ray_sphere_intersect(
    origin: Vec3,
    dir: Vec3,
    center: Vec3,
    radius: f32,
) -> Option<f32> {
    let oc = origin - center;
    let a = dir.dot(dir);
    let b = 2.0 * oc.dot(dir);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let mut t = (-b - sqrt_disc) / (2.0 * a);
    if t < 0.0 {
        t = (-b + sqrt_disc) / (2.0 * a);
        if t < 0.0 {
            return None;
        }
    }
    Some(t)
}

fn pick_entity_at_cursor(
    scene: &Scene,
    camera: &Camera,
    width: u32,
    height: u32,
    x: f64,
    y: f64,
) -> Option<usize> {
    let (origin, dir) = screen_ray(camera, width, height, x, y)?;
    let mut best_t = f32::MAX;
    let mut best_index = None;

    for (i, entity) in scene.entities.iter().enumerate() {
        let center = entity.transform.translation;
        let scale = entity.transform.scale;
        let mut radius = scale.x.abs().max(scale.y.abs()).max(scale.z.abs());
        if radius <= 0.0 {
            radius = 0.5;
        }

        if let Some(t) = ray_sphere_intersect(origin, dir, center, radius) {
            if t < best_t {
                best_t = t;
                best_index = Some(i);
            }
        }
    }

    best_index
}

fn ray_plane_y(origin: Vec3, dir: Vec3, plane_y: f32) -> Option<Vec3> {
    let denom = dir.y;
    if denom.abs() < 1e-4 {
        return None;
    }
    let t = (plane_y - origin.y) / denom;
    if t < 0.0 {
        return None;
    }
    Some(origin + dir * t)
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

// Scene save/load lives in the engine crate now (engine::scene_io).
