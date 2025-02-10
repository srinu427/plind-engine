use std::sync::{Arc, Mutex};

use rhi::{
  CommandBufferID,
  FenceID,
  FramebufferID,
  InputSetID,
  PipelineID,
  RenderBackend,
};

pub struct MeshVertex{
  position: glam::Vec4,
  normal: glam::Vec4,
  tangent: glam::Vec4,
  bi_tangent: glam::Vec4,
  tex_coord: glam::Vec4,
}

pub struct MeshCPU{
  verts: Vec<MeshVertex>,
  indices: Vec<u32>,
}

pub struct PbrRenderer<B: RenderBackend>{
  pipeline: PipelineID,
  input_sets: Vec<InputSetID>,
  framebuffers: Vec<FramebufferID>,
  command_buffers: Vec<CommandBufferID>,
  fences: Vec<FenceID>,
  backend: Arc<Mutex<B>>
}

impl<B: RenderBackend> PbrRenderer<B>{
  pub fn new(backend: Arc<Mutex<B>>) -> Result<Self, String>{
    let mut backend_lock = backend
      .lock()
      .map_err(|e| format!("at backend lock: {e}"))?;
    let frame_count = backend_lock.get_swapchain_images().len();
    let fences = (0..frame_count)
      .map(|_| backend_lock.create_fence(false))
      .collect::<Result<Vec<_>, String>>()?;
    todo!();
  }
}

pub struct Renderer<B: RenderBackend>{
  backend: Arc<Mutex<B>>,
  pbr_renderer: PbrRenderer<B>,
}

impl<B: RenderBackend> Renderer<B>{
  pub fn new(backend: Arc<Mutex<B>>) -> Result<Renderer<B>, String>{
    let pbr_renderer = PbrRenderer::new(backend.clone())?;
    Ok(Self{ backend, pbr_renderer })
  }
}
