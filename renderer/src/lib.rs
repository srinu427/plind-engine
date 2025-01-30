use rhi::{
  CommandBufferID,
  FenceID,
  FramebufferID,
  InputSetID,
  PipelineID,
  RenderBackend,
  RenderBackendInitializer
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

pub struct PbrRenderer{
  pipeline: PipelineID,
  input_sets: Vec<InputSetID>,
  framebuffers: Vec<FramebufferID>,
  command_buffers: Vec<CommandBufferID>,
  fences: Vec<FenceID>,
}

impl PbrRenderer{
  pub async fn new(backend: &mut impl RenderBackend) -> Self{
    backend.create_fence(false);
  }
}

pub struct Renderer<B: RenderBackend>{
  backend: B,
  pbr_renderer: PbrRenderer,
}

impl<B: RenderBackend> Renderer<B>{
  pub fn new(backend: impl RenderBackend) -> Result<Renderer<B>, String>{
    Ok(Self{ backend })
  }
}
