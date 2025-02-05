use std::path::PathBuf;
use bitflags::bitflags;
pub use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

#[derive(Debug, Clone)]
pub struct GPUInfo{
  pub id: u32,
  pub name: String,
  pub integrated: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Resolution2D {
  pub width: u32,
  pub height: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct SwapchainInfo{
  pub res: Resolution2D,
  pub image_count: u32,
}

pub enum MemoryLocation{
  Any,
  GPU,
  Shared,
}

bitflags! {
  #[derive(Debug, Clone, Copy)]
  pub struct ImageUsage: u32 {
    const COPY_SRC = 0b00000001;
    const COPY_DST = 0b00000010;
    const BLIT_SRC = 0b00000100;
    const BLIT_DST = 0b00001000;
    const SHADER_SAMPLED = 0b00010000;
    const SHADER_STORAGE = 0b00100000;
  }
}

#[derive(Debug, Clone, Copy)]
pub enum ImageFormat {
  Texture,
  Float,
  Depth,
  RenderIntermediate,
  Presentation,
}

#[derive(Debug, Clone, Copy)]
pub enum MemAccessType {
  TransferRead,
  TransferWrite,
  HostRead,
  HostWrite,
  ShaderRead,
  ShaderWrite,
  ColorAttachmentRead,
  ColorAttachmentWrite,
  DepthAttachmentRead,
  DepthAttachmentWrite,
  MemoryRead,
  MemoryWrite,
}

#[derive(Debug, Clone, Copy)]
pub enum ImageLayoutType {
  Undefined,
  General,
  ColorAttachment,
  DepthAttachment,
  DepthRO,
  ShaderRO,
  TransferSrc,
  TransferDst,
}

#[derive(Debug, Clone, Copy)]
pub enum ImageSampleCount {
  E1,
  E2,
  E4,
  E8,
  E16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageID(pub u32);

#[derive(Debug, Clone, Copy)]
pub struct ImageViewID(pub u32);

#[derive(Debug, Clone, Copy)]
pub struct SamplerID(pub u32);

bitflags! {
  #[derive(Debug, Clone, Copy)]
  pub struct BufferUsage: u32 {
    const COPY_SRC = 0b00000001;
    const COPY_DST = 0b00000010;
    const UNIFORM = 0b00000100;
    const STORAGE = 0b00001000;
  }
}

#[derive(Debug, Clone, Copy)]
pub struct BufferID(pub u32);

bitflags! {
  #[derive(Debug, Clone, Copy)]
  pub struct ShaderStageFlags: u32 {
    const VERTEX = 0b00000001;
    const FRAGMENT = 0b00000010;
  }
}

#[derive(Debug, Clone, Copy)]
pub enum DescriptorType {
  Uniform,
  Storage,
  Sampler2D,
}

#[derive(Debug, Clone, Copy)]
pub struct DescriptorSetID(pub u32);

#[derive(Debug, Clone, Copy)]
pub struct PipelineID(pub u32);

#[derive(Debug, Clone, Copy)]
pub struct FramebufferID(pub u32);

#[derive(Debug, Clone, Copy)]
pub enum RasterStyle {
  Fill,
  WireFrame{thickness: u32}
}

#[derive(Debug, Clone, Copy)]
pub struct InputSetID(pub u32);

#[derive(Debug, Clone, Copy)]
pub struct FenceID(pub u32);

#[derive(Debug, Clone, Copy)]
pub struct CommandBufferID(pub u32);

#[derive(Debug, Clone, Copy)]
pub struct DrawInfo{
  offset: u32,
  count: u32,
}

#[derive(Debug, Clone)]
pub enum GPUCommands{
  CopyBufferToBuffer{src: BufferID, dst: BufferID},
  CopyBufferToImage{src: BufferID, dst: ImageID},
  BlitImage{src: ImageID, dst: ImageID},
  RunGraphicsPipeline{
    pipeline: PipelineID,
    framebuffer: FramebufferID,
    input_set: InputSetID,
    draw_infos: Vec<DrawInfo>
  },
}

#[trait_variant::make(RenderBackend: Send)]
pub trait LocalRenderBackend {
  fn get_swapchain_info(&self) -> SwapchainInfo;
  fn create_buffer(
    &mut self,
    size: u64,
    usage: BufferUsage,
    memory_location: MemoryLocation
  ) -> Result<BufferID, String>;

  fn create_texture_2d(
    &mut self,
    res: Resolution2D,
    format: ImageFormat,
    usage: ImageUsage,
    memory_location: MemoryLocation
  ) -> Result<ImageID, String>;

  async fn create_graphics_pipeline(
    &mut self,
    raster_style: RasterStyle,
    color_attachment_formats: Vec<ImageFormat>,
    depth_attachment_formats: Option<ImageFormat>,
    max_buffer_count: u32,
    max_texture_count: u32,
    vertex_shader: PathBuf,
    fragment_shader: PathBuf,
  ) -> Result<PipelineID, String>;

  fn create_frame_buffer(
    &mut self,
    pipeline_id: PipelineID,
    color_attachments: Vec<ImageID>,
    depth_attachment: Option<ImageID>,
  ) -> Result<FramebufferID, String>;

  fn create_input_set(&mut self, pipeline_id: PipelineID) -> Result<InputSetID, String>;

  fn update_input_set(
    &mut self,
    input_set: InputSetID,
    buffers: Vec<BufferID>,
    textures: Vec<ImageID>
  ) -> Result<(), String>;

  fn create_fence(&mut self, signaled: bool) -> Result<FenceID, String>;

  async fn wait_for_fence(&self, fence_id: FenceID) -> Result<(), String>;

  fn create_command_buffer(&mut self) -> Result<CommandBufferID, String>;

  fn compile_commands(
    &self,
    command_buffer: CommandBufferID,
    commands: Vec<GPUCommands>
  ) -> Result<(), String>;

  fn run_commands(&self, command_buffer: CommandBufferID, fence_id: FenceID) -> Result<(), String>;
}
