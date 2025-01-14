use bitflags::bitflags;

#[derive(Debug, Clone)]
pub struct GPUInfo{
  pub id: u32,
  pub name: String,
  pub integrated: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Resolution2D {
  width: u32,
  height: u32,
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

#[derive(Debug, Clone, Copy)]
pub struct ImageID(u32);

#[derive(Debug, Clone, Copy)]
pub struct ImageViewID(u32);

#[derive(Debug, Clone, Copy)]
pub struct SamplerID(u32);

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
pub struct BufferID(u32);

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
pub struct DescriptorLayoutID(u32);

#[derive(Debug, Clone, Copy)]
pub struct DescriptorPoolID(u32);

#[derive(Debug, Clone, Copy)]
pub struct DescriptorSetID(u32);

#[derive(Debug, Clone, Copy)]
pub struct PipelineAttchmentConfig {
  format: ImageFormat,
  initial_layout: ImageLayoutType,
  final_layout: ImageLayoutType,
}

#[derive(Debug, Clone, Copy)]
pub struct PipelineID(u32);

#[derive(Debug, Clone, Copy)]
pub enum RasterStyle {
  Fill,
  WireFrame{thickness: u32}
}

pub enum RenderBackendTask {
  // Meta
  OrderedTasks(Vec<Self>),
  UnorderedTasks(Vec<Self>),
  // Image Related
  Create2DImage{res: Resolution2D, format: ImageFormat, usage: ImageUsage},
  DestroyImage{id: ImageID},
  CreateImageView{image_id: ImageID},
  DestroyImageView{image_view_id: ImageViewID},
  // Buffer Related
  CreateBuffer{size: u64, usage: BufferUsage},
  DestroyBuffer{id: BufferID},
  // Descriptor Related
  CreateDescriptorLayout{binding_types: Vec<(u32, DescriptorType, ShaderStageFlags)>},
  DestroyDescriptorLayout{id: DescriptorLayoutID},
  CreateDescriptorPool{freeable: bool, limits: Vec<(DescriptorType, u32)>},
  DestroyDescriptorPool{id: DescriptorPoolID},
  AllocateDescriptorSet{pool: DescriptorPoolID, set_layouts: DescriptorLayoutID},
  UpdateDescriptorSetBufferBinding{
    set: DescriptorSetID,
    binding: u32,
    buffer_id: BufferID,
    offset: u64,
    len: Option<u64>,
  },
  UpdateDescriptorSetImageBinding{
    set: DescriptorSetID,
    binding: u32,
    image_view_id: ImageViewID,
    sampler: SamplerID,
    layout: ImageLayoutType,
  },
  // Pipeline Related
  CreateGraphicsPipeline{
    raster_style: RasterStyle,
    color_attachment_configs: Vec<PipelineAttchmentConfig>, 
    depth_attachment_config: Option<PipelineAttchmentConfig>,
  },
  DestroyPipeline{id: PipelineID},
}

pub enum RenderBackendTaskOutput {
  // Meta
  OrderedTasksOutput(Vec<Self>),
  UnorderedTasksOutput(Vec<Self>),
  // Image Related
  Create2DImageOutput(Result<ImageID, String>),
  DestroyImageOutput(Result<(), String>),
  CreateImageViewOutput(Result<ImageViewID, String>),
  DestroyImageViewOutput(Result<(), String>),
  // Buffer Related
  CreateBufferOutput(Result<BufferID, String>),
  DestroyBufferOutput(Result<(), String>),
  // Descriptor Related
  CreateDescriptorLayoutOutput(Result<DescriptorLayoutID, String>),
  DestroyDescriptorLayoutOutput(Result<(), String>),
  CreateDescriptorPoolOutput(Result<DescriptorPoolID, String>),
  DestroyDescriptorPoolOutput(Result<(), String>),
  AllocateDescriptorSetOuput(Result<DescriptorSetID, String>),
  UpdateDescriptorSetBufferBindingOutput(Result<(), String>),
  UpdateDescriptorSetImageBindingOutput(Result<(), String>),
  // Pipeline Related
  CreateGraphicsPipelineOutput(Result<PipelineID, String>),
  DestroyPipelineOutout(Result<(), String>),
}

pub trait RenderBackend {
  fn new() -> Result<Self, String> where Self: Sized;
  fn get_gpu_infos(&self) -> Vec<GPUInfo>;
  fn init(gpu_id: u32) -> Self;
  fn run_task(&mut self, task: RenderBackendTask) -> RenderBackendTaskOutput;
  fn destroy(&mut self);
}
