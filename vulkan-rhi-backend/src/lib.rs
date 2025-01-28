use std::fmt::Debug;
use std::path::PathBuf;
pub use rhi;
use ash::{vk, ext, khr};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{
  Allocation,
  AllocationCreateDesc,
  AllocationScheme,
  Allocator,
  AllocatorCreateDesc
};
use seq_id_store::SequentialIDStore;
use tokio::fs;

fn translate_memory_location(memory_location: rhi::MemoryLocation) -> MemoryLocation {
  match memory_location {
    rhi::MemoryLocation::Any => {MemoryLocation::GpuOnly}
    rhi::MemoryLocation::GPU => {MemoryLocation::GpuOnly}
    rhi::MemoryLocation::Shared => {MemoryLocation::CpuToGpu}
  }
}
fn translate_image_format(format: rhi::ImageFormat) -> vk::Format {
  match format {
    rhi::ImageFormat::Texture => {vk::Format::R8G8B8A8_UNORM}
    rhi::ImageFormat::Float => {vk::Format::R32G32B32_SFLOAT}
    rhi::ImageFormat::Depth => {vk::Format::D24_UNORM_S8_UINT}
    rhi::ImageFormat::RenderIntermediate => {vk::Format::R8G8B8A8_UNORM}
    rhi::ImageFormat::Presentation => {vk::Format::B8G8R8A8_SRGB}
  }
}

fn translate_image_usage(usage: rhi::ImageUsage) -> vk::ImageUsageFlags {
  let mut flags = vk::ImageUsageFlags::empty();
  if usage.contains(rhi::ImageUsage::COPY_SRC){
    flags |= vk::ImageUsageFlags::TRANSFER_SRC;
  }
  if usage.contains(rhi::ImageUsage::COPY_DST) {
    flags |= vk::ImageUsageFlags::TRANSFER_DST;
  }
  if usage.contains(rhi::ImageUsage::SHADER_SAMPLED) {
    flags |= vk::ImageUsageFlags::SAMPLED;
  }
  if usage.contains(rhi::ImageUsage::SHADER_STORAGE) {
    flags |= vk::ImageUsageFlags::STORAGE;
  }
  flags
}

fn translate_buffer_usage(usage: rhi::BufferUsage) -> vk::BufferUsageFlags {
  let mut flags = vk::BufferUsageFlags::empty();
  if usage.contains(rhi::BufferUsage::COPY_SRC) {
    flags |= vk::BufferUsageFlags::TRANSFER_SRC;
  }
  if usage.contains(rhi::BufferUsage::COPY_DST) {
    flags |= vk::BufferUsageFlags::TRANSFER_DST;
  }
  if usage.contains(rhi::BufferUsage::STORAGE) {
    flags |= vk::BufferUsageFlags::STORAGE_BUFFER;
  }
  if usage.contains(rhi::BufferUsage::UNIFORM) {
    flags |= vk::BufferUsageFlags::UNIFORM_BUFFER;
  }
  flags
}

fn translate_descriptor_type(_type: rhi::DescriptorType) -> vk::DescriptorType {
  match _type {
    rhi::DescriptorType::Uniform => {
      vk::DescriptorType::UNIFORM_BUFFER
    }
    rhi::DescriptorType::Storage => {
      vk::DescriptorType::STORAGE_BUFFER
    }
    rhi::DescriptorType::Sampler2D => {
      vk::DescriptorType::COMBINED_IMAGE_SAMPLER
    }
  }
}

fn translate_shader_stage_flags(
  shader_stage_flags: rhi::ShaderStageFlags
) -> vk::ShaderStageFlags {
  let mut flags = vk::ShaderStageFlags::empty();
  if shader_stage_flags.contains(rhi::ShaderStageFlags::VERTEX) {
    flags |= vk::ShaderStageFlags::VERTEX;
  }
  if shader_stage_flags.contains(rhi::ShaderStageFlags::FRAGMENT) {
    flags |= vk::ShaderStageFlags::FRAGMENT;
  }
  flags
}

fn translate_raster_style(
  raster_style: rhi::RasterStyle
) -> vk::PipelineRasterizationStateCreateInfo{
  match raster_style {
    rhi::RasterStyle::Fill => {
      vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .cull_mode(vk::CullModeFlags::BACK)
        .line_width(1.0)
    }
    rhi::RasterStyle::WireFrame { thickness } => {
      vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .cull_mode(vk::CullModeFlags::BACK)
        .line_width(thickness as f32)
    }
  }
}

pub struct AllocatedBuffer{
  buffer: vk::Buffer,
  size: u64,
  allocation: Option<Allocation>,
}

pub struct AllocatedTexture{
  image: vk::Image,
  view: vk::ImageView,
  resolution: rhi::Resolution2D,
  format: rhi::ImageFormat,
  allocation: Option<Allocation>,
}

pub struct GraphicsPipeline{
  pipeline: vk::Pipeline,
  pipeline_layout: vk::PipelineLayout,
  render_pass: vk::RenderPass,
  buffer_set_layout: vk::DescriptorSetLayout,
  texture_set_layout: vk::DescriptorSetLayout,
}

pub struct BindlessDescriptorSets {
  buffer_set: vk::DescriptorSet,
  texture_set: vk::DescriptorSet,
}

pub struct VulkanBackendInitializer {
  vk_gpus: Vec<vk::PhysicalDevice>,
  ash_instance: ash::Instance,
  ash_entry: ash::Entry,
}

impl rhi::RenderBackendInitializer<VulkanBackend> for VulkanBackendInitializer {
  fn new() -> Result<Self, String> where Self: Sized {
    unsafe {
      let ash_entry = ash::Entry::load().map_err(|e| format!("at VK load: {e}"))?;
      let layers = [
        #[cfg(debug_assertions)]
        c"VK_LAYER_KHRONOS_validation".as_ptr(),
      ];
      let extensions = [
        #[cfg(debug_assertions)]
        ext::debug_utils::NAME.as_ptr(),
        khr::get_physical_device_properties2::NAME.as_ptr(),
        khr::surface::NAME.as_ptr(),
        #[cfg(target_os = "windows")]
        khr::win32_surface::NAME.as_ptr(),
        #[cfg(target_os = "linux")]
        khr::xlib_surface::NAME.as_ptr(),
        #[cfg(target_os = "linux")]
        khr::wayland_surface::NAME.as_ptr(),
        #[cfg(target_os = "macos")]
        khr::portability_enumeration::NAME.as_ptr(),
        #[cfg(target_os = "macos")]
        ext::metal_surface::NAME.as_ptr(),
        #[cfg(target_os = "android")]
        khr::android_surface::NAME.as_ptr(),
      ];
    
      let app_info = vk::ApplicationInfo::default()
        .application_name(c"Plind VK App")
        .application_version(0)
        .engine_name(c"Plind Engine")
        .engine_version(0)
        .api_version(vk::API_VERSION_1_0);
    
      #[cfg(target_os = "macos")]
      let vk_instance_create_info = vk::InstanceCreateInfo::default()
        .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR)
        .application_info(&app_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions);
    
      #[cfg(not(target_os = "macos"))]
      let vk_instance_create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions);
    
      let ash_instance = ash_entry
        .create_instance(&vk_instance_create_info, None)
        .map_err(|e| format!("at instance create: {e}"))?;

      let vk_gpus = ash_instance
        .enumerate_physical_devices()
        .map_err(|e| format!("at getting GPU list: {e}"))?;

      Ok(Self { vk_gpus, ash_instance, ash_entry })
    }
  }

  fn get_gpu_infos(&self) -> Vec<rhi::GPUInfo> {
    unsafe {
      self
        .vk_gpus
        .iter()
        .enumerate()
        .map(|(i, gpu)| {
          let props = self.ash_instance.get_physical_device_properties(*gpu);
          rhi::GPUInfo{
            id: i as _,
            name: props
              .device_name_as_c_str()
              .unwrap_or(c"Unknown Device")
              .to_string_lossy()
              .to_string(),
            integrated: props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU,
          }
        })
        .collect()
    }
  }

  fn init_backend(self, gpu_id: u32) -> Result<VulkanBackend, String> {
    let gpu = self.vk_gpus[gpu_id as usize];
    unsafe {
      let gpu_queue_family_props =
        self.ash_instance.get_physical_device_queue_family_properties(*gpu);
      let graphics_queue_family_id = gpu_queue_family_props
        .iter()
        .enumerate()
        .filter(|(_, x)| x.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .max_by_key(|(_, x)| x.queue_count)
        .map(|(x, _)| x as u32)
        .ok_or("no suitable GPU graphics queue found".to_string())?;
      let device_extensions = [
        khr::swapchain::NAME.as_ptr(),
        #[cfg(target_os = "macos")]
        khr::portability_subset::NAME.as_ptr(),
      ];
      let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&[
          vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics_queue_family_id)
            .queue_priorities(&[1.0])
        ])
        .enabled_extension_names(&device_extensions)
        .enabled_features(&vk::PhysicalDeviceFeatures::default());
      let ash_device = self
        .ash_instance
        .create_device(gpu, &device_create_info, None)
        .map_err(|e| format!("at vk device create: {e}"))?;
      let graphics_queue = ash_device.get_device_queue(graphics_queue_family_id, 0);
      let allocator = Allocator::new(
        &AllocatorCreateDesc {
          instance: self.ash_instance.clone(),
          device: ash_device.clone(),
          physical_device: gpu,
          debug_settings: Default::default(),
          buffer_device_address: false,
          allocation_sizes: Default::default(),
        }
      )
        .map_err(|e| format!("at allocator create: {e}"))?;

      let mut pool_create_info = vk::DescriptorPoolCreateInfo::default()
        .flags(
          vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND |
          vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET
        )
        .pool_sizes(
          &[
            vk::DescriptorPoolSize::default()
              .ty(vk::DescriptorType::STORAGE_BUFFER)
              .descriptor_count(512),
            vk::DescriptorPoolSize::default()
              .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
              .descriptor_count(8192),
          ]
        )
        .max_sets(512);
      let descriptor_pool = unsafe {
        ash_device
          .create_descriptor_pool(&pool_create_info, None)
          .map_err(|e| format!("at vk descriptor pool create: {e}"))?
      };

      Ok(VulkanBackend {
        descriptor_sets: SequentialIDStore::new(512),
        frame_buffers: SequentialIDStore::new(256),
        pipelines: SequentialIDStore::new(32),
        images: SequentialIDStore::new(1024),
        buffers: SequentialIDStore::new(1024),
        allocator,
        descriptor_pool,
        graphics_queue,
        graphics_queue_family_id,
        gpu,
        ash_device,
        ash_instance: self.ash_instance,
      })
    }
  }
}

pub struct VulkanBackend {
  descriptor_sets: SequentialIDStore<BindlessDescriptorSets>,
  frame_buffers: SequentialIDStore<vk::Framebuffer>,
  pipelines: SequentialIDStore<GraphicsPipeline>,
  images: SequentialIDStore<AllocatedTexture>,
  buffers: SequentialIDStore<AllocatedBuffer>,
  allocator: Allocator,
  descriptor_pool: vk::DescriptorPool,
  graphics_queue: vk::Queue,
  graphics_queue_family_id: u32,
  gpu: vk::PhysicalDevice,
  ash_device: ash::Device,
  ash_instance: ash::Instance,
}

impl VulkanBackend {
  fn destroy_image(&mut self, image_id: rhi::ImageID) -> Result<(), String> {
    let rhi::ImageID(image_id) = image_id;
    let a_image = self.images.remove_obj(image_id)?;
    unsafe {
      self.ash_device.destroy_image_view(a_image.view, None);
      self.ash_device.destroy_image(a_image.image, None);
      a_image.allocation.map(|a| self.allocator.free(a));
    }
    Ok(())
  }

  fn destroy_buffer(&mut self, buffer_id: rhi::BufferID) -> Result<(), String> {
    let rhi::BufferID(buffer_id) = buffer_id;
    let a_buffer = self.buffers.remove_obj(buffer_id)?;
    unsafe {
      self.ash_device.destroy_buffer(a_buffer.buffer, None);
      a_buffer.allocation.map(|a| self.allocator.free(a));
    }
    Ok(())
  }

  unsafe fn create_render_pass(
    &self,
    color_attachment_formats: &[rhi::ImageFormat],
    depth_attachment_formats: Option<&rhi::ImageFormat>,
  ) -> Result<vk::RenderPass, String> {
    let mut attachments = color_attachment_formats
      .into_iter()
      .map(|x| vk::AttachmentDescription::default()
        .format(translate_image_format(*x))
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .samples(vk::SampleCountFlags::TYPE_1)
      )
      .collect::<Vec<_>>();
    depth_attachment_formats.map(|x| attachments.push(
      vk::AttachmentDescription::default()
        .format(translate_image_format(*x))
        .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
        .stencil_store_op(vk::AttachmentStoreOp::STORE)
        .samples(vk::SampleCountFlags::TYPE_1)
    ));
    let mut subpass_desc = vk::SubpassDescription::default()
      .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
      .color_attachments(
        &(0..color_attachment_formats.len() as u32)
          .map(|i| vk::AttachmentReference::default()
            .attachment(i)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
          )
          .collect::<Vec<_>>()
      );
    let subpass_desc = match depth_attachment_formats {
      None => subpass_desc,
      Some(_) => subpass_desc.depth_stencil_attachment(
        &vk::AttachmentReference::default()
          .attachment(color_attachment_formats.len() as _)
          .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
      ),
    };
    let render_pass_create_info = vk::RenderPassCreateInfo::default()
      .attachments(&attachments)
      .subpasses(&[subpass_desc]);
    self
      .ash_device
      .create_render_pass(&render_pass_create_info, None)
      .map_err(|e| format!("at render pass creation: {e}"))
  }
}

impl rhi::RenderBackend for VulkanBackend {
  fn create_buffer(
    &mut self,
    size: u64,
    usage: rhi::BufferUsage,
    memory_location: rhi::MemoryLocation
  ) -> Result<rhi::BufferID, String> {
    unsafe {
      let buffer_create_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(translate_buffer_usage(usage))
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
      let buffer = self
        .ash_device
        .create_buffer(&buffer_create_info, None)
        .map_err(|e| format!("at vk buffer create: {e}"))?;
      let memory_requirements = self.ash_device.get_buffer_memory_requirements(buffer);
      let allocation = self
        .allocator
        .allocate(
          &AllocationCreateDesc{
            name: &format!("buffer_{buffer_id_u32}"),
            requirements: memory_requirements,
            location: translate_memory_location(memory_location),
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
          })
        .map_err(|e| format!("at allocator alloc: {e}"))?;
      let a_buffer = AllocatedBuffer{
        buffer,
        size,
        allocation: Some(allocation),
      };
      let buffer_id_u32 = self
        .buffers
        .add_obj(a_buffer)
        .map_err(|e| format!("max buffer count reached: {e}"))?;
      Ok(rhi::BufferID(buffer_id_u32))
    }
  }

  fn create_texture_2d(
    &mut self,
    res: rhi::Resolution2D,
    format: rhi::ImageFormat,
    usage: rhi::ImageUsage,
    memory_location: rhi::MemoryLocation
  ) -> Result<rhi::ImageID, String> {
    unsafe {
      let image_create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(translate_image_format(format))
        .usage(translate_image_usage(usage))
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .tiling(vk::ImageTiling::OPTIMAL)
        .extent(vk::Extent3D { width: res.width, height: res.height, depth: 1, });
      let image = self
        .ash_device
        .create_image(&image_create_info, None)
        .map_err(|e| format!("at vk image create: {e}"))?;
      let memory_requirements = self.ash_device.get_image_memory_requirements(image);
      let allocation = self
        .allocator
        .allocate(
          &AllocationCreateDesc{
            name: &format!("image_{image_id_u32}"),
            requirements: memory_requirements,
            location: translate_memory_location(memory_location),
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
          })
        .map_err(|e| format!("at allocator alloc: {e}"))?;
      let view = self
        .ash_device
        .create_image_view(
          &vk::ImageViewCreateInfo::default()
            .image(image)
            .format(translate_image_format(format))
            .view_type(vk::ImageViewType::TYPE_2D)
            .subresource_range(
              vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .layer_count(1)
                .base_mip_level(0)
                .level_count(1)
            ),
          None
        )
        .map_err(|e| format!("at view creation: {e}"))?;
      let a_image = AllocatedTexture{
        image,
        view,
        resolution: res,
        format,
        allocation: Some(allocation),
      };
      let image_id_u32 = self
        .images
        .add_obj(a_image)
        .map_err(|e| format!("max image count reached: {e}"))?;
      Ok(rhi::ImageID(image_id_u32))
    }
  }

  async fn create_graphics_pipeline(
    &mut self,
    raster_style: rhi::RasterStyle,
    color_attachment_formats: Vec<rhi::ImageFormat>,
    depth_attachment_formats: Option<rhi::ImageFormat>,
    max_buffer_count: u32,
    max_texture_count: u32,
    vertex_shader: PathBuf,
    fragment_shader: PathBuf
  ) -> Result<rhi::PipelineID, String> {
    unsafe {
      // Render pass
      let render_pass = self.create_render_pass(
        &color_attachment_formats,
        depth_attachment_formats.as_ref()
      )?;
      // Pipeline layout
      let buffer_set_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(&[
          vk::DescriptorSetLayoutBinding::default()
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(max_buffer_count)
            .stage_flags(vk::ShaderStageFlags::ALL)
        ]);
      let buffer_set_layout = self
        .ash_device
        .create_descriptor_set_layout(&buffer_set_layout_info, None)
        .map_err(|e| format!("at buffer set layout creation: {e}"))?;
      let texture_set_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(&[
          vk::DescriptorSetLayoutBinding::default()
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(max_texture_count)
            .stage_flags(vk::ShaderStageFlags::ALL)
        ]);
      let texture_set_layout = self
        .ash_device
        .create_descriptor_set_layout(&texture_set_layout_info, None)
        .map_err(|e| format!("at texture set layout creation: {e}"))?;
      let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&[buffer_set_layout, texture_set_layout]);
      let pipeline_layout = self
        .ash_device
        .create_pipeline_layout(&pipeline_layout_create_info, None)
        .map_err(|e| format!("at pipeline layout creation: {e}"))?;
      // Pipeline
      let mut vert_fr =
        fs::read(&vertex_shader).await.map_err(|e| format!("at read vertex shader file: {e}"))?;
      let vert_data = ash::util::read_spv(&mut vert_fr[..])
        .map_err(|e| format!("at read vertex shader: {e}"))?;
      let vert_shader_vk = self.ash_device.create_shader_module(
        &vk::ShaderModuleCreateInfo::default().code(&vert_data),
        None
      )
        .map_err(|e| format!("at vert shader module creation: {e}"))?;
      let mut frag_fr =
        fs::read(&fragment_shader).await.map_err(|e| format!("at read fragment shader file: {e}"))?;
      let frag_data = ash::util::read_spv(&mut frag_fr[..])
        .map_err(|e| format!("at read fragment shader: {e}"))?;
      let frag_shader_vk = self.ash_device.create_shader_module(
        &vk::ShaderModuleCreateInfo::default().code(&frag_data),
        None
      )
        .map_err(|e| format!("at frag shader module creation: {e}"))?;
      let vert_input_info = vk::PipelineVertexInputStateCreateInfo::default();
      let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
      let dynamic_states = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
      let msaa_info = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false);
      let vp_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
      let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
        .render_pass(render_pass)
        .subpass(0)
        .layout(pipeline_layout)
        .vertex_input_state(&vert_input_info)
        .input_assembly_state(&input_assembly_info)
        .dynamic_state(&dynamic_states)
        .multisample_state(&msaa_info)
        .viewport_state(&vp_state)
        .rasterization_state(&translate_raster_style(raster_style))
        .stages(&[
          vk::PipelineShaderStageCreateInfo::default()
            .name(c"main")
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_vk),
          vk::PipelineShaderStageCreateInfo::default()
            .name(c"main")
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_vk),
        ]);
      let pipeline = self
        .ash_device
        .create_graphics_pipelines(
          vk::PipelineCache::null(),
          &[pipeline_create_info],
          None
        )
        .map_err(|e| format!("at create pipeline: {}", *e.1))?
        .remove(0);
      let g_pipeline = GraphicsPipeline{
        pipeline,
        pipeline_layout,
        render_pass,
        buffer_set_layout,
        texture_set_layout,
      };
      let g_pipeline_id = self.pipelines.add_obj(g_pipeline)?;
      Ok(rhi::PipelineID(g_pipeline_id))
    }
  }

  fn create_frame_buffer(
    &mut self,
    pipeline_id: rhi::PipelineID,
    color_attachments: Vec<rhi::ImageID>,
    depth_attachment: Option<rhi::ImageID>
  ) -> Result<rhi::FramebufferID, String>{
    unsafe {
      let g_pipeline = self.pipelines.get_obj(pipeline_id.0)?;
      let mut attachments = color_attachments
        .iter()
        .map(|x| self.images.get_obj(x.0).map(|img| img.view))
        .collect::<Result<Vec<_>, &str>>()?;
      if let Some(depth_attach_id) = depth_attachment{
        attachments.push(self.images.get_obj(depth_attach_id.0).map(|img| img.view)?);
      }
      let res = self.images.get_obj(color_attachments[0].0)?.resolution;
      let fb_create_info = vk::FramebufferCreateInfo::default()
        .render_pass(g_pipeline.render_pass)
        .attachments(&attachments)
        .layers(1)
        .width(res.width)
        .height(res.height);
      let frame_buffer = self
        .ash_device
        .create_framebuffer(&fb_create_info, None)
        .map_err(|e| format!("at create framebuffer: {e}"))?;
      let fb_id_u32 = self.frame_buffers.add_obj(frame_buffer)?;
      Ok(rhi::FramebufferID(fb_id_u32))
    }
  }

  fn create_input_set(
    &mut self,
    pipeline_id: rhi::PipelineID,
  ) -> Result<rhi::InputSetID, String> {
    unsafe {
      let pipeline = self.pipelines.get_obj(pipeline_id.0)?;
      let buffer_set_layout = pipeline.buffer_set_layout;
      let texture_set_layout = pipeline.texture_set_layout;
      let desc_sets = self
        .ash_device
        .allocate_descriptor_sets(
          &vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&[buffer_set_layout, texture_set_layout]),
        )
        .map_err(|e| format!("at allocate buffer descriptor set: {e}"))?;
      let buffer_set = desc_sets[0];
      let texture_set = desc_sets[1];
      let b_descriptor_sets = BindlessDescriptorSets{ buffer_set, texture_set };
      let bds_id_u32 = self.descriptor_sets.add_obj(b_descriptor_sets)?;
      Ok(rhi::InputSetID(bds_id_u32))
    }
  }

  fn update_input_set(
    &mut self,
    input_set: rhi::InputSetID,
    buffers: Vec<rhi::BufferID>,
    textures: Vec<rhi::ImageID>
  ) -> Result<(), String> {
    unsafe {
      let b_desc_sets = self.descriptor_sets.get_obj(input_set.0)?;
      let buffer_write_info = vk::WriteDescriptorSet::default()
        .dst_set(b_desc_sets.buffer_set)
        .dst_binding(0)
        .descriptor_type(translate_descriptor_type(rhi::DescriptorType::Storage))
        .descriptor_count(1)
        .buffer_info(&buffers
          .into_iter()
          .map(|x| self.buffers.get_obj(x.0))
          .collect::<Result<Vec<_>, _>>()?
          .into_iter()
          .map(|x| vk::DescriptorBufferInfo::default()
            .buffer(x.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE))
          .collect::<Vec<_>>()
        );
      let texture_write_info = vk::WriteDescriptorSet::default()
        .dst_set(b_desc_sets.texture_set)
        .dst_binding(0)
        .descriptor_type(translate_descriptor_type(rhi::DescriptorType::Sampler2D))
        .descriptor_count(1)
        .image_info(&textures
          .into_iter()
          .map(|x| self.images.get_obj(x.0))
          .collect::<Result<Vec<_>, _>>()?
          .into_iter()
          .map(|x| vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(x.view))
          .collect::<Vec<_>>()
        );
      self.ash_device.update_descriptor_sets(
        &[buffer_write_info, texture_write_info],
        &[]
      );
    }
    Ok(())
  }
}