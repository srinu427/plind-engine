mod helpers;

use std::collections::HashMap;
use std::path::PathBuf;
pub use rhi;
use ash::{vk, khr};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{
  Allocation,
  AllocationCreateDesc,
  AllocationScheme,
  Allocator,
  AllocatorCreateDesc
};
use rhi::{HasDisplayHandle, HasWindowHandle};
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

fn translate_raster_style<'a>(
  raster_style: rhi::RasterStyle
) -> vk::PipelineRasterizationStateCreateInfo<'a>{
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

fn get_aspect_mask(format: rhi::ImageFormat) -> vk::ImageAspectFlags {
  match format {
    rhi::ImageFormat::Texture => { vk::ImageAspectFlags::COLOR }
    rhi::ImageFormat::Float => { vk::ImageAspectFlags::COLOR }
    rhi::ImageFormat::Depth => { vk::ImageAspectFlags::DEPTH }
    rhi::ImageFormat::RenderIntermediate => { vk::ImageAspectFlags::COLOR }
    rhi::ImageFormat::Presentation => { vk::ImageAspectFlags::COLOR }
  }
}

fn infer_access_from_layout(layout: vk::ImageLayout) -> vk::AccessFlags{
  if layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL{
    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
  } else if layout == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL{
    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
  } else if layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL{
    vk::AccessFlags::SHADER_READ
  } else if layout == vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
    vk::AccessFlags::TRANSFER_READ
  } else if layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL {
    vk::AccessFlags::TRANSFER_WRITE
  } else if layout == vk::ImageLayout::UNDEFINED {
    vk::AccessFlags::NONE
  } else {
    vk::AccessFlags::NONE
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

pub struct InputSetVK {
  buffer_set: vk::DescriptorSet,
  texture_set: vk::DescriptorSet,
  bound_buffers: Vec<rhi::BufferID>,
  bound_textures: Vec<rhi::ImageID>,
}

pub struct FramebufferVK {
  framebuffer: vk::Framebuffer,
  color_attachments: Vec<rhi::ImageID>,
  depth_attachment: Option<rhi::ImageID>,
}

pub struct VulkanBackend {
  command_buffers: SequentialIDStore<vk::CommandBuffer>,
  command_pool: vk::CommandPool,
  fences: SequentialIDStore<vk::Fence>,
  descriptor_sets: SequentialIDStore<InputSetVK>,
  frame_buffers: SequentialIDStore<FramebufferVK>,
  pipelines: SequentialIDStore<GraphicsPipeline>,
  images: SequentialIDStore<AllocatedTexture>,
  buffers: SequentialIDStore<AllocatedBuffer>,
  allocator: Allocator,
  descriptor_pool: vk::DescriptorPool,
  graphics_queue: vk::Queue,
  graphics_queue_family_id: u32,
  gpu: vk::PhysicalDevice,
  swapchain_images: Vec<rhi::ImageID>,
  swapchain: vk::SwapchainKHR,
  swapchain_res: vk::Extent2D,
  surface_format: vk::SurfaceFormatKHR,
  swapchain_device: khr::swapchain::Device,
  surface: vk::SurfaceKHR,
  ash_device: ash::Device,
  surface_instance: khr::surface::Instance,
  ash_instance: ash::Instance,
  ash_entry: ash::Entry,
}

impl VulkanBackend {
  fn new(window: &(impl HasWindowHandle + HasDisplayHandle)) -> Result<Self, String> {
    unsafe {
      let (ash_entry, ash_instance) = helpers::create_vk_instance()?;
      let vk_gpus = ash_instance
        .enumerate_physical_devices()
        .map_err(|e| format!("at getting GPU list: {e}"))?;
      let gpu_info = vk_gpus
        .into_iter()
        .map(|gpu| {
          let props = ash_instance.get_physical_device_properties(gpu);
          (
            gpu,
            props
              .device_name_as_c_str()
              .unwrap_or(c"Unknown Device")
              .to_string_lossy()
              .to_string(),
            props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
          )
        })
        .collect::<Vec<_>>();
      if gpu_info.is_empty() {
        return Err(String::from("no GPU found"));
      }
      let gpu = gpu_info.iter().find(|info| {info.2}).map(|x| x.0).unwrap_or(gpu_info[0].0);

      let gpu_queue_family_props = ash_instance.get_physical_device_queue_family_properties(gpu);
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
      let ash_device = ash_instance
        .create_device(
          gpu,
          &vk::DeviceCreateInfo::default()
          .queue_create_infos(&[
            vk::DeviceQueueCreateInfo::default()
              .queue_family_index(graphics_queue_family_id)
              .queue_priorities(&[1.0])
          ])
          .enabled_extension_names(&device_extensions)
          .enabled_features(&vk::PhysicalDeviceFeatures::default()),
          None
        )
        .map_err(|e| format!("at vk device create: {e}"))?;
      let graphics_queue = ash_device.get_device_queue(graphics_queue_family_id, 0);
      let surface_instance = khr::surface::Instance::new(&ash_entry, &ash_instance);
      let surface = ash_window::create_surface(
        &ash_entry,
        &ash_instance,
        window.display_handle().map_err(|_| "invalid window".to_string())?.as_raw(),
        window.window_handle().map_err(|_| "invalid window".to_string())?.as_raw(),
        None
      )
        .map_err(|e| format!("at surface creation: {e}"))?;
      let swapchain_device = khr::swapchain::Device::new(&ash_instance, &ash_device);

      let allocator = Allocator::new(
        &AllocatorCreateDesc {
          instance: ash_instance.clone(),
          device: ash_device.clone(),
          physical_device: gpu,
          debug_settings: Default::default(),
          buffer_device_address: false,
          allocation_sizes: Default::default(),
        }
      )
        .map_err(|e| format!("at allocator create: {e}"))?;

      let descriptor_pool = ash_device
        .create_descriptor_pool(
          &vk::DescriptorPoolCreateInfo::default()
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
          .max_sets(512),
          None
        )
        .map_err(|e| format!("at vk descriptor pool create: {e}"))?;
      let command_pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(graphics_queue_family_id)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
      let command_pool = ash_device
        .create_command_pool(&command_pool_info, None)
        .map_err(|e| format!("at command pool create: {e}"))?;
      let (swapchain_res, surface_format, swapchain_image_count, swapchain) =
        helpers::make_swapchain(gpu, &surface_instance, surface, &swapchain_device)?;
      let swapchain_images_vk = swapchain_device
        .get_swapchain_images(swapchain)
        .map_err(|e| format!("at getting swapchain images: {e}"))?;
      let swapchain_image_views = swapchain_images_vk.iter().map(|x| {
        ash_device.create_image_view(
          &vk::ImageViewCreateInfo::default()
            .image(*x)
            .format(surface_format.format)
            .view_type(vk::ImageViewType::TYPE_2D)
            .subresource_range(
              vk::ImageSubresourceRange::default()
                .aspect_mask(get_aspect_mask(rhi::ImageFormat::Presentation))
                .base_array_layer(0)
                .layer_count(1)
                .base_mip_level(0)
                .level_count(1)
            ),
          None
        ).map_err(|e| format!("at swapchain image view: {e}"))
      }).collect::<Result<Vec<_>, String>>()?;
      let mut images = SequentialIDStore::new(1024);
      let swapchain_images = (0..swapchain_images_vk.len())
        .map(|i| {
          let a_image = AllocatedTexture{
            image: swapchain_images_vk[i],
            view: swapchain_image_views[i],
            resolution: rhi::Resolution2D { width: swapchain_res.width, height: swapchain_res.height },
            format: rhi::ImageFormat::Presentation,
            allocation: None
          };
          images.add_obj(a_image).map(rhi::ImageID)
        })
        .collect::<Result<Vec<_>, &str>>()?;

      Ok(Self {
        command_buffers: SequentialIDStore::new(256),
        command_pool,
        fences: SequentialIDStore::new(256),
        descriptor_sets: SequentialIDStore::new(512),
        frame_buffers: SequentialIDStore::new(256),
        pipelines: SequentialIDStore::new(32),
        images,
        buffers: SequentialIDStore::new(1024),
        allocator,
        descriptor_pool,
        graphics_queue,
        graphics_queue_family_id,
        gpu,
        swapchain_images,
        swapchain,
        swapchain_res,
        surface_format,
        swapchain_device,
        surface,
        ash_device,
        surface_instance,
        ash_instance,
        ash_entry,
      })
    }
  }

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
    let subpass_color_attach_infos = (0..color_attachment_formats.len() as u32)
      .map(|i| vk::AttachmentReference::default()
        .attachment(i)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
      )
      .collect::<Vec<_>>();
    let subpass_depth_attach_info = depth_attachment_formats
      .map(|_| vk::AttachmentReference::default()
        .attachment(color_attachment_formats.len() as _)
        .layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
      );
    let subpass_desc = vk::SubpassDescription::default()
      .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
      .color_attachments(&subpass_color_attach_infos);
    let subpass_desc = match subpass_depth_attach_info.as_ref() {
      None => subpass_desc,
      Some(x) => {
        subpass_desc.depth_stencil_attachment(x)},
    };
    let subpass_descs = [subpass_desc];
    let render_pass_create_info = vk::RenderPassCreateInfo::default()
      .attachments(&attachments)
      .subpasses(&subpass_descs);
    self
      .ash_device
      .create_render_pass(&render_pass_create_info, None)
      .map_err(|e| format!("at render pass creation: {e}"))
  }
}

impl rhi::RenderBackend for VulkanBackend {
  fn get_swapchain_info(&self) -> rhi::SwapchainInfo {
    rhi::SwapchainInfo{
      res: rhi::Resolution2D{width: self.swapchain_res.width, height: self.swapchain_res.height},
      image_count: self.swapchain_images.len() as _,
    }
  }

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
      let a_buffer = AllocatedBuffer{
        buffer,
        size,
        allocation: None,
      };
      let buffer_id_u32 = self
        .buffers
        .add_obj(a_buffer)
        .map_err(|e| format!("max buffer count reached: {e}"))?;
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
      self.buffers.get_obj_mut(buffer_id_u32)?.allocation = Some(allocation);
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
      let memory_requirements = self.ash_device.get_image_memory_requirements(image);
      let a_image = AllocatedTexture{
        image,
        view,
        resolution: res,
        format,
        allocation: None,
      };
      let image_id_u32 = self
        .images
        .add_obj(a_image)
        .map_err(|e| format!("max image count reached: {e}"))?;
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
      self.images.get_obj_mut(image_id_u32)?.allocation = Some(allocation);
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
      let buffer_dset_bindings = [
        vk::DescriptorSetLayoutBinding::default()
          .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
          .descriptor_count(max_buffer_count)
          .stage_flags(vk::ShaderStageFlags::ALL)
      ];
      let buffer_set_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(&buffer_dset_bindings);
      let buffer_set_layout = self
        .ash_device
        .create_descriptor_set_layout(&buffer_set_layout_info, None)
        .map_err(|e| format!("at buffer set layout creation: {e}"))?;
      let texture_dset_bindings = [
        vk::DescriptorSetLayoutBinding::default()
          .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
          .descriptor_count(max_texture_count)
          .stage_flags(vk::ShaderStageFlags::ALL)
      ];
      let texture_set_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(&texture_dset_bindings);
      let texture_set_layout = self
        .ash_device
        .create_descriptor_set_layout(&texture_set_layout_info, None)
        .map_err(|e| format!("at texture set layout creation: {e}"))?;
      let pipeline_set_layouts = [buffer_set_layout, texture_set_layout];
      let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&pipeline_set_layouts);
      let pipeline_layout = self
        .ash_device
        .create_pipeline_layout(&pipeline_layout_create_info, None)
        .map_err(|e| format!("at pipeline layout creation: {e}"))?;
      // Pipeline
      let vert_fr = fs::read(&vertex_shader)
        .await
        .map_err(|e| format!("at read vertex shader file: {e}"))?;
      let vert_data = ash::util::read_spv(&mut std::io::Cursor::new(&vert_fr))
        .map_err(|e| format!("at read vertex shader: {e}"))?;
      let vert_shader_vk = self.ash_device.create_shader_module(
        &vk::ShaderModuleCreateInfo::default().code(&vert_data),
        None
      )
        .map_err(|e| format!("at vert shader module creation: {e}"))?;
      let frag_fr =
        fs::read(&fragment_shader).await.map_err(|e| format!("at read fragment shader file: {e}"))?;
      let frag_data = ash::util::read_spv(&mut std::io::Cursor::new(&frag_fr))
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
      let raster_style_vk = translate_raster_style(raster_style);
      let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
          .name(c"main")
          .stage(vk::ShaderStageFlags::VERTEX)
          .module(vert_shader_vk),
        vk::PipelineShaderStageCreateInfo::default()
          .name(c"main")
          .stage(vk::ShaderStageFlags::FRAGMENT)
          .module(frag_shader_vk),
      ];
      let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
        .render_pass(render_pass)
        .subpass(0)
        .layout(pipeline_layout)
        .vertex_input_state(&vert_input_info)
        .input_assembly_state(&input_assembly_info)
        .dynamic_state(&dynamic_states)
        .multisample_state(&msaa_info)
        .viewport_state(&vp_state)
        .rasterization_state(&raster_style_vk)
        .stages(&shader_stages);
      let pipeline = self
        .ash_device
        .create_graphics_pipelines(
          vk::PipelineCache::null(),
          &[pipeline_create_info],
          None
        )
        .map_err(|e| format!("at create pipeline: {}", e.1))?
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
      let mut attachment_ids = color_attachments.clone();
      depth_attachment.map(|d| attachment_ids.push(d));
      let attachments = attachment_ids
        .iter()
        .map(|x| self.images.get_obj(x.0).map(|img| img.view))
        .collect::<Result<Vec<_>, &str>>()?;
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
        .map(|x| FramebufferVK{framebuffer: x, color_attachments, depth_attachment})
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
      let b_descriptor_sets = InputSetVK {
        buffer_set,
        texture_set,
        bound_buffers: vec![],
        bound_textures: vec![]
      };
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
      let buffer_infos = buffers
        .into_iter()
        .map(|x| self.buffers.get_obj(x.0))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|x| vk::DescriptorBufferInfo::default()
          .buffer(x.buffer)
          .offset(0)
          .range(vk::WHOLE_SIZE))
        .collect::<Vec<_>>();
      let buffer_write_info = vk::WriteDescriptorSet::default()
        .dst_set(b_desc_sets.buffer_set)
        .dst_binding(0)
        .descriptor_type(translate_descriptor_type(rhi::DescriptorType::Storage))
        .descriptor_count(1)
        .buffer_info(&buffer_infos);
      let image_infos = textures
        .into_iter()
        .map(|x| self.images.get_obj(x.0))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|x| vk::DescriptorImageInfo::default()
          .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
          .image_view(x.view))
        .collect::<Vec<_>>();
      let texture_write_info = vk::WriteDescriptorSet::default()
        .dst_set(b_desc_sets.texture_set)
        .dst_binding(0)
        .descriptor_type(translate_descriptor_type(rhi::DescriptorType::Sampler2D))
        .descriptor_count(1)
        .image_info(&image_infos);
      self.ash_device.update_descriptor_sets(
        &[buffer_write_info, texture_write_info],
        &[]
      );
    }
    Ok(())
  }

  fn create_fence(&mut self, signaled: bool) -> Result<rhi::FenceID, String> {
    unsafe {
      let fence_create_flags = if signaled {
        vk::FenceCreateFlags::SIGNALED
      } else {
        vk::FenceCreateFlags::empty()
      };
      let fence_vk = self
        .ash_device
        .create_fence(&vk::FenceCreateInfo::default().flags(fence_create_flags), None)
        .map_err(|e| format!("at create fence: {e}"))?;
      let fence_id_u32 = self.fences.add_obj(fence_vk)?;
      Ok(rhi::FenceID(fence_id_u32))
    }
  }

  async fn wait_for_fence(&self, fence_id: rhi::FenceID) -> Result<(), String> {
    unsafe {
      let fence = self.fences.get_obj(fence_id.0)?;
      self
        .ash_device
        .wait_for_fences(&[*fence], true, u64::MAX)
        .map_err(|e| format!("at wait_for_fence: {e}"))
    }
  }

  fn create_command_buffer(&mut self) -> Result<rhi::CommandBufferID, String> {
    unsafe {
      let command_buffer = self
        .ash_device
        .allocate_command_buffers(
          &vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1)
        )
        .map_err(|e| format!("at allocate command buffers: {e}"))?
        .remove(0);
      let cmd_buffer_id_u32 = self.command_buffers.add_obj(command_buffer)?;
      Ok(rhi::CommandBufferID(cmd_buffer_id_u32))
    }
  }

  fn compile_commands(&self, command_buffer: rhi::CommandBufferID, commands: Vec<rhi::GPUCommands>) -> Result<(), String> {
    // Figure out image layout transitions
    let mut image_needed_state = HashMap::new();
    for (i, command) in commands.iter().enumerate() {
      match command {
        rhi::GPUCommands::CopyBufferToBuffer { .. } => {}
        rhi::GPUCommands::CopyBufferToImage { src, dst } => {
          image_needed_state
            .entry(*dst)
            .or_insert(HashMap::new())
            .insert(i, (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::PipelineStageFlags::TRANSFER));
        }
        rhi::GPUCommands::BlitImage { src, dst } => {
          image_needed_state
            .entry(*src)
            .or_insert(HashMap::new())
            .insert(i, (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::PipelineStageFlags::TRANSFER));
          image_needed_state
            .entry(*dst)
            .or_insert(HashMap::new())
            .insert(i, (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::PipelineStageFlags::TRANSFER));
        }
        rhi::GPUCommands::RunGraphicsPipeline { pipeline, framebuffer, input_set, draw_infos } => {
          let frame_buffer_vk = self.frame_buffers.get_obj(framebuffer.0)?;
          for att_id in frame_buffer_vk.color_attachments.iter() {
            image_needed_state
              .entry(*att_id)
              .or_insert(HashMap::new())
              .insert(i, (
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
              ));
          }
          if let Some(att_id) = frame_buffer_vk.depth_attachment.as_ref() {
            image_needed_state
              .entry(*att_id)
              .or_insert(HashMap::new())
              .insert(i, (
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
              ));
          }
          let input_set_vk = self.descriptor_sets.get_obj(input_set.0)?;
          for tex_id in input_set_vk.bound_textures.iter() {
            image_needed_state
              .entry(*tex_id)
              .or_insert(HashMap::new())
              .insert(i, (
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags::FRAGMENT_SHADER
              ));
          }
        }
      }
    };
    // Fill command buffer
    let command_buffer_vk = self.command_buffers.get_obj(command_buffer.0)?.clone();
    unsafe {
      self
        .ash_device
        .begin_command_buffer(command_buffer_vk, &vk::CommandBufferBeginInfo::default())
        .map_err(|e| format!("at begin_command_buffer: {e}"))?;
      for (i, command) in commands.iter().enumerate() {
        match command {
          rhi::GPUCommands::CopyBufferToBuffer { src, dst } => {
            let src_buffer_vk = self.buffers.get_obj(src.0)?;
            let dst_buffer_vk = self.buffers.get_obj(dst.0)?;
            self.ash_device.cmd_copy_buffer(
              command_buffer_vk,
              src_buffer_vk.buffer,
              dst_buffer_vk.buffer,
              &[vk::BufferCopy::default().src_offset(0).dst_offset(0).size(src_buffer_vk.size)]
            );
          }
          rhi::GPUCommands::CopyBufferToImage { .. } => {}
          rhi::GPUCommands::BlitImage { .. } => {}
          rhi::GPUCommands::RunGraphicsPipeline { .. } => {}
        }
        for (img, states) in image_needed_state.iter() {
          let img_vk = self.images.get_obj(img.0)?;
          let Some(curr_state) = states.get(&i).cloned() else { continue };
          let prev_state = if i > 0{
             states
              .get(&(i - 1))
              .cloned()
              .unwrap_or((curr_state.0, vk::PipelineStageFlags::BOTTOM_OF_PIPE))
          } else {
            (curr_state.0, vk::PipelineStageFlags::BOTTOM_OF_PIPE)
          };
          self.ash_device.cmd_pipeline_barrier(
            command_buffer_vk,
            prev_state.1,
            curr_state.1,
            vk::DependencyFlags::BY_REGION,
            &[],
            &[],
            &[
              vk::ImageMemoryBarrier::default()
                .image(img_vk.image)
                .old_layout(prev_state.0)
                .new_layout(curr_state.0)
                .src_access_mask(infer_access_from_layout(prev_state.0))
                .dst_access_mask(infer_access_from_layout(curr_state.0))
                .src_queue_family_index(self.graphics_queue_family_id)
                .dst_queue_family_index(self.graphics_queue_family_id)
                .subresource_range(
                  vk::ImageSubresourceRange::default()
                    .aspect_mask(get_aspect_mask(img_vk.format))
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                )
            ]
          );
        }
      };
      self
        .ash_device
        .end_command_buffer(command_buffer_vk)
        .map_err(|e| format!("at end_command_buffer: {e}"))?;
    }
    Ok(())
  }

  fn run_commands(
    &self,
    command_buffer: rhi::CommandBufferID,
    fence_id: rhi::FenceID
  ) -> Result<(), String> {
    let command_buffer_vk = self.command_buffers.get_obj(command_buffer.0)?.clone();
    let fence_vk = self.fences.get_obj(fence_id.0)?;
    unsafe {
      self
        .ash_device
        .queue_submit(
          self.graphics_queue,
          &[vk::SubmitInfo::default().command_buffers(&[command_buffer_vk])],
          *fence_vk
        )
        .map_err(|e| format!("at submit queue submit: {e}"))
    }
  }
  
  fn get_swapchain_images(&self) -> Vec<rhi::ImageID>{
    self.swapchain_images.clone()
  }
  
  fn present_swapchain_image(&self, id: u32) -> Result<bool,String> {
    unsafe {
      self
        .swapchain_device
        .queue_present(
          self.graphics_queue,
          &vk::PresentInfoKHR::default().image_indices(&[id]).swapchains(&[self.swapchain])
        )
        .map_err(|e| format!("at presenting: {e}"))
    }
  }
  
  fn acquire_present_image(&self, fence_id: rhi::FenceID) -> Result<u32,String>{
    unsafe {
      self
        .swapchain_device
        .acquire_next_image(
          self.swapchain,
          999999,
          vk::Semaphore::null(),
          self.fences.get_obj(fence_id.0)?.clone()
        )
        .map(|x| x.0)
        .map_err(|e| format!("at acquiring present image: {e}"))
    }
  }
}

impl Drop for VulkanBackend{
  fn drop(&mut self){
    unsafe {
      let image_ids = self.images.get_all().keys().cloned().collect::<Vec<_>>();
      for image_id in image_ids {
        self.destroy_image(rhi::ImageID(image_id));
      }
      let buffer_ids = self.buffers.get_all().keys().cloned().collect::<Vec<_>>();
      for buffer_id in buffer_ids {
        self.destroy_buffer(rhi::BufferID(buffer_id));
      }
      self.surface_instance.destroy_surface(self.surface, None);
      self.ash_device.destroy_device(None);
      self.ash_instance.destroy_instance(None);
    }
  }
}
