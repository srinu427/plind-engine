pub use rhi;
use ash::{vk, ext, khr};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc};
use seq_id_store::SequentialIDStore;

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

      Ok(VulkanBackend {
        descriptor_set_layouts: SequentialIDStore::new(1024),
        descriptor_pools: SequentialIDStore::new(1024),
        images: SequentialIDStore::new(1024),
        buffers: SequentialIDStore::new(1024),
        allocator,
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
  descriptor_set_layouts: SequentialIDStore<vk::DescriptorSetLayout>,
  descriptor_pools: SequentialIDStore<vk::DescriptorPool>,
  images: SequentialIDStore<(vk::Image, Option<Allocation>)>,
  buffers: SequentialIDStore<(vk::Buffer, Option<Allocation>)>,
  allocator: Allocator,
  graphics_queue: vk::Queue,
  graphics_queue_family_id: u32,
  gpu: vk::PhysicalDevice,
  ash_device: ash::Device,
  ash_instance: ash::Instance,
}

impl VulkanBackend {
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

  fn create_2d_image(
    &mut self,
    res: rhi::Resolution2D,
    format: rhi::ImageFormat,
    usage: rhi::ImageUsage,
    memory_location: rhi::MemoryLocation,
  ) -> Result<rhi::ImageID, String> {
    let image_create_info = vk::ImageCreateInfo::default()
      .image_type(vk::ImageType::TYPE_2D)
      .format(Self::translate_image_format(format))
      .usage(Self::translate_image_usage(usage))
      .sharing_mode(vk::SharingMode::EXCLUSIVE)
      .initial_layout(vk::ImageLayout::UNDEFINED)
      .tiling(vk::ImageTiling::OPTIMAL)
      .extent(vk::Extent3D { width: res.width, height: res.height, depth: 1, });
    let image = unsafe {
      self
        .ash_device
        .create_image(&image_create_info, None)
        .map_err(|e| format!("at vk image create: {e}"))?
    };
    let image_id_u32 = self
      .images
      .add_obj((image, None))
      .map_err(|e| format!("max image count reached: {e}"))?;
    let memory_requirements = unsafe {
      self.ash_device.get_image_memory_requirements(image)
    };
    let allocation = self
      .allocator
      .allocate(
        &AllocationCreateDesc{
          name: &format!("image_{image_id_u32}"),
          requirements: memory_requirements,
          location: Self::translate_memory_location(memory_location),
          linear: false,
          allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
      .map_err(|e| format!("at allocator alloc: {e}"))?;
     self
      .images
      .get_obj_mut(image_id_u32)
      .map_err(|e| format!("at image allocation insert: {e}"))?
      .1
      .replace(allocation);
    Ok(rhi::ImageID(image_id_u32))
  }

  fn destroy_image(&mut self, image_id: rhi::ImageID) -> Result<(), String> {
    let rhi::ImageID(image_id) = image_id;
    let (image, allocation) = self.images.remove_obj(image_id)?;
    unsafe {
      self.ash_device.destroy_image(image, None);
      allocation.map(|a| self.allocator.free(a));
    }
    Ok(())
  }

  fn create_buffer(
    &mut self,
    size: u64,
    usage: rhi::BufferUsage,
    memory_location: rhi::MemoryLocation,
  ) -> Result<rhi::BufferID, String> {
    let buffer_create_info = vk::BufferCreateInfo::default()
      .size(size)
      .usage(Self::translate_buffer_usage(usage))
      .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe {
      self
        .ash_device
        .create_buffer(&buffer_create_info, None)
        .map_err(|e| format!("at vk buffer create: {e}"))?
    };
    let buffer_id_u32 = self
      .buffers
      .add_obj((buffer, None))
      .map_err(|e| format!("max buffer count reached: {e}"))?;
    let memory_requirements = unsafe {
      self.ash_device.get_buffer_memory_requirements(buffer)
    };
    let allocation = self
      .allocator
      .allocate(
        &AllocationCreateDesc{
          name: &format!("buffer_{buffer_id_u32}"),
          requirements: memory_requirements,
          location: Self::translate_memory_location(memory_location),
          linear: false,
          allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
      .map_err(|e| format!("at allocator alloc: {e}"))?;
    self
      .buffers
      .get_obj_mut(buffer_id_u32)
      .map_err(|e| format!("at buffer allocation insert: {e}"))?
      .1
      .replace(allocation);
    Ok(rhi::BufferID(buffer_id_u32))
  }

  fn destroy_buffer(&mut self, buffer_id: rhi::BufferID) -> Result<(), String> {
    let rhi::BufferID(buffer_id) = buffer_id;
    let (buffer, allocation) = self.buffers.remove_obj(buffer_id)?;
    unsafe {
      self.ash_device.destroy_buffer(buffer, None);
      allocation.map(|a| self.allocator.free(a));
    }
    Ok(())
  }

  fn create_descriptor_pool(
    &mut self,
    free_able: bool,
    limits: Vec<(rhi::DescriptorType, u32)>
  ) -> Result<rhi::DescriptorPoolID, String> {
    let mut pool_create_info = vk::DescriptorPoolCreateInfo::default()
      .pool_sizes(
        &limits
          .iter()
          .map(|(ty, count)| vk::DescriptorPoolSize::default()
            .ty(Self::translate_descriptor_type(*ty))
            .descriptor_count(*count)
          )
          .collect::<Vec<_>>()
      )
      .max_sets(limits.iter().map(|x| x.1).sum::<u32>());
    if free_able {
      pool_create_info.flags |= vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET;
    }
    let descriptor_pool = unsafe {
      self
        .ash_device
        .create_descriptor_pool(&pool_create_info, None)
        .map_err(|e| format!("at vk descriptor pool create: {e}"))?
    };
    let descriptor_pool_id_u32 = self
      .descriptor_pools
      .add_obj(descriptor_pool)
      .map_err(|e| format!("max descriptor pool count reached: {e}"))?;
    Ok(rhi::DescriptorPoolID(descriptor_pool_id_u32))
  }

  fn destroy_descriptor_pool(
    &mut self,
    descriptor_pool_id: rhi::DescriptorPoolID
  ) -> Result<(), String> {
    let rhi::DescriptorPoolID(descriptor_pool) = descriptor_pool_id;
    let descriptor_pool = self.descriptor_pools.remove_obj(descriptor_pool)?;
    unsafe {
      self.ash_device.destroy_descriptor_pool(descriptor_pool, None);
    }
    Ok(())
  }

  fn create_descriptor_layout(
    &mut self,
    binding_types: Vec<(u32, rhi::DescriptorType, rhi::ShaderStageFlags)>
  ) -> Result<rhi::DescriptorLayoutID, String> {
    let vk_descriptor_bindings = binding_types
      .iter()
      .enumerate()
      .map(|(i, binding)| {
        vk::DescriptorSetLayoutBinding::default()
          .binding(i as _)
          .stage_flags(Self::translate_shader_stage_flags(binding.2))
          .descriptor_type(Self::translate_descriptor_type(binding.1))
          .descriptor_count(1)
      })
      .collect::<Vec<_>>();
    let descriptor_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
      .bindings(&vk_descriptor_bindings);
    let descriptor_layout = unsafe {
      self
        .ash_device
        .create_descriptor_set_layout(&descriptor_layout_create_info, None)
        .map_err(|e| format!("at vk descriptor set layout: {e}"))?
    };
    let descriptor_set_layout_id_u32 = self
      .descriptor_set_layouts
      .add_obj(descriptor_layout)
      .map_err(|e| format!("max descriptor set layout reached: {e}"))?;
    Ok(rhi::DescriptorLayoutID(descriptor_set_layout_id_u32))
  }

  fn destroy_descriptor_layout(
    &mut self,
    descriptor_layout_id: rhi::DescriptorLayoutID
  ) -> Result<(), String> {
    let rhi::DescriptorLayoutID(descriptor_layout_id) = descriptor_layout_id;
    let descriptor_layout = self
        .descriptor_set_layouts
        .remove_obj(descriptor_layout_id)
        .map_err(|e| format!("max descriptor set layout reached: {e}"))?;
    unsafe {
      self.ash_device.destroy_descriptor_set_layout(descriptor_layout, None);
    }
    Ok(())
  }
}

impl rhi::RenderBackend for VulkanBackend {
  fn run_task(&mut self, task: rhi::RenderBackendTask) -> rhi::RenderBackendTaskOutput {
    match task {
      rhi::RenderBackendTask::OrderedTasks(tasks) => {
        let mut outputs = Vec::new();
        for task in tasks {
          outputs.push(self.run_task(task));
        }
        return rhi::RenderBackendTaskOutput::OrderedTasksOutput(outputs);
      }
      rhi::RenderBackendTask::UnorderedTasks(tasks) => {
        let mut outputs = Vec::new();
        for task in tasks {
          outputs.push(self.run_task(task));
        }
        return rhi::RenderBackendTaskOutput::UnorderedTasksOutput(outputs);
      }
      rhi::RenderBackendTask::Create2DImage { res, format, usage, memory_location } => {
        return rhi::RenderBackendTaskOutput::Create2DImageOutput(
          self.create_2d_image(res, format, usage, memory_location)
        );
      }
      rhi::RenderBackendTask::DestroyImage { id } => {
        return rhi::RenderBackendTaskOutput::DestroyImageOutput(
          self.destroy_image(id)
        );
      }
      rhi::RenderBackendTask::CreateImageView { .. } => {}
      rhi::RenderBackendTask::DestroyImageView { .. } => {}
      rhi::RenderBackendTask::CreateBuffer { size, usage, memory_location } => {
        return rhi::RenderBackendTaskOutput::CreateBufferOutput(
          self.create_buffer(size, usage, memory_location)
        );
      }
      rhi::RenderBackendTask::DestroyBuffer { id } => {
        return rhi::RenderBackendTaskOutput::DestroyBufferOutput(
          self.destroy_buffer(id)
        );
      }
      rhi::RenderBackendTask::CreateDescriptorLayout { binding_types } => {
        return rhi::RenderBackendTaskOutput::CreateDescriptorLayoutOutput(
          self.create_descriptor_layout(binding_types)
        );
      }
      rhi::RenderBackendTask::DestroyDescriptorLayout { id } => {
        return rhi::RenderBackendTaskOutput::DestroyDescriptorLayoutOutput(
          self.destroy_descriptor_layout(id)
        );
      }
      rhi::RenderBackendTask::CreateDescriptorPool { free_able, limits } => {
        return rhi::RenderBackendTaskOutput::CreateDescriptorPoolOutput(
          self.create_descriptor_pool(free_able, limits)
        );
      }
      rhi::RenderBackendTask::DestroyDescriptorPool { id } => {
        return rhi::RenderBackendTaskOutput::DestroyDescriptorPoolOutput(
          self.destroy_descriptor_pool(id)
        );
      }
      rhi::RenderBackendTask::AllocateDescriptorSet { .. } => {}
      rhi::RenderBackendTask::UpdateDescriptorSetBufferBinding { .. } => {}
      rhi::RenderBackendTask::UpdateDescriptorSetImageBinding { .. } => {}
      rhi::RenderBackendTask::CreateGraphicsPipeline { .. } => {}
      rhi::RenderBackendTask::DestroyPipeline { .. } => {}
    }
    todo!()
  }

  fn destroy(&mut self) {
    unsafe {
      self.ash_device.destroy_device(None);
      self.ash_instance.destroy_instance(None);
    }
  }
}