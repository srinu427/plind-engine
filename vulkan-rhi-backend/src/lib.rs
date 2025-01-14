pub use rhi;
use ash::{vk, ext, khr};
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
    }
    Ok(VulkanBackend {
      images: SequentialIDStore::new(1024),
      buffers: SequentialIDStore::new(1024),
      graphics_queue,
      graphics_queue_family_id,
      gpu,
      ash_device,
      ash_instance: self.ash_instance,
    })
  }
}

pub struct VulkanBackend {
  images: SequentialIDStore<vk::Image>,
  buffers: SequentialIDStore<vk::Buffer>,
  graphics_queue: vk::Queue,
  graphics_queue_family_id: u32,
  gpu: vk::PhysicalDevice,
  ash_device: ash::Device,
  ash_instance: ash::Instance,
}

impl VulkanBackend {
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

  fn create_2d_image(
    &mut self,
    res: rhi::Resolution2D,
    format: rhi::ImageFormat,
    usage: rhi::ImageUsage
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
      .add_obj(image)
      .map_err(|e| format!("max image count reached: {e}"))?;
    Ok(rhi::ImageID(image_id_u32))
  }

  fn destroy_image(&mut self, image_id: rhi::ImageID) -> Result<(), String> {
    let rhi::ImageID(image_id) = image_id;
    let image = self.images.remove_obj(image_id)?;
    unsafe {
      self.ash_device.destroy_image(image, None);
    }
    Ok(())
  }

  fn create_buffer(
    &mut self,
    size: u64,
    usage: rhi::BufferUsage
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
      .add_obj(buffer)
      .map_err(|e| format!("max buffer count reached: {e}"))?;
    Ok(rhi::BufferID(buffer_id_u32))
  }

  fn destroy_buffer(&mut self, buffer_id: rhi::BufferID) -> Result<(), String> {
    let rhi::BufferID(buffer_id) = buffer_id;
    let buffer = self.buffers.remove_obj(buffer_id)?;
    unsafe {
      self.ash_device.destroy_buffer(buffer, None);
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
      rhi::RenderBackendTask::Create2DImage { res, format, usage } => {
        return rhi::RenderBackendTaskOutput::Create2DImageOutput(
          self.create_2d_image(res, format, usage)
        );
      }
      rhi::RenderBackendTask::DestroyImage { id } => {
        return rhi::RenderBackendTaskOutput::DestroyImageOutput(
          self.destroy_image(id)
        );
      }
      rhi::RenderBackendTask::CreateImageView { .. } => {}
      rhi::RenderBackendTask::DestroyImageView { .. } => {}
      rhi::RenderBackendTask::CreateBuffer { size, usage } => {
        return rhi::RenderBackendTaskOutput::CreateBufferOutput(
          self.create_buffer(size, usage)
        );
      }
      rhi::RenderBackendTask::DestroyBuffer { id } => {
        return rhi::RenderBackendTaskOutput::DestroyBufferOutput(
          self.destroy_buffer(id)
        );
      }
      rhi::RenderBackendTask::CreateDescriptorLayout { .. } => {}
      rhi::RenderBackendTask::DestroyDescriptorLayout { .. } => {}
      rhi::RenderBackendTask::CreateDescriptorPool { .. } => {}
      rhi::RenderBackendTask::DestroyDescriptorPool { .. } => {}
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