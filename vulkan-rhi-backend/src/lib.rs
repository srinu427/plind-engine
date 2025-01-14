use std::collections::HashSet;

pub use rhi;
use ash::{vk, ext, khr};

pub struct VulkanBackend {
  vk_gpus: Vec<vk::PhysicalDevice>,
  ash_instance: ash::Instance,
  ash_entry: ash::Entry,
}

impl rhi::RenderBackend for VulkanBackend {
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
        .enabled_layer_names(&all_layers)
        .enabled_extension_names(&all_extensions);
    
      let ash_instance = ash_entry
        .create_instance(&vk_instance_create_info, None)
        .map_err(|e| format!("at instance create: {e}"))?;

      let vk_gpus = ash_instance
        .enumerate_physical_devices()
        .map_err(|e| format!("at geting GPU list: {e}"))?;

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

  fn init(gpu_id: u32) -> Self {
    todo!()
  }

  fn run_task(&mut self, task: rhi::RenderBackendTask) -> rhi::RenderBackendTaskOutput {
    todo!()
  }

  fn destroy(&mut self) {
    todo!()
  }
}