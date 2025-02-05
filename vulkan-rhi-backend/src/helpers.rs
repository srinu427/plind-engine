use ash::{ext, khr, vk};

pub unsafe fn create_vk_instance() -> Result<(ash::Entry, ash::Instance), String> {
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
  Ok((ash_entry, ash_instance))
}

pub unsafe fn make_swapchain(
  gpu: vk::PhysicalDevice,
  surface_instance: &khr::surface::Instance,
  surface: vk::SurfaceKHR,
  swapchain_device: &khr::swapchain::Device,
) -> Result<(vk::Extent2D, vk::SurfaceFormatKHR, u32, vk::SwapchainKHR), String> {
  let surface_caps = surface_instance
    .get_physical_device_surface_capabilities(gpu, surface)
    .map_err(|e| format!("at getting surface capabilities: {e}"))?;
  let swapchain_res = surface_caps.current_extent;
  let swapchain_img_count = std::cmp::min(
    surface_caps.min_image_count + 1,
    if surface_caps.max_image_count == 0 {
      surface_caps.min_image_count + 1
    } else {
      surface_caps.max_image_count
    },
  );
  let surface_formats = surface_instance
    .get_physical_device_surface_formats(gpu, surface)
    .map_err(|e| format!("at getting surface formats: {e}"))?;
  let surface_format = surface_formats
    .iter()
    .find(|f| f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
    .cloned()
    .unwrap_or(surface_formats[0]);
  let present_mode = surface_instance
    .get_physical_device_surface_present_modes(gpu, surface)
    .map_err(|e| format!("at getting surface present mode: {e}"))?;
  let swapchain = swapchain_device.create_swapchain(
    &vk::SwapchainCreateInfoKHR::default()
      .surface(surface)
      .min_image_count(swapchain_img_count)
      .image_extent(swapchain_res)
      .image_color_space(surface_format.color_space)
      .image_format(surface_format.format)
      .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
      .image_usage(
        vk::ImageUsageFlags::COLOR_ATTACHMENT |
          vk::ImageUsageFlags::TRANSFER_DST |
          vk::ImageUsageFlags::STORAGE
      )
      .image_array_layers(1),
    None
  )
    .map_err(|e| format!("at swapchain creation: {e}"))?;
  Ok((swapchain_res, surface_format, swapchain_img_count, swapchain))
}
