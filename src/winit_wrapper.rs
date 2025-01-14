use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes, WindowId};

static WINDOW_ICON_BYTES: &[u8] = include_bytes!("../assets/icon.ico");

pub struct AppActivity {
  window: Option<Window>,
}

impl AppActivity {
  pub fn new() -> Result<Self, String> {
    Ok(Self {
      window: None,
    })
  }
}

impl ApplicationHandler for AppActivity {
  fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    if self.window.is_none() {
      let Ok(w) = event_loop
        .create_window(
          WindowAttributes::default()
            //.with_taskbar_icon(icon.clone())
            //.with_window_icon(icon)
            .with_title("Plint Engine"),
        )
        .inspect_err(|e| eprintln!("error creating window: {e}"))
      else {
        event_loop.exit();
        return;
      };
    }
  }

  fn window_event(
    &mut self,
    event_loop: &ActiveEventLoop,
    _window_id: WindowId,
    event: WindowEvent,
  ) {
    // println!("event: {event:?}");
    match event {
      WindowEvent::ActivationTokenDone { .. } => {}
      WindowEvent::Resized(_) => {}
      WindowEvent::Moved(_) => {}
      WindowEvent::CloseRequested => {
        // #[cfg(target_os = "macos")]
        // let _ = self.window.take();
        event_loop.exit();
      }
      WindowEvent::Destroyed => {}
      WindowEvent::DroppedFile(_) => {}
      WindowEvent::HoveredFile(_) => {}
      WindowEvent::HoveredFileCancelled => {}
      WindowEvent::Focused(_) => {}
      WindowEvent::KeyboardInput { device_id, event, is_synthetic } => match event.state {
        winit::event::ElementState::Pressed => {
          //self.input_aggregator.update_key_pressed(event.key_without_modifiers());
        }
        winit::event::ElementState::Released => {
          //self.input_aggregator.update_key_released(event.key_without_modifiers());
        }
      },
      WindowEvent::ModifiersChanged(_) => {}
      WindowEvent::Ime(_) => {}
      WindowEvent::CursorMoved { .. } => {}
      WindowEvent::CursorEntered { .. } => {}
      WindowEvent::CursorLeft { .. } => {}
      WindowEvent::MouseWheel { .. } => {}
      WindowEvent::MouseInput { .. } => {}
      WindowEvent::PinchGesture { .. } => {}
      WindowEvent::PanGesture { .. } => {}
      WindowEvent::DoubleTapGesture { .. } => {}
      WindowEvent::RotationGesture { .. } => {}
      WindowEvent::TouchpadPressure { .. } => {}
      WindowEvent::AxisMotion { .. } => {}
      WindowEvent::Touch(_) => {}
      WindowEvent::ScaleFactorChanged { .. } => {}
      WindowEvent::ThemeChanged(_) => {}
      WindowEvent::Occluded(_) => {}
      WindowEvent::RedrawRequested => {}
    }
  }

  fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {}
}
