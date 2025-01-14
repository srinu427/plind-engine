use std::collections::HashMap;

pub struct SequentialIDStore<T>{
  max_id: u32,
  store: HashMap<u32, T>,
  freed: Vec<u32>,
}

impl<T> SequentialIDStore<T>{
  pub fn new(capacity: u32) -> Self {
    Self {
      max_id: 0,
      store: HashMap::with_capacity(capacity as _),
      freed: Vec::with_capacity(capacity as _)
    }
  }

  pub fn add_obj(&mut self, obj: T) -> Result<u32, &'static str>{
    match self.freed.pop() {
      Some(id) => {
        self.store.insert(id, obj);
        Ok(id)
      },
      None => {
        if self.max_id == u32::MAX {
          Err("max items reached")
        } else {
          self.max_id += 1;
          self.store.insert(self.max_id - 1, obj);
          Ok(self.max_id - 1)
        }
      },
    }
  }

  pub fn remove_obj(&mut self, id: u32) -> Result<T, &'static str>{
    let obj = self.store.remove(&id).ok_or("item not found")?;
    self.freed.push(id);
    Ok(obj)
  }

  pub fn get_obj(&self, id: u32) -> Result<&T, &'static str>{
    self.store.get(&id).ok_or("item not found")
  }

  pub fn get_obj_mut(&mut self, id: u32) -> Result<&mut T, &'static str>{
    self.store.get_mut(&id).ok_or("item not found")
  }
}