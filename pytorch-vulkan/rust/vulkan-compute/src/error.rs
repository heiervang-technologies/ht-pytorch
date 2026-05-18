use thiserror::Error;

#[derive(Error, Debug)]
pub enum VkcError {
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] ash::vk::Result),

    #[error("No suitable GPU found")]
    NoGpu,

    #[error("No compute queue family found")]
    NoComputeQueue,

    #[error("Allocation failed: {0}")]
    Allocation(String),

    #[error("Shader compilation failed: {0}")]
    Shader(String),

    #[error("Device not initialized")]
    NotInitialized,

    #[error("Invalid SPIR-V: length {0} is not a multiple of 4")]
    InvalidSpirv(usize),
}
