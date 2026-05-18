import torch
from pytorch_vulkan import init
init()
device = "vkgpu:0"

x = torch.arange(8).view(1, 8, 1)
x_vk = x.to(device)

y_vk = x_vk.expand(2, 8, 4)
z2_vk = y_vk.clone(memory_format=torch.contiguous_format)
print("Vulkan clone(contiguous):")
print(z2_vk.cpu().flatten()[:16])

