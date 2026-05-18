import logging
logging.basicConfig(level=logging.INFO)
import torch
import torch.nn as nn
import pytorch_vulkan
pytorch_vulkan.init()
from pytorch_vulkan import register_training, get_backward_op_report

register_training()

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleMLP().to("vkgpu:0")
opt = torch.optim.SGD(model.parameters(), lr=0.01)

def train_step(model, x, target):
    out = model(x)
    loss = (out - target).pow(2).mean()
    return loss

compiled_train_step = torch.compile(train_step, backend="vulkan_train")

# Single training step.
x = torch.randn(8, 32).to("vkgpu:0")
target = torch.randn(8, 10).to("vkgpu:0")
x.requires_grad = True

loss = compiled_train_step(model, x, target)
loss.backward()
opt.step()
opt.zero_grad()

report = get_backward_op_report()
print("\n" + report)