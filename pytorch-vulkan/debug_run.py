import torch
from pytorch_vulkan import init
init()
device = "vkgpu:0"

# test 1
print("Test 1: arange")
try:
    x = torch.arange(8, device=device)
    print("arange OK")
except Exception as e:
    print("arange FAIL:", e)

# test 2
print("Test 2: clone contiguous")
try:
    x2 = x.expand(2, 8)
    x3 = x2.contiguous()
    print("contiguous OK")
except Exception as e:
    print("contiguous FAIL:", e)

# test 3
print("Test 3: argmax")
try:
    y = torch.randn(1, 4, device=device)
    z = y.argmax(dim=-1)
    print("argmax OK")
except Exception as e:
    print("argmax FAIL:", e)

# test 4
print("Test 4: copy slice")
try:
    a = torch.randn(4, 4, device=device)
    b = torch.randn(1, 4, device=device)
    a[1:2, :] = b
    print("copy slice OK")
except Exception as e:
    print("copy slice FAIL:", e)

# test 5
print("Test 5: rmsnorm")
try:
    import torch.nn.functional as F
    w = torch.ones(4, device=device)
    r = torch.ops.aten._fused_rms_norm.default(a, (4,), w, 1e-6)
    print("rmsnorm OK")
except Exception as e:
    print("rmsnorm FAIL:", e)
