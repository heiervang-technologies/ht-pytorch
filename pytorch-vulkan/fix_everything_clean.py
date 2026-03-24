import re

with open('csrc/shim.cpp', 'r') as f:
    content = f.read()

# 1. Fix copy_vulkan_ fallback bug (remove memcpy on non-contiguous)
copy_old = """  if (!self.is_contiguous() || !src.is_contiguous() || self.scalar_type() != src.scalar_type()) {
      vkc_flush();
      auto self_cpu = at::empty(self.sizes(), self.options().device(c10::kCPU));
      auto src_cpu = at::from_blob(const_cast<void*>(src.data_ptr()), src.sizes(), src.strides(), at::TensorOptions().dtype(src.scalar_type()).device(c10::kCPU));
      self_cpu.copy_(src_cpu, non_blocking);
      std::memcpy(self.data_ptr(), self_cpu.data_ptr(), self.nbytes());
      return self;
  }"""

# We just remove this top fallback because the bottom fallback (Mixed device fallback)
# handles it perfectly with `from_blob` and `copy_`.
content = content.replace(copy_old, "")

# 2. Fix copy_vulkan_ fast path to check offset and restrict shader types
fast_old = """  // Fast path: contiguous same-dtype where at least one side is Vulkan.
  if (self.is_contiguous() && src.is_contiguous() && self.dtype() == src.dtype()
      && self.nbytes() == src.nbytes()) {"""

fast_new = """  // Fast path: contiguous same-dtype where at least one side is Vulkan.
  if (self.is_contiguous() && src.is_contiguous() && self.dtype() == src.dtype()
      && self.nbytes() == src.nbytes() && self.storage_offset() == 0 && src.storage_offset() == 0) {"""

content = content.replace(fast_old, fast_new)

async_old = """  // Both on Vulkan: try async GPU copy shader (supports dtype cast and strided)
  if (self_vk && src_vk) {
      std::string shader_name = "copy";
      if (self.scalar_type() == c10::kHalf && src.scalar_type() == c10::kHalf) {
          shader_name = "copy_f16";
      } else if (self.scalar_type() == c10::kBFloat16 && src.scalar_type() == c10::kBFloat16) {
          shader_name = "copy_bf16";
      } else if (self.scalar_type() == c10::kFloat && src.scalar_type() == c10::kHalf) {
          shader_name = "copy_f16_to_f32";
      } else if (self.scalar_type() == c10::kHalf && src.scalar_type() == c10::kFloat) {
          shader_name = "copy_f32_to_f16";
      }

      auto it = g_shader_handles.find(shader_name);
      if (it != g_shader_handles.end() && self.dim() <= 4) {"""

async_new = """  // Both on Vulkan: try async GPU copy shader (supports dtype cast and strided)
  if (self_vk && src_vk) {
      std::string shader_name = "";
      if (self.scalar_type() == c10::kFloat && src.scalar_type() == c10::kFloat) shader_name = "copy";
      else if (self.scalar_type() == c10::kInt && src.scalar_type() == c10::kInt) shader_name = "copy";
      else if (self.scalar_type() == c10::kHalf && src.scalar_type() == c10::kHalf) shader_name = "copy_f16";
      else if (self.scalar_type() == c10::kBFloat16 && src.scalar_type() == c10::kBFloat16) shader_name = "copy_bf16";
      else if (self.scalar_type() == c10::kFloat && src.scalar_type() == c10::kHalf) shader_name = "copy_f16_to_f32";
      else if (self.scalar_type() == c10::kHalf && src.scalar_type() == c10::kFloat) shader_name = "copy_f32_to_f16";

      auto it = g_shader_handles.find(shader_name);
      if (!shader_name.empty() && it != g_shader_handles.end() && self.dim() <= 4) {"""

content = content.replace(async_old, async_new)

# 3. Fix binary ops contiguous clones
pattern = re.compile(r"""  auto \[a, b\] = needs_broadcast\(self, other\)
      \? broadcast_operands\(self, other\)
      : std::make_pair\(self\.contiguous\(\), other\.contiguous\(\)\);

  uint32_t num_elements = a\.numel\(\);""")

replacement = """  auto [a, b] = needs_broadcast(self, other)
      ? broadcast_operands(self, other)
      : std::make_pair(self.contiguous(), other.contiguous());

  if (!a.is_contiguous() || a.storage_offset() > 0) a = a.clone(at::MemoryFormat::Contiguous);
  if (!b.is_contiguous() || b.storage_offset() > 0) b = b.clone(at::MemoryFormat::Contiguous);

  uint32_t num_elements = a.numel();"""

content = pattern.sub(replacement, content)

# 4. Fix rmsnorm offset bug
rms_old = """  auto work = input.contiguous();
  int64_t dim_size = 1;"""

rms_new = """  at::Tensor work = input;
  if (!work.is_contiguous() || work.storage_offset() > 0) {
      work = work.clone(at::MemoryFormat::Contiguous);
  }
  int64_t dim_size = 1;"""

content = content.replace(rms_old, rms_new)

# 5. Fix layernorm offset bug
ln_old = """  auto work = input.contiguous();
  int64_t ndim = work.dim();"""

ln_new = """  at::Tensor work = input;
  if (!work.is_contiguous() || work.storage_offset() > 0) {
      work = work.clone(at::MemoryFormat::Contiguous);
  }
  int64_t ndim = work.dim();"""

content = content.replace(ln_old, ln_new)

with open('csrc/shim.cpp', 'w') as f:
    f.write(content)

print("Applied fix_everything_clean.")
