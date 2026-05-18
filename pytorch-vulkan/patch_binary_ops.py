import re

with open('csrc/shim.cpp', 'r') as f:
    content = f.read()

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

with open('csrc/shim.cpp', 'w') as f:
    f.write(content)

print("Patched binary ops.")
