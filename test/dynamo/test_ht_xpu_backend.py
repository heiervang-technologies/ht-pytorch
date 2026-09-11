import operator

import torch
from torch import fx
from torch.testing._internal.common_utils import run_tests, TestCase

from torch._dynamo.backends.ht_xpu import profile_graph


class TestHTXPUBackend(TestCase):
    def test_graph_profile(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        added = graph.call_function(operator.add, (x, y))
        graph.output(added)
        profile = profile_graph(fx.GraphModule({}, graph))
        self.assertEqual(profile.nodes, 4)
        self.assertEqual(profile.call_function, 1)
        self.assertEqual(profile.call_method, 0)
        self.assertEqual(profile.call_module, 0)
        self.assertEqual(sum(profile.operators.values()), 1)


if __name__ == "__main__":
    run_tests()
