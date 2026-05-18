import pytest


@pytest.fixture(autouse=True)
def flush_vulkan_queue(request):
    """Flush the Vulkan async command queue before and after every test.

    This ensures descriptor pool state from prior tests doesn't leak,
    and that all GPU work completes before the next test starts.
    """
    try:
        from pytorch_vulkan import _C
        _C.flush()
    except (ImportError, AttributeError):
        pass
    yield
    try:
        from pytorch_vulkan import _C
        _C.flush()
    except (ImportError, AttributeError):
        pass
    # Reset dynamo cache to prevent compiled function state from leaking
    # between tests (especially torch.compile backends).
    try:
        import torch._dynamo
        torch._dynamo.reset()
    except Exception:
        pass
