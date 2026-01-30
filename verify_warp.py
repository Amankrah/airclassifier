import warp as wp

# Initialize Warp
wp.init()

# Check available devices
print(f"CUDA available: {wp.is_cuda_available()}")
print(f"Devices: {wp.get_devices()}")

# Simple test kernel
@wp.kernel
def hello_kernel(data: wp.array(dtype=float)):
    tid = wp.tid()
    data[tid] = float(tid) * 2.0

# Run test
n = 10
arr = wp.zeros(n, dtype=float, device="cuda")
wp.launch(hello_kernel, dim=n, inputs=[arr])
print(arr.numpy())