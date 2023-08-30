import time

import tensorrt as trt
# import pycuda.autoinit
# import pycuda.driver as cuda
import torch
import numpy as np
from cuda import cuda, cudart
from common import allocate_buffers, memcopy_device_to_device

class Engine(object):
    def __init__(self, engine_path):
        device = torch.device("cuda")
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, '')
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.imgsz = model.get_binding_shape(0)[2:]
        self.dtype = trt.nptype(model.get_binding_dtype(0))
        self.context = model.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(model)
        # self.stream = cuda.Stream()
        # for binding in model:
        #     size = trt.volume(model.get_binding_shape(binding))
        #     dtype = trt.nptype(model.get_binding_dtype(binding))
        #     # np.dtype.itemsize
        #     # host_mem = cuda.pagelocked_empty(size, dtype)
        #     mem_size = size * np.dtype(dtype).itemsize
        #     device_mem = cuda.mem_alloc(mem_size)
        #     self.bindings.append(int(device_mem))
        #     if model.binding_is_input(binding):
        #         self.inputs.append({'device': int(device_mem), 'size': int(mem_size)})#{'host': host_mem, 'device': device_mem})
        #     else:
        #         self.outputs.append({'device': int(device_mem), 'size': int(mem_size)})#{'host': host_mem, 'device': device_mem})
        self.out_tensor = torch.zeros([1, 257, 1408], dtype=torch.float16, device=device)
    def forward(self, img:torch.Tensor): #np.array

        self.bindings[0] = int(img.data_ptr())
        # self.inputs[0]['host'] = np.ravel(img)
        # cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream)
        # fetch outputs from gpu
        cuda.cuStreamSynchronize(self.stream)
        # self.stream.synchronize()
        memcopy_device_to_device(self.out_tensor.data_ptr(), self.outputs[0].device, self.outputs[0].nbytes)
        # cuda.memcpy_dtod_async(self.out_tensor.data_ptr(), self.outputs[0]['device'], self.outputs[0]['size'], self.stream)
        # self.stream.synchronize()
        return self.out_tensor

if __name__ =='__main__':
    inputs = np.random.randn(32, 3, 640, 640)
    engine = Engine("./best_folded_v3_tiny.engine")
    inputs = inputs.astype(engine.dtype)
    s = time.time()
    res = engine.forward(inputs)
    torch.cuda.synchronize()
    e = time.time()
    print("Inference time: ", round(e - s, 4))

    print("Result size: ", res.size())