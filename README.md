# cpucuda runtime
Implementation of the CUDA API for cpucuda forked from hipCPU

## build

```
mkdir build
cd build
cmake .. -DCUDA_PATH=<path_to_cuda>
make
```

results in static library `build/src/libcpucudart.a`
