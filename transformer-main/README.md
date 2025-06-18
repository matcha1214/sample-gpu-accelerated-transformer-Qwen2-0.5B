# transformer

Caltech CS179 Transformer Project

final score for the current implementation: 98/100

-1: superfluous add kernel 
-1: incorrect implementation, query head should be divided by kv heads

# Background

I recommend you read [Attention Is All You Need](https://arxiv.org/abs/1706.03762) many times, this paper is critical to all transformer models.

We will be implementing the [Qwen2](https://arxiv.org/pdf/2407.10671) model, an open-weights LLMs.
Qwen2's architecture is copied from [Meta's Llama](https://arxiv.org/abs/2407.21783), except Qwen2 uses bias in the QKV matrices.
Both of these architectures implement [grouped-query attention](https://arxiv.org/abs/2305.13245),
[layer normalization](https://arxiv.org/abs/1607.06450v1) with zero mean,
[SwiGLU feed forward networks](https://arxiv.org/abs/2002.05202), and
[rotary positional embeddings](https://arxiv.org/abs/2104.09864).

# Implementation

We will only implement autoregressive decoding, which supports inference (generation) with one token at a time.
With single-token decoding, we avoid the masked attention operator which is the focus of most transformer optimization such as [FlashAttention](https://arxiv.org/pdf/2205.14135).
Additionally, we will not support batching.

We will use the [`bfloat16`](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) data type to store
model weights and key/value cache to reduce memory bandwidth relative to `float32`. However,
internally, we will use `float32` for accumulation within kernels, to minimize unnecessary floating-point rounding errors.

For simplicity, all tensors in this implementation will use row-major ordering, i.e. the memory is contiguous along the last dimension.
However, this layout is not optimal for performance is all cases.

# Real world LLM inference

This project is missing many features in real-world production LLM inference, such as:
- Batching
- Prefill phase with masked matrix multiply
- Tensor cores
- Quantization
- Advanced model architectures, such as Mixture-of-Experts (MoE)
- Multi-GPU and multi-node parallelization
- KV cache offloading
- Sampling methods, such as Top-K

# Assignment
You will implement all kernels necessary for the LLM. Libraries are not allowed (such as cuBLAS, CUTLASS, cub, and thrust),
you must code all kernels from scratch.

## Part 1 (first week)

For the questions, cite sources you used. To submit, zip your repository to `~/lab5_2025_submission.zip`.

### Question 1.1 (5 points)
In this assignment, we will not be using tensor cores, because they require advanced data transfer layouts.
Instead, we will implement matrix-vector multiply with standard fused-multiply-add operators.
What is ratio of BF16 tensor core FLOPS to BF16 non-tensor core FLOPS on an A100-PCIE-40GB GPU?
Note: NVIDIA and AMD marketing both try to inflate their performance by measuring "sparse" tensor core operations, but nobody uses those.

https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf

"BFLOAT16
Tensor Core
312 TFLOPS"

https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf

"
Peak BF16 TFLOPS
 (non-Tensor) NVIDIA A100: 39"

Therefore, the ratio of BF16 Tensor Core FLOPS to BF16 non-Tensor Core FLOPS (without sparsity) on an A100-PCIE-40GB GPU is:

312 TFLOPS / 39 TFLOPS = 8

### Question 1.2 (5 points)
What is the expected speedup of tensor cores vs non-tensor cores for matrix-vector multiplication on an A100-PCIE-40GB GPU?
Make an argument based on arithmetic intensity (FLOPS is not the whole story).
Assume the matrix and vector are read from off-chip memory.

https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf

"1555 GB/sec of memory bandwidth"

AI = FLOPs / Bytes Read
AI = (2 * M * K) / (2 * K * (M+1))
AI = M / (M+1) FLOPs/byte
AI ≈ 1 FLOP/byte

P_MemBound = Memory Bandwidth * Arithmetic Intensity = 1.555 TFLOPS

Perf_NTC = min(39 TFLOPS, 1.555 TFLOPS) Perf_NTC = 1.555 TFLOPS
Perf_TC = min(312 TFLOPS, 1.555 TFLOPS) Perf_TC = 1.555 TFLOPS

Expected Speedup = Perf_TC / Perf_NTC = 1x

Both the non-Tensor Cores and the Tensor Cores cannot compute faster than the rate at which data is supplied from memory. Since both are capped by the same memory bandwidth limit for this operation, the Tensor Cores offer no significant speedup over non-Tensor Cores for matrix-vector multiplication under these conditions. The bottleneck is not the compute units themselves, but the data pipeline feeding them. Therefore, the expected speedup is negligible.

### Coding (80 points)
Implement GPU operators:
- ArgMax
- LayerNorm
- MatrixVectorMultiply
- RoPE
- SiLUMult

### Profiling (10 points)
Profile all your kernels with `ncu`, with input sizes matching what you'd expect for Qwen2 0.5B.
For each kernel, provide a screenshot and explain something interesting you noticed.
For example:
- Explain why your kernel is memory-bandwidth limited, latency/occupancy-limited, compute-limited, or limited by some other overhead.
- Explain why your kernel has suboptimal memory accesses, and a potential strategy to improve the kernel with expected performance increase.
- Explain which kernels are the most important to optimize, and which ones are less important.
- Explain how the performance would be different in another scenario (e.g. longer sequence length, larger model, increased batch size)
- Explain similarities across the kernels

### see the profiling README

## Part 2 (second week)

To submit, zip your repository to `~/lab6_2025_submission.zip`.

### Question 2.1 (3 points)
List all the matrix-vector multiplies in a Qwen2 0.5B layer, including the (M, K) dimensions of the matrix.
(Do not include grouped-query attention).

A single layer of the Qwen2 0.5B model involves seven key matrix-vector multiplication operations when processing input data. Within the attention mechanism, these include: the query projection, where an (896, 896) matrix transforms the 896-dimensional input hidden state into query vectors; the key projection, using a (128, 896) matrix to produce key vectors from the same input; the value projection, with a (128, 896) matrix creating value vectors; and finally, the output projection, where an (896, 896) matrix combines the attention outputs back into the hidden state dimension. The feed-forward network part of the layer also contains three such operations: a gate projection with a (4864, 896) matrix, an up projection also with a (4864, 896) matrix, and a down projection employing an (896, 4864) matrix to process the activations. The first dimension (M) of these (M, K) pairs indicates the output size, while the second dimension (K) matches the input vector's size.

### Question 2.2 (2 points)
Treating each query head as a row of a matrix, what are the dimensions of the matrix-matrix multiply in a
Qwen2 0.5B layer grouped-query attention operation? Assume current sequence length is 1234 tokens.

In the Qwen2 0.5B model's grouped-query attention mechanism, when processing a sequence of 1234 tokens, the core matrix-matrix multiplication to calculate attention scores involves query heads divided by KV heads. With 14 query heads and 2 KV heads, we have 7 query heads per KV head group. For each group, the matrix-matrix multiplication is between a (7, 64) query matrix and a (64, 1234) transposed key matrix, resulting in a (7, 1234) attention score matrix per group. Since there are 2 such groups (corresponding to the 2 KV heads), the total computation involves two separate (7, 64) × (64, 1234) = (7, 1234) matrix multiplications, effectively giving us the attention scores for all 14 query heads across the 1234 sequence positions.

### Question 2.3 (5 points)
Assuming off-chip memory bandwidth is the limiting factor, what is the theoretical minimum inference latency (in ms)
for Qwen2 0.5B on an A100-PCIE-40GB, with BF16 weights? Assume small sequence length (i.e. KV cache size is negligible).

The theoretical minimum time (latency) to generate a single token with the Qwen2 0.5B model on an A100-PCIE-40GB GPU, using BF16 precision for its weights and assuming memory bandwidth is the sole bottleneck, is approximately 0.635 milliseconds. This calculation is based on the model's specific size of 494 million parameters, which translates to 988 million bytes (988 MB) when stored in BF16 format (2 bytes per parameter). Given the A100-PCIE-40GB's memory bandwidth of 1,555 GB per second, the latency is found by dividing the total model size in bytes by the memory bandwidth in bytes per second. This figure represents an ideal scenario where data transfer is perfectly efficient and computational overhead is negligible.

### Question 2.4 (5 points)
Determine the sequence length at which the KV cache becomes non-negligible in terms of performance;
specifically, at what sequence length in Qwen2 0.5B would the KV cache become 10% the size of the model parameters?

For the Qwen2 0.5B model, the KV cache (which stores past key and value states for efficient generation) would reach 10% of the total model parameter size at a sequence length of approximately 8,040 tokens. The model has 494 million parameters, amounting to 988 million bytes in BF16 format; 10% of this is 98.8 million bytes. Each token added to the sequence requires storing its key and value vectors across all 24 layers. For Qwen2 0.5B, this means storing 2 key vectors of 64 dimensions each and 2 value vectors of 64 dimensions each, per layer, all in BF16 (2 bytes per value). This sums up to 12,288 bytes per token for the entire KV cache. Dividing the 98.8 million byte threshold by 12,288 bytes per token gives the sequence length where the KV cache size becomes a significant fraction of the model's weight storage.

### Coding (75 points)
Complete:
- GroupQueryAttention
  - Must use online numerically stable softmax, see section 3.1 of [Online normalizer calculation for softmax](https://arxiv.org/pdf/1805.02867)
- Qwen2Layer
- Qwen2Model

You should not allocate or free any memory inside `Qwen2Model::forward`;
scratch space should be allocated only at model initialization, in constructors.

Test by running `./transformer`, and 100 tokens will be produced, matching the python reference implementation 

Once working, you can run `./transformer --interactive --max-seq-len 10000` to send messages with a chatbot interface.

### Profiling (10 points)

Once working, profile your implementation with:
```bash
ncu --set full --nvtx --nvtx-include last_token/ -c100 -o profile ./transformer
```
(may adjust -c parameter to number of kernels per layer)

How many microseconds per layer does your implementation take?
What is the slowest part of the layer and why?
Include screenshots of the something interesting you notice, and explain.

### see the profiling README

## Assignment notes

- Your kernels must fully occupy the GPU when possible (i.e. do not launch with only 1 block, launch with many).
- Kernels should have optimal memory access (coalesced gmem, and no smem bank conflicts) when possible.
- Always use CUDA streams when launching kernels, such as:
  - `my_kernel<<<grid_dim, block_dim, 0, stream>>>(my_arg);`
- Use the test cases and python reference to check the correctness of your implementation.

## Debugging tips

- Add print statements in the python implementation and equivalents in the CUDA implementation, such as:
```python
print('after q proj:', queries[0, 0])
```
corresponding to
```c++
std::cerr << "after q proj: " << static_cast<float>(*static_cast<__nv_bfloat16*>(queries->data)) << std::endl;
```
and check when they diverge.

- All GPU memory is allocated with `cudaMallocManaged`, which allows you to access the GPU memory from the CPU.
  Therefore, with plain GDB, we can run:
  - `CUDA_LAUNCH_BLOCKING=1 gdb ./transformer`
  - Set breakpoints
  - Save a tensor to disk: `dump binary memory /tmp/queries.bin queries->data ((uint8_t*)queries->data)+queries->size`
  - Load the tensor in python: `torch.from_file('/tmp/queries.bin',size=head_size*num_query_heads,dtype=torch.bfloat16).reshape(num_query_heads, head_size)`

## Test cases
Run all test cases with:
- `cd build`
- `cmake --build .`
- `ctest`

For tests that are failing, you can run them individually to see which elements were incorrect, for example:
- `cd build`
- `cmake --build .`
- `./silumulttest`

Failing output:
```
difference at index 0: GPU calculated 10.875, CPU calculated 108.5
difference at index 1: GPU calculated -5.65625, CPU calculated 0.332031
difference at index 2: GPU calculated 7.3125, CPU calculated -76.5
...
```

Also, note that passing all test cases does not mean you will get an A.
The test cases only check for correctness, not for performance.
If your kernels have needless suboptimal memory access, poor occupancy, or other performance issues,
the tests will still pass, but you will not get a good grade.

### AI is used for debugging

## Author
Sam Foxman 2025
