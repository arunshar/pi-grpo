# Profiler pass: one llm-grpo training step

- device: `NVIDIA H100` (key `h100`, peak 989 TFLOPS bf16 dense)
- torch 2.11.0+cu130, CUDA available: True
- timed steps: 3 (after 1 warmup); median step time **4352.0 ms** (min 4138.9 / max 4355.0 ms)
- achieved FLOPs/step: 2.009e+14 (source: profiler (with_flops))
- achieved 46.2 TFLOP/s -> **MFU 4.7%** (analytic peak denominator)

## Top ops by self-CUDA time (MEASURED)

| rank | op | self CUDA us | calls | profiler FLOPs |
|---|---|---:|---:|---:|
| 1 | `aten::mm` | 0 | 68305 | 1.90e+14 |
| 2 | `cudaLaunchKernel` | 0 | 305329 | 0.00e+00 |
| 3 | `aten::mul` | 0 | 75518 | 2.03e+10 |
| 4 | `aten::addmm` | 0 | 16716 | 1.06e+13 |
| 5 | `aten::matmul` | 0 | 66590 | 0.00e+00 |
| 6 | `aten::_cudnn_attention_forward` | 0 | 5572 | 0.00e+00 |
| 7 | `aten::copy_` | 0 | 72420 | 0.00e+00 |
| 8 | `aten::add` | 0 | 56273 | 5.86e+09 |
| 9 | `aten::_to_copy` | 0 | 69610 | 0.00e+00 |
| 10 | `aten::linear` | 0 | 83107 | 0.00e+00 |

## Caveats

- Step time and the op ranking are measured by `torch.profiler`.
- MFU uses an ANALYTIC vendor peak (gpu_estimator) as the denominator; it is an upper-bound ratio, not a hardware SM-utilization counter.
