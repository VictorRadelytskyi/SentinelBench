# vLLM Serving

Serve a post-trained adapter merged into its base model with tensor parallelism:

```powershell
vllm serve <merged-model-path> --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 8192
```

For high-throughput tool-use traffic, benchmark request concurrency and prompt-token distributions separately. The serving rollout must use the same chat template and tool schema as the post-training data.