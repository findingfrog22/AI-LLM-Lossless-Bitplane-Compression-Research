This guide is for this simulation: RAGEmbeddingSimulation.py R1.0.0+

Hardware Requirements:
- Of course, for this simulation, stronger hardware is always better, but here is the recommended baseline
Baseline:
- CPU: 8 cores or higher
- GPU: Intel ARC iGPU (ArrowLake or later), AMD HD 780m (Ryzen 7000 series or later)
- RAM: 16GB+ (32 Recommended)
- SSD: 100+GB (To hold embedding models, source data, etc)
- OS: Windows 11 or Linux (kernel 6.8+)

Recommended Specs:
- CPU: 16+ Cores
- GPU: Nvidia RTX 3000 series or newer (with CUDA 13.0+ support), Intel ARC A/B series, AMD RX 7700 and higher
- VRAM Requirements: 16GB (recommended), 32GB+ (for heavier models like Qwen3 8B or Gemma 2 9B)
- RAM: 32GB+ (64-128GB recommended for larger files)
- SSD: 200GB+ (to hold all embedding models and source files)
- OS: Linux (newer kernels, 6.11+)

Library Requirements:
- There are a lot of library requirements. The simulation will tell you too which ones to install.
- Here are the most important and hardest ones to get working:
- Pytorch 2.11.0+ or newer (compiled for either cpu, cuda, xpu, rocm, etc)
- llama_cpp python compiled with gpu support (or cpu if needed)

Simulation Tuning Parameters:

This simulation has many settings and parameters for the user to tune, but here are the only important ones that the User may need to tune in between runs or devices:
- NUM_ROWS
- NUM_LINES
- SAMPLE_RATE
- QUANTIZATION_TYPE
- BLOCK_SIZE
- ACCELERATION_DEVICE
For any other settings, I would advise against changing them.

NUM_ROWS: This is the number of vector embedding rows that the simulation will analyze
- 1024 is default
- NUM_ROWS should be less than or equal to NUM_LINES
- Will be autoadjusted in certain cases if there aren't enough vector embeddings

NUM_LINES: This is the number of lines/samples that the embedding model will use to compute embeddings.
- Only applies to vectors that aren't precomputed
- For text: is the number of lines of text that will be fed into the embedding model (1 line --> 1 embedding row)
- For video: is the max number of frames that will be analyzed by the embedding model (1 sampled frame --> 1 embedding row)

SAMPLE_RATE: For Video embeddings only. Determines the frequency in Hz for how often to sample frames.
- Default is -1. This will make it sample every 1/FPS per second, where FPS is the detected frames per second of the video. For example, if FPS is 60, then it will sample every 1/60th of a second.
- if 1, then it will sample once per second

QUANTIZATION_TYPE: Determines what datatype the quantized tensor will be.
- Default is torch.int8
- This is quite important, and will effect how you should calculate BLOCK_SIZE

BLOCK_SIZE: Determines the block size of the ZSTD and LZ4 compressors
- Default is 4096 (=4KB)
- You will be changing this option very often, even between runs
- NOTE: this is highly reccommended to ensure accurate results: BLOCK_SIZE = (EmbeddingDimension * Bitcount(QUANTIZATION_TYPE) ) / 8
- I will explain what this is saying:
- EmbeddingDimension: This is the dimension of a single vector embedding from your file (computed or precomputed) [Ex: 768, 1024, etc]
- Bitcount(QUANTIZATION_TYPE): This is the number of bits that represents your QUANTIZATION_TYPE. [Ex: torch.int8 --> 8 bits]
- /8: The reason for this is due to the bitpacking of bits into bytes, so your size has to account for bytes, not bits
- So for example:
- Dimension: 1024
- QUANTIZATION_TYPE: torch.int32
- Then, BLOCK_SIZE = (1024 * 32) / 8, Which means BLOCK_SIZE = 4096

ACCELERATION_DEVICE: This is crucial to making sure it runs with high performance.
- "cpu": default setting, highest compatibility and reliability, but slowest in terms of performance
- "xpu": uses Intel ARC gpu backend, is quite fast, even on ARC iGPUs, but can be unstable or give garbage results
- "cuda": supports Nvidia CUDA and AMD ROCm. AMD ROCm isn't fully tested, but Nvidia CUDA is fully validated. Fast and compatible
- Make sure you have the libraries installed if you want to use anything besides "cpu". For gpu acceleration, you need the GPU compiled versions of llama_cpp python AND pytorch

Simulation How to Interpret Results:

You will see results that give percentages or ratios. There are two major metric formats used in this simulation:
- Compression Ratio: The ratio of compressed size to baseline size. Ratio = Compressed / Baseline
- % Space Savings: The percent space savings of compressed size vs baseline size. % Space Savings = (1 - (Compressed / Baseline)) * 100%
- You will see tables that show stuff like % space savings vs ########. Here is what they mean:
  
- For quantized data, you will see:
- These Vs values will be your bitplane+quantized+compression vs [baseline]
- vs Original Data Size: How much space is saved vs the size of your original data (no modifications)
- vs Quantized+Scalar: How much space is saved vs only quantized+scalar data (no compression)
- vs Direct Compression: How much space is saved vs the original data being compressed directly (no quantization)
- vs Quant Compressed+Scale (No BP): How much space is saved vs not using bitplane (quantization and compression only)

- Note: for mentions of "raw" or "direct", it means that there is no bitplaning involved in those
- For Unquantized data, you will see:
- No Bitplane Compressed vs Original Data Size: This is unquantized compression (without bitplane) vs the original unmodified data
- Bitplane Compressed vs Original Data Size: This is the compression of the bitplaned unquantized data vs the original unmodified data
- Bitplane Compressed vs No Bitplane Compressed: This is the effect of bitplaning vs no bitplaning on compression

There will also be analytics showing compression ratios per bitplane, and metrics showing % space savings per row (for more granularity)
