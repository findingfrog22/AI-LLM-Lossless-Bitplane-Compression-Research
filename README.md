RPI Spring 2026 URP: 
This research is heavily involved in addressing the current memory bottlenecks of current AI inferencing solutions. The end goal is to have specialized hardware built into SSD controllers to compress and decompress vector embeddings and other inference data to reduce its memory footprint, reduce bandwidth constraints, and improve $/GB metrics.

Current Goal:
- Determine the effects of bitplaning on the compressibility of vector embeddings for LZ4 and ZSTD compression codecs of various block sizes

Results:
- Compression results show up to a 53% reduction in memory losslessly
- Experimental multimodal embedding results show up to a 92% reduction in memory

Conclusions:
- Bitplaning only showed better results when not combined with lossy quantization.
- Bitplaning vastly improves LZ4 compressibility regardless if it is quantized or not.

See the results of this research in this repository (slides, data, working document, code, etc)

Future Goals:
- Implementing Elastic Precision Retrieval to vector embedding RAG retrieval to alleviate bandwidth further
- Implementing computational pipeline into TRACE Interface for higher memory and bandwidth savings

Datasets for this project:*
- Wikipedia DPR
- Fineweb Edu Embeddings
- MS MARCO V2.1 English V3 (Based on TREC RAG 2024)
- Booksum
- Wikitext-103
- Amazon Vector Database
- BEIR nfcorpus
- BEIR fiqa
- BEIR touche-2020
- BEIR dbpedia
- BEIR fever
*Even though the simulation will download the embedding models for you, keep in mind that you must download your own source data to do embeddings with (.parquet, .txt, etc). It must be in a similar relative directory to the simulation file itself

Embedding Models Included in this project:
Text Embedding Models (Validated for research):
- BAAI BGE-M3 (FP16)
- EmbeddingGemma 300M (FP16) [Based on Gemma 3]
- Gemma 2 9B (FP16)
- Qwen3 4B (FP16)
- Qwen3 8B (FP16)
- Harrier OSS V1 0.6B (BF16)
- Arctic Snowflake Embed V2.0 (FP32)
- E5 Large V2 (FP32)
- Nomic Embed V1.5 (FP32)
- E5 Mistral 7B (FP16)
Multimodal Embedding Models (Experimental, not validated for research):
- Laion.ai CLAP (FP64)
- OpenAI CLIP ViT-L/14 (FP32)
- Alibaba Qwen3-VL Embed 2B (FP32)

Multimodal Support:
- CLAP: [Image: Unsupported | Video: Unsupported | Audio: Supported]
- CLIP ViT: [Image: Supported | Video: Supported* | Audio: Unsupported]
- Qwen3-VL: [Image: Supported | Video: Supported | Audio: Unsupported]
*video support is not natively supported in this model, this is experimental feature

Supported Filetypes:
- .parquet
- .csv
- .npy (precomputed only)
- Images: .jpg, .png
- Videos: .mp4, .mov
- Audio: .mp3, .wav
- Note: Images, Videos, and Audio are only supported on computed (not precomputed)

Performance and Implementation Details:
- Utilizes CPU multithreading & Intel Arc for highly parallel acceleration (supports other GPU vendors too)
- --> GPU mainly accelerates matrix and tensor manipulations and calculations on large blocks of data
- --> GPU Support 
- --> CPU mainly performs data conversion/transformation operations in a pool of multiple threads, processes GPU output in a Python and user readable format
- For new users, I highly recommend Nvidia CUDA (for speed) or CPU (for reliability). The other backends are much harder to get set up and working. CUDA is fully validated for even the experimental parts of this simulation.

Recommended Hardware Specifications:
Hardware Requirements:
- Of course, for this simulation, stronger hardware is always better, but here is the recommended baseline
Baseline Specifications:
- CPU: 8 cores or higher (AVX2 or newer)
- GPU: Intel ARC iGPU (ArrowLake or later), AMD HD 780m (Ryzen 7000 series or later)
- RAM: 16GB+ (32 Recommended)
- SSD: 100+GB (To hold embedding models, source data, etc)
- OS: Windows 11 or Linux (kernel 6.8+)
  
Recommended Specifications:
- CPU: 16+ Cores (AVX512 or AVX10 reccommended)
- GPU: Nvidia RTX 3000 series or newer (with CUDA 13.0+ support), Intel ARC A/B series, AMD RX 7700 and higher
- VRAM Requirements: 16GB (recommended), 32GB+ (for heavier models like Qwen3 8B or Gemma 2 9B)
- RAM: 32GB+ (64-128GB recommended for larger files)
- SSD: 200GB+ (to hold all embedding models and source files)
- OS: Linux (newer kernels, 6.11+)

Library Notes:
- There are so many libraries used in this simulation, that it would not be countable for me, so I will include the most important libraries that require the most work to set up:
- llama_cpp python: You need to custom compile a GPU accelerated one from source. The default one only uses CPU
- pytorch 2.11.0+ or newer: Must also be compiled for GPU acceleration, as the default one only does CPU.

Known Issues:
- Intel ARC backend is broken on certain text embedding models (E5 large v2, embeddinggemma, harrier OSS)
- Vulkan and AMD backend are not fully tested, may cause similar instability issues as Intel GPU backend
- Intel ARC backend is broken with multimodal embeddings
