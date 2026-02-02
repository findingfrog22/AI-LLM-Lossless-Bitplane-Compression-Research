RPI Spring 2026 URP: 
This research is heavily involved in addressing the current memory bottlenecks of current AI inferencing solutions. The end ideal is to have specialized hardware built in to SSD controllers to compress and decompress vector embeddings and other inference data to reduce its memory footprint, reduce bandwidth constraints, and improve $/GB metrics.
Current Goal:
- Determine the best orientation for maximizing lossless compressibility of vector embeddings using LZ4 and ZSTD in 4kb block sizes

Current Methodologies:
- Bitplane disaggregation of vector embeddings in both vertical (multiple vector embeddings) and horizontal (single vector embedding) formats
- Symmetric scalar quantization (Ex: FP32 --> INT8 + Scalar)

Datasets for this project:
- Wikipedia DPR
- Fineweb Edu Embeddings
- TREC-RAG 2024

Performance and Implementation Details:
- Utilizes CPU multithreading & Intel Arc for highly parallel acceleration
- --> GPU mainly accelerates matrix and tensor manipulations and calculations on large blocks of data
- --> Technically supports other GPU frameworks like CUDA and ROCm, but isn't tested, so use at your own risk
- --> CPU mainly performs data conversion/transformation operations in a pool of multiple threads, processes GPU output in a Python and user readable format
- File Selection (newer versions)
- --> Gives options for locally scanned files or online streaming*

*online streaming isn't available for Wikipedia DPR dataset due to being a legacy database
