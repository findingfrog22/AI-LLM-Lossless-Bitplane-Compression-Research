
WikipediaDPR_path = "C:/Users/findi/OneDrive/Desktop/AI Research Datasets/Wikipedia DPR Mini/" #base directory of Wikipedia DPR mini dataset
testfile = WikipediaDPR_path + "Multiset/train-00000-of-00157.parquet" #current file selection

NUM_ROWS = 1 #this is how many rows of vector embeddings will be processed
#Note about NUM_ROWS: for LZ4 and ZSTD compression analysis, make sure this number is divisible by 8 (for multivectors)
NUM_DIMS = 768 #this is the number of weights per vector embedding

#some more global settings
ACCELERATION_DEVICE = "xpu"
PRINT_DEBUG = False #determines if you want to print information relevant for debugging
#this includes:
# - quantized + dequantized + scale data (for both vertical and horizontal)
# - bit conversions for quantized and scale data
# - bitplanes for quantized and scale data
# [False, default] off
# [True] on

#symmetric scalar quantization settings
MULTI_SCALAR = False
#MULTI_SCALAR determines if you want to do symmetric scalar quantization on your columns of vector embeddings
# --> note that this only works if your NUM_ROWS > 1 (allows for vertical bit plane disaggregation)
# True - [Multiscalar on]
# False - [Multiscalar off, default]
# --> note: this will print after your per-vector symmetric scalar quantization, as this will always be on regardless
FLOAT_BITS = 32 #number of bits in input float [FP32 - 32 default, FP64 - 64, etc]
INT_BITS = 8 #number of bits in the output integer [INT8 - 8 default, INT4 - 4, etc]

#lz4 and zstd compression settings
BLOCK_SIZE = 4096 #block size for compression, [4KB default - 4096, 8KB - 8192, 16KB - 16384]

#Some helper functions
#symmetric scalar quantization and dequantization functions
def Symmetric_Scalar_Quantization_ROW(tens_emb):
    import torch
    import numpy
    #per vector scale first
    #first, get max absolute value per vector (dim=1 is row-wise, keepdim=true for easy broadcasting)(uses intel XVE vector engines across rows)
    mxvals = torch.max(torch.abs(tens_emb), dim=1, keepdim=True).values
    #second, vectorized scale generation
    scales = mxvals / 127.0 #for now
    scales = torch.clamp(scales, min=1e-9) #for now, and these are the scales per vector embedding row
    #third, element-wise quantization and rounding (uses XPU backend to minimize memory trips)
    quantized = torch.round(tens_emb / scales).to(torch.int8) #for now, quantized to int8
    if(PRINT_DEBUG == True):
        print("\nRow-Wise Symmetric Scalar Quantization (Quantized Values): ")
        print(quantized)
        print("\nRow-Wise Symmetric Scalar Quantization (Scalar Values): ")
        print(scales)
    return quantized, scales

def Symmetric_Scalar_Dequantization_ROW(data, scalars):
    import torch
    import numpy
    #first, cast INT8 data to target float type
    fl = data.to(torch.float32)
    #second, do the multiplication across iGPU
    dequantized = fl * scalars
    if(PRINT_DEBUG == True):
        print("\nRow-Wise Symmetric Scalar Dequantization (Dequantized values): ")
        print(dequantized)
    return dequantized

def Symmetric_Scalar_Error_ROW(orig, dequant):
    import torch
    import torch.nn.functional as f
    import numpy
    print("\nRow-Wise Symmetric Scalar Dequantization Error (Mean Square Error, MSE) Stats: ")
    #overall loss
    overall_loss = f.mse_loss(orig, dequant)
    print("\nOverall Loss (MSE): ")
    print(overall_loss)
    #row-wise
    #elementwise
    element_loss = f.mse_loss(orig, dequant, reduction='none')
    print("\nElement-Wise Loss (MSE): ")
    print(element_loss)
    #row-wise avgs
    vector_loss = element_loss.mean(dim=1)
    print("\nLoss per Vector (MSE): ")
    print(vector_loss)
    #finding worst vector
    worst_vector = torch.argmax(vector_loss)
    print("\nLeast accurate vector (MSE): ")
    print(worst_vector)
    return vector_loss

def Symmetric_Scalar_Quantization_COL(tens_emb):
    import torch
    import numpy
    #per vector scale first
    #first, get max absolute value per vector (dim=1 is row-wise, keepdim=true for easy broadcasting)(uses intel XVE vector engines across rows)
    mxvals = torch.max(torch.abs(tens_emb), dim=0, keepdim=True).values
    #second, vectorized scale generation
    scales = mxvals / 127.0 #for now
    scales = torch.clamp(scales, min=1e-9) #for now, and these are the scales per vector embedding row
    #third, element-wise quantization and rounding (uses XPU backend to minimize memory trips)
    quantized = torch.round(tens_emb / scales).to(torch.int8) #for now, quantized to int8
    if(PRINT_DEBUG == True):
        print("\nColumn-Wise Symmetric Scalar Quantization (Quantized Values): ")
        print(quantized)
        print("\nColumn-Wise Symmetric Scalar Quantization (Scalar Values): ")
        print(scales)
    return quantized, scales

def Symmetric_Scalar_Dequantization_COL(data, scalars):
    import torch
    import numpy
    #first, cast INT8 data to target float type
    fl = data.to(torch.float32)
    #second, do the multiplication across iGPU
    dequantized = fl * scalars
    if(PRINT_DEBUG == True):
        print("\nColumn-Wise Symmetric Scalar Dequantization (Dequantized values): ")
        print(dequantized)
    return dequantized

def Symmetric_Scalar_Error_COL(orig, dequant):
    import torch
    import torch.nn.functional as f
    import numpy
    print("\nColumn-Wise Symmetric Scalar Dequantization Error (Mean Square Error, MSE) Stats: ")
    #overall loss
    overall_loss = f.mse_loss(orig, dequant)
    print("\nOverall Loss (MSE): ")
    print(overall_loss)
    #row-wise
    #elementwise
    element_loss = f.mse_loss(orig, dequant, reduction='none')
    print("\nElement-Wise Loss (MSE): ")
    print(element_loss)
    #col-wise avgs
    vector_loss = element_loss.mean(dim=0)
    print("\nLoss per column (position in embedding vector of 768 dims) (MSE): ")
    print(vector_loss)
    #finding worst vector
    worst_vector = torch.argmax(vector_loss)
    print("\nLowest accuracy column (of all embedding vectors of 768 dims) (MSE): ")
    print(worst_vector)
    return vector_loss

#bitplane stuff
def rowbinary_quant(row):
    return [format(x, '08b') for x in row]
def rowbinary_scale(row):
    return [format(x, '032b') for x in row]

def binarytolist(element):
    return [int(x) for x in element]

def binaryvectortolist(row):
    return [[int(x) for x in y] for y in row]

#this function returns the compressibility of every bit plane in the matrix
#Here is some info about compressibility:
#Ratio Value | Meaning
#0.05 --> Extremelly compressible (95% reduction)
#0.50 --> Moderately compressible (takes half the space)
#1.00 --> No benefit (compressed size == original size)
#>1.00 --> Negative compression (compressed size > original size)
#I plan to add Arithmetic Coding to this too to have even further compression analysis
def vertical_bitplane_compressibility(matrix, num_dims): #analyzes compressibility of multiple vectors
    import torch
    import numpy
    from multiprocessing import Pool
    
    tensor = torch.from_numpy(matrix).to(ACCELERATION_DEVICE)
    differentials = tensor[:, :, 1:] != tensor[:, :, :-1]
    
    RPV = differentials.sum(dim=2) + 1 #runs per vector
    ratio = (RPV * 2) / num_dims
    return ratio

#this function returns the compressibility of every bit plane horizontally in a single vector embedding
#Here is some info about compressibility:
#Ratio Value | Meaning
#0.05 --> Extremelly compressible (95% reduction)
#0.50 --> Moderately compressible (takes half the space)
#1.00 --> No benefit (compressed size == original size)
#>1.00 --> Negative compression (compressed size > original size)
#I plan to add Arithmetic Coding to this too to have even further compression analysis
#Tensor output interpretation:
#Each horizontal row represents a single dimension of a vector embedding (out of 768)
#Each element within a row represents the compressibility ratio of a single bit plane (out of 32 bit planes)
# --> leftmost plane is MSB, and will typically have the highest compressibility
# --> rightmost plane is LSB, and will typically have the lowest compressibility
# --> each ratio is out of all of the rows of vector embeddings that you have analyzed (you set this with NUM_ROWS)
def horizontal_bitplane_compressibility(matrix): #analyzes compressibility of single vector
    import torch
    import numpy
    from multiprocessing import Pool
    
    tensor = matrix.to(ACCELERATION_DEVICE)
    differentials = tensor[:, 1:] != tensor[:, :-1]
    
    RPV = differentials.sum(dim=1) + 1 #runs per vector
    planelength = tensor.shape[1]
    ratio = (RPV * 2) / planelength
    return ratio

#more analytics stuff
def quantization_histogram(quant, mode):
    import torch
    import numpy
    import matplotlib.pyplot as plt
    # 4. Move to CPU for plotting
    quantized_cpu = quant.cpu().numpy().flatten()

    # 5. Plotting the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(quantized_cpu, bins=255, range=(-128, 128), color='#4682B4', edgecolor='black', alpha=0.7)
    plt.title('Distribution of INT8 Quantized Vector Embeddings (' + mode + ')')
    plt.xlabel('Quantized Integer Value')
    plt.ylabel('Frequency (Total Elements)')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig('quantized_distribution[' + mode + '].png')

#now on to LZ4 and ZSTD [THIS IS FOR MONDAY 2/9/26 AND LATER BTW, PLANNING STAGE AND IMPLEMENTATION DETAILS]
#There are 2-3 different analysis techniques
#1.) [vertical compression] "sliced" compression, where we get entire 2D bitplanes 768 tall x #rows deep per bitplane and perform LZ4/ZSTD on each of these (Vertical compression)
# --> tells how compressible entire bitplanes (of all vector dims and rows) are
#2.) [depth compression] "tiled" compression, where we get the compressibility of each 1 x #rows deep per bitplane per dim of vector embedding, results in 2d plane of compressibility for each bitplane for each vector dim (Depth compression)
# --> tells how compressible bitplanes per vector dim are (across all rows)
#3.) [optional, topological compression] performs compressibility of each bitplane for the entire vector dim, takes each 1x768 [0] of each and compresses it
# --> tells how compressible each row's bitplanes are (across all vector embedding dims)
#now ideally, we should probably make them compatible with quantized, scales, horizontal, and vertical, lets think about the logistics of this
# - Quantized (vertical and horizontal) (768 tall, 8 wide, #rows deep)
# --> Vertical: output --> (1 tall, 8 wide)
# --> Depth: output --> (768 tall, 8 wide)
# --> Topological: output --> (1 tall, 8 wide, #rows deep)
# - Scales (vertical) [per dim] (32 tall, 1 wide, 768 deep)
# --> Vertical: output --> (not really possible)
# --> Depth: output --> (32 tall, 1 wide) [need custom implementation]
# --> Topological: output --> (not really possible)
# - Scales (horizontal) [per vector embedding] (32 tall, 1 wide, #rows deep)
# --> Vertical: output --> (not really possible)
# --> Depth: output --> (32 tall, 1 wide) [semi custom impelementation, same as scales vertical depth]
# --> Topological: output --> (not really possible)
# now for single vector embedding input:
# - Quantized: horizontal only (768 tall, 8 wide) # note that vertical and topological are the same thing for this one
# --> Vertical: output --> (1 tall, 8 wide)
# --> Depth: output --> (very useless)
# --> Topological: output --> (1 tall, 8 wide)
# Scales (vertical) [per dim] (not possible)
# --> Vertical: output --> (not possible)
# --> Depth: output --> (not possible)
# --> Topological: output --> (not possible)
# Scales (horizontal) [per vector embedding] (not possible)
# --> Vertical: output --> (not possible)
# --> Depth: output --> (not possible)
# --> Topological: output --> (not possible)
# so Scale for Single vector embedding is a simple LZ4/ZSTD Compression analysis value

#2-3 more vibe coded functions, be careful

def lz4_compress_list(data_list):
    import lz4.frame
    return sum(len(lz4.frame.compress(a, block_size=BLOCK_SIZE)) for a in data_list)

def zstd_compress_list(data_list):
    import zstandard as zstd
    """Compresses a list of byte-buffers and returns the total compressed size."""
    c = zstd.ZstdCompressor(level=3)
    return sum(len(c.compress(b)) for b in data_list)

def run_phased_benchmark(tensor_d): #results: < 1 is good
    import torch
    import multiprocessing as mp
    import zstandard as zstd
    import pandas as pd
    D, B, N = tensor_d.shape
    # 1. PACKING PHASE (XPU)
    # Collapse N (depth) into N//8 bytes
    tensor_dbn = torch.from_numpy(tensor_d).to(ACCELERATION_DEVICE)
    reshaped = tensor_dbn.view(D, B, N // 8, 8).to(torch.uint8)
    packed = (reshaped[..., 0] << 7) | (reshaped[..., 1] << 6) | \
             (reshaped[..., 2] << 5) | (reshaped[..., 3] << 4) | \
             (reshaped[..., 4] << 3) | (reshaped[..., 5] << 2) | \
             (reshaped[..., 6] << 1) | (reshaped[..., 7] << 0)
    
    # Move to CPU: Shape (Bitplane=8, Dim=768, Packed_Depth=N/8)
    brick = packed.permute(1, 0, 2).cpu().numpy()
    original_size_kb = (D * N / 8) / 1024
    results = {k: {"Bitplane": k} for k in range(8)}

    with mp.Pool(processes=8) as pool:
        # --- PHASE 1: GLOBAL WALL (8 planes in parallel) ---
        print("Running Global Wall Analysis...")
        global_buffers = [brick[k].tobytes() for k in range(8)]
        # Using a simple list comprehension for global since it's just 8 tasks
        g_sizes = pool.map(zstd_compress_list, [[b] for b in global_buffers])
        for k, size in enumerate(g_sizes):
            results[k]["Global(ZSTD)"] = round((size / 1024) / original_size_kb, 3)
        g4_sizes = pool.map(lz4_compress_list, [[b] for b in global_buffers])
        for k, size in enumerate(g4_sizes):
            results[k]["Global(LZ4)"] = round((size / 1024) / original_size_kb, 3)

        # --- PHASE 2: TEMPORAL FIBRES (8 planes in parallel) ---
        print("Running Temporal Fibre Analysis...")
        # We prepare a list of lists: 8 bitplanes, each containing 768 strings
        temporal_tasks = [[brick[k, i].tobytes() for i in range(D)] for k in range(8)]
        t_sizes = pool.map(zstd_compress_list, temporal_tasks)
        for k, size in enumerate(t_sizes):
            results[k]["Temporal(ZSTD)"] = round((size / 1024) / original_size_kb, 3)
        t4_sizes = pool.map(lz4_compress_list, temporal_tasks)
        for k, size in enumerate(t4_sizes):
            results[k]["Temporal(LZ4)"] = round((size / 1024) / original_size_kb, 3)

        # --- PHASE 3: SPATIAL COLUMNS (8 planes in parallel) ---
        print("Running Spatial Column Analysis...")
        # We prepare a list of lists: 8 bitplanes, each containing N/8 strings
        spatial_tasks = [[brick[k, :, j].tobytes() for j in range(N // 8)] for k in range(8)]
        s_sizes = pool.map(zstd_compress_list, spatial_tasks)
        for k, size in enumerate(s_sizes):
            results[k]["Spatial(ZSTD)"] = round((size / 1024) / original_size_kb, 3)
        s4_sizes = pool.map(lz4_compress_list, spatial_tasks)
        for k, size in enumerate(s4_sizes):
            results[k]["Spatial(LZ4)"] = round((size / 1024) / original_size_kb, 3)

    # Final Table
    df = pd.DataFrame(list(results.values())).sort_values("Bitplane", ascending=False)
    print("\n" + df.to_string(index=False))
    return df
    #BTW:
    # --> Global == "sliced" or "vertical"
    # --> Temporal == "depth"
    # --> Spatial == "topological"

def vertical_scale_compression_benchmark(Scale):
    import torch
    import multiprocessing as mp
    import zstandard as zstd
    import pandas as pd
    scale = Scale[0]
    print(scale.shape)
    D, B = scale.shape
    # 1. PACKING PHASE (XPU)
    # Collapse N (depth) into N//8 bytes
    tensor_dbn = torch.from_numpy(scale).to(ACCELERATION_DEVICE)
    reshaped = tensor_dbn.view(D, B // 8, 8).to(torch.uint8)
    packed = (reshaped[..., 0] << 7) | (reshaped[..., 1] << 6) | \
             (reshaped[..., 2] << 5) | (reshaped[..., 3] << 4) | \
             (reshaped[..., 4] << 3) | (reshaped[..., 5] << 2) | \
             (reshaped[..., 6] << 1) | (reshaped[..., 7] << 0)
    #get original size
    orig_size = (D * (B // 8)) / 1024 #in KB
    brick = packed.cpu().numpy()
    results = {k: {"Bitplane": k} for k in range(32)}
    with mp.Pool(processes=8) as pool:
        # --- PHASE 1: GLOBAL WALL (8 planes in parallel) ---
        print("Running Global Wall Analysis...")
        global_buffers = [brick[k].tobytes() for k in range(32)]
        # Using a simple list comprehension for global since it's just 32 tasks
        g_sizes = pool.map(zstd_compress_list, [[b] for b in global_buffers])
        for k, size in enumerate(g_sizes):
            results[k]["Scale Vertical (ZSTD)"] = round((size / 1024) / orig_size, 3)
        g4_sizes = pool.map(lz4_compress_list, [[b] for b in global_buffers])
        for k, size in enumerate(g4_sizes):
            results[k]["Scale Vertical (LZ4)"] = round((size / 1024) / orig_size, 3)
    # Final Table
    df = pd.DataFrame(list(results.values())).sort_values("Bitplane", ascending=False)
    print("\n" + df.to_string(index=False))
    return df

def horizontal_scale_compression_benchmark(scale):
    import torch
    import multiprocessing as mp
    import zstandard as zstd
    import pandas as pd
    #scale = Scale[0]
    print(scale.shape)
    D, B = scale.shape
    # 1. PACKING PHASE (XPU)
    # Collapse N (depth) into N//8 bytes
    tensor_dbn = torch.from_numpy(scale).to(ACCELERATION_DEVICE)
    reshaped = tensor_dbn.view(D, B // 8, 8).to(torch.uint8)
    packed = (reshaped[..., 0] << 7) | (reshaped[..., 1] << 6) | \
             (reshaped[..., 2] << 5) | (reshaped[..., 3] << 4) | \
             (reshaped[..., 4] << 3) | (reshaped[..., 5] << 2) | \
             (reshaped[..., 6] << 1) | (reshaped[..., 7] << 0)
    #get original size
    orig_size = (D * (B // 8)) / 1024 #in KB
    brick = packed.cpu().numpy()
    results = {k: {"Bitplane": k} for k in range(32)}
    with mp.Pool(processes=8) as pool:
        # --- PHASE 1: GLOBAL WALL (8 planes in parallel) ---
        print("Running Global Wall Analysis...")
        global_buffers = [brick[k].tobytes() for k in range(32)]
        # Using a simple list comprehension for global since it's just 32 tasks
        g_sizes = pool.map(zstd_compress_list, [[b] for b in global_buffers])
        for k, size in enumerate(g_sizes):
            results[k]["Scale Vertical (ZSTD)"] = round((size / 1024) / orig_size, 3)
        g4_sizes = pool.map(lz4_compress_list, [[b] for b in global_buffers])
        for k, size in enumerate(g4_sizes):
            results[k]["Scale Vertical (LZ4)"] = round((size / 1024) / orig_size, 3)
    # Final Table
    df = pd.DataFrame(list(results.values())).sort_values("Bitplane", ascending=False)
    print("\n" + df.to_string(index=False))
    return df

def single_embedding_run_phased_benchmark(tensor_d): #results: < 1 is good
    import torch
    import multiprocessing as mp
    import zstandard as zstd
    import pandas as pd
    D, B = tensor_d.shape
    print("\nDEBUG: TENSOR SHAPE: ")
    print(tensor_d.shape)
    # 1. PACKING PHASE (XPU)
    # Collapse N (depth) into N//8 bytes
    #tensor_dbn = torch.from_numpy(tensor_d).to(ACCELERATION_DEVICE)
    reshaped = tensor_d.view(D, B // 8, 8).to(torch.uint8)
    packed = (reshaped[..., 0] << 7) | (reshaped[..., 1] << 6) | \
             (reshaped[..., 2] << 5) | (reshaped[..., 3] << 4) | \
             (reshaped[..., 4] << 3) | (reshaped[..., 5] << 2) | \
             (reshaped[..., 6] << 1) | (reshaped[..., 7] << 0)
    
    # Move to CPU: Shape (Bitplane=8, Dim=768, Packed_Depth=N/8)
    #brick = packed.permute(1, 0).cpu().numpy()
    brick = packed.cpu().numpy()
    #D, B = brick.shape
    #print("\nD: " + str(D))
    #for s in brick: #notice that there is something weird here, horizontal characters have more redundancy than vertically
        #print(s)
    original_size_kb = (D * B / 8) / 1024
    results = {k: {"Bitplane": k} for k in range(D)}

    with mp.Pool(processes=8) as pool:
        # --- PHASE 1: GLOBAL WALL (8 planes in parallel) ---
        print("Running Global Wall Analysis...")
        global_buffers = [brick[k].tobytes() for k in range(8)]
        # Using a simple list comprehension for global since it's just 8 tasks
        g_sizes = pool.map(zstd_compress_list, [[b] for b in global_buffers])
        for k, size in enumerate(g_sizes):
            results[k]["Global(ZSTD)"] = round((size / 1024) / original_size_kb, 3)
        g4_sizes = pool.map(lz4_compress_list, [[b] for b in global_buffers])
        for k, size in enumerate(g4_sizes):
            results[k]["Global(LZ4)"] = round((size / 1024) / original_size_kb, 3)

    # Final Table
    df = pd.DataFrame(list(results.values())).sort_values("Bitplane", ascending=False)
    print("\n" + df.to_string(index=False)) #confirmed correct results
    return df
    #BTW:
    # --> Global == "sliced" or "vertical"
    # --> Temporal == "depth"
    # --> Spatial == "topological"
    
def single_scale_compression(Scale): # < 1 good
    import torch
    import multiprocessing as mp
    import zstandard as zstd
    import lz4.frame
    import pandas as pd
    import numpy
    scale = Scale[0][0]
    dat = str(scale)
    bat = int(dat, 2)
    raw_data = bat.to_bytes(4, byteorder='big')
    zstd_compression = zstd.ZstdCompressor(level=3).compress(raw_data)
    lz4_compression = lz4.frame.compress(raw_data, store_size=False)
    print("\nSingle Scale Compression: \nOriginal Size: " + str(len(raw_data)) + " bytes\nZSTD Ratio: " + str(len(zstd_compression) / len(raw_data)) + "\nLZ4 Ratio: " + str(len(lz4_compression) / len(raw_data)))

#lets keep this simulation simpler and more focused on given priorities
#What we need:
# - Perform symmetric scalar quantization for raw embedding FP32 --> INT8 + scalar (for begginning, use single vector embedding dim=768 as granularity for calculating scale)
# --> Table: quantization error (mean squared error) between original FP32 and INT8 vectors to ensure scale is applied correctly
# --> Histogram: A distribution plot of the INT8 value (from -128 to 127) to see if the range is fully utilized
# - Fixed block size compression: Apply bitplane lossless compression to INT8 data, use LZ4 and ZSTD of fixed block sizes (4KB,8KB,16KB). This helps us simulate real world SSD IO behavior and page level compression
# --> Bar chart: compression ratios for both LZ4 and ZSTD across different block sizes
# --> Bar chart: average compression ratio for each bit plane
# - Explore compression ratios in the columnar layout (basically what you did with the vertical multi-vector bit plane compression)
# --> Documentation: Record your findings in documentation showing your new findings and insights on which dimensions (MSBs,LSBs) show the most gain in this vertical arrangement
# --> Bar chart: average compression ratio for each bit plane

#Here is what I see the process looking like:
#0.) Getting and parsing the file data
#1.) Symmetric Scalar quantization: Raw embedding FP32 --> INT8 + scalar (have 2 versions: per vector scale (this is what you should start with, single and multivector), per column of all vectors (multivector only))
#1.1.) table: quantization error (mean squared error) between original FP32 and INT8 vectors (to ensure scale is applied correctly)
#1.2.) histogram: distribution plot of INT8 value (from -128 to 127) to see if range is fully utilized
#2.) Turn quantized vector embeddings into both types of bitplanes (vertical (multivector) and horizontal (single vector))
#2.1.) Bar chart: average compression ratio for each bit plane
#2.2.) Document findings and insights on which dimensions (MSBs/LSBs) show most gain in vertical compression
#3.)[basically 2.0.5] Perform LZ4 and ZSTD compression per bit plane using specific block sizes (4KB,8KB,16KB)
#3.1.) Bar chart: compression ratios for both LZ4 and ZSTD across different block sizes
#3.2.) Bar chart: average compression ratio for each bit-plane
def initialization():
    #0.) Get and parse the file data
    import pandas
    global testfile
    global NUM_ROWS
    #allows to select specific columns to save memory/time
    datafile_subset = pandas.read_parquet(testfile, columns=['embeddings']) #now we have access to raw embeddings
    raw_embeddings = datafile_subset['embeddings'][:NUM_ROWS]
    if(PRINT_DEBUG == True):
        print("\nDEBUG: raw_embeddings: ")
        print(raw_embeddings)
    
    #1.) Symmetric Scalar quantization: Raw embedding FP32 --> INT8 + scalar (have 2 versions: per vector scale (this is what you should start with, single and multivector), per column of all vectors (multivector only))
    import torch
    import numpy
    global ACCELERATION_DEVICE
    #get a tensor version of our raw embeddings
    numpy_emb = numpy.stack(raw_embeddings)
    tens_emb = torch.tensor(numpy_emb).to(ACCELERATION_DEVICE)
    
    #now we do default row-wise quantization
    quantized,scales = Symmetric_Scalar_Quantization_ROW(tens_emb) #get the quantization and scale
    #now we should try to dequantize it
    quantized = quantized.to(ACCELERATION_DEVICE)
    quantization_histogram(quantized, "horizontal") #1.2.) histogram: distribution plot of INT8 value (from -128 to 127) to see if range is fully utilized
    quantized = quantized.to(ACCELERATION_DEVICE)
    scales = scales.to(ACCELERATION_DEVICE)
    dequantized = Symmetric_Scalar_Dequantization_ROW(quantized, scales) #gets it back in dequantized format
    #now, we want to calculate the accuracy/loss of our quantization
    #original = tens_emb.to(ACCELERATION_DEVICE)
    #dequantized = dequantized.to(ACCELERATION_DEVICE)
    err = Symmetric_Scalar_Error_ROW(tens_emb, dequantized) #1.1.) table: quantization error (mean squared error) between original FP32 and INT8 vectors (to ensure scale is applied correctly)
    
    #now lets try element/column wise quantization (ONLY IF MULTI_SCALAR == True and NUM_ROWS > 1)
    global MULTI_SCALAR
    if((NUM_ROWS > 1) and (MULTI_SCALAR == True)):
        #perform multiple vector columnar symmetric scalar quantization
        quantized_C,scales_C = Symmetric_Scalar_Quantization_COL(tens_emb) #get the quantization and scale
        #now we should try to dequantize it
        quantized_C = quantized_C.to(ACCELERATION_DEVICE)
        quantization_histogram(quantized_C, "vertical") #1.2.) histogram: distribution plot of INT8 value (from -128 to 127) to see if range is fully utilized
        quantized_C = quantized_C.to(ACCELERATION_DEVICE)
        scales_C = scales_C.to(ACCELERATION_DEVICE)
        dequantized_C = Symmetric_Scalar_Dequantization_COL(quantized_C, scales_C) #gets it back in dequantized format
        #now, we want to calculate the accuracy/loss of our quantization
        #original = tens_emb.to(ACCELERATION_DEVICE)
        #dequantized = dequantized.to(ACCELERATION_DEVICE)
        err_C = Symmetric_Scalar_Error_COL(tens_emb, dequantized_C) #1.1.) table: quantization error (mean squared error) between original FP32 and INT8 vectors (to ensure scale is applied correctly)
    
    from multiprocessing import Pool
    #lets now bitplane the quantized data and scalars separately
    #here is for the horizontal symmetric quantization
    #quantized data bitplane
    bit_patterns_gpu = quantized.view(torch.uint8)
    bit_patterns_cpu = bit_patterns_gpu.cpu()
        
    #converts it to binary
    binstring_qr = Pool().map(rowbinary_quant, bit_patterns_cpu)
    if(PRINT_DEBUG == True):
        print("\nDEBUG: binstring_qr: ")
        print(numpy.array(binstring_qr)) #works, prints whole thing for debugging
        #print(binstring_qr) #works, prints abbreviated version
    #scalar bitplane
    bpg_sr = scales.view(torch.uint32)
    bpc_sr = bpg_sr.cpu()
    #converts it to binary
    binstring_sr = Pool().map(rowbinary_scale, bpc_sr)
    if(PRINT_DEBUG == True):
        print("\nDEBUG: binstring_sr: ")
        print(numpy.array(binstring_sr)) #works, prints whole thing for debugging
        #print(binstring_sr) #works, prints abbreviated version
    
    #now we need to do this for the vertical symmetric scalar quantization too, (only if applicable)
    if((NUM_ROWS > 1) and (MULTI_SCALAR == True)):
        #first, we need bitplanes of quantized data
        bpg_qc = quantized_C.view(torch.uint8)
        bpc_qc = bpg_qc.cpu()
        #convert it to binary
        binstring_qc = Pool().map(rowbinary_quant, bpc_qc)
        if(PRINT_DEBUG == True):
            print("\nDEBUG: binstring_qc: ")
            print(numpy.array(binstring_qc)) #works, prints whole thing for debugging
            #print(binstring_qc) #works, prints abbreviated version
        #then, we want bitplanes of the scalar data
        bpg_sc = scales_C.view(torch.uint32)
        bpc_sc = bpg_sc.cpu()
        #convert it to binary
        binstring_sc = Pool().map(rowbinary_scale, bpc_sc)
        if(PRINT_DEBUG == True):
            print("\nDEBUG: binstring_sc: ")
            print(numpy.array(binstring_sc)) #basically works (so close, but lsbs are a little off), probably a rounding thing [prints whole thing for debugging]
            #print(binstring_sc) #works, prints abbreviated version
    
    #now we need to handle both horizontal and vertical bitplaning
    if(NUM_ROWS == 1):
        #first, use cpu parallel acceleration to split each element into its own list
        #(we need to repeat this process for both quantized data and the scales)
        #binstring_sr
        #binstring_qr
        #first, the quantized data
        vector_embedding_qr = binstring_qr[0]
        print("\nDEBUG: binstring_qr: ")
        print(numpy.asarray(vector_embedding_qr))
        print(numpy.asarray(vector_embedding_qr).shape)
        binsplit_qr = Pool().map(binarytolist, vector_embedding_qr)
        if(PRINT_DEBUG == True):
            print("\nDEBUG: binsplit_qr: ")
            print(numpy.array(binsplit_qr))#works
        print("\nDEBUG: binsplit_qr: ")
        print(numpy.asarray(binsplit_qr))
        print(numpy.asarray(binsplit_qr).shape)
        
        #now we need to transpose the matrix
        Matrix_qr = torch.tensor(binsplit_qr)
        matrixXPU_qr = Matrix_qr.to(ACCELERATION_DEVICE)
        bitplanes_qr = matrixXPU_qr.T
        if(PRINT_DEBUG == True):
            print("\nDEBUG: bitplanes_qr: ")
            print(bitplanes_qr)
        #print(bitplanes_qr.to("cpu"))#works
        print("\nDEBUG: bitplanes_qr: ")
        print(bitplanes_qr)
        print(bitplanes_qr.shape)
        
        #now analyze the compressibility of the bitplanes
        print("\n[single horizontal quant, quant] HORIZONTAL BITPLANE COMPRESSION (Single plane) (-->0 good): ")
        comp_qr = horizontal_bitplane_compressibility(bitplanes_qr)
        print(comp_qr)
        
        print("\n[single horizontal quant, quant] Average Compression Ratio: ")
        avg_sr = torch.mean(comp_qr)
        print(avg_sr)
        
        #test
        single_embedding_run_phased_benchmark(bitplanes_qr)
        #
        
        #now, the scale data
        vector_embedding_sr = binstring_sr
        binsplit_sr = Pool().map(binarytolist, vector_embedding_sr)
        if(PRINT_DEBUG == True):
            print("\nDEBUG: binsplit_sr: ")
            print(binsplit_sr)#works
        print("\nDEBUG: binsplit_sr: ")
        print(binsplit_sr)
        
        #test
        single_scale_compression(binsplit_sr)
        #
        
        #since the scale only has 1 value, we cannot decompose it into bitplanes for compression analysis
        '''
        #now we need to transpose the matrix
        Matrix_sr = torch.tensor(binsplit_sr)
        matrixXPU_sr = Matrix_sr.to(ACCELERATION_DEVICE)
        bitplanes_sr = matrixXPU_sr.T
        #print(bitplanes.to("cpu"))#works
        
        #now analyze the compressibility of the bitplanes
        comp_sr = horizontal_bitplane_compressibility(bitplanes_sr)
        '''
        pass
    elif(NUM_ROWS > 1):
        #going to try a similar method as before on cpu, but it will be much slower due to multiple rows
        #(we need to repeat this process for both quantized data and the scales)
        #binstring_sc
        #binstring_qc
        #first, try it with quantized data (row-wise)
        print("\nDEBUG: binstring_qr: ")
        print(numpy.asarray(binstring_qr))
        print(numpy.asarray(binstring_qr).shape)
        binsplit_qr = Pool().map(binaryvectortolist, binstring_qr)
        if(PRINT_DEBUG == True):
            print("\nDEBUG: binsplit_qr: ")
            print(binsplit_qr)
        print("\nDEBUG: binsplit_qr: ")
        print(numpy.asarray(binsplit_qr))
        print(numpy.asarray(binsplit_qr).shape)
    
        #then transpose time
        matrix_qr = numpy.stack(binsplit_qr, axis=-1)
        if(PRINT_DEBUG == True):
            print("\nDEBUG: matrix_qr: ")
            print(matrix_qr)
        print("\nDEBUG: matrix_qr: ")
        print(matrix_qr)
        print(matrix_qr.shape)
        
        comp_qr = vertical_bitplane_compressibility(matrix_qr, NUM_DIMS)
        print("\n[horizontal quant, quant] MULTI VECTOR VERTICAL BIT PLANE COMPRESSION RATIOS [Run Length Encoding](-->0 good): ")
        print(comp_qr)
        
        #for now, test
        run_phased_benchmark(matrix_qr)
        #
        
        print("\n[horizontal quant, quant] AVERAGE COMPRESSION RATIOS PER BIT PLANE: ")
        avg_qr = torch.mean(comp_qr, dim=0)
        print(avg_qr)
        #print("\nexplanation of results above:\n - leftmost column is MSB\n - rightmost column is LSB\n - There is a single row, gets average of all 768 dimensions' bit planes of the # of rows you selected\n - compression ratio -> 0 means perfect compressibility, >1 means negative compressibility (takes more space to compress it than original)")

        print("\n[horizontal quant, quant] AVG COMPRESSION RATIO PER VECTOR DIM: ")
        avg2_qr = torch.mean(comp_qr, dim=1)
        print(avg2_qr)
        
        print("\n[horizontal quant, quant] AVG COMPRESSION RATIO TOTAL BLOCK: ")
        avg3_qr = torch.mean(avg_qr, dim=0)
        print(avg3_qr)
        
        #then try it with scales (row-wise)
        binsplit_sr = Pool().map(binaryvectortolist, binstring_sr)
        if(PRINT_DEBUG == True):
            print("\nDEBUG: binsplit_sr: ")
            print(binsplit_sr)
    
        #then transpose time
        matrix_sr = numpy.stack(binsplit_sr, axis=-1)
        if(PRINT_DEBUG == True):
            print("\nDEBUG: matrix_sr: ")
            print(matrix_sr)
        
        comp_sr = vertical_bitplane_compressibility(matrix_sr, NUM_DIMS)
        print("\n[horizontal quant, scale] MULTI VECTOR VERTICAL BIT PLANE COMPRESSION RATIOS [Run Length Encoding](-->0 good): ")
        print(comp_sr)
        
        print("\n[horizontal quant, scale] AVERAGE COMPRESSION RATIO ACROSS ALL BIT PLANES ACROSS ALL ROWS/VECTORS: ")
        avg_sr = torch.mean(comp_sr)
        print(avg_sr)
        
        #for now, test
        vertical_scale_compression_benchmark(matrix_sr)
        #
        
        #now we also need to account for vertical symmetric quantization too
        if(MULTI_SCALAR == True):
            #first, try it with quantized data (col-wise)
            binsplit_qc = Pool().map(binaryvectortolist, binstring_qc)
            #print("\nDEBUG: binsplit_qc: ")
            #print(binsplit_qc)
    
            #then transpose time
            matrix_qc = numpy.stack(binsplit_qc, axis=-1)
            if(PRINT_DEBUG):
                print("\nDEBUG: matrix_qc: ")
                print(matrix_qc)
        
            comp_qc = vertical_bitplane_compressibility(matrix_qc, NUM_DIMS)
            print("\n[vertical quant, quant]MULTI VECTOR VERTICAL BIT PLANE COMPRESSION RATIOS [Run Length Encoding](-->0 good): ")
            print(comp_qc)
            
            print("\n[vertical quant, quant] AVERAGE COMPRESSION RATIOS PER BIT PLANE: ")
            avg_qc = torch.mean(comp_qc, dim=0)
            print(avg_qc)
            #print("\nexplanation of results above:\n - leftmost column is MSB\n - rightmost column is LSB\n - There is a single row, gets average of all 768 dimensions' bit planes of the # of rows you selected\n - compression ratio -> 0 means perfect compressibility, >1 means negative compressibility (takes more space to compress it than original)")
        
            print("\n[vertical quant, quant] AVG COMPRESSION RATIO PER DIM OF ALL VECTOR EMBEDDINGS: ")
            avg2_qc = torch.mean(comp_qc, dim=1)
            print(avg2_qc)
            
            print("\n[vertical quant, quant] AVG COMPRESSION RATIO TOTAL BLOCK: ")
            avg3_qc = torch.mean(avg_qc, dim=0)
            print(avg3_qc)
            
            #try this
            run_phased_benchmark(matrix_qc)
            #
        
            #then try it with scales (row-wise)
            binsplit_sc = Pool().map(binaryvectortolist, binstring_sc)
            if(PRINT_DEBUG == True):
                print("\nDEBUG: binsplit_sc: ")
                print(binsplit_sc)
    
            #then transpose time
            binsplit_sc = binsplit_sc[0]
            matrix_sc = numpy.stack(binsplit_sc, axis=-1)
            if(PRINT_DEBUG == True):
                print("\nDEBUG: matrix_sc: ")
                print(matrix_sc)
        
            #comp_sc = vertical_bitplane_compressibility(matrix_sc, NUM_DIMS)#causes issues here, might try horizontal bitplane compressibility
            comp_sc = horizontal_bitplane_compressibility(torch.from_numpy(matrix_sc)) #works
            #through analysis, the structure of matrix_sc is the same as single vector horizontal bit plane compression, so I can verify that it works
            print("\n[vertical quant, scale] MULTI VECTOR HORIZONTAL BIT PLANE COMPRESSION RATIOS [Run Length Encoding](-->0 good): ")
            print(comp_sc) #confirmed to work correctly
            
            print("\n[vertical quant, scale] AVERAGE COMPRESSIBILITY BETWEEN ALL COMPRESSION RATIOS [RLE]: ")
            print(torch.mean(comp_sc))
            
            #test it
            horizontal_scale_compression_benchmark(matrix_sc)
            pass
        pass
    pass

#for thread safety
if __name__ == '__main__':
    initialization()
    #TO DO LIST FOR 2/8/26+:
    # - I already checked the correctness of (multiscalar stuff for the most part, and some other main stuff), but make sure the rest is correct?
    # - You implemented compression analytics for ZSTD and LZ4, you did analyze correctness and didnt see anything out of the ordinary (is it okay??)
    # - Add print info for what the analytics are (LZ4 and ZSTD) for each quant, scale, vertical, horizontal, etc
    # - Remove debug print statements, and clean up code a bit
    # - add print information under each results to explain how to interpret the results
    # - work on developing the other steps
    
    #once the main stuff is down: we can add the nice stuff:
    # - localized file choosing
    # - file and block randomization
    # - all the compression ratio metrics should be standard (final size / original size out of 1.0)
    pass