#We should be able to handle these file types:
# - .parquet, .csv, .npy, etc
#Initial Setup
import os
base_directory = os.path.dirname(os.path.abspath(__file__))
import torch
file_path = "" #don't touch this unless you know what you are doing
NUM_ROWS = 32768 #This is for you to choose, it is the number of vector embeddings (essentially # of tokens)[must be a multiple of 8]
#Note: your NUM_ROWS should ideally exceed your BLOCK_SIZE, and for best results, should be a multiple of BLOCK_SIZE
NUM_DIMS = 768 #768 is default, but will automatically change with shape recognition
MAX_ROWS = 10000 #10k is default, but will automatically change with shape recognition

#quantization settings
BASE_TYPE = torch.float32 #torch.float32 is default, will autochange during shape detection
QUANTIZATION_TYPE = torch.int8 #You change this value, this is the resulting quantized values [Default, torch.int8]
SCALE_TYPE = torch.float32 #torch.float32 is detault, will autochange during quantization

#some LZ4 and ZSTD Compression Settings
BLOCK_SIZE = 4096 #default - 4096 = 4KB

#performance settings
ACCELERATION_DEVICE = "xpu" #"xpu" is default
#Other options for ACCELERATION_DEVICE may include:
# - "cuda" - this is for Nvidia GPUs and AMD ROCm-compatible GPUs [In theory should be compatible, but it isn't tested]
# - "xpu" - this is for Intel ARC GPUS (iGPU and dGPU) [This is what it was developed on]
# - "cpu" - this is a fallback that runs on your CPU if you don't have a compatible GPU (warning, it will be pretty slow)

#result printing settings (may improve performance too)
PRINT_ORIGINAL = False #prints original input data from file + basic analytics [default True]
PRINT_QUANT = False #prints quantized and dequantized values at beginning, along with MSE error [default True]
PRINT_RLE = False #prints the Run Length Encoding compressibility for every tensor [default True]
PRINT_COMP = True #prints the LZ4 and ZSTD Compression analytics for the bitplanes for every tensor [default True]

#debug settings
PRINT_DEBUG = False #prints useful info for debugging
SHOW_EXTRANEOUS_RESULTS = False #default false to improve performance

SHOW_NEGATIVE_COMPRESSION_RATIOS = False
#Determines whether to compress the uncompressible or not
# - True: will show negative compression ratios (>1.0) [default]
# - False: will change any negative compression ratios to 1.0

IGNORE_RAW = True
#Will skip the raw compression permutations. This saves on performance and VRAM,
#,especially with large # of rows
# - True: doesn't perform calculations on Prequantized data [default]
# - False: performs calculations on prequantized data [warning, performance may drop]

#some functions

#raw vector tensor retrieval from files
def _find_vector_column(sample_row):
    """
    Scans a dictionary or Series to find a column containing a list/array of floats.
    """
    import os
    import json
    import numpy as np
    import pandas as pd
    import zstandard as zstd
    import io
    
    if isinstance(sample_row, pd.Series):
        sample_row = sample_row.to_dict()

    for key, value in sample_row.items():
        # Check if the value is a list or numpy array
        if isinstance(value, (list, np.ndarray)):
            # Ensure it's not empty and contains numbers
            if len(value) > 0 and isinstance(value[0], (int, float, np.float32, np.float64)):
                return key
        # Handle cases where vectors are stored as JSON strings in CSVs
        elif isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            return key
            
    raise ValueError("Could not automatically detect a vector column.")

def extract_vectors(file_path, vector_column_name='embeddings'):
    import os
    import numpy as np
    import pandas as pd
    import json
    import zstandard as zstd
    """
    Extracts vector data from various file formats and returns a NumPy array.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    print(f"--- Processing {ext} file ---")

    try:
        # 1. NumPy Files (Pure numerical data)
        if ext == '.npy':
            return np.load(file_path)
        
        # 2. Parquet Files (Common for Large Datasets)
        elif ext == '.parquet':
            df = pd.read_parquet(file_path)
            print(".parquet KEYS: ")
            print(df.keys())
            col = _find_vector_column(df.iloc[0])
            print(f"Autodetected vector column: '{col}'")
            return np.stack(df[col].values)

        # 3. JSON Lines (Common for API exports like OpenAI)
        elif ext == '.jsonl':
            vectors = []
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    # Adjust 'embedding' key based on your specific JSON structure
                    print("JSONL KEYS: ")
                    print(data.keys())
                    vectors.append(data[vector_column_name])
            return np.array(vectors)
        elif(file_path.endswith('.jsonl.zst')):
            vectors = []
            dctx = zstd.ZstdDecompressor()
            with open(file_path, 'rb') as fh:
                # Create a stream reader to decompress on the fly
                with dctx.stream_reader(fh) as reader:
                    # Wrap the binary stream in a text-mode wrapper
                    import io
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                    for line in text_stream:
                        if line.strip():
                            data = json.loads(line)
                            print("JSONL.ZST KEYS: ")
                            print(data.keys())
                            col = _find_vector_column(data)
                            print(f"Autodetected vector column: '{col}'")
                            vectors.append(data[col])
            return np.array(vectors)
        elif(file_path.endswith('.jsonl.offsets')):
            jsonl_path = file_path.replace('.offsets', '')
            # Read offsets (usually stored as 8-byte integers)
            offsets = np.fromfile(file_path, dtype=np.int64)
        
            # Example: Extracting the first NUM_ROWS vectors using offsets
            vectors = []
            with open(jsonl_path, 'r') as f:
                for off in offsets[:NUM_ROWS]:
                    f.seek(off)
                    line = f.readline()
                    print("JSONL.OFFSETS KEYS: ")
                    print(json.loads(line).keys())
                    vectors.append(json.loads(line)[vector_column_name])
            return np.array(vectors)

        # 4. CSV Files (Text-heavy/Debug formats)
        elif ext == '.csv':
            df = pd.read_csv(file_path)
            # CSV vectors are often stored as strings like "[0.1, 0.2...]"
            # This converts those strings back into actual lists/arrays
            print("CSV KEYS: ")
            print(df.keys())
            if isinstance(df[vector_column_name].iloc[0], str):
                df[vector_column_name] = df[vector_column_name].apply(lambda x: json.loads(x))
            return np.stack(df[vector_column_name].values)

        # 5. SafeTensors (Modern Model Weights)
        elif ext == '.safetensors':
            from safetensors.numpy import load_file
            tensors = load_file(file_path)
            # Safetensors usually have multiple keys; we return the first one or a specific one
            key = list(tensors.keys())[0]
            print(f"Extracted key: {key}")
            return tensors[key]

        # 6. HDF5 (Scientific Data)
        elif ext in ['.h5', '.hdf5']:
            import h5py
            with h5py.File(file_path, 'r') as f:
                # Assuming vectors are in a dataset named 'vectors'
                key = 'vectors' if 'vectors' in f else list(f.keys())[0]
                return np.array(f[key])

        else:
            print(f"Unsupported extension: {ext}")
            return None

    except Exception as e:
        print(f"Error extracting from {ext}: {e}")
        return None

# --- Quick Usage Example ---
# vectors = extract_vectors("my_embeddings.parquet")
# print(f"Shape: {vectors.shape}")

#Quantization Functions
def symmetric_scalar_quantization(tens_emb, direc):
    import torch
    import numpy
    #per vector scale first
    dimen = -1
    if(direc == "row"):
        dimen = 1
        print("\nROW: ")
    elif(direc == "col"):
        dimen = 0
        print("\nCOL: ")
    #first, get max absolute value per vector (dim=1 is row-wise, keepdim=true for easy broadcasting)(uses intel XVE vector engines across rows)
    mxvals = torch.max(torch.abs(tens_emb), dim=dimen, keepdim=True).values
    #second, vectorized scale generation
    global QUANTIZATION_TYPE
    bitsize = 0
    if(QUANTIZATION_TYPE.is_floating_point):
        bitsize += torch.finfo(QUANTIZATION_TYPE).bits
    else:
        bitsize += torch.iinfo(QUANTIZATION_TYPE).bits
    scales = mxvals.float() / (float((2**bitsize)/2) - 1) #improved precision (becuase of new .float(), forces it to FP32), only works when wanting to quantize to an integer datatype
    global SCALE_TYPE
    SCALE_TYPE = scales.dtype
    print("Scales dtype: " + str(scales.dtype))
    scales = torch.clamp(scales, min=1e-9) #for now, and these are the scales per vector embedding row
    #third, element-wise quantization and rounding (uses XPU backend to minimize memory trips)
    quantized = torch.round(tens_emb / scales).to(QUANTIZATION_TYPE) #for now, quantized to int8
    if(PRINT_DEBUG == True):
        print(direc)
        print("\nSymmetric Scalar Quantization (Quantized Values): ")
        print(quantized)
        print("\nSymmetric Scalar Quantization (Scalar Values): ")
        print(scales)
    return quantized, scales

def symmetric_scalar_dequantization(data, scalars):
    import torch
    import numpy
    #first, cast Quantization dtype data to target float type
    fl = data.to(scalars.dtype)
    #second, do the multiplication across iGPU
    dequantized = fl * scalars
    if(PRINT_DEBUG == True):
        print("\nSymmetric Scalar Dequantization (Dequantized values): ")
        print(dequantized)
    return dequantized

def Symmetric_Scalar_Error(orig, dequant, direc):
    import torch
    import torch.nn.functional as f
    import numpy
    dimen = -1
    if(direc == "row"):
        dimen = 1
        print("\nROW-WISE: ")
    elif(direc == "col"):
        dimen = 0
        print("\nCOL-WISE: ")
    print("\nSymmetric Scalar Dequantization Error (Mean Square Error, MSE) Stats: ")
    #overall loss
    overall_loss = f.mse_loss(orig, dequant)
    print("\nOverall Loss (MSE): ")
    print(overall_loss)
    #row-wise
    #elementwise
    element_loss = f.mse_loss(orig, dequant, reduction='none')
    print("\nElement-Wise Loss (MSE): ")
    print(element_loss)
    vector_loss = element_loss.mean(dim=dimen)
    if(direc == "row"):
        #row-wise avgs
        print("\nLoss per Vector (MSE): ")
        print(vector_loss)
    elif(direc == "col"):
        print("\nLoss per column (position in embedding vector of " + str(NUM_DIMS) + " dims) (MSE): ")
        print(vector_loss)
    #finding worst vector
    worst_vector = torch.argmax(vector_loss)
    print("\nLeast accurate vector (MSE): ")
    print(worst_vector)
    return vector_loss

#Bitplane Disaggregation Functions

def direct_bitplane(tens_emb): #replaces the older version
    import torch
    tens_bits = tens_emb.element_size() * 8
    if(tens_bits == 32):
        view_type = torch.int32
    elif(tens_bits == 8):
        view_type = torch.int8
    elif(tens_bits == 16):
        view_type = torch.int16
    else:
        view_type = torch.int32 #default fallback
    
    tens_signed = tens_emb.detach().view(view_type)
    bits = torch.arange(tens_bits - 1, -1, -1, device=tens_emb.device, dtype=torch.int32)
    mask = torch.tensor(1, device=tens_emb.device, dtype=view_type)
    planes_horizontal = (tens_signed.unsqueeze(1) >> bits.view(1, -1, 1)) & mask
    planes_horizontal = planes_horizontal.to(torch.uint8).contiguous()
    planes_vertical = (tens_signed.unsqueeze(0) >> bits.view(-1, 1, 1)) & mask
    planes_vertical = planes_vertical.to(torch.uint8).contiguous()
    planes_vertical = planes_vertical.permute(2,0,1).contiguous()
    #if(print_stage3 == True):
        #print("\nSTAGE 3: Bitplane Disaggregation (raw/direct): H: " + str(planes_horizontal.shape) + "; V: " + str(planes_vertical.shape) + ": ")
        #print("Horizontal Raw: ")
        #print(planes_horizontal)
        #print("Vertical Raw: ")
        #print(planes_vertical)
    return planes_horizontal, planes_vertical

def bitplane(quant, scale, direc): #FIXED
    import torch

    def get_planes(tensor, num_bits, direction):
        # 1. Map to SIGNED type because XPU supports rshift for Int32/Int8
        # float32 -> int32, uint8/int8 -> int8
        if num_bits == 32:
            view_type = torch.int32
        elif num_bits == 8:
            view_type = torch.int8
        elif num_bits == 16:
            view_type = torch.int16
        else:
            view_type = torch.int32 # Default fallback
        
        # 2. View data as signed
        t_signed = tensor.detach().view(view_type)
        
        # 3. Create bit indices (Int32 is safe on XPU)
        bits = torch.arange(num_bits - 1, -1, -1, device=tensor.device, dtype=torch.int32)
        
        # 4. Create the mask (1) 
        mask = torch.tensor(1, device=tensor.device, dtype=view_type)

        # 5. Extract bits
        # Note: (x >> bits) & 1 works the same for signed/unsigned 
        # as long as we only care about the resulting 0 or 1.
        if direction == "horizontal":
            # Target: (N, B, W)
            planes = (t_signed.unsqueeze(1) >> bits.view(1, -1, 1)) & mask
            #planes = planes.to(torch.uint8).contiguous()
            #return planes.permute(2,0,1).contiguous()
        elif direction == "vertical":
            # Target: (B, N, W)
            planes = (t_signed.unsqueeze(0) >> bits.view(-1, 1, 1)) & mask
            planes = planes.to(torch.uint8).contiguous()
            return planes.permute(2,0,1).contiguous() #now returns in form (W,B,N)
        
        # 6. Convert to uint8 (0 or 1) for your RLE and packing analysis
        return planes.to(torch.uint8).contiguous()

    # Determine bit depth
    q_bits = quant.element_size() * 8
    s_bits = scale.element_size() * 8
    
    q_planes = get_planes(quant, q_bits, direc)
    s_planes = get_planes(scale, s_bits, direc)

    return q_planes, s_planes

#Compression Analysis Functions
    
#ported RLE Compression analysis

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
#RLE_bitplane_compressibility notes:
# - Since the input matrix is already in the correct orientation, then the code will be the same for
# both vertical and horizontal compressibility analysis, depending on the matrix
# - but the further more specific analytics depend on if form is vertical or horizontal
def RLE_bitplane_compressibility(matrix, form, title):
    import torch
    import numpy
    #we need to standardize input to tensor form (no numpy.ndarrays)
    MAT = matrix
    if(isinstance(matrix, numpy.ndarray)):
        MAT = torch.from_numpy(matrix)
    #now, we get initial compression analysis:
    tensor = MAT.to(ACCELERATION_DEVICE)
    Differentials = tensor[:, :, 1:] != tensor[:, :, :-1]
    RunsPerBitplane = Differentials.sum(dim=2) + 1 #runs per bitplane
    PlaneLength = tensor.shape[2]
    print("\nDEBUG: PLANELENGTH: " + str(PlaneLength))
    ratio = (RunsPerBitplane * 2) / PlaneLength #this is the overall results
    if(SHOW_NEGATIVE_COMPRESSION_RATIOS == False):
        ratio = torch.clamp(ratio, max=1.0)
    print("\nDEBUG: " + str(title) + " Overall results (RLE): " + str(ratio.shape))
    print(ratio)
    #return ratio?
    if(form == "horizontal"):
        bp_avg = torch.mean(ratio, dim=0) #per bitplane, originally dim=1
        print("\nHorizontal Per Bitplane: ")
        print(bp_avg)
        row_avg = torch.mean(ratio, dim=1) #per vector embedding, originally dim=0
        print("\nHorizontal Per Vector Embedding: ")
        print(row_avg)
    elif(form == "vertical"):
        bp_avg = torch.mean(ratio, dim=0) #per bitplane
        print("\nVertical Per Bitplane: ")
        print(bp_avg)
        dim_avg = torch.mean(ratio, dim=1) #per vector embedding dim
        print("\nVertical Per Dimension: ")
        print(dim_avg)
    print("\nDEBUG: Overall Average RLE Compression Ratio: " + str(torch.mean(ratio)))
    return torch.mean(ratio)

def calculate_overall_RLE_ratio(quant, scale, quant_ratio, scale_ratio):
    import torch
    import numpy
    #first, we need to get the overall # of bits that quant and scale take
    QA, QB, QC = quant.shape
    quant_bits = QA * QB * QC
    
    SA, SB, SC = scale.shape
    scale_bits = SA * SB * SC
    #then, get the total original size in bits, and the total compressed size to the nearest bit
    quant_rle = int(quant_ratio * quant_bits)
    scale_rle = int(scale_ratio * scale_bits)
    tot_rle = quant_rle + scale_rle
    #then get the total ratio
    tot_bits = quant_bits + scale_bits
    tot_ratio = float(tot_rle/tot_bits)
    return tot_ratio

def pack_bits(tensor_r, dim): #NOTE: both of these modes are confirmed correct and working. Keep in mind it does 0 padding for N and/or B if they aren't a multiple of 8, which might inflate compression ratios, but this is done to some degree on real devices anyways
    """
    Packs a binary tensor (0, 1) into uint8 bytes along the N (0) or B (2) dimension.
    Input Shape: (N, W, B)
    """
    import torch
    #clear xpu memory: this is especially important for very large tensors that fill your VRAM
    torch.xpu.empty_cache()
    stats = torch.xpu.memory_stats()
    print(f"Allocated: {stats['allocated_bytes.all.current'] / 1024**3:.2f} GB")
    print(f"Reserved: {stats['reserved_bytes.all.current'] / 1024**3:.2f} GB")
    # 0. Make sure the input is a tensor, turn it to tensor if it is a numpy array
    if(torch.is_tensor(tensor_r) == False):
        tensor_s = torch.tensor(tensor_r)
    else:
        tensor_s = tensor_r
    tensor_s = tensor_s.to(ACCELERATION_DEVICE)
    
    if(dim == 0): #packed_n (for spatial)
        N, W, B = tensor_s.shape
        remainder = N % 8
        if remainder != 0:
            pad_amt = 8 - remainder
            # Pad the first dimension (N). F.pad uses (left, right, top, bottom, front, back)
            # For 3D (N, W, B), the order is (B_front, B_back, W_front, W_back, N_front, N_back)
            tensor_s = torch.nn.functional.pad(tensor_s, (0, 0, 0, 0, 0, pad_amt))
        # Reshape to (N_padded//8, 8, W, B)
        padded_N = tensor_s.shape[0]
        reshaped = tensor_s.reshape(padded_N // 8, 8, W, B)
        # Permute to move the '8' to the end: (N//8, W, B, 8)
        reshaped = reshaped.permute(0, 2, 3, 1)
        
        weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], device=ACCELERATION_DEVICE, dtype=torch.uint8)
        packed = (reshaped.to(torch.uint8) * weights).sum(dim=-1).to(torch.uint8)
        
        return packed
    
    elif(dim == 2): #packed_b (for global and temporal)
        # 1. Safely move the dimension you want to pack to the end
        # Works for (N, W, B) or even (Batch, N, W, B)
        # If dim=0 (N), shape becomes (W, B, N)
        # If dim=2 (B), shape stays (N, W, B)
        tensor = torch.movedim(tensor_s, dim, -1)
        orig_shape = list(tensor_r.shape)
        target_size = orig_shape[-1]

        # 2. Padding (Crucial to avoid Shape Errors)
        remainder = target_size % 8
        if remainder != 0:
            pad_size = 8 - remainder
            # Padding only the last dimension (which is our target dim)
            tensor = torch.nn.functional.pad(tensor, (0, pad_size), "constant", 0)
            orig_shape[-1] += pad_size # Update shape for the reshape

        # 3. Reshape: Isolate groups of 8 bits
        # Shape becomes (..., Target_Size // 8, 8)
        reshaped = tensor.reshape(*orig_shape[:-1], -1, 8)

        # 4. Bit-weighting Math
        # We use powers of 2 (128, 64, 32, 16, 8, 4, 2, 1)
        weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], 
                               device=ACCELERATION_DEVICE, dtype=torch.uint8)
    
        # Multiply and sum to create the byte (0-255)
        packed = (reshaped.to(torch.uint8) * weights).sum(dim=-1).to(torch.uint8)

        # 5. Move it back to where it started
        # (W, B, N/8) -> (N/8, W, B)  OR  (N, W, B/8)
        packed = torch.movedim(packed, -1, dim)
    
        return packed
    elif(dim == 1): #packed_w, purely optional and more for experimental and curiosity purposes
        # 1. Padding Logic (Left Pad)
        W = tensor_s.shape[1]
        remainder = W % 8
        if remainder != 0:
            pad_size = 8 - remainder
            # F.pad for (N, W, B) to pad dim 1 (W) from the left:
            # (B_front, B_back, W_front, W_back, N_front, N_back)
            tensor = torch.nn.functional.pad(tensor_s, (0, 0, 0, pad_size, 0, 0), "constant", 0)
        else:
            tensor = tensor_s

        # 2. Reshape: Split W into (W//8, 8)
        # Shape: (N, W_padded // 8, 8, B)
        padded_W = tensor.shape[1]
        reshaped = tensor.reshape(tensor.shape[0], padded_W // 8, 8, tensor.shape[2])

        # 3. Permute: Move the '8' to the end for the dot product
        # (N, W//8, 8, B) -> (N, W//8, B, 8)
        reshaped = reshaped.permute(0, 1, 3, 2)

        # 4. Bit-weighting Math
        weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], 
                               device=ACCELERATION_DEVICE, dtype=torch.uint8)
    
        # Result: (N, W//8, B)
        packed = (reshaped.to(torch.uint8) * weights).sum(dim=-1).to(torch.uint8)

        return packed
    else:
        raise ValueError("Dimension for pack_bits must be either 0 or 2...")

def lz4_compress_list(data_list): #takes in a list
    import lz4.frame
    if(isinstance(data_list, list)):
        group = b"".join(data_list)
    else:
        group = bytes([data_list])
    #print("\nDEBUG: LZ4 precompressed bytes: " + str(len(group)) + " : ")
    #print(group)
    #print(BLOCK_SIZE)
    return len(lz4.frame.compress(group, block_size=BLOCK_SIZE))
    #return sum(len(lz4.frame.compress(a, block_size=BLOCK_SIZE)) for a in data_list)

def zstd_compress_list(data_list): #takes in a list
    import zstandard as zstd
    """Compresses a list of byte-buffers and returns the total compressed size."""
    if(isinstance(data_list, list)):
        group = b"".join(data_list)
    else:
        group = bytes([data_list])
    #print("\nDEBUG: ZSTD precompressed bytes: " + str(len(group)) + " : ")
    #print(group)
    compressed_size = 0
    c = zstd.ZstdCompressor(level=3)
    total_len = len(group)
    blocks = []
    for i in range(0, total_len, BLOCK_SIZE):
        block = group[i : i + BLOCK_SIZE]
        #padding
        if(len(block) < BLOCK_SIZE):
            block = block.ljust(BLOCK_SIZE, b'\x00')
        compressed_size += len(c.compress(block))
    
    return compressed_size
    #return len(c.compress(group))
    #return sum(len(c.compress(b)) for b in data_list)

#added some new features, like showing negative compression ratios or now
#now it also returns some stuff:
# - returns a dictionary: res
# - res["Global(ZSTD):"], res["Global(LZ4):"], res["Temporal(ZSTD):"], res["Temporal(LZ4):"], res["Spatial(ZSTD):"], res["Spatial(LZ4):"], res["Original Size(Bits):"]
# - keep in mind that the return results will change depending on if negative compression ratios are active or not
# - keep in mind that all results are returned in the number of bits, not bytes
def run_phased_benchmark(tensor_d, form, variant): 
    import torch
    import multiprocessing
    import zstandard as zstd
    import pandas as pd
    import numpy
    import lz4.frame
    
    #if the original input is not a tensor, convert it to a tensor
    if(torch.is_tensor(tensor_d) == False):
        tensor_d = torch.from_numpy(tensor_d).to(ACCELERATION_DEVICE)
    
    #new, for return value for further metrics about space saving
    res = {"Global(ZSTD):": 0, "Global(LZ4):": 0, "Temporal(ZSTD):": 0, "Temporal(LZ4):": 0, "Spatial(ZSTD):": 0, "Spatial(LZ4):": 0, "Original Size(Bits):": tensor_d.shape[0] * tensor_d.shape[1] * tensor_d.shape[2]}
    
    #get the proper N,W,B values depending on if it is vertical or horizontal (if orig tensor is (N,W,B)), and do rest of setup
    if(form == "horizontal"): #typically how the CPU handles tokens
        N,B,W = tensor_d.shape
        #get the required packed_values
        packed_n = pack_bits(tensor_d, dim=0) #packed across n in this case
        packed_b = pack_bits(tensor_d, dim=2) #packed across w in this case
        #get the required tensor forms for Global, Temporal, and Spatial
    elif(form == "vertical"): #common for KV-cache optimizations and columnar databases
        W,B,N = tensor_d.shape
        #get the required packed_values
        packed_b = pack_bits(tensor_d, dim=2) #packed across n in this case
        packed_n = pack_bits(tensor_d, dim=0) #packed across w in this case
        #get the required tensor forms for Global, Temporal, and Spatial
    
    #now universal benchmark setup
    print("\nLZ4+ZSTD Results for " + form + "-" + variant + ": " + str(tensor_d.shape))#tag modifier title
    orig_size_kb = (N * W) / (1024 * 8) #gets it in bytes becuase the compressors return size in bytes
    print("B: " + str(B))
    B_b = B
    results = {k: {"Bitplane": k} for k in range(B_b)} #setting up the title
    cores = min(B_b, multiprocessing.cpu_count())
    check_tensor_entropy(tensor_d)
    with multiprocessing.Pool(processes=cores) as pool:
        if(form == "horizontal"): #(N,B,W)
            print("Stage 1: Global Analysis...") #overall compression per bitplane
            #packed_b (packs across w)
            b_global = packed_b.permute(1,0,2).contiguous().cpu().numpy()
            globaltasks = [[b_global[k].tobytes()] for k in range(B_b)]
            z_gsizes = pool.map(zstd_compress_list, globaltasks)
            l_gsizes = pool.map(lz4_compress_list, globaltasks)
            if(SHOW_NEGATIVE_COMPRESSION_RATIOS == False):
                for k in range(B_b):
                    gh_z = round((z_gsizes[k]/1024) / orig_size_kb, 3)
                    gh_l = round((l_gsizes[k]/1024) / orig_size_kb, 3)
                    results[k]["Global(ZSTD):"] = gh_z if gh_z < 1.0 else 1.000
                    results[k]["Global(LZ4):"] = gh_l if gh_l < 1.0 else 1.000
                    #add to return results too
                    res["Global(ZSTD):"] += z_gsizes[k]*8 if z_gsizes[k] < (orig_size_kb*1024) else orig_size_kb*1024*8
                    res["Global(LZ4):"] += l_gsizes[k]*8 if l_gsizes[k] < (orig_size_kb*1024) else orig_size_kb*1024*8
            else:
                for k in range(B_b):
                    results[k]["Global(ZSTD):"] = round((z_gsizes[k]/1024) / orig_size_kb, 3)
                    results[k]["Global(LZ4):"] = round((l_gsizes[k]/1024) / orig_size_kb, 3)
                    #add to return results too
                    res["Global(ZSTD):"] += z_gsizes[k]*8
                    res["Global(LZ4):"] += l_gsizes[k]*8
            print("Stage 2: Temporal Analysis...") #consistency across all rows/embeddings of a single dim, there are multiple dims
            #packed_n (packs across n) #this one is pretty important for redundancy across all rows per dim per bitplane [main horizontal benchmark]
            n_spatial = packed_n.permute(1,2,0).contiguous().cpu().numpy()
            spatialtasks = list()
            if(SHOW_NEGATIVE_COMPRESSION_RATIOS == False):
                for k in range(B_b):
                    bitplanetasks = [[n_spatial[k][x].tobytes()] for x in range(n_spatial.shape[1])]
                    z_ssizes = sum(pool.map(zstd_compress_list, bitplanetasks))
                    l_ssizes = sum(pool.map(lz4_compress_list, bitplanetasks))
                    sh_z = round((z_ssizes/1024) / orig_size_kb, 3)
                    sh_l = round((l_ssizes/1024) / orig_size_kb, 3)
                    results[k]["Temporal(ZSTD):"] = sh_z if sh_z < 1.0 else 1.000
                    results[k]["Temporal(LZ4):"] = sh_l if sh_l < 1.0 else 1.000
                    #add to return results too
                    res["Temporal(ZSTD):"] += z_ssizes*8 if z_ssizes < (orig_size_kb*1024) else orig_size_kb*1024*8
                    res["Temporal(LZ4):"] += l_ssizes*8 if l_ssizes < (orig_size_kb*1024) else orig_size_kb*1024*8
            else:
                for k in range(B_b):
                    bitplanetasks = [[n_spatial[k][x].tobytes()] for x in range(n_spatial.shape[1])]
                    z_ssizes = sum(pool.map(zstd_compress_list, bitplanetasks))
                    l_ssizes = sum(pool.map(lz4_compress_list, bitplanetasks))
                    results[k]["Temporal(ZSTD):"] = round((z_ssizes/1024) / orig_size_kb, 3)
                    results[k]["Temporal(LZ4):"] = round((l_ssizes/1024) / orig_size_kb, 3)
                    #add to return results too
                    res["Temporal(ZSTD):"] += z_ssizes*8
                    res["Temporal(LZ4):"] += l_ssizes*8
            print("Stage 3: Spatial Analysis...") #consistency within the dim of a single embedding, across multiple embeddings
            #packed_b (packs across w)
            b_temporal = packed_b.permute(1,0,2).contiguous().cpu().numpy()
            temporaltasks = list()
            if(SHOW_NEGATIVE_COMPRESSION_RATIOS == False):
                for k in range(B_b):
                    bitplanetasks = [[b_temporal[k][x].tobytes()] for x in range(b_temporal.shape[1])]
                    z_tsizes = sum(pool.map(zstd_compress_list, bitplanetasks))
                    l_tsizes = sum(pool.map(lz4_compress_list, bitplanetasks))
                    th_z = round((z_tsizes/1024) / orig_size_kb, 3)
                    th_l = round((l_tsizes/1024) / orig_size_kb, 3)
                    results[k]["Spatial(ZSTD):"] = th_z if th_z < 1.0 else 1.000
                    results[k]["Spatial(LZ4):"] = th_l if th_l < 1.0 else 1.000
                    #add to return results too
                    res["Spatial(ZSTD):"] += z_tsizes*8 if z_tsizes < (orig_size_kb*1024) else orig_size_kb*1024*8
                    res["Spatial(LZ4):"] += l_tsizes*8 if l_tsizes < (orig_size_kb*1024) else orig_size_kb*1024*8
            else:
                for k in range(B_b):
                    bitplanetasks = [[b_temporal[k][x].tobytes()] for x in range(b_temporal.shape[1])]
                    z_tsizes = sum(pool.map(zstd_compress_list, bitplanetasks))
                    l_tsizes = sum(pool.map(lz4_compress_list, bitplanetasks))
                    results[k]["Spatial(ZSTD):"] = round((z_tsizes/1024) / orig_size_kb, 3)
                    results[k]["Spatial(LZ4):"] = round((l_tsizes/1024) / orig_size_kb, 3)
                    #add to the return results too
                    res["Spatial(ZSTD):"] += z_tsizes*8
                    res["Spatial(LZ4):"] += l_tsizes*8
        elif(form == "vertical"): #(W,B,N)
            print("Stage 1: Global Analysis...") #overall compression per bitplane
            #packed_b (packs across n), Seems like an important benchmark?
            b_global = packed_b.permute(1,0,2).contiguous().cpu().numpy() #(W,B,N) --> (B,W,N) for access reasons
            globaltasks = [[b_global[x].tobytes()] for x in range(B_b)]
            z_gsizes = pool.map(zstd_compress_list, globaltasks)
            l_gsizes = pool.map(lz4_compress_list, globaltasks)
            if(SHOW_NEGATIVE_COMPRESSION_RATIOS == False):
                for k in range(B_b):
                    gv_z = round((z_gsizes[k]/1024) / orig_size_kb, 3)
                    gv_l = round((l_gsizes[k]/1024) / orig_size_kb, 3)
                    results[k]["Global(ZSTD):"] = gv_z if gv_z < 1.0 else 1.000
                    results[k]["Global(LZ4):"] = gv_l if gv_l < 1.0 else 1.000
                    #add to the return results too
                    res["Global(ZSTD):"] += z_gsizes[k]*8 if z_gsizes[k] < (orig_size_kb*1024) else orig_size_kb*1024*8
                    res["Global(LZ4):"] += l_gsizes[k]*8 if l_gsizes[k] < (orig_size_kb*1024) else orig_size_kb*1024*8
            else:
                for k in range(B_b):
                    results[k]["Global(ZSTD):"] = round((z_gsizes[k]/1024) / orig_size_kb, 3)
                    results[k]["Global(LZ4):"] = round((l_gsizes[k]/1024) / orig_size_kb, 3)
                    #add to the return results too
                    res["Global(ZSTD):"] += z_gsizes[k]*8
                    res["Global(LZ4):"] += l_gsizes[k]*8
            print("Stage 2: Temporal Analysis...") #consistency across all rows/embeddings of a single dim, there are multiple dims
            #packed_b (packs across n), This is an important benchmark, finds the redundancy per bitplane across all vector embeddings/tokens, per vector dim
            b_temporal = packed_b.permute(1,0,2).contiguous().cpu().numpy() #(W,B,N) --> (B,W,N) for access reasons
            temporaltasks = list()
            if(SHOW_NEGATIVE_COMPRESSION_RATIOS == False):
                for k in range(B_b):
                    bitplanetasks = [[b_temporal[k][x].tobytes()] for x in range(b_temporal.shape[1])]
                    z_tsizes = sum(pool.map(zstd_compress_list, bitplanetasks))
                    l_tsizes = sum(pool.map(lz4_compress_list, bitplanetasks))
                    tv_z = round((z_tsizes/1024) / orig_size_kb, 3)
                    tv_l = round((l_tsizes/1024) / orig_size_kb, 3)
                    results[k]["Temporal(ZSTD):"] = tv_z if tv_z < 1.0 else 1.000
                    results[k]["Temporal(LZ4):"] = tv_l if tv_l < 1.0 else 1.000
                    #add to the return results too
                    res["Temporal(ZSTD):"] += z_tsizes*8 if z_tsizes < (orig_size_kb*1024) else orig_size_kb*1024*8
                    res["Temporal(LZ4):"] += l_tsizes*8 if l_tsizes < (orig_size_kb*1024) else orig_size_kb*1024*8
            else:
                for k in range(B_b):
                    bitplanetasks = [[b_temporal[k][x].tobytes()] for x in range(b_temporal.shape[1])]
                    z_tsizes = sum(pool.map(zstd_compress_list, bitplanetasks))
                    l_tsizes = sum(pool.map(lz4_compress_list, bitplanetasks))
                    results[k]["Temporal(ZSTD):"] = round((z_tsizes/1024) / orig_size_kb, 3)
                    results[k]["Temporal(LZ4):"] = round((l_tsizes/1024) / orig_size_kb, 3)
                    #add to the return results too
                    res["Temporal(ZSTD):"] += z_tsizes*8
                    res["Temporal(LZ4):"] += l_tsizes*8
            print("Stage 3: Spatial Analysis...") #consistency across all dims of a single embedding/row, there are multiple embeddings/rows
            #packed_n (packs across w)
            n_spatial = packed_n.permute(1,2,0).contiguous().cpu().numpy() #(W,B,N) --> (B,N,W) for access reasons
            spatialtasks = list()
            if(SHOW_NEGATIVE_COMPRESSION_RATIOS == False):
                for k in range(B_b):
                    bitplanetasks = [[n_spatial[k][x].tobytes()] for x in range(n_spatial.shape[1])]
                    z_ssizes = sum(pool.map(zstd_compress_list, bitplanetasks))
                    l_ssizes = sum(pool.map(lz4_compress_list, bitplanetasks))
                    sv_z = round((z_ssizes/1024) / orig_size_kb, 3)
                    sv_l = round((l_ssizes/1024) / orig_size_kb, 3)
                    results[k]["Spatial(ZSTD):"] = sv_z if sv_z < 1.0 else 1.000
                    results[k]["Spatial(LZ4):"] = sv_l if sv_l < 1.0 else 1.000
                    #add to the return results too
                    res["Spatial(ZSTD):"] += z_ssizes*8 if z_ssizes < (orig_size_kb*1024) else orig_size_kb*1024*8
                    res["Spatial(LZ4):"] += l_ssizes*8 if l_ssizes < (orig_size_kb*1024) else orig_size_kb*1024*8
            else:
                for k in range(B_b):
                    bitplanetasks = [[n_spatial[k][x].tobytes()] for x in range(n_spatial.shape[1])]
                    z_ssizes = sum(pool.map(zstd_compress_list, bitplanetasks))
                    l_ssizes = sum(pool.map(lz4_compress_list, bitplanetasks))
                    results[k]["Spatial(ZSTD):"] = round((z_ssizes/1024) / orig_size_kb, 3)
                    results[k]["Spatial(LZ4):"] = round((l_ssizes/1024) / orig_size_kb, 3)
                    #add to the return results too
                    res["Spatial(ZSTD):"] += z_ssizes*8
                    res["Spatial(LZ4):"] += l_ssizes*8
    
    #now, print and return the results
    df = pd.DataFrame(list(results.values())).sort_values("Bitplane", ascending=False)
    pd.set_option('display.max_columns', None)
    print(df)
    #print("\nDEBUG: Compressed Sizes per Technique in Bits: ")
    #print(res)
    return res

def space_saving_metrics(bit_comp_quant, bit_comp_scale, Raw_size):
    import pandas as pd
    import numpy as np
    framedata = [[],[],[0,0,0,0,0,0],[0,0,0,0,0,0]]
    #most important: Compressed Bit Quant+Scale vs Original Prequantized Uncompressed
    #Global(ZSTD):
    framedata[0].append(round((1-((bit_comp_quant["Global(ZSTD):"]+bit_comp_scale["Global(ZSTD):"])/Raw_size))*100, 3))
    #Global(LZ4):
    framedata[0].append(round((1-((bit_comp_quant["Global(LZ4):"]+bit_comp_scale["Global(LZ4):"])/Raw_size))*100, 3))
    #Temporal(ZSTD):
    framedata[0].append(round((1-((bit_comp_quant["Temporal(ZSTD):"]+bit_comp_scale["Temporal(ZSTD):"])/Raw_size))*100, 3))
    #Temporal(LZ4):
    framedata[0].append(round((1-((bit_comp_quant["Temporal(LZ4):"]+bit_comp_scale["Temporal(LZ4):"])/Raw_size))*100, 3))
    #Spatial(ZSTD):
    framedata[0].append(round((1-((bit_comp_quant["Spatial(ZSTD):"]+bit_comp_scale["Spatial(ZSTD):"])/Raw_size))*100, 3))
    #Spatial(LZ4):
    framedata[0].append(round((1-((bit_comp_quant["Spatial(LZ4):"]+bit_comp_scale["Spatial(LZ4):"])/Raw_size))*100, 3))
    #now do: Compressed Bit Quant+Scale vs Bit Quant+Scale Uncompressed
    #Global(ZSTD):
    framedata[1].append(round((1-((bit_comp_quant["Global(ZSTD):"]+bit_comp_scale["Global(ZSTD):"])/(bit_comp_quant["Original Size(Bits):"]+bit_comp_scale["Original Size(Bits):"])))*100, 3))
    #Global(LZ4):
    framedata[1].append(round((1-((bit_comp_quant["Global(LZ4):"]+bit_comp_scale["Global(LZ4):"])/(bit_comp_quant["Original Size(Bits):"]+bit_comp_scale["Original Size(Bits):"])))*100, 3))
    #Temporal(ZSTD):
    framedata[1].append(round((1-((bit_comp_quant["Temporal(ZSTD):"]+bit_comp_scale["Temporal(ZSTD):"])/(bit_comp_quant["Original Size(Bits):"]+bit_comp_scale["Original Size(Bits):"])))*100, 3))
    #Temporal(LZ4):
    framedata[1].append(round((1-((bit_comp_quant["Temporal(LZ4):"]+bit_comp_scale["Temporal(LZ4):"])/(bit_comp_quant["Original Size(Bits):"]+bit_comp_scale["Original Size(Bits):"])))*100, 3))
    #Spatial(ZSTD):
    framedata[1].append(round((1-((bit_comp_quant["Spatial(ZSTD):"]+bit_comp_scale["Spatial(ZSTD):"])/(bit_comp_quant["Original Size(Bits):"]+bit_comp_scale["Original Size(Bits):"])))*100, 3))
    #Spatial(LZ4):
    framedata[1].append(round((1-((bit_comp_quant["Spatial(LZ4):"]+bit_comp_scale["Spatial(LZ4):"])/(bit_comp_quant["Original Size(Bits):"]+bit_comp_scale["Original Size(Bits):"])))*100, 3))
    #do the others later
    df = pd.DataFrame(np.array(framedata))
    df.columns = ['Global(ZSTD):','Global(LZ4):', 'Temporal(ZSTD):', 'Temporal(LZ4):', 'Spatial(ZSTD):', 'Spatial(LZ4):']
    df.index = ['vs Original Data Size:', 'vs Quantized+Scalar:', 'vs Direct Compression:', 'vs Quant+Scale Compressed (No BP):'] #rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    print(df)
    print("\n Extra: Uncompressed Bitplaned Quant+Scalar vs Uncompressed Bitplaned Original Tensor: " + str(round((1-((bit_comp_quant["Original Size(Bits):"]+bit_comp_scale["Original Size(Bits):"])/Raw_size))*100, 3)))

def check_tensor_entropy(tensor_d):
    import torch
    # Expects shape (N, B, W)
    N, B, W = tensor_d.shape
    print(f"\n--- BITPLANE ENTROPY CHECK (Shape: {N}x{B}x{W}) ---")
    print(f"{'Plane':<10} | {'Mean (Bias)':<15} | {'Status'}")
    print("-" * 45)

    for k in range(B):
        # Isolate plane k and calculate mean
        # .float() is necessary because uint8 means will truncate
        bias = tensor_d[:, k, :].to(torch.float32).mean().item()
        
        # Determine status
        if 0.48 <= bias <= 0.52:
            status = "Random (Incompressible)"
        elif bias < 0.1 or bias > 0.9:
            status = "Highly Structured"
        else:
            status = "Partial Structure"

        # Only print the first few, middle, and last few to avoid spam
        if k < 4 or k > B - 5 or k == B // 2:
            print(f"Bitplane {k:<2} | {bias:<15.4f} | {status}")
        elif k == 4:
            print("...")

#further analytics
def quantization_histogram(quant, mode):
    import torch
    import numpy
    import matplotlib.pyplot as plt
    # 4. Move to CPU for plotting
    quantized_cpu = quant.cpu().numpy().flatten()
    
    global QUANTIZATION_TYPE
    bitsize = 0
    if(QUANTIZATION_TYPE.is_floating_point):
        bitsize += torch.finfo(QUANTIZATION_TYPE).bits
    else:
        bitsize += torch.iinfo(QUANTIZATION_TYPE).bits

    # 5. Plotting the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(quantized_cpu, bins=int((2**bitsize)-1), range=(-1*int((2**bitsize)/2), int((2**bitsize)/2)), color='#4682B4', edgecolor='black', alpha=0.7)
    plt.title('Distribution of ' + str(QUANTIZATION_TYPE) + ' Quantized Vector Embeddings (' + mode + ')')
    plt.xlabel('Quantized Integer Value')
    plt.ylabel('Frequency (Total Elements)')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig('quantized_distribution[' + mode + ', ' + str(QUANTIZATION_TYPE) + '].png')
    
def plot_embedding_histogram(tensor,filepath): #gets the histogram of the range of input data from vector embedding
    import matplotlib.pyplot as plt
    flat_data = tensor.cpu().numpy().flatten()
    plt.hist(flat_data, bins=100, color='blue', alpha=0.7)
    plt.title("Range of Vector Embedding Input Data: [path=" + str(filepath) + "]")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    #plt.show()
    plt.savefig('EmbeddingRange.png')

#main control logic flow
def initialization():
    #lets try starting with autoscan
    global base_directory
    print(base_directory)
    
    #try the autoscan
    #local relative file autoscan, scans nearby folders for .parquet files
    from pathlib import Path
    files = list(Path('.').rglob('*.*'))
    print("vector embedding files found, select via number: ")
    ind = 0
    while(ind < len(files)):
        print(str(ind) + ".) " + str(files[ind]))
        ind += 1
    print("\nNOTE: You must make sure that the file you are choosing is a pre-computed non-compressed vector embedding of numeric values, (or else the program will fail)\n")
    ms = int(input("Input path number here: "))
    testfile = files[ms]
    #join it together
    global file_path
    file_path = os.path.join(base_directory, testfile)
    #print(file_path)
    result = extract_vectors(file_path)
    if(PRINT_ORIGINAL == True):
        print("\nORIGINAL VECTOR EMBEDDING DATA: " + str(result.shape))
        print(result)
    #now adjust the NUM_DIMS and MAX_ROWS parameters, will be useful later
    global MAX_ROWS
    global NUM_DIMS
    MAX_ROWS, NUM_DIMS = result.shape
    if(PRINT_ORIGINAL == True):
        print("MAX_ROWS: " + str(MAX_ROWS))
        print("NUM_DIMS: " + str(NUM_DIMS))
    import torch
    import numpy
    global BASE_TYPE
    BASE_TYPE = result.dtype
    if(PRINT_ORIGINAL == True):
        print("BASE_TYPE: " + str(BASE_TYPE))
    #get it into a tensor
    global ACCELERATION_DEVICE
    res = result[:NUM_ROWS] #reduces it to your specified number of rows to analyze
    numpy_emb = numpy.stack(res)
    tens_emb = torch.tensor(numpy_emb).to(ACCELERATION_DEVICE) #NOTE: we will keep this for later when we do unquantized comparisons to the quantized stuff
    #now the range of input from the file
    if(PRINT_ORIGINAL == True):
        print("\nFIGURE 1.0: Range of input tensor: " + str(file_path) + " [stored on a file called EmbeddingRange.png]")
    raw_emb = torch.tensor(numpy_emb).to(ACCELERATION_DEVICE)
    plot_embedding_histogram(raw_emb,file_path)
    #now symmetric scalar quantization
    #row wise
    if(PRINT_QUANT == True):
        print("\nROW Quantization: ")
    row_quant_raw, row_scale_raw = symmetric_scalar_quantization(tens_emb,"row")
    if(PRINT_QUANT == True):
        print("QUANT: " + str(row_quant_raw.shape))
        print(row_quant_raw)
        print("SCALE: " + str(row_scale_raw.shape))
        print(row_scale_raw)
        row_dequant_raw = symmetric_scalar_dequantization(row_quant_raw, row_scale_raw)
        print("DEQUANT: " + str(row_dequant_raw.shape))
        print(row_dequant_raw)
        print("\nFIGURE 1.1: Row-wise Symmetric Scalar Quantization Error: ")
        row_error = Symmetric_Scalar_Error(tens_emb, row_dequant_raw,"row")
        #figure 1.2 is a histogram of quantized values, it is only saved as a .png
        row_quant_raw = row_quant_raw.to(ACCELERATION_DEVICE)
        quantization_histogram(row_quant_raw, "row")
    row_quant_raw = row_quant_raw.to(ACCELERATION_DEVICE)
    
    #col wise
    if(PRINT_QUANT == True):
        print("\nCOL Quantization: ")
    col_quant_raw, col_scale_raw = symmetric_scalar_quantization(tens_emb,"col")
    if(PRINT_QUANT == True):
        print("QUANT: " + str(col_quant_raw.shape))
        print(col_quant_raw)
        print("SCALE: " + str(col_scale_raw.shape))
        print(col_scale_raw)
        col_dequant_raw = symmetric_scalar_dequantization(col_quant_raw, col_scale_raw)
        print("DEQUANT: " + str(col_dequant_raw.shape))
        print(col_dequant_raw)
        print("\nFIGURE 1.3: Column-wise Symmetric Scalar Quantization Error: ")
        col_error = Symmetric_Scalar_Error(tens_emb, col_dequant_raw,"col")
        #figure 1.4 is a histogram of quantized values, it is only saved as a .png
        col_quant_raw = col_quant_raw.to(ACCELERATION_DEVICE)
        quantization_histogram(col_quant_raw, "col")
    col_quant_raw = col_quant_raw.to(ACCELERATION_DEVICE)
    
    #now next step, is to do binary string conversions
    #then, lets get the bitplanes for both vertical and horizontal
    
    #horizontal_row
    if(SHOW_EXTRANEOUS_RESULTS == True):
        #get bitplanes
        horizontal_bitplane_quant_row, horizontal_bitplane_scale_row = bitplane(row_quant_raw, row_scale_raw, "horizontal")
        print("\nDEBUG: horizontal_bitplane_quant_row: " + str(horizontal_bitplane_quant_row.shape))
        #RLE
        if(PRINT_RLE == True):
            quant_horizontal_row = RLE_bitplane_compressibility(horizontal_bitplane_quant_row, "horizontal", "horizontal_bitplane_quant_row")
        #LZ4 + ZSTD
        if(PRINT_COMP == True):
            comp_horizontal_bitplane_quant_row = run_phased_benchmark(horizontal_bitplane_quant_row, "horizontal", "quant")
        #Entropy
        print("\nDEBUG: horizontal_bitplane_scale_row: " + str(horizontal_bitplane_scale_row.shape))
        #RLE
        if(PRINT_RLE == True):
            #the RLE below isn't surprising at all, since each bitplane here only has 1 element
            scale_horizontal_row = RLE_bitplane_compressibility(horizontal_bitplane_scale_row, "horizontal", "horizontal_bitplane_scale_row")
        #LZ4 + ZSTD
        if(PRINT_COMP == True):
            comp_horizontal_bitplane_scale_row = run_phased_benchmark(horizontal_bitplane_scale_row, "horizontal", "scale")
        #Entropy
        
        #overall RLE
        if(PRINT_RLE == True):
            print("\nDEBUG: horizontal_row_ratio_overall: " + str(calculate_overall_RLE_ratio(horizontal_bitplane_quant_row, horizontal_bitplane_scale_row, quant_horizontal_row, scale_horizontal_row)))
    
    #vertical_row
    #get bitplanes
    vertical_bitplane_quant_row, vertical_bitplane_scale_row = bitplane(row_quant_raw, row_scale_raw, "vertical")
    print("\nDEBUG: vertical_bitplane_quant_row: " + str(vertical_bitplane_quant_row.shape))
    #RLE
    #RLE below seems interesting
    if(PRINT_RLE == True):
        quant_vertical_row = RLE_bitplane_compressibility(vertical_bitplane_quant_row, "vertical", "vertical_bitplane_quant_row")
    #LZ4 + ZSTD
    if(PRINT_COMP == True):
        comp_vertical_bitplane_quant_row = run_phased_benchmark(vertical_bitplane_quant_row, "vertical", "quant")
    #Entropy
    print("\nDEBUG: vertical_bitplane_scale_row: " + str(vertical_bitplane_scale_row.shape))
    #RLE
    if(PRINT_RLE == True):
        scale_vertical_row = RLE_bitplane_compressibility(vertical_bitplane_scale_row, "vertical", "vertical_bitplane_scale_row")
    #LZ4 + ZSTD
    if(PRINT_COMP == True):
        comp_vertical_bitplane_scale_row = run_phased_benchmark(vertical_bitplane_scale_row, "vertical", "scale")
    #Entropy
    
    #Overall
    if(PRINT_RLE == True):
        print("\nDEBUG: vertical_row_ratio_overall: " + str(calculate_overall_RLE_ratio(vertical_bitplane_quant_row, vertical_bitplane_scale_row, quant_vertical_row, scale_vertical_row)))
    
    #horizontal_col
    #get bitplanes
    horizontal_bitplane_quant_col, horizontal_bitplane_scale_col = bitplane(col_quant_raw, col_scale_raw, "horizontal")
    print("\nDEBUG: horizontal_bitplane_quant_col: " + str(horizontal_bitplane_quant_col.shape))
    #RLE
    #very bad compression on RLE below
    if(PRINT_RLE == True):
        quant_horizontal_col = RLE_bitplane_compressibility(horizontal_bitplane_quant_col, "horizontal", "horizontal_bitplane_quant_col")
    #LZ4 + ZSTD
    if(PRINT_COMP == True):
        comp_horizontal_bitplane_quant_col = run_phased_benchmark(horizontal_bitplane_quant_col, "horizontal", "quant")
    #Entropy
    print("\nDEBUG: horizontal_bitplane_scale_col: " + str(horizontal_bitplane_scale_col.shape))
    #RLE
    #RLE below reasonable compression on first few bits, then almost no compression on rest
    if(PRINT_RLE == True):
        scale_horizontal_col = RLE_bitplane_compressibility(horizontal_bitplane_scale_col, "horizontal", "horizontal_bitplane_scale_col")
    #LZ4 + ZSTD
    if(PRINT_COMP == True):
        comp_horizontal_bitplane_scale_col = run_phased_benchmark(horizontal_bitplane_scale_col, "horizontal", "scale")
    #Entropy
    
    #Overall
    if(PRINT_RLE == True):
        print("\nDEBUG: horizontal_col_ratio_overall: " + str(calculate_overall_RLE_ratio(horizontal_bitplane_quant_col, horizontal_bitplane_scale_col, quant_horizontal_col, scale_horizontal_col)))
    
    #vertical_col
    if(SHOW_EXTRANEOUS_RESULTS == True):
        #get bitplanes
        vertical_bitplane_quant_col, vertical_bitplane_scale_col = bitplane(col_quant_raw, col_scale_raw, "vertical")
        print("\nDEBUG: vertical_bitplane_quant_col: " + str(vertical_bitplane_quant_col.shape))
        #RLE
        #meh RLE compressibility below
        if(PRINT_RLE == True):
            quant_vertical_col = RLE_bitplane_compressibility(vertical_bitplane_quant_col, "vertical", "vertical_bitplane_quant_col")
        #LZ4 + ZSTD
        if(PRINT_COMP == True):
            comp_vertical_bitplane_quant_col = run_phased_benchmark(vertical_bitplane_quant_col, "vertical", "quant")
        #Entropy
        
        #Overall
        print("\nDEBUG: vertical_bitplane_scale_col: " + str(vertical_bitplane_scale_col.shape))
        #RLE
        #For RLE below, not surprising since each bitplane only has 1 element
        if(PRINT_RLE == True):
            scale_vertical_col = RLE_bitplane_compressibility(vertical_bitplane_scale_col, "vertical", "vertical_bitplane_scale_col")
        #LZ4 + ZSTD
        if(PRINT_COMP == True):
            comp_vertical_bitplane_scale_col = run_phased_benchmark(vertical_bitplane_scale_col, "vertical", "scale")
        #Entropy
        
        #Overall
        if(PRINT_RLE == True):
            print("\nDEBUG: vertical_col_ratio_overall: " + str(calculate_overall_RLE_ratio(vertical_bitplane_quant_col, vertical_bitplane_scale_col, quant_vertical_col, scale_vertical_col)))
    
    #try also getting the bitplanes for non quantized too
    if(IGNORE_RAW == False):
        #horizontal_raw
        #get bitplanes
        horizontal_bitplane_raw, vertical_bitplane_raw = direct_bitplane(tens_emb) #no quantization
        print("\nDEBUG: vertical_bitplane_raw: " + str(vertical_bitplane_raw.shape))
        #RLE
        #RLE results below are surprisingly very compressible
        if(PRINT_RLE == True):
            raw_vertical = RLE_bitplane_compressibility(vertical_bitplane_raw, "vertical", "vertical_bitplane_raw")
        #LZ4 + ZSTD
        if(PRINT_COMP == True):
            comp_vertical_bitplane_raw = run_phased_benchmark(vertical_bitplane_raw, "vertical", "direct")
        #Entropy
    
        print("\nDEBUG: horizontal_bitplane_raw: " + str(horizontal_bitplane_raw.shape))
        #RLE
        #RLE results below have mixed results of compressibility
        if(PRINT_RLE == True):
            raw_horizontal = RLE_bitplane_compressibility(horizontal_bitplane_raw, "horizontal", "horizontal_bitplane_raw")
        #LZ4 + ZSTD
        if(PRINT_COMP == True):
            comp_horizontal_bitplane_raw = run_phased_benchmark(horizontal_bitplane_raw, "horizontal", "direct")
        #Entropy
    
    #first, lets get the total bit size of the original input tensor
    #tens_emb
    bit_rep = 0
    if(tens_emb.dtype.is_floating_point == True):
        bit_rep += torch.finfo(tens_emb.dtype).bits
    else:
        bit_rep += torch.iinfo(tens_emb.dtype).bits
    raw_size = tens_emb.shape[0] * tens_emb.shape[1] * bit_rep #full raw size of original tensor in bits
    #now, for space-saving-metrics
    if(SHOW_EXTRANEOUS_RESULTS == True):
        print("\nSpace Saving Metrics: Horizontal_Row (Quant+Scale): ")
        space_saving_metrics(comp_horizontal_bitplane_quant_row, comp_horizontal_bitplane_scale_row, raw_size) #for horizontal_row
    
    print("\nSpace Saving Metrics: Vertical_Row (Quant+Scale): ")
    space_saving_metrics(comp_vertical_bitplane_quant_row, comp_vertical_bitplane_scale_row, raw_size) #for vertical_row (most promising, across all vector embeddings)
    
    print("\nSpace Saving Metrics: Horizontal_Col (Quant+Scale): ")
    space_saving_metrics(comp_horizontal_bitplane_quant_col, comp_horizontal_bitplane_scale_col, raw_size) #for horizontal_col (less promising, across each vector embedding for all vector embeddings)
    
    if(SHOW_EXTRANEOUS_RESULTS == True):
        print("\nSpace Saving Metrics: Vertical_Col (Quant+Scale): ")
        space_saving_metrics(comp_vertical_bitplane_quant_col, comp_vertical_bitplane_scale_col, raw_size) #for vertical_col
    
    if(IGNORE_RAW == False):
        print("\nSpace Saving Metrics: Vertical (Raw): ")
        space_saving_metrics(comp_vertical_bitplane_raw, comp_vertical_bitplane_raw, raw_size) #for vertical_raw
    
        print("\nSpace Saving Metrics: Horizontal (Raw): ")
        space_saving_metrics(comp_horizontal_bitplane_raw, comp_horizontal_bitplane_raw, raw_size) #for horizontal_raw
    
if __name__ == '__main__':
    initialization() #call it, starts the simulation
    #to do 3/14+/26:
    # - more metrics:
    # --> Direct file compression
    # --> metric: vs Direct Compression
    # --> metric: vs Quant+Scale Compressed (No Bitplaning)
    # --> (optional) avg compression ratio for global, temporal, spatial (for each LZ4 and ZSTD) [averages across all bitplanes]
    # --> (optional) actual compression ratio for global, temporal, spatial (for each LZ4 and ZSTD) [gets sizes of all compressed bitplanes, adds it up, divides by total quantized size of before compression]
    # --> (optional) The previous 2 metrics, but with scalar + quantized
    # --> (optional) The previous 2 metrics, but compared to pre-quantized tensor
    # --> (optional) Raw/Direct versions of these metrics
    
    #Some of the lower priority tasks:
    # - adding explantions
    # - direct compression
    # - adding options for randomized file selection
    # - adding options for randomized starting row selection
    # - [basically done?, only tested working on .parquet and .npy, havent tested uncompressed jsonl or csv yet] Finish working on the vector embedding extraction (heuristic?) [NOTE: don't do jsonl.zst stuff, Rui said]
    # - maybe direct compression and block size heuristics