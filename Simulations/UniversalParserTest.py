#We should be able to handle these file types:
#

import os
base_directory = os.path.dirname(os.path.abspath(__file__))
import torch
file_path = ""
NUM_ROWS = 512
NUM_DIMS = 768 #768 is default, but will automatically change with shape recognition
MAX_ROWS = 10000 #10k is default, but will automatically change with shape recognition

#quantization settings
BASE_TYPE = torch.float32 #torch.float32 is default, will autochange during shape detection
QUANTIZATION_TYPE = torch.int8 #You change this value, this is the resulting quantized values
SCALE_TYPE = torch.float32 #torch.float32 is detault, will autochange during quantization

#some LZ4 and ZSTD Compression Settings
BLOCK_SIZE = 4096 #default - 4096 = 4KB

#performance settings
ACCELERATION_DEVICE = "xpu"

#debug settings
PRINT_DEBUG = False
SHOW_EXTRANEOUS_RESULTS = True #default false to improve performance

#don't worry about these
qmat = 0
smat = 0

#some functions

#raw vector tensor retrieval from files
#This one is also vibe coded, be careful, will modify if needed
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

#this one is vibe coded, be careful, modify it too

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

def rowbinary_quant(row, qmat):
    return [format(x, f'0{qmat}b') for x in row] #only works in python 3.6 and newer

def rowbinary_scale(row, smat):
    return [format(x, f'0{smat}b') for x in row] #only works in python 3.6 and newer

def set_bitplane_precision(): #sets the precision for bitstring conversion of quantized and scaled values (must use before converting to bitstring)
    global QUANTIZATION_TYPE
    global qmat
    if(QUANTIZATION_TYPE.is_floating_point):
        qmat = int(torch.finfo(QUANTIZATION_TYPE).bits)
    else:
        qmat = int(torch.iinfo(QUANTIZATION_TYPE).bits)
    global SCALE_TYPE
    global smat
    smat = int(torch.finfo(SCALE_TYPE).bits)
    #print("\nDEBUG: QMAT: ")
    #print(qmat)
    #print("\nDEBUG SMAT: ")
    #print(smat)

def binarytolist(element):
    return [int(x) for x in element]

def binaryvectortolist(row):
    return [[int(x) for x in y] for y in row]

def get_unsigned_dtype(basetype):
    import torch
    #import torchao
    mapping = {torch.int4: torch.uint4, torch.int8: torch.uint8, torch.int16: torch.uint16, torch.int32: torch.uint32, torch.int64: torch.uint64, torch.bfloat16: torch.uint16, torch.float16: torch.uint16, torch.float32: torch.uint32, torch.float64: torch.uint64}
    bits = 0
    default_return_type = torch.uint8 #default return type is autoadjusted depending on basetype, and will return if there is no mapping for it
    #the code below determines the number of bits of the basetype so we can choose the default return type if it fails to map, it also returns proper mapping if found
    if(basetype.is_floating_point):
        bits += torch.finfo(basetype).bits
    else:
        bits += torch.iinfo(basetype).bits
    if(bits == 8):
        return mapping.get(basetype, torch.uint8)
    elif(bits == 16):
        return mapping.get(basetype, torch.uint16)
    elif(bits == 32):
        return mapping.get(basetype, torch.uint32)
    return mapping.get(basetype, default_return_type)

def direct_bitplane(matr): #this is for non-quantized bitplanes, returns a pair, one is vertical and the other is horizontal
    import torch
    from multiprocessing import Pool
    import numpy
    #lets get the formatting first
    vmode = get_unsigned_dtype(matr.dtype)
    bit_gpu = matr.view(vmode)
    bit_cpu = bit_gpu.cpu()
    #print("\nDEBUG: direct bit_cpu: ")
    #print(bit_cpu)
    #get dtype into bits form
    currtype = matr.dtype
    bits = 0
    if(currtype.is_floating_point):
        bits += torch.finfo(currtype).bits
    else:
        bits += torch.iinfo(currtype).bits
    #now continue the formatting into binstrings
    binstring = []
    with Pool() as p:
        binstring = p.starmap(rowbinary_quant, [(x, bits) for x in bit_cpu])
    #print("\nDEBUG: binstring: ")
    #print(binstring)
    #then format them into lists then tensors to send to device
    bit_split = Pool().map(binaryvectortolist, binstring)
    #print("\nDEBUG: bit_split: ")
    #print(bit_split)
    bitplane_horizontal = torch.tensor(bit_split).to(ACCELERATION_DEVICE)
    #perform bitplane rearrangement
    bitplane_h = bitplane_horizontal.permute(2,0,1)
    #print("\nDEBUG: bitplane_h direct: ")
    #print(bitplane_h)
    bitplane_v = numpy.stack(bit_split, axis=-1)
    #print("\nDEBUG: bitplane_v direct: ")
    #print(bitplane_v)
    #now return the values
    return bitplane_h, bitplane_v

def bitplane(quant, scale, direc): #converts it into a binstring first, then converts it to bitplanes
    import torch
    from multiprocessing import Pool
    global QUANTIZATION_TYPE
    global SCALE_TYPE
    import numpy
    
    #1. convert both quantized and scale to binary strings (Accelerated on CPU multithreading)
    quant_view_type = get_unsigned_dtype(QUANTIZATION_TYPE)
    bit_quant_gpu = quant.view(quant_view_type)
    bit_quant_cpu = bit_quant_gpu.cpu()
    binstring_quant = []
    with Pool() as p:
        binstring_quant = p.starmap(rowbinary_quant, [(x, qmat) for x in bit_quant_cpu])
    #print("\nDEBUG: binstring_quant: ")
    #print(binstring_quant)
    
    scale_view_type = get_unsigned_dtype(SCALE_TYPE)
    bit_scale_gpu = scale.view(scale_view_type)
    bit_scale_cpu = bit_scale_gpu.cpu()
    binstring_scale = []
    with Pool() as p:
        binstring_scale = p.starmap(rowbinary_scale, [(x, smat) for x in bit_scale_cpu])
    
    #1.5. get the binary split for both quantized values and scalars
    bit_split_quantized = Pool().map(binaryvectortolist, binstring_quant)
    bit_split_scalars = Pool().map(binaryvectortolist, binstring_scale)
    #2. convert to bitplanes (vertical and horizontal)
    
    #horizontal (single and multivector) [originally was single vector only]
    #print("\nDEBUG: bit_split_quantized: ")
    #print(bit_split_quantized)
    if(direc == "horizontal"):
        bitplane_tens_quant_horizontal = torch.tensor(bit_split_quantized)
        bitplane_matr_quant_horizontal = bitplane_tens_quant_horizontal.to(ACCELERATION_DEVICE)
        #bitplanes_quantized_horizontal = bitplane_matr_quant_horizontal.T
        bitplanes_quantized_horizontal = bitplane_matr_quant_horizontal.permute(2,0,1)
        #print("\nDEBUG: bitplanes_quantized_horizontal: " + str(bitplanes_quantized_horizontal.shape))
        #print(bitplanes_quantized_horizontal)
    
        bitplane_tens_scale_horizontal = torch.tensor(bit_split_scalars)
        bitplane_matr_scale_horizontal = bitplane_tens_scale_horizontal.to(ACCELERATION_DEVICE)
        #bitplanes_scalars_horizontal = bitplane_matr_scale_horizontal.T
        bitplanes_scalars_horizontal = bitplane_matr_scale_horizontal.permute(2,0,1)
        
        #return
        return bitplanes_quantized_horizontal, bitplanes_scalars_horizontal
    
    #vertical (multivector only)
    elif(direc == "vertical"):
        bitplanes_quantized_vertical = numpy.stack(bit_split_quantized, axis=-1)
        bitplanes_scalars_vertical = numpy.stack(bit_split_scalars, axis=-1)
        
        #return
        return bitplanes_quantized_vertical, bitplanes_scalars_vertical

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
    print("\nDEBUG: " + str(title) + " Overall results (RLE): " + str(ratio.shape))
    print(ratio)
    #return ratio?
    if(form == "horizontal"):
        bp_avg = torch.mean(ratio, dim=1) #per bitplane
        print("\nHorizontal Per Bitplane: ")
        print(bp_avg)
        row_avg = torch.mean(ratio, dim=0) #per vector embedding
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
    pass

#ported over LZ4+ZSTD Compression analysis

def lz4_compress_list(data_list):
    import lz4.frame
    return sum(len(lz4.frame.compress(a, block_size=BLOCK_SIZE)) for a in data_list)

def zstd_compress_list(data_list):
    import zstandard as zstd
    """Compresses a list of byte-buffers and returns the total compressed size."""
    c = zstd.ZstdCompressor(level=3)
    return sum(len(c.compress(b)) for b in data_list)

def pmark(tensor_d):
    import torch
    import multiprocessing as mp
    import zstandard as zstd
    import pandas as pd
    D, B, N = tensor_d.shape
    pass

#vibe coded pack_bits, modify and customize and verify it
def pack_bits(tensor_r, dim): #NOTE: both of these modes are confirmed correct and working. Keep in mind it does 0 padding for N and/or B if they aren't a multiple of 8, which might inflate compression ratios, but this is done to some degree on real devices anyways
    """
    Packs a binary tensor (0, 1) into uint8 bytes along the N (0) or B (2) dimension.
    Input Shape: (N, W, B)
    """
    import torch
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
    else:
        raise ValueError("Dimension for pack_bits must be either 0 or 2...")

# Example for your (N, W, B) tensor:
# packed_N = pack_bits(tensor, dim=0) # Packs rows into bytes
# packed_B = pack_bits(tensor, dim=2) # Packs bitplanes into bytes

#vibe modded version of previous code, modify it further
def run_phased_benchmark(tensor_d, form, variant):
    import torch
    import multiprocessing
    import zstandard as zstd
    import pandas as pd
    
    #First, get the dimensions/shape of the incoming tensor
    N, W, B = tensor_d.shape
    #where:
    # - N = height (typically the number of rows)
    # - W = width (typically dim of vector embedding)
    # - B = depth (typically the length of bitplane or number of bits in msb)
    
    #print("\nDEBUG: packed input tensor: " + str(tensor_d.shape) + " [Original Bitplanes]: ")
    #print(tensor_d)
    packed_n = pack_bits(tensor_d, dim=0)#for spatial
    print("\nDEBUG: packed_n tensor: " + str(packed_n.shape) + " [for Spatial]: ")
    print(packed_n)
    packed_b = pack_bits(tensor_d, dim=2) #for global and temporal
    print("\nDEBUG: packed_b tensor: " + str(packed_b.shape) + " [for Global and Temporal]: ")
    print(packed_b)
    pass

def run_phased_benchmark0(tensor_d, form, variant):
    import torch
    import multiprocessing as mp
    import zstandard as zstd
    import pandas as pd
    
    # D=768, B=8 (Bitplanes), N=Depth
    D, B, N = tensor_d.shape
    
    # 1. XPU PACKING PHASE
    tensor_dbn = torch.from_numpy(tensor_d).to(ACCELERATION_DEVICE)
    
    # FIX: Use int64 for math to avoid NotImplementedError: lshift_xpu
    # We pack N into bytes for Global/Temporal
    reshaped_n = tensor_dbn.view(D, B, N // 8, 8).to(torch.int64)
    packed_n = (reshaped_n[..., 0] << 7) | (reshaped_n[..., 1] << 6) | \
               (reshaped_n[..., 2] << 5) | (reshaped_n[..., 3] << 4) | \
               (reshaped_n[..., 4] << 3) | (reshaped_n[..., 5] << 2) | \
               (reshaped_n[..., 6] << 1) | (reshaped_n[..., 7] << 0)
    
    # NEW: Pack D into bytes for Spatial analysis (to measure feature correlation)
    # We permute so D is at the end, then pack D into D//8
    reshaped_d = tensor_dbn.permute(2, 1, 0).reshape(N, B, (D // 8)+(D % 8), 8).to(torch.int64)
    packed_d = (reshaped_d[..., 0] << 7) | (reshaped_d[..., 1] << 6) | \
               (reshaped_d[..., 2] << 5) | (reshaped_d[..., 3] << 4) | \
               (reshaped_d[..., 4] << 3) | (reshaped_d[..., 5] << 2) | \
               (reshaped_d[..., 6] << 1) | (reshaped_d[..., 7] << 0)

    # Move to CPU and fix orientation [N, W, B]
    #n: (N,W,B//8) --> 
    # brick_n: (D, B, N//8) -> we want (B, D, N//8) for your loops
    brick_n = packed_n.permute(1, 0, 2).to(torch.uint8).cpu().numpy()
    # brick_d: (N, B, D//8) -> we want (B, N, D//8)
    brick_d = packed_d.permute(1, 0, 2).to(torch.uint8).cpu().numpy()

    original_size_kb = (D * N / 8) / 1024
    num_bitplanes = B
    results = {k: {"Bitplane": k} for k in range(num_bitplanes)}
    
    cores = min(num_bitplanes, mp.cpu_count())

    with mp.Pool(processes=cores) as pool:
        # --- PHASE 1 & 2: Use brick_n (Packed along N) ---
        print("Running Global & Temporal Analysis...")
        global_tasks = [[brick_n[k].tobytes()] for k in range(num_bitplanes)]
        temporal_tasks = [[brick_n[k, i].tobytes() for i in range(D)] for k in range(num_bitplanes)]
        
        # ZSTD Mapping
        for k, size in enumerate(pool.map(zstd_compress_list, global_tasks)):
            results[k]["Global(ZSTD)"] = round((size / 1024) / original_size_kb, 3)
        for k, size in enumerate(pool.map(zstd_compress_list, temporal_tasks)):
            results[k]["Temporal(ZSTD)"] = round((size / 1024) / original_size_kb, 3)

        # --- PHASE 3: Use brick_d (Packed along D) ---
        print("Running Spatial Analysis...")
        # Now each task is N "layers", where each layer is a D/8 byte-string
        spatial_tasks = [[brick_d[k, j].tobytes() for j in range(N)] for k in range(num_bitplanes)]
        
        for k, size in enumerate(pool.map(zstd_compress_list, spatial_tasks)):
            results[k]["Spatial(ZSTD)"] = round((size / 1024) / original_size_kb, 3)
            
        # ... Repeat for LZ4 ...

    df = pd.DataFrame(list(results.values())).sort_values("Bitplane", ascending=False)
    print(df)
    return df

def run_phased_benchmark2(tensor_r, form, variant): #results: < 1 is good
    import torch
    import multiprocessing as mp
    import zstandard as zstd
    import pandas as pd
    import numpy
    #account for tensor vs ndarray
    tensor_d = tensor_r
    if(torch.is_tensor(tensor_r)):
        tensor_d = numpy.array(tensor_r)
    D, B, N = tensor_d.shape
    #account for variants: quant, scale, direct
    shapetype = torch.uint8
    if(variant == "quant"):
        shapetype = get_unsigned_dtype(QUANTIZATION_TYPE)
    elif(variant == "scale"):
        shapetype = get_unsigned_dtype(SCALE_TYPE)
    elif(variant == "direct"):
        shapetype = get_unsigned_dtype(BASE_TYPE)
    shapetype = torch.int64
    # 1. PACKING PHASE (XPU)
    # Collapse N (depth) into N//8 bytes
    tensor_dbn = torch.from_numpy(tensor_d).to(ACCELERATION_DEVICE)
    reshaped = tensor_dbn.view(D, B, N // 8, 8).to(shapetype)
    packed = (reshaped[..., 0] << 7) | (reshaped[..., 1] << 6) | \
             (reshaped[..., 2] << 5) | (reshaped[..., 3] << 4) | \
             (reshaped[..., 4] << 3) | (reshaped[..., 5] << 2) | \
             (reshaped[..., 6] << 1) | (reshaped[..., 7] << 0)
    
    # Move to CPU: Shape (Bitplane=8, Dim=768, Packed_Depth=N/8)
    brick = packed.cpu().numpy()
    if(form == "vertical"):
        brick = packed.permute(1, 0, 2).cpu().numpy()
    original_size_kb = (D * N / 8) / 1024
    num_bitplanes = brick.shape[0]
    results = {k: {"Bitplane": k} for k in range(num_bitplanes)}
    cores = 8
    if(num_bitplanes <= mp.cpu_count()): #auto core detection for cpu
        cores = num_bitplanes #assigns as bitplanes to each cpu core (within the limits of your core/thread count)

    with mp.Pool(processes=cores) as pool:
        # --- PHASE 1: GLOBAL WALL (8 planes in parallel) ---
        print("Running Global Wall Analysis...")
        global_buffers = [brick[k].tobytes() for k in range(num_bitplanes)]
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
        temporal_tasks = [[brick[k, i].tobytes() for i in range(D)] for k in range(num_bitplanes)]
        t_sizes = pool.map(zstd_compress_list, temporal_tasks)
        for k, size in enumerate(t_sizes):
            results[k]["Temporal(ZSTD)"] = round((size / 1024) / original_size_kb, 3)
        t4_sizes = pool.map(lz4_compress_list, temporal_tasks)
        for k, size in enumerate(t4_sizes):
            results[k]["Temporal(LZ4)"] = round((size / 1024) / original_size_kb, 3)

        # --- PHASE 3: SPATIAL COLUMNS (8 planes in parallel) ---
        print("Running Spatial Column Analysis...")
        # We prepare a list of lists: 8 bitplanes, each containing N/8 strings
        spatial_tasks = [[brick[k, :, j].tobytes() for j in range(N // 8)] for k in range(num_bitplanes)]
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

#ported over entropy compression analysis

#ported over direct compression analysis

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
    print(file_path)
    result = extract_vectors(file_path)
    print("\nORIGINAL VECTOR EMBEDDING DATA: " + str(result.shape))
    print(result)
    #now adjust the NUM_DIMS and MAX_ROWS parameters, will be useful later
    global MAX_ROWS
    global NUM_DIMS
    MAX_ROWS, NUM_DIMS = result.shape
    print("MAX_ROWS: " + str(MAX_ROWS))
    print("NUM_DIMS: " + str(NUM_DIMS))
    import torch
    import numpy
    global BASE_TYPE
    BASE_TYPE = result.dtype
    print("BASE_TYPE: " + str(BASE_TYPE))
    #get it into a tensor
    global ACCELERATION_DEVICE
    res = result[:NUM_ROWS] #reduces it to your specified number of rows to analyze
    numpy_emb = numpy.stack(res)
    tens_emb = torch.tensor(numpy_emb).to(ACCELERATION_DEVICE) #NOTE: we will keep this for later when we do unquantized comparisons to the quantized stuff
    #now symmetric scalar quantization
    #row wise
    print("\nROW Quantization: ")
    row_quant_raw, row_scale_raw = symmetric_scalar_quantization(tens_emb,"row")
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
    print("\nCOL Quantization: ")
    col_quant_raw, col_scale_raw = symmetric_scalar_quantization(tens_emb,"col")
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
    #first, lets try to set the string formatting precisions
    set_bitplane_precision()
    #then, lets get the bitplanes for both vertical and horizontal
    
    #horizontal_row
    if(SHOW_EXTRANEOUS_RESULTS == True):
        #get bitplanes
        horizontal_bitplane_quant_row, horizontal_bitplane_scale_row = bitplane(row_quant_raw, row_scale_raw, "horizontal")
        #RLE
        print("\nDEBUG: horizontal_bitplane_quant_row: " + str(horizontal_bitplane_quant_row.shape))
        quant_horizontal_row = RLE_bitplane_compressibility(horizontal_bitplane_quant_row, "horizontal", "horizontal_bitplane_quant_row")
        #LZ4 + ZSTD
        run_phased_benchmark(horizontal_bitplane_quant_row, "horizontal", "quant")
        #Entropy
        
        #RLE
        print("\nDEBUG: horizontal_bitplane_scale_row: " + str(horizontal_bitplane_scale_row.shape))
        #the RLE below isn't surprising at all, since each bitplane here only has 1 element
        scale_horizontal_row = RLE_bitplane_compressibility(horizontal_bitplane_scale_row, "horizontal", "horizontal_bitplane_scale_row")
        #LZ4 + ZSTD
        run_phased_benchmark(horizontal_bitplane_scale_row, "horizontal", "scale")
        #Entropy
        
        #overall RLE
        print("\nDEBUG: horizontal_row_ratio_overall: " + str(calculate_overall_RLE_ratio(horizontal_bitplane_quant_row, horizontal_bitplane_scale_row, quant_horizontal_row, scale_horizontal_row)))
    
    #vertical_row
    #get bitplanes
    vertical_bitplane_quant_row, vertical_bitplane_scale_row = bitplane(row_quant_raw, row_scale_raw, "vertical")
    print("\nDEBUG: vertical_bitplane_quant_row: " + str(vertical_bitplane_quant_row.shape))
    #RLE
    #RLE below seems interesting
    quant_vertical_row = RLE_bitplane_compressibility(vertical_bitplane_quant_row, "vertical", "vertical_bitplane_quant_row")
    #LZ4 + ZSTD
    run_phased_benchmark(vertical_bitplane_quant_row, "vertical", "quant")
    #Entropy
    
    print("\nDEBUG: vertical_bitplane_scale_row: " + str(vertical_bitplane_scale_row.shape))
    #RLE
    scale_vertical_row = RLE_bitplane_compressibility(vertical_bitplane_scale_row, "vertical", "vertical_bitplane_scale_row")
    #LZ4 + ZSTD
    run_phased_benchmark(vertical_bitplane_scale_row, "vertical", "scale")
    #Entropy
    
    #Overall
    print("\nDEBUG: vertical_row_ratio_overall: " + str(calculate_overall_RLE_ratio(vertical_bitplane_quant_row, vertical_bitplane_scale_row, quant_vertical_row, scale_vertical_row)))
    
    #horizontal_col
    #get bitplanes
    horizontal_bitplane_quant_col, horizontal_bitplane_scale_col = bitplane(col_quant_raw, col_scale_raw, "horizontal")
    print("\nDEBUG: horizontal_bitplane_quant_col: " + str(horizontal_bitplane_quant_col.shape))
    #RLE
    #very bad compression on RLE below
    quant_horizontal_col = RLE_bitplane_compressibility(horizontal_bitplane_quant_col, "horizontal", "horizontal_bitplane_quant_col")
    #LZ4 + ZSTD
    run_phased_benchmark(horizontal_bitplane_quant_col, "horizontal", "quant")
    #Entropy
    
    print("\nDEBUG: horizontal_bitplane_scale_col: " + str(horizontal_bitplane_scale_col.shape))
    #RLE
    #RLE below reasonable compression on first few bits, then almost no compression on rest
    scale_horizontal_col = RLE_bitplane_compressibility(horizontal_bitplane_scale_col, "horizontal", "horizontal_bitplane_scale_col")
    #LZ4 + ZSTD
    run_phased_benchmark(horizontal_bitplane_scale_col, "horizontal", "scale")
    #Entropy
    
    #Overall
    print("\nDEBUG: horizontal_col_ratio_overall: " + str(calculate_overall_RLE_ratio(horizontal_bitplane_quant_col, horizontal_bitplane_scale_col, quant_horizontal_col, scale_horizontal_col)))
    
    #vertical_col
    if(SHOW_EXTRANEOUS_RESULTS == True):
        #get bitplanes
        vertical_bitplane_quant_col, vertical_bitplane_scale_col = bitplane(col_quant_raw, col_scale_raw, "vertical")
        print("\nDEBUG: vertical_bitplane_quant_col: " + str(vertical_bitplane_quant_col.shape))
        #RLE
        #meh RLE compressibility below
        quant_vertical_col = RLE_bitplane_compressibility(vertical_bitplane_quant_col, "vertical", "vertical_bitplane_quant_col")
        #LZ4 + ZSTD
        run_phased_benchmark(vertical_bitplane_quant_col, "vertical", "quant")
        #Entropy
        
        #Overall
        print("\nDEBUG: vertical_bitplane_scale_col: " + str(vertical_bitplane_scale_col.shape))
        #RLE
        #For RLE below, not surprising since each bitplane only has 1 element
        scale_vertical_col = RLE_bitplane_compressibility(vertical_bitplane_scale_col, "vertical", "vertical_bitplane_scale_col")
        #LZ4 + ZSTD
        run_phased_benchmark(vertical_bitplane_scale_col, "vertical", "scale")
        #Entropy
        
        #Overall
        print("\nDEBUG: vertical_col_ratio_overall: " + str(calculate_overall_RLE_ratio(vertical_bitplane_quant_col, vertical_bitplane_scale_col, quant_vertical_col, scale_vertical_col)))
    
    #try also getting the bitplanes for non quantized too
    #horizontal_raw
    #get bitplanes
    horizontal_bitplane_raw, vertical_bitplane_raw = direct_bitplane(tens_emb) #no quantization
    print("\nDEBUG: vertical_bitplane_raw: " + str(vertical_bitplane_raw.shape))
    #RLE
    #RLE results below are surprisingly very compressible
    RLE_bitplane_compressibility(vertical_bitplane_raw, "vertical", "vertical_bitplane_raw")
    #LZ4 + ZSTD
    run_phased_benchmark(vertical_bitplane_raw, "vertical", "direct")
    #Entropy
    
    print("\nDEBUG: horizontal_bitplane_raw: " + str(horizontal_bitplane_raw.shape))
    #RLE
    #RLE results below have mixed results of compressibility
    RLE_bitplane_compressibility(horizontal_bitplane_raw, "horizontal", "horizontal_bitplane_raw")
    #LZ4 + ZSTD
    run_phased_benchmark(horizontal_bitplane_raw, "horizontal", "direct")
    #Entropy
    
    #then perform RLE Compression analysis
    
    #then perform LZ4 and ZSTD Compression analysis
    
    #Then perform direct compression
    
if __name__ == '__main__':
    initialization() #call it
    #to do 2/20+/26:
    # - Highest priority: Port your LZ4+ZSTD simulation code that is already here with the pack_bits function //
    # //so you can start actually doing analysis
    # - Check the accuracy of this simulation vs previous simulation (and check if formatting and conversions and bitplane stuff is correct)
    #Some of the lower priority tasks:
    # - adding explantions
    # - custom block sizes for compression
    # - direct compression
    
    #To do 2_11+_26:
    # - [basically done?, only tested working on .parquet and .npy, havent tested uncompressed jsonl or csv yet] Finish working on the vector embedding extraction (heuristic?) [NOTE: don't do jsonl.zst stuff, Rui said]
    # - start preparing for shape formatting and bitplane stuff
    # - maybe direct compression and block size heuristics
    
    #To do specifically for 2_13_26:
    # - Set all scalars to FP32 (Rui wants that)
    # - Finish binary string conversions
    # - Try to finish binary string --> bitplane conversions (int --> uint etc, try to abstract them to functions if possible)
    # - maybe clean up code and comments? (some prints should probably be commented out for now)
    # - (LOOK AT PAGE 9 OF NOTES ON TABLET)
    pass