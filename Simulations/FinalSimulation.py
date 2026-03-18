#settings
#We should be able to handle these file types:
# - .parquet, .csv, .npy, etc
#Initial Setup
import os
base_directory = os.path.dirname(os.path.abspath(__file__))
import torch
file_path = "" #don't touch this unless you know what you are doing
NUM_ROWS = 1024 #This is for you to choose, it is the number of vector embeddings (essentially # of tokens)[must be a multiple of 8]
#Note: your NUM_ROWS should ideally exceed your BLOCK_SIZE, and for best results, should be a multiple of BLOCK_SIZE
NUM_DIMS = 4096 #768 is default, but will automatically change with shape recognition
MAX_ROWS = 10000 #10k is default, but will automatically change with shape recognition

#quantization settings
BASE_TYPE = torch.float32 #torch.float32 is default, will autochange during shape detection
QUANTIZATION_TYPE = torch.int8 #You change this value, this is the resulting quantized values [Default, torch.int8]
SCALE_TYPE = torch.float32 #torch.float32 is detault, will autochange during quantization

#some LZ4 and ZSTD Compression Settings
BLOCK_SIZE = 4096 #default - 4096 = 4KB
#Note: make sure that this BLOCK_SIZE divides evenly into (QuantTypeBits * NUM_DIMS)/8 (this is the Byte size per row)
#, or else it will zero pad at the end and slightly throw off your results
#Note: Also make sure that your BLOCK_SIZE <= (NUM_DIMS * QuantTypeBits)/8, or else your compression ratios are negative

#performance settings
ACCELERATION_DEVICE = "xpu" #"xpu" is default
#Other options for ACCELERATION_DEVICE may include:
# - "cuda" - this is for Nvidia GPUs and AMD ROCm-compatible GPUs [In theory should be compatible, but it isn't tested]
# - "xpu" - this is for Intel ARC GPUS (iGPU and dGPU) [This is what it was developed on]
# - "cpu" - this is a fallback that runs on your CPU if you don't have a compatible GPU (warning, it will be pretty slow)

#result printing settings (may improve performance too)
PRINT_ORIGINAL = True #prints original input data from file + basic analytics [default True]
PRINT_QUANT = True #prints quantized and dequantized values at beginning, along with MSE error [default True]
PRINT_RLE = False #prints the Run Length Encoding compressibility for every tensor [default False]
PRINT_COMP = True #prints the LZ4 and ZSTD Compression analytics for the bitplanes for every tensor [default True]
PRINT_EXTRA_SPACE_SAVING_METRICS = False #prints extra metrics (vs Direct Compression, vs Quant+Scale Compressed No BP) [default False]

PRINT_DELTA_TRANSFORMATION = False #prints version with delta transformation (optional due to diminishing returns) [default False]

#debug settings
PRINT_DEBUG = False #prints useful info for debugging [default False]
SHOW_EXTRANEOUS_RESULTS = False #default false to improve performance [default False]

SHOW_NEGATIVE_COMPRESSION_RATIOS = False
#Determines whether to compress the uncompressible or not
# - True: will show negative compression ratios (>1.0) [default]
# - False: will change any negative compression ratios to 1.0

IGNORE_RAW = True
#Will skip the raw compression permutations. This saves on performance and VRAM,
#,especially with large # of rows
# - True: doesn't perform calculations on Prequantized data [default]
# - False: performs calculations on prequantized data [warning, performance may drop]


#Step -1: vector embedding generation

#Step 0: Vector embedding file handling
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
        
        # 7. .txt (Create the vector embedding yourself)
        elif ext == '.txt':
            import os
            import numpy as np
            from pathlib import Path
            from huggingface_hub import hf_hub_download
            from llama_cpp import Llama
            from transformers import AutoModel, AutoTokenizer
            import torch
            import accelerate
            
            repository_id = "Qwen/Qwen3-Embedding-8B-GGUF"
            filename = "Qwen3-Embedding-8B-f16.gguf"
            current_directory = Path(__file__).parent.resolve()
            model_dir = current_directory / "models"
            #get uncompressed_embeddings
            model_dir.mkdir(parents=True, exist_ok=True)
            #check for file or download
            #only downloads a single 16gb file
            model_path = hf_hub_download(repo_id=repository_id, filename=filename, local_dir=str(model_dir))
            #initialize the engine
            engine = Llama(model_path=model_path, embedding=True, n_gpu_layers=-1, n_ctx=32768, n_batch=1024, pooling_type=0, verbose=False)
            #NOTE: n_ctx means how much memory you need for context
            #NOTE: n_batch probably means how many tokens it can batch together
            #NOTE: pooling_type=2 means that it does last token pooling (normal for Qwen3 embedding), and =0 means it gets a vector for every token
            
            #generate the 4096+ dim vector
            #text = "".join(open(file_path, 'r'))
            text = open(file_path, 'r', encoding='utf-8').read()
            tokens = engine.tokenize(text.encode('utf-8'))
            #with text as f:
                #lines = [line.strip() for line in f if line.strip()]
            #output = engine.create_embedding(lines)
            #vectors = [item['embedding'] for item in output['data']]
            #return np.array(vectors)
            #if(len(tokens) > 32768):
                #print("Failure")
            result = np.array(engine.create_embedding(text)['data'][0]['embedding'])
            print("ORIGINAL .TXT VECTOR EMBEDDING DATA: " + str(result.shape))
            print(result)
            return result
            #print(output)
            #raw_vector = np.array(output['data'][0]['embedding'])
            #return raw_vector
            '''
            current_directory = Path(__file__).parent.resolve()
            #load uncompressed_engine
            model_id = "Qwen/Qwen3-Embedding-8B"
            model_path = current_directory/"models"/"qwen3-8b-f16"
            #download fp16 safetensors
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=model_path)
            model = AutoModel.from_pretrained(model_id, dtype=torch.float16, device_map=ACCELERATION_DEVICE, cache_dir=model_path, trust_remote_code=True)
            #now text and stuff
            text = "".join(open(file_path, 'r'))
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                #extract hidden state of the last token from the last layer
                last_hidden_state = outputs.last_hidden_state[0, -1, :]
            #convert to CPU/Numpy for your custom quantization logic
            return last_hidden_state.cpu().numpy()
            '''
            '''
            #initialize model
            model = Llama(model_path="./Embedding_Models/Qwen3-Embedding-8B-f16.gguf", embedding=True, n_gpu_layers=-1)
            #generate the 4096+ dim embedding
            text = "".join(open(file_path, 'r'))
            output = model.create_embedding(text)
            #extract the tensor
            print("Dim of custom embedding: " + str(len(output['data'][0]['embedding'])))
            return output['data'][0]['embedding']
            '''

        else:
            print(f"Unsupported extension: {ext}")
            return None

    except Exception as e:
        print(f"Error extracting from {ext}: {e}")
        return None

#Step 1+2: Vector Embedding Input Normalization (to -1 to 1, with norm of 1)(if not already normalized) + Quantization (Assymetric Scalar Quantization or Symmetric Scalar Quantization)
#normalization
def normalize(np_emb):
    import numpy
    norm = numpy.linalg.norm(np_emb[0])
    if((norm >= 0.999) and (norm <= 1.001)):
        #is normalized, no need to normalize it, just return the already normalized embedding data
        return np_emb
    else: #now, if not normalized, then normalize it
        #normalize it
        norms = numpy.linalg.norm(np_emb, axis=1, keepdims=True)
        normalizedembeddings = np_emb / norms
        #verify
        verify = numpy.mean(numpy.linalg.norm(normalizedembeddings, axis=1))
        if((verify >= 0.999) and (verify <= 1.001)):
            return normalizedembeddings
        print("Not normalized. Normalization failed.")
        return normalizedembeddings

#quantization functions
def symmetric_scalar_quantization(tens_emb):
    import torch
    import numpy
    #per vector scale first
    #first, get max absolute value per vector (dim=1 is row-wise, keepdim=true for easy broadcasting)(uses intel XVE vector engines across rows)
    mxvals = torch.max(torch.abs(tens_emb), dim=1, keepdim=True).values
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

def Symmetric_Scalar_Error(orig, dequant):
    import torch
    import torch.nn.functional as f
    import numpy
    
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
    vector_loss = element_loss.mean(dim=1)
    print("\nLoss per Vector (MSE): ")
    print(vector_loss)
    #finding worst vector
    worst_vector = torch.argmax(vector_loss)
    print("\nLeast accurate vector (MSE): ")
    print(worst_vector)
    return vector_loss

#Step 3+4: Binary Conversion + Bitplane Disaggregation (Horizontal)

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
    
    #if(print_stage3 == True):
        #print("\nSTAGE 3: Bitplane Disaggregation (raw/direct): H: " + str(planes_horizontal.shape) + "; V: " + str(planes_vertical.shape) + ": ")
        #print("Horizontal Raw: ")
        #print(planes_horizontal)
    
    return planes_horizontal

def bitplane(quant, scale): #FIXED
    import torch

    def get_planes(tensor, num_bits):
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
        
        # Target: (N, B, W)
        planes = (t_signed.unsqueeze(1) >> bits.view(1, -1, 1)) & mask
        #planes = planes.to(torch.uint8).contiguous()
        #return planes.permute(2,0,1).contiguous()
        
        # 6. Convert to uint8 (0 or 1) for your RLE and packing analysis
        return planes.to(torch.uint8).contiguous()

    # Determine bit depth
    q_bits = quant.element_size() * 8
    s_bits = scale.element_size() * 8
    
    q_planes = get_planes(quant, q_bits)
    s_planes = get_planes(scale, s_bits)

    return q_planes, s_planes

#Step 5+6: Bitpacking (horizontally, packed_b) + LZ4&ZSTD Compression (and metrics)

def pack_bits(tensor_r): #NOTE: both of these modes are confirmed correct and working. Keep in mind it does 0 padding for N and/or B if they aren't a multiple of 8, which might inflate compression ratios, but this is done to some degree on real devices anyways
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
    
    #packed_b (for global and temporal)
    # 1. Safely move the dimension you want to pack to the end
    # Works for (N, W, B) or even (Batch, N, W, B)
    # If dim=0 (N), shape becomes (W, B, N)
    # If dim=2 (B), shape stays (N, W, B)
    tensor = torch.movedim(tensor_s, 2, -1)
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
    packed = torch.movedim(packed, -1, 2)
    
    return packed

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
    #print("TOTAL LEN: " + str(total_len))
    blocks = []
    for i in range(0, total_len, BLOCK_SIZE):
        block = group[i : i + BLOCK_SIZE]
        #padding
        if(len(block) < BLOCK_SIZE):
            #print("BLOCK: " + str(len(block)) + " : " + str(BLOCK_SIZE))
            block = block.ljust(BLOCK_SIZE, b'\x00')
        #print("Before Size: " + str(len(block)))
        compressed_size += len(c.compress(block))
        #print("After Size: " + str(len(c.compress(block))))
    
    return compressed_size
    #return len(c.compress(group))
    #return sum(len(c.compress(b)) for b in data_list)


def calculate_shannon_entropy_per_row(bit_tensor):
    import torch
    import numpy as np
    """
    Input: bit_tensor (N, B, W) containing only 0s and 1s.
    Output: entropy_matrix (N, B) where each cell is the Shannon Entropy of that row's bitplane.
    """
    # 1. Ensure we are on CPU and using floats for math
    # Shape is (N, B, W)
    N, B, W = bit_tensor.shape
    
    # 2. Calculate P(1): the mean of the bits along the W dimension
    # This gives us a (N, B) matrix of probabilities
    p1 = bit_tensor.to(torch.float32).mean(dim=-1).cpu().numpy()
    p0 = 1.0 - p1
    
    # 3. Apply the Shannon Entropy formula: H = -sum(p * log2(p))
    # We use numpy for the log2 and handling of 0 values
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -(p1 * np.log2(p1) + p0 * np.log2(p0))
        
    # 4. Clean up: log2(0) results in NaN, but in entropy, 0 * log(0) is defined as 0
    h = np.nan_to_num(h)
    #print(h)
    return h # Returns (N, B) matrix

def bitplane_compression_benchmark(tensor_d, title): #repolaces run_phased_benchmark
    import torch
    import multiprocessing
    import zstandard as zstd
    import pandas as pd
    import numpy
    import lz4.frame
    
    #clear xpu memory: this is especially important for very large tensors that fill your VRAM
    torch.xpu.empty_cache()
    
    #if the original input is not a tensor, convert it to a tensor
    if(torch.is_tensor(tensor_d) == False):
        tensor_d = torch.from_numpy(tensor_d).to(ACCELERATION_DEVICE)
    
    #get dimensions
    N,B,W = tensor_d.shape
    
    #pack the bits
    packed = pack_bits(tensor_d)
    
    #now some initial setup
    print("\nLZ4+ZSTD Results for " + title + ": " + str(tensor_d.shape))#tag modifier title
    orig_size_kb = (N * W) / (1024 * 8) #gets it in bytes becuase the compressors return size in bytes
    print("B: " + str(B))
    B_b = B
    concat = {n: {"Vector Embedding": n} for n in range(N)}
    metrics = {n: {"Vector Embedding": n} for n in range(N)}
    results = {k: {"Bitplane": k} for k in range(B_b)} #setting up the title
    cores = min(B_b+1, multiprocessing.cpu_count()) #add an extra core for the loops?
    #cores = multiprocessing.cpu_count() #just use ALL of the cores this time
    check_tensor_entropy(tensor_d)
    #lets try the most important one first, concat
    with multiprocessing.Pool(processes=min(N, multiprocessing.cpu_count())) as p:
        #set up the data
        temp = packed.contiguous().cpu().numpy()
        #set up the tasks
        concat_tasks = [[temp[n].tobytes()] for n in range(N)]
        #perform the compression
        concat_zstd_sizes = p.map(zstd_compress_list, concat_tasks)
        concat_lz4_sizes = p.map(lz4_compress_list, concat_tasks)
        #get the % savings
        if(SHOW_NEGATIVE_COMPRESSION_RATIOS == False):
            for n in range(N):
                concat[n]["Concatenated % Savings (ZSTD):"] = max(round((1-(concat_zstd_sizes[n]/((W*B_b)/8)))*100, 3), 0.000)
                concat[n]["Concatenated % Savings (LZ4):"] = max(round((1-(concat_lz4_sizes[n]/((W*B_b)/8)))*100, 3), 0.000)
        else:
            for n in range(N):
                concat[n]["Concatenated % Savings (ZSTD):"] = round((1-(concat_zstd_sizes[n]/((W*B_b)/8)))*100, 3)
                concat[n]["Concatenated % Savings (LZ4):"] = round((1-(concat_lz4_sizes[n]/((W*B_b)/8)))*100, 3)
        #now get the overall % savings per LZ4 and ZSTD
        concat_zstd_bytes = sum(concat_zstd_sizes)
        concat_lz4_bytes = sum(concat_lz4_sizes)
        total_size_bytes = (N * B * W)/8
        if(SHOW_NEGATIVE_COMPRESSION_RATIOS == False):
            concat_perc_saving_zstd = max(round((1 - (concat_zstd_bytes/total_size_bytes))*100, 3), 0.000)
            concat_perc_saving_lz4 = max(round((1 - (concat_lz4_bytes/total_size_bytes))*100, 3), 0.000)
        else:
            concat_perc_saving_zstd = max(round((1 - (concat_zstd_bytes/total_size_bytes))*100, 3), 0.000)
            concat_perc_saving_lz4 = max(round((1 - (concat_lz4_bytes/total_size_bytes))*100, 3), 0.000)
        #get the average too
        concat_zstd_s = [concat[n]["Concatenated % Savings (ZSTD):"] for n in range(N)]
        concat_lz4_s = [concat[n]["Concatenated % Savings (LZ4):"] for n in range(N)]
        concat_zstd_avg = float(sum(concat_zstd_s) / N)
        concat_lz4_avg = float(sum(concat_lz4_s) / N)
    
    #print concat metrics
    df1 = pd.DataFrame(list(concat.values())).sort_values("Vector Embedding", ascending=False)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    print(df1)
    print("Concat % Saving Overall (ZSTD): " + str(concat_perc_saving_zstd) + ", Average: " + str(concat_zstd_avg))
    print("Concat % Saving Overall (LZ4): " + str(concat_perc_saving_lz4) + ", Average: " + str(concat_lz4_avg))
    print((W * B_b)/8)
        
    with multiprocessing.Pool(processes=cores) as pool: #will come back to this later. will need to change a bunch of stuff, add new metrics too
        #try it by row first
        
        temp = packed.contiguous().cpu().numpy()
        tasks = list()
        
        for n in range(N):
            bitplanetasks = [[temp[n][b].tobytes()] for b in range(B_b)]
            z_tsizes = pool.map(zstd_compress_list, bitplanetasks)
            l_tsizes = pool.map(lz4_compress_list, bitplanetasks)
            if(SHOW_NEGATIVE_COMPRESSION_RATIOS == False):
                z_ratios = [min(round(z/(W/8), 3), 1.000) for z in z_tsizes]
                l_ratios = [min(round(l/(W/8), 3), 1.000) for l in l_tsizes]
            else:
                z_ratios = [round(z/(W/8), 3) for z in z_tsizes]
                l_ratios = [round(l/(W/8), 3) for l in l_tsizes]
                
            metrics[n]["Bit Plane Ratios (ZSTD):"] = z_ratios
            metrics[n]["Bit Plane Ratios (LZ4):"] = l_ratios
        
        '''
        #packed_b (packs across w)
        b_temporal = packed.permute(1,0,2).contiguous().cpu().numpy()
        temporaltasks = list()
        if(SHOW_NEGATIVE_COMPRESSION_RATIOS == False): #use this for the rewrite
            for k in range(B_b):
                bitplanetasks = [[b_temporal[k][x].tobytes()] for x in range(b_temporal.shape[1])]
                z_tsizes = sum(pool.map(zstd_compress_list, bitplanetasks))
                l_tsizes = sum(pool.map(lz4_compress_list, bitplanetasks))
                th_z = round((z_tsizes/1024) / orig_size_kb, 3)
                th_l = round((l_tsizes/1024) / orig_size_kb, 3)
                results[k]["Spatial(ZSTD):"] = th_z if th_z < 1.0 else 1.000
                results[k]["Spatial(LZ4):"] = th_l if th_l < 1.0 else 1.000
                #add to return results too
                #res["Spatial(ZSTD):"] += z_tsizes*8 if z_tsizes < (orig_size_kb*1024) else orig_size_kb*1024*8
                #res["Spatial(LZ4):"] += l_tsizes*8 if l_tsizes < (orig_size_kb*1024) else orig_size_kb*1024*8
        else:
            for k in range(B_b):
                bitplanetasks = [[b_temporal[k][x].tobytes()] for x in range(b_temporal.shape[1])]
                z_tsizes = sum(pool.map(zstd_compress_list, bitplanetasks))
                l_tsizes = sum(pool.map(lz4_compress_list, bitplanetasks))
                results[k]["Spatial(ZSTD):"] = round((z_tsizes/1024) / orig_size_kb, 3)
                results[k]["Spatial(LZ4):"] = round((l_tsizes/1024) / orig_size_kb, 3)
                #add to the return results too
                #res["Spatial(ZSTD):"] += z_tsizes*8
                #res["Spatial(LZ4):"] += l_tsizes*8
    '''
    #now, print and return the results
    
    df0 = pd.DataFrame(list(metrics.values())).sort_values("Vector Embedding", ascending=False)
    #df = pd.DataFrame(list(results.values())).sort_values("Bitplane", ascending=False)
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.width', 150)
    #print(df)
    print(df0)

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
    
    #clear xpu memory: this is especially important for very large tensors that fill your VRAM
    torch.xpu.empty_cache()
    
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
            if(SHOW_NEGATIVE_COMPRESSION_RATIOS == False): #use this for the rewrite
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

def space_saving_metrics(bit_comp_quant, bit_comp_scale, Raw_size, direct_size, direct_quantized_size):
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
    #others now are optional, denoted by [0] = -1
    if(direct_size[0] != -1):
        #Global(ZSTD):
        framedata[2][0] = round((1-((bit_comp_quant["Global(ZSTD):"]+bit_comp_scale["Global(ZSTD):"])/direct_size[0]))*100, 3)
        #Global(LZ4):
        framedata[2][1] = round((1-((bit_comp_quant["Global(LZ4):"]+bit_comp_scale["Global(LZ4):"])/direct_size[1]))*100, 3)
        #Temporal(ZSTD):
        framedata[2][2] = round((1-((bit_comp_quant["Temporal(ZSTD):"]+bit_comp_scale["Temporal(ZSTD):"])/direct_size[0]))*100, 3)
        #Temporal(LZ4):
        framedata[2][3] = round((1-((bit_comp_quant["Temporal(LZ4):"]+bit_comp_scale["Temporal(LZ4):"])/direct_size[1]))*100, 3)
        #Spatial(ZSTD):
        framedata[2][4] = round((1-((bit_comp_quant["Spatial(ZSTD):"]+bit_comp_scale["Spatial(ZSTD):"])/direct_size[0]))*100, 3)
        #Spatial(LZ4):
        framedata[2][5] = round((1-((bit_comp_quant["Spatial(LZ4):"]+bit_comp_scale["Spatial(LZ4):"])/direct_size[1]))*100, 3)
    if(direct_quantized_size[0] != -1):
        #Global(ZSTD):
        framedata[3][0] = round((1-((bit_comp_quant["Global(ZSTD):"]+bit_comp_scale["Global(ZSTD):"])/direct_quantized_size[0]))*100, 3)
        #Global(LZ4):
        framedata[3][1] = round((1-((bit_comp_quant["Global(LZ4):"]+bit_comp_scale["Global(LZ4):"])/direct_quantized_size[1]))*100, 3)
        #Temporal(ZSTD):
        framedata[3][2] = round((1-((bit_comp_quant["Temporal(ZSTD):"]+bit_comp_scale["Temporal(ZSTD):"])/direct_quantized_size[0]))*100, 3)
        #Temporal(LZ4):
        framedata[3][3] = round((1-((bit_comp_quant["Temporal(LZ4):"]+bit_comp_scale["Temporal(LZ4):"])/direct_quantized_size[1]))*100, 3)
        #Spatial(ZSTD):
        framedata[3][4] = round((1-((bit_comp_quant["Spatial(ZSTD):"]+bit_comp_scale["Spatial(ZSTD):"])/direct_quantized_size[0]))*100, 3)
        #Spatial(LZ4):
        framedata[3][5] = round((1-((bit_comp_quant["Spatial(LZ4):"]+bit_comp_scale["Spatial(LZ4):"])/direct_quantized_size[1]))*100, 3)
    #now print
    df = pd.DataFrame(np.array(framedata))
    df.columns = ['Global(ZSTD):','Global(LZ4):', 'Temporal(ZSTD):', 'Temporal(LZ4):', 'Spatial(ZSTD):', 'Spatial(LZ4):']
    df.index = ['vs Original Data Size:', 'vs Quantized+Scalar:', 'vs Direct Compression:', 'vs Quant+Scale Compressed (No BP):'] #rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    print(df)
    print("\n Extra: Uncompressed Bitplaned Quant+Scalar vs Uncompressed Bitplaned Original Tensor: " + str(round((1-((bit_comp_quant["Original Size(Bits):"]+bit_comp_scale["Original Size(Bits):"])/Raw_size))*100, 3)))

def raw_space_metrics(bit_comp, Raw_size, direct_size):
    import pandas as pd
    import numpy as np
    #note: direct_size is optional, put [-1] if you don't want it to be used
    # --> direct_size is if you want to compare bitplane compressed size to the direct compressed size
    
    framedata = [[]] #set up data
    #get data
    #Global(ZSTD):
    framedata[0].append(round((1-((bit_comp["Global(ZSTD):"])/Raw_size))*100, 3))
    #Global(LZ4):
    framedata[0].append(round((1-((bit_comp["Global(LZ4):"])/Raw_size))*100, 3))
    #Temporal(ZSTD):
    framedata[0].append(round((1-((bit_comp["Temporal(ZSTD):"])/Raw_size))*100, 3))
    #Temporal(LZ4):
    framedata[0].append(round((1-((bit_comp["Temporal(LZ4):"])/Raw_size))*100, 3))
    #Spatial(ZSTD):
    framedata[0].append(round((1-((bit_comp["Spatial(ZSTD):"])/Raw_size))*100, 3))
    #Spatial(LZ4):
    framedata[0].append(round((1-((bit_comp["Spatial(LZ4):"])/Raw_size))*100, 3))
    #then print
    df = pd.DataFrame(np.array(framedata))
    df.columns = ['Global(ZSTD):','Global(LZ4):', 'Temporal(ZSTD):', 'Temporal(LZ4):', 'Spatial(ZSTD):', 'Spatial(LZ4):']
    df.index = ['vs Original Data Size:']
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    print(df)
    
    #optional direct size too
    if(direct_size[0] != -1): #note: for this part, you must run direct_compression(tens) before running this function
        framedata2 = [[]] #set up data
        #get data
        #Global(ZSTD):
        framedata2[0].append(round((1-((bit_comp["Global(ZSTD):"])/direct_size[0]))*100, 3))
        #Global(LZ4):
        framedata2[0].append(round((1-((bit_comp["Global(LZ4):"])/direct_size[1]))*100, 3))
        #Temporal(ZSTD):
        framedata2[0].append(round((1-((bit_comp["Temporal(ZSTD):"])/direct_size[0]))*100, 3))
        #Temporal(LZ4):
        framedata2[0].append(round((1-((bit_comp["Temporal(LZ4):"])/direct_size[1]))*100, 3))
        #Spatial(ZSTD):
        framedata2[0].append(round((1-((bit_comp["Spatial(ZSTD):"])/direct_size[0]))*100, 3))
        #Spatial(LZ4):
        framedata2[0].append(round((1-((bit_comp["Spatial(LZ4):"])/direct_size[1]))*100, 3))
        #then print
        df2 = pd.DataFrame(np.array(framedata2))
        df2.columns = ['Global(ZSTD):','Global(LZ4):', 'Temporal(ZSTD):', 'Temporal(LZ4):', 'Spatial(ZSTD):', 'Spatial(LZ4):']
        df2.index = ['vs Direct [ZSTD,LZ4]:']
        print(df2)

def direct_compression(tens):
    import zstandard as zstd
    import numpy as np
    import lz4.frame
    import torch
    
    if(torch.is_tensor(tens) == False):
        tens = torch.from_numpy(tens).to(ACCELERATION_DEVICE)
    
    #get the raw bytes
    rawbytes = tens.cpu().numpy().tobytes()
    
    #do zstd compression first
    c_z = zstd.ZstdCompressor(level=3)
    zstd_size = 0 #in bytes
    for i in range(0, len(rawbytes), BLOCK_SIZE):
        block = rawbytes[i : i + BLOCK_SIZE]
        #handle with zero padding if needed
        if(len(block) < BLOCK_SIZE):
            block = block.ljust(BLOCK_SIZE, b'\x00')
        
        zstd_size += len(c_z.compress(block))
    direc_ratio_z = round(float(zstd_size/len(rawbytes)), 3)
    direc_perc_z = round((1-(zstd_size/len(rawbytes)))*100, 3)
    if(SHOW_NEGATIVE_COMPRESSION_RATIOS == False):
        direc_ratio_z = direc_ratio_z if direc_ratio_z < 1.000 else 1.000
        direc_perc_z = direc_perc_z if direc_perc_z >= 0.000 else 0.000
    print("\nDirect Compression(ZSTD) Ratio & % Memory Saving: " + str(direc_ratio_z) + " , " + str(direc_perc_z) + "%")
    zstd_bits = zstd_size * 8 #size of zstd compressed in bits
    
    #then, do lz4 compression
    lz4_size = len(lz4.frame.compress(rawbytes, block_size=BLOCK_SIZE)) #in bytes
    direc_ratio_l = round(float(lz4_size/len(rawbytes)), 3)
    direc_perc_l = round((1-(lz4_size/len(rawbytes)))*100, 3)
    if(SHOW_NEGATIVE_COMPRESSION_RATIOS == False):
        direc_ratio_l = direc_ratio_l if direc_ratio_l < 1.000 else 1.000
        direc_perc_l = direc_perc_l if direc_perc_l >= 0.000 else 0.000
    print("\nDirect Compression(LZ4) Ratio & % Memory Saving: " + str(direc_ratio_l) + " , " + str(direc_perc_l) + "%")
    lz4_bits = lz4_size * 8 #in bits
    
    return zstd_bits, lz4_bits

def direct_quantized_compression(quant, scale):
    import zstandard as zstd
    import numpy as np
    import lz4.frame
    import torch
    
    if(torch.is_tensor(quant) == False):
        quant = torch.from_numpy(quant).to(ACCELERATION_DEVICE)
    if(torch.is_tensor(scale) == False):
        scale = torch.from_numpy(scale).to(ACCELERATION_DEVICE)
    
    #get the raw bytes
    rawbytes_q = quant.cpu().numpy().tobytes()
    rawbytes_s = scale.cpu().numpy().tobytes()
    
    #do zstd compression first
    c_z = zstd.ZstdCompressor(level=3)
    zstd_size = 0 #in bytes
    for i in range(0, len(rawbytes_q), BLOCK_SIZE):
        block = rawbytes_q[i : i + BLOCK_SIZE]
        #handle with zero padding if needed
        if(len(block) < BLOCK_SIZE):
            block = block.ljust(BLOCK_SIZE, b'\x00')
        
        zstd_size += len(c_z.compress(block))
    for i in range(0, len(rawbytes_s), BLOCK_SIZE):
        block = rawbytes_s[i : i + BLOCK_SIZE]
        #handle with zero padding if needed
        if(len(block) < BLOCK_SIZE):
            block = block.ljust(BLOCK_SIZE, b'\x00')
        
        zstd_size += len(c_z.compress(block))
    direc_ratio_z = round(float(zstd_size/(len(rawbytes_q)+len(rawbytes_s))), 3)
    direc_perc_z = round((1-(zstd_size/(len(rawbytes_q)+len(rawbytes_s))))*100, 3)
    if(SHOW_NEGATIVE_COMPRESSION_RATIOS == False):
        direc_ratio_z = direc_ratio_z if direc_ratio_z < 1.000 else 1.000
        direc_perc_z = direc_perc_z if direc_perc_z >= 0.000 else 0.000
    print("\nDirect Compression(ZSTD, Quant+Scale) Ratio & % Memory Saving: " + str(direc_ratio_z) + " , " + str(direc_perc_z) + "%")
    zstd_bits = zstd_size * 8 #size of zstd compressed in bits
    
    #then, do lz4 compression
    lz4_size = len(lz4.frame.compress(rawbytes_q, block_size=BLOCK_SIZE)) + len(lz4.frame.compress(rawbytes_s, block_size=BLOCK_SIZE)) #in bytes
    direc_ratio_l = round(float(lz4_size/(len(rawbytes_q)+len(rawbytes_s))), 3)
    direc_perc_l = round((1-(lz4_size/(len(rawbytes_q)+len(rawbytes_s))))*100, 3)
    if(SHOW_NEGATIVE_COMPRESSION_RATIOS == False):
        direc_ratio_l = direc_ratio_l if direc_ratio_l < 1.000 else 1.000
        direc_perc_l = direc_perc_l if direc_perc_l >= 0.000 else 0.000
    print("\nDirect Compression(LZ4, Quant+Scale) Ratio & % Memory Saving: " + str(direc_ratio_l) + " , " + str(direc_perc_l) + "%")
    lz4_bits = lz4_size * 8 #in bits
    
    return zstd_bits, lz4_bits

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
    
#extra RLE stuff
def RLE_bitplane_compressibility(matrix, title):
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
    
    bp_avg = torch.mean(ratio, dim=0) #per bitplane, originally dim=1
    print("\nHorizontal Per Bitplane: ")
    print(bp_avg)
    row_avg = torch.mean(ratio, dim=1) #per vector embedding, originally dim=0
    print("\nHorizontal Per Vector Embedding: ")
    print(row_avg)
    
    print("\nDEBUG: Overall Average RLE Compression Ratio: " + str(torch.mean(ratio)))
    return torch.mean(ratio)

#Main program logic

def run_simulation():
    import numpy
    import torch
    
    #stage -1: vector embedding generation (not doing yet)
    
    #stage 0: getting the file
    
    #this is initialization, and file handling (stage 0: getting the file)
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
    
    #stage 1: normalize vector embedding input
    print("\nSTAGE 0: Normalization: ")
    normalized_emb = normalize(numpy_emb)
    normalized_tensor = torch.tensor(normalized_emb).to(ACCELERATION_DEVICE)
    print("Normalized Embedding Input: ")
    print(normalized_tensor)
    
    #stage 2: quantize it (quant+scalar)
    print("\nSTAGE 1: Symmetric Scalar Quantization: ")
    quantized, scalars = symmetric_scalar_quantization(normalized_tensor)
    print("Quantized Values: ")
    print(quantized)
    print("Scalar Values: ")
    print(scalars)
    
    #mse error
    dequantized = symmetric_scalar_dequantization(quantized, scalars)
    Symmetric_Scalar_Error(normalized_tensor, dequantized)
    
    #stage 3+4: binary conversion + bitplane disaggregation (horizontal, do mainly for quant, not scale)
    print("\nSTAGE 3+4: Binary Conversion + Bitplane Disaggregation: ")
    quantized_bitplane, scalar_bitplane = bitplane(quantized, scalars)
    print("Quantized Bitplanes: " + str(quantized_bitplane.shape))
    print(quantized_bitplane)
    print("Scalar Bitplanes: " + str(scalar_bitplane.shape))
    print(scalar_bitplane)
    raw_bitplane = direct_bitplane(normalized_tensor)
    print("Raw Bitplanes: " + str(raw_bitplane.shape))
    print(raw_bitplane)
    
    #stage 4.5 optional: RLE compression analysis
    print("\nDEBUG: RLE Compression analysis on QUANT Horizontal: ")
    rle_res = RLE_bitplane_compressibility(quantized_bitplane,"Quant-Horizontal")
    #print(rle_res)
    
    #stage 5+6: bitpacking (horizontally, packed_b) + LZ4 & ZSTD Bitplane compression (with metrics)
    print("\nSTAGE 5+6: Bitpacking + LZ4 & ZSTD Compression: ")
    bitplane_compression_benchmark(quantized_bitplane, "bitplane quant horizontal")
    #I want to test delta transformation too
    #it gives ~1% improvement, which is not good becuase it introduces computational complexity which hurts latency
    if(PRINT_DELTA_TRANSFORMATION == True):
        tensor_delta = quantized_bitplane.clone()
        tensor_delta[:, 1:] = quantized_bitplane[:, 1:] - quantized_bitplane[:, :-1]
        bitplane_compression_benchmark(tensor_delta, "delta transformation - horizontal")
    
    #entropy
    entropy_results = calculate_shannon_entropy_per_row(quantized_bitplane)
    # Print average for each bitplane to the console
    for b in range(entropy_results.shape[1]):
        avg_h = entropy_results[:, b].mean()
        # Compressibility = (1 - Entropy)
        theoretical_savings = (1.0 - avg_h) * 100
        print(f"Bitplane {b} | Avg Entropy: {avg_h:.4f} | Max Potential Savings: {theoretical_savings:.2f}%")
    pass

if  __name__ == "__main__":
    import numpy
    import torch
    
    #then call simulation
    run_simulation()
    