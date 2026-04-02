#settings
#We should be able to handle these file types:
# - .parquet, .csv, .npy, etc
#Initial Setup
import os
base_directory = os.path.dirname(os.path.abspath(__file__))
import torch
file_path = "" #don't touch this unless you know what you are doing
NUM_ROWS = 100 #This is for you to choose, it is the number of vector embeddings (essentially # of tokens)[must be a multiple of 8]
#Note: your NUM_ROWS should ideally exceed your BLOCK_SIZE, and for best results, should be a multiple of BLOCK_SIZE
NUM_DIMS = 768 #768 is default, but will automatically change with shape recognition
MAX_ROWS = 10000 #10k is default, but will automatically change with shape recognition

#Embedding model settings
CHUNK_SIZE = 512
#Only applies to average pooling (1) and CLS pooling (2), and only applies to embedding models
#Determines the token chunking context mainly for average pooling. Higher chunk size has higher RAM/VRAM requirements
#chunk size is essentially token context per vector embedding row, so CHUNK_SIZE = 512 means 512 tokens per 1 vector embedding

BATCH_SIZE = 512
#for hardware efficiency, doesn't effect results.
#use lower batch sizes for weaker machines like iGPUs (256, 512, 1024, 2048)
#use higher batch sizes for stronger dedicated GPUs (4096, 8192, etc)
#For loose rule of thumb, CUDA Core count = CHUNK_SIZE * BATCH_SIZE
#default: 512

POOLING_TYPE = 0 #most embedding models will use their default pooling method, this setting overrides it
#Determines how token semantics are pooled together during embedding model computations (changes results)
# - 0: per-token vector embeddings (1 row per 1 token)
# - 1: mean-row vector embeddings (averaged 1 line of tokens (chunk size) for 1 row) [common in RAG]
# - 2: CLS/Last Token vector embedding (single vector, for all tokens) [common in certain embedding models]
# - 3: use default pooling mode for that embedding [default]
#note: 1 and 2 both automatically change the number of rows:
# --> 1: changes it to min(NUM_ROWS, available rows)
# --> 2: changes NUM_ROWS == 1

TOKEN_CONTEXT = -1
#this determines the context of tokens to generate each embedding row during the pooling phase
#only applies to local embedding models, so precomputed ones don't use this
#will only use this value if it is inside the limits of the model
# - other
# - -1 [default, uses model default, native setting]

NUM_LINES = 100
#this is the number of lines of text that the embedding model reads to convert to embeddings
#make sure that it gives enough text for the embedding model to use, or else it might crash the program or
# give unpredictable results
# - 1024 [default]

CHUNKING_MODE = True
#This determines how things are chunked
# - True: Uses sentence-based chunking, (Used for research, preserves semantic meaning) [default]
# - False: Uses size/token-based chunking, (Typically used in RAG industry, loses semantic meaning)

#quantization settings
BASE_TYPE = torch.float32 #torch.float32 is default, will autochange during shape detection (embedding models will use their defaults)
QUANTIZATION_TYPE = torch.int8 #You change this value, this is the resulting quantized values [Default, torch.int8]
SCALE_TYPE = torch.float32 #torch.float32 is detault, will autochange during quantization

#some LZ4 and ZSTD Compression Settings
BLOCK_SIZE = 1024 #default - 4096 = 4KB
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
# - True: will show negative compression ratios (>1.0)
# - False: will change any negative compression ratios to 1.0 [default]

IGNORE_RAW = False
#Will skip the raw compression permutations. This saves on performance and VRAM,
#,especially with large # of rows
# - True: doesn't perform calculations on Prequantized data [default]
# - False: performs calculations on prequantized data [warning, performance may drop]

IGNORE_MANTISSA = False
#Will only use the sign and exponent bits for compression analysis. Only applies to RAW bitplane (IGNORE_RAW must be False)
# - True: only uses sign and exponent bits for compression analysis on RAW tensors (not quantized)
# - False: Uses all bitplanes for compression analysis on RAW tensors (not quantized) [default]

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

def generate_local_embeddings(text, model_name, pooling_mode, dim_count, max_context, native_context, bert, data_type):
    import os
    import json
    import io
    import numpy as np
    import pandas as pd
    import zstandard as zstd
    from pathlib import Path
    import sentence_transformers
    """Checks local 'models' folder, downloads if missing, and runs inference."""
    script_dir = Path(__file__).parent.resolve()
    model_dir = script_dir / "models"
    model_dir.mkdir(exist_ok=True)
    
    #allows the arc IGPU to use more vram than 4GB
    if(ACCELERATION_DEVICE == "xpu"):
        oneapi_path = r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin"
        os.environ["PATH"] = oneapi_path + os.pathsep + os.environ["PATH"]
        # Force the level_zero driver to select the GPU (iGPU/Arc)
        os.environ["ONEAPI_DEVICE_SELECTOR"] = "level_zero:gpu"
        # Optional: Helps with task submission performance on Arc
        os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"
        #os.environ["GGML_SYCL_DEBUG"] = "1"
        os.environ["SYCL_UR_USE_LEVEL_ZERO_V2"] = "1"
        os.environ["UR_L0_USE_RELAXED_ALLOCATION_LIMITS"] = "1"
        os.environ["ZES_ENABLE_SYSMAN"] = "1"
        os.environ["GGML_SYCL_DEVICE"] = "0" # Forces the first Intel GPU
        if(model_name == "e5-large-v2"):
            os.environ["GGML_SYCL_F16_ACC"] = "0"
    
    #sets up some parameters
    if(TOKEN_CONTEXT == -1):
        context_n = native_context
    else:
        context_n = min(TOKEN_CONTEXT, max_context)

    # GGUF MODELS (Llama-cpp-python) #removed nomic from GGUF models, as it causes compatibility issues with certain backends, like vulkan
    if model_name in ["qwen3", "e5-mistral", "nomic", "bge-m3", "e5-large-v2", "gemma-2", "snowflake-arctic-embed-v2.0"]:
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download
        import llama_cpp
        
        configs = {
            "qwen3": ("Qwen/Qwen3-Embedding-8B-GGUF", "Qwen3-Embedding-8B-f16.gguf"),
            "e5-mistral": ("dranger003/e5-mistral-7b-instruct-GGUF", "ggml-e5-mistral-7b-instruct-f16.gguf"),
            "nomic": ("nomic-ai/nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.f32.gguf"),
            "e5-large-v2": ("ChristianAzinn/e5-large-v2-gguf", "e5-large-v2_fp32.gguf"),
            "bge-m3": ("gpustack/bge-m3-GGUF", "bge-m3-FP16.gguf"),
            "gemma-2": ("alamios/gemma-2-9b-it-GGUF-f16", "gemma-2-9b-it.f16.gguf"),
            "snowflake-arctic-embed-v2.0": ("Casual-Autopsy/snowflake-arctic-embed-l-v2.0-gguf", "snowflake-arctic-embed-l-v2.0-f32.gguf")
        }
        #original e5-mistral: ("intfloat/e5-mistral-7b-instruct", "e5-mistral-7b-instruct-f16.gguf")
        repo, filename = configs[model_name]
        m_path = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(model_dir))
        
        #setting up engine parameters
        if(ACCELERATION_DEVICE == "xpu"): #xpu sycl backend has some bugs/quirks that require attention
            if(model_name == "e5-large-v2"):
                engine = Llama(model_path=m_path, embedding=True, flash_attn=False, logits_all=True, pooling_type=pooling_mode, n_parallel=1, n_gpu_layers=0, n_threads=14, main_gpu=-1, offload_kqv=False, use_mmap=True, n_ctx=context_n, n_batch=BATCH_SIZE, n_ubatch=BATCH_SIZE, verbose=True)
            else:
                engine = Llama(model_path=m_path, embedding=True, flash_attn=False, logits_all=True, pooling_type=pooling_mode, n_parallel=1, n_gpu_layers=-1, use_mmap=True, n_ctx=context_n, n_batch=BATCH_SIZE, n_ubatch=BATCH_SIZE, verbose=False)
        else:
            engine = Llama(model_path=m_path, embedding=True, pooling_type=pooling_mode, n_gpu_layers=-1, main_gpu=0, n_ctx=context_n, n_batch=BATCH_SIZE, n_ubatch=BATCH_SIZE, verbose=False)
        
        if((pooling_mode == 1) or (pooling_mode == 2)):
            #text = ''.join(text[:NUM_ROWS])
            all_results = []
            #print("print1")
            if(isinstance(text, np.ndarray)):
                text = text.tolist()
            elif(isinstance(text, str)):
                text = [text]
            #print("print2")
            total_text = text[:NUM_LINES]
            #NOTE FOR 4/2+: You need to account for when lines are too long or too short
            full_text = []
            if(ACCELERATION_DEVICE == "xpu"): #xpu sycl library has a quirk where it can only embed 1 string line at a time
                '''
                if(CHUNKING_MODE == True):
                    for line in total_text:
                        tokens = engine.tokenize(line.encode('utf-8'))
                        if(len(tokens) > context_n):
                            engine.reset()
                            res = engine.embed(tokens)
                            full_text.append(np.array(res[0], dtype=data_type))
                            #full_text.append(engine.create_embedding(input=tokens)['data'][0]['embedding'])
                        else:
                            for i in range(0, len(tokens), context_n):
                                engine.reset()
                                res = engine.embed(tokens[i:i+context_n])
                                full_text.append(np.array(res[0], dtype=data_type))
                                #full_text.append(engine.create_embedding(input=tokens[i:i+context_n])['data'][0]['embedding'])
                        return np.array(full_text, dtype=data_type)
                '''
                vectors = [engine.create_embedding(line)['data'][0]['embedding'] for line in total_text]
                return np.array(vectors, dtype=data_type)
            else: #other backends can do all the embeddings at once
                response = engine.create_embedding(total_text)
                vectors = [row['embedding'] for row in response['data']]
                return np.array(vectors, dtype=data_type)
        elif(pooling_mode == 0):
            text = ''.join(text[:NUM_ROWS])
        return np.array(engine.create_embedding(text)['data'][0]['embedding'])
    

def generate_local_embeddings0(text, model_name, pooling_mode, dim_count, max_context, native_context, bert):
    import os
    import json
    import io
    import numpy as np
    import pandas as pd
    import zstandard as zstd
    from pathlib import Path
    import sentence_transformers
    """Checks local 'models' folder, downloads if missing, and runs inference."""
    script_dir = Path(__file__).parent.resolve()
    model_dir = script_dir / "models"
    model_dir.mkdir(exist_ok=True)
    
    #allows the arc IGPU to use more vram than 4GB
    if(ACCELERATION_DEVICE == "xpu"):
        oneapi_path = r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin"
        os.environ["PATH"] = oneapi_path + os.pathsep + os.environ["PATH"]
        # Force the level_zero driver to select the GPU (iGPU/Arc)
        os.environ["ONEAPI_DEVICE_SELECTOR"] = "level_zero:gpu"
        # Optional: Helps with task submission performance on Arc
        os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"
        os.environ["GGML_SYCL_DEBUG"] = "1"
        os.environ["SYCL_UR_USE_LEVEL_ZERO_V2"] = "1"
        os.environ["UR_L0_USE_RELAXED_ALLOCATION_LIMITS"] = "1"
        os.environ["ZES_ENABLE_SYSMAN"] = "1"
        os.environ["GGML_SYCL_DEVICE"] = "0" # Forces the first Intel GPU
    
    #sets up some parameters
    if(TOKEN_CONTEXT == -1):
        context_n = native_context
    else:
        context_n = min(TOKEN_CONTEXT, max_context)

    # GGUF MODELS (Llama-cpp-python) #removed nomic from GGUF models, as it causes compatibility issues with certain backends, like vulkan
    if model_name in ["qwen3", "e5-mistral", "nomic"]:
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download
        import llama_cpp
        
        configs = {
            "qwen3": ("Qwen/Qwen3-Embedding-8B-GGUF", "Qwen3-Embedding-8B-f16.gguf"),
            "e5-mistral": ("dranger003/e5-mistral-7b-instruct-GGUF", "ggml-e5-mistral-7b-instruct-f16.gguf"),
            "nomic": ("nomic-ai/nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.f32.gguf")
        }
        #original e5-mistral: ("intfloat/e5-mistral-7b-instruct", "e5-mistral-7b-instruct-f16.gguf")
        repo, filename = configs[model_name]
        m_path = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(model_dir))
        
        #setting up engine parameters
        engine = Llama(model_path=m_path, embedding=True, flash_attn=False, logits_all=True, pooling_type=pooling_mode, n_parallel=4, n_gpu_layers=-1, main_gpu=0, use_mmap=False, n_ctx=context_n, n_batch=BATCH_SIZE, n_ubatch=BATCH_SIZE, verbose=True)
        
        if((pooling_mode == 1) or (pooling_mode == 2)):
            #text = ''.join(text[:NUM_ROWS])
            all_results = []
            #print("print1")
            if(isinstance(text, np.ndarray)):
                text = text.tolist()
            elif(isinstance(text, str)):
                text = [text]
            #print("print2")
            total_text = text[:NUM_LINES]
            #print(total_text)
            #print("print3")
            response = engine.create_embedding(total_text)
            #print("print4")
            vectors = [row['embedding'] for row in response['data']]
            #print("print5")
            return np.array(vectors)
            #for i in range(0, NUM_ROWS):
                #batch = text[i]
                #response = engine.create_embedding(batch)
                #vectors = [row['embedding'] for row in response['data']]
                #all_results.append(vectors)
            #return np.vstack(all_results)
            '''
            text = text[:NUM_ROWS]
            return np.vstack(engine.create_embedding(text)['data'][0]['embedding'])
            '''
        elif(pooling_mode == 0):
            text = ''.join(text[:NUM_ROWS])
        return np.array(engine.create_embedding(text)['data'][0]['embedding'])
    
    #
    '''
    elif model_name in ["nomic"]:
        from transformers import AutoModel, AutoTokenizer
        import torch.nn.functional as F
        import torch
        import numpy as np
        import einops
        if(isinstance(text, np.ndarray)):
            text = text.tolist()
        elif(isinstance(text, str)):
            text = text.split("\n")
            text = [t + "\n" for t in text]
        #0. set up the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        if(CHUNKING_MODE == True):
            model.eval()
            text = text[:NUM_LINES]
            text_lines = []
            prefix = "search_document:"
            prefix_overhead = len(tokenizer.encode(prefix, add_special_tokens=False))
            budget = context_n - prefix_overhead - 2
            for line in text:
                print("Before")
                tokens = tokenizer.encode(line, add_special_tokens=False)
                print("After")
                if(len(tokens) <= budget):
                    text_lines.append(line)
                else:
                    for i in range(0, len(tokens), budget):
                        chunk = tokens[i : i + budget]
                        # 3. Decode back to text so ST can re-tokenize it with special tokens
                        decoded_chunk = tokenizer.decode(chunk, skip_special_tokens=True)
                        text_lines.append(decoded_chunk)
                del tokens
            text_lines = [prefix + t for t in text_lines] #add the prefix, makes it more accurate
            embeddings = []
            #print(len(all_ids))
            rows = 0
            for chunk in text_lines:
                #5. add the structural anchors for nomic/BERT
                # [101] = CLS, [102] = SEP
                print("5")
                # This handles the [101] and [102] (CLS/SEP) automatically
                inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(model.device)
                
                print("6")
                with torch.no_grad():
                    outputs = model(**inputs)
                print("7")
            
                # 2. Robust Mean Pooling (Using Attention Mask)
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
            
                # Expand mask to match hidden state shape [Batch, Seq, Dim]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            
                # Sum the visible tokens and divide by the sum of the mask
                sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                chunk_val = sum_embeddings / sum_mask
            
                # 3. Normalize to Unit Hypersphere
                # Essential for your bit-plane range stability
                chunk_val = F.normalize(chunk_val, p=2, dim=1)
            
                embeddings.append(chunk_val.cpu().numpy())
            
                rows += 1
                if(rows >= NUM_ROWS):
                    break
            return np.vstack(embeddings)
            #embedding = model.encode(text_lines, batch_size=1, convert_to_numpy=True)
            #if(len(embedding.shape) == 1):
                #embedding = embedding.reshape(1,embedding.shape[0])
            #return embedding
        elif(CHUNKING_MODE == False):
            #force to CPU to avoid the "Memory not initialized" iGPU/Vulkan error
            print("1")
            model.eval()
            #1. tokenize everything into one giant list of IDs. Turn off add_special_tokens so we can handle [CLS]/[SEP] manually
            all_ids = []
            prefix = "search_document:" #this is required for nomic to give accurate RAG compression results
            for t in text[:NUM_LINES]:
                full_text = prefix + t
                all_ids.extend(tokenizer.encode(full_text, add_special_tokens=False))
        
            #2. define payload size (context_len - 2 for CLS and SEP)
            payload_size = context_n - 2
        
            #3. create chunks of exactly payload_size
            #we drop the last chunk if it's incomplete to keey your data "pure"
            print("3")
            chunks = [all_ids[i : i + payload_size] for i in range(0, len(all_ids), payload_size) if len(all_ids[i : i + payload_size]) == payload_size]
            print("4")
        
            #4. generate embeddings
            embeddings = []
            print(len(all_ids))
            rows = 0
            for chunk in chunks:
                #5. add the structural anchors for nomic/BERT
                # [101] = CLS, [102] = SEP
                print("5")
                full_chunk = [[101] + chunk + [102]]
                input_ids = torch.tensor(full_chunk).to(model.device)
                attention_mask = torch.ones((1, len(full_chunk))).to(model.device)
                input_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
                print("6")
                with torch.no_grad():
                    outputs = model(**input_dict)
                print("7")
            
                #6. mean pooling: average the hidden states
                last_hidden = outputs.last_hidden_state
                print("before dim1")
                chunk_val = torch.mean(last_hidden, dim=1)
            
                #7. normalize to the unit hypersphere (standard for RAG/Nomic)
                print("before dim2")
                chunk_val = torch.nn.functional.normalize(chunk_val, p=2, dim=1)
                embeddings.append(chunk_val.cpu().numpy())
            
                rows += 1
                if(rows >= NUM_ROWS):
                    break
            return np.vstack(embeddings)
    
    # TRANSFORMERS MODELS (Sentence-Transformers)
    elif model_name in ["bge-m3", "e5-large-v2"]:
        from sentence_transformers import SentenceTransformer
        import torch
        hf_id = {"bge-m3": "BAAI/bge-m3", "e5-large-v2": "intfloat/e5-large-v2"}[model_name]
        # SentenceTransformer handles local caching automatically
        model = SentenceTransformer(hf_id, cache_folder=str(model_dir))
        model.gradient_checkpointing_enable() #memory optimization, slightly reduces speed
        query_overhead = 0 #certain datasets need this
        if(model_name == "e5-large-v2"):
            query_overhead = len(model.tokenizer.encode("passage:", add_special_tokens=False))
        if(CHUNKING_MODE == True):
            text = text[:NUM_LINES]
            #try ruis new method
            text_lines = []
            tokenizer = model.tokenizer
            budget = context_n - query_overhead - 2 #tries to not account for special tokens
            for line in text:
                tokens = tokenizer.encode(line, add_special_tokens=False)
                if(len(tokens) <= budget):
                    text_lines.append(line)
                else:
                    for i in range(0, len(tokens), budget):
                        chunk = tokens[i : i + budget]
                        # 3. Decode back to text so ST can re-tokenize it with special tokens
                        decoded_chunk = tokenizer.decode(chunk, skip_special_tokens=True)
                        text_lines.append(decoded_chunk)
                del tokens
            #
            if(model_name == "e5-large-v2"):
                text_lines = ["passage:" + t for t in text_lines]
            embedding = model.encode(text_lines, batch_size=1, convert_to_numpy=True)
            if(len(embedding.shape) == 1):
                embedding = embedding.reshape(1,embedding.shape[0])
            return embedding
        elif((CHUNKING_MODE == False) and (pooling_mode == 0)):
            text = text[:NUM_LINES]
            inputs = model.tokenize([text])
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output = model[0](inputs)
                raw_matrix = output['token_embeddings']
            return np.array(raw_matrix[0].to("cpu"))
        elif((CHUNKING_MODE == False) and (pooling_mode == 1)): #mainly for e5
            tokenizer = model.tokenizer
            text = text[:NUM_LINES]
            full_ids = []
            for t in text:
                full_text = t + "passage: " #e5 is sensitive to prefixes
                full_ids.extend(tokenizer.encode(full_text, batch_size=1, add_special_tokens=False))
            #create chunks of text strings
            chunks = [full_ids[i : i + max_context] for i in range(0, len(full_ids), context_n)]
            #convert IDs back to strings so .encode() can handle them
            processed_chunks = [tokenizer.decode(c) for c in chunks if len(c) == context_n]
            #now do the embedding
            if(ACCELERATION_DEVICE == "xpu"):
                torch.xpu.empty_cache()
            elif(ACCELERATION_DEVICE == "cuda"):
                torch.cuda.empty_cache()
            embedding = model.encode(processed_chunks, convert_to_numpy=True)
            #return result
            return embedding
        elif((CHUNKING_MODE == False) and (pooling_mode == 2)): #mainly for BAAI-BGE-M3
            #1. use BGE-M3 tokenizer to wrap into 8190 token chunks
            tokenizer = model.tokenizer
            text = text[:NUM_LINES]
            full_ids = []
            for t in text:
                full_ids.extend(tokenizer.encode(t, batch_size=1, add_special_tokens=False))
            print(len(full_ids))
            #create chunks of text strings
            chunks = [full_ids[i : i + max_context] for i in range(0, len(full_ids), context_n)]
            #convert IDs back to strings so .encode() can handle them
            processed_chunks = [tokenizer.decode(c) for c in chunks if len(c) == context_n]
            #now do the embedding
            if(ACCELERATION_DEVICE == "xpu"):
                torch.xpu.empty_cache()
            elif(ACCELERATION_DEVICE == "cuda"):
                torch.cuda.empty_cache()
            embedding = model.encode(processed_chunks, batch_size=1, convert_to_numpy=True, normalize_embeddings=True)
            #now return the results
            return embedding

    #note: these others below are not going to be used
    # API MODELS
    elif model_name == "openai-3-large":
        from openai import OpenAI
        client = OpenAI()
        return np.array(client.embeddings.create(input=text, model="text-embedding-3-large").data[0].embedding)
    
    # 4. GOOGLE GEMINI (API-based)
    elif model_name == "gemini-2":
        import google.generativeai as genai
        Google_api_key = str(input("Enter your Google API Key: "))
        # Requires GOOGLE_API_KEY env variable
        genai.configure(api_key=os.environ[Google_api_key])
        # 'models/text-embedding-004' or the new 'gemini-embedding-2-preview'
        result = genai.embed_content(
            model="models/gemini-embedding-2-preview",
            content=text,
            task_type="retrieval_document"
        )
        return np.array(result['embedding'])

    # 5. NVIDIA NV-EMBED (Local via Transformers or NIM API)
    elif model_name == "nv-embed-v2":
        # Strategy A: Using NVIDIA's API (NIM)
        nvidia_api_key = str(input("Enter your Nvidia API Key: "))
        from openai import OpenAI # NVIDIA NIM uses OpenAI-compatible headers
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ[nvidia_api_key]
        )
        response = client.embeddings.create(
            input=[text],
            model="nvidia/nv-embed-v2"
        )
        return np.array(response.data[0].embedding)

    # 6. NVIDIA NEMOTRON (Local via Sentence-Transformers)
    elif model_name == "nemotron-embed":
        from sentence_transformers import SentenceTransformer
        # These are often hosted on HF as 'nvidia/llama-nemotron-embed-1b-v2'
        model = SentenceTransformer("nvidia/llama-nemotron-embed-1b-v2", cache_folder=str(model_dir))
        return model.encode(text)
    pass
    '''

def extract_vectors(file_path, vector_column_name='embeddings'):
    import os
    import numpy as np
    import pandas as pd
    import json
    import zstandard as zstd
    import ml_dtypes
    import pyarrow.feather as feather
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
        elif ext == '.csv': #don't use this either
            df = pd.read_csv(file_path)
            # CSV vectors are often stored as strings like "[0.1, 0.2...]"
            # This converts those strings back into actual lists/arrays
            print([df.iloc[i] for i in range(8)])
            key = _find_vector_column(df.iloc[0])
            #print("CSV KEYS: ")
            #print(df.keys())
            if isinstance(df[key].iloc[0], str):
                df[key] = df[key].apply(json.loads)
            return np.stack(df[key].values)

        # 5. SafeTensors (Modern Model Weights)
        elif ext == '.safetensors': #don't use this at all. It is more for model weights than vector embeddings
            from safetensors.numpy import load_file
            tensors = load_file(file_path)
            # Safetensors usually have multiple keys; we return the first one or a specific one
            #key = list(tensors.keys())[0]
            print(tensors)
            key = _find_vector_column(tensors)
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
            return embedding_model_selection(file_path, "", ".txt")
            '''
            text = open(file_path, 'r', encoding='utf-8').read()
            #print(text)
            framedata = [[4096, 40960, "Alibaba Group", "Quite advanced"],[3072, 8192, "OpenAI", "Used in ChatGPT"],[4096, 32768, "Microsoft", "Used in Bing Search, and Even Copilot and RAG applications"],[1024, 512, "Microsoft", "Warning: pretty low token range"],[1024, 8192, "Beijing Academy of Artificial Intelligence", ""],[768,8192, "Nomic AI", "new, and focused on performance"],[4096,32768, "Nvidia", "Based on E5 Mistral 7B, optimized for performance and accuracy. Typically used in research rather than industry."],[2048,8192, "Nvidia", "Custom Nemotron architecture, good for efficiency"],[3072,8192, "Google", "Used in Google Gemini Models."]]
            df = pd.DataFrame(np.array(framedata))
            df.columns = ['Maximum Dims:','Maximum # Tokens:', 'Source:', 'Notes:']
            df.index = ['0.) Qwen3 Embedding 8B FP16: ', '1.) OpenAI V3 Large: ', '2.) E5 Mistral 7B FP16: ', '3.) E5 Large V2: ', '4.) BAAI BGE-M3: ', '5.) Nomic Embed V1.5 FP32: ', '6.) NV-Embed V2: ', '7.) Llama Nemotron Embedding 1B: ', '8.) Gemini Embedding 2'] #rows
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 150)
            print(df)
            print("NOTE: These embedding models won't work properly, due to either being paid, requiring API keys, or not running locally:")
            print("1.) OpenAI V3 Large\n6.) NV-Embed V2\n7.) Llama Nemotron Embedding 1B\n8.) Gemini Embedding 2\n")
            choice = int(input("Enter your number choice here: "))
            
            options = ["qwen3","openai-3-large","e5-mistral","e5-large-v2","bge-m3","nomic","nv-embed-v2","nemotron-embed","gemini-2"]
            return generate_local_embeddings(text, options[choice])
            '''
        
        # 8. .arrow or .feather; extremelly rare, but huggingface datasets sometimes use this
        elif ext == '.arrow' or ext == '.feather':
            df = feather.read_feather(file_path)
            key = _find_vector_column(df.iloc[0])
            return np.stack(df[key].values)

        else:
            print(f"Unsupported extension: {ext}")
            return None

    except Exception as e:
        print(f"Error extracting from {ext}: {e}")
        return None

def embedding_model_selection(file_path, Text, mode):
    import os
    import numpy as np
    import pandas as pd
    
    if(mode == ".txt"):
        text = open(file_path, 'r', encoding='utf-8').read()
    elif(mode == "dataset"):
        text = Text
    
    pool_mapping = {"qwen3": 2, "openai-3-large": 1, "e5-mistral": 2, "e5-large-v2": 1, "bge-m3": 2, "nomic": 1, "nv-embed-v2": 0, "nemotron-embed": 0, "gemini-2": 1, "gemma-2": 1, "snowflake-arctic-embed-v2.0": 2}
    dim_mapping = {"qwen3": 4096, "openai-3-large": 3072, "e5-mistral": 4096, "e5-large-v2": 1024, "bge-m3": 1024, "nomic": 768, "nv-embed-v2": 4096, "nemotron-embed": 2048, "gemini-2": 3072, "gemma-2": 3584, "snowflake-arctic-embed-v2.0": 1024}
    context_mapping = {"qwen3": 40960, "e5-mistral": 32768, "e5-large-v2": 512, "bge-m3": 8192, "nomic": 8192, "gemma-2": 8192, "snowflake-arctic-embed-v2.0": 8192}
    native_context_mapping = {"qwen3": 32768, "e5-mistral": 4096, "e5-large-v2": 512, "bge-m3": 8192, "nomic": 2048, "gemma-2": 8192, "snowflake-arctic-embed-v2.0": 8192}
    datatype_mapping = {"qwen3": np.float16, "e5-mistral": np.float16, "e5-large-v2": np.float32, "bge-m3": np.float16, "nomic": np.float32, "gemma-2": np.float16, "snowflake-arctic-embed-v2.0": np.float32}
    #print(text)
    framedata = [[4096, 40960, 32768, "CLS", "Alibaba Group", "Quite advanced"],[4096, 32768, 4096, "CLS", "Microsoft", "Used in Bing Search, and Even Copilot and RAG applications"],[1024, 512, 512, "Mean", "Microsoft", "Warning: pretty low token range"],[1024, 8192, 8192, "CLS", "Beijing Academy of Artificial Intelligence", ""],[768, 8192, 2048, "Mean", "Nomic AI", "new, and focused on performance"], [3584, 8192, 8192, "Mean(Adaptive)", "Google", "Open Source version of Google Gemini Embedding model"], [1024, 8192, 8192, "CLS", "Snowflake", "Commonly used for large-scale RAG in industry"]]
    df = pd.DataFrame(np.array(framedata))
    df.columns = ['Maximum Dims:','Maximum # Tokens:', 'Native Token Context:', 'Pooling Type:', 'Source:', 'Notes:']
    df.index = ['0.) Qwen3 Embedding 8B (FP16): ', '1.) E5 Mistral 7B (FP16): ', '2.) E5 Large V2 (FP32): ', '3.) BAAI BGE-M3 (FP16): ', '4.) Nomic Embed V1.5 (FP32): ', '5.) Gemma 2 Embedding 9B (FP16):', '6.) Snowflake Arctic Embedding L V2.0 (FP32):'] #rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    print(df)
    choice = int(input("Enter your number choice here: "))
            
    options = ["qwen3","e5-mistral","e5-large-v2","bge-m3","nomic","gemma-2","snowflake-arctic-embed-v2.0"]
    print("before generating local embeddings")
    
    #text formatting by pooling type
    '''
    pool_map = pool_mapping[options[choice]]
    if(pool_map == 0): #by token
        pass
    elif(pool_map == 1): #average by row
        pass
    elif(pool_map == 2): #CLS / last token
        pass
    '''
    BERT = False
    BERT_LIST = ['nomic']
    if(options[choice] in BERT_LIST):
        BERT = True
    return generate_local_embeddings(text, options[choice], pool_mapping[options[choice]], dim_mapping[options[choice]], context_mapping[options[choice]], native_context_mapping[options[choice]], BERT, datatype_mapping[options[choice]])

def dataset_embedding(file_path):
    import os
    import numpy as np
    import pandas as pd
    import json
    import zstandard as zstd
    import ml_dtypes
    import pyarrow.feather as feather
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
            print(".parquet KEYS --> Values: ")
            i = 0
            for k in df.keys():
                print(str(i) + ".) " + str(k) + " --> " + str(df[k].values))
                i += 1
            selec = int(input("Select the column to choose by number: "))
            text = df[df.keys()[selec]].values
            print("Before selecting embedding model")
            return embedding_model_selection(file_path, text, "dataset")

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
        elif ext == '.csv': #don't use this either
            df = pd.read_csv(file_path)
            # CSV vectors are often stored as strings like "[0.1, 0.2...]"
            # This converts those strings back into actual lists/arrays
            print([df.iloc[i] for i in range(8)])
            key = _find_vector_column(df.iloc[0])
            #print("CSV KEYS: ")
            #print(df.keys())
            if isinstance(df[key].iloc[0], str):
                df[key] = df[key].apply(json.loads)
            return np.stack(df[key].values)

        # 5. SafeTensors (Modern Model Weights)
        elif ext == '.safetensors': #don't use this at all. It is more for model weights than vector embeddings
            from safetensors.numpy import load_file
            tensors = load_file(file_path)
            # Safetensors usually have multiple keys; we return the first one or a specific one
            #key = list(tensors.keys())[0]
            print(tensors)
            key = _find_vector_column(tensors)
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
            return embedding_model_selection(file_path, "", ".txt")
            '''
            text = open(file_path, 'r', encoding='utf-8').read()
            #print(text)
            framedata = [[4096, 40960, "Alibaba Group", "Quite advanced"],[3072, 8192, "OpenAI", "Used in ChatGPT"],[4096, 32768, "Microsoft", "Used in Bing Search, and Even Copilot and RAG applications"],[1024, 512, "Microsoft", "Warning: pretty low token range"],[1024, 8192, "Beijing Academy of Artificial Intelligence", ""],[768,8192, "Nomic AI", "new, and focused on performance"],[4096,32768, "Nvidia", "Based on E5 Mistral 7B, optimized for performance and accuracy. Typically used in research rather than industry."],[2048,8192, "Nvidia", "Custom Nemotron architecture, good for efficiency"],[3072,8192, "Google", "Used in Google Gemini Models."]]
            df = pd.DataFrame(np.array(framedata))
            df.columns = ['Maximum Dims:','Maximum # Tokens:', 'Source:', 'Notes:']
            df.index = ['0.) Qwen3 Embedding 8B FP16: ', '1.) OpenAI V3 Large: ', '2.) E5 Mistral 7B FP16: ', '3.) E5 Large V2: ', '4.) BAAI BGE-M3: ', '5.) Nomic Embed V1.5 FP32: ', '6.) NV-Embed V2: ', '7.) Llama Nemotron Embedding 1B: ', '8.) Gemini Embedding 2'] #rows
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 150)
            print(df)
            print("NOTE: These embedding models won't work properly, due to either being paid, requiring API keys, or not running locally:")
            print("1.) OpenAI V3 Large\n6.) NV-Embed V2\n7.) Llama Nemotron Embedding 1B\n8.) Gemini Embedding 2\n")
            choice = int(input("Enter your number choice here: "))
            
            options = ["qwen3","openai-3-large","e5-mistral","e5-large-v2","bge-m3","nomic","nv-embed-v2","nemotron-embed","gemini-2"]
            return generate_local_embeddings(text, options[choice])
            '''
        
        # 8. .arrow or .feather; extremelly rare, but huggingface datasets sometimes use this
        elif ext == '.arrow' or ext == '.feather':
            df = feather.read_feather(file_path)
            key = _find_vector_column(df.iloc[0])
            return np.stack(df[key].values)

        else:
            print(f"Unsupported extension: {ext}")
            return None

    except Exception as e:
        print(f"Error extracting from {ext}: {e}")
        return None
    pass

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
        print("Not normalized. Normalization failed. The mean deviation is: " + str(verify))
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
    #print("TENS BITS___________: " + str(tens_bits))
    if(tens_bits == 32):
        view_type = torch.int32
    elif(tens_bits == 8):
        view_type = torch.int8
    elif(tens_bits == 16):
        view_type = torch.int16
    elif(tens_bits == 64):
        view_type = torch.int64
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
        elif num_bits == 64:
            view_type = torch.int64
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
    if(ACCELERATION_DEVICE == "xpu"):
        torch.xpu.empty_cache()
    elif(ACCELERATION_DEVICE == "cuda"):
        torch.cuda.empty_cache()
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

def bitplane_compression_benchmark(tensor_d, title): #replaces run_phased_benchmark
    import torch
    import multiprocessing
    import zstandard as zstd
    import pandas as pd
    import numpy
    import lz4.frame
    
    #clear xpu memory: this is especially important for very large tensors that fill your VRAM
    if(ACCELERATION_DEVICE == "xpu"):
        torch.xpu.empty_cache()
    elif(ACCELERATION_DEVICE == "cuda"):
        torch.cuda.empty_cache()
    
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
    ret = {"Concatenated Bits (ZSTD):": 0, "Concatenated Bits (LZ4):": 0, "Original Quantized Bits:": 0}
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
        ret["Original Quantized Bits:"] = total_size_bytes*8
        if(SHOW_NEGATIVE_COMPRESSION_RATIOS == False):
            concat_perc_saving_zstd = max(round((1 - (concat_zstd_bytes/total_size_bytes))*100, 3), 0.000)
            concat_perc_saving_lz4 = max(round((1 - (concat_lz4_bytes/total_size_bytes))*100, 3), 0.000)
            ret["Concatenated Bits (ZSTD):"] = min(concat_zstd_bytes, total_size_bytes)*8
            ret["Concatenated Bits (LZ4):"] = min(concat_lz4_bytes, total_size_bytes)*8
        else:
            concat_perc_saving_zstd = max(round((1 - (concat_zstd_bytes/total_size_bytes))*100, 3), 0.000)
            concat_perc_saving_lz4 = max(round((1 - (concat_lz4_bytes/total_size_bytes))*100, 3), 0.000)
            ret["Concatenated Bits (ZSTD):"] = concat_zstd_bytes*8
            ret["Concatenated Bits (LZ4):"] = concat_lz4_bytes*8
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
    print("Concat % Saving Overall (ZSTD) vs Quantized: " + str(concat_perc_saving_zstd) + ", Average: " + str(concat_zstd_avg))
    print("Concat % Saving Overall (LZ4) vs Quantized: " + str(concat_perc_saving_lz4) + ", Average: " + str(concat_lz4_avg))
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
    return ret #for further analytics

def space_saving_metrics(bit_comp_quant, scale_tens, raw_tens, quant_raw):
    import pandas as pd
    import numpy as np
    import torch
    framedata = [[],[],[],[]]
    #preparing for calculations
    scalar_bits = scale_tens.shape[0]*scale_tens.shape[1]*scale_tens.shape[2] #note: scale_tens must be the bitplane one
    global BASE_TYPE
    bitsize = 0
    BASE_TYPE = torch.as_tensor(np.array([], dtype=BASE_TYPE)).dtype
    if(BASE_TYPE.is_floating_point):
        bitsize += torch.finfo(BASE_TYPE).bits
    else:
        bitsize += torch.iinfo(BASE_TYPE).bits
    Raw_size = raw_tens.shape[0]*raw_tens.shape[1]*bitsize #note: raw_tens must be the initial tensor before any quantization or bitplaning
    print("DEBUG: DIRECT SIZE RIGHT BEFORE COMPRESSION: ")
    direct_size = direct_compression(raw_tens) #this is the size in bits of directly compressing raw_tens
    print("DEBUG: DIRECT QUANTIZED SIZE RIGHT BEFORE COMPRESSION: ")
    direct_quantized_size = direct_compression(quant_raw) #this is the size in bits of directly compressing quant_raw (without bitplanes)
    #most important: Compressed Bit Quant+Scale vs Original Prequantized Uncompressed
    #(ZSTD):
    framedata[0].append(round((1-((bit_comp_quant["Concatenated Bits (ZSTD):"]+scalar_bits)/Raw_size))*100, 3))
    #(LZ4):
    framedata[0].append(round((1-((bit_comp_quant["Concatenated Bits (LZ4):"]+scalar_bits)/Raw_size))*100, 3))
    #now do: Compressed Bit Quant+Scale vs Bit Quant+Scale Uncompressed
    #(ZSTD):
    framedata[1].append(round((1-((bit_comp_quant["Concatenated Bits (ZSTD):"]+scalar_bits)/(bit_comp_quant["Original Quantized Bits:"]+scalar_bits)))*100, 3))
    #(LZ4):
    framedata[1].append(round((1-((bit_comp_quant["Concatenated Bits (LZ4):"]+scalar_bits)/(bit_comp_quant["Original Quantized Bits:"]+scalar_bits)))*100, 3))
    #do the others later
    #vs Direct Compression
    #(ZSTD):
    framedata[2].append(round((1-((bit_comp_quant["Concatenated Bits (ZSTD):"]+scalar_bits)/direct_size[0]))*100, 3))
    #(LZ4):
    framedata[2].append(round((1-((bit_comp_quant["Concatenated Bits (LZ4):"]+scalar_bits)/direct_size[1]))*100, 3))
    #vs Quant Compressed + Scale (No bitplanes)
    #(ZSTD):
    framedata[3].append(round((1-((bit_comp_quant["Concatenated Bits (ZSTD):"]+scalar_bits)/(direct_quantized_size[0]+scalar_bits)))*100, 3))
    #(LZ4):
    framedata[3].append(round((1-((bit_comp_quant["Concatenated Bits (LZ4):"]+scalar_bits)/(direct_quantized_size[1]+scalar_bits)))*100, 3))
    #now print
    df = pd.DataFrame(np.array(framedata))
    df.columns = ['Concatenated % Savings (ZSTD):','Concatenated % Savings (LZ4):']
    df.index = ['vs Original Data Size:', 'vs Quantized+Scalar:', 'vs Direct Compression:', 'vs Quant Compressed+Scale (No BP):'] #rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    print(df)
    print("\n Extra: Uncompressed Bitplaned Quant+Scalar vs Uncompressed Bitplaned Original Tensor: " + str(round((1-((bit_comp_quant["Original Quantized Bits:"]+scalar_bits)/Raw_size))*100, 3)))

def direct_compression(tens):
    import zstandard as zstd
    import numpy as np
    import lz4.frame
    import torch
    
    if(torch.is_tensor(tens) == False):
        tens = torch.from_numpy(tens).to(ACCELERATION_DEVICE)
    
    #get the raw bytes
    rawbytes = tens.cpu().numpy().tobytes()
    print("DIRECT BYTES: " + str(len(rawbytes)))
    
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
    #clear xpu memory: this is especially important for very large tensors that fill your VRAM
    if(ACCELERATION_DEVICE == "xpu"):
        torch.xpu.empty_cache()
    elif(ACCELERATION_DEVICE == "cuda"):
        torch.cuda.empty_cache()
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
    #print("\nNOTE: You must make sure that the file you are choosing is a pre-computed non-compressed vector embedding of numeric values, (or else the program will fail)\n")
    ms = int(input("Input path number here: "))
    testfile = files[ms]
    #join it together
    global file_path
    file_path = os.path.join(base_directory, testfile)
    #print(file_path)
    computation_choice = str(input("Is this data precomputed?[Y/N]: ")).lower()
    if(computation_choice == "y"):
        result = extract_vectors(file_path)
    elif(computation_choice == "n"):
        result = dataset_embedding(file_path)
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
    if(IGNORE_RAW == False):
        raw_bitplane = direct_bitplane(normalized_tensor)
        print("Raw Bitplanes: " + str(raw_bitplane.shape))
        print(raw_bitplane)
    
    #stage 4.5 optional: RLE compression analysis
    if(PRINT_RLE == True):
        print("\nDEBUG: RLE Compression analysis on QUANT Horizontal: ")
        rle_res = RLE_bitplane_compressibility(quantized_bitplane,"Quant-Horizontal")
        if(IGNORE_RAW == False):
            print("\nDEBUG: RLE Compression analysis on DIRECT Horizontal: ")
            rle_raw = RLE_bitplane_compressibility(raw_bitplane,"Direct-Horizontal")
        #print(rle_res)
    
    #stage 5+6: bitpacking (horizontally, packed_b) + LZ4 & ZSTD Bitplane compression (with metrics)
    print("\nSTAGE 5+6: Bitpacking + LZ4 & ZSTD Compression: ")
    quant_comp_results = bitplane_compression_benchmark(quantized_bitplane, "bitplane quant horizontal")
    #now for some extra metrics:
    space_saving_metrics(quant_comp_results, scalar_bitplane, normalized_tensor, quantized)
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
    
    #raw
    if(IGNORE_RAW == False):
        bitplane_compression_benchmark(raw_bitplane, "bitplane direct horizontal")
        if(PRINT_DELTA_TRANSFORMATION == True):
            tensor_delta_raw = raw_bitplane.clone()
            tensor_delta_raw[:, 1:] = raw_bitplane[:, 1:] - raw_bitplane[:, :-1]
            bitplane_compression_benchmark(tensor_delta_raw, "delta transformation - horizontal raw")
    pass

if  __name__ == "__main__":
    import numpy
    import torch
    
    #then call simulation
    run_simulation()
    