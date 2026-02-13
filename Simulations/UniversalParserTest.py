#We should be able to handle these file types:
#

import os
base_directory = os.path.dirname(os.path.abspath(__file__))
import torch
file_path = ""
NUM_ROWS = 0
NUM_DIMS = 768 #768 is default, but will automatically change with shape recognition
MAX_ROWS = 10000 #10k is default, but will automatically change with shape recognition

#quantization settings
BASE_TYPE = torch.float32 #torch.float32 is default, will autochange during shape detection
QUANTIZATION_TYPE = torch.int8 #You change this value, this is the resulting quantized values
SCALE_TYPE = torch.float32 #torch.float32 is detault, will autochange during quantization

#performance settings
ACCELERATION_DEVICE = "xpu"

#debug settings
PRINT_DEBUG = False

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
    scales = mxvals / (float((2**bitsize)/2) - 1) #for now
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
def rowbinary(row, typ): #Dont use this
    if(typ == "quant"):
        global QUANTIZATION_TYPE
        qmat = '0'
        if(QUANTIZATION_TYPE.is_floating_point):
            qmat += torch.finfo(QUANTIZATION_TYPE).bits
        else:
            qmat += torch.iinfo(QUANTIZATION_TYPE).bits
        qmat += 'b'
        return [format(x, qmat) for x in row]
    elif(typ == "scale"):
        global SCALE_TYPE
        smat = '0' + str(torch.finfo(SCALE_TYPE).bits) + 'b'
        return [format(x, smat) for x in row]

def rowbinary_quant(row):
    global QUANTIZATION_TYPE
    qmat = '0'
    if(QUANTIZATION_TYPE.is_floating_point):
        qmat += torch.finfo(QUANTIZATION_TYPE).bits
    else:
        qmat += torch.iinfo(QUANTIZATION_TYPE).bits
    qmat += 'b'
    return [format(x, qmat) for x in row]

def rowbinary_scale(row):
    global SCALE_TYPE
    smat = '0' + str(torch.finfo(SCALE_TYPE).bits) + 'b'
    return [format(x, smat) for x in row]

def binarytolist(element):
    return [int(x) for x in element]

def binaryvectortolist(row):
    return [[int(x) for x in y] for y in row]

def horizontal_bitplane():
    pass

def vertical_bitplane():
    pass

#Compression Analysis Functions

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
    res = extract_vectors(file_path)
    print("\nORIGINAL VECTOR EMBEDDING DATA: " + str(res.shape))
    print(res)
    #now adjust the NUM_DIMS and MAX_ROWS parameters, will be useful later
    global MAX_ROWS
    global NUM_DIMS
    MAX_ROWS, NUM_DIMS = res.shape
    print("MAX_ROWS: " + str(MAX_ROWS))
    print("NUM_DIMS: " + str(NUM_DIMS))
    import torch
    import numpy
    global BASE_TYPE
    BASE_TYPE = res.dtype
    print("BASE_TYPE: " + str(BASE_TYPE))
    #get it into a tensor
    global ACCELERATION_DEVICE
    numpy_emb = numpy.stack(res)
    tens_emb = torch.tensor(numpy_emb).to(ACCELERATION_DEVICE)
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
    
    #now next step, is to do binary conversions
    
    
if __name__ == '__main__':
    initialization() #call it
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