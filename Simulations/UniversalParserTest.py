#We should be able to handle these file types:
#

import os
base_directory = os.path.dirname(os.path.abspath(__file__))
file_path = ""
NUM_ROWS = 0

#some functions
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
                            vectors.append(data[vector_column_name])
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
    ms = int(input("Input path number here: "))
    testfile = files[ms]
    #join it together
    global file_path
    file_path = os.path.join(base_directory, testfile)
    print(file_path)
    res = extract_vectors(file_path)
    print(res)
    
if __name__ == '__main__':
    initialization() #call it
    #To do 2_11+_26:
    # - Finish working on the vector embedding extraction (heuristic?)
    # - start preparing for shape formatting and bitplane stuff
    # - maybe direct compression and block size heuristics
    pass