#Wikipedia DPR paths
#/"Wikipedia DPR Mini"/
#Index File: /"Wikipedia DPR Mini"/"Index File"/psgs_w100.multiset.HNSW128_SQ8-IP-train.faiss
#note: we haven't tested compression of the index file, it might not work for that
#Multiset: /"Wikipedia DPR Mini"/Multiset/
#--> train-00156-of-00157.parquet
#--> train-00078-of-00157.parquet
#--> train-00000-of-00157.parquet
#DummyMultiset: /"Wikipedia DPR Mini"/DummyMultiset/train-00000-of-00001.parquet

#Global variables, these are like tuning parameters for this "simulation" that analyzes compressibility of vector embeddings data
WikipediaDPR_path = "C:/Users/findi/OneDrive/Desktop/AI Research Datasets/Wikipedia DPR Mini/" #base directory of Wikipedia DPR mini dataset
testfile = WikipediaDPR_path + "Multiset/train-00000-of-00157.parquet" #current file selection

#New Source settings 2/1/26+ [note: these 3 options only take effect when DATA_SELECTION_MODE = 3]
DATA_SELECTION_METHOD = True
#This determines if you directly choose the source file or if it is randomly chosen
#True - [Default] Manual Selection
#False - Random Selection

RAND_BLOCK = True
#This determines whether it chooses a random block for automation
#True - [Random] will select a random valid starting location in the selected file
#False - [Default] uses specified START_IND for starting location of block in selected file
START_IND = 0 #starting index of block that is scanned

NUM_ROWS = 768 #this is how many rows of vector embeddings will be processed
NUM_DIMS = 768 #this is the number of weights per vector embedding

#some settings
CALCULATE_ENTROPY = True #determines whether to calculate entropy [False, Default]
ENTROPY_MODEL = 0 #0-2, only 0 works for now
#ENTROPY_MODEL determines how entropy is calculated on your data
#ENTROPY_MODEL has multiple settings:
#0 - [General Entropy, Default] entropy of each of the 32 bit planes
#1 - [Marginal Entropy] entropy for each of the 32 bit planes (32 entropy values)
#2 - [Conditional(Context) Entropy] entropy context between the bit planes
#It will only go into effect if CALCULATE_ENTROPY = True

AC_MODE = 0 #0-4
#AC_MODE is the data orientation that you want to perform Arithmetic Coding on
#AC_MODE has multiple settings
#0 - [ALL, default] will print out all analytics regarding AC compressibility
#1 - [Theoretical] will print out all theoretical analytics regarding AC compressibility ratios
#2 - [All Row Bitplane Analysis] will print out the compressibility of all rows into 1 value for each bit plane (768 tall for vector dimension, 32 wide for bit plane bit number)
#3 - [Individual Row Bitplane Analysis] will print out compressibility of each row with 32 compressibility values
#4 - [OFF]

DATA_SELECTION_MODE = 3 #0-3
#DATA_SELECTION_MODE specifies how files and data are selected for the simulation
#DATA_SELECTION_MODE has multiple settings:
#0 - [Manual Selection] will only use the data you specify
#1 - [File Selection - Random Block] will use the data file you specify, but will choose a random block of data within the file
#2 - [Fully Random] will pick a random data file in your local dataset, then also pick a random block
#3 - [Full Automation, default] will utilize file scanning locally or through internet streaming. This makes DATA_SELECTION_METHOD actually do something
#new note: only use 0-2 when autoscan isn't finding your file (you would need to manually specify the file path)

#helper functions below, they are used in parallel processing on the CPU side (and GPU side) to accelerate formatting and cast conversions

DIRECT_COMPRESSION_COMPARISON = True
#False - [Default] No Direct Compression analytics for LZ4 and ZSTD after bitplane compression
#True - [ON] Shows Direct Compression analytics for LZ4 and ZSTD after bitplane compression

#Performance settings
#NOTE: this is highly experimental, as CUDA and CPU were not tested on this build, only "xpu" which is intel arc acceleration
ACCELERATION_DEVICE = "xpu"
#ACCELERATION_DEVICE has multiple options
#"xpu" - [Default] Intel Arc GPU + XMX acceleration, this is what the simulation was developed on
#"cuda" - [Nvidia GPU + Tensor Core acceleration] not tested, use at your own risk
#"cpu" - [CPU multicore acceleration] much slower than the other options, use if you don't have a compatible GPU. not tested, use at your own risk

#helper functions below
def rowbinary(row):
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

#how to interpret the results:
#Density: P
# P = 0.5 (balanced): random noise, not compressible
# P = 0 (sparse): plane is mostly 0s. common when MSBs are small
# P = 1 (saturated): plane is mostly 1s. common when MSBs are either very large or negative (2s complement)
#Theoretical Entropy: H in bpS
# H = 1.0 bpS : max disorder, random noise, low compressibility (0%, arithmetic coding won't help)
# H = 0.8 bpS : modest bias, 1s : 0s is a 70 : 30 split (20% reduction with arithmetic coding)
# H = 0.1 bpS : high predictability, plane is nearly all 0s or all 1s (90% reduction with arithmetic coding)
# H = 0.0 bpS : static, every single bit in that bit plane is the same (~100% reduction with arithmetic coding)
def ThreeD_AC_Context(matrix): #theoretical entropy of each of the 32 bit planes in 3 dimensions
    import torch
    import numpy
    from multiprocessing import Pool
    
    if(isinstance(matrix, numpy.ndarray)):
        bitvolume = torch.from_numpy(matrix).to(ACCELERATION_DEVICE)
        plane_densities = bitvolume.to(torch.float32).mean(dim=(0, 2))
    else:
        bitvolume = matrix.to(ACCELERATION_DEVICE)
        plane_densities = bitvolume.to(torch.float32).mean(dim=(1))
    for ind, density in enumerate(plane_densities):
        p = density.item()
        entropy = 0 if p <= 0 or p >= 1 else -(p * torch.log2(torch.tensor(p)) + (1-p) * torch.log2(torch.tensor(1-p)))
        print(f"Bit Plane {ind}: Density {p:.4f}, Theoretical Entropy {entropy:.4f} bpS")
    
#What do the results mean?: BTW, this is for theoretical compressibility per bit plane for Arithmetic Coding
#Marginal Ratio (1:Hmarg): This is the compression you get if each plane is encoded individually using a simple arithmetic coder
#EX: 10.50:1 means very sparse bit plane, only need ~1/10th the space to store it
#Contextual Ratio (1:Hcontext): This is the compression you get if your arithmetic coder "remembers" the bit it just saw in the more significant plane
#--> Gain: If Marginal is 2:1 but Contextual is 4:1, then it means there is a strong correlation between bit planes. Using the bit above as "context" doubles your storage efficiency
#The MSB (0) vs LSB (31) pattern:
#--> your higher msb stuff should have higher ratios than at msb where it drops off towards 1.0:1, which is effectively noise
def AC_Analysis(matrix): #all vibe coded, should probably test out next time
    import torch
    import numpy as np
    device = ACCELERATION_DEVICE
    
    if(isinstance(matrix, np.ndarray)):
        bit_volume = torch.from_numpy(matrix).to(device)
        num_dims, num_planes, num_rows = bit_volume.shape
        # Total bits in one plane across all vectors and dimensions
        bits_per_plane = num_dims * num_rows
    else:
        bit_volume = matrix.to(device)
        num_planes, num_dims = bit_volume.shape
        num_rows = 1
        # Total bits in one plane across all vectors and dimensions
        bits_per_plane = num_dims * num_rows

    print(f"{'Plane':<6} | {'Marginal Ratio':<15} | {'Contextual Ratio':<18} | {'Savings %'}")
    print("-" * 65)
    
    if(num_rows > 1):
        for i in range(num_planes):
            # 1. Marginal Entropy H(Xi)
            p1 = torch.mean(bit_volume[:, i, :].to(torch.float32)).item()
            h_marg = 0.0 if p1 <= 0 or p1 >= 1 else -(p1 * np.log2(p1) + (1-p1) * np.log2(1-p1))
        
            # Calculate Marginal Ratio
            # If entropy is 0, ratio is infinite (effectively perfectly compressible)
            #marg_ratio = 1.0 / h_marg if h_marg > 0.0001 else 1000.0
            marg_ratio = 1.0 / h_marg
        
            # 2. Contextual Entropy H(Xi | Xi+1)
            if i < num_planes - 1:
                # Combine current plane i and next plane i+1 to find joint probability
                # Xi is current, Xi+1 is the context (more significant bit)
                joint_state = (bit_volume[:, i, :].to(torch.int32) << 1) | bit_volume[:, i+1, :].to(torch.int32)
                counts = torch.bincount(joint_state.flatten(), minlength=4).to(torch.float32)
                probs = counts / counts.sum()
            
                h_joint = -torch.sum(probs * torch.log2(probs + 1e-9)).item()
            
                # H(Xi+1)
                p_next1 = torch.mean(bit_volume[:, i+1, :].to(torch.float32)).item()
                h_next = 0.0 if p_next1 <= 0 or p_next1 >= 1 else -(p_next1 * np.log2(p_next1) + (1-p_next1) * np.log2(1-p_next1))
            
                h_context = h_joint - h_next
            else:
                h_context = h_marg # MSB has no context above it

            # Calculate Contextual Ratio
            #cont_ratio = 1.0 / h_context if h_context > 0.0001 else 1000.0
            cont_ratio = 1.0 / h_context
        
            # Overall savings compared to uncompressed
            savings = (1.0 - h_context) * 100
        
            print(f"{i:<6} | {marg_ratio:<15.2f}:1 | {cont_ratio:<18.2f}:1 | {max(0, savings):.1f}%")
    elif(num_rows == 1):
        for i in range(num_planes):
            # 1. Marginal Entropy H(Xi)
            p1 = torch.mean(bit_volume[i, :].to(torch.float32)).item()
            h_marg = 0.0 if p1 <= 0 or p1 >= 1 else -(p1 * np.log2(p1) + (1-p1) * np.log2(1-p1))
        
            # Calculate Marginal Ratio
            # If entropy is 0, ratio is infinite (effectively perfectly compressible)
            #marg_ratio = 1.0 / h_marg if h_marg > 0.0001 else 1000.0
            marg_ratio = 1.0 / h_marg
        
            # 2. Contextual Entropy H(Xi | Xi+1)
            if i < num_planes - 1:
                # Combine current plane i and next plane i+1 to find joint probability
                # Xi is current, Xi+1 is the context (more significant bit)
                joint_state = (bit_volume[i, :].to(torch.int32) << 1) | bit_volume[i+1, :].to(torch.int32)
                counts = torch.bincount(joint_state.flatten(), minlength=4).to(torch.float32)
                probs = counts / counts.sum()
            
                h_joint = -torch.sum(probs * torch.log2(probs + 1e-9)).item()
            
                # H(Xi+1)
                p_next1 = torch.mean(bit_volume[i+1, :].to(torch.float32)).item()
                h_next = 0.0 if p_next1 <= 0 or p_next1 >= 1 else -(p_next1 * np.log2(p_next1) + (1-p_next1) * np.log2(1-p_next1))
            
                h_context = h_joint - h_next
            else:
                h_context = h_marg # MSB has no context above it

            # Calculate Contextual Ratio
            #cont_ratio = 1.0 / h_context if h_context > 0.0001 else 1000.0
            cont_ratio = 1.0 / h_context
        
            # Overall savings compared to uncompressed
            savings = (1.0 - h_context) * 100
        
            print(f"{i:<6} | {marg_ratio:<15.2f}:1 | {cont_ratio:<18.2f}:1 | {max(0, savings):.1f}%")

#gets actual AC compressibility of a bit plane slice, so the compressibility of each bit plane of all vector embeddings
#what do the results mean?
#Density Bias: If plane has density 0.1 or 0.9, the Static AC Ratio will be arund 2:1. This means that just knowing the average number of 1s allows you to cut the storage in half
#Predictive Boost: If Predictive Ratio is much higher than Static ratio, then it proves that the bit planes within each vector embedding are vertically correlated
#the "Similarity" Factor: Since similarity is calculated on these planes, a high predictive ratio suggests that dimensions 768 aren't independent. This can be used to skip encoding redundant dimensions
def AC_Analysis_BitPlaneSlice(bit_volume): #this is vibe coded
    import numpy as np
    import torch
    device = ACCELERATION_DEVICE
    
    if(isinstance(bit_volume, np.ndarray)):
        bit_volume = torch.from_numpy(bit_volume).to(device)
        num_dims, num_planes, num_rows = bit_volume.shape
    else:
        bit_volume = bit_volume.to(device)
        num_planes, num_dims = bit_volume.shape
        num_rows = 1
    bits_per_plane = num_dims * num_rows
    
    print(f"{'Bit-Plane':<10} | {'Density':<10} | {'Static AC Ratio':<18} | {'Predictive AC Ratio'}")
    print("-" * 75)
    
    if(num_rows > 1):
        for i in range(num_planes):
            # Extract the current plane
            plane = bit_volume[:, i, :].to(torch.float32)
        
            # 1. Static Probability (Probability of '1' in this plane)
            p1 = torch.mean(plane).item()
            p0 = 1.0 - p1
        
            # Calculate Static Arithmetic Coding Length (Shannon limit)
            # Length = -Sum(count_i * log2(prob_i))
            if p1 <= 0 or p1 >= 1:
                static_bits = 1.0 # Negligible (header only)
            else:
                static_bits = -( (p1 * bits_per_plane * np.log2(p1)) + 
                                 (p0 * bits_per_plane * np.log2(p0)) )
        
            static_ratio = bits_per_plane / static_bits
        
            # 2. Predictive/Contextual Ratio (Using Plane i+1 as context)
            if i < num_planes - 1:
                context_plane = bit_volume[:, i+1, :].to(torch.float32)
            
                # Find P(Bit=1 | Context=0) and P(Bit=1 | Context=1)
                c0_mask = (context_plane == 0)
                c1_mask = (context_plane == 1)
            
                p1_given_c0 = torch.mean(plane[c0_mask]).item() if torch.any(c0_mask) else 0.5
                p1_given_c1 = torch.mean(plane[c1_mask]).item() if torch.any(c1_mask) else 0.5
            
                # Calculate total bits for this context model
                # Bits = -(Sum of logs of probabilities for every bit based on its context)
                # This simulates a real adaptive Arithmetic Coder
                h_c0 = -(p1_given_c0 * np.log2(p1_given_c0 + 1e-9) + (1-p1_given_c0) * np.log2(1-p1_given_c0 + 1e-9))
                h_c1 = -(p1_given_c1 * np.log2(p1_given_c1 + 1e-9) + (1-p1_given_c1) * np.log2(1-p1_given_c1 + 1e-9))
            
                predictive_bits = (h_c0 * torch.sum(c0_mask).item()) + (h_c1 * torch.sum(c1_mask).item())
                predictive_ratio = bits_per_plane / predictive_bits
            else:
                predictive_ratio = static_ratio # No higher plane context
            
            print(f"Plane {i:>2}      | {p1:<10.4f} | {static_ratio:<18.2f}:1 | {predictive_ratio:.2f}:1")
    elif(num_rows == 1):
        for i in range(num_planes):
            # Extract the current plane
            plane = bit_volume[i, :].to(torch.float32)
        
            # 1. Static Probability (Probability of '1' in this plane)
            p1 = torch.mean(plane).item()
            p0 = 1.0 - p1
        
            # Calculate Static Arithmetic Coding Length (Shannon limit)
            # Length = -Sum(count_i * log2(prob_i))
            if p1 <= 0 or p1 >= 1:
                static_bits = 1.0 # Negligible (header only)
            else:
                static_bits = -( (p1 * bits_per_plane * np.log2(p1)) + 
                                 (p0 * bits_per_plane * np.log2(p0)) )
        
            static_ratio = bits_per_plane / static_bits
        
            # 2. Predictive/Contextual Ratio (Using Plane i+1 as context)
            if i < num_planes - 1:
                context_plane = bit_volume[i+1, :].to(torch.float32)
            
                # Find P(Bit=1 | Context=0) and P(Bit=1 | Context=1)
                c0_mask = (context_plane == 0)
                c1_mask = (context_plane == 1)
            
                p1_given_c0 = torch.mean(plane[c0_mask]).item() if torch.any(c0_mask) else 0.5
                p1_given_c1 = torch.mean(plane[c1_mask]).item() if torch.any(c1_mask) else 0.5
            
                # Calculate total bits for this context model
                # Bits = -(Sum of logs of probabilities for every bit based on its context)
                # This simulates a real adaptive Arithmetic Coder
                h_c0 = -(p1_given_c0 * np.log2(p1_given_c0 + 1e-9) + (1-p1_given_c0) * np.log2(1-p1_given_c0 + 1e-9))
                h_c1 = -(p1_given_c1 * np.log2(p1_given_c1 + 1e-9) + (1-p1_given_c1) * np.log2(1-p1_given_c1 + 1e-9))
            
                predictive_bits = (h_c0 * torch.sum(c0_mask).item()) + (h_c1 * torch.sum(c1_mask).item())
                predictive_ratio = bits_per_plane / predictive_bits
            else:
                predictive_ratio = static_ratio # No higher plane context
            
            print(f"Plane {i:>2}      | {p1:<10.4f} | {static_ratio:<18.2f}:1 | {predictive_ratio:.2f}:1")

#gets actual AC compressibility of each bit plane per vector embedding
#what do the results mean?:
#Vector Sparsity: If certain vectors have high ratios across many planes, they probably have many near-zero values. These vectors cost less to store in arithmetic coding
#Predictive Indexing: if top 5% most compressible vectors all share high ratio in same bit plane, you can use that specific bit plane as a filter or pre-index for similarity search
#Outlier Detection: Vectors with a ratio very close to 1.0:1 across all planes are "high-entropy" (very noisy) and cannot be compressed.
def AC_Analysis_BitPlane(bit_volume): #this is vibe coded, per vector compressibility
    import torch
    import numpy as np
    device = ACCELERATION_DEVICE
    
    # Move to Intel Arc
    # We transpose to (13348, 32, 768) to process vectors in parallel
    if(isinstance(bit_volume, np.ndarray)):
        v_volume = torch.from_numpy(bit_volume).permute(2, 1, 0).to(device).to(torch.float32)
        num_vectors, num_planes, num_dims = v_volume.shape
    else:
        v_volume = bit_volume.permute(1, 0).to(device).to(torch.float32)
        num_planes, num_dims = v_volume.shape
        num_vectors = 1
    
    # 1. Calculate P(1) for every bit plane of every vector
    # Shape: (13348, 32)
    if(num_vectors > 1):
        p1 = torch.mean(v_volume, dim=2)
    elif(num_vectors == 1):
        p1 = torch.mean(v_volume, dim=1)
    p0 = 1.0 - p1
    
    # 2. Calculate Binary Entropy for every plane/vector combination
    # H = -(p1*log2(p1) + p0*log2(p0))
    # We use a small epsilon to avoid log(0)
    eps = 1e-9
    entropy_matrix = -(p1 * torch.log2(p1 + eps) + p0 * torch.log2(p0 + eps))
    
    # 3. Convert Entropy to Theoretical AC Ratios
    # Ratio = 1 / Entropy (clamped to 1.0)
    ratios = 1.0 / torch.clamp(entropy_matrix, min=0.001)
    
    # Average ratio per vector across all 32 planes
    if(num_vectors > 1):
        avg_vector_ratio = torch.mean(ratios, dim=1)
    elif(num_vectors == 1):
        avg_vector_ratio = torch.mean(ratios, dim=0)
    
    # Find the most and least compressible vectors
    max_comp_idx = torch.argmax(avg_vector_ratio).item()
    min_comp_idx = torch.argmin(avg_vector_ratio).item()
    
    print(f"Analysis complete for {num_vectors} vectors.")
    if(num_vectors > 1):
        print(f"Most compressible vector index: {max_comp_idx} (Ratio: {avg_vector_ratio[max_comp_idx]:.2f}:1)")
        print(f"Least compressible vector index: {min_comp_idx} (Ratio: {avg_vector_ratio[min_comp_idx]:.2f}:1)")
    
    return avg_vector_ratio.cpu(), ratios.cpu()

#for performance reasons, past print statements will be commented out when not needed. for debugging purposes
def initialization():
    import pandas
    import random
    if((DATA_SELECTION_MODE > 0) and (DATA_SELECTION_MODE != 3)):
        global testfile
        selec = random.randint(0,3)
        choices = [WikipediaDPR_path + "Multiset/train-00156-of-00157.parquet", WikipediaDPR_path + "Multiset/train-00078-of-00157.parquet", WikipediaDPR_path + "Multiset/train-00000-of-00157.parquet", WikipediaDPR_path + "DummyMultiset/train-00000-of-00001.parquet"]
        testfile = choices[selec]
        print("File used: " + testfile)
    elif(DATA_SELECTION_MODE == 3):
        if(DATA_SELECTION_METHOD == True): #Manual file selection (autoscan)
            #local relative file autoscan, scans nearby folders for .parquet files
            from pathlib import Path
            parquetfiles = list(Path('.').rglob('*.parquet'))
            print("parquet files found, select via number: ")
            ind = 0
            while(ind < len(parquetfiles)):
                print(str(ind) + ".) " + str(parquetfiles[ind]))
                ind += 1
            ms = int(input("Input path number here: "))
            testfile = parquetfiles[ms]
        elif(DATA_SELECTION_METHOD == False): #Randomized file selection (also with autoscan)
            #local randomized file selection (autoscans local directories)
            from pathlib import Path
            import random
            parquetfiles = list(Path('.').rglob('*.parquet'))
            ms = random.randint(0, len(parquetfiles)-1)
            testfile = parquetfiles[ms]

    #allows to select specific columns to save memory/time
    datafile_subset = pandas.read_parquet(testfile, columns=['embeddings'])

    import sys
    import torch #Note: For intel arc systems, Pytorch 2.9.1+xpu build is required, if before that, it must import intel IPEX
    import numpy
    from multiprocessing import Pool

    #GPU does initial parallel bit pattern conversions (to binary 32 bit int)
    num_rows = len(datafile_subset['embeddings'])
    num_cols = len(datafile_subset['embeddings'][0])
    ind = 0
    
    import struct
    
    if(NUM_ROWS > 0): #does the data parsing
        import random
        if(DATA_SELECTION_MODE == 2):
            startind = random.randint(0,len(datafile_subset['embeddings']) - NUM_ROWS - 1)
            stockdata = datafile_subset['embeddings'][startind:startind + NUM_ROWS]
            print("Block used: [" + str(startind) + ":" + str(startind+NUM_ROWS) + "], #Vector embeddings calculated: " + str(NUM_ROWS))
        elif(DATA_SELECTION_MODE == 3):
            if(RAND_BLOCK == True):
                startind = random.randint(0,len(datafile_subset['embeddings']) - NUM_ROWS - 1)
                stockdata = datafile_subset['embeddings'][startind:startind + NUM_ROWS]
                print("Block used: [" + str(startind) + ":" + str(startind+NUM_ROWS) + "], #Vector embeddings calculated: " + str(NUM_ROWS))
            elif(RAND_BLOCK == False):
                global START_IND
                stockdata = datafile_subset['embeddings'][START_IND:START_IND + NUM_ROWS + 1]
                print("Block used: [" + str(START_IND) + ":" + str(START_IND+NUM_ROWS-1) + "], #Vector embeddings calculated: " + str(NUM_ROWS))
        else:
            stockdata = datafile_subset['embeddings'][:NUM_ROWS]
        
        matrix_numpy = numpy.stack(stockdata)
        matrix_gpu = torch.from_numpy(matrix_numpy).to(ACCELERATION_DEVICE)
        bit_patterns_gpu = matrix_gpu.view(torch.uint32)
        bit_patterns_cpu = bit_patterns_gpu.cpu().numpy()
        
        #converts it to binary
        binstring = Pool().map(rowbinary, bit_patterns_cpu)
        #print(binstring)
    
    #Split into bit planes
    #use GPU acceleration tensor cores (XMX cores) to transpose the binstring into a bitplane matrix
    if(NUM_ROWS == 1): #this is for horizontal bit plane compression of a single vector embedding
        #first, use cpu parallel acceleration to split each element into its own list
        vector_embedding = binstring[0]
        binsplit = Pool().map(binarytolist, vector_embedding)
        #print(binsplit)#works
        
        #now we need to transpose the matrix
        Matrix = torch.tensor(binsplit)
        matrixXPU = Matrix.to(ACCELERATION_DEVICE)
        bitplanes = matrixXPU.T
        #print(bitplanes.to("cpu"))#works
        
        #now analyze the compressibility of the bitplanes
        comp = horizontal_bitplane_compressibility(bitplanes)
        print("\nSINGLE VECTOR HORIZONTAL BIT PLANE COMPRESSION RATIOS [Run Length Encoding](-->0 good): ")
        print(comp)
        print("\nexplanation of results above:\n - leftmost column is MSB\n - rightmost column is LSB\n - each row represents 1 of the 768 dimensions in a single vector embedding\n - compression ratio -> 0 means perfect compressibility, >1 means negative compressibility (takes more space to compress it than original)")
        
        if(CALCULATE_ENTROPY == True):
            if(ENTROPY_MODEL == 0):
                print("\nENTROPY RESULTS: ")
                ThreeD_AC_Context(bitplanes)
                print("\nHow to interpret the results above:\n - P = Density [0 sparse | 0.5 balanced | 1 saturated]\n - H = Theoretical Entropy (bpS)\n --> H = 1.0 bpS : max disorder, random noise, low compressibility (0%, arithmetic coding won't help)\n --> H = 0.8 bpS : modest bias, 1s : 0s is a 70 : 30 split (20% reduction with arithmetic coding)\n --> H = 0.1 bpS : high predictability, plane is nearly all 0s or all 1s (90% reduction with arithmetic coding)\n --> H = 0.0 bpS : static, every single bit in that bit plane is the same (~100% reduction with arithmetic coding)")
        if(AC_MODE != 4):
            if(AC_MODE == 0):
                print("\nTHEORETICAL AC COMPRESSIBILITY: ")
                AC_Analysis(bitplanes)
                print("\nWhat do the results above mean?:\n - Marginal Ratio (1:Hmarg): This is the compression you get if each plane is encoded individually using a simple arithmetic coder\n --> EX: 10.50:1 means very sparse bit plane, only need ~1/10th the space to store it\n - Contextual Ratio (1:Hcontext): This is the compression you get if your arithmetic coder remembers the bit it just saw in the more significant plane\n --> Gain: If Marginal is 2:1 but Contextual is 4:1, then it means there is a strong correlation between bit planes. Using the bit above as context doubles your storage efficiency\n - MSB (0) vs LSB (31)\n --> your higher msb stuff should have higher ratios than at msb where it drops off towards 1.0:1, which is effectively noise")
                print("\nAC COMPRESSIBILITY PER BIT PLANE FOR ALL ROWS: ")
                AC_Analysis_BitPlaneSlice(bitplanes)
                print("\nWhat do the results above mean?:\n - Density Bias: we want either very close to 0 or very close to 1 \n - Predictive Boost: if Predictive Ratio >> Static Ratio, then bit planes within each vector embedding are vertically correlated\n - Similarity Factor: High predictive ratio suggests that the dimensions within a single vector embedding are similar, hence highly compressible")
                print("\nAC COMPRESSIBILITY PER BIT PLANE PER ROW: ")
                AC_Analysis_BitPlane(bitplanes)
                print("\nWhat do the results above mean?:\n - Vector Sparsity: vectors with high ratios across many planes are highly compressible\n - Predictive Indexing: the most compressible vectors could be used to shorten the similarity search through pre-indexing\n - Outlier Detection: anything with a ratio 1.0:1 across all planes are high entropy noise that cannot be compressed effectively")
            elif(AC_MODE == 1):
                print("\nTHEORETICAL AC COMPRESSIBILITY: ")
                AC_Analysis(bitplanes)
                print("\nWhat do the results above mean?:\n - Marginal Ratio (1:Hmarg): This is the compression you get if each plane is encoded individually using a simple arithmetic coder\n --> EX: 10.50:1 means very sparse bit plane, only need ~1/10th the space to store it\n - Contextual Ratio (1:Hcontext): This is the compression you get if your arithmetic coder remembers the bit it just saw in the more significant plane\n --> Gain: If Marginal is 2:1 but Contextual is 4:1, then it means there is a strong correlation between bit planes. Using the bit above as context doubles your storage efficiency\n - MSB (0) vs LSB (31)\n --> your higher msb stuff should have higher ratios than at msb where it drops off towards 1.0:1, which is effectively noise")
            elif(AC_MODE == 2):
                print("\nAC COMPRESSIBILITY PER BIT PLANE FOR ALL ROWS: ")
                AC_Analysis_BitPlaneSlice(bitplanes)
                print("\nWhat do the results above mean?:\n - Density Bias: we want either very close to 0 or very close to 1 \n - Predictive Boost: if Predictive Ratio >> Static Ratio, then bit planes within each vector embedding are vertically correlated\n - Similarity Factor: High predictive ratio suggests that the dimensions within a single vector embedding are similar, hence highly compressible")
            elif(AC_MODE == 3):
                print("\nAC COMPRESSIBILITY PER BIT PLANE PER ROW: ")
                AC_Analysis_BitPlane(bitplanes)
                print("\nWhat do the results above mean?:\n - Vector Sparsity: vectors with high ratios across many planes are highly compressible\n - Predictive Indexing: the most compressible vectors could be used to shorten the similarity search through pre-indexing\n - Outlier Detection: anything with a ratio 1.0:1 across all planes are high entropy noise that cannot be compressed effectively")
        
        if(DIRECT_COMPRESSION_COMPARISON == True):
            #original size in storage capacity
            DataBytes = bitplanes.cpu().numpy().tobytes()
            print("\nOriginal Size: " + str(len(DataBytes)) + " Bytes")
            OriginalSize = len(DataBytes)
        
            #we will also perform timing on it
            import time
            print("\n")
        
            #ZSTD Compression
            import zstandard as zstd
            CompressionConfig = zstd.ZstdCompressor(level=9) #9 is high compression level
            TimeStart = time.time()
            compressedZSTD = CompressionConfig.compress(DataBytes) #compresses in ZSTD
            TimeEnd = time.time()
        
            CompressedZSTD_Size = len(compressedZSTD)
            ZSTD_CompressionRatio = OriginalSize / CompressedZSTD_Size
            CompressionPercent = (float)(CompressedZSTD_Size / OriginalSize) * 100
            print("ZSTD Results: ")
            print("Compressed Size: " + str(CompressedZSTD_Size) + " Bytes [" + str(CompressionPercent) + "%]")
            print("Compression Ratio: " + str(ZSTD_CompressionRatio) + " (<1 good, >1 bad)")
            print("Time Elapsed: " + str(TimeEnd - TimeStart) + " seconds")
        
            print("\n")
            #LZ4 Compression
            import lz4.frame
            TimeStart = time.time()
            compressedLZ4 = lz4.frame.compress(DataBytes) #compress in LZ4
            TimeEnd = time.time()
            CompressedLZ4_Size = len(compressedLZ4)
            LZ4_CompressionRatio = OriginalSize / CompressedLZ4_Size
            CompressionPercent2 = (float)(CompressedLZ4_Size / OriginalSize) * 100
            print("LZ4 Results: ")
            print("Compressed Size: " + str(CompressedLZ4_Size) + " Bytes [" + str(CompressionPercent2) + "%]")
            print("Compression Ratio: " + str(LZ4_CompressionRatio) + " (<1 good, >1 bad)")
            print("Time Elapsed: " + str(TimeEnd - TimeStart) + " seconds")
    elif(NUM_ROWS > 1): #this is for vertical bit plane compression of multiple vector embeddings
        #going to try a similar method as before on cpu, but it will be much slower due to multiple rows
        binsplit = Pool().map(binaryvectortolist, binstring)
        #print(binsplit)
    
        #then transpose time
        matrix = numpy.stack(binsplit, axis=-1)
        #print(matrix)
        
        comp = vertical_bitplane_compressibility(matrix, NUM_DIMS)
        print("\nMULTI VECTOR VERTICAL BIT PLANE COMPRESSION RATIOS [Run Length Encoding](-->0 good): ")
        print(comp)
        print("\nexplanation of results above:\n - leftmost column is MSB\n - rightmost column is LSB\n - each row represents 1 of the 768 dimensions in a single vector embedding\n - compression ratio -> 0 means perfect compressibility, >1 means negative compressibility (takes more space to compress it than original)")
        
        print("\nAVERAGE COMPRESSION RATIOS PER BIT PLANE: ")
        avg = torch.mean(comp, dim=0)
        print(avg)
        print("\nexplanation of results above:\n - leftmost column is MSB\n - rightmost column is LSB\n - There is a single row, gets average of all 768 dimensions' bit planes of the # of rows you selected\n - compression ratio -> 0 means perfect compressibility, >1 means negative compressibility (takes more space to compress it than original)")
        
        print("\nAVG COMPRESSION RATIO PER VECTOR EMBEDDING: ")
        avg2 = torch.mean(comp, dim=1)
        print(avg2)
        print("\nAVG COMPRESSION RATIO TOTAL BLOCK: ")
        avg3 = torch.mean(avg, dim=0)
        print(avg3)
        
        if(CALCULATE_ENTROPY == True):
            if(ENTROPY_MODEL == 0):
                print("\nENTROPY RESULTS: ")
                ThreeD_AC_Context(matrix)
                print("\nHow to interpret the results above:\n - P = Density [0 sparse | 0.5 balanced | 1 saturated]\n - H = Theoretical Entropy (bpS)\n --> H = 1.0 bpS : max disorder, random noise, low compressibility (0%, arithmetic coding won't help)\n --> H = 0.8 bpS : modest bias, 1s : 0s is a 70 : 30 split (20% reduction with arithmetic coding)\n --> H = 0.1 bpS : high predictability, plane is nearly all 0s or all 1s (90% reduction with arithmetic coding)\n --> H = 0.0 bpS : static, every single bit in that bit plane is the same (~100% reduction with arithmetic coding)")
        if(AC_MODE != 4):
            if(AC_MODE == 0):
                print("\nTHEORETICAL AC COMPRESSIBILITY: ")
                AC_Analysis(matrix)
                print("\nWhat do the results above mean?:\n - Marginal Ratio (1:Hmarg): This is the compression you get if each plane is encoded individually using a simple arithmetic coder\n --> EX: 10.50:1 means very sparse bit plane, only need ~1/10th the space to store it\n - Contextual Ratio (1:Hcontext): This is the compression you get if your arithmetic coder remembers the bit it just saw in the more significant plane\n --> Gain: If Marginal is 2:1 but Contextual is 4:1, then it means there is a strong correlation between bit planes. Using the bit above as context doubles your storage efficiency\n - MSB (0) vs LSB (31)\n --> your higher msb stuff should have higher ratios than at msb where it drops off towards 1.0:1, which is effectively noise")
                print("\nAC COMPRESSIBILITY PER BIT PLANE FOR ALL ROWS: ")
                AC_Analysis_BitPlaneSlice(matrix)
                print("\nWhat do the results above mean?:\n - Density Bias: we want either very close to 0 or very close to 1 \n - Predictive Boost: if Predictive Ratio >> Static Ratio, then bit planes within each vector embedding are vertically correlated\n - Similarity Factor: High predictive ratio suggests that the dimensions within a single vector embedding are similar, hence highly compressible")
                print("\nAC COMPRESSIBILITY PER BIT PLANE PER ROW: ")
                AC_Analysis_BitPlane(matrix)
                print("\nWhat do the results above mean?:\n - Vector Sparsity: vectors with high ratios across many planes are highly compressible\n - Predictive Indexing: the most compressible vectors could be used to shorten the similarity search through pre-indexing\n - Outlier Detection: anything with a ratio 1.0:1 across all planes are high entropy noise that cannot be compressed effectively")
            elif(AC_MODE == 1):
                print("\nTHEORETICAL AC COMPRESSIBILITY: ")
                AC_Analysis(matrix)
                print("\nWhat do the results above mean?:\n - Marginal Ratio (1:Hmarg): This is the compression you get if each plane is encoded individually using a simple arithmetic coder\n --> EX: 10.50:1 means very sparse bit plane, only need ~1/10th the space to store it\n - Contextual Ratio (1:Hcontext): This is the compression you get if your arithmetic coder remembers the bit it just saw in the more significant plane\n --> Gain: If Marginal is 2:1 but Contextual is 4:1, then it means there is a strong correlation between bit planes. Using the bit above as context doubles your storage efficiency\n - MSB (0) vs LSB (31)\n --> your higher msb stuff should have higher ratios than at msb where it drops off towards 1.0:1, which is effectively noise")
            elif(AC_MODE == 2):
                print("\nAC COMPRESSIBILITY PER BIT PLANE FOR ALL ROWS: ")
                AC_Analysis_BitPlaneSlice(matrix)
                print("\nWhat do the results above mean?:\n - Density Bias: we want either very close to 0 or very close to 1 \n - Predictive Boost: if Predictive Ratio >> Static Ratio, then bit planes within each vector embedding are vertically correlated\n - Similarity Factor: High predictive ratio suggests that the dimensions within a single vector embedding are similar, hence highly compressible")
            elif(AC_MODE == 3):
                print("\nAC COMPRESSIBILITY PER BIT PLANE PER ROW: ")
                AC_Analysis_BitPlane(matrix)
                print("\nWhat do the results above mean?:\n - Vector Sparsity: vectors with high ratios across many planes are highly compressible\n - Predictive Indexing: the most compressible vectors could be used to shorten the similarity search through pre-indexing\n - Outlier Detection: anything with a ratio 1.0:1 across all planes are high entropy noise that cannot be compressed effectively")
        
        if(DIRECT_COMPRESSION_COMPARISON == True):
            #original size in storage capacity
            DataBytes = matrix.tobytes()
            print("\nOriginal Size: " + str(len(DataBytes)) + " Bytes")
            OriginalSize = len(DataBytes)
        
            #we will also perform timing on it
            import time
            print("\n")
        
            #ZSTD Compression
            import zstandard as zstd
            CompressionConfig = zstd.ZstdCompressor(level=9) #9 is high compression level
            TimeStart = time.time()
            compressedZSTD = CompressionConfig.compress(DataBytes) #compresses in ZSTD
            TimeEnd = time.time()
        
            CompressedZSTD_Size = len(compressedZSTD)
            ZSTD_CompressionRatio = OriginalSize / CompressedZSTD_Size
            CompressionPercent = (float)(CompressedZSTD_Size / OriginalSize) * 100
            print("ZSTD Results: ")
            print("Compressed Size: " + str(CompressedZSTD_Size) + " Bytes [" + str(CompressionPercent) + "%]")
            print("Compression Ratio: " + str(ZSTD_CompressionRatio) + " (>1 good, <1 bad)")
            print("Time Elapsed: " + str(TimeEnd - TimeStart) + " seconds")
        
            print("\n")
            #LZ4 Compression
            import lz4.frame
            TimeStart = time.time()
            compressedLZ4 = lz4.frame.compress(DataBytes) #compress in LZ4
            TimeEnd = time.time()
            CompressedLZ4_Size = len(compressedLZ4)
            LZ4_CompressionRatio = OriginalSize / CompressedLZ4_Size
            CompressionPercent2 = (float)(CompressedLZ4_Size / OriginalSize) * 100
            print("LZ4 Results: ")
            print("Compressed Size: " + str(CompressedLZ4_Size) + " Bytes [" + str(CompressionPercent2) + "%]")
            print("Compression Ratio: " + str(LZ4_CompressionRatio) + " (>1 good, <1 bad)")
            print("Time Elapsed: " + str(TimeEnd - TimeStart) + " seconds")
    elif(NUM_ROWS == 0): #this is for direct compression
        #Have to use compression techniques: lz4 or zstd
        #first, get the binary string
        matrix_gpu = torch.from_numpy(datafile_subset['embeddings'][0]).to(ACCELERATION_DEVICE)
        bit_patterns_gpu = matrix_gpu.view(torch.uint32)#put back to int32
        bit_patterns_cpu = bit_patterns_gpu.cpu().numpy()
        binstring = [format(x, '032b') for x in bit_patterns_cpu] #correct
        #print(binstring)
        
        #original size in storage capacity
        DataBytes = datafile_subset['embeddings'][0].tobytes()
        print("Original Size: " + str(len(DataBytes)) + " Bytes")
        OriginalSize = len(DataBytes)
        
        #we will also perform timing on it
        import time
        print("\n")
        
        #ZSTD Compression
        import zstandard as zstd
        CompressionConfig = zstd.ZstdCompressor(level=9) #9 is high compression level
        TimeStart = time.time()
        compressedZSTD = CompressionConfig.compress(DataBytes) #compresses in ZSTD
        TimeEnd = time.time()
        
        CompressedZSTD_Size = len(compressedZSTD)
        ZSTD_CompressionRatio = OriginalSize / CompressedZSTD_Size
        CompressionPercent = (float)(CompressedZSTD_Size / OriginalSize) * 100
        print("ZSTD Results: ")
        print("Compressed Size: " + str(CompressedZSTD_Size) + " Bytes [" + str(CompressionPercent) + "%]")
        print("Compression Ratio: " + str(ZSTD_CompressionRatio) + " (>1 good, <1 bad)")
        print("Time Elapsed: " + str(TimeEnd - TimeStart) + " seconds")
        
        print("\n")
        #LZ4 Compression
        import lz4.frame
        TimeStart = time.time()
        compressedLZ4 = lz4.frame.compress(DataBytes) #compress in LZ4
        TimeEnd = time.time()
        CompressedLZ4_Size = len(compressedLZ4)
        LZ4_CompressionRatio = OriginalSize / CompressedLZ4_Size
        CompressionPercent2 = (float)(CompressedLZ4_Size / OriginalSize) * 100
        print("LZ4 Results: ")
        print("Compressed Size: " + str(CompressedLZ4_Size) + " Bytes [" + str(CompressionPercent2) + "%]")
        print("Compression Ratio: " + str(LZ4_CompressionRatio) + " (>1 good, <1 bad)")
        print("Time Elapsed: " + str(TimeEnd - TimeStart) + " seconds")

if __name__ == '__main__':
    initialization()
    #To do for next revisions monday 2/2/26 and later:
    # - You also need to standardize the ratios a bit so you can compare them more easily
    #You generally need to add your new ease-of-use features from your FineWebEDU code:
    # - Local and online streaming/scanning
    # - Manual and random selection of these files
    # - local file overhaul to work on more machines
    
    
    #STARTING GUIDE:
    # - There are settings and parameters at the top of this code file for you to tweak
    # - [new] I added file directory scanning so it makes it easier to work on other machines
    # - [new] I couldn't add online file streaming due to security on legacy repositories, so you still have to manually download the vector embedding files (.parquet)
    
    #Sorry that the code is so messy, it was worse before, but it could be a lot better
    
    #Some system compatability notes:
    #Since I am dealing with a lot of data at once, I utilized parallel programming on the CPU and GPU to accelerate matrix and transformation workloads.
    #Because of this, I have been developing this simulation code for Intel Arc GPUs with Pytorch 2.9.1+xpu
    #This means that Nvidia CUDA, AMD ROCm, or CPU matrix accleration isn't tested on here, so run it at your own risk
    
    #Some Performance Notes regarding Arithmetic Coding (AC):
    #Arithmetic Coding might not be viable for compression in our use case.
    #While it is excellent for AI embeddings and context awareness,
    #it might be too slow for dedicated hardware,
    #it is very computationally intensive, even with custom hardware
    #it has extremelly high compression ratios
    #it is highly sequential/single threaded, making it hard to acclerate with parallelism


