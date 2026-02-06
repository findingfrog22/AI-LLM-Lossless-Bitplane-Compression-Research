
WikipediaDPR_path = "C:/Users/findi/OneDrive/Desktop/AI Research Datasets/Wikipedia DPR Mini/" #base directory of Wikipedia DPR mini dataset
testfile = WikipediaDPR_path + "Multiset/train-00000-of-00157.parquet" #current file selection

NUM_ROWS = 2048 #this is how many rows of vector embeddings will be processed
NUM_DIMS = 768 #this is the number of weights per vector embedding

#some more global settings
ACCELERATION_DEVICE = "xpu"
#symmetric scalar quantization settings
MULTI_SCALAR = True
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
    print(quantized)
    print(scales)
    return quantized, scales

def Symmetric_Scalar_Dequantization_ROW(data, scalars):
    import torch
    import numpy
    #first, cast INT8 data to target float type
    fl = data.to(torch.float32)
    #second, do the multiplication across iGPU
    dequantized = fl * scalars
    print(dequantized)
    return dequantized

def Symmetric_Scalar_Error_ROW(orig, dequant):
    import torch
    import torch.nn.functional as f
    import numpy
    #overall loss
    overall_loss = f.mse_loss(orig, dequant)
    print(overall_loss)
    #row-wise
    #elementwise
    element_loss = f.mse_loss(orig, dequant, reduction='none')
    print(element_loss)
    #row-wise avgs
    vector_loss = element_loss.mean(dim=1)
    print(vector_loss)
    #finding worst vector
    worst_vector = torch.argmax(vector_loss)
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
    print(quantized)
    print(scales)
    return quantized, scales

def Symmetric_Scalar_Dequantization_COL(data, scalars):
    import torch
    import numpy
    #first, cast INT8 data to target float type
    fl = data.to(torch.float32)
    #second, do the multiplication across iGPU
    dequantized = fl * scalars
    print(dequantized)
    return dequantized

def Symmetric_Scalar_Error_COL(orig, dequant):
    import torch
    import torch.nn.functional as f
    import numpy
    #overall loss
    overall_loss = f.mse_loss(orig, dequant)
    print(overall_loss)
    #row-wise
    #elementwise
    element_loss = f.mse_loss(orig, dequant, reduction='none')
    print(element_loss)
    #col-wise avgs
    vector_loss = element_loss.mean(dim=0)
    print(vector_loss)
    #finding worst vector
    worst_vector = torch.argmax(vector_loss)
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
    print(binstring_qr) #works
    #scalar bitplane
    bpg_sr = scales.view(torch.uint32)
    bpc_sr = bpg_sr.cpu()
    #converts it to binary
    binstring_sr = Pool().map(rowbinary_scale, bpc_sr)
    print(binstring_sr) #works
    
    #now we need to do this for the vertical symmetric scalar quantization too, (only if applicable)
    if((NUM_ROWS > 1) and (MULTI_SCALAR == True)):
        #first, we need bitplanes of quantized data
        bpg_qc = quantized_C.view(torch.uint8)
        bpc_qc = bpg_qc.cpu()
        #convert it to binary
        binstring_qc = Pool().map(rowbinary_quant, bpc_qc)
        print(binstring_qc) #works
        #then, we want bitplanes of the scalar data
        bpg_sc = scales_C.view(torch.uint32)
        bpc_sc = bpg_sc.cpu()
        #convert it to binary
        binstring_sc = Pool().map(rowbinary_scale, bpc_sc)
        print(binstring_sc) #basically works (so close, but lsbs are a little off), probably a rounding thing
    
    #now we need to handle both horizontal and vertical bitplaning
    if(NUM_ROWS == 1):
        #first, use cpu parallel acceleration to split each element into its own list
        #(we need to repeat this process for both quantized data and the scales)
        #binstring_sr
        #binstring_qr
        #first, the quantized data
        vector_embedding_qr = binstring_qr[0]
        binsplit_qr = Pool().map(binarytolist, vector_embedding_qr)
        #print(binsplit)#works
        
        #now we need to transpose the matrix
        Matrix_qr = torch.tensor(binsplit_qr)
        matrixXPU_qr = Matrix_qr.to(ACCELERATION_DEVICE)
        bitplanes_qr = matrixXPU_qr.T
        #print(bitplanes.to("cpu"))#works
        
        #now analyze the compressibility of the bitplanes
        comp_qr = horizontal_bitplane_compressibility(bitplanes_qr)
        
        #now, the scale data
        vector_embedding_sr = binstring_sr
        binsplit_sr = Pool().map(binarytolist, vector_embedding_sr)
        #print(binsplit)#works
        
        #now we need to transpose the matrix
        Matrix_sr = torch.tensor(binsplit_sr)
        matrixXPU_sr = Matrix_sr.to(ACCELERATION_DEVICE)
        bitplanes_sr = matrixXPU_sr.T
        #print(bitplanes.to("cpu"))#works
        
        #now analyze the compressibility of the bitplanes
        comp_sr = horizontal_bitplane_compressibility(bitplanes_sr)
        pass
    elif(NUM_ROWS > 1):
        #going to try a similar method as before on cpu, but it will be much slower due to multiple rows
        #(we need to repeat this process for both quantized data and the scales)
        #binstring_sc
        #binstring_qc
        #first, try it with quantized data (row-wise)
        binsplit_qr = Pool().map(binaryvectortolist, binstring_qr)
        print("\nDEBUG: binsplit_qr: ")
        print(binsplit_qr)
    
        #then transpose time
        matrix_qr = numpy.stack(binsplit_qr, axis=-1)
        #print(matrix)
        
        comp_qr = vertical_bitplane_compressibility(matrix_qr, NUM_DIMS)
        print("\n[horizontal quant, quant] MULTI VECTOR VERTICAL BIT PLANE COMPRESSION RATIOS [Run Length Encoding](-->0 good): ")
        print(comp_qr)
        
        #then try it with scales (row-wise)
        binsplit_sr = Pool().map(binaryvectortolist, binstring_sr)
        print("\nDEBUG: binsplit_sr: ")
        print(binsplit_sr)
    
        #then transpose time
        matrix_sr = numpy.stack(binsplit_sr, axis=-1)
        #print(matrix)
        
        comp_sr = vertical_bitplane_compressibility(matrix_sr, NUM_DIMS)
        print("\n[horizontal quant, scale] MULTI VECTOR VERTICAL BIT PLANE COMPRESSION RATIOS [Run Length Encoding](-->0 good): ")
        print(comp_sr)
        
        #now we also need to account for vertical symmetric quantization too
        if(MULTI_SCALAR == True):
            #first, try it with quantized data (col-wise)
            binsplit_qc = Pool().map(binaryvectortolist, binstring_qc)
            print("\nDEBUG: binsplit_qc: ")
            print(binsplit_qc)
    
            #then transpose time
            matrix_qc = numpy.stack(binsplit_qc, axis=-1)
            #print(matrix)
        
            comp_qc = vertical_bitplane_compressibility(matrix_qc, NUM_DIMS)
            print("\n[vertical quant, quant]MULTI VECTOR VERTICAL BIT PLANE COMPRESSION RATIOS [Run Length Encoding](-->0 good): ")
            print(comp_qc)
        
            #then try it with scales (row-wise)
            binsplit_sc = Pool().map(binaryvectortolist, binstring_sc)
            print("\nDEBUG: binsplit_sc: ")
            print(binsplit_sc)
    
            #then transpose time
            matrix_sc = numpy.stack(binsplit_sc, axis=-1)
            #print(matrix)
        
            comp_sc = vertical_bitplane_compressibility(matrix_sc, NUM_DIMS)
            print("\n[vertical quant, scale] MULTI VECTOR VERTICAL BIT PLANE COMPRESSION RATIOS [Run Length Encoding](-->0 good): ")
            print(comp_sc)
            pass
        pass
    pass

#for thread safety
if __name__ == '__main__':
    initialization()
    #TO DO LIST FOR 2/7/26+:
    # - KNOWN ISSUE: the binary plane splits used right before the compressibility have formatted it wrong, look at binsplit_qc vs the array under tensor308
    # - once that is ironed out, we make versions of the compressibility analytics that use LZ4 and ZSTD per row,column, etc (with block sizes being customizable)
    # - work on developing the other steps
    
    #once the main stuff is down: we can add the nice stuff:
    # - localized file choosing
    # - file and block randomization
    # - all the compression ratio metrics should be standard (final size / original size out of 1.0)
    pass