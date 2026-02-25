import torch
#some settings and universal variables
BASE_TYPE = torch.float32
QUANTIZATION_TYPE = torch.int8
SCALE_TYPE = torch.float32

ACCELERATION_DEVICE = "xpu"

PRINT_DEBUG = True
print_stage0 = False #prints the original tensor
print_stage1 = False #prints symmetric scalar quantization details
print_stage2 = False #prints binary string conversion (optional)
print_stage3 = False #prints bitplane disaggregation
print_stage4 = False #prints RLE analysis
print_stage5 = True #prints bitpacking
print_stage6 = True #prints LZ4+ZSTD Compression analysis

#Quantization Functions (Stage 1)
def symmetric_scalar_quantization(tens_emb, direc):
    import torch
    import numpy
    #per vector scale first
    dimen = -1
    if(direc == "row"):
        dimen = 1
        if(print_stage1 == True):
            print("\nROW: ")
    elif(direc == "col"):
        dimen = 0
        if(print_stage1 == True):
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
    if(print_stage1 == True):
        print("Scales dtype: " + str(scales.dtype))
    scales = torch.clamp(scales, min=1e-9) #for now, and these are the scales per vector embedding row
    #third, element-wise quantization and rounding (uses XPU backend to minimize memory trips)
    quantized = torch.round(tens_emb / scales).to(QUANTIZATION_TYPE) #for now, quantized to int8
    if((PRINT_DEBUG == True)and(print_stage1 == True)):
        print(direc)
        print("\nSymmetric Scalar Quantization (Quantized Values): ")
        print(quantized)
        print("\nSymmetric Scalar Quantization (Scalar Values): ")
        print(scales)
    return quantized, scales

#Bitplane Disaggregation Functions (stage 2+3)
def raw_bitplane(tens_emb):
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
    if(print_stage3 == True):
        print("\nSTAGE 3: Bitplane Disaggregation (raw/direct): H: " + str(planes_horizontal.shape) + "; V: " + str(planes_vertical.shape) + ": ")
        print("Horizontal Raw: ")
        print(planes_horizontal)
        print("Vertical Raw: ")
        print(planes_vertical)
    return planes_horizontal, planes_vertical

#vibe coded, modify and make sure it is robust
def bitplane(quant, scale, direc):
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
            return planes.permute(2,0,1).contiguous() #now it seems to work correctly (results in (W,B,N))
        
        # 6. Convert to uint8 (0 or 1) for your RLE and packing analysis
        return planes.to(torch.uint8).contiguous()

    # Determine bit depth
    q_bits = quant.element_size() * 8
    s_bits = scale.element_size() * 8
    
    q_planes = get_planes(quant, q_bits, direc)
    s_planes = get_planes(scale, s_bits, direc)
    
    if(print_stage3 == True):
        print("\nSTAGE 3: Bitplane Disaggregation (Quant+Scalar): Q: " + str(q_planes.shape) + ", S: " + str(s_planes.shape) + " : ")
        print("Quant: ")
        print(q_planes)
        print("Scalars: ")
        print(s_planes)

    return q_planes, s_planes

#RLE Compression Analysis (Stage 4)
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
    if(print_stage4 == True):
        print("\nDEBUG: PLANELENGTH: " + str(PlaneLength))
    ratio = (RunsPerBitplane * 2) / PlaneLength #this is the overall results
    if(print_stage4 == True):
        print("\nDEBUG: " + str(title) + " Overall results (RLE): " + str(ratio.shape))
        print(ratio)
    #return ratio?
    if(form == "horizontal"):
        bp_avg = torch.mean(ratio, dim=0) #per bitplane, originally dim=1
        if(print_stage4 == True):
            print("\nHorizontal Per Bitplane: ")
            print(bp_avg)
        row_avg = torch.mean(ratio, dim=1) #per vector embedding, originally dim=0
        if(print_stage4 == True):
            print("\nHorizontal Per Vector Embedding: ")
            print(row_avg)
    elif(form == "vertical"):
        bp_avg = torch.mean(ratio, dim=0) #per bitplane
        if(print_stage4 == True):
            print("\nVertical Per Bitplane: ")
            print(bp_avg)
        dim_avg = torch.mean(ratio, dim=1) #per vector embedding dim
        if(print_stage4 == True):
            print("\nVertical Per Dimension: ")
            print(dim_avg)
    if(print_stage4 == True):
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

#Bit packing (Stage 5)
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

def initialization(tens_emb):
    #Step 1.) Test Symmetric Scalar Quantization (row-wise, col-wise, raw)
    row_quant, row_scale = symmetric_scalar_quantization(tens_emb, "row")
    col_quant, col_scale = symmetric_scalar_quantization(tens_emb, "col")
    #Step 2.) (optional?) Test Binary String Conversion
    
    #Step 3.) Test Bitplane Disaggregation (horizontal, vertical)
    
    #first, raw conversions
    bitplane_horizontal, bitplane_vertical = raw_bitplane(tens_emb)
    
    #now try horizontal
    bitplane_quant_row_horizontal, bitplane_scale_row_horizontal = bitplane(row_quant, row_scale, "horizontal")
    bitplane_quant_col_horizontal, bitplane_scale_col_horizontal = bitplane(col_quant, col_scale, "horizontal")
    #now try vertical
    bitplane_quant_row_vertical, bitplane_scale_row_vertical = bitplane(row_quant, row_scale, "vertical")
    bitplane_quant_col_vertical, bitplane_scale_col_vertical = bitplane(col_quant, col_scale, "vertical")
    
    #Step 4.) Test RLE Analysis (across bitplanes)
    RLE_bitplane_compressibility(bitplane_horizontal, "horizontal", "raw_horizontal")
    RLE_bitplane_compressibility(bitplane_vertical, "vertical", "raw_vertical")
    RLE_bitplane_compressibility(bitplane_quant_row_horizontal, "horizontal", "quant_row_horizontal")
    RLE_bitplane_compressibility(bitplane_scale_row_horizontal, "horizontal", "scale_row_horizontal")
    RLE_bitplane_compressibility(bitplane_quant_col_horizontal, "horizontal", "quant_col_horizontal")
    RLE_bitplane_compressibility(bitplane_scale_col_horizontal, "horizontal", "scale_col_horizontal")
    RLE_bitplane_compressibility(bitplane_quant_row_vertical, "vertical", "quant_row_vertical")
    RLE_bitplane_compressibility(bitplane_scale_row_vertical, "vertical", "scale_row_vertical")
    RLE_bitplane_compressibility(bitplane_quant_col_vertical, "vertical", "quant_col_vertical")
    RLE_bitplane_compressibility(bitplane_scale_col_vertical, "vertical", "scale_col_vertical")
    
    #Step 5.) Test Bitpacking (n_packed, w_packed, b_packed)
    import numpy
    import torch
    import sys
    #bit_data_test = [[[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]] #tensor in the form (N,W,B), which is (8,8,8) in this case
    bit_data_test = torch.randint(0, 2, (8,8,8), dtype=torch.uint8)
    #we need to set the print options for both so they don't truncate the results, good for manual testing and comparison
    numpy.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(profile="full")
    #print the original data
    if(print_stage5 == True):
        print("\nStage 5: Bitpacking: Original Tensor: " + str(bit_data_test.shape))
        print(bit_data_test)
        packed_n = pack_bits(bit_data_test,0)
        print("\npack_n: " + str(packed_n.shape))
        print(packed_n)
        packed_w = pack_bits(bit_data_test,1)
        print("\npack_w: " + str(packed_w.shape))
        print(packed_w)
        packed_b = pack_bits(bit_data_test,2)
        print("\npack_b: " + str(packed_b.shape))
        print(packed_b)
    
    #Step 6.) Test LZ4+ZSTD Bitplane Compression Analysis (Global, Temporal, Spatial)
    pass

if  __name__ == "__main__":
    import numpy
    import torch
    parquet_data = [[0.5,0.33,0.125],[0.75,0.25,0.375]] #tensor is (N,W), (in this case, (2,3))
    #parquet_data = [[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]] #tensor is (N,W), (in this case, (8,8))
    Tensor = torch.tensor(parquet_data, dtype=torch.float32)
    #print(format("032b", Tensor.view(torch.uint32)))
    if(print_stage0 == True):
        print("\nInitial Tensor Data: " + str(Tensor.shape))
        print(Tensor)
    initialization(Tensor)