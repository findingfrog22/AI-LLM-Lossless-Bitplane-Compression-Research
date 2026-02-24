import torch
#some settings and universal variables
BASE_TYPE = torch.float32
QUANTIZATION_TYPE = torch.int8
SCALE_TYPE = torch.float32

ACCELERATION_DEVICE = "xpu"

PRINT_DEBUG = True
print_stage0 = True #prints the original tensor
print_stage1 = True #prints symmetric scalar quantization details
print_stage2 = True #prints binary string conversion (optional)
print_stage3 = True #prints bitplane disaggregation
print_stage4 = True #prints RLE analysis
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

def rowbinary_quant(row, qmat):
    return [format(x, f'0{qmat}b') for x in row] #only works in python 3.6 and newer

def rowbinary_scale(row, smat):
    return [format(x, f'0{smat}b') for x in row] #only works in python 3.6 and newer

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
    print("\nDEBUG: binstring: ")
    print(binstring)
    #then format them into lists then tensors to send to device
    bit_split = Pool().map(binaryvectortolist, binstring)
    #print("\nDEBUG: bit_split: ")
    #print(bit_split)
    bitplane_horizontal = torch.tensor(bit_split).to(ACCELERATION_DEVICE)
    #perform bitplane rearrangement
    bitplane_h = bitplane_horizontal.permute(2,0,1).contiguous()
    #print("\nDEBUG: bitplane_h direct: ")
    #print(bitplane_h)
    bitplane_v = numpy.stack(bit_split, axis=-1)
    #print("\nDEBUG: bitplane_v direct: ")
    #print(bitplane_v)
    
    if(print_stage3 == True):
        print("\nSTAGE 2+3 RAW: ")
        print("\nBitplane RAW Horizontal: " + str(bitplane_h.shape))
        print(bitplane_h)
        print("\nBitplane RAW Vertical: " + str(bitplane_v.shape))
        print(bitplane_v)
    #now return the values
    return bitplane_h, bitplane_v

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
    
    #Step 5.) Test Bitpacking (n_packed, w_packed, b_packed)
    
    #Step 6.) Test LZ4+ZSTD Bitplane Compression Analysis (Global, Temporal, Spatial)
    pass

if  __name__ == "__main__":
    import numpy
    import torch
    parquet_data = [[0.5,0.33,0.125],[0.75,0.25,0.375]] #tensor is (N,W), (in this case, (2,3))
    Tensor = torch.tensor(parquet_data, dtype=torch.float32)
    #print(format("032b", Tensor.view(torch.uint32)))
    if(print_stage0 == True):
        print("\nInitial Tensor Data: " + str(Tensor.shape))
        print(Tensor)
    initialization(Tensor)