

FineWebEdu_path = "C:/Users/findi/OneDrive/Desktop/AI Research Datasets/FineWeb-Edu Mini/Data/2025-2026/" #base directory of Wikipedia DPR mini dataset
testfile = FineWebEdu_path + "000_00000.parquet" #current file selection
#current local options:
#000_00000.parquet
#000_00025.parquet
#000_00049.parquet

#Some Global parameters
NUM_ROWS = 0
START_IND = 0

#Some Global options
FILE_STREAMING = False
#This determines the source of your files that the simulation will use
#True - Streams the files from the internet (note that this requires internet)
#False - [Default] Uses local files (make sure your local path is correct, and you have the files)

DATA_SELECTION_METHOD = True
#This determines if you directly choose the source file or if it is randomly chosen
#True - [Default] Manual Selection
#False - Random Selection

ENTROPY = False
#This determines if you want to see the Entropy stats on the bitplanes or not
#True - shows entropy stats
#False - [Default] doesn't show entropy stats

DIRECT_COMPRESSION_COMPARISON = False
#This determines if you also want to compare the direct compression of the bitplanes (ZSTD, LZ4) to your RLE compression of your bitplanes
#True - shows LZ4 and ZSTD compression ratios of the same data
#False - [Default] doesn't show it

AC_MODE = 0
#This determines if you want to see Arithmetic Coding (AC) Compression stats on your current bitplane data
#0 - [None, Default] no AC compression stats
#1 - [Theoretical] will print out all theoretical analytics regarding AC compressibility ratios
#2 - [All Row Bitplane Analysis] will print out the compressibility of all rows into 1 value for each bit plane (768 tall for vector dimension, 32 wide for bit plane bit number)
#3 - [Individual Row Bitplane Analysis] will print out compressibility of each row with 32 compressibility values
#4 - [ALL, default] will print out all analytics regarding AC compressibility

#some performance settings
#NOTE: this is highly experimental, as CUDA and CPU were not tested on this build, only "xpu" which is intel arc acceleration
ACCELERATION_DEVICE = "xpu"
#ACCELERATION_DEVICE has multiple options
#"xpu" - [Default] Intel Arc GPU + XMX acceleration, this is what the simulation was developed on
#"cuda" - [Nvidia GPU + Tensor Core acceleration](Also AMD ROCm compatible) not tested, use at your own risk
#"cpu" - [CPU multicore acceleration] much slower than the other options, use if you don't have a compatible GPU. not tested, use at your own risk

def Simulation():
    #first, we need to take in the file and parse it
    
    #NOTE: you can comment out this initial user input if you want it to be fast between runs, just modify the global variables above
    #actually, I should probably make this simulation more user friendly than the previous one
    #planning for input configuration, these toggle global variables
    #input "Choose Data Source [Online, Local]: "
    global FILE_STREAMING
    fs = input("Choose Data Source [Online, Local]: ")
    if(fs.lower() == "online"):
        FILE_STREAMING = True
    elif(fs.lower() == "local"):
        FILE_STREAMING = False
    else:
        print("Invalid data source, choosing local...")
        FILE_STREAMING = False
    # --> FILE_STREAMING, Online -> True, Local -> False
    #if local or online:
    # --> input "Choose data selection method [Random, Manual]: "
    # ------> if Manual, input "Local .parquet files found, select via number:\n0.)...\nEnter Choice Here: "
    ds = input("Choose data selection method [Random, Manual]: ")
    if(ds.lower() == "manual"):
        global DATA_SELECTION_METHOD
        DATA_SELECTION_METHOD = True
    elif(ds.lower() == "random"):
        #global DATA_SELECTION_METHOD
        DATA_SELECTION_METHOD = False
    else:
        print("Invalid selection, choosing manual...")
        #global DATA_SELECTION_METHOD
        DATA_SELECTION_METHOD = True
    
    if(DATA_SELECTION_METHOD == True):
        if(FILE_STREAMING == True): #online manual selection
            from huggingface_hub import HfFileSystem
            fs = HfFileSystem()
            # List all parquet files within that specific snapshot directory
            # The path structure is: datasets/HuggingFaceFW/fineweb-edu/data/CC-MAIN-2024-10/*.parquet
            snapshot_files = fs.glob(f"datasets/HuggingFaceFW/fineweb-edu/data/CC-MAIN-2025-26/*.parquet")

            #print(f"Found {len(snapshot_files)} files in snapshot {snapshot}.")
            #print(f"Example file: {snapshot_files[0]}")
            print("Found " + str(len(snapshot_files)) + " .parquet files:")
            ind = 0
            while(ind < len(snapshot_files)):
                print(str(ind) + ".) " + str(snapshot_files[ind]))
                ind += 1
            sl = int(input("Enter index here: "))
            selected_path = snapshot_files[sl]
            
            global testfile
            testfile = selected_path.replace(f"datasets/HuggingFaceFW/fineweb-edu/", f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/")
        elif(FILE_STREAMING == False): #local manual selection
            from pathlib import Path
            parquetfiles = list(Path('.').rglob('*.parquet'))
            print("parquet files found, select via number: ")
            ind = 0
            while(ind < len(parquetfiles)):
                print(str(ind) + ".) " + str(parquetfiles[ind]))
                ind += 1
            ms = int(input("Input path number here: "))
            #global testfile
            testfile = parquetfiles[ms]
    elif(DATA_SELECTION_METHOD == False):
        if(FILE_STREAMING == True): #online random selection
            from huggingface_hub import HfFileSystem
            import random
            fs = HfFileSystem()
            # List all parquet files within that specific snapshot directory
            # The path structure is: datasets/HuggingFaceFW/fineweb-edu/data/CC-MAIN-2024-10/*.parquet
            snapshot_files = fs.glob(f"datasets/HuggingFaceFW/fineweb-edu/data/CC-MAIN-2025-26/*.parquet")

            #print(f"Found {len(snapshot_files)} files in snapshot {snapshot}.")
            #print(f"Example file: {snapshot_files[0]}")
            #print("Found " + str(len(snapshot_files)) + " .parquet files:")
            #ind = 0
            #while(ind < len(snapshot_files)):
                #print(str(ind) + ".) " + str(snapshot_files[ind]))
                #ind += 1
            #sl = int(input("Enter index here: "))
            sl = random.randint(0, len(snapshot_files)-1)
            selected_path = snapshot_files[sl]
            
            #global testfile
            testfile = selected_path.replace(f"datasets/HuggingFaceFW/fineweb-edu/", f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/")
        elif(FILE_STREAMING == False):#local random selection
            from pathlib import Path
            import random
            parquetfiles = list(Path('.').rglob('*.parquet'))
            #print("parquet files found, select via number: ")
            #ind = 0
            #while(ind < len(parquetfiles)):
                #print(str(ind) + ".) " + str(parquetfiles[ind]))
                #ind += 1
            #ms = int(input("Input path number here: "))
            ms = random.randint(0, len(parquetfiles)-1)
            #global testfile
            testfile = parquetfiles[ms]
    
    #This part is integral, dont comment it out
    #first, get the file
    import pandas
    #global testfile
    FileData = pandas.read_parquet(testfile, columns=["embeddings"])
    maxblocksize = len(FileData["embeddings"])
    
    
    #Further global parameter input options below
    #input "Choose Block Size in number of rows (0-maxblocksize [must get this before input]): "
    # "note that compression method depends on block size.\n0 - direct compression of whole file\n1 - Horizontal Bit Plane compression of single vector embedding\n>1 - Vertical Bit Plane compression of multiple vector embeddings"
    # --> NUM_ROWS = #
    bs = int(input("Choose Block Size in number of rows [0 --> " + str(maxblocksize-1) + "]: \nNote that compression method depends on block size.\n0 - direct compression on whole file\n1 - Horizontal Bit Plane compression of single vector embedding\n>1 - Vertical Bit Plane compression of multiple vector embeddings\nEnter your choice here: "))
    global NUM_ROWS
    NUM_ROWS = bs
    #input "Choose Block starting index [0-maxblocksize-NUM_ROWS, enter -1 for random]: "
    # --> START_IND = value or random
    si = int(input("Choose Block starting index [0 --> " + str(maxblocksize - 1 - NUM_ROWS) + ", enter -1 for random]: "))
    if(si == -1):
        global START_IND
        import random
        #global NUM_ROWS
        #nonlocal maxblocksize
        rs = random.randint(0, maxblocksize-1-NUM_ROWS)
        START_IND = rs
    elif(si >= 0):
        #global START_IND
        START_IND = si
    else:
        print("invalid input, setting starting index to 0...")
        #global START_IND
        START_IND = 0
    
    #Don't comment out this block of code, it is used to get rows block
    SelectedRows = FileData["embeddings"][0+START_IND:START_IND+NUM_ROWS+1]
    
    # "Now on to extra analytic settings"
    print("Now on to extra analytic settings...")
    # "Entropy: "
    print("Entropy: ")
    #input "Display Entropy analytics? (Yes/No): "
    # --> ENTROPY = True/False
    ea = input("Display Entropy analytics? (Yes/No): ")
    if(ea.lower() == "yes"):
        global ENTROPY
        ENTROPY = True
    elif(ea.lower() == "no"):
        #global ENTROPY
        ENTROPY = False
    else:
        print("invalid setting, entropy is disabled...")
        #global ENTROPY
        ENTROPY = False
    # "Direct Compression Comparison: "
    print("Direct Compression Comparison (LZ4, ZSTD): ")
    #input "Display Direct Compression for Comparison? (Yes/No): "
    dc = input("Display Direct Compression for Comparison? (Yes/No): ")
    if(dc.lower() == "yes"):
        global DIRECT_COMPRESSION_COMPARISON
        DIRECT_COMPRESSION_COMPARISON = True
    elif(dc.lower() == "no"):
        #global DIRECT_COMPRESSION_COMPARISON
        DIRECT_COMPRESSION_COMPARISON = False
    else:
        print("invalid option, defaulting to no direct compression...")
        #global DIRECT_COMPRESSION_COMPARISON
        DIRECT_COMPRESSION_COMPARISON = False
    # --> DIRECT_COMPRESSION_COMPARISON = True/False
    # "Arithmetic Coding Compression: "
    print("Arithmetic Coding Compression (AC): ")
    #input "Want to compare AC bitplane compression to your RLE bitplane compression?:\n0 - No AC\n1 - Theoretical Analysis\n2 - All Row Bitplane Analysis: prints out compressibility of all rows into 1 value for each bit plane\n3 - Individual Row Bitplane Analysis: prints out compressibility of each row with #bits compressibility values\n4 - ALL: prints all of these\nEnter your choice here: "
    ac = int(input("Want to compare AC bitplane compression to your RLE bitplane compression?:\n0 - No AC\n1 - Theoretical Analysis\n2 - All Row Bitplane Analysis: prints out compressibility of all rows into 1 value for each bit plane\n3 - Individual Row Bitplane Analysis: prints out compressibility of each row with #bits compressibility values\n4 - ALL: prints all of these\nEnter your choice here: "))
    # --> AC_MODE = 0-4
    if((ac >= 0) and (ac <= 4)):
        global AC_MODE
        AC_MODE = ac
    else:
        print("Invalid AC mode setting, setting off by default...")
        #global AC_MODE
        AC_MODE = 0
    # "Performance Settings: "
    print("Performance Settings [Hardware Acceleration]: ")
    #input "What device will you use for parallel acceleration?:\nxpu - intel arc gpu\ncuda - nvidia gpu and amd rocm\ncpu - slowest option\nNote that only xpu was tested, so use the others at your own risk\nEnter your choice here: "
    ps = input("What device will you use for parallel acceleration?:\nxpu - intel arc gpu\ncuda - nvidia gpu and amd rocm\ncpu - slowest option\nNote that only xpu was tested, so use the others at your own risk\nEnter your choice here: ")
    if((ps == "xpu") or (ps == "cuda") or (ps == "cpu")):
        global ACCELERATION_DEVICE
        ACCELERATION_DEVICE = ps
    else:
        print("Invalid hardware acceleration setting, will default to CPU for compatibility...")
        #global ACCELERATION_DEVICE
        ACCELERATION_DEVICE = "cpu"
    # --> ACCELERATION_DEVICE = "xpu"/"cuda"/"cpu"
    
    #now we do more actual script parts
    
    #To do monday 2/2/26 and later:
    # - Found out that you are using the wrong files, the .parquet files aren't vector embeddings.
    # - You want the .npy files (these are precomputed vector embeddings)
    # - You will have to rewrite and modify parts of your code to now handle .npy files
    # --> This includes:
    # ---> panda.read_npy or something like that
    # ---> Your fineweb-edu library stuff for streaming
    # ---> Your local file selection
    # ---> The random selection stuff
    # --> New parsing
    #Here is the link to the website with vector embeddings: https://huggingface.co/datasets/Cohere/fineweb-edu-emb/tree/main/emb
    
    pass

if __name__ == '__main__':
    Simulation() #starts our simulation
    #this is necessary due to the quirks of parallel programming on cpu for kernel programming