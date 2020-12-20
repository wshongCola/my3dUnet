import os
import scipy.io as scio
import numpy as np

datadir = "/home/wshong/Documents/MATLAB/compress_sensing/code/cs_CT_Lp_recon/results/GOTCHA/"
carname = "ToyotaCamry"
# carname = "ChevyMalibu"
pol = "HH"
prefix = "Normal"
config_dir = prefix + "/Lambda10.0000_Pol" + pol + "_Wx6.0_Wy8.0_Wz4.0_Wins6.0_res0.1"
datadir = datadir + carname + os.sep + pol + os.sep + config_dir + os.sep

# Load data dirs
BPLoadDir = "CS-INP/"
CSLoadDir = "CS-GT/"
BPLoadDir = datadir + BPLoadDir
CSLoadDir = datadir + CSLoadDir

# Save data dirs.
SaveDir = "/home/wshong/Documents/data/unet3d_car/narrow_elev/GOTCHA/train/"
SaveGTDir = SaveDir + "GT/"
SaveINPDir = SaveDir + "INP/"

if not os.path.exists(SaveGTDir):
    os.makedirs(SaveGTDir)
if not os.path.exists(SaveINPDir):
    os.makedirs((SaveINPDir))

# Load GT data. Uniform it and save as npy format.
LoadCSFileName = carname + ".mat"

LoadCSFilePath = CSLoadDir + LoadCSFileName

CSData = scio.loadmat(LoadCSFilePath)
CSImg = np.abs(CSData['img']) / 8  # Abs and Uniform Operation.
image_x, image_y, image_z = np.shape(CSImg)
output_size = [max(image_y, image_x), max(image_y, image_x), image_z]

for x_offset in np.arange(-20, 20, 5):
    for y_offset in np.arange(-10, 10, 5):
        CSImgAug = np.zeros(output_size)
        CSImgAug[
            max(x_offset, 0): min(x_offset + image_x, output_size[0]),
            max(y_offset, 0): min(y_offset + image_y, output_size[1]), :
        ] = CSImg[
            max(-x_offset, 0): image_x + min(output_size[0] - image_x - x_offset, 0),
            max(-y_offset, 0): image_y + min(output_size[1] - image_y - y_offset, 0),:
            ]
        for r in np.arange(1, 4):
            CSImgAug = np.rot90(CSImgAug)
            aug_str = "+r" + str(r) + "ox" + str(x_offset) + "oy" + str(y_offset)
            SaveGTFileName = pol + carname + aug_str + "_gt.npy"
            SaveGTFilePath = SaveGTDir + SaveGTFileName
            np.save(SaveGTFilePath, CSImgAug)

for x_offset in np.arange(-20, 20, 10):
    for y_offset in np.arange(-10, 10, 5):
        DesiredOrbs = ["148", "158", "168", "137", "147", "157"]
        for Orb in DesiredOrbs:
            LoadBPFileName = carname + "_csinp_" + Orb + ".mat"
            LoadBPFilePath = BPLoadDir + LoadBPFileName
            BPData = scio.loadmat(LoadBPFilePath)
            BPImg = np.abs(BPData['bp_slc_img']) / 3

            INPImgAug = np.zeros(output_size)
            INPImgAug[
                max(x_offset, 0): min(x_offset + image_x, output_size[0]),
                max(y_offset, 0): min(y_offset + image_y, output_size[1]), :
            ] = BPImg[
                max(-x_offset, 0): image_x + min(output_size[0] - image_x - x_offset, 0),
                max(-y_offset, 0): image_y + min(output_size[1] - image_y - y_offset, 0), :
                ]
            for r in np.arange(1, 4):
                # Load Desired INP data. Uniformed it and save as npy format.
                aug_str = "+r" + str(r) + "ox" + str(x_offset) + "oy" + str(y_offset)
                SaveINPFilename = pol + carname + aug_str + "_inp" + Orb + ".npy"
                SaveINPFilePath = SaveINPDir + SaveINPFilename
                INPImgAug = np.rot90(INPImgAug)
                np.save(SaveINPFilePath, INPImgAug)





