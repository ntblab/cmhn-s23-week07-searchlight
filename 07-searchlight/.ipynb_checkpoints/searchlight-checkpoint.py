#!/gpfs/gibbs/project/cmhn/share/conda_envs/mybrainiak/bin/python

# Run a whole brain searchlight

# Import libraries
import nibabel as nib
import numpy as np
from mpi4py import MPI
from brainiak.searchlight.searchlight import Searchlight
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import os

# Define some constant variables for running and saving data.
scratch_directory =  os.path.expanduser("~/palmer_scratch/")
# This is where the preprocessed data created from the 07-searchlight notebook will go:
fs_data_dir = os.path.join(scratch_directory, 'brainiak_results/searchlight_data/')
results_path = os.path.join(scratch_directory, 'brainiak_results/searchlight_results/') # change these paths only if you want them to go elsewhere!
    
# Specify the number of subjects to run    
num_subj = 3

############ Define functions ###################

# Load and prepare data for one subject
def load_fs_data(sub_id, mask=None):
    # Find the file path
    sub = f'sub-{sub_id:02d}'
    data_dir = f'{fs_data_dir}/{sub}'
    data_file = f'{data_dir}/data.nii.gz'
    label_file =  f'{data_dir}/label.npy'
    
    if not mask:
        mask = 'wb'
    mask_file = f'{fs_data_dir}/{mask}_mask.nii.gz'
    
    # Load bold data and some header information so that we can save searchlight results as nifti later.
    nii = nib.load(data_file)
    bold_data = nii.get_fdata()
    affine_mat = nii.affine
    dimensions = nii.header.get_zooms() 
    
    # Load mask.
    brain_mask = nib.load(mask_file).get_fdata()

    return bold_data, brain_mask, affine_mat, dimensions

# Load the face/scene labels for this subject
def load_fs_label(sub_id):
    sub = f'sub-{sub_id:02d}'
    input_dir = os.path.join(fs_data_dir, sub)
    label = np.load(os.path.join(input_dir, 'label.npy'))
    return label
    
# Set up the kernel function, in this case an SVM
def calc_svm(data, sl_mask, myrad, bcvar):
    
    num_voxels_in_sl = sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2]
    num_epoch = data[0].shape[3]
    
    accuracy = []
    # Loop over subjects to leave each subject out once: 
    for idx in range(len(data)):
        # Pull out the data
        # Testing data
        data4D_test = data[idx]
        labels_test = bcvar[idx]
        bolddata_sl_test = data4D_test.reshape(num_voxels_in_sl, num_epoch).T  
        
        # Training data
        labels_train = []
        bolddata_sl_train = np.empty((0, num_voxels_in_sl))
        for train_id in range(len(data)):
            if train_id != idx:
                labels_train.extend(list(bcvar[train_id]))
                bolddata_sl_train = np.concatenate((bolddata_sl_train, data[train_id].reshape(num_voxels_in_sl, num_epoch).T))
        labels_train = np.array(labels_train)
        
        # Train classifier
        clf = SVC(kernel='linear', C=1)
        clf.fit(bolddata_sl_train, labels_train)
        
        # Test classifier
        score = clf.score(bolddata_sl_test, labels_test)
        accuracy.append(score) 
        
    return accuracy

########### This is the code that actually runs the analysis, using the functions defined above. 

# Pull out the MPI information
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Load mask as a binary numpy array.
mask_file = os.path.join(fs_data_dir, 'wb_mask.nii.gz')
mask = nib.load(mask_file).get_fdata()

data = []
bcvar = []
masks = []
affines = []
dimsizes = []
# Loop over subjects.

for sub_id in range(1,num_subj+1):
    # Only load the data on the first core.
    if rank == 0:
        data_i, mask, affine_mat, dimsize = load_fs_data(sub_id)
        data.append(data_i)
        masks.append(mask)
        affines.append(affine_mat)
        dimsizes.append(dimsize)
    else:
        data.append(None)
    bcvar_i = load_fs_label(sub_id)
    bcvar.append(bcvar_i)

sl_rad = 1
max_blk_edge = 5
pool_size = 1

coords = np.where(mask)


# Create the searchlight object
sl = Searchlight(sl_rad=sl_rad,max_blk_edge=max_blk_edge)

# Distribute the information to the searchlights (preparing it to run)
sl.distribute(data, mask)

# Broadcast variables
sl.broadcast(bcvar)


# Run the searchlight analysis
print(f"Begin Searchlight in rank {rank}")
all_sl_result = sl.run_searchlight(calc_svm, pool_size=pool_size)
print(f"End Searchlight in rank {rank}")

# Only save the data if this is the first core.
if rank == 0: 
    all_sl_result = all_sl_result[mask==1]
    all_sl_result = [num_subj*[0] if not n else n for n in all_sl_result] # replace all None
    # The average result
    avg_vol = np.zeros(mask.shape)  
    
    # Loop over subjects
    for sub_id in range(1,num_subj+1):
        sl_result = [r[sub_id-1] for r in all_sl_result]
        # reshape
        result_vol = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]))  
        result_vol[coords] = sl_result   
        # Convert the output into what can be used
        result_vol = result_vol.astype('double')
        # Turn any missing values to 0
        result_vol = np.nan_to_num(result_vol) 
        # Add the processed result_vol into avg_vol
        avg_vol += result_vol
        # Save the volume
        output_name = os.path.join(results_path, f'sub-{sub_id:02d}_whole_brain_SL.nii.gz')
        sl_nii = nib.Nifti1Image(result_vol, affine_mat)
        sl_nii.header.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
        nib.save(sl_nii, output_name)  # Save
    
    # Save the average result
    output_name = os.path.join(results_path, f'avg-{num_subj:02d}_whole_brain_SL.nii.gz')
    sl_nii = nib.Nifti1Image(avg_vol/num_subj, affine_mat)
    sl_nii.header.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
    nib.save(sl_nii, output_name)  # Save    
    
    print('Finished searchlight')