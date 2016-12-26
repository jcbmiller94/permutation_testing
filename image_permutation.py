"""
Code for permutation testing for group-level fMRI images (here, beta maps)

"""
#: standard imports
import numpy as np
import numpy.linalg as npl
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import colors
np.set_printoptions(precision=4)  # print arrays to 4 DP
import nibabel as nib
import nilearn
from nilearn import plotting, masking
import glob
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
from apply_glm import glm_4d, t_test_3d

"""
Code to look through folder of (group-level) image files and determine voxel- and cluster- wise
 thresholds for statistical significance employing a permutation method.

Each permutation, the n images will have (1) or (-1) randomly assigned. Under the null,
 such labeling still shoudl yield no relevant t-values caluclated at the group level.

In this specific case, the option for continuous variables of interest and non-interest will
 be included as well.


Flow:

- establish design matrix for series of images with continuous variables of interest and non-interest
    Y = X.dot(B) + e     (where B are beta maps, X is design matrix, e is residuals)

- sample from binomial distribution of (-1, 1) to assign random values to each of 73 images

- calculate t-map for group level analysis with beta weight signs randomly permuted

- determine max t-value (voxel-wise) or cluster size (cluster-wise, use z = 3 or p=0.0001 for cluster-
   level trheshold) for each permutation and add to distribution

- sample top 2.5pct. from end of each side of distribution and use as threshold of interest

Y: array size (I, J, K, T)
    - beta maps from contrast of interest, each image is from an individual subject
X_g: aray size (scans x p)
    - group-level design matrix
B_g: array size (p x voxels)
    - group-level beta map for each parameter p of the design matrix X_g
c: contrast of interest (p,)
    - weight for each of the parameters p, must sum to 0 or 1

"""

# load files

def load_group_data(dirpath, filt = '*', suffix = '.nii'):
    """ Descends through a directory to load given files and concatenate into 4D image file

    Parameters
    ----------
    dirpath: string
        asbolute or relative filepath of directory containing group level images

    filter: string (default = *)
        used to select files of interest
    suffix: string (default = '.nii')

    Returns
    ----------
    group_img: img size ((3D image size) x scans)

    """
    #img_list = []
    img_dict = {}
    imgpaths = dirpath + filt + suffix
    img_names = []
    i = 0
    for filename in glob.glob(imgpaths):
        img = nib.load(filename, mmap = False)
        #
        img_dict['img{0}'.format(i)] = img
        img_names.append(filename)
        i += 1
    #print(img_dict)
    img_list = list(img_dict.values())
    group_img = nib.concat_images(img_list)
    # load the data from the grup image
    group_data = group_img.get_data()

    return group_data, img_names

def design_matrix(filepath, labels = [], img_names = [], plot = True):
    """ Constructs a group-level design matrix from the given variables

    Parameters
    ----------
    filepath: string
        asbolute or relative filepath to .txt file containing regressors
    labels: sequence
        list of strings to label each regressor of design matrix
    img_names: sequence
        list of img filenames to label design matrix
    plot: boolean (default = True)
        if True, plot the design matrix

    Returns
    ----------
    X: array size (scans, p)
        design matrix

    """
    var_arr = np.loadtxt(filepath)

    # check argument lengths are maching label lengths
    if len(img_names) > 0 and (var_arr.shape[0] != len(img_names)):
        raise ValueError('Number of scans in regressors file and imagelist must be equal')
    if len(labels) > 0 and (var_arr.shape[1] != len(labels)):
        raise ValueError('Number of regressors and labels must be equal')

    # add columns of ones to end of design matrix to model the mean
    X = np.column_stack((var_arr, np.ones(var_arr.shape[0])))
    #for i in range(X.shape[0]):

    # plot the design matrix unless specified otherwise
    if plot == True:
        fig, ax = plt.subplots(1)
        ax.set_yticks(np.arange(X.shape[0]))
        # set y labels to the list of image names provided from loading the data
        if len(img_names) > 0:
            y_labels = list(img_names)
            ax.set_yticklabels(y_labels)
        # set the x labels to the labels passed for the regressors
        if len(labels) > 0:
            x_labels = list(labels)
            x_labels.insert(0, ' ')
            ax.set_xticklabels(x_labels, rotation='vertical')
        plt.title('Group-level design matrix', fontsize = 16)
        # Pad margins so that markers don't get clipped by the axes
        plt.margins(0.2)
        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.15, left = 0.2)
        ax.imshow(X, aspect=0.4,vmin=X.min(), vmax=X.max())
        plt.show()
    return X

def shuffle_signs(input_arr):
    """ Assigns a positive (1) or negaive (-1) weight across last dimension of array

        Parameters
        ----------
        input_arr: array size (I, J, K, scans)

        Returns
        ----------
        signed_arr: array size (I,J,K, scans)

    """
    signed_arr = np.empty((input_arr.shape))
    last_dim = input_arr.shape[-1]
    # random distribution of 0s and 1s
    rand_ints = np.random.randint(2, size=last_dim)
    # change 0s to -1s
    rand_ints[rand_ints == 0] = -1
    for i in range(last_dim):
        signed_arr[...,i] = input_arr[...,i]*rand_ints[i]

    return signed_arr

def group_glm(group_data, X):
    """ Run GLM on 4D data group_data design matrix 'X'

    Parameters
    ----------
    group_data : array shape (I, J, K, scans)
        4D array to fit to model with design `X`.  Column vectors are vectors
        over the final length T dimension.
    X : array ahape (scans, P)
        2D design matrix to fit to data `Y`.

    Returns
    -------
    B : array shape (I, J, K, P)
        parameter array, one length P vector of parameters for each voxel.
    sigma_2 : array shape (I, J, K)
        unbiased estimate of variance for each voxel.
    df : int
        degrees of freedom due to error.
    """
    if group_data.shape[-1] != X.shape[0]:
        raise ValueError('Number of regressors and labels must be equal')

    B, s_2, df = glm_4d(group_data, X)

    return B, s_2, df

def group_t_test(c, X, B, s_2, df):
    """ Two-tailed t-test on 3D estimates given contrast `c`, design `X`

        Parameters
        ----------
        c : array shape (P,)
            contrast specifying conbination of parameters to test.
        X : array shape (N, P)
            design matrix.
        B : array shape (I, J, K, P)
            parameter array, one length P vector of parameters for each voxel.
        sigma_2 : array shape (I, J, K)
            unbiased estimate of variance for each voxel.
        df : int
            degrees of freedom due to error.

        Returns
        -------
        t : array shape (I, J, K)
            t statistics for each data vector.
        p : array shape (V,)
            two-tailed probability value for each t statistic.
    """
    t, p = t_test_3d(c, X, B, s_2, df)

    return t, p

def peak_t(t):
    """ Return pos or neg t-stat with highest absolute value
    Parameters
    ----------
    t : array shape (I, J, K)
        t statistics for each data vector.

    Returns
    -------
    peak_t: float with highest absolute value

    """
    t_max = np.nanmax(t)
    t_min = np.nanmin(t)

    return max(t_max, t_min, key = abs)

group_data, img_names = load_group_data('../PPI_images/',suffix = '.img')
X = design_matrix('../PPI_images/regressors.txt', labels = ('RPS1', 'MDS_C1', 'MDS_C2', 'E_RCJ_acc', 'Sex'), img_names = img_names)
#B, s_2, df = group_glm(group_data, X)
#c = np.array([1, 0, 0, 0, 0, 0])
#t, p = group_t_test(c, X, B, s_2, df)

# iterations to obtain distribution for t-values
iterations = 1000
t_dist = np.empty(iterations)
c = np.array([1, 0, 0, 0, 0, 0])

for i in range(iterations):
    group_data_shuffled = shuffle_signs(group_data)
    B_group, s_2, df = group_glm(group_data_shuffled, X)
    t, p = group_t_test(c, X, B_group, s_2, df)
    t_iter = peak_t(t)
    t_dist[i] = t_iter
