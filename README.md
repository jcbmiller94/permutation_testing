# permutation_testing

### Framework for permutation testing for group (2nd) level fMRI analysis 
Will include voxel-wise and cluster-wise permutation testing 

In brief, images are permuted with labels of (1) or (-1) and a max t-vaue is obtained after applying
 group-level GLM. This is done ~1000 times and to create a distribution of t-statistics that is free 
 of several assumptions, including Gaussian-distributed noise in the data at the 2nd level 
