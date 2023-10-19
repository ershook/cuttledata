from collections import OrderedDict 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import plenoptic as po
import scipy.io as sio
import os
import os.path as op
import glob
import math
import pyrtools as pt
from tqdm import tqdm
from PIL import Image
import plenoptic



class PortillaSimoncelliMinimalStats(po.simul.PortillaSimoncelli):
    r"""Model for obtaining the exact set of statistics reported in
    Portilla and Simoncelli (2000).
    Parameters
    ----------
    im_shape: int
        the size of the images being processed by the model
    n_scales: int
        number of scales of the steerable pyramid
    n_orientations: int
        number of orientations of the steerable pyramid
    spatial_corr_width: int
        width of the spatial correlation window
    use_true_correlations: bool
        if True, use the true correlations, otherwise use covariances
    """
    def __init__(
        self,
        im_shape,
        n_scales=4,
        n_orientations=4,
        spatial_corr_width=9,
        use_true_correlations=False,
    ):
        super().__init__(im_shape, n_scales=n_scales,
                         n_orientations=n_orientations,
                         spatial_corr_width=spatial_corr_width,
                         use_true_correlations=use_true_correlations)
        self.statistics_mask = self.mask_extra_statistics()

    def mask_extra_statistics(self):
        r"""Generate a dictionary with the same structure as the statistics
        dictionary, containing masks that indicate for each statistics
        whether it is part of the minimal set of original statistics (True)
        or not (False).
        """
        n = self.spatial_corr_width
        n_scales = self.n_scales
        n_orientations = self.n_orientations
        mask_original = OrderedDict()  # Masks of original statistics
        # Add mask elements in same order as the po statistics dict
        
        #### pixel_statistics ####
        # All in original statistics
        mask_original['pixel_statistics'] = torch.tensor([True] * 6)
        
        #### magnitude_means ####
        # Not in original paper
        mask_original['magnitude_means'] = torch.tensor([False] * ((n_scales * \
            n_orientations) + 2))
        
        #### auto_correlation_magnitude ####
        # Symmetry M_{i,j} = M_{n-i+1, n_j+1}
        # Start with 0's and put 1's in original elements
        acm_mask = torch.zeros((n, n, n_scales, n_orientations))
        # Lower triangular (including diagonal) to ones
        tril_inds = torch.tril_indices(n, n)
        acm_mask[tril_inds[0], tril_inds[1], :, :] = 1
        # Set repeated diagnoal elements to 0
        diag_repeated = torch.arange(start=(n+1)/2, end=n, dtype=torch.long)
        acm_mask[diag_repeated, diag_repeated, :, :] = 0
        mask_original['auto_correlation_magnitude'] = acm_mask.bool()
        
        #### skew_reconstructed, kurtosis_reconstructed ####
        # All in original paper
        mask_original['skew_reconstructed'] = torch.tensor([True] * (n_scales + 1))
        mask_original['kurtosis_reconstructed'] = torch.tensor([True] * (n_scales + 1))
        
        #### auto_correlation_reconstructed ####
        # Symmetry M_{i,j} = M_{n-i+1, n-j+1}
        acr_mask = torch.zeros((n, n, n_scales+1))
        # Reuse templates from acm
        acr_mask[tril_inds[0], tril_inds[1], :] = 1
        acr_mask[diag_repeated, diag_repeated, :] = 0
        mask_original['auto_correlation_reconstructed'] = acr_mask.bool()
        if self.use_true_correlations:
            # std_reconstructed holds the center values of the
            # auto_correlation_reconstructed matrices. Which are turned
            # to 1's when using correlations
            mask_original['std_reconstructed'] = \
                torch.tensor([True] * (n_scales + 1))
            
        #### cross_orientation_correlation magnitude ####
        # Symmetry M_{i,j} = M_{j,i}. Diagonal elements are redundant with the
        # central elements of acm matrices. Last scale is full of 0's
        # Start with 1's and set redundant elements to 0
        cocm_mask = torch.ones((n_orientations, n_orientations, n_scales+1))
        # Template of redundant indices (diagonals are redundant)
        triu_inds = torch.triu_indices(n_orientations, n_orientations)
        cocm_mask[triu_inds[0], triu_inds[1], :] = 0
        # Set to 0 last scale that is not in the paper
        cocm_mask[:, :, -1] = 0
        mask_original['cross_orientation_correlation_magnitude'] = cocm_mask.bool()
        
        #### cross_scale_correlation_magnitude ####
        # No symmetry. Last scale is always 0
        cscm_mask = torch.ones((n_orientations, n_orientations, n_scales))
        cscm_mask[:,:,-1] = 0
        mask_original['cross_scale_correlation_magnitude'] = cscm_mask.bool()
        
        #### cross_orientation_correlation_real ####
        # Not included in paper's statistics
        mask_original['cross_orientation_correlation_real'] = torch.zeros( \
            (n_orientations*2, n_orientations*2, n_scales+1)).bool()
        
        #### cross_scale_correlation_real ####
        # No symmetry. Bottom half of matrices are 0's always.
        # Last scale is not included in paper's statistics
        cscr_mask = torch.ones((n_orientations*2, n_orientations*2, n_scales))
        cscr_mask[(n_orientations):,:,:] = 0
        cscr_mask[:,:,(n_scales-1):] = 0
        mask_original['cross_scale_correlation_real'] = cscr_mask.bool()
        
        ### var highpass residual ####
        # Not redundant
        mask_original['var_highpass_residual'] = torch.tensor(True)
        
        ### Adjust dictionary for correlation matrices ####
        if self.use_true_correlations:
            # Constant 1's in the correlation matrices not in original set
            ctrind = torch.tensor([n//2])
            mask_original['auto_correlation_reconstructed'][ctrind, ctrind, :] = False
            mask_original['auto_correlation_magnitude'][ctrind, ctrind, :, :] = False
            # Remove from original set the diagonal elements of
            # cross_orientation_correlation_magnitude matrices
            # that are 1's in correlation matrices
            dgind = torch.arange(n_orientations)
            mask_original['cross_orientation_correlation_magnitude'][dgind, dgind,:-1] = True
        return mask_original


    def forward(self, image, scales=None):
        r"""Generate Texture Statistics representation of an image with
        the statistics not reported in the paper turned to 0.

        Parameters
        ----------
        image : torch.Tensor
            A tensor containing the image to analyze.
        scales : list, optional
            Which scales to include in the returned representation. If an empty
            list (the default), we include all scales. Otherwise, can contain
            subset of values present in this model's ``scales`` attribute.

        Returns
        -------
        representation_vector: torch.Tensor
            A flattened tensor (1d) containing the measured representation statistics.

        """
        # create the representation vector with (with all scales)
        stats_vec = super().forward(image)
        # Turn the mask dictionary into a vector
        mask_vec = self.convert_to_vector(self.statistics_mask)
        # Multiply the statistics vector by the mask vector
        stats_vec = torch.mul(stats_vec, mask_vec)
        return stats_vec