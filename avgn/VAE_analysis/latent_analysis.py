import tensorflow as tf
import numpy as np
from tqdm.autonotebook import tqdm

def get_latent_vectors(
    model=None, 
    data=None, 
    num_samples=None, #Sets number of samples to extract, if None, extracts all
    verbose=False, 
    metadata_index=[], 
):
    """ Extracts latent vectors from a model for a given dataset.
    Args:
        model: A trained model with an encode method that returns mean and logvar.
        data: A dataset to extract latent vectors from.
        num_samples: Number of samples to extract. If None, extracts all.
        verbose: If True, prints additional information.
        metadata_index: List of indices to extract metadata from the batch.
    Returns:
        all_means: Concatenated means of the latent vectors.
        all_logvars: Concatenated log variances of the latent vectors.
        all_metadata: List of metadata dictionaries for each batch.
    """

    if model is None:  
        raise ValueError("Model must be provided to extract latent vectors.")
    
    if data is None:
        raise ValueError("Data must be provided to extract latent vectors.")

    if num_samples is not None:
        total = num_samples
    elif num_samples is None:
        total = sum(1 for _ in data) 

    all_means = []
    all_logvars = []
    all_metadata = []
    
    for idx, batch in tqdm(enumerate(data),total=total, desc="Extracting latent vectors"):
        images = batch[0]  # assuming (images, labels)
        images = tf.cast(images, tf.float32) / 255.0
        images = tf.reshape(images, (-1, 128, 128, 1))

        if len(metadata_index) > 0:
            all_metadata.append({idn+1 : key for idn, key in enumerate(batch) if idn in metadata_index})
            if verbose:
                print(all_metadata[idx])

        # Encode the batch
        mean, logvar = model.encode(images)
        all_means.append(mean.numpy())
        all_logvars.append(logvar.numpy())

        if num_samples is not None and idx >= num_samples - 1:
            break

    # Concatenate all latent vectors from batches
    all_means = np.vstack(all_means)
    all_logvars = np.vstack(all_logvars)

    if verbose:
        print('Total latent space shape:', all_means.shape, all_logvars.shape)

    return all_means, all_logvars, all_metadata

def get_row_medians(
    latent_space = None, 
    verbose=False, 
    window_size=5
):
    """ Computes the row median of the latent space.
    Args:
        latent_space: A numpy array of shape (i, d) where i is the number of samples and d is the dimensionality of the latent space.
        verbose: If True, prints additional information.
        window_size: Size of the window to compute the rolling median.
    Returns:
        row_median: A numpy array of shape (i,) containing the median for each row.
    """

    if latent_space is None:
        raise ValueError("Latent space must be provided to compute row variance.")

    specs = latent_space # shape (i, d)
    
    # Compute pairwise differences, result will be (i, d, j)
    diff_matrix = specs[:, None, :] - specs[None, :, :]  # shape (i, j, d)
    
    # Example input
    x = diff_matrix  # shape (i=10, j=10, d=32)
    
    # Compute L2 norm along the 'd' dimension (axis=2)
    l2_norm = np.linalg.norm(x, axis=2, keepdims=True)  # shape will be (i, j, 1)
    
    # Remove the last singleton dimension to work easily with sorting
    l2_flat = l2_norm.squeeze(axis=2)  # shape (i, j)

    half_w = window_size // 2
    i, j = l2_flat.shape

    if half_w < 1:
        raise ValueError("Window size must be greater than 1 to compute a median.")
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number to compute a centered median.")
    
    # Compute median from centered window for each row
    row_median = np.array([[np.median(row[idx-half_w:idx+half_w])] for idx, row in enumerate(l2_flat) if (idx+half_w < j) & (idx > half_w)]).flatten()  
    
    if verbose:
        print('shape of specs:', specs.shape)#,'\n', specs)
        print('shape of piecewise:', diff_matrix.shape)#,'\n', diff_matrix)
        print('shape of l2 norms:', l2_norm.shape)#,'\n', l2_norm)
        print('shape of row_medians:', row_median.shape)#,'\n', mins)
    
    return row_median