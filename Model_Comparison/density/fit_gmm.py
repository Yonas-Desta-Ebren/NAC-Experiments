import torch
import pickle
import os
from .gmm import GMM

def fit_gmm(data_loader, model, latent_dim, gmm_file="gmm.pth", n_components=75, max_iter=5,
                        assume_diag_cov=False, init_kmeans=True, device=None):
    """
    Fit a GMM using latent representations extracted from a model.

    Args:
        data_loader: DataLoader providing data batches.
        model: The trained model to extract latent representations.
        latent_dim: Dimension of the latent space.
        gmm_file: File to save the fitted GMM model.
        n_components: Number of GMM components.
        max_iter: Maximum iterations for EM algorithm.
        assume_diag_cov: Use diagonal covariance if True.
        init_kmeans: Use K-Means for initialization if True.
        device: Device to use for computation (default: auto-detect).

    Returns:
        gmm: Trained GMM model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    data_list = []

    with torch.no_grad():
        for batch in data_loader:
            batch_data = batch[0].to(device)
            batch_data = (batch_data > 0.5).float()
            batch_data = batch_data.view(batch_data.size(0), -1)
            output = model.encoder(batch_data)

            if isinstance(output, tuple):  
                latent_vectors = output[0] 
            else:
                latent_vectors = output 

            data_list.append(latent_vectors)

    data = torch.cat(data_list, dim=0)
    assert data.size(1) == latent_dim, f"Expected latent dimension {latent_dim}, but got {data.size(1)}"

    print(f"Collected latent data shape: {data.shape}")

    if not isinstance(gmm_file, str):
        gmm_file = f"gmm.pth"

    if os.path.exists(gmm_file):
        os.remove(gmm_file)
        print(f"Removed existing {gmm_file}")
    
    gmm = GMM(k=n_components, max_iter=max_iter, assume_diag_cov=assume_diag_cov, init_kmeans=init_kmeans)
    gmm.fit(data)

    # Save the GMM model
    print(f"Saving GMM to file: {gmm_file}")
    torch.save(gmm, gmm_file)
    return gmm