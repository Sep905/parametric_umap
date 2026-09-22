"""Parametric UMAP implementation for dimensionality reduction using neural networks."""

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from parametric_umap.datasets.covariates_datasets import VariableDataset
from parametric_umap.datasets.edge_dataset import EdgeDataset
from parametric_umap.models.mlp import MLP
from parametric_umap.utils.graph import compute_all_p_umap
from parametric_umap.utils.losses import compute_correlation_loss


class ParametricUMAP:
    """A parametric implementation of UMAP (Uniform Manifold Approximation and Projection).

    This class implements a parametric version of UMAP that learns a neural network to perform
    dimensionality reduction. The model can transform new data points without having to recompute
    the entire embedding.

    Attributes:
        n_components (int): Number of dimensions in the output embedding
        hidden_dim (int): Dimension of hidden layers in the MLP
        n_layers (int): Number of hidden layers in the MLP
        n_neighbors (int): Number of neighbors to consider for each point
        a (float): UMAP parameter controlling local connectivity
        b (float): UMAP parameter controlling the strength of repulsion between points
        correlation_weight (float): Weight of the correlation loss term
        learning_rate (float): Learning rate for the optimizer
        n_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        device (str or torch.device): Device to use for computations ('cpu' or 'cuda')
        use_batchnorm (bool): Whether to use batch normalization in the MLP
        use_dropout (bool): Whether to use dropout in the MLP
        compile_model (bool): Whether to apply ``torch.compile`` to the MLP
        parametric_reconstruction (bool): Whether to train a decoder
        autoencoder_loss (bool): Whether reconstruction loss updates the encoder
        parametric_reconstruction_loss_weight (float): Weight of reconstruction loss
        reconstruction_loss (str): Reconstruction loss type (``"mse"`` or ``"bce"``)
        model (Optional[MLP]): The neural network model (possibly compiled)
        decoder (Optional[MLP]): The reconstruction network (possibly compiled)
        is_fitted (bool): Whether the model has been fitted

    """

    def __init__(
        self,
        n_components: int = 2,
        hidden_dim: int = 1024,
        n_layers: int = 3,
        n_neighbors: int = 15,
        a: float = 0.1,
        b: float = 1.0,
        correlation_weight: float = 0.1,
        learning_rate: float = 1e-4,
        n_epochs: int = 10,
        batch_size: int = 32,
        device: str | torch.device | None = None,
        use_batchnorm: bool = False,
        use_dropout: bool = False,
        compile_model: bool = False,
        parametric_reconstruction: bool = False,
        autoencoder_loss: bool = False,
        parametric_reconstruction_loss_weight: float = 1.0,
        reconstruction_loss: str = "mse",
    ) -> None:
        """Initialize ParametricUMAP.

        Parameters
        ----------
        n_components : int
            Number of dimensions in the output embedding
        hidden_dim : int
            Dimension of hidden layers in the MLP
        n_layers : int
            Number of hidden layers in the MLP
        n_neighbors : int
            Number of neighbors to consider for each point
        a, b : float
            UMAP parameters for the optimization
        correlation_weight : float
            Weight of the correlation loss term
        learning_rate : float
            Learning rate for the optimizer
        n_epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        device : str or torch.device, optional
            Device to use for computations ('cpu', 'cuda', or 'mps').
            Auto-detected if not specified (CUDA > MPS > CPU).
        use_batchnorm : bool
            Whether to use batch normalization in the MLP
        use_dropout : bool
            Whether to use dropout in the MLP
        compile_model : bool
            Whether to apply ``torch.compile`` to the MLP. Can yield
            10-30 % faster training on PyTorch 2.x at the cost of a
            one-time compilation delay on the first forward pass.
        parametric_reconstruction : bool
            Whether to train a decoder from the embedding to the input space.
        autoencoder_loss : bool
            Whether reconstruction loss also updates the encoder. Requires
            ``parametric_reconstruction=True``.
        parametric_reconstruction_loss_weight : float
            Weight of reconstruction loss relative to the UMAP loss.
        reconstruction_loss : {"mse", "bce"}
            Reconstruction loss. BCE requires input features in [0, 1].

        """
        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_neighbors = n_neighbors
        self.a = a
        self.b = b
        self.correlation_weight = correlation_weight
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.compile_model = compile_model
        self.parametric_reconstruction = parametric_reconstruction
        self.autoencoder_loss = autoencoder_loss
        self.parametric_reconstruction_loss_weight = parametric_reconstruction_loss_weight
        self.reconstruction_loss = reconstruction_loss.lower()

        if self.autoencoder_loss and not self.parametric_reconstruction:
            raise ValueError("autoencoder_loss=True requires parametric_reconstruction=True")
        if self.reconstruction_loss not in {"mse", "bce"}:
            raise ValueError("reconstruction_loss must be either 'mse' or 'bce'")
        if self.parametric_reconstruction_loss_weight < 0:
            raise ValueError("parametric_reconstruction_loss_weight must be non-negative")

        self.model = None
        self.decoder = None
        self.loss_fn = nn.BCELoss()
        self.reconstruction_loss_fn = nn.MSELoss() if self.reconstruction_loss == "mse" else nn.BCEWithLogitsLoss()
        self.loss_history_: dict[str, list[float]] = {}
        self.is_fitted = False

    @property
    def _unwrapped_model(self) -> MLP | None:
        """Return the underlying MLP, unwrapping ``torch.compile`` if needed."""
        if self.model is None:
            return None
        return getattr(self.model, "_orig_mod", self.model)

    @property
    def _unwrapped_decoder(self) -> MLP | None:
        """Return the underlying decoder MLP, unwrapping ``torch.compile`` if needed."""
        if self.decoder is None:
            return None
        return getattr(self.decoder, "_orig_mod", self.decoder)

    def _init_model(self, input_dim: int) -> None:
        """Initialize the MLP model.

        Parameters
        ----------
        input_dim : int
            The input dimension of the data

        """
        model = MLP(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_components,
            num_layers=self.n_layers,
            use_batchnorm=self.use_batchnorm,
            use_dropout=self.use_dropout,
        ).to(self.device)
        self.model = torch.compile(model) if self.compile_model else model

        if self.parametric_reconstruction:
            decoder = MLP(
                input_dim=self.n_components,
                hidden_dim=self.hidden_dim,
                output_dim=input_dim,
                num_layers=self.n_layers,
                use_batchnorm=self.use_batchnorm,
                use_dropout=self.use_dropout,
            ).to(self.device)
            self.decoder = torch.compile(decoder) if self.compile_model else decoder

    @staticmethod
    def _precompute_edge_tensors(
        X: np.ndarray, edges: np.ndarray, weights: np.ndarray, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build device-placed tensors for edge weights and input-space distances."""
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
        x_dists = np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1).astype(np.float32)
        x_dists_t = torch.tensor(x_dists, dtype=torch.float32, device=device)
        return weights_t, x_dists_t

    def _validate_reconstruction_input(self, X: np.ndarray) -> None:
        """Validate input constraints imposed by the reconstruction loss."""
        if self.parametric_reconstruction and self.reconstruction_loss == "bce" and np.any((X < 0) | (X > 1)):
            raise ValueError("BCE reconstruction loss requires all input features to be in [0, 1]")

    def _training_parameters(self) -> list[nn.Parameter]:
        """Return all parameters optimized during fitting."""
        parameters = list(self.model.parameters())
        if self.decoder is not None:
            parameters.extend(self.decoder.parameters())
        return parameters

    def _prepare_training(
        self,
    ) -> None:
        """Set model modes and initialize per-fit loss history."""
        self.model.train()
        if self.decoder is not None:
            self.decoder.train()
        self.loss_history_ = {
            "loss": [],
            "umap_loss": [],
            "correlation_loss": [],
            "reconstruction_loss": [],
        }

    def _prepare_training_batch(
        self,
        dataset: VariableDataset,
        edge_batch: np.ndarray,
        batch_idx: np.ndarray,
        all_weights_t: torch.Tensor,
        all_x_dists_t: torch.Tensor,
        low_memory: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather one edge batch and move it to the compute device if needed."""
        src_values = dataset[edge_batch[:, 0]]
        dst_values = dataset[edge_batch[:, 1]]
        targets = all_weights_t[batch_idx]
        X_distances = all_x_dists_t[batch_idx]

        if low_memory:
            src_values = src_values.to(self.device)
            dst_values = dst_values.to(self.device)
            targets = targets.to(self.device)
            X_distances = X_distances.to(self.device)

        return src_values, dst_values, targets, X_distances

    def _compute_training_loss(
        self,
        src_values: torch.Tensor,
        dst_values: torch.Tensor,
        targets: torch.Tensor,
        X_distances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total and component losses for one edge batch."""
        src_embeddings = self.model(src_values)
        dst_embeddings = self.model(dst_values)
        Z_distances = torch.norm(src_embeddings - dst_embeddings, dim=1, p=2 * self.b)
        qs = torch.pow(1 + self.a * Z_distances, -1).clamp(1e-7, 1 - 1e-7)
        umap_loss = self.loss_fn(qs, targets)
        corr_loss = compute_correlation_loss(X_distances, Z_distances)
        reconstruction_loss = torch.zeros((), dtype=umap_loss.dtype, device=umap_loss.device)

        if self.decoder is not None:
            reconstruction_loss = self._compute_reconstruction_loss(src_values, src_embeddings)

        loss = (
            umap_loss
            + self.correlation_weight * corr_loss
            + self.parametric_reconstruction_loss_weight * reconstruction_loss
        )
        return loss, umap_loss, corr_loss, reconstruction_loss

    def fit(
        self,
        X: np.ndarray | torch.Tensor,
        resample_negatives: bool = False,
        low_memory: bool = False,
        random_state: int = 0,
        verbose: bool = True,
    ) -> "ParametricUMAP":
        """Fit the model using X as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can be numpy array or torch tensor.
        resample_negatives : bool, optional (default=False)
            Whether to resample negative edges at each epoch.
        low_memory : bool, optional (default=False)
            If True, keeps the data and edge weights on CPU and transfers
            per batch.  Trades speed for lower accelerator memory usage.
        random_state : int, optional (default=0)
            Random state for reproducibility.
        verbose : bool, optional (default=True)
            Whether to display progress bars and print statements.

        Returns
        -------
        self : ParametricUMAP
            The fitted model.

        """
        X = np.asarray(X).astype(np.float32)
        self._validate_reconstruction_input(X)

        # Initialize model if not already done
        if self.model is None:
            self._init_model(X.shape[1])

        # Create datasets.  In low_memory mode, keep data on CPU and
        # transfer each batch to the compute device during training.
        dataset = VariableDataset(X) if low_memory else VariableDataset(X).to(self.device)
        P_sym = compute_all_p_umap(X, k=self.n_neighbors)
        ed = EdgeDataset(P_sym)

        # Initialize optimizer
        optimizer = AdamW(self._training_parameters(), lr=self.learning_rate)

        # Training loop
        self._prepare_training()

        loader = ed.get_loader(
            batch_size=self.batch_size,
            sample_first=True,
            random_state=random_state,
            verbose=verbose,
        )

        # Pre-place edge weights and input-space distances on the compute
        # device (or CPU in low_memory mode) so batch slicing is fast.
        _tensor_device = "cpu" if low_memory else self.device
        all_weights_t, all_x_dists_t = self._precompute_edge_tensors(X, ed.all_edges, ed.all_weights, _tensor_device)

        if verbose:
            print("Training...")

        pbar = tqdm(range(self.n_epochs), desc="Epochs", position=0, disable=not verbose)
        for epoch in pbar:
            epoch_loss = 0.0
            epoch_umap_loss = 0.0
            epoch_corr_loss = 0.0
            epoch_reconstruction_loss = 0.0
            num_batches = 0

            for edge_batch, _weight_batch, batch_idx in tqdm(
                loader, desc=f"Epoch {epoch + 1}", position=1, leave=False, disable=not verbose
            ):
                optimizer.zero_grad(set_to_none=True)

                src_values, dst_values, targets, X_distances = self._prepare_training_batch(
                    dataset,
                    edge_batch,
                    batch_idx,
                    all_weights_t,
                    all_x_dists_t,
                    low_memory,
                )
                loss, umap_loss, corr_loss, reconstruction_loss = self._compute_training_loss(
                    src_values,
                    dst_values,
                    targets,
                    X_distances,
                )

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_umap_loss += umap_loss.item()
                epoch_corr_loss += corr_loss.item()
                epoch_reconstruction_loss += reconstruction_loss.item()
                num_batches += 1

            if resample_negatives:
                loader = ed.get_loader(
                    batch_size=self.batch_size,
                    sample_first=True,
                    random_state=random_state + epoch + 1,
                )
                all_weights_t, all_x_dists_t = self._precompute_edge_tensors(
                    X, ed.all_edges, ed.all_weights, _tensor_device
                )

            avg_loss = epoch_loss / num_batches
            self.loss_history_["loss"].append(avg_loss)
            self.loss_history_["umap_loss"].append(epoch_umap_loss / num_batches)
            self.loss_history_["correlation_loss"].append(epoch_corr_loss / num_batches)
            self.loss_history_["reconstruction_loss"].append(epoch_reconstruction_loss / num_batches)

            # Update progress bar with current loss
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            if verbose:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {avg_loss:.4f}")

        self.is_fitted = True
        return self

    def _compute_reconstruction_loss(
        self,
        values: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction loss with optional encoder gradient flow."""
        decoder_input = embeddings if self.autoencoder_loss else embeddings.detach()
        reconstructed = self.decoder(decoder_input)
        return self.reconstruction_loss_fn(reconstructed, values)

    def transform(self, X: np.ndarray | torch.Tensor, *, batch_size: int | None = None) -> np.ndarray:
        """Apply dimensionality reduction to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform. Can be numpy array or torch tensor.
        batch_size : int, optional
            If provided, process the input in batches of this size to avoid
            out-of-memory errors on large inputs. By default the entire input
            is processed in a single forward pass.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data in the low-dimensional space.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before transform")

        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self._unwrapped_model.input_dim:
            msg = f"X has {X.shape[1]} features, but model was fitted with {self._unwrapped_model.input_dim} features"
            raise ValueError(msg)

        self.model.eval()

        with torch.no_grad():
            if batch_size is None:
                X_t = torch.as_tensor(X, dtype=torch.float32).to(self.device)
                return self.model(X_t).cpu().numpy()

            parts = []
            for start in range(0, X.shape[0], batch_size):
                batch = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32).to(self.device)
                parts.append(self.model(batch).cpu())
            return torch.cat(parts, dim=0).numpy()

    def inverse_transform(
        self,
        X: np.ndarray | torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Decode embedding-space points into the original input space."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before inverse_transform")
        if self.decoder is None:
            raise RuntimeError("Model was fitted without parametric reconstruction")

        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self._unwrapped_decoder.input_dim:
            msg = f"X has {X.shape[1]} components, but decoder expects {self._unwrapped_decoder.input_dim} components"
            raise ValueError(msg)

        self.decoder.eval()

        def decode(batch: torch.Tensor) -> torch.Tensor:
            output = self.decoder(batch)
            return torch.sigmoid(output) if self.reconstruction_loss == "bce" else output

        with torch.no_grad():
            if batch_size is None:
                X_t = torch.as_tensor(X, dtype=torch.float32).to(self.device)
                return decode(X_t).cpu().numpy()

            parts = []
            for start in range(0, X.shape[0], batch_size):
                batch = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32).to(self.device)
                parts.append(decode(batch).cpu())
            return torch.cat(parts, dim=0).numpy()

    def reconstruct(
        self,
        X: np.ndarray | torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode and reconstruct input-space points."""
        embeddings = self.transform(X, batch_size=batch_size)
        return self.inverse_transform(embeddings, batch_size=batch_size)

    def fit_transform(
        self,
        X: np.ndarray | torch.Tensor,
        resample_negatives: bool = False,
        low_memory: bool = False,
        random_state: int = 0,
        verbose: bool = True,
    ) -> np.ndarray:
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can be numpy array or torch tensor.
        resample_negatives : bool, optional (default=False)
            Whether to resample negative edges at each epoch.
        low_memory : bool, optional (default=False)
            If True, keeps the data and edge weights on CPU and transfers
            per batch.  Trades speed for lower accelerator memory usage.
        random_state : int, optional (default=0)
            Random state for reproducibility.
        verbose : bool, optional (default=True)
            Whether to display progress bars and print statements.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data in the low-dimensional space.

        """
        self.fit(
            X,
            resample_negatives=resample_negatives,
            low_memory=low_memory,
            random_state=random_state,
            verbose=verbose,
        )
        return self.transform(X)

    def save(self, path: str) -> None:
        """Save the model to a file.

        Parameters
        ----------
        path : str
            Path to save the model.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")

        save_dict = {
            "checkpoint_version": 2,
            "model_state_dict": self._unwrapped_model.state_dict(),
            "decoder_state_dict": (
                self._unwrapped_decoder.state_dict() if self._unwrapped_decoder is not None else None
            ),
            "input_dim": self._unwrapped_model.input_dim,
            "n_components": self.n_components,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "a": self.a,
            "b": self.b,
            "correlation_weight": self.correlation_weight,
            "use_batchnorm": self.use_batchnorm,
            "use_dropout": self.use_dropout,
            "parametric_reconstruction": self.parametric_reconstruction,
            "autoencoder_loss": self.autoencoder_loss,
            "parametric_reconstruction_loss_weight": self.parametric_reconstruction_loss_weight,
            "reconstruction_loss": self.reconstruction_loss,
        }

        torch.save(save_dict, path)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "ParametricUMAP":
        """Load a saved model.

        Parameters
        ----------
        path : str
            Path to the saved model.
        device : str, optional
            Device to load the model to. Auto-detected if not specified.

        Returns
        -------
        model : ParametricUMAP
            The loaded model instance.

        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        save_dict = torch.load(path, map_location=device, weights_only=True)

        # Create instance with saved parameters
        instance = cls(
            n_components=save_dict["n_components"],
            hidden_dim=save_dict["hidden_dim"],
            n_layers=save_dict["n_layers"],
            a=save_dict["a"],
            b=save_dict["b"],
            correlation_weight=save_dict["correlation_weight"],
            device=device,
            use_batchnorm=save_dict["use_batchnorm"],
            use_dropout=save_dict["use_dropout"],
            parametric_reconstruction=save_dict.get("parametric_reconstruction", False),
            autoencoder_loss=save_dict.get("autoencoder_loss", False),
            parametric_reconstruction_loss_weight=save_dict.get(
                "parametric_reconstruction_loss_weight",
                1.0,
            ),
            reconstruction_loss=save_dict.get("reconstruction_loss", "mse"),
        )

        # Initialize model architecture (fall back to weight introspection for old checkpoints)
        input_dim = save_dict.get("input_dim") or save_dict["model_state_dict"]["model.0.weight"].shape[1]
        instance._init_model(input_dim=input_dim)

        # Load state dict
        instance.model.load_state_dict(save_dict["model_state_dict"])
        decoder_state_dict = save_dict.get("decoder_state_dict")
        if decoder_state_dict is not None:
            instance.decoder.load_state_dict(decoder_state_dict)
        instance.is_fitted = True

        return instance
