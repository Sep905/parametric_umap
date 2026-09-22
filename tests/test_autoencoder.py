"""Tests for parametric reconstruction and autoencoder training."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from scipy import sparse
from torch import nn

from parametric_umap import ParametricUMAP


class TestAutoencoderInitialization:
    """Test autoencoder configuration and model initialization."""

    def test_defaults_preserve_encoder_only_behavior(self):
        """Test that reconstruction remains disabled by default."""
        pumap = ParametricUMAP(device="cpu")
        pumap._init_model(input_dim=5)

        assert not pumap.parametric_reconstruction
        assert not pumap.autoencoder_loss
        assert pumap.decoder is None

    def test_decoder_uses_existing_mlp_architecture(self):
        """Test decoder input and output dimensions."""
        pumap = ParametricUMAP(
            n_components=3,
            hidden_dim=16,
            n_layers=2,
            device="cpu",
            parametric_reconstruction=True,
        )
        pumap._init_model(input_dim=7)

        assert pumap.decoder is not None
        assert pumap._unwrapped_decoder.input_dim == 3
        assert pumap._unwrapped_decoder.model[-1].out_features == 7

    @pytest.mark.parametrize(
        ("name", "expected_type"),
        [("mse", nn.MSELoss), ("bce", nn.BCEWithLogitsLoss), ("BCE", nn.BCEWithLogitsLoss)],
    )
    def test_reconstruction_loss_selection(self, name, expected_type):
        """Test supported reconstruction losses."""
        pumap = ParametricUMAP(reconstruction_loss=name)
        assert isinstance(pumap.reconstruction_loss_fn, expected_type)

    def test_autoencoder_loss_requires_decoder(self):
        """Test invalid autoencoder configuration."""
        with pytest.raises(ValueError, match="requires parametric_reconstruction"):
            ParametricUMAP(autoencoder_loss=True)

    def test_invalid_reconstruction_loss(self):
        """Test rejection of unsupported reconstruction losses."""
        with pytest.raises(ValueError, match="must be either"):
            ParametricUMAP(reconstruction_loss="mae")

    def test_negative_reconstruction_weight(self):
        """Test rejection of negative reconstruction weights."""
        with pytest.raises(ValueError, match="non-negative"):
            ParametricUMAP(parametric_reconstruction_loss_weight=-1.0)


class TestAutoencoderGradientFlow:
    """Test reconstruction gradient routing."""

    @pytest.mark.parametrize("autoencoder_loss", [False, True])
    def test_reconstruction_gradient_routing(self, autoencoder_loss):
        """Test that reconstruction optionally updates the encoder."""
        pumap = ParametricUMAP(
            n_components=2,
            hidden_dim=8,
            n_layers=1,
            device="cpu",
            parametric_reconstruction=True,
            autoencoder_loss=autoencoder_loss,
        )
        pumap._init_model(input_dim=4)
        values = torch.rand(6, 4)
        embeddings = pumap.model(values)
        embeddings.retain_grad()

        loss = pumap._compute_reconstruction_loss(values, embeddings)
        loss.backward()

        assert any(parameter.grad is not None for parameter in pumap.decoder.parameters())
        if autoencoder_loss:
            assert embeddings.grad is not None
            assert any(parameter.grad is not None for parameter in pumap.model.parameters())
        else:
            assert embeddings.grad is None
            assert all(parameter.grad is None for parameter in pumap.model.parameters())


class TestAutoencoderInference:
    """Test reconstruction and inverse transformation."""

    @pytest.fixture
    def fitted_autoencoder(self):
        """Create an initialized autoencoder for inference tests."""
        pumap = ParametricUMAP(
            n_components=2,
            hidden_dim=8,
            n_layers=1,
            device="cpu",
            parametric_reconstruction=True,
            autoencoder_loss=True,
        )
        pumap._init_model(input_dim=4)
        pumap.is_fitted = True
        return pumap

    def test_inverse_transform_shape(self, fitted_autoencoder):
        """Test decoding embedding-space points."""
        embeddings = np.random.randn(9, 2).astype(np.float32)
        reconstructed = fitted_autoencoder.inverse_transform(embeddings)

        assert reconstructed.shape == (9, 4)
        assert reconstructed.dtype == np.float32

    def test_batched_inverse_transform(self, fitted_autoencoder):
        """Test that batched decoding matches a single forward pass."""
        embeddings = np.random.randn(9, 2).astype(np.float32)

        expected = fitted_autoencoder.inverse_transform(embeddings)
        actual = fitted_autoencoder.inverse_transform(embeddings, batch_size=4)

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_reconstruct_shape(self, fitted_autoencoder):
        """Test the encode-decode convenience method."""
        X = np.random.randn(9, 4).astype(np.float32)
        reconstructed = fitted_autoencoder.reconstruct(X, batch_size=4)

        assert reconstructed.shape == X.shape

    def test_bce_inverse_transform_range(self):
        """Test sigmoid output for BCE reconstruction."""
        pumap = ParametricUMAP(
            n_components=2,
            hidden_dim=8,
            n_layers=1,
            device="cpu",
            parametric_reconstruction=True,
            reconstruction_loss="bce",
        )
        pumap._init_model(input_dim=4)
        pumap.is_fitted = True

        reconstructed = pumap.inverse_transform(np.random.randn(9, 2).astype(np.float32))

        assert np.all(reconstructed >= 0)
        assert np.all(reconstructed <= 1)

    def test_inverse_transform_requires_fitted_model(self):
        """Test inverse transformation before fitting."""
        pumap = ParametricUMAP(parametric_reconstruction=True)
        with pytest.raises(RuntimeError, match="must be fitted"):
            pumap.inverse_transform(np.zeros((2, 2), dtype=np.float32))

    def test_inverse_transform_requires_decoder(self):
        """Test inverse transformation on an encoder-only model."""
        pumap = ParametricUMAP(device="cpu")
        pumap._init_model(input_dim=4)
        pumap.is_fitted = True

        with pytest.raises(RuntimeError, match="without parametric reconstruction"):
            pumap.inverse_transform(np.zeros((2, 2), dtype=np.float32))

    def test_inverse_transform_validates_components(self, fitted_autoencoder):
        """Test validation of decoder input dimensions."""
        with pytest.raises(ValueError, match="decoder expects"):
            fitted_autoencoder.inverse_transform(np.zeros((2, 3), dtype=np.float32))


class TestAutoencoderTraining:
    """Test autoencoder integration with the UMAP training loop."""

    def test_fit_trains_decoder_and_records_losses(self):
        """Test one complete autoencoder training epoch."""
        X = np.random.rand(12, 4).astype(np.float32)
        nodes = np.arange(len(X))
        rows = np.concatenate([nodes, nodes])
        cols = np.concatenate([np.roll(nodes, 1), np.roll(nodes, -1)])
        graph = sparse.csr_matrix((np.full(len(rows), 0.8), (rows, cols)), shape=(len(X), len(X)))
        pumap = ParametricUMAP(
            n_components=2,
            hidden_dim=8,
            n_layers=1,
            n_neighbors=2,
            n_epochs=1,
            batch_size=32,
            device="cpu",
            correlation_weight=0.0,
            parametric_reconstruction=True,
            autoencoder_loss=True,
            reconstruction_loss="mse",
        )

        with patch("parametric_umap.core.compute_all_p_umap", return_value=graph):
            pumap.fit(X, verbose=False)

        assert pumap.is_fitted
        assert any(parameter.grad is not None for parameter in pumap.decoder.parameters())
        assert set(pumap.loss_history_) == {
            "loss",
            "umap_loss",
            "correlation_loss",
            "reconstruction_loss",
        }
        assert all(len(values) == 1 for values in pumap.loss_history_.values())
        assert np.isfinite(pumap.loss_history_["reconstruction_loss"][0])

    def test_bce_fit_rejects_unscaled_input(self):
        """Test BCE target range validation before graph construction."""
        X = np.array([[0.0, 1.1], [-0.1, 0.5]], dtype=np.float32)
        pumap = ParametricUMAP(parametric_reconstruction=True, reconstruction_loss="bce")

        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            pumap.fit(X, verbose=False)


class TestAutoencoderPersistence:
    """Test autoencoder checkpoint handling."""

    @pytest.mark.parametrize("reconstruction_loss", ["mse", "bce"])
    def test_save_load_roundtrip(self, reconstruction_loss, temp_model_file):
        """Test that encoder and decoder outputs survive a round trip."""
        pumap = ParametricUMAP(
            n_components=2,
            hidden_dim=8,
            n_layers=1,
            device="cpu",
            parametric_reconstruction=True,
            autoencoder_loss=True,
            parametric_reconstruction_loss_weight=0.25,
            reconstruction_loss=reconstruction_loss,
        )
        pumap._init_model(input_dim=4)
        pumap.is_fitted = True
        X = np.random.rand(6, 4).astype(np.float32)
        expected_embeddings = pumap.transform(X)
        expected_reconstructions = pumap.reconstruct(X)

        pumap.save(str(temp_model_file))
        restored = ParametricUMAP.load(str(temp_model_file), device="cpu")

        assert restored.parametric_reconstruction
        assert restored.autoencoder_loss
        assert restored.parametric_reconstruction_loss_weight == 0.25
        assert restored.reconstruction_loss == reconstruction_loss
        np.testing.assert_allclose(restored.transform(X), expected_embeddings, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(restored.reconstruct(X), expected_reconstructions, rtol=1e-6, atol=1e-6)

    def test_load_legacy_encoder_checkpoint(self, temp_model_file):
        """Test loading a checkpoint without reconstruction fields."""
        pumap = ParametricUMAP(n_components=2, hidden_dim=8, n_layers=1, device="cpu")
        pumap._init_model(input_dim=4)
        checkpoint = {
            "model_state_dict": pumap.model.state_dict(),
            "input_dim": 4,
            "n_components": 2,
            "hidden_dim": 8,
            "n_layers": 1,
            "a": 0.1,
            "b": 1.0,
            "correlation_weight": 0.1,
            "use_batchnorm": False,
            "use_dropout": False,
        }
        torch.save(checkpoint, temp_model_file)

        restored = ParametricUMAP.load(str(temp_model_file), device="cpu")

        assert restored.is_fitted
        assert not restored.parametric_reconstruction
        assert restored.decoder is None
