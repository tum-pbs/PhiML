"""
Dimensionality reduction and latent space models.

This module provides functionality for fitting and using latent models, such as UMAP, to reduce the dimensionality of data.
"""
from dataclasses import dataclass
from typing import Union, Any, Optional

from . import math, channel, instance, Tensor, wrap, Shape
from .math import DimFilter
from .math._shape import auto


@dataclass
class LatentModel:
    model_type: str
    model: Any
    feature_dim: Shape
    latent_dim: Shape
    can_reconstruct: bool

    def embed(self, data: Tensor) -> Tensor:
        """
        Embed the data using the fitted model or by training a new dimensionality reduction model.

        Args:
            data: Tensor to embed. Must contain `list_dim`. If no `feature_dim` is present, 1D data is assumed.

        Returns:
            The embedded data as a `Tensor`.
        """
        list_dim = data.shape.without(self.feature_dim)
        if self.model_type == 'UMAP':
            def embed_single(x):
                embedded_np = self.model.transform(x.numpy([list_dim, self.feature_dim]))
                embedded_nat = data.backend.as_tensor(embedded_np)
                return wrap(embedded_nat, [list_dim, self.latent_dim])
            return math.map(embed_single, data, dims=data.shape - self.feature_dim - list_dim)

    __call__ = embed

    def reconstruct(self, latent: Tensor) -> Tensor:
        """
        Reconstruct the data from the latent representation if supported.
        Check `self.can_reconstruct` to see if reconstruction is supported.

        Args:
            latent: Tensor containing the latent representation. Must contain `self.latent_dim`.

        Returns:
            The reconstructed data as a `Tensor`.
        """
        list_dim = latent.shape.without(self.latent_dim)
        if self.model_type == 'UMAP':
            def embed_single(x):
                reconstructed_np = self.model.inverse_transform(x.numpy([list_dim, self.latent_dim]))
                reconstructed_nat = latent.backend.as_tensor(reconstructed_np)
                return wrap(reconstructed_nat, [list_dim, self.feature_dim])
            return math.map(embed_single, latent, dims=latent.shape - self.latent_dim - list_dim)


def fit(data: Tensor, model: str, latent_dim: Union[str, Shape, int] = channel(vector='l1,l2'), feature_dim: DimFilter = channel, list_dim: DimFilter = instance, **model_kwargs) -> LatentModel:
    """
    Fit a latent model to the data.

    Args:
        data: Tensor to fit the model to. Must contain `list_dim`. If no `feature_dim` is present, 1D data is assumed.
        model: Model type to fit. Currently only 'UMAP' is supported.
        latent_dim: The dimension of the latent space to use. This determines the number of components in the latent representation.
            The shape can be passed as a string spec, e.g. '(x,y)'.
            Alternatively, the number of components can be passed as an `int`, in which case the shape will be channel of name 'vector'.
        feature_dim: The dimension of the feature space used in distance computations.
        list_dim: Dimension along which data points belonging to one model are listed. Any dims not marked as list or feature are considered as batch dims.

    Returns:
        LatentModel: A LatentModel object containing the fitted model and its parameters.
    """
    if isinstance(latent_dim, int):
        latent_dim = channel(vector=['l'+str(i) for i in range(latent_dim)])
    elif isinstance(latent_dim, str):
        latent_dim = auto(latent_dim)
    assert isinstance(latent_dim, Shape), f"Expected latent_dim to be a Shape, shape-string or int, but got {type(latent_dim)}"
    feature_dim = data.shape.only(feature_dim)
    if model == 'UMAP':
        from umap import UMAP
        def fit_single(x):
            umap_model = UMAP(n_components=latent_dim.volume, **model_kwargs)
            umap_model.fit(x.numpy([list_dim, feature_dim]))
            return LatentModel(model, umap_model, feature_dim, latent_dim, latent_dim.volume > 1)
        return math.map(fit_single, data, dims=data.shape - feature_dim - list_dim)
    else:
        raise ValueError(f"Unknown model name: {model}")
