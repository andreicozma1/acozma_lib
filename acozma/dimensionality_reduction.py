import colorsys
from enum import Enum
from typing import Optional


import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import (
    TSNE,
    SpectralEmbedding,
    MDS,
    Isomap,
    LocallyLinearEmbedding,
)
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from textwrap import wrap


class EmbedMethods(str, Enum):
    PCA = "Principal Component Analysis (PCA)"

    TSNE = "T-distributed Stochastic Neighbor Embedding (TSNE)"
    UMAP = "Uniform Manifold Approximation and Projection (UMAP)"

    SVD = "Truncated Singular Value Decomposition (SVD)"
    ICA = "Independent Component Analysis (ICA)"
    SPEC_NN = "Spectral Embedding (NN)"
    SPEC_RBF = "Spectral Embedding (RBF)"
    MDS = "Multidimensional Scaling (MDS)"
    LLE = "Locally Linear Embedding (LLE)"

    ISOMAP = "ISOMAP"

    LDA = "Linear Dinscriminant Analysis (LDA)"
    NCA = "Neighborhood Components Analysis (NCA)"

    def __str__(self):
        return self.value


def reduce_dimension(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    method: EmbedMethods = EmbedMethods.PCA,
    **kwargs,
):
    kwargs.setdefault("n_components", 3)
    print(f"reduce_dimension: {method} ({x.shape}) ({kwargs})")

    match method:
        case EmbedMethods.PCA:
            m = PCA(**kwargs)
            low_dim = m.fit_transform(x)

        case EmbedMethods.TSNE:
            kwargs.setdefault("n_iter", 2500)
            m = TSNE(**kwargs)
            low_dim = m.fit_transform(x)

        case EmbedMethods.UMAP:
            import umap

            m = umap.UMAP(**kwargs)
            low_dim = m.fit_transform(x)

        case EmbedMethods.SVD:
            m = TruncatedSVD(**kwargs)
            low_dim = m.fit_transform(x)

        case EmbedMethods.ICA:
            m = FastICA(**kwargs)
            low_dim = m.fit_transform(x)

        case EmbedMethods.SPEC_NN:
            kwargs = {
                **kwargs,
                "affinity": "nearest_neighbors",
            }
            m = SpectralEmbedding(**kwargs)
            low_dim = m.fit_transform(x)

        case EmbedMethods.SPEC_RBF:
            kwargs = {
                **kwargs,
                "affinity": "rbf",
            }
            m = SpectralEmbedding(**kwargs)
            low_dim = m.fit_transform(x)

        case EmbedMethods.MDS:
            m = MDS(**kwargs)
            low_dim = m.fit_transform(x)

        case EmbedMethods.LLE:
            m = LocallyLinearEmbedding(**kwargs)
            low_dim = m.fit_transform(x)

        case EmbedMethods.ISOMAP:
            m = Isomap(**kwargs)
            low_dim = m.fit_transform(x)

        case EmbedMethods.LDA:
            # fix n_components cannot be larger than min(n_features, n_classes - 1).
            kwargs["n_components"] = min(
                kwargs["n_components"], np.unique(y).shape[0] - 1
            )
            m = LinearDiscriminantAnalysis(**kwargs)
            low_dim = m.fit_transform(x, y)

        case EmbedMethods.NCA:
            m = NeighborhoodComponentsAnalysis(**kwargs)
            low_dim = m.fit(x, y).transform(x)
        case _:
            raise ValueError(f"Unknown Metric: {method}")

    return m, low_dim


def generate_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 1, 1)
        colors.append(rgb)
    return colors


def get_scatter(
    features: np.ndarray, captions: Optional[list[str]] = None, **marker_kwargs
):
    print("get_scatter")
    marker_kwargs.setdefault("size", 5)
    marker_kwargs.setdefault("colorscale", "Turbo")

    scatter_kwargs = {}

    if captions is not None:
        scatter_kwargs |= {
            "hoverinfo": "text",
            "hovertemplate": "%{text}",
            "text": ["<br>".join(wrap(caption, width=75)) for caption in captions],
        }

    match features.shape[-1]:
        case 1:
            return go.Scatter(
                x=features[:, 0],
                y=np.zeros(features.shape[0]),
                mode="markers",
                marker=dict(
                    **marker_kwargs,
                ),
                **scatter_kwargs,
            )
        case 2:
            return go.Scatter(
                x=features[:, 0],
                y=features[:, 1],
                mode="markers",
                marker=dict(
                    **marker_kwargs,
                ),
                **scatter_kwargs,
            )
        case 3:
            return go.Scatter3d(
                x=features[:, 0],
                y=features[:, 1],
                z=features[:, 2],
                mode="markers",
                marker=dict(
                    **marker_kwargs,
                ),
                **scatter_kwargs,
            )
        case _:
            raise ValueError(f"Unknown dimensionality: {features.shape[-1]}")


def get_fig(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    method: EmbedMethods = EmbedMethods.PCA,
    captions: Optional[list[str]] = None,
):
    print("plot_fig")

    # n_colors = generate_colors(np.unique(y).shape[0])
    # n_colors = list(
    #     map(lambda x: f"rgb{tuple(map(lambda y: int(y * 255), x))}", n_colors)
    # )
    # color_map = {pid: n_colors[i] for i, pid in enumerate(np.unique(y))}
    # colors = [color_map[pid] for pid in y]

    m, low_dim = reduce_dimension(x, y, method=method)

    print("creating figure")
    fig = go.Figure(
        data=[
            get_scatter(low_dim),
        ]
    )

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
    )

    # fig.update_traces(
    #     hoverinfo="none",
    #     hovertemplate=None,
    # )

    if captions is not None:
        fig.update_traces(
            hoverinfo="text",
            # hovertemplate="%{text}",
            text=["<br>".join(wrap(caption, width=75)) for caption in captions],
        )

    fig.update_layout(height=800)

    return fig
