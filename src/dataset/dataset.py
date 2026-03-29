#!/usr/bin/env/python

import torch
import warnings
from src.paths import DATA_DIR


VALID_SETS = ['dblp', 'pokec_z', 'pokec_n', 'credit']


class GraphDataset:
    """
    A unified data loader for graph datasets supporting both DGL and PyTorch formats.

    This class standardizes access to node features, labels, sensitive attributes,
    and graph topology across several fairness-oriented benchmarks.
    It provides utilities for adjacency matrix normalization required by GCN-based architectures.

    Parameters
    ----------
    dataset : str
        Name of the dataset.
        Must be one of: {VALID_SETS}.
    dgl_data : bool, optional
        If True, loads data from a DGL `.bin` file.
        If False, expects a standard PyTorch `.pt` dictionary.
        Defaults to False.

    Attributes
    ----------
    x : torch.Tensor
        Node feature matrix of shape [num_nodes, num_features].
    y : torch.Tensor
        Ground truth labels for node classification.
    sensitive : torch.Tensor or None
        Sensitive demographic attributes used for fairness constraints.
    edge_index : torch.Tensor
        Graph connectivity in COO format [2, num_edges].
    train_idx, val_idx, test_idx : torch.Tensor
        Long tensors containing the indices for the respective data splits.
    """
    def __init__(self, dataset: str, dgl_data: bool = False):
        self.dataset = dataset.strip().lower()
        if self.dataset not in VALID_SETS:
            raise ValueError(f"dataset should be {', '.join(VALID_SETS)}")
        self.dgl_data = dgl_data

        if self.dgl_data:
            if self.dataset not in ['dblp', 'pokec_z', 'pokec_n']:
                raise ValueError("This dataset is only available as `.pt` please set dgl_data=False")
            try:
                import dgl
            except ImportError:
                raise ImportError("Could not import dgl, please install dgl or set `dgl_data=False`")

            graph_list, _ = dgl.load_graphs(DATA_DIR + "/" + self.dataset + ".bin")
            self._graph = graph_list[0]
            self._graph_pt = None

            # Graph attributes
            self.num_nodes = self._graph.num_nodes()
            self.edge_index = torch.stack(self._graph.edges(), dim=0)
            # Node attributes
            self.x = self._graph.ndata['feature']
            self.y = self._graph.ndata['label']
            self.sensitive = self._graph.ndata.get('sensitive', None)
            # Splits
            self.train_idx = torch.where(self._graph.ndata['train_index'])[0]
            self.val_idx = torch.where(self._graph.ndata['val_index'])[0]
            self.test_idx = torch.where(self._graph.ndata['test_index'])[0]
        else:
            self._graph = None
            self._graph_pt = torch.load(DATA_DIR + "/" + self.dataset + ".pt")

            # Graph attributes
            self.num_nodes = self._graph_pt['num_nodes']
            self.edge_index = self._graph_pt['edge_index']
            # Node attributes
            self.x = self._graph_pt['x']
            self.y = self._graph_pt['y']
            self.sensitive = self._graph_pt.get('sensitive', None)
            # Splits
            self.train_idx = self._graph_pt['split']['train_index']
            self.val_idx = self._graph_pt['split']['val_index']
            self.test_idx = self._graph_pt['split']['test_index']

        self.num_classes = int(len(self.y[self.test_idx].unique()))

    def _add_self_loops(self):
        """
        Adds self-loops $(i, i)$ to the current `edge_index`.

        This is a standard pre-processing step for GCNs to ensure that
        a node's own features are included in its neighborhood aggregation.
        """
        self_loops = torch.arange(self.num_nodes, dtype=torch.long)
        self_loop_edges = torch.stack([self_loops, self_loops], dim=0)
        self.edge_index = torch.cat([self.edge_index, self_loop_edges], dim=1)

    def sample(self, split: str, num_samples: int = 1024, replace: bool = False) -> torch.Tensor:
        """
        Randomly samples node indices from a specific split.

        Parameters
        ----------
        split : str
            The split to sample from ('train', 'val', or 'test').
        num_samples : int, optional
            Number of indices to return. Defaults to 1024.
        replace : bool, optional
            Whether to sample with replacement. Defaults to False.

        Returns
        -------
        torch.Tensor
            A tensor of sampled node indices.
        """
        idx = getattr(self, f'{split.lower()}_idx')
        if replace:
            return idx[torch.randint(0, len(idx), (num_samples,))]
        else:
            return idx[torch.randperm(len(idx))[:num_samples]]

    def summary(self):
        """Returns a dictionary of graph statistics."""
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.edge_index.shape[1],
            "num_features": self.x.shape[1],
            "num_classes": self.num_classes,
            "train_size": len(self.train_idx),
            "val_size": len(self.val_idx),
            "test_size": len(self.test_idx),
        }

    def get_adj_matrix(self, norm: str = 'both') -> torch.Tensor:
        """
        Computes a sparse normalized adjacency matrix $\hat{A}$.

        Supports various normalization schemes used in Graph Neural Networks:
        - 'none': Raw adjacency matrix $A$.
        - 'left': Random-walk normalization $D^{-1}A$.
        - 'right': $AD^{-1}$.
        - 'both': Symmetric normalization $D^{-1/2}AD^{-1/2}$ (Default for GCN).

        Parameters
        ----------
        norm : str, optional
            The normalization method.
            Defaults to 'both'.

        Returns
        -------
        torch.sparse_coo_tensor
            A sparse, coalesced tensor representing the normalized adjacency matrix.

        Raises
        ------
        ValueError
            If the graph contains isolated nodes with an in-degree of 0
            when normalization is required.
        """
        src, dst = self.edge_index

        # Calculate in- and out-degrees of each node
        in_degs = torch.zeros(self.num_nodes)
        in_degs.index_add_(0, dst, torch.ones(dst.shape[0]))
        out_degs = torch.zeros(self.num_nodes)
        out_degs.index_add_(0, src, torch.ones(src.shape[0]))

        # Ensure correct updates and correct adjacency matrix calculation
        if (in_degs == 0).any():
            raise ValueError("the adjacency matrix includes nodes with in-degrees of 0")
        out_degs.clamp_(min=1)

        # Calculate normalization coefficients
        if norm == 'left':
            left_norm = 1.0 / out_degs
            right_norm = torch.ones(self.num_nodes)
        elif norm == 'right':
            left_norm = torch.ones(self.num_nodes)
            right_norm = 1.0 / in_degs
        elif norm == 'both':
            left_norm = torch.pow(out_degs, -0.5)
            right_norm = torch.pow(in_degs, -0.5)
        else:
            if norm is not None and norm != 'none':
                warnings.warn("no valid normalization specified, adjacency matrix not normalized. "
                              "If intended set `norm=None`.", Warning)
            left_norm = torch.ones(self.num_nodes)
            right_norm = torch.ones(self.num_nodes)

        # For each edge, this will be 1/D for left or right and 1/sqrt(Di)sqrt(Dj) for both
        values = left_norm[src] * right_norm[dst]

        # Build adj matrix as sparse tensor
        assert self.edge_index.device != "mps", "sparse tensor support does not exist for MPS on this pytorch version"
        adj_matrix = torch.sparse_coo_tensor(indices=self.edge_index, values=values,
                                             size=(self.num_nodes, self.num_nodes))
        return adj_matrix.coalesce()

    def __len__(self):
        return self.num_nodes

    def __str__(self):
        s = self.summary()
        return (
            f"GraphDataset(\n"
            f"  num_nodes={s['num_nodes']},\n"
            f"  num_edges={s['num_edges']},\n"
            f"  num_features={s['num_features']},\n"
            f"  num_classes={s['num_classes']},\n"
            f"  train/val/test={s['train_size']}/{s['val_size']}/{s['test_size']}\n"
            f")"
        )

    __repr__ = __str__


# Update docstring dynamically
GraphDataset.__doc__ = GraphDataset.__doc__.replace(
    "{VALID_SETS}",
    ", ".join([f"'{s}'" for s in VALID_SETS])
)


if __name__ == '__main__':
    data = GraphDataset('credit', dgl_data=False)
    print(data)
