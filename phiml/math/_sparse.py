import warnings
from functools import partial
from numbers import Number
from typing import List, Callable, Tuple, Union

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix

from ._shape import Shape, non_batch, merge_shapes, instance, batch, non_instance, shape, channel, spatial, DimFilter, concat_shapes, EMPTY_SHAPE, dual, DUAL_DIM, SPATIAL_DIM, \
    non_channel, DEBUG_CHECKS
from ._magic_ops import concat, pack_dims, expand, rename_dims, stack, unpack_dim, unstack
from ._tensors import Tensor, TensorStack, NativeTensor, cached, wrap
from ..backend import choose_backend, NUMPY, Backend
from ..backend._dtype import DType


def sparse_tensor(indices: Tensor,
                  values: Tensor,
                  dense_shape: Shape,
                  can_contain_double_entries=True,
                  indices_sorted=False,
                  format='coo',
                  default: Number or None = 0) -> Tensor:
    """
    Construct a sparse tensor that stores `values` at the corresponding `indices` and is 0 everywhere else.
    In addition to the sparse dimensions indexed by `indices`, the tensor inherits all batch and channel dimensions from `values`.

    Args:
        indices: `Tensor` encoding the positions of stored values. It has the following dimensions:

            * One instance dimension exactly matching the instance dimension on `values`.
              It enumerates the positions of stored entries.
            * One channel dimension.
              Its item names must match the dimension names of `dense_shape` but the order can be arbitrary.
            * Any number of batch dimensions

        values: `Tensor` containing the stored values at positions given by `indices`. It has the following dimensions:

            * One instance dimension exactly matching the instance dimension on `indices`.
              It enumerates the values of stored entries.
            * Any number of channel dimensions if multiple values are stored at each index.
            * Any number of batch dimensions

        dense_shape: Dimensions listed in `indices`.
            The order can differ from the item names of `indices`.
        can_contain_double_entries: Whether some indices might occur more than once.
            If so, values at the same index will be summed.
        indices_sorted: Whether the indices are sorted in ascending order given the dimension order of the item names of `indices`.
        format: Sparse format in which to store the data, such as `'coo'` or `'csr'`. See `phiml.math.get_format`.
        default: Value the sparse tensor returns for non-stored values. Must be `0` or `None`.

    Returns:
        Sparse `Tensor` with the specified `format`.
    """
    assert default in [0, None], f"default value must be either 0 or None but got '{default}'"
    coo = SparseCoordinateTensor(indices, values, dense_shape, can_contain_double_entries, indices_sorted, default)
    return to_format(coo, format)


def tensor_like(existing_tensor: Tensor, values: Union[Tensor, Number, bool], value_order: str = None):
    """
    Creates a tensor with the same format and shape as `existing_tensor`.

    Args:
        existing_tensor: Any `Tensor`, sparse or dense.
        values: New values to replace the existing values by.
            If `existing_tensor` is sparse, `values` must have an instance dimension to list the stored values, matching the sparse indices.
        value_order: Order of `values` compared to `existing_tensor`.
            If `'original'`, the values are ordered like the values that was used to create the first tensor with this sparsity pattern.
            If `'as existing'`, the values match the current order of `existing_tensor`.
            Note that the order of values may be changed upon creating a sparse tensor.

    Returns:
        `Tensor`
    """
    assert value_order in ['original', 'as existing', None]
    if isinstance(existing_tensor, (SparseCoordinateTensor, CompressedSparseMatrix)):
        if value_order is None:
            assert not instance(values), f"When creating a sparse tensor from a list of values, value_order must be specified."
        if instance(values):
            values = rename_dims(values, instance, instance(existing_tensor._values))
        values = expand(values, instance(existing_tensor._values))
        if value_order == 'original' and isinstance(existing_tensor, CompressedSparseMatrix) and existing_tensor._uncompressed_indices_perm is not None:
            values = values[existing_tensor._uncompressed_indices_perm]
        if isinstance(existing_tensor, CompressedSparseMatrix) and existing_tensor._uncompressed_offset is not None:
            from ._ops import where
            values = where(existing_tensor._valid_mask(), values, 0)
        return existing_tensor._with_values(values)
    if not is_sparse(existing_tensor):
        return unpack_dim(values, instance, existing_tensor.shape.non_channel.non_batch)
    raise NotImplementedError


class SparseCoordinateTensor(Tensor):
    """
    indices: Tensor whose instance dimensions list the sparse entries and whose single channel dimension indexes the sparse dims. Can also have batch dimensions.
    values: Tensor with any of the instance dimensions of `indices` but no others. Can have any other non-instance dimensions to represent batched values.

    Batched-value matrices can be converted to explicit sparsity using sparsify_batch_dims().
    When reducing a batch dim (e.g. by summing over it), the dual dim remains, i.e. the input space keeps that dimension.
    """

    def __init__(self, indices: Tensor, values: Tensor, dense_shape: Shape, can_contain_double_entries: bool, indices_sorted: bool, default: Number):
        """
        Construct a sparse tensor with any number of sparse, dense and batch dimensions.
        """
        super().__init__()
        assert isinstance(indices, Tensor), f"indices must be a Tensor but got {type(indices)}"
        assert isinstance(values, Tensor), f"values must be a Tensor but got {type(values)}"
        assert instance(indices), f"indices must have an instance dimension but got {indices.shape}"
        assert channel(indices.shape).rank == 1, f"indices must have one channel dimension but got {indices.shape}"
        indices = rename_dims(indices, channel, 'sparse_idx')
        assert set(indices.sparse_idx.item_names) == set(dense_shape.names), f"The 'sparse_idx' dimension of indices must list the dense dimensions {dense_shape} as item names but got {indices.sparse_idx.item_names}"
        assert len(set(indices.sparse_idx.item_names)) == indices.sparse_idx.size, f"Duplicate sparse dimensions in indices {indices} with index {indices.sparse_idx.item_names}"
        assert indices.dtype.kind == int, f"indices must have dtype=int but got {indices.dtype}"
        assert instance(values) in instance(indices), f"All instance dimensions of values must exist in indices. values={values.shape}, indices={indices.shape}"
        assert set(indices.shape.only(instance(values))) == set(instance(values)), f"indices and values must have equal number of elements but got {instance(indices)} indices and {instance(values)} values"
        if not instance(values) and (spatial(values) or dual(values)):
            warnings.warn(f"You are creating a sparse tensor with only constant values {values.shape}. To have values vary along indices, add the corresponding instance dimension.", RuntimeWarning, stacklevel=3)
        self._shape = merge_shapes(dense_shape, batch(indices), non_instance(values))
        self._dense_shape = dense_shape
        self._indices = indices
        self._values = values
        self._can_contain_double_entries = can_contain_double_entries
        self._indices_sorted = indices_sorted
        self._default = default
        # if DEBUG_CHECKS:
        #     self.compress_rows()

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def dtype(self) -> DType:
        return self._values.dtype

    @property
    def sparse_dims(self):
        return self._dense_shape

    @property
    def sparsity_batch(self):
        return batch(self._indices)

    def native(self, order: Union[str, tuple, list, Shape] = None, singleton_for_const=False):
        assert order is None, f"sparse matrices are always ordered (primal, dual). For custom ordering, use math.dense(tensor).native() instead."
        return native_matrix(self, self.default_backend)

    @property
    def _is_tracer(self) -> bool:
        return self._indices._is_tracer or self._values._is_tracer

    def _with_values(self, new_values: Tensor):
        return SparseCoordinateTensor(self._indices, new_values, self._dense_shape, self._can_contain_double_entries, self._indices_sorted, self._default)

    def _natives(self) -> tuple:
        indices_const = self._indices.default_backend is not self._values.default_backend
        if indices_const:
            return self._values._natives()  # If we return NumPy arrays, they might get converted in function transformations
        else:
            return self._values._natives() + self._indices._natives()

    def _spec_dict(self) -> dict:
        indices_const = self._indices.default_backend is not self._values.default_backend
        return {'type': SparseCoordinateTensor,
                'shape': self._shape,
                'dense_shape': self._dense_shape,
                'indices': self._indices if indices_const else self._indices._spec_dict(),
                'values': self._values._spec_dict(),
                'can_contain_double_entries': self._can_contain_double_entries,
                'indices_sorted': self._indices_sorted,
                'default': self._default}

    @classmethod
    def _from_spec_and_natives(cls, spec: dict, natives: list):
        values = spec['values']['type']._from_spec_and_natives(spec['values'], natives)
        indices_or_spec = spec['indices']
        if isinstance(indices_or_spec, Tensor):
            indices = indices_or_spec
        else:
            indices = spec['indices']['type']._from_spec_and_natives(spec['indices'], natives)
        return SparseCoordinateTensor(indices, values, spec['dense_shape'], spec['can_contain_double_entries'], spec['indices_sorted'], spec['default'])

    def _native_coo_components(self, col_dims: DimFilter, matrix=False):
        col_dims = self._shape.only(col_dims)
        row_dims = self._dense_shape.without(col_dims)
        row_idx_packed, col_idx_packed = self._pack_indices(row_dims, col_dims)
        from . import reshaped_native
        ind_batch = batch(self._indices)
        channels = non_instance(self._values).without(ind_batch)
        if matrix:
            native_indices = choose_backend(row_idx_packed, col_idx_packed).stack([row_idx_packed, col_idx_packed], -1)
            native_shape = (row_dims.volume, col_dims.volume)
        else:
            native_indices = reshaped_native(self._indices, [ind_batch, instance, 'sparse_idx'])
            native_shape = self._dense_shape.sizes
        native_values = reshaped_native(self._values, [ind_batch, instance, channels])
        return ind_batch, channels, native_indices, native_values, native_shape

    def dual_indices(self, to_primal=False):
        """ Unpacked column indices """
        idx = self._indices[list(self._dense_shape.dual.names)]
        if to_primal:
            dual_names = self._dense_shape.dual.names
            primal_names = spatial(*dual_names).names
            idx = rename_dims(idx, 'sparse_idx', channel(sparse_idx=primal_names))
        return idx

    def primal_indices(self):
        """ Unpacked row indices """
        return self._indices[list(self._dense_shape.non_dual.names)]

    def _pack_indices(self, row_dims: Shape, col_dims: Shape):
        assert row_dims, f"Requires at least on row dim to pack"
        assert col_dims, f"Requires at least on col dim to pack"
        assert row_dims in self._dense_shape, f"Can only compress sparse dims but got {row_dims} which contains non-sparse dims"
        from ._ops import reshaped_native
        b = self._indices.default_backend
        row_idx = self._indices[row_dims.name_list]
        col_idx = self._indices[self._dense_shape.without(row_dims).name_list]
        # ToDo if not row_dims: idx = [0]
        row_idx_packed = b.ravel_multi_index(reshaped_native(row_idx, [batch, instance, channel]), row_dims.sizes)
        col_idx_packed = b.ravel_multi_index(reshaped_native(col_idx, [batch, instance, channel]), col_dims.sizes)
        return row_idx_packed, col_idx_packed

    def _unpack_indices(self, row_idx_packed, col_idx_packed, row_dims: Shape, col_dims: Shape, ind_batch: Shape):
        row_idx = np.stack(np.unravel_index(row_idx_packed, row_dims.sizes), -1)
        col_idx = np.stack(np.unravel_index(col_idx_packed, col_dims.sizes), -1)
        np_indices = np.concatenate([row_idx, col_idx], -1)
        from ._ops import reshaped_tensor
        idx_dim = channel(**{channel(self._indices).name: row_dims.names + col_dims.names})
        indices = reshaped_tensor(np_indices, [ind_batch, instance(self._indices), idx_dim], convert=False)
        return indices

    def compress_rows(self):
        return self.compress(self._dense_shape.non_dual)

    def compress_cols(self):
        return self.compress(self._dense_shape.dual)

    def compress(self, dims: DimFilter):
        c_dims = self._shape.only(dims, reorder=True)
        u_dims = self._dense_shape.without(c_dims)
        c_idx_packed, u_idx_packed = self._pack_indices(c_dims, u_dims)
        entries_dims = instance(self._indices)
        values = pack_dims(self._values, entries_dims, instance('sp_entries'))
        uncompressed_indices = pack_dims(self._indices, entries_dims, instance('sp_entries'))
        if self._can_contain_double_entries:
            bi = self._indices.default_backend
            assert c_idx_packed.shape[0] == 1, f"sparse compress() not supported for batched indices"
            lin_idx = bi.ravel_multi_index(bi.stack([c_idx_packed, u_idx_packed], -1)[0], (c_dims.volume, u_dims.volume))
            u_idx, u_ptr = bi.unique(lin_idx, return_inverse=True, return_counts=False, axis=-1)
            num_entries = u_idx.shape[-1]
            if num_entries < instance(values).volume:
                b = self.default_backend
                from ._ops import reshaped_native, reshaped_tensor
                if non_instance(values):
                    batched_values = reshaped_native(values, [non_instance, 'sp_entries'])
                    values_nat = b.batched_bincount(u_ptr[None, :], weights=batched_values, bins=num_entries)
                    values = wrap(values_nat, non_instance(values), instance('sp_entries'))
                else:
                    values = b.bincount(u_ptr, weights=values.native(), bins=num_entries)
                    values = reshaped_tensor(values, [instance('sp_entries')])
                idx_packed = bi.unravel_index(u_idx, (c_dims.volume, u_dims.volume))
                c_idx_packed = idx_packed[None, :, 0]
                u_idx_packed = idx_packed[None, :, 1]
                uncompressed_indices = bi.unravel_index(u_idx, c_dims.sizes + u_dims.sizes)
                uncompressed_indices = wrap(uncompressed_indices, instance('sp_entries'), channel(self._indices))
        # --- Use scipy.sparse.csr_matrix to reorder values ---
        c_idx_packed = choose_backend(c_idx_packed).numpy(c_idx_packed)
        u_idx_packed = choose_backend(u_idx_packed).numpy(u_idx_packed)
        idx = np.arange(1, c_idx_packed.shape[-1] + 1)  # start indexing at 1 since 0 might get removed
        scipy_csr = csr_matrix((idx, (c_idx_packed[0], u_idx_packed[0])), shape=(c_dims.volume, u_dims.volume))
        assert c_idx_packed.shape[1] == len(scipy_csr.data), "Failed to create CSR matrix because the CSR matrix contains fewer non-zero values than COO. This can happen when the `x` tensor is too small for the stencil."
        # --- Construct CompressedSparseMatrix ---
        entries_dim = instance(values).name
        perm = None
        if np.any(scipy_csr.data != idx):
            perm = {entries_dim: wrap(scipy_csr.data - 1, instance(entries_dim))}
            values = values[perm]  # Change order accordingly
        indices = wrap(scipy_csr.indices, instance(entries_dim))
        pointers = wrap(scipy_csr.indptr, instance('pointers'))
        return CompressedSparseMatrix(indices, pointers, values, u_dims, c_dims, self._default, uncompressed_indices=uncompressed_indices, uncompressed_indices_perm=perm)

    def __pack_dims__(self, dims: Tuple[str, ...], packed_dim: Shape, pos: Union[int, None], **kwargs) -> 'Tensor':
        dims = self._shape.only(dims)
        assert dims in self._dense_shape, f"Can only pack sparse dimensions on SparseCoordinateTensor but got {dims} of which {dims.without(self._dense_shape)} are not sparse"
        assert self._indices.default_backend is NUMPY, "Can only pack NumPy indices as of yet"
        from ._ops import reshaped_native, reshaped_tensor
        inst_dim_order = instance(self._indices)
        indices = pack_dims(self._indices, inst_dim_order, instance('sp_entries'))
        values = pack_dims(self._values, inst_dim_order, instance('sp_entries'))
        idx_to_pack = indices.sparse_idx[dims.names]
        idx_packed = np.ravel_multi_index(reshaped_native(idx_to_pack, [channel, instance(idx_to_pack)]), dims.sizes)
        idx_packed = expand(reshaped_tensor(idx_packed, [instance('sp_entries')]), channel(sparse_idx=packed_dim.name))
        indices = concat([indices.sparse_idx[list(self._dense_shape.without(dims).names)], idx_packed], 'sparse_idx')
        dense_shape = concat_shapes(self._dense_shape.without(dims), packed_dim.with_size(dims.volume))
        idx_sorted = self._indices_sorted and False  # ToDo still sorted if dims are ordered correctly and no other dim in between and inserted at right point
        return SparseCoordinateTensor(indices, values, dense_shape, self._can_contain_double_entries, idx_sorted, self._default)

    def _with_shape_replaced(self, new_shape: Shape):
        assert self._shape.rank == new_shape.rank
        dense_shape = new_shape[self._shape.indices(self._dense_shape)]
        new_item_names = new_shape[self._shape.indices(self._indices.shape.get_item_names('sparse_idx'))].names
        values = self._values._with_shape_replaced(self._values.shape.replace(self._shape, new_shape))
        non_vec = self._shape.without('sparse_idx')
        new_non_vec = new_shape[self._shape.indices(non_vec)]
        indices = self._indices._with_shape_replaced(self._indices.shape.replace(non_vec, new_non_vec).with_dim_size('sparse_idx', new_item_names))
        return SparseCoordinateTensor(indices, values, dense_shape, self._can_contain_double_entries, self._indices_sorted, self._default)

    def _op1(self, native_function):
        return self._with_values(self._values._op1(native_function))

    def _op2(self, other, operator: Callable, native_function: Callable, op_name: str = 'unknown', op_symbol: str = '?') -> 'Tensor':
        other_shape = shape(other)
        affects_only_values = self._dense_shape.isdisjoint(other_shape)
        if affects_only_values:
            return self._with_values(operator(self._values, other))
        if isinstance(other, CompressedSparseMatrix):
            other = other.decompress()
        if isinstance(other, SparseCoordinateTensor):
            if same_sparsity_pattern(self, other):
                return self._with_values(operator(self._values, other._values))
            else:
                assert op_name in ['add', 'radd', 'sub', 'rsub']
                all_sparse_dims = sparse_dims(other) & sparse_dims(self)
                self_indices = pack_dims(self._indices, instance, instance('sp_entries'))
                other_indices = pack_dims(other._indices, instance, instance('sp_entries'))
                self_values = pack_dims(self._values, instance(self._indices), instance('sp_entries'))
                other_values = pack_dims(other._values, instance(other._indices), instance('sp_entries'))
                self_indices, self_values = with_sparsified_dim(self_indices, self_values, all_sparse_dims)
                other_indices, other_values = with_sparsified_dim(other_indices, other_values, all_sparse_dims)
                indices = concat([self_indices, other_indices], 'sp_entries')
                if op_symbol == '+':
                    values = concat([self_values, other_values], instance(self_values), expand_values=True)
                elif op_name == 'sub':
                    values = concat([self_values, -other_values], instance(self_values), expand_values=True)
                else:  # op_name == 'rsub':
                    values = concat([-self_values, other_values], instance(self_values), expand_values=True)
                return SparseCoordinateTensor(indices, values, self._dense_shape & other._dense_shape, can_contain_double_entries=True, indices_sorted=False, default=self._default)
        else:  # other is dense
            if self._dense_shape in other.shape:  # all dims dense -> convert to dense
                return dense(self)._op2(other, operator, native_function, op_name, op_symbol)
            else:  # only some dims dense -> stay sparse
                dense_dims = self._dense_shape.only(other.shape)
                other_values = other[self._indices.sparse_idx[dense_dims.name_list]]
                values = operator(self._values, other_values)
                return self._with_values(values)

    def _getitem(self, selection: dict) -> 'Tensor':
        batch_selection = {dim: selection[dim] for dim in self._shape.only(tuple(selection)).names}
        indices = self._indices[{dim: sel for dim, sel in batch_selection.items() if dim != 'sparse_idx'}]
        values = self._values[batch_selection]
        if self._dense_shape.only(tuple(selection)):
            keep = expand(True, instance(self._indices))
            for dim, sel in selection.items():
                dim_indices = self._indices[dim]
                if isinstance(sel, int):
                    item_names = list(channel(indices).item_names[0])
                    item_names.remove(dim)
                    indices = indices[item_names]
                    sel = slice(sel, sel + 1)
                elif isinstance(sel, str):
                    raise NotImplementedError
                assert isinstance(sel, slice)
                assert sel.step in (None, 1), f"Only step size 1 supported for sparse indexing but got {sel.step}"
                start = sel.start or 0
                stop = self._dense_shape[dim].size if sel.stop is None else sel.stop
                keep &= (start <= dim_indices) & (dim_indices < stop)
                from . import vec
                indices -= vec('sparse_idx', **{d: start if d == dim else 0 for d in indices.sparse_idx.item_names})
            from ._ops import boolean_mask
            indices = boolean_mask(indices, instance(indices), keep)
            values = boolean_mask(values, instance(indices), keep)
            dense_shape = self._dense_shape.after_gather(selection)
            return SparseCoordinateTensor(indices, values, dense_shape, self._can_contain_double_entries, self._indices_sorted, self._default)
        else:
            return SparseCoordinateTensor(indices, values, self._dense_shape, self._can_contain_double_entries, self._indices_sorted, self._default)

    def __concat__(self, tensors: tuple, dim: str, **kwargs) -> 'SparseCoordinateTensor':
        if not all(isinstance(t, SparseCoordinateTensor) for t in tensors):
            return NotImplemented
        if dim in self._dense_shape:
            # assert all default equal
            from . import vec
            indices = []
            values = []
            offset = 0
            for t in tensors:
                t_indices = stored_indices(t, list_dim=instance(self._indices), index_dim=channel(self._indices))
                t_values = stored_values(t, list_dim=instance(self._values))
                t_indices += vec('sparse_idx', **{d: offset if d == dim else 0 for d in t_indices.sparse_idx.item_names})
                offset += t.shape.get_size(dim)
                indices.append(t_indices)
                values.append(t_values)
            indices = concat(indices, instance(self._indices))
            values = concat(values, instance(self._values))
            dense_shape = self._dense_shape.with_dim_size(dim, sum([t.shape.get_size(dim) for t in tensors]))
            can_contain_double_entries = any([t._can_contain_double_entries for t in tensors])
            indices_sorted = all([t._indices_sorted for t in tensors])
            return SparseCoordinateTensor(indices, values, dense_shape, can_contain_double_entries, indices_sorted, self._default)
        else:
            raise NotImplementedError("concatenating compressed sparse tensors along non-sparse dims not yet supported")


class CompressedSparseMatrix(Tensor):

    def __init__(self,
                 indices: Tensor,
                 pointers: Tensor,
                 values: Tensor,
                 uncompressed_dims: Shape,
                 compressed_dims: Shape,
                 default: Number,
                 uncompressed_offset: int = None,
                 uncompressed_indices: Tensor = None,
                 uncompressed_indices_perm: Tensor = None):
        """

        Args:
            indices: indices must be sorted in ascending order by compressed_dim and other sparse dims.
                Must have one or multiple instance dimensions and can have any number of batch dimensions.
                No spatial and channel dimensions allowed.
            pointers:
            values:
            compressed_dims: Sparse dimensions with compressed pointer representation.
                Only one pointer array is used per matrix, i.e. the dimensions are packed internally.
                These dimensions are indexed by `pointers`.
            uncompressed_dims: Sparse dimensions with full index storage.
                These dimensions are indexed by `indices`.
            uncompressed_offset: For sliced sparse tensors.
                If `None`, indicates that all entries lie within bounds.
                If an `int`, indicate that this is a slice of a larger compressed sparse matrix.
                Indices actually refer to `indices - uncompressed_offset` within this matrix, i.e. they may reference phantom values to the left or right of the matrix.
                The `values` corresponding to phantom entries must all be 0.
                The size of the slice is given by `compressed_dims.volume`.
        """
        super().__init__()
        assert instance(indices), "indices must have an instance dimension"
        assert instance(pointers), "pointers must have an instance dimension"
        assert instance(values) == instance(indices), "Instance dimensions of values and indices must match exactly"
        assert not channel(indices) and not spatial(indices), f"channel and spatial dimensions not allowed on indices but got {shape(indices)}"
        assert not channel(pointers) and not spatial(pointers), f"channel and spatial dimensions not allowed on pointers but got {shape(pointers)}"
        assert uncompressed_dims.isdisjoint(compressed_dims), f"Dimensions cannot be compressed and uncompressed at the same time but got compressed={compressed_dims}, uncompressed={uncompressed_dims}"
        assert instance(pointers).size == compressed_dims.volume + 1
        if uncompressed_indices is not None:
            assert instance(uncompressed_indices) == instance(indices), f"Number of uncompressed indices {instance(uncompressed_offset)} does not match compressed indices {instance(indices)}"
        self._shape = merge_shapes(compressed_dims, uncompressed_dims, batch(indices), batch(pointers), non_instance(values))
        self._indices = indices
        self._pointers = rename_dims(pointers, instance, 'pointers')
        self._values = values
        self._uncompressed_dims = uncompressed_dims
        self._compressed_dims = compressed_dims
        self._default = default
        self._uncompressed_offset = uncompressed_offset
        self._uncompressed_indices = uncompressed_indices
        self._uncompressed_indices_perm = uncompressed_indices_perm

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def sparse_dims(self):
        return self._compressed_dims & self._uncompressed_dims

    @property
    def sparsity_batch(self):
        return batch(self._indices) & batch(self._pointers)

    @property
    def dtype(self) -> DType:
        return self._values.dtype

    @property
    def _is_tracer(self) -> bool:
        return self._values._is_tracer or self._indices._is_tracer or self._pointers._is_tracer

    def _natives(self) -> tuple:
        indices_const = self._indices.default_backend is not self._values.default_backend
        pointers_const = self._pointers.default_backend is not self._values.default_backend
        result = self._values._natives()
        if not indices_const:
            result += self._indices._natives()
        if not pointers_const:
            result += self._pointers._natives()
        return result

    def _spec_dict(self) -> dict:
        indices_const = self._indices.default_backend is not self._values.default_backend
        pointers_const = self._pointers.default_backend is not self._values.default_backend
        return {'type': CompressedSparseMatrix,
                'shape': self._shape,
                'values': self._values._spec_dict(),
                'indices': self._indices if indices_const else self._indices._spec_dict(),
                'pointers': self._pointers if pointers_const else self._pointers._spec_dict(),
                'uncompressed_dims': self._uncompressed_dims,
                'compressed_dims': self._compressed_dims,
                'uncompressed_offset': self._uncompressed_offset,
                'uncompressed_indices': self._uncompressed_indices,
                'uncompressed_indices_perm': self._uncompressed_indices_perm,
                'default': self._default,
                }

    @classmethod
    def _from_spec_and_natives(cls, spec: dict, natives: list):
        values = spec['values']['type']._from_spec_and_natives(spec['values'], natives)
        indices_or_spec = spec['indices']
        if isinstance(indices_or_spec, Tensor):
            indices = indices_or_spec
        else:
            indices = spec['indices']['type']._from_spec_and_natives(spec['indices'], natives)
        pointers_or_spec = spec['pointers']
        if isinstance(pointers_or_spec, Tensor):
            pointers = pointers_or_spec
        else:
            pointers = spec['pointers']['type']._from_spec_and_natives(spec['pointers'], natives)
        return CompressedSparseMatrix(indices, pointers, values, spec['uncompressed_dims'], spec['compressed_dims'], spec['default'], spec['uncompressed_offset'], spec['uncompressed_indices'], spec['uncompressed_indices_perm'])

    def _getitem(self, selection: dict) -> 'Tensor':
        batch_selection = {dim: selection[dim] for dim in self._shape.only(tuple(selection)).names}
        indices = self._indices[batch_selection]
        pointers = self._pointers[batch_selection]
        values = self._values[batch_selection]
        uncompressed = self._uncompressed_dims
        compressed = self._compressed_dims
        uncompressed_offset = self._uncompressed_offset
        if compressed.only(tuple(selection)):
            if compressed.rank > 1:
                raise NotImplementedError
            ptr_sel = selection[compressed.name]
            if isinstance(ptr_sel, int):
                raise NotImplementedError(f"Slicing with int not yet supported for sparse tensors. Use a range instead, e.g. [{ptr_sel}:{ptr_sel+1}] instead of [{ptr_sel}]")
            elif isinstance(ptr_sel, slice):
                assert ptr_sel.step in (None, 1), f"Only step size 1 supported for sparse indexing but got {ptr_sel.step}"
                if batch(indices):
                    raise NotImplementedError("Slicing not yet supported for batched sparse tensors")
                start = ptr_sel.start or 0
                stop = compressed.volume if ptr_sel.stop is None else ptr_sel.stop
                pointers = pointers[start:stop+1]
                indices = indices[{instance(indices).name: slice(int(pointers[0]), int(pointers[-1]))}]
                values = values[{instance(values).name: slice(int(pointers[0]), int(pointers[-1]))}]
                pointers -= pointers[0]
                compressed = compressed.after_gather({compressed.name: ptr_sel})
            else:
                raise NotImplementedError
        if uncompressed.only(tuple(selection)):
            if self._uncompressed_dims.rank > 1:
                raise NotImplementedError
            ind_sel = selection[uncompressed.name]
            if isinstance(ind_sel, int):
                raise NotImplementedError(f"Slicing with int not yet supported for sparse tensors. Use a range instead, e.g. [{ind_sel}:{ind_sel+1}] instead of [{ind_sel}]")
            elif isinstance(ind_sel, slice):
                assert ind_sel.step in (None, 1), f"Only step size 1 supported for sparse indexing but got {ind_sel.step}"
                start = ind_sel.start or 0
                stop = uncompressed.volume if ind_sel.stop is None else ind_sel.stop
                keep = (start <= self._indices) & (self._indices < stop)
                from ._ops import where
                values = where(keep, values, 0)
                uncompressed_offset = start
                uncompressed = uncompressed.after_gather({uncompressed.name: ind_sel})
            else:
                raise NotImplementedError
        return CompressedSparseMatrix(indices, pointers, values, uncompressed, compressed, self._default, uncompressed_offset)

    def __concat__(self, tensors: tuple, dim: str, **kwargs) -> 'CompressedSparseMatrix':
        if not all(isinstance(t, CompressedSparseMatrix) for t in tensors):
            return NotImplemented
        if dim == self._compressed_dims[0].name:
            indices = concat([t._indices for t in tensors], instance(self._indices), **kwargs)
            values = concat([t._values for t in tensors], instance(self._values), **kwargs)
            pointers = []
            pointer_offset = 0
            for i, t in enumerate(tensors):
                pointers.append((t._pointers[1:] if i else t._pointers) + pointer_offset)
                pointer_offset += t._pointers[-1]
            assert pointer_offset == instance(indices).volume
            pointers = concat(pointers, instance(self._pointers))
            compressed = self._compressed_dims.with_dim_size(dim, sum(t.shape.get_size(dim) for t in tensors))
            return CompressedSparseMatrix(indices, pointers, values, self._uncompressed_dims, compressed, self._default, self._uncompressed_offset)
        elif dim == self._uncompressed_dims[0].name:
            if all([same_sparsity_pattern(self, t) for t in tensors]):
                # ToDo test if offsets match and ordered correctly
                from ._ops import sum_
                values = sum_([t._values for t in tensors], '0')
                uncompressed = self._uncompressed_dims.with_dim_size(dim, sum(t.shape.get_size(dim) for t in tensors))
                return CompressedSparseMatrix(self._indices, self._pointers, values, uncompressed, self._compressed_dims, self._default, uncompressed_offset=None)
            else:
                raise NotImplementedError("concatenating arbitrary compressed sparse tensors along uncompressed dim is not yet supported")
        else:
            raise NotImplementedError("concatenating compressed sparse tensors along non-sparse dims not yet supported")

    def _op1(self, native_function):
        return self._with_values(self._values._op1(native_function))

    def _op2(self, other, operator: Callable, native_function: Callable, op_name: str = 'unknown', op_symbol: str = '?') -> 'Tensor':
        other_shape = shape(other)
        affects_only_values = self.sparse_dims.isdisjoint(other_shape) and non_instance(self._indices).isdisjoint(other_shape)
        if affects_only_values:
            return self._with_values(operator(self._values, other))
        elif isinstance(other, CompressedSparseMatrix):
            if same_sparsity_pattern(self, other):
                result = operator(self._values, other._values)
                if self._uncompressed_offset is not None:
                    from ._ops import where
                    result = where(self._valid_mask(), result, 0)
                return self._with_values(result)
            elif op_symbol == '+':
                raise NotImplementedError("Compressed addition not yet implemented")
            else:
                # convert to COO, then perform operation
                raise NotImplementedError
        elif self._uncompressed_dims in other_shape and self._compressed_dims.isdisjoint(other_shape):
            from ._ops import gather, boolean_mask, clip, where
            if self._uncompressed_offset is None:
                other_values = gather(other, self._indices, self._uncompressed_dims)
                return self._with_values(operator(self._values, other_values))
            # if bake_slice:
            #     baked = self._bake_slice()
            #     other_values = gather(other, baked._indices, self._uncompressed_dims)
            #     return baked._with_values(operator(baked._values, other_values))
            indices = clip(self._indices - self._uncompressed_offset, 0, self._uncompressed_dims.volume - 1)
            other_values = gather(other, indices, self._uncompressed_dims)
            return self._with_values(where(self._valid_mask(), operator(self._values, other_values), 0))
        elif self._compressed_dims in other_shape and self._uncompressed_dims.isdisjoint(other_shape):
            from ._ops import gather, boolean_mask, clip, where
            row_indices, _ = self._coo_indices('clamp')
            other_values = gather(other, row_indices, self._compressed_dims)
            result_values = operator(self._values, other_values)
            if self._uncompressed_offset is not None:
                result_values = where(self._valid_mask(), result_values, 0)
            return self._with_values(result_values)
        else:
            raise NotImplementedError

    def _with_values(self, new_values: Tensor):
        return CompressedSparseMatrix(self._indices, self._pointers, new_values, self._uncompressed_dims, self._compressed_dims, self._default, self._uncompressed_offset, self._uncompressed_indices, self._uncompressed_indices_perm)

    def _with_shape_replaced(self, new_shape: Shape):
        assert self._shape.rank == new_shape.rank
        values = self._values._with_shape_replaced(self._values.shape.replace(self._shape, new_shape))
        indices = self._indices._with_shape_replaced(self._indices.shape.replace(self._shape, new_shape))
        pointers = self._pointers._with_shape_replaced(self._pointers.shape.replace(self._shape, new_shape))
        uncompressed_indices = self._uncompressed_indices._with_shape_replaced(self._uncompressed_indices.shape.replace(self._shape, new_shape, replace_item_names=channel)) if self._uncompressed_indices is not None else None
        return CompressedSparseMatrix(indices, pointers, values, self._uncompressed_dims.replace(self._shape, new_shape), self._compressed_dims.replace(self._shape, new_shape), self._default, self._uncompressed_offset, uncompressed_indices, self._uncompressed_indices_perm)

    def _native_csr_components(self, invalid='clamp', get_values=True):
        assert invalid in ['clamp', 'discard', 'keep']
        from ._ops import reshaped_native
        ind_batch = batch(self._indices) & batch(self._pointers)
        channels = non_instance(self._values).without(ind_batch)
        native_indices = reshaped_native(self._indices, [ind_batch, instance])
        native_pointers = reshaped_native(self._pointers, [ind_batch, instance])
        native_values = reshaped_native(self._values, [ind_batch, instance, channels]) if get_values else None
        native_shape = self._compressed_dims.volume, self._uncompressed_dims.volume
        if self._uncompressed_offset is not None:
            native_indices -= self._uncompressed_offset
            if invalid == 'clamp':
                native_indices = choose_backend(native_indices).clip(native_indices, 0, self._uncompressed_dims.volume - 1)
            elif invalid == 'discard':
                assert ind_batch.volume == 1, f"Variable number of indices not supported, batch shape = {ind_batch}"
                b = choose_backend(native_indices, native_pointers)
                in_range = (0 <= native_indices) & (native_indices < self._uncompressed_dims.volume)
                native_indices = b.boolean_mask(native_indices, in_range[0], 1)
                native_values = choose_backend(native_values).boolean_mask(native_values, in_range[0], 1) if get_values else None
                removed = b.cumsum(~in_range, 1)
                removed = b.batched_gather_1d(removed, native_pointers[:, 1:]-1)
                removed = b.concat([b.zeros((b.staticshape(removed)[0], 1), b.dtype(removed)), removed], 1)
                native_pointers -= removed
        return ind_batch, channels, native_indices, native_pointers, native_values, native_shape

    def _bake_slice(self) -> 'CompressedSparseMatrix':
        from ._ops import boolean_mask, cumulative_sum, pad
        valid = (self._uncompressed_offset <= self._indices) & (self._indices < self._uncompressed_offset + self._uncompressed_dims.volume)
        indices = boolean_mask(self._indices, instance(self._indices), valid)
        values = boolean_mask(self._values, instance(self._values), valid)
        removed = cumulative_sum(~valid, instance(valid))
        removed = removed[self._pointers.pointers[1:] - 1]
        removed = pad(removed, {'pointers': (1, 0)}, 1)
        pointers = self._pointers - removed
        return CompressedSparseMatrix(indices, pointers, values, self._uncompressed_dims, self._compressed_dims, self._default)

    def _valid_mask(self):
        return (self._uncompressed_offset <= self._indices) & (self._indices < self._uncompressed_offset + self._uncompressed_dims.volume)

    def _coo_indices(self, invalid='clamp', stack_dim: Shape = None):
        ind_batch, channels, native_indices, native_pointers, _, native_shape = self._native_csr_components(invalid, get_values=False)
        native_indices = choose_backend(native_indices, native_pointers).csr_to_coo(native_indices, native_pointers)
        from ._ops import reshaped_tensor
        if stack_dim is not None:
            item_names = self._compressed_dims.name, self._uncompressed_dims.name
            indices = reshaped_tensor(native_indices, [ind_batch, instance(self._indices), stack_dim.with_size(item_names)], convert=False)
            return indices
        else:
            rows = reshaped_tensor(native_indices[..., 0], [ind_batch, instance(self._indices)], convert=False)
            cols = reshaped_tensor(native_indices[..., 1], [ind_batch, instance(self._indices)], convert=False)
            return rows, cols

    def decompress(self):
        if self._uncompressed_indices is None:
            ind_batch, channels, native_indices, native_pointers, native_values, native_shape = self._native_csr_components(invalid='discard')
            native_indices = choose_backend(native_indices, native_pointers).csr_to_coo(native_indices, native_pointers)
            from ._ops import reshaped_tensor
            if self._compressed_dims.rank == self._uncompressed_dims.rank == 1:
                indices = reshaped_tensor(native_indices, [ind_batch, instance(self._indices), channel(sparse_idx=(self._compressed_dims.name, self._uncompressed_dims.name))], convert=False)
                values = reshaped_tensor(native_values, [ind_batch, instance(self._values), channel(self._values)])
            else:
                raise NotImplementedError()
            return SparseCoordinateTensor(indices, values, concat_shapes(self._compressed_dims, self._uncompressed_dims), False, True, self._default)
        if self._uncompressed_indices_perm is not None:
            self._uncompressed_indices = self._uncompressed_indices[self._uncompressed_indices_perm]
            self._uncompressed_indices_perm = None
        return SparseCoordinateTensor(self._uncompressed_indices, self._values, self._compressed_dims & self._uncompressed_dims, False, False, self._default)

    def native(self, order: Union[str, tuple, list, Shape] = None, singleton_for_const=False):
        assert order is None, f"sparse matrices are always ordered (primal, dual). For custom ordering, use math.dense(tensor).native() instead."
        return native_matrix(self, self.default_backend)

    def __pack_dims__(self, dims: Tuple[str, ...], packed_dim: Shape, pos: Union[int, None], **kwargs) -> 'Tensor':
        assert all(d in self._shape for d in dims)
        dims = self._shape.only(dims, reorder=True)
        if dims.only(self._compressed_dims).is_empty:  # pack cols
            assert self._uncompressed_dims.are_adjacent(dims), f"Can only compress adjacent dims but got {dims} for matrix {self._shape}"
            uncompressed_dims = self._uncompressed_dims.replace(dims, packed_dim.with_size(dims.volume))
            return CompressedSparseMatrix(self._indices, self._pointers, self._values, uncompressed_dims, self._compressed_dims, self._default, self._uncompressed_offset)
        elif dims.only(self._uncompressed_dims).is_empty:   # pack rows
            assert self._compressed_dims.are_adjacent(dims), f"Can only compress adjacent dims but got {dims} for matrix {self._shape}"
            compressed_dims = self._compressed_dims.replace(dims, packed_dim.with_size(dims.volume))
            return CompressedSparseMatrix(self._indices, self._pointers, self._values, self._uncompressed_dims, compressed_dims, self._default, self._uncompressed_offset)
        else:
            raise NotImplementedError(f"Cannot pack dimensions from both columns and rows with compressed sparse matrices but got {dims}")


def get_format(x: Tensor) -> str:
    """
    Returns the sparse storage format of a tensor.

    Args:
        x: `Tensor`

    Returns:
        One of `'coo'`, `'csr'`, `'csc'`, `'dense'`.
    """
    if isinstance(x, SparseCoordinateTensor):
        return 'coo'
    elif isinstance(x, CompressedSparseMatrix):
        if dual(x._uncompressed_dims):
            return 'csr'
        else:
            assert not dual(x._uncompressed_dims), f"Compressed matrix {x.shape} does not match 'csr' or 'csc' because dual dimensions are present in rows and columns."
            return 'csc'
    elif isinstance(x, TensorStack):
        formats = [get_format(t) for t in x._tensors]
        if all(f == formats[0] for f in formats):
            return formats[0]
        return 'mixed'
    elif isinstance(x, Tensor):
        return 'dense'
    b = choose_backend(x)
    if not b.is_sparse(x):
        return 'dense'
    return b.get_sparse_format(x)


def is_sparse(x: Tensor):
    """
    Checks whether a tensor is represented in COO, CSR or CSC format.
    If the tensor is neither sparse nor dense, this function raises an error.

    Args:
        x: `Tensor` to test.

    Returns:
        `True` if `x` is sparse, `False` if `x` is dense.

    Raises:
        `AssertionError` if `x` is neither sparse nor fully dense.
    """
    f = get_format(x)
    if f == 'dense':
        return False
    if f in ['csr', 'csc', 'coo']:
        return True
    raise AssertionError(f"Tensor {x} is neither sparse nor dense")


def to_format(x: Tensor, format: str):
    """
    Converts a `Tensor` to the specified sparse format or to a dense tensor.

    Args:
        x: Sparse or dense `Tensor`
        format: Target format. One of `'dense'`, `'coo'`, `'csr'`, or `'csc'`.

    Returns:
        `Tensor` of the specified format.
    """
    assert format in ('coo', 'csr', 'csc', 'dense'), f"Invalid format: '{format}'. Must be one of 'coo', 'csr', 'csc', 'dense'"
    if get_format(x) == format:
        return x
    if format == 'dense':
        return dense(x)
    if isinstance(x, SparseCoordinateTensor):
        if format == 'csr':
            return x.compress_rows()
        elif format == 'csc':
            return x.compress_cols()
    elif isinstance(x, CompressedSparseMatrix):
        if format == 'coo':
            return x.decompress()
        else:
            return to_format(x.decompress(), format)
    else:  # dense to sparse
        from ._ops import nonzero
        indices = nonzero(rename_dims(x, channel, instance))
        values = x[indices]
        coo = SparseCoordinateTensor(indices, values, x.shape, can_contain_double_entries=False, indices_sorted=False, default=0)
        return to_format(coo, format)


def sparse_dims(x: Tensor) -> Shape:
    """
    Returns the dimensions of a `Tensor` that are explicitly stored in a sparse format.

    Args:
        x: Any `Tensor`

    Returns:
        `Shape`
    """
    if isinstance(x, SparseCoordinateTensor):
        return x._dense_shape
    elif isinstance(x, CompressedSparseMatrix):
        return concat_shapes(x._compressed_dims, x._uncompressed_dims)
    else:
        return EMPTY_SHAPE


def get_sparsity(x: Tensor):
    """
    Fraction of values currently stored on disk for the given `Tensor` `x`.
    For sparse tensors, this is `nnz / shape`.

    This is a lower limit on the number of values that will need to be processed for operations involving `x`.
    The actual number is often higher since many operations require data be laid out in a certain format.
    In these cases, missing values, such as zeros, are filled in before the operation.

    The following operations may return tensors whose values are only partially stored:

    * `phiml.math.expand()`
    * `phiml.math.pairwise_distance()` with `max_distance` set.
    * Tracers used in `phiml.math.jit_compile_linear()`
    * Stacking any of the above.

    Args:
        x: `Tensor`

    Returns:
        The number of values that are actually stored on disk.
        This does not include additional information, such as position information / indices.
        For sparse matrices, this is equal to the number of nonzero values.
    """
    return stored_values(x, invalid='keep').shape.volume / x.shape.volume


def stored_values(x: Tensor, list_dim=instance('entries'), invalid='discard') -> Tensor:
    """
    Returns the stored values for a given `Tensor``.
    For sparse tensors, this will return only the stored entries.
    For collapsed tensors, only the stored dimensions will be returned.
    Dense tensors are returned as-is.

    Args:
        x: `Tensor`
        list_dim: Dimension along which stored values should be laid out.
        invalid: One of `'discard'`, `'clamp'`, `'keep'` Filter result by valid indices.
            Internally, invalid indices may be stored for performance reasons.

    Returns:
        `Tensor` representing all values stored to represent `x`.
    """
    assert invalid in ['discard', 'clamp', 'keep'], f"invalid handling must be one of 'discard', 'clamp', 'keep' but got {invalid}"
    if isinstance(x, NativeTensor):
        return expand(NativeTensor(x._native, x._native_shape, x._native_shape), list_dim.with_size(1))
    if isinstance(x, TensorStack):
        if x.is_cached:
            return stored_values(cached(x))
        return stack([stored_values(t, list_dim) for t in x._tensors], x._stack_dim)
    elif isinstance(x, CompressedSparseMatrix):
        if invalid in ['keep', 'clamp']:
            return rename_dims(x._values, instance, list_dim)
        else:
            x = x.decompress()  # or apply slices, then return values
    if isinstance(x, SparseCoordinateTensor):
        if x._can_contain_double_entries:
            warnings.warn(f"stored_values of sparse tensor {x.shape} may contain multiple values for the same position.")
        return rename_dims(x._values, instance, list_dim)
    raise ValueError(x)


def stored_indices(x: Tensor, list_dim=instance('entries'), index_dim=channel('index'), invalid='discard') -> Tensor:
    """
    Returns the indices of the stored values for a given `Tensor``.
    For sparse tensors, this will return the stored indices tensor.
    For collapsed tensors, only the stored dimensions will be returned.

    Args:
        x: `Tensor`
        list_dim: Dimension along which stored indices should be laid out.
        invalid: One of `'discard'`, `'clamp'`, `'keep'` Filter result by valid indices.
            Internally, invalid indices may be stored for performance reasons.

    Returns:
        `Tensor` representing all indices of stored values.
    """
    assert invalid in ['discard', 'clamp', 'keep'], f"invalid handling must be one of 'discard', 'clamp', 'keep' but got {invalid}"
    if isinstance(x, NativeTensor):
        from ._ops import meshgrid
        if batch(x):
            raise NotImplementedError
        indices = meshgrid(x._native_shape.non_batch.non_channel)
        return pack_dims(indices, non_channel, list_dim)
    if isinstance(x, TensorStack):
        if x.is_cached or not x.requires_broadcast:
            return stored_indices(cached(x))
        raise NotImplementedError
        return stack([stored_indices(t, list_dim) for t in x._tensors], x._stack_dim)  # ToDo add index for stack dim
    elif isinstance(x, CompressedSparseMatrix):
        return rename_dims(x._coo_indices(invalid, stack_dim=index_dim), instance, list_dim)
    if isinstance(x, SparseCoordinateTensor):
        if x._can_contain_double_entries:
            warnings.warn(f"stored_values of sparse tensor {x.shape} may contain multiple values for the same position.")
        new_index_dim = index_dim.with_size(channel(x._indices).item_names[0])
        return rename_dims(x._indices, [instance(x._indices).name, channel(x._indices).name], [list_dim, new_index_dim])
    raise ValueError(x)


def same_sparsity_pattern(t1: Tensor, t2: Tensor, allow_const=False):
    if isinstance(t1, TensorStack):
        raise NotImplementedError
    if isinstance(t2, TensorStack):
        raise NotImplementedError
    from ._ops import close
    if allow_const:
        if is_sparse(t1) and not is_sparse(t2) and sparse_dims(t1) not in t2.shape:
            return True
        if is_sparse(t2) and not is_sparse(t1) and sparse_dims(t2) not in t2.shape:
            return True
    if type(t1) != type(t2):
        return False
    if isinstance(t1, CompressedSparseMatrix):
        if t2._indices is t1._indices and t2._pointers is t1._pointers:
            return True
        return close(t1._indices, t2._indices) and close(t1._pointers, t2._pointers)
    if isinstance(t1, SparseCoordinateTensor):
        if t1._indices is t2._indices:
            return True
        if set(t1._indices.shape) != set(t2._indices.shape):
            return False
        return close(t1._indices, t2._indices)
    raise NotImplementedError


def dense(x: Tensor) -> Tensor:
    """
    Convert a sparse tensor representation to an equivalent dense one in which all values are explicitly stored contiguously in memory.

    Args:
        x: Any `Tensor`.
            Python primitives like `float`, `int` or `bool` will be converted to `Tensors` in the process.

    Returns:
        Dense tensor.
    """
    from . import reshaped_tensor
    if isinstance(x, SparseCoordinateTensor):
        from ._ops import scatter
        return scatter(x.shape, x._indices, x._values, mode='add', outside_handling='undefined')
    elif isinstance(x, CompressedSparseMatrix):
        ind_batch, channels, native_indices, native_pointers, native_values, native_shape = x._native_csr_components()
        native_dense = x.default_backend.csr_to_dense(native_indices, native_pointers, native_values, native_shape)
        return reshaped_tensor(native_dense, [ind_batch, x._compressed_dims, x._uncompressed_dims, channels])
    elif isinstance(x, NativeTensor):
        return x
    elif isinstance(x, Tensor):
        return cached(x)
    elif isinstance(x, (Number, bool)):
        return wrap(x)


def dot_compressed_dense(compressed: CompressedSparseMatrix, cdims: Shape, dense: Tensor, ddims: Shape):
    if dense._is_tracer:
        return dense.matmul(compressed, cdims, ddims)
    from . import reshaped_native, reshaped_tensor
    backend = choose_backend(*compressed._natives() + dense._natives())
    if compressed._uncompressed_dims in cdims:  # proper matrix-vector multiplication
        ind_batch, channels, native_indices, native_pointers, native_values, native_shape = compressed._native_csr_components()
        rhs_channels = shape(dense).without(ddims).without(channels)
        dense_native = reshaped_native(dense, [ind_batch, ddims, channels, rhs_channels])
        result_native = backend.mul_csr_dense(native_indices, native_pointers, native_values, native_shape, dense_native)
        result = reshaped_tensor(result_native, [ind_batch, compressed._compressed_dims, channels, rhs_channels])
        return result
    else:  # transposed matrix vector multiplication. This is inefficient
        raise NotImplementedError("Transposed sparse matrix multiplication not yet implemented")


def dot_coordinate_dense(sparse: SparseCoordinateTensor, sdims: Shape, dense: Tensor, ddims: Shape):
    from ._ops import reshaped_native, reshaped_tensor, dot
    if dense._is_tracer:
        return dense.matmul(sparse, sdims, ddims)
    if sdims.as_instance() in instance(sparse._indices):  # no arbitrary sparse reduction needed
        const_entries_dim = instance(sparse._indices).only(sdims.as_instance())
        dense_dims = sparse._dense_shape.only(dense.shape)
        needed_indices = unstack(sparse._indices, const_entries_dim)[0].sparse_idx[dense_dims.name_list]
        dense_gathered = dense[needed_indices]
        values = sparse._values * rename_dims(dense_gathered, ddims, const_entries_dim)
        dense_shape = sparse._dense_shape.without(sdims) & dense.shape.without(ddims)
        indices = sparse._indices[sparse_dims(sparse).without(sdims).name_list]
        return SparseCoordinateTensor(indices, values, dense_shape, sparse._can_contain_double_entries, sparse._indices_sorted, sparse._default)
    backend = choose_backend(*sparse._natives() + dense._natives())
    ind_batch, channels, native_indices, native_values, native_shape = sparse._native_coo_components(sdims, matrix=True)
    rhs_channels = shape(dense).without(ddims).without(channels)
    dense_native = reshaped_native(dense, [ind_batch, ddims, channels, rhs_channels])
    result_native = backend.mul_coo_dense(native_indices, native_values, native_shape, dense_native)
    result = reshaped_tensor(result_native, [ind_batch, sparse._dense_shape.without(sdims), channels, rhs_channels])
    return result


def dot_sparse_sparse(a: Tensor, a_dims: Shape, b: Tensor, b_dims: Shape):
    b = to_format(b, 'coo')
    assert a_dims.rank == b_dims.rank
    remaining_a = sparse_dims(a).without(a_dims)
    remaining_b = sparse_dims(b).without(b_dims)
    list_dim = instance(b._values)

    a_gathered = a[{a_dim.name: b._indices[b_dim.name] for a_dim, b_dim in zip(a_dims, b_dims)}]
    values = a_gathered * b._values  # for each value in B, we have all
    i = values._indices[remaining_a]
    j = b._indices[remaining_b][{instance: values._indices[list_dim.name]}]
    indices = concat([i, j], 'sparse_idx')
    values = values._values
    return SparseCoordinateTensor(indices, values, channel(a) & dual(b), can_contain_double_entries=True, indices_sorted=False, default=a._default)


def native_matrix(value: Tensor, target_backend: Backend):
    target_backend = target_backend or value.default_backend
    cols = dual(value)
    rows = non_batch(value).non_dual
    if not value.available and target_backend != value.default_backend:
        raise RuntimeError(f"Cannot compute native_matrix with target_backend={target_backend} because the {value.default_backend}-value is not available")
    b = target_backend
    if isinstance(value, CompressedSparseMatrix):
        assert not non_instance(value._values), f"native_matrix does not support vector-valued matrices. Vector dims: {non_batch(value).without(sparse_dims(value))}"
        ind_batch, channels, indices, pointers, values, shape = value._native_csr_components()
        if dual(value._uncompressed_dims):  # CSR
            assert not dual(value._compressed_dims), "Dual dimensions on both compressed and uncompressed dimensions"
            if ind_batch.volume > 1 or channels.volume > 1:
                if b.supports(Backend.csr_matrix_batched):
                    return b.csr_matrix_batched(indices, pointers, values, shape)
            else:
                if b.supports(Backend.csr_matrix):
                    return b.csr_matrix(indices[0], pointers[0], values[0, :, 0], shape)
        else:  # CSC
            assert not dual(value._uncompressed_dims)
            if ind_batch.volume > 1 or channels.volume > 1:
                if b.supports(Backend.csc_matrix_batched):
                    return b.csc_matrix_batched(pointers, indices, values, shape)
            else:
                if b.supports(Backend.csc_matrix):
                    return b.csc_matrix(pointers[0], indices[0], values[0, :, 0], shape)
        value = value.decompress()  # backend does not support CSR/CSC, use COO instead
    if isinstance(value, SparseCoordinateTensor):  # COO
        ind_batch, channels, indices, values, shape = value._native_coo_components(dual, matrix=True)
        if ind_batch.volume > 1 or channels.volume > 1:
            return b.sparse_coo_tensor_batched(indices, values, shape)
        else:
            return b.sparse_coo_tensor(indices[0], values[0, :, 0], shape)
    else:
        if batch(value):
            raise NotImplementedError
        v = pack_dims(value, rows, channel('_row'))
        v = pack_dims(v, cols, channel('_col'))
        from ._ops import reshaped_native
        return reshaped_native(v, ['_row', '_col'])


def sparse_dot(x: Tensor, x_dims: Shape, y: Tensor, y_dims: Shape):
    if is_sparse(x) and is_sparse(y) and x_dims in x._values.shape.non_instance and y_dims in y._values.shape.non_instance:
        if same_sparsity_pattern(x, y):
            from ._ops import dot
            new_values = dot(x._values, x_dims, y._values, y_dims)
            return x._with_values(new_values)
        raise NotImplementedError
    if isinstance(x, CompressedSparseMatrix):
        if isinstance(y, (CompressedSparseMatrix, SparseCoordinateTensor)):
            if x_dims.only(sparse_dims(x)) and y_dims.only(sparse_dims(y)):
                return dot_sparse_sparse(x, x_dims, y, y_dims)
            raise NotImplementedError
        return dot_compressed_dense(x, x_dims, y, y_dims)
    elif isinstance(y, CompressedSparseMatrix):
        if isinstance(x, (CompressedSparseMatrix, SparseCoordinateTensor)):
            return dot_sparse_sparse(x, x_dims, y, y_dims)
        return dot_compressed_dense(y, y_dims, x, x_dims)
    if isinstance(x, SparseCoordinateTensor):
        if isinstance(y, (CompressedSparseMatrix, SparseCoordinateTensor)):
            return dot_sparse_sparse(x, x_dims, y, y_dims)
        return dot_coordinate_dense(x, x_dims, y, y_dims)
    elif isinstance(y, SparseCoordinateTensor):
        if isinstance(x, (CompressedSparseMatrix, SparseCoordinateTensor)):
            return dot_sparse_sparse(x, x_dims, y, y_dims)
        return dot_coordinate_dense(y, y_dims, x, x_dims)
    raise NotImplementedError


def add_sparse_batch_dim(matrix: Tensor, in_dims: Shape, out_dims: Shape):
    """
    Bakes dimensions non-instance dimensions of the non-zero values in `matrix` into the sparsity pattern.
    The number of values stays unchanged but the number of indices increases.
    """
    assert instance(out_dims).is_empty
    assert instance(in_dims).is_empty
    from ._ops import arange, meshgrid
    if isinstance(matrix, SparseCoordinateTensor):
        assert out_dims not in matrix.shape
        assert in_dims not in matrix.shape
        indices = concat([arange(channel(sparse_idx=in_dims)), matrix._indices, arange(channel(sparse_idx=out_dims))], 'sparse_idx', expand_values=True)
        offsets = [wrap([*idx.values()] + [0] * non_instance(matrix._indices).volume + [*idx.values()], channel(indices)) for idx in out_dims.meshgrid()]
        offsets = stack(offsets, out_dims.as_instance())
        indices += offsets
        # values = expand(matrix._values, out_dims.as_instance())
        values = matrix._values
        dense_shape = in_dims & matrix._dense_shape & out_dims
        return SparseCoordinateTensor(indices, values, dense_shape, matrix._can_contain_double_entries, matrix._indices_sorted, matrix._default)
    raise NotImplementedError


# def sparsify_batch_dims(matrix: Tensor, in_dims: Shape, out_dims: Shape):
#     """
#     Bakes dimensions non-instance dimensions of the non-zero values in `matrix` into the sparsity pattern.
#     The number of values stays unchanged but the number of indices increases.
#     """
#     assert instance(out_dims).is_empty
#     assert instance(in_dims).is_empty
#     from ._ops import arange, meshgrid
#     if isinstance(matrix, SparseCoordinateTensor):
#         if out_dims in matrix._values.shape:
#             already_sparse = out_dims in matrix._indices.shape
#             if already_sparse:
#                 raise AssertionError(f"Cannot sparsify {out_dims} because dims are are already part of the sparsity pattern {matrix._indices.shape}.")
#             assert out_dims in matrix._values.shape
#             indices = concat([arange(channel(sparse_idx=in_dims)), matrix._indices, arange(channel(sparse_idx=out_dims))], 'sparse_idx', expand_values=True)
#             offsets = [wrap([*idx.values()] + [0] * non_instance(matrix._indices).volume + [*idx.values()], channel(indices)) for idx in out_dims.meshgrid()]
#             offsets = stack(offsets, out_dims.as_instance())
#             indices += offsets
#             values = rename_dims(matrix._values, out_dims, instance)
#             # all_indices = [indices]
#             # for idx in dims.meshgrid():
#             #     if any(i != 0 for i in idx.values()):
#             #         offset = wrap([*idx.values()] * 2 + [0] * non_instance(matrix._indices).volume, channel(indices))
#             #         all_indices.append(indices + offset)
#             # indices = concat(all_indices, instance(indices))
#             # values = pack_dims(matrix._values, concat_shapes(dims, instance(matrix._values)), instance(matrix._values))
#             dense_shape = in_dims & matrix._dense_shape & out_dims
#             return SparseCoordinateTensor(indices, values, dense_shape, matrix._can_contain_double_entries, matrix._indices_sorted, matrix._default)
#     raise NotImplementedError


def with_sparsified_dim(indices: Tensor, values: Tensor, dims: Shape):
    if indices.sparse_idx.item_names == dims.names and dims.only(values.shape).is_empty:
        return indices, values
    assert indices.sparse_idx.item_names in dims, f"dims must include all sparse dims {dims} but got {indices.sparse_idx.item_names}"
    components = []
    for dim in dims:
        if dim.name in indices.sparse_idx.item_names:
            components.append(indices[[dim.name]])
        else:
            from ._ops import meshgrid
            components.append(rename_dims(meshgrid(dim, stack_dim=channel('sparse_idx')), dim, instance))
    indices = concat(components, 'sparse_idx', expand_values=True)
    entries_dims = instance(indices)
    indices = pack_dims(indices, entries_dims, instance('sp_entries'))
    values = rename_dims(values, dims, instance)
    values = pack_dims(expand(values, entries_dims), entries_dims, instance('sp_entries'))
    return indices, values


def sparse_reduce(value: Tensor, dims: Shape, mode: str):
    from ._ops import _sum, _max, _min, scatter, dot, ones, zeros
    reduce = {'add': _sum, 'max': _max, 'min': _min}[mode]
    if value.sparse_dims in dims:  # reduce all sparse dims
        return reduce(value._values, dims.without(value.sparse_dims) & instance(value._values))
    value_only_dims = dims.only(value._values.shape).without(value.sparse_dims)
    if value_only_dims:
        value = value._with_values(reduce(value._values, value_only_dims))
    dims = dims.without(value_only_dims)
    if not dims:
        return value
    if isinstance(value, CompressedSparseMatrix):
        if value._compressed_dims in dims and value._uncompressed_dims.isdisjoint(dims):  # We can ignore the pointers
            result_base = value.shape.without(value._compressed_dims)
            indices = expand(value._indices, channel(index=value._uncompressed_dims))
            return scatter(result_base, indices, value._values, mode=mode, outside_handling='undefined')
        elif value.sparse_dims.only(dims):  # reduce some sparse dims
            if mode == 'add':
                return dot(value, dims, ones(dims), dims)  # this is what SciPy does in both axes, actually.
            else:
                value = value.decompress()
                return sparse_reduce(value, dims, mode)
        else:
            return value  # we have already reduced the value dimensions above
    elif isinstance(value, SparseCoordinateTensor):
        if value._dense_shape in dims:  # sum all sparse dims
            v_dims = dims.without(value._dense_shape) & instance(value._values)
            return reduce(value._values, v_dims)
        else:
            remaining_sparse_dims = value._dense_shape.without(dims)
            indices = value._indices.sparse_idx[remaining_sparse_dims.names]
            if remaining_sparse_dims.rank == 1:  # return dense result
                result = scatter(value.shape.without(dims), indices, value._values, mode=mode, outside_handling='undefined')
                return result
            elif dims.as_instance() in value._values.shape:  # We sum the output batch but keep the input.
                dense_shape = value._dense_shape.without(dims)
                return SparseCoordinateTensor(indices, value._values, dense_shape, value._can_contain_double_entries, value._indices_sorted, value._default)
            else:  # return sparse result
                keep_sparse = sparse_dims(value).without(dims)
                value = pack_dims(value, keep_sparse, channel('_not_summed'))
                summed = sparse_reduce(value, dims, mode)
                return unpack_dim(summed, '_not_summed', keep_sparse)
    raise ValueError(value)


sparse_sum = partial(sparse_reduce, mode='add')
sparse_max = partial(sparse_reduce, mode='max')
sparse_min = partial(sparse_reduce, mode='min')


def sum_equal_entries(matrix: Tensor, flatten_entries=True):
    assert flatten_entries
    from ._ops import reshaped_native, reshaped_tensor
    if not isinstance(matrix, SparseCoordinateTensor):
        return matrix
    if not matrix._can_contain_double_entries:
        return matrix
    b = matrix._indices.default_backend
    entries_dims = instance(matrix._indices)
    values = pack_dims(matrix._values, entries_dims, instance('sp_entries'))
    indices = pack_dims(matrix._indices, entries_dims, instance('sp_entries'))
    dims = matrix._dense_shape.only(channel(indices).item_names[0], reorder=True)
    assert not batch(indices), f"sparse compress() not supported for batched indices"
    idx_packed = b.ravel_multi_index(reshaped_native(indices, [instance, channel]), dims.sizes)
    u_idx, u_ptr = b.unique(idx_packed, return_inverse=True, return_counts=False, axis=-1)
    num_entries = u_idx.shape[-1]
    if num_entries == instance(values).volume:
        matrix._can_contain_double_entries = False
        return matrix
    b = matrix.default_backend
    if non_instance(values):
        batched_values = reshaped_native(values, [non_instance, 'sp_entries'])
        values_nat = b.batched_bincount(u_ptr[None, :], weights=batched_values, bins=num_entries)
        values = wrap(values_nat, non_instance(values), instance('sp_entries'))
    else:
        values = b.bincount(u_ptr, weights=values.native(), bins=num_entries)
        values = reshaped_tensor(values, [instance('sp_entries')])
    idx_packed = b.unravel_index(u_idx, dims.sizes)
    indices = wrap(idx_packed, instance('sp_entries'), channel(matrix._indices))
    return SparseCoordinateTensor(indices, values, matrix._dense_shape, False, True, matrix._default)


def sparse_gather(matrix: Tensor, indices: Tensor):
    if isinstance(matrix, CompressedSparseMatrix):
        indexed = channel(indices).item_names[0]
        if matrix._uncompressed_dims.only(indexed):
            matrix = matrix.decompress()
        else:  # gathering variable-length slices of the values tensor
            matrix = matrix.decompress()  # maybe there is a more efficient way of doing this, but we'd have to rebuild the pointers anyway
    if isinstance(matrix, SparseCoordinateTensor):
        b = matrix._indices.default_backend
        matrix = sum_equal_entries(matrix, flatten_entries=True)
        placeholders = np.arange(1, instance(matrix._values).volume + 1)  # start indexing at 1 since 0 might get removed
        row_dims = matrix._dense_shape.only(channel(indices).item_names[0], reorder=True)
        col_dims = matrix._dense_shape.without(row_dims)
        row_indices = matrix._indices[row_dims.name_list]
        col_indices = matrix._indices[col_dims.name_list]
        # --- Construct SciPy matrix for efficient slicing ---
        from ._ops import reshaped_numpy, reshaped_tensor
        np_rows = NUMPY.ravel_multi_index(reshaped_numpy(row_indices, [instance, channel]), row_dims.sizes)
        np_cols = NUMPY.ravel_multi_index(reshaped_numpy(col_indices, [instance, channel]), col_dims.sizes)
        scipy_mat = csr_matrix((placeholders, (np_rows, np_cols)), shape=(row_dims.volume, col_dims.volume))
        if channel(indices).size > 1 or row_dims.rank > 1:
            raise NotImplementedError  # ravel indices
        else:
            lin_indices = reshaped_numpy(unstack(indices, channel)[0], [shape])
        row_counts = scipy_mat.getnnz(axis=1)  # how many elements per matrix row
        lookup = scipy_mat[lin_indices, :].data - 1
        lookup = expand(wrap(lookup, instance('sp_entries')), channel(sparse_idx=instance(col_indices).name))
        # --- Perform resulting gather on tensors ---
        gathered_cols = col_indices[lookup]
        if non_channel(indices).non_batch:
            row_count_out = row_counts[lin_indices]  # row count for each i in indices
            rows = b.repeat(b.range(len(lin_indices)), row_count_out, 0)
            rows = b.unravel_index(rows, non_channel(indices).non_batch.sizes)
            rows = wrap(rows, instance('sp_entries'), channel(sparse_idx=non_channel(indices).names))
            gathered_indices = concat([rows, gathered_cols], 'sparse_idx')
        else:
            gathered_indices = gathered_cols
        gathered_values = matrix._values[lookup]
        dense_shape = matrix._dense_shape.without(channel(indices).item_names[0]) & non_channel(indices)
        return SparseCoordinateTensor(gathered_indices, gathered_values, dense_shape, can_contain_double_entries=False, indices_sorted=False, default=matrix._default)
    raise NotImplementedError
