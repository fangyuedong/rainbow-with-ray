import sys
from collections import namedtuple
sys.path.append("./")
from utils.wrap_lmdb import lmdb_init, lmdb_write, lmdb_read, lmdb_clean, lmdb_len, lmdb_sample
from utils.mmdb import mmdb_init, mmdb_write, mmdb_read, mmdb_clean, mmdb_len, mmdb_sample

"""
define dataset operators
data_op.init(dataset)
data_op.write(dataset, data)
data_op.read(dataset, idxs)
data_op.sample(dataset, nb, shullfer)
data_op.clean(dataset)
data_op.len(dataset)
"""
DataOp = namedtuple("DataOp",["init", "write", "read", "clean", "len", "sample"])

lmdb_op = DataOp(lmdb_init, lmdb_write, lmdb_read, lmdb_clean, lmdb_len, lmdb_sample)

mmdb_op = DataOp(mmdb_init, mmdb_write, mmdb_read, mmdb_clean, mmdb_len, mmdb_sample)