import sys
from collections import namedtuple
sys.path.append("./")
from utils.wrap_lmdb import lmdb_init, lmdb_write, lmdb_read, lmdb_clean, lmdb_len, lmdb_sample
from utils.mmdb import mmdb_init, mmdb_write, mmdb_read, mmdb_clean, mmdb_len, mmdb_sample, mmdb_update
from utils.pmdb import pmdb_init, pmdb_write, pmdb_read, pmdb_clean, pmdb_len, pmdb_sample, pmdb_update

"""
define dataset operators
data_op.init(dataset)
data_op.write(dataset, data)
data_op.read(dataset, idxs)
data_op.sample(dataset, nb, shullfer)
data_op.clean(dataset)
data_op.len(dataset)
"""
DataOp = namedtuple("DataOp",["init", "write", "read", "clean", "len", "sample", "update"])

lmdb_op = DataOp(lmdb_init, lmdb_write, lmdb_read, lmdb_clean, lmdb_len, lmdb_sample, None)

mmdb_op = DataOp(mmdb_init, mmdb_write, mmdb_read, mmdb_clean, mmdb_len, mmdb_sample, mmdb_update)

pmdb_op = DataOp(pmdb_init, pmdb_write, pmdb_read, pmdb_clean, pmdb_len, pmdb_sample, pmdb_update)