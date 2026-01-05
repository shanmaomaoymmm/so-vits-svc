from time import time

import numpy as np
import psutil
import torch
from torch.nn.functional import normalize


# device=torch.device("cuda:0")
def _kpp(data: torch.Tensor, k: int, sample_size: int = -1):
    """ Picks k points in the data based on the kmeans++ method.

    Parameters
    ----------
    data : torch.Tensor
        Expect a rank 1 or 2 array. Rank 1 is assumed to describe 1-D
        data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    sample_size : int
        sample data to avoid memory overflow during calculation

    Returns
    -------
    init : ndarray
        A 'k' by 'N' containing the initial centroids.

    References
    ----------
    .. [1] D. Arthur and S. Vassilvitskii, "k-means++: the advantages of
       careful seeding", Proceedings of the Eighteenth Annual ACM-SIAM Symposium
       on Discrete Algorithms, 2007.
    .. [2] scipy/cluster/vq.py: _kpp
    """
    batch_size=data.shape[0]
    if batch_size>sample_size:
        data = data[torch.randint(0, batch_size,[sample_size], device=data.device)]
    dims = data.shape[1] if len(data.shape) > 1 else 1
    init = torch.zeros((k, dims)).to(data.device)
    r = torch.distributions.uniform.Uniform(0, 1)
    for i in range(k):
        if i == 0:
            init[i, :] = data[torch.randint(data.shape[0], [1])]
        else:
            D2 = torch.cdist(init[:i, :][None, :], data[None, :], p=2)[0].amin(dim=0)
            probs = D2 / torch.sum(D2)
            cumprobs = torch.cumsum(probs, dim=0)
            init[i, :] = data[torch.searchsorted(cumprobs, r.sample([1]).to(data.device))]
    return init
class KMeansGPU:
  '''
  Kmeans clustering algorithm implemented with PyTorch

  Parameters:
    n_clusters: int, 
      Number of clusters

    max_iter: int, default: 100
      Maximum number of iterations

    tol: float, default: 0.0001
      Tolerance
    
    verbose: int, default: 0
      Verbosity

    mode: {'euclidean', 'cosine'}, default: 'euclidean'
      Type of distance measure
      
    init_method: {'random', 'point', '++'}
      Type of initialization

    minibatch: {None, int}, default: None
      Batch size of MinibatchKmeans algorithm
      if None perform full KMeans algorithm
      
  Attributes:
    centroids: torch.Tensor, shape: [n_clusters, n_features]
      cluster centroids
  '''
  def __init__(self, n_clusters, max_iter=200, tol=1e-4, verbose=0, mode="euclidean",device=torch.device("cuda:0")):
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    self.tol = tol
    self.verbose = verbose
    self.mode = mode
    self.device=device
    if device.type == 'cuda':
        # 使用 torch.cuda 获取 GPU 内存信息
        gpu_memory = torch.cuda.get_device_properties(device.index).total_memory
        # 计算可用内存，使用总内存的一个比例作为估计值
        available_memory = gpu_memory * 0.8  # 假设80%为可用内存
        self.minibatch = int(33e6/self.n_clusters*available_memory/ 1024 / 1024 / 1024)
        print("gpu_memory/GB:", gpu_memory/ 1024 / 1024 / 1024,"minibatch:",self.minibatch)
    elif device.type == 'xpu':
        # XPU设备使用不同的内存管理方式
        xpu_memory = torch.xpu.get_device_properties(device.index).total_memory
        # 使用可用内存的一个比例，而不是直接查询可用内存
        available_memory = xpu_memory * 0.8  # 假设80%为可用内存
        self.minibatch = int(33e6/self.n_clusters*available_memory/ 1024 / 1024 / 1024)
        print("xpu_memory/GB:", xpu_memory/ 1024 / 1024 / 1024,"minibatch:",self.minibatch)
    else:
        # CPU设备使用系统内存信息
        total_memory = psutil.virtual_memory().total
        # 计算可用内存，使用总内存的一个比例作为估计值
        available_memory = total_memory * 0.5  # 假设50%为可用内存
        self.minibatch = int(33e6/self.n_clusters*available_memory/ 1024 / 1024 / 1024)
        print("cpu_available_memory/GB:", available_memory/ 1024 / 1024 / 1024,"minibatch:",self.minibatch)
    
  @staticmethod
  def cos_sim(a, b):
    """
      Compute cosine similarity of 2 sets of vectors

      Parameters:
      a: torch.Tensor, shape: [m, n_features]

      b: torch.Tensor, shape: [n, n_features]
    """
    return normalize(a, dim=-1) @ normalize(b, dim=-1).transpose(-2, -1)

  @staticmethod