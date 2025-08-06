import copy
import os
import pickle
from typing import List
from typing import Union

import faiss
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F


class FaissNN(object):
    """
    FAISS 最近邻搜索类。
    它封装了 FAISS 库，提供了一个统一的接口来执行暴力（Flat）最近邻搜索，
    并支持在 CPU 和 GPU 之间切换。
    """
    def __init__(self, on_gpu: bool = False, num_workers: int = 4) -> None:
        """FAISS 最近邻搜索初始化.

        Args:
            on_gpu: 如果为 True，搜索将在 GPU 上执行。
            num_workers: 用于 FAISS 相似性搜索的线程数。
        """
        # 设置 FAISS 的 OpenMP 线程数，用于并行计算
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index = None

    def _gpu_cloner_options(self):
        """返回 GPU 索引克隆选项，留作子类重写用。"""
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index):
        """将 FAISS 索引从 CPU 转移到 GPU。"""
        if self.on_gpu:
            # 这是一个关键步骤，将索引从 CPU 内存移动到 GPU 显存
            return faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, index, self._gpu_cloner_options()
            )
        return index

    def _index_to_cpu(self, index):
        """将 FAISS 索引从 GPU 转移到 CPU。"""
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension):
        """
        根据 on_gpu 参数创建 FAISS 索引。
        这里创建的是最简单的暴力（Flat）L2 距离索引，即计算所有向量的欧氏距离。
        """
        if self.on_gpu:
            # 创建 GPU 上的暴力 L2 索引
            return faiss.GpuIndexFlatL2(
                faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig()
            )
        # 创建 CPU 上的暴力 L2 索引
        return faiss.IndexFlatL2(dimension)

    def fit(self, features: np.ndarray) -> None:
        """
        构建 FAISS 搜索索引，将训练特征添加到其中。
        这相当于 PatchCore 算法中“构建记忆库”的步骤。

        Args:
            features: 训练集的特征向量数组，形状为 [N, D]，N是向量数量，D是维度。
        """
        # 如果已存在索引，先清空
        if self.search_index:
            self.reset_index()
        # 创建一个新索引，维度与特征匹配
        self.search_index = self._create_index(features.shape[-1])
        # _train 方法留作子类重写，用于近似索引的训练
        self._train(self.search_index, features)
        # 将所有特征向量添加到索引中
        self.search_index.add(features)

    def _train(self, _index, _features):
        """
        这个方法在 FaissNN 中是空的，因为它使用的是 Flat 索引，不需要训练。
        子类如 ApproximateFaissNN 会重写此方法以实现索引训练。
        """
        pass

    def run(
        self,
        n_nearest_neighbours,
        query_features: np.ndarray,
        index_features: np.ndarray = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行最近邻搜索。

        Args:
            n_nearest_neighbours: 需要返回的最近邻数量 K。
            query_features: 要查询的特征向量数组，形状为 [M, D]。
            index_features: [可选] 如果提供，将使用这些特征临时创建一个索引来搜索。
                            如果没有提供，则在之前 fit 的索引中搜索。

        Returns:
            返回 (distances, indices, ...) 元组。
            distances: 距离数组，形状为 [M, K]。
            indices: 索引数组，形状为 [M, K]。
        """
        if index_features is None:
            # 在已构建的索引中搜索
            return self.search_index.search(query_features, n_nearest_neighbours)

        # 临时创建一个新的索引，只用于这次搜索
        search_index = self._create_index(index_features.shape[-1])
        self._train(search_index, index_features)
        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        """将索引保存到文件中。"""
        # 确保索引在 CPU 上才能写入文件
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        """从文件中加载索引。"""
        # 读取索引，并根据配置将其移动到 GPU
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        """重置索引，清空其中的所有向量。"""
        if self.search_index:
            self.search_index.reset()
            self.search_index = None


class ApproximateFaissNN(FaissNN):
    """
    FaissNN 的子类，实现近似最近邻搜索。
    """
    def _train(self, index, features):
        """重写父类方法，用于训练 IndexIVFPQ 索引。"""
        index.train(features)

    def _gpu_cloner_options(self):
        """为 GPU 索引设置选项，这里使用半精度浮点数（float16）以节省显存。"""
        cloner = faiss.GpuClonerOptions()
        cloner.useFloat16 = True
        return cloner

    def _create_index(self, dimension):
        """
        重写父类方法，创建 IndexIVFPQ 索引。
        这是一个近似最近邻索引，比 Flat 索引速度更快、内存占用更小。
        - IndexIVFPQ: 倒排索引（IVF）+ 乘积量化（PQ）。
        - 512: 倒排索引的聚类中心数。
        - 64: 乘积量化的子向量数量。
        - 8: 每个子向量的编码位数。
        """
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(dimension),
            dimension,
            512,  # 聚类中心数
            64,   # 乘积量化的子量化器数量
            8,    # 每个子量化器的编码位数
        )
        return self._index_to_gpu(index)
class _BaseMerger:
    """
    特征合并基类。
    它定义了合并的通用流程，具体的降维或处理逻辑由子类实现。
    """
    def __init__(self):
        """
        初始化特征合并器。
        """
        pass  # 构造函数目前为空，但保留作为接口

    def merge(self, features: list):
        """
        核心方法：将一个特征列表合并成一个单一的特征向量。

        Args:
            features: 包含多个特征数组的列表，每个数组通常来自模型不同层。

        Returns:
            np.ndarray: 合并后的特征向量数组，形状为 [N, D_total]。
        """
        # 遍历所有特征，对每个特征调用子类实现的 _reduce 方法进行降维
        features = [self._reduce(feature) for feature in features]
        # 将降维后的所有特征沿通道维度（axis=1）拼接起来
        return np.concatenate(features, axis=1)


class AverageMerger(_BaseMerger):
    """
    平均值合并器。
    它通过全局平均池化的方式，将特征图降维成一个特征向量。
    """
    @staticmethod
    def _reduce(features):
        """
        将特征图从 NxCxWxH 降维到 NxC。

        Args:
            features: 形状为 [N, C, W, H] 的特征数组。

        Returns:
            np.ndarray: 形状为 [N, C] 的特征向量。
        """
        # 将后两个维度 WxH 展平为一维，形状变为 [N, C, W*H]
        # 然后对 W*H 这一维度求平均值，实现全局平均池化
        return features.reshape([features.shape[0], features.shape[1], -1]).mean(
            axis=-1
        )


class ConcatMerger(_BaseMerger):
    """
    拼接合并器。
    它通过直接展平并拼接的方式，将特征图降维成一个特征向量。
    """
    @staticmethod
    def _reduce(features):
        """
        将特征图从 NxCxWxH 降维到 NxCWH。

        Args:
            features: 形状为 [N, C, W, H] 的特征数组。

        Returns:
            np.ndarray: 形状为 [N, C*W*H] 的特征向量。
        """
        # 将除了批量维度 N 之外的所有维度（C, W, H）全部展平
        return features.reshape(len(features), -1)
class Preprocessing(torch.nn.Module):
    """
    特征预处理模块。
    它负责对从模型不同层提取出的特征进行统一的降维处理。
    """
    def __init__(self, input_dims, output_dim):
        """
        初始化预处理模块。

        Args:
            input_dims: 一个列表，包含每一层特征的输入维度。
            output_dim: 预处理后所有特征统一的目标维度。
        """
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        # 使用 ModuleList 来保存每个特征层的预处理模块
        self.preprocessing_modules = torch.nn.ModuleList()
        # 为每一层特征都创建一个 MeanMapper 实例
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        """
        前向传播，对每个特征进行预处理。

        Args:
            features: 一个包含多层特征张量的列表。

        Returns:
            torch.Tensor: 一个新的张量，将所有预处理后的特征堆叠在一起。
        """
        _features = []
        # 遍历每个预处理模块和对应的特征
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        # 将所有预处理后的特征沿维度1（层维度）堆叠起来
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    """
    平均映射器。
    它使用一维自适应平均池化来将特征向量映射到目标维度。
    """
    def __init__(self, preprocessing_dim):
        """
        初始化映射器。

        Args:
            preprocessing_dim: 映射的目标维度。
        """
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        """
        将特征降维。

        Args:
            features: 输入特征张量。

        Returns:
            torch.Tensor: 降维后的特征张量。
        """
        # 将特征张量重塑，使其适合一维池化操作
        features = features.reshape(len(features), 1, -1)
        # 执行一维自适应平均池化，将特征维度压缩到 preprocessing_dim
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    """
    特征聚合器。
    它将预处理后的多层特征进一步聚合成一个最终的特征向量。
    """
    def __init__(self, target_dim):
        """
        初始化聚合器。
        target_dim: 最终特征向量的目标维度。
        """
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """
        前向传播，将多层特征聚合。
            features: 形状为 [batch_size, num_layers, input_dim] 的特征张量
            torch.Tensor: 聚合后的特征向量，形状为 [batch_size, target_dim]。
        """
        # 将张量重塑，使其适合一维池化
        # [batch_size, num_layers * input_dim] -> [batch_size, 1, num_layers * input_dim]
        features = features.reshape(len(features), 1, -1)
        # 执行一维自适应平均池化，将所有层的特征聚合成 target_dim
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        # 移除维度1，得到最终的特征向量
        return features.reshape(len(features), -1)


class RescaleSegmentor:
    def __init__(self, device, target_size=224):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores):

        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()

        return [
            ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ]


class NetworkFeatureAggregator(torch.nn.Module):
    """
    网络特征聚合器。
    该类通过注册 hook，可以高效地从骨干网络的指定层提取特征。
    它只运行到需要提取的最后一层，然后通过抛出异常来提前终止计算。
    """
    def __init__(self, backbone, layers_to_extract_from, device):
        """
        初始化特征聚合器。
            backbone: 一个 torchvision.model 类型的骨干网络，例如 ResNet。
            layers_to_extract_from: 一个字符串列表，包含需要提取特征的层名。
            device: 指定运行的设备，
        """
        super(NetworkFeatureAggregator, self).__init__()
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        
        # 确保骨干网络有 hook_handles 属性，用于存储 hook 的句柄
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        # 清除骨干网络上已有的 hook，防止重复注册
        for handle in self.backbone.hook_handles:
            handle.remove()
        
        self.outputs = {}  # 字典，用于存储提取到的特征

        # 遍历需要提取的所有层，并为每一层注册一个前向 hook
        for extract_layer in layers_to_extract_from:
            # 创建一个 ForwardHook 实例
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            
            # 根据层名的字符串格式，找到对应的网络层
            if "." in extract_layer:
                # 处理如 "layer1.2" 这样的子模块
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                # 处理如 "layer1" 这样的顶级模块
                network_layer = backbone.__dict__["_modules"][extract_layer]

            # 注册 hook：如果层是 Sequential，则挂在最后一层；否则直接挂在该层
            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.to(self.device)

    def forward(self, images):
        """
        前向传播方法。
            images: 输入的图像张量。
            dict: 包含所有提取到的特征的字典。
        """
        self.outputs.clear()  # 每次前向传播前清空旧的特征
        with torch.no_grad():
            try:
                # 运行骨干网络。当达到最后一个 hook 时，会抛出异常并中断计算
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass  # 捕获异常，正常退出前向传播
        return self.outputs

    def feature_dimensions(self, input_shape):
        """
        计算给定输入形状下，所有提取到的特征层的维度。
        """
        # 使用一个虚拟输入来运行一次前向传播，以获取各层的输出维度
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class ForwardHook:
    """
    自定义前向 hook 类。
    它作为一个回调函数，用于捕获和存储中间层的输出。
    """
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict  # 存储特征的字典
        self.layer_name = layer_name  # 当前 hook 所在的层名
        # 标记是否为最后一个需要提取的层，如果是，则在执行后抛出异常
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        """
        Hook 的回调方法，在前向传播时被调用。
        """
        # 将当前层的输出存储到字典中
        self.hook_dict[self.layer_name] = output
        # 如果是最后一个 hook，则抛出异常以终止计算
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    """
    自定义异常，用于在特征提取完成后，优雅地终止模型的前向传播。
    """
    pass


class NearestNeighbourScorer(object):
    """
    最近邻异常分数计算器类。
    该类使用最近邻搜索方法来计算图像或像素的异常分数。
    它整合了特征合并和最近邻搜索功能，是 PatchCore 算法的核心组件。
    """
    def __init__(self, n_nearest_neighbours: int, nn_method=FaissNN(False, 4)) -> None:
        """
        初始化异常分数计算器。

        Args:
            n_nearest_neighbours: 用于确定异常像素的最近邻数量。
            nn_method: 最近邻搜索方法实例，默认为 FaissNN。
        """
        # 使用 ConcatMerger 来合并特征，这表明它主要用于像素级检测
        self.feature_merger = ConcatMerger()

        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method = nn_method

        # 定义一个用于图像级（或补丁级）搜索的 lambda 函数
        # 它会查找 k 个最近邻
        self.imagelevel_nn = lambda query: self.nn_method.run(
            n_nearest_neighbours, query
        )
        # 定义一个用于像素级搜索的 lambda 函数
        # 它只查找 1 个最近邻
        self.pixelwise_nn = lambda query, index: self.nn_method.run(1, query, index)

    def fit(self, detection_features: List[np.ndarray]) -> None:
        """
        训练模型。这包括构建特征记忆库。

        Args:
            detection_features: 训练集的所有特征列表，每个元素是一个图像的特征数组。
        """
        # 使用合并器将所有特征合并成一个单一的巨大特征数组
        self.detection_features = self.feature_merger.merge(
            detection_features,
        )
        # 将合并后的特征添加到最近邻方法（FaissNN）的索引中
        self.nn_method.fit(self.detection_features)

    def predict(
        self, query_features: List[np.ndarray]
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """
        预测异常分数。

        Args:
            query_features: 测试集的特征列表。

        Returns:
            anomaly_scores: 异常分数，形状为 [N]。
            query_distances: 距离矩阵，形状为 [N, K]。
            query_nns: 最近邻索引矩阵，形状为 [N, K]。
        """
        # 合并查询特征，使其与训练特征格式一致
        query_features = self.feature_merger.merge(
            query_features,
        )
        # 在训练好的索引中搜索查询特征的 k 个最近邻
        query_distances, query_nns = self.imagelevel_nn(query_features)
        # 异常分数被定义为 k 个最近邻距离的平均值
        anomaly_scores = np.mean(query_distances, axis=-1)
        return anomaly_scores, query_distances, query_nns

    @staticmethod
    def _detection_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename, features):
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(
        self,
        save_folder: str,
        save_features_separately: bool = False,
        prepend: str = "",
    ) -> None:
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(
                self._detection_file(save_folder, prepend), self.detection_features
            )

    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.nn_method.load(self._index_file(load_folder, prepend))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(
                self._detection_file(load_folder, prepend)
            )
