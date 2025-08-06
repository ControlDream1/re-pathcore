import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
          

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]
        '''
        此时的features的特征形式为 features = [特征图1, 特征图2, 特征图3, ...]
        eg features = [Tensor(1, 256, 56, 56), Tensor(1, 512, 28, 28)]
        '''
        

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]  
        '''
        这里便是引入path补丁特征,通过parhify方法将特征图分割成多个patch,并返回补丁数量
        此时的features的特征形式为 features = [(补丁特征张量1, 补丁数量1), (补丁特征张量2, 补丁数量2), ...]
        eg features = [(Tensor(1, 56*56, 256, 3, 3), [56, 56]), (Tensor(1, 28*28, 512, 3, 3), [28, 28])]
        '''


        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]


        '''
        下面便是我认为最复杂的板块 我们经过不同层的卷积网络提取特征，那么特征图的大小必然不统一，为了方面后面计算，我们需要将不同层的特征进行聚合和降维，得到最终的特征图
        '''

        '''
        第一部分是for循环 通过双线性插值的方法，将所有低分辨率的补丁特征都上采样到高分辨率的补丁数量一致 这样不同层的特征图大小变成一致
        在这我们详细解释一下双线性插值的方法：
        在图像处理中，双线性插值主要用于缩放图像。无论是放大（upsampling）还是缩小（downsampling）图像，它都能产生比最近邻插值更平滑、质量更高的结果
        双线性插值是一种平滑且高效的重采样方法，它通过计算周围补丁特征的加权平均值，来生成新的特征点，从而在不引入明显伪影的情况下，将低分辨率特征图上采样到高分辨率
        '''
        for i in range(1, len(features)):
            _features = features[i]      #当前处理的低分辨率特征图 目的是为了将低特征图片上采样成第一张特征图
            patch_dims = patch_shapes[i]    #当前处理的低分辨率特征图的补丁数量

           
            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )  
            '''
            代码执行后，_features 的形状将变为 [batch_size, patch_height_count, patch_width_count, channels, patch_height, patch_width]  后面的双线性插值（F.interpolate）做准备
            '''

            _features = _features.permute(0, -3, -2, -1, 1, 2)
            '''
            调整特征维度 调整后为[batch_size, channels, patch_height, patch_width, patch_height_count, patch_width_count]
            '''

            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            '''
            [batch_size * channels * patch_height * patch_width, patch_height_count, patch_width_count]
            '''

            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),   # 需要上采集的大小 就是高分辨率的信息
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

       
        features = self.forward_modules["preprocessing"](features)
        '''
        这是一个预处理模块:
        1.统一特征维度： 如果不同层级的特征通道数 channels 不同，这个模块可以将其投影到相同的维度上
        2.特征降维： 它可以将高维的特征向量投影到较低维度的空间，从而降低存储和计算成本。
        '''
        features = self.forward_modules["preadapt_aggregator"](features)
        '''
        这是一个聚合模块:
        将不同层级的特征图聚合在一起，形成一个统一的特征表示，eg：[batch_size, 1024, 56, 56]

        '''

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)





    '''
    下面也是很重要的部分 构建正常样本的记忆库
    '''

    def fit(self, training_data):
        
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        #构建普通的特征库
        features = np.concatenate(features, axis=0)
        #构建贪婪核心子库 贪婪核心子库是基于特征库的子集，通过贪婪算法选择最具代表性的特征点，从而减少特征库的大小，提高计算效率
        features = self.featuresampler.run(features) 
        #将贪婪核心子库存储到记忆库中
        self.anomaly_scorer.fit(detection_features=[features])




    '''
    下面是算法的核心推理 接收输入图像 并输出异常分数和异常检测定位
    '''
    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images):
        
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]   # 像素级和图像级分数

            '''
            调用path_maker中的函数 来将扁平化后的补丁分数重新恢复形状 方便后面聚合
            [batchsize, num_patches_height, num_patches_width, ...]
            注意此处是图片级分数
            '''
            
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)


            '''
            这里是像素级分数
            无需分数的聚合 只用保证维度和最高分辨率图片相等即可
            '''
            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize      
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            '''
            根据像素级的补丁分数 生成热力图 从而实现异常分割
            '''
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]


    '''
    下面两个方法分别为模型保存和模型加载
    '''
    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)
 
    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)



class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """
        将输出特征转成path特征
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    '''
    将补丁分数聚合 成单一图片级分数
    '''
    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
