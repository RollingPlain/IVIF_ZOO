
## Latest News 🔥🔥
[2024-12-12] Our survey paper [__Infrared and Visible Image Fusion: From Data Compatibility to Task Adaption.__] has been accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence!
([Paper](https://ieeexplore.ieee.org/abstract/document/10812907))([中文版](https://pan.baidu.com/s/1EIRYSULa-pd2FRmIdG693g?pwd=aiey))

# IVIF Zoo
Welcome to IVIF Zoo, a comprehensive repository dedicated to Infrared and Visible Image Fusion (IVIF). Based on our survey paper [__Infrared and Visible Image Fusion: From Data Compatibility to Task Adaption.__ *Jinyuan Liu, Guanyao Wu, Zhu Liu, Di Wang, Zhiying Jiang, Long Ma, Wei Zhong, Xin Fan, Risheng Liu**], this repository aims to serve as a central hub for researchers, engineers, and enthusiasts in the field of IVIF. Here, you'll find a wide array of resources, tools, and datasets, curated to accelerate advancements and foster collaboration in infrared-visible image fusion technologies.

***

![preview](assets/light2.png)
<sub>A detailed spectrogram depicting almost all wavelength and frequency ranges, particularly expanding the range of the human visual system and annotating corresponding computer vision and image fusion datasets.</sub>

![preview](assets/pipeline1.png)
The diagram of infrared and visible image fusion for practical applications. Existing image fusion methods majorly focus on the design of architectures and training strategies for visual enhancement, few considering the adaptation for downstream visual perception tasks. Additionally, from the data compatibility perspective, pixel misalignment and adversarial attacks of image fusion are two major challenges. Additionally, integrating comprehensive semantic information for tasks like semantic segmentation, object detection, and salient object detection remains underexplored, posing a critical obstacle in image fusion.

![preview](assets/sankey1.png)
 A classification sankey diagram containing typical fusion methods.

***

## 导航(Navigation)

- [数据集 (Datasets)](#数据集datasets)
- [方法集 (Method Set)](#方法集method-set)
  - [纯融合方法 (Fusion for Visual Enhancement)](#纯融合方法fusion-for-visual-enhancement)
  - [数据兼容方法 (Data Compatible)](#数据兼容方法data-compatible)
  - [面向应用方法 (Application-oriented)](#面向应用方法application-oriented)
- [评价指标 (Evaluation Metric)](#评价指标evaluation-metric)
###  [🔥🚀资源库 (Resource Library)](#资源库resource-library)  
`It covers all results of our survey paper, available for download from Baidu Cloud.`
  - 💥[融合 (Fusion)](#融合fusion) 
  - ✂️[分割 (Segmentation)](#分割segmentation) `Based on SegFormer`
  - 🔍[检测 (Detection)](#检测detection) `Based on YOLO-v5`
  - [计算效率 (Computational Efficiency)](#计算效率computational-efficiency)
# 数据集(Datasets)
<table>
    <thead>
        <tr>
            <th>Dataset</th>
            <th>Img pairs</th>
            <th>Resolution</th>
            <th>Color</th>
            <th>Obj/Cats</th>
            <th>Cha-Sc</th>
            <th>Anno</th>
            <th>DownLoad</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>TNO</td>
            <td>261</td>
            <td>768×576</td>
            <td>❌</td>
            <td>few</td>
            <td>✔</td>
            <td>❌</td>
            <td><a href="https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029">Link</a></td>
        </tr>
        <tr>
            <td>RoadScene 🔥</td>
            <td>221</td>
            <td>Various</td>
            <td>✔</td>
            <td>medium</td>
            <td>❌</td>
            <td>❌</td>
            <td><a href="https://github.com/hanna-xu/RoadScene">Link</a></td>
        </tr>
        <tr>
            <td>VIFB</td>
            <td>21</td>
            <td>Various</td>
            <td>Various</td>
            <td>few</td>
            <td>❌</td>
            <td>❌</td>
            <td><a href="https://github.com/xingchenzhang/Visible-infrared-image-fusion-benchmark">Link</a></td>
        </tr>
        <tr>
            <td>MS</td>
            <td>2999</td>
            <td>768×576</td>
            <td>✔</td>
            <td>14146 / 6</td>
            <td>❌</td>
            <td>✔</td>
            <td><a href="https://www.mi.t.u-tokyo.ac.jp/projects/mil_multispectral/index.html">Link</a></td>
        </tr>
        <tr>
            <td>LLVIP</td>
            <td>16836</td>
            <td>1280×720</td>
            <td>✔</td>
            <td>pedestrian / 1</td>
            <td>❌</td>
            <td>✔</td>
            <td><a href="https://bupt-ai-cz.github.io/LLVIP/">Link</a></td>
        </tr>
        <tr>
            <td>M<sup>3</sup>FD 🔥</td>
            <td>4200</td>
            <td>1024×768</td>
            <td>✔</td>
            <td>33603 / 6</td>
            <td>✔</td>
            <td>✔</td>
            <td><a href="https://github.com/JinyuanLiu-CV/TarDAL">Link</a></td>
        </tr>
        <tr>
            <td>MFNet</td>
            <td>1569</td>
            <td>640×480</td>
            <td>✔</td>
            <td>abundant / 8</td>
            <td>❌</td>
            <td>✔</td>
            <td><a href="https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/">Link</a></td>
        </tr>
        <tr>
            <td>FMB 🔥</td>
            <td>1500</td>
            <td>800×600</td>
            <td>✔</td>
            <td>abundant / 14</td>
            <td>❌</td>
            <td>✔</td>
            <td><a href="https://github.com/JinyuanLiu-CV/SegMiF">Link</a></td>
        </tr>
    </tbody>
</table>


If the M<sup>3</sup>FD and FMB datasets are helpful to you, please cite the following paper:

```
@inproceedings{liu2022target,
  title={Target-aware dual adversarial learning and a multi-scenario multi-modality benchmark to fuse infrared and visible for object detection},
  author={Liu, Jinyuan and Fan, Xin and Huang, Zhanbo and Wu, Guanyao and Liu, Risheng and Zhong, Wei and Luo, Zhongxuan},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={5802--5811},
  year={2022}
}
```

```
@inproceedings{liu2023multi,
  title={Multi-interactive feature learning and a full-time multi-modality benchmark for image fusion and segmentation},
  author={Liu, Jinyuan and Liu, Zhu and Wu, Guanyao and Ma, Long and Liu, Risheng and Zhong, Wei and Luo, Zhongxuan and Fan, Xin},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={8115--8124},
  year={2023}
}
```

# 方法集(Method Set)
## 纯融合方法(Fusion for Visual Enhancement)
<table>
    <thead>
        <tr>
            <th>Aspects<br>(分类)</th>
            <th>Methods<br>(方法)</th>
            <th>Title<br>(标题)</th>
            <th>Venue<br>(发表场所)</th>
            <th>Source<br>(资源)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Auto-Encoder</td>
            <td>DenseFuse</td>
            <td>Densefuse: A fusion approach to infrared and visible images</td>
            <td>TIP '18</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/8580578/">Paper</a>/<a href="https://github.com/hli1221/imagefusion_densefuse">Code</a></td>
        </tr>
        <tr>
            <td>Auto-Encoder</td>
            <td>SEDRFuse</td>
            <td>Sedrfuse: A symmetric encoder–decoder with residual block network for infrared and visible image fusion</td>
            <td>TIM '20</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/9187663/">Paper</a>/<a href="https://github.com/jianlihua123/SEDRFuse">Code</a></td>
        </tr>
        <tr>
            <td>Auto-Encoder</td>
            <td>DIDFuse</td>
            <td>Didfuse: Deep image decomposition for infrared and visible image fusion</td>
            <td>IJCAI '20</td>
            <td><a href="https://arxiv.org/abs/2003.09210">Paper</a>/<a href="https://github.com/Zhaozixiang1228/IVIF-DIDFuse">Code</a></td>
        </tr>
        <tr>
            <td>Auto-Encoder</td>
            <td>MFEIF</td>
            <td>Learning a deep multi-scale feature ensemble and an edge-attention guidance for image fusion</td>
            <td>TCSVT '21</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/9349250/">Paper</a>/<a href="https://github.com/JinyuanLiu-CV/MFEIF">Code</a></td>
        </tr>
        <tr>
            <td>Auto-Encoder</td>
            <td>RFN-Nest</td>
            <td>Rfn-nest: An end-to-end residual fusion network for infrared and visible images</td>
            <td>TIM '21</td>
            <td><a href="https://www.sciencedirect.com/science/article/pii/S1566253521000440">Paper</a>/<a href="https://github.com/hli1221/imagefusion-rfn-nest">Code</a></td>
        </tr>
        <tr>
            <td>Auto-Encoder</td>
            <td>SFAFuse</td>
            <td>Self-supervised feature adaption for infrared and visible image fusion</td>
            <td>InfFus '21</td>
            <td><a href="https://www.sciencedirect.com/science/article/pii/S1566253521001287">Paper</a>/<a href="https://github.com/zhoafan/SFA-Fuse">Code</a></td>
        </tr>
        <tr>
            <td>Auto-Encoder</td>
            <td>SMoA</td>
            <td>Smoa: Searching a modality-oriented architecture for infrared and visible image fusion</td>
            <td>SPL '21</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/9528046/">Paper</a>/<a href="https://github.com/JinyuanLiu-CV/SMoA">Code</a></td>
        </tr>
        <tr>
            <td>Auto-Encoder</td>
            <td>Re2Fusion</td>
            <td>Res2fusion: Infrared and visible image fusion based on dense res2net and double nonlocal attention models</td>
            <td>TIM '22</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/9670874/">Paper</a>/<a href="https://github.com/Zhishe-Wang/Res2Fusion">Code</a></td>
        </tr>
             <tr>
        <td>Auto-Encoder</td>
        <td>RPFNet</td>
        <td>Residual Prior-driven Frequency-aware Network for Image Fusion</td>
        <td>ACM MM '25</td>
        <td><a href="https://arxiv.org/abs/2507.06735">Paper</a>/<a href="https://github.com/wang-x-1997/RPFNet">Code</a></td>
    </tr>
    <tr>
        <td>Auto-Encoder</td>
        <td>TTD</td>
        <td>Test-Time Dynamic Image Fusion</td>
        <td>NeurIPS '24</td>
        <td><a href="https://nips.cc/virtual/2024/poster/95415">Paper</a>/<a href="https://github.com/Yinan-Xia/TTD">Code</a></td>
    </tr>
        <tr>
            <td>GAN</td>
            <td>FusionGAN</td>
            <td>Fusiongan: A generative adversarial network for infrared and visible image fusion</td>
            <td>InfFus '19</td>
            <td><a href="https://www.sciencedirect.com/science/article/pii/S1566253518301143">Paper</a>/<a href="https://github.com/jiayi-ma/FusionGAN">Code</a></td>
        </tr>
        <tr>
            <td>GAN</td>
            <td>DDcGAN</td>
            <td>Learning a generative model for fusing infrared and visible images via conditional generative adversarial network with dual discriminators</td>
            <td>TIP '19</td>
            <td><a href="https://www.ijcai.org/proceedings/2019/0549.pdf">Paper</a>/<a href="https://github.com/hanna-xu/DDcGAN">Code</a></td>
        </tr>
        <tr>
            <td>GAN</td>
            <td>AtFGAN</td>
            <td>Attentionfgan: Infrared and visible image fusion using attention-based generative adversarial networks</td>
            <td>TMM '20</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/9103116">Paper</a></td>
        </tr>
        <tr>
            <td>GAN</td>
            <td>DPAL</td>
            <td>Infrared and visible image fusion via detail preserving adversarial learning</td>
            <td>InfFus '20</td>
            <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S1566253519300314">Paper</a>/<a href="https://github.com/StaRainJ/ResNetFusion">Code</a></td>
        </tr>
        <tr>
            <td>GAN</td>
            <td>D2WGAN</td>
            <td>Infrared and visible image fusion using dual discriminators generative adversarial networks with wasserstein distance</td>
            <td>InfSci '20</td>
            <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S0020025520303431">Paper</a></td>
        </tr>
        <tr>
            <td>GAN</td>
            <td>GANMcC</td>
            <td>Ganmcc: A generative adversarial network with multiclassification constraints for infrared and visible image fusion</td>
            <td>TIM '20</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/9274337/">Paper</a>/<a href="https://github.com/HaoZhang1018/GANMcC">Code</a></td>
        </tr>
        <tr>
            <td>GAN</td>
            <td>ICAFusion</td>
            <td>Infrared and visible image fusion via interactive compensatory attention adversarial learning</td>
            <td>TMM '22</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/9982426/">Paper</a>/<a href="https://github.com/Zhishe-Wang/ICAFusion">Code</a></td>
        </tr>
        <tr>
            <td>GAN</td>
            <td>TCGAN</td>
            <td>Transformer based conditional gan for multimodal image fusion</td>
            <td>TMM '23</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/10041783/">Paper</a>/<a href="https://github.com/jinxiqinghuan/TCGAN">Code</a></td>
        </tr>
        <tr>
        <tr>
            <td>GAN</td>
            <td>DCFusion</td>
            <td>DCFusion: A Dual-Frequency Cross-Enhanced Fusion Network for Infrared and Visible Image Fusion</td>
            <td>TIM '23</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/10102546">Paper</a></td>
        </tr>
        <tr>
            <td>GAN</td>
            <td>FreqGAN</td>
            <td>Freqgan: Infrared and visible image fusion via unified frequency adversarial learning</td>
            <td>TCSVT '24</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/10680110/">Paper</a>/<a href="https://github.com/Zhishe-Wang/FreqGAN">Code</a></td>
        </tr>
     <tr>
        <td>GAN</td>
        <td>DDBF</td>
        <td>Dispel Darkness for Better Fusion: A Controllable Visual Enhancer based on Cross-modal Conditional Adversarial Learning</td>
        <td>CVPR '24</td>
        <td><a href="https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Dispel_Darkness_for_Better_Fusion_A_Controllable_Visual_Enhancer_based_CVPR_2024_paper.html">Paper</a>/<a href="https://github.com/HaoZhang1018/DDBF">Code</a></td>
    </tr>
    <tr>
        <td>GAN</td>
        <td>CCF</td>
        <td>Conditional Controllable Image Fusion</td>
        <td>NeurIPS '24</td>
        <td><a href="https://proceedings.neurips.cc/paper_files/paper/2024/file/d99e8e80a6c41e148db686918dd7eab3-Paper-Conference.pdf">Paper</a>/<a href="https://github.com/jehovahxu/CCF">Code</a></td>
    </tr>
        <tr>
            <td>CNN</td>
            <td>BIMDL</td>
            <td>A bilevel integrated model with data-driven layer ensemble for multi-modality image fusion</td>
            <td>TIP '20</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/9293146">Paper</a></td>
        </tr>
        <tr>
            <td>CNN</td>
            <td>MgAN-Fuse</td>
            <td>Multigrained attention network for infrared and visible image fusion</td>
            <td>TIM '20</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/9216075">Paper</a></td>
        </tr>
        <tr>
            <td>CNN</td>
            <td>AUIF</td>
            <td>Efficient and model-based infrared and visible image fusion via algorithm unrolling</td>
            <td>TCSVT '21</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/9416456">Paper</a>/<a href="https://github.com/Zhaozixiang1228/IVIF-AUIF-Net">Code</a></td>
        </tr>
        <tr>
            <td>CNN</td>
            <td>RXDNFuse</td>
            <td>Rxdnfuse: A aggregated residual dense network for infrared and visible image fusion</td>
            <td>InfFus '21</td>
            <td><a href="https://www.sciencedirect.com/science/article/pii/S1566253520304152">Paper</a></td>
        </tr>
        <tr>
            <td>CNN</td>
            <td>STDFusionNet</td>
            <td>Stdfusionnet: An infrared and visible image fusion network based on salient target detection</td>
            <td>TIM '21</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/9416507">Paper</a>/<a href="https://github.com/jiayi-ma/STDFusionNet">Code</a></td>
        </tr>
        <tr>
            <td>CNN</td>
            <td>CUFD</td>
            <td>Cufd: An encoder–decoder network for visible and infrared image fusion based on common and unique feature decomposition</td>
            <td>CVIU '22</td>
            <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S1077314222000352">Paper</a>/<a href="https://github.com/Meiqi-Gong/CUFD">Code</a></td>
        </tr>
        <tr>
            <td>CNN</td>
            <td>Dif-Fusion</td>
            <td>Dif-fusion: Towards high color fidelity in infrared and visible image fusion with diffusion models</td>
            <td>TIP '23</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/10286359/">Paper</a>/<a href="https://github.com/GeoVectorMatrix/Dif-Fusion">Code</a></td>
        </tr>
        <tr>
            <td>CNN</td>
            <td>L2Net</td>
            <td>L2Net: Infrared and Visible Image Fusion Using Lightweight Large Kernel Convolution Network</td>
            <td>TIP '23</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/10301581">Paper</a>/<a href="https://github.com/chang-le-11/L2Net">Code</a></td>
        </tr>
        <tr>
            <td>CNN</td>
            <td>IGNet</td>
            <td>Learning a graph neural network with cross modality interaction for image fusion</td>
            <td>ACMMM '23</td>
            <td><a href="https://dl.acm.org/doi/abs/10.1145/3581783.3612135">Paper</a>/<a href="https://github.com/lok-18/IGNet">Code</a></td>
        </tr>
        <tr>
            <td>CNN</td>
            <td>LRRNet</td>
            <td>Lrrnet: A novel representation learning guided fusion network for infrared and visible images</td>
            <td>TPAMI '23</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/10105495/">Paper</a>/<a href="https://github.com/hli1221/imagefusion-LRRNet">Code</a></td>
        </tr>
        <tr>
            <td>CNN</td>
            <td>MetaFusion</td>
            <td>Metafusion: Infrared and visible image fusion via meta-feature embedding from object detection</td>
            <td>CVPR '23</td>
            <td><a href="https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_MetaFusion_Infrared_and_Visible_Image_Fusion_via_Meta-Feature_Embedding_From_CVPR_2023_paper.html">Paper</a>/<a href="https://github.com/wdzhao123/MetaFusion">Code</a></td>
        </tr>
        <tr>
            <td>CNN</td>
            <td>PSFusion</td>
            <td>Rethinking the necessity of image fusion in high-level vision tasks: A practical infrared and visible image fusion network based on progressive semantic injection and scene fidelity</td>
            <td>InfFus '23</td>
            <td><a href="https://www.sciencedirect.com/science/article/pii/S1566253523001860">Paper</a>/<a href="https://github.com/Linfeng-Tang/PSFusion">Code</a></td>
        </tr>
         <tr>
        <td>CNN</td>
        <td>LUT-Fuse</td>
        <td>LUT-Fuse: Towards Extremely Fast Infrared and Visible Image Fusion via Distillation to Learnable Look-Up Tables</td>
        <td>ICCV '25</td>
        <td><a href="https://arxiv.org/abs/2509.00346">Paper</a>/<a href="https://github.com/zyb5/LUT-Fuse">Code</a></td>
    </tr>
        <tr>
        <td>CNN</td>
        <td>PMAINet</td>
        <td>Progressive Modality-Adaptive Interactive Network for Multi-Modality Image Fusion</td>
        <td>IJCAI '25</td>
        <td><a href="https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/1791.pdf">Paper</a></td>
    </tr>
        <tr>
            <td>Transformer</td>
            <td>SwinFusion</td>
            <td>Swinfusion: Cross-domain long-range learning for general image fusion via swin transformer</td>
            <td>JAS '22</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/9812535">Paper</a>/<a href="https://github.com/Linfeng-Tang/SwinFusion">Code</a></td>
        </tr>
        <tr>
            <td>Transformer</td>
            <td>YDTR</td>
            <td>Ydtr: Infrared and visible image fusion via y-shape dynamic transformer</td>
            <td>TMM '22</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/9834137">Paper</a>/<a href="https://github.com/tthinking/YDTR">Code</a></td>
        </tr>
        <tr>
            <td>Transformer</td>
            <td>IFT</td>
            <td>Image fusion transformer</td>
            <td>ICIP '22</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/9897280">Paper</a>/<a href="https://github.com/Vibashan/Image-Fusion-Transformer">Code</a></td>
        </tr>
        <tr>
            <td>Transformer</td>
            <td>CDDFuse</td>
            <td>Cddfuse: Correlation-driven dual-branch feature decomposition for multi-modality image fusion</td>
            <td>CVPR '23</td>
            <td><a href="https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_CDDFuse_Correlation-Driven_Dual-Branch_Feature_Decomposition_for_Multi-Modality_Image_Fusion_CVPR_2023_paper.html">Paper</a>/<a href="https://github.com/Zhaozixiang1228/MMIF-CDDFuse">Code</a></td>
        </tr>
        <tr>
            <td>Transformer</td>
            <td>TGFuse</td>
            <td>Tgfuse: An infrared and visible image fusion approach based on transformer and generative adversarial network</td>
            <td>TIP '23</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/10122870">Paper</a>/<a href="https://github.com/dongyuya/TGFuse">Code</a></td>
        </tr>
        <tr>
            <td>Transformer</td>
            <td>CMTFusion</td>
            <td>Cross-modal transformers for infrared and visible image fusion</td>
            <td>TCSVT '23</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/10163247">Paper</a>/<a href="https://github.com/seonghyun0108/CMTFusion">Code</a></td>
        </tr>
        <tr>
            <td>Transformer</td>
            <td>Text-IF</td>
            <td>Text-if: Leveraging semantic text guidance for degradation-aware and interactive image fusion</td>
            <td>CVPR '24</td>
            <td><a href="https://openaccess.thecvf.com/content/CVPR2024/html/Yi_Text-IF_Leveraging_Semantic_Text_Guidance_for_Degradation-Aware_and_Interactive_Image_CVPR_2024_paper.html">Paper</a>/<a href="https://github.com/XunpengYi/Text-IF">Code</a></td>
        </tr>
        <tr>
            <td>Transformer</td>
            <td>PromptF</td>
            <td>Promptfusion: Harmonized semantic prompt learning for infrared and visible image fusion</td>
            <td>JAS '24</td>
            <td></td>
        </tr>
        <tr>
            <td>Transformer</td>
            <td>MaeFuse</td>
            <td>MaeFuse: Transferring Omni Features With Pretrained Masked Autoencoders for Infrared and Visible Image Fusion via Guided Training</td>
            <td>TIP '25</td>
            <td><a href="https://arxiv.org/pdf/2404.11016">Paper</a>/<a href="https://github.com/Henry-Lee-real/MaeFuse">Code</a></td>
        </tr>
    <tr>
        <td>Transformer</td>
        <td>Fusion with Language-driven</td>
        <td>Infrared and Visible Image Fusion with Language-Driven Loss in CLIP Embedding Space</td>
        <td>ACM MM '24</td>
        <td><a href="https://arxiv.org/abs/2402.16267">Paper</a>/<a href="null">Code</a></td>
    </tr>
    </tbody>
</table>

## 数据兼容方法(Data Compatible)
<table>
            <thead>
                <tr>
                    <th>Aspects<br>(分类)</th>
                    <th>Methods<br>(方法)</th>
                    <th>Title<br>(标题)</th>
                    <th>Venue<br>(发表场所)</th>
                    <th>Source<br>(资源)</th>
                </tr>
            </thead>
    <tbody>
        <tr>
            <td>Registration</td>
            <td>UMIR</td>
            <td>Unsupervised multi-modal image registration via geometry preserving image-to-image translation</td>
            <td>CVPR ‘20</td>
            <td><a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Arar_Unsupervised_Multi-Modal_Image_Registration_via_Geometry_Preserving_Image-to-Image_Translation_CVPR_2020_paper.html">Paper</a>/<a href="https://github.com/moabarar/nemar">Code</a></td>
        </tr>
        <tr>
            <td>Registration</td>
            <td>ReCoNet</td>
            <td>Reconet: Recurrent correction network for fast and efficient multi-modality image fusion</td>
            <td>ECCV ‘22</td>
            <td><a href="https://link.springer.com/chapter/10.1007/978-3-031-19797-0_31">Paper</a>/<a href="https://github.com/dlut-dimt/ReCoNet">Code</a></td>
        </tr>
        <tr>
            <td>Registration</td>
            <td>SuperFusion</td>
            <td>Superfusion: A versatile image registration and fusion network with semantic awareness</td>
            <td>JAS ‘22</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/9970457">Paper</a>/<a href="https://github.com/Linfeng-Tang/SuperFusion">Code</a></td>
        </tr>
        <tr>
            <td>Registration</td>
            <td>UMFusion</td>
            <td>Unsupervised misaligned infrared and visible image fusion via cross-modality image generation and registration</td>
            <td>IJCAI ‘22</td>
            <td><a href="https://arxiv.org/abs/2205.11876">Paper</a>/<a href="https://github.com/wdhudiekou/UMF-CMGR">Code</a></td>
        </tr>
        <tr>
            <td>Registration</td>
            <td>GCRF</td>
            <td>General cross-modality registration framework for visible and infrared UAV target image registration</td>
            <td>SR ‘23</td>
            <td><a href="https://www.nature.com/articles/s41598-023-39863-3">Paper</a></td>
        </tr>
        <tr>
            <td>Registration</td>
            <td>MURF</td>
            <td>MURF: mutually reinforcing multi-modal image registration and fusion</td>
            <td>TPAMI ‘23</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/10145843">Paper</a>/<a href="https://github.com/hanna-xu/MURF">Code</a></td>
        </tr>
        <tr>
            <td>Registration</td>
            <td>SemLA</td>
            <td>Semantics lead all: Towards unified image registration and fusion from a semantic perspective</td>
            <td>InfFus ‘23</td>
            <td><a href="https://www.sciencedirect.com/science/article/pii/S1566253523001513">Paper</a>/<a href="https://github.com/xiehousheng/SemLA">Code</a></td>
        </tr>
        <tr>
            <td>Registration</td>
            <td>-</td>
            <td>A Deep Learning Framework for Infrared and Visible Image Fusion Without Strict Registration</td>
            <td>IJCV ‘23</td>
            <td><a href="https://link.springer.com/article/10.1007/s11263-023-01948-x">Paper</a></td>
        </tr>
        <tr>
            <td>Attack</td>
            <td>PAIFusion</td>
            <td>PAIF: Perception-aware infrared-visible image fusion for attack-tolerant semantic segmentation</td>
            <td>ACMMM ‘23</td>
            <td><a href="https://dl.acm.org/doi/abs/10.1145/3581783.3611928">Paper</a>/<a href="https://github.com/LiuZhu-CV/PAIF">Code</a></td>
        </tr>
        <tr>
            <td>General</td>
            <td>FusionDN</td>
            <td>FusionDN: A unified densely connected network for image fusion</td>
            <td>AAAI ‘20</td>
            <td><a href="https://aaai.org/ojs/index.php/AAAI/article/view/6936">Paper</a>/<a href="https://github.com/hanna-xu/FusionDN">Code</a></td>
        </tr>
        <tr>
            <td>General</td>
            <td>IFCNN</td>
            <td>IFCNN: A general image fusion framework based on convolutional neural network</td>
            <td>InfFus ‘20</td>
            <td><a href="https://www.sciencedirect.com/science/article/pii/S1566253518305505">Paper</a>/<a href="https://github.com/uzeful/IFCNN">Code</a></td>
        </tr>
        <tr>
            <td>General</td>
            <td>PMGI</td>
            <td>Rethinking the image fusion: A fast unified image fusion network based on proportional maintenance of gradient and intensity</td>
            <td>AAAI ‘20</td>
            <td><a href="https://ojs.aaai.org/index.php/AAAI/article/view/6975">Paper</a>/<a href="https://github.com/HaoZhang1018/PMGI_AAAI2020">Code</a></td>
        </tr>
        <tr>
            <td>General</td>
            <td>U2Fusion</td>
            <td>U2Fusion: A unified unsupervised image fusion network</td>
            <td>TPAMI ‘20</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/9151265">Paper</a>/<a href="https://github.com/hanna-xu/U2Fusion">Code</a></td>
        </tr>
        <tr>
            <td>General</td>
            <td>SDNet</td>
            <td>SDNet: A versatile squeeze-and-decomposition network for real-time image fusion</td>
            <td>IJCV ‘21</td>
            <td><a href="https://link.springer.com/article/10.1007/s11263-021-01501-8">Paper</a>/<a href="https://github.com/HaoZhang1018/SDNet">Code</a></td>
        </tr>
        <tr>
            <td>General</td>
            <td>CoCoNet</td>
            <td>CoCoNet: Coupled contrastive learning network with multi-level feature ensemble for multi-modality image fusion</td>
            <td>IJCV ‘23</td>
            <td><a href="https://link.springer.com/article/10.1007/s11263-023-01952-1">Paper</a>/<a href="https://github.com/runjia0124/CoCoNet">Code</a></td>
        </tr>
        <tr>
            <td>General</td>
            <td>DDFM</td>
            <td>DDFM: Denoising diffusion model for multi-modality image fusion</td>
            <td>ICCV ‘23</td>
            <td><a href="https://openaccess.thecvf.com/content/ICCV2023/html/Zhao_DDFM_Denoising_Diffusion_Model_for_Multi-Modality_Image_Fusion_ICCV_2023_paper.html">Paper</a>/<a href="https://github.com/Zhaozixiang1228/MMIF-DDFM">Code</a></td>
        </tr>
        <tr>
            <td>General</td>
            <td>EMMA</td>
            <td>Equivariant multi-modality image fusion</td>
            <td>CVPR ‘24</td>
            <td><a href="https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Equivariant_Multi-Modality_Image_Fusion_CVPR_2024_paper.html">Paper</a>/<a href="https://github.com/Zhaozixiang1228/MMIF-EMMA">Code</a></td>
        </tr>
        <tr>
            <td>General</td>
            <td>FILM</td>
            <td>Image fusion via vision-language model</td>
            <td>ICML ‘24</td>
            <td><a href="https://arxiv.org/abs/2402.02235">Paper</a>/<a href="https://github.com/Zhaozixiang1228/IF-FILM">Code</a></td>
        </tr>
        <tr>
            <td>General</td>
            <td>VDMUFusion</td>
            <td>VDMUFusion: A Versatile Diffusion Model-Based Unsupervised Framework for Image Fusion</td>
            <td>TIP ‘24</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/10794610">Paper</a>/<a href="https://github.com/yuliu316316/VDMUFusion">Code</a></td>
        </tr>
             <tr>
           <td>General</td>
           <td>TC-MoA</td>
           <td>Task-Customized Mixture of Adapters for General Image Fusion</td>
           <td>CVPR '24</td>
           <td><a href="https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_Task-Customized_Mixture_of_Adapters_for_General_Image_Fusion_CVPR_2024_paper.html">Paper</a>/<a href="https://github.com/YangSun22/TC-MoA">Code</a></td>
        </tr>
        <tr>
           <td>General</td>
           <td>SHIP</td>
           <td>Probing Synergistic High-Order Interaction in Infrared and Visible Image Fusion</td>
           <td>CVPR '24</td>
           <td><a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Zheng_Probing_Synergistic_High-Order_Interaction_in_Infrared_and_Visible_Image_Fusion_CVPR_2024_paper.pdf">Paper</a>/<a href="https://github.com/zheng980629/SHIP">Code</a></td>
        </tr>
              <tr>
        <td>Transformer</td>
        <td>GIFNet</td>
        <td>One Model for ALL: Low-Level Task Interaction Is a Key to Task-Agnostic Image Fusion</td>
        <td>CVPR '25</td>
        <td><a href="https://openaccess.thecvf.com/content/CVPR2025/html/Cheng_One_Model_for_ALL_Low-Level_Task_Interaction_Is_a_Key_CVPR_2025_paper.html">Paper</a>/<a href="https://github.com/AWCXV/GIFNet">Code</a></td>
    </tr>
    </tbody>
</table>

## 面向应用方法(Application-oriented)
<table>
    <thead>
        <tr>
            <th>Aspects<br>(分类)</th>
            <th>Methods<br>(方法)</th>
            <th>Title<br>(标题)</th>
            <th>Venue<br>(发表场所)</th>
            <th>Source<br>(资源)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Perception</td>
            <td>DetFusion</td>
            <td>A detection-driven infrared and visible image fusion network</td>
            <td>ACMMM ‘22</td>
            <td><a href="https://dl.acm.org/doi/abs/10.1145/3503161.3547902">Paper</a>/<a href="https://github.com/SunYM2020/DetFusion">Code</a></td>
        </tr>
        <tr>
            <td>Perception</td>
            <td>SeAFusion</td>
            <td>Image fusion in the loop of high-level vision tasks: A semantic-aware real-time infrared and visible image fusion network</td>
            <td>InfFus ‘22</td>
            <td><a href="https://www.sciencedirect.com/science/article/pii/S1566253521002542">Paper</a>/<a href="https://github.com/Linfeng-Tang/SeAFusion">Code</a></td>
        </tr>
        <tr>
            <td>Perception</td>
            <td>TarDAL</td>
            <td>Target-aware dual adversarial learning and a multi-scenario multimodality benchmark to fuse infrared and visible for object detection</td>
            <td>CVPR ‘22</td>
            <td><a href="https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Target-Aware_Dual_Adversarial_Learning_and_a_Multi-Scenario_Multi-Modality_Benchmark_To_CVPR_2022_paper.html">Paper</a>/<a href="https://github.com/dlut-dimt/TarDAL">Code</a></td>
        </tr>
        <tr>
            <td>Perception</td>
            <td>BDLFusion</td>
            <td>Bi-level dynamic learning for jointly multi-modality image fusion and beyond</td>
            <td>IJCAI ‘23</td>
            <td><a href="https://arxiv.org/abs/2305.06720">Paper</a>/<a href="https://github.com/LiuZhu-CV/BDLFusion">Code</a></td>
        </tr>
        <tr>
            <td>Perception</td>
            <td>IRFS</td>
            <td>An interactively reinforced paradigm for joint infrared-visible image fusion and saliency object detection</td>
            <td>InfFus ‘23</td>
            <td><a href="https://www.sciencedirect.com/science/article/pii/S1566253523001446">Paper</a>/<a href="https://github.com/wdhudiekou/IRFS">Code</a></td>
        </tr>
        <tr>
            <td>Perception</td>
            <td>MetaFusion</td>
            <td>Metafusion: Infrared and visible image fusion via meta-feature embedding from object detection</td>
            <td>CVPR ‘23</td>
            <td><a href="https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_MetaFusion_Infrared_and_Visible_Image_Fusion_via_Meta-Feature_Embedding_From_CVPR_2023_paper.html">Paper</a>/<a href="https://github.com/wdzhao123/MetaFusion">Code</a></td>
        </tr>
        <tr>
            <td>Perception</td>
            <td>MoE-Fusion</td>
            <td>Multi-modal gated mixture of local-to-global experts for dynamic image fusion</td>
            <td>ICCV ‘23</td>
            <td><a href="https://openaccess.thecvf.com/content/ICCV2023/html/Cao_Multi-Modal_Gated_Mixture_of_Local-to-Global_Experts_for_Dynamic_Image_Fusion_ICCV_2023_paper.html">Paper</a>/<a href="https://github.com/SunYM2020/MoE-Fusion">Code</a></td>
        </tr>
        <tr>
            <td>Perception</td>
            <td>SegMiF</td>
            <td>Multi-interactive feature learning and a full-time multimodality benchmark for image fusion and segmentation</td>
            <td>ICCV ‘23</td>
            <td><a href="https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Multi-interactive_Feature_Learning_and_a_Full-time_Multi-modality_Benchmark_for_Image_ICCV_2023_paper.html">Paper</a>/<a href="https://github.com/JinyuanLiu-CV/SegMiF">Code</a></td>
        </tr>
        <tr>
            <td>Perception</td>
            <td>CAF</td>
            <td>Where elegance meets precision: Towards a compact, automatic, and flexible framework for multi-modality image fusion and applications</td>
            <td>IJCAI ‘24</td>
            <td><a href="https://www.ijcai.org/proceedings/2024/0123.pdf">Paper</a>/<a href="https://github.com/RollingPlain/CAF_IVIF">Code</a></td>
        </tr>
        <tr>
            <td>Perception</td>
            <td>MRFS</td>
            <td>Mrfs: Mutually reinforcing image fusion and segmentation</td>
            <td>CVPR ‘24</td>
            <td><a href="https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_MRFS_Mutually_Reinforcing_Image_Fusion_and_Segmentation_CVPR_2024_paper.html">Paper</a>/<a href="https://github.com/HaoZhang1018/MRFS">Code</a></td>
        </tr>
        <tr>
            <td>Perception</td>
            <td>TIMFusion</td>
            <td>A task-guided, implicitly searched and meta-initialized deep model for image fusion</td>
            <td>TPAMI ‘24</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/10480582">Paper</a>/<a href="https://github.com/LiuZhu-CV/TIMFusion">Code</a></td>
        </tr>
        <tr>
            <td>Perception</td>
            <td>SAGE</td>
            <td>Every SAM Drop Counts: Embracing Semantic Priors for Multi-Modality Image Fusion and Beyond</td>
            <td>CVPR ‘25</td>
            <td><a href="https://arxiv.org/pdf/2503.01210">Paper</a>/<a href="https://github.com/RollingPlain/SAGE_IVIF">Code</a></td>
        </tr>
             <tr>
            <td>Perception</td>
            <td>DCEvo</td>
            <td>DCEvo: Discriminative Cross-Dimensional Evolutionary Learning for Infrared and Visible Image Fusion</td>
            <td>CVPR ‘25</td>
            <td><a href="https://openaccess.thecvf.com/content/CVPR2025/html/Liu_DCEvo_Discriminative_Cross-Dimensional_Evolutionary_Learning_for_Infrared_and_Visible_Image_CVPR_2025_paper.html">Paper</a>/<a href="https://github.com/Beate-Suy-Zhang/DCEvo">Code</a></td>
        </tr>
        <tr>
            <td>Perception</td>
            <td>TDFusion</td>
            <td>Task-driven Image Fusion with Learnable Fusion Loss</td>
            <td>CVPR ‘25</td>
            <td><a href="https://openaccess.thecvf.com/content/CVPR2025/html/Bai_Task-driven_Image_Fusion_with_Learnable_Fusion_Loss_CVPR_2025_paper.html">Paper</a>/<a href="https://github.com/HaowenBai/TDFusion">Code</a></td>
        </tr>
        <tr>
            <td>Perception</td>
            <td>EVIF</td>
            <td>Event-based Visible and Infrared Fusion via Multi-task Collaboration </td>
            <td>CVPR ‘24</td>
            <td><a href="https://openaccess.thecvf.com/content/CVPR2024/html/Geng_Event-based_Visible_and_Infrared_Fusion_via_Multi-task_Collaboration_CVPR_2024_paper.html">Paper</a>/<a href="https://github.com/NetaPanda/EVIF">Code</a></td>
        </tr>
        <tr>
            <td>Perception</td>
            <td>MMAIF</td>
            <td>MMAIF: Multi-task and Multi-degradation All-in-One for Image Fusion with Language Guidance</td>
            <td>ICCV ‘25</td>
            <td><a href="https://arxiv.org/abs/2503.14944">Paper</a>/<a href="https://github.com/294coder/MMAIF">Code</a></td>
        </tr>
        <tr>
            <td>Perception</td>
            <td>CMFS</td>
            <td>CMFS: CLIP-Guided Modality Interaction for Mitigating Noise in Multi-Modal Image Fusion and Segmentation</td>
            <td>IJCAI ‘25</td>
            <td><a href="https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/2440.pdf">Paper</a>/<a href="https://github.com/SuGuilin/IJCAI2025-CMFS">Code</a></td>
        </tr>
        <tr>
            <td>Perception</td>
            <td>A²RNet</td>
            <td>A²RNet: Adversarial Attack Resilient Network for Robust Infrared and Visible Image Fusion</td>
            <td>AAAI ‘25</td>
            <td><a href="https://ojs.aaai.org/index.php/AAAI/article/view/32504">Paper</a>/<a href="https://github.com/lok-18/A2RNet">Code</a></td>
        </tr>
        <tr>
            <td>Perception</td>
            <td>SDSFusion</td>
            <td>SDSFusion: A Semantic-Aware Infrared and Visible Image Fusion Network for Degraded Scenes</td>
            <td>TIP ‘25</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/11014600">Paper</a>/<a href="https://github.com/Liling-yang/SDSFusion">Code</a></td>
        </tr>        
        <tr>
            <td>Perception</td>
            <td>S4Fusion</td>
            <td>S4Fusion: Saliency-Aware Selective State Space Model for Infrared and Visible Image Fusion</td>
            <td>TIP ‘25</td>
            <td><a href="https://ieeexplore.ieee.org/document/11062462">Paper</a>/<a href="https://github.com/zipper112/S4Fusion">Code</a></td>
        </tr>  
        <tr>
            <td>Perception</td>
            <td>FreeFusion</td>
            <td>FreeFusion: Infrared and Visible Image Fusion via Cross Reconstruction Learning</td>
            <td>TPAMI ‘25</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/11010882">Paper</a></td>
        </tr>  
        <tr>
            <td>Perception</td>
            <td>MulFS-CAP</td>
            <td>MulFS-CAP: Multimodal Fusion-Supervised Cross-Modality Alignment Perception for Unregistered Infrared-Visible Image Fusion</td>
            <td>TPAMI ‘25</td>
            <td><a href="https://ieeexplore.ieee.org/document/10856402">Paper</a>/<a href="https://github.com/YR0211/MulFS-CAP">Code</a></td>
        </tr>  
    </tbody>
</table>

#  评价指标(Evaluation Metric)
We integrated the code for calculating metrics and used GPU acceleration with PyTorch, significantly improving the speed of computing metrics across multiple methods and images.
You can find it at [Metric](https://github.com/RollingPlain/IVIF_ZOO/tree/main/Metric)

If you want to calculate metrics using our code, you can run:
```python
# Please modify the data path in 'eval_torch.py'.
python eval_torch.py
 ```

#  资源库(Resource Library)
##  融合(Fusion)

Fusion images from multiple datasets in the IVIF domain are organized in the following form: each subfolder contains fusion images generated by different methods, facilitating research and comparison for users.
```
Fusion ROOT
├── IVIF
|   ├── FMB
|   |   ├── ... 
|   |   ├── CAF # All the file names are named after the methods
|   |   └── ...
|   ├── # The other files follow the same structure shown above.
|   ├── M3FD_300 # Mini version of M3FD dataset with 300 images
|   ├── RoadScene
|   ├── TNO
|   └── M3FD_4200.zip # Full version of the M3FD dataset with 4200 images
```
You can directly download from here.

Download：[Baidu Yun](https://pan.baidu.com/s/1S6l-CUqE2nRPXeX2P_VScg?pwd=wgtn)


##  分割(Segmentation)

Segmentation data is organized in the following form: it contains multiple directories to facilitate the management of segmentation-related data and results.

```
Segmentation ROOT
├── Segformer
|   ├── datasets
|   |   ├── ... 
|   |   ├── CAF # All the file names are named after the methods
|   |   |    └──VOC2007
|   |   |         ├── JPEGImages # Fusion result images in JPG format
|   |   |         └── SegmentationClass # Ground truth for segmentation
|   |   └── ... # The other files follow the same structure shown above.
|   ├── model_data 
|   |   ├── backbone # Backbone used for segmentation
|   |   └── model # Saved model files
|   |        ├── ...
|   |        ├── CAF.pth # All the model names are named after the methods
|   |        └── ... 
|   ├── results # Saved model files and training results
|   |   ├── iou # IoU results for segmentation validation
|   |        ├── ...
|   |        ├── CAF.txt # All the file names are named after the methods
|   |        └── ... 
|   |   └── predict #Visualization of segmentation
|   |        ├── ...
|   |        ├── CAF # All the file names are named after the methods
|   |        └── ... 
|   └── hyperparameters.md # Hyperparameter settings
```

You can directly download from here.

Download：[Baidu Yun](https://pan.baidu.com/s/1IZOZU17CA6-zeR8zb1LW3Q?pwd=5rcp)

##  检测(Detection)
Detection data is organized in the following form:
it contains multiple directories to facilitate the management of detection-related data and results.

```
Detection ROOT
├── M3FD
|   ├── Fused Results
|   |   ├── ... 
|   |   ├── CAF # All the file names are named after the methods
|   |   |   ├── Images # Fusion result images in PNG format
|   |   |   └── Labels # Ground truth for detection
|   |   └── ... # The other files follow the same structure shown above.
|   ├── model_data 
|   |   └── model # Saved model files
|   |        ├── ...
|   |        ├── CAF.pth # All the model names are named after the methods
|   |        └── ... 
|   ├── results # Saved model files and training results
|   |   └── predict #Visualization of detection
|   |        ├── ...
|   |        ├── CAF # All the file names are named after the methods
|   |        └── ... 
|   └── hyperparameters.md # Hyperparameter settings
```

You can directly download from here.

Download：[Baidu Yun](https://pan.baidu.com/s/1mC3wTM1DjbBz5mIaDYJLDQ?pwd=a36k)
##  计算效率(Computational Efficiency)
- **FLOPS and Params**:
    - We utilize the `profile` function from the `thop` package to compute the FLOPs (G) and Params (M) counts of the model.
```python
from thop import profile

# Create ir, vi input tensor
ir = torch.randn(1, 1, 1024, 768).to(device)
vi = torch.randn(1, 3, 1024, 768).to(device)
# Assume 'model' is your network model
flops, params = profile(model, inputs=(ir, vi))
 ```

- **Time**:
    - To measure the Time (ms) of the model, we exclude the initial image to compute the average while testing a random selection of 10 image sets from the M3FD dataset, each with a resolution of 1024×768, on the Nvidia GeForce 4090. To eliminate CPU influence, we employ CUDA official event functions to measure running time on the GPU.

```python
import torch
  
# Create CUDA events
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
# Record the start time
start.record()
# Execute the model
# Assume 'model' is your network model
fus = model(ir, vi)   
# Record the end time
end.record()
```

