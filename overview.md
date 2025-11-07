# log and index

## 2025.9&10

### 2025.9.5-9.23

阶段任务：接触了解现有**参数高效微调方法(PEFT)**，阅读相关论文，读微调骨干模型**vision transformer**, **CLIP**的代码实现，做一些基础性的复现实验



阅读文献：

1. [**CLIP**](###CLIP)
   - Adapter方法
2. [**AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition**](###AdaptFormer)
3. [**Convolutional Bypasses Are Better Vision Transformer Adapters**](###convpass) 
4. [**COMPACTER: Efficient Low-Rank Hypercomplex Adapter Layers **](###COMPACTER)
   - LOAR方法
5. [**1% VS 100%: Parameter-Efficient Low Rank Adapter for Dense Predictions **](LoRA)
6. [**SCT: A Simple Baseline for Parameter-Efficient Fine-Tuning via Salient Channels **](###SCT)
   - VPT方法
7. [**Visual Prompt Tuning**](###VPT)
8. [**VISUAL PROMPT TUNING FOR TEST-TIME DOMAIN ADAPTATION**](###DePT) 



### 2025.9.23-10.14

阶段任务：重点学习VPT方法，研究VPT方法在CLIP模型上的运用，阅读针对CLIP模型的Prompt tuning方法的论文，初步了解这一方向研究的工作，baseline，benchmark等



阅读文献：

- VPT

1. [**LPT: LONG-TAILED PROMPT TUNING FOR IMAGE CLASSIFICATION**](###LPT)
2. [**Instance-aware Dynamic Prompt Tuning for Pre-trained Point Cloud Models**](###IDPT)
3. [**LION: Implicit Vision Prompt Tuning**](###LION)



- CLIP

1. [**Learning to Prompt for Vision-Language Models**](###CoOp) 
2. [**Conditional Prompt Learning for Vision-Language Models**](###CoCoOp) 
3. [**MaPLe: Multi-modal Prompt Learning**](###MaPLe)
4. [**Self-regulating Prompts: Foundational Model Adaptation without Forgetting**](###PromptSRC) 
5. [**PromptKD: Unsupervised Prompt Distillation for Vision Language Models**](###PromptKD)
6. [**ArGue: Attribute-Guided Prompt Tuning for Vision-Language Models**](###ArGue)
7. [**TextRefiner: Internal Visual Feature as Efficient Refiner for Vision-Language Models Prompt Tuning**](###TextRefiner)



### 2025.10.14-10.21

阶段任务：继续阅读与CLIP参数微调相关论文，以及读CoOp，CoCoOp方法的源代码以及实验部分，复现部分实验结果，熟悉代码运行的控制流，学习这些方法在代码上的具体实现，继续推进论文阅读



阅读文献：

1. [**Visual-Language Prompt Tuning with Knowledge-guided Context Optimization**](###KgCoOp)
2. [**Read-only Prompt Optimization for Vision-Language Few-shot Learning**](###PRO)
3. [**DePT: Decoupled Prompt Tuning**](###DePT)
4. [**What does a platypus look like? Generating customized prompts for zero-shot image classification.**](###CuPL)





### 2025.10.21-10.30

阶段任务：重点放在MaPLe和CoPrompt方法的代码实现上，以及论文中的实验部分，跑通其中几个小数据集的结果，对代码核心模块实现进行学习，以及尝试对部分模块进行调整



阅读文献：

1. [**CONSISTENCY-GUIDED PROMPT LEARNING FOR VISION-LANGUAGE MODELS**](###CoPrompt)
2. [**Prompt-based Adaptation in Large-scale Vision Models: A Survey**](###PAmethod)



## 2025.11

### 2025.11.2

**今日进展**

- 阅读文献：[**PLOT: Prompt Learning with Optimal Transport for Vision-Language Models.**](###PLOT)

  PLOT主要对多个文本Prompt特征与图像特征进行对齐的问题进行优化，该方法思路为：将文本与图像特征的余弦相似度计算 **优化为** 最优通道算法，以此提点，但是同时降低了训练和推理的效率。

  **可能改进点**：针对图文相似度拉近的算法继续进行优化；作为新方法用来提点的一个组件来使用

  

- 继续对CoPrompt的源代码进行阅读学习，此前已看完总体框架的构建，今日主要针对训练的过程。

  主要针对其与maple方法的不同点：

  1. 冻结模型蒸馏指导可学习模型的模块：冻结模块的输入为扰动输入，具体查看了**数据是如何处理的**
  2. Adapter模块：Adapter模块的加入很容易导致过拟合，代码中**使用一个置信度超参数进行了平衡(在论文中并没有提及)**
  3. LLM生成增强文本：并非将所有Prompt的embedding进行平均，而是在训练过程**动态随机选择某一条作为扰动输入**



**下阶段目标**：针对PromptSRC的源代码进行阅读学习，分析PromptSRC中的代码设计思路，PromptSRC与CoPrompt的差异点。继续阅读相关论文



### 2025.11.4

**今日进展**

- 看PromptSRC的源代码, 对于训练过程，从总体框架到细节的具体实现, 重点是与先前已经看过的代码项目的不同之处

  1. 将包含学习参数的编码器与冻结的编码器分开, **采用知识蒸馏的思想**, 用冻结的去指导包含可学习的, 具体实现即对其做L1范数损失调节
  2. 不同于MaPLe在设计Prompt learner类时候，在其中设计好VPTdeep，而是**在clip.py中的注意力模块部分实现VPTshallow**

  但忽略了自集成模块的实现方式。

  发现之前读的maple源代码中**被忽略的部分细节**：主要来自于clip.py文件中maple方法引入在attention层的跨模态交互部分



**下阶段目标**：查看高斯权重集成模块的具体实现，读PromptSRC的代码的测试阶段部分, 回头看maple方法之前被忽略的细节。



### 2025.11.6

**今日进展**

- [**Prompt-based Adaptation in Large-scale Vision Models: A Survey**](###PAmethod) 针对此前阅读的这篇有关视觉模型Prompt方法综述的文章进行总结, 将相关的新论文拉表汇总: 主要是 2025年pub的 PA相关文章；现阶段学习有关的CLIP或者多模态语言模型的文章 
- 查看了PromptSRC的权重自集成模块的具体实现



**下阶段目标**：看完PromptSRC的测试部分, 结束PromptSRC代码的阅读。进行下一个论文`TCP`和`BTP`的阅读。





### **CLIP**

**CLIP(Contrastive Language–Image Pre-training)**:  对比 语言-图像 预训练。一种强大的多模态模型，能够将图像和文本嵌入到同一个特征空间中。主要由图像编码器和文本编码器组成。核心是对比学习：图像嵌入与正确文本描述的相似度

CLIP没有分类头, 因此做的不是严格分类，这种图像文本对齐的方式，模型获得的对图像的语义性更强，在做迁移学习的时候的效果比其他模型要好得多





通过CLIP出现了prompt engineering这样的工程，这是在CLIP做迁移学习时候，由于CLIP是在一个由句子文本和图像的数据集上进行对比学习的，因此在迁移学习的时候，需要将数据集的标签加工成为与训练时相似的句子文本，从而达到zero shot迁移学习的效果，不仅如此，效果还十分优秀

![image-20251002185502402](./assets/image-20251002185502402.png)

文本编码通过单词与index的编码对实现，而图像通过一个网络

将图文进行embedding，经过encoder，然后将两者的共空间拉近。

在实际适配到下游任务过程中，需要进行prompt engineering，如 给模型一个"a photo of {}", 然后输入图像数据，模型根据prompt对{}进行图文匹配，此时{}的context会对产生结果影响

CLIP在0 shot情况下表现由于1,2 shot，这是因为用少的数据进行fine tuning会破坏原本CLIP学到的庞大的图文分布特征，使得CLIP过拟合到这一个图像小的图文分布特征内

few shot应用于下游任务时，利用few shot样本，使用交叉熵损失将预测逻辑与真实标签对齐，优化可学习的prompt



# Adapter

**Source**：[Awesome-Parameter-Efficient-Transfer-Learning](https://github.com/synbol/Awesome-Parameter-Efficient-Transfer-Learning?tab=readme-ov-file)

adapter: 1 2 6 9 12 13 15 17 20 26 30

prompt: 1 2 3 5 7 17 18 22

side tuning: 1 2 6 10 

reparameter tuning: 1 2 4 

unified tuning: 2 3



高效参数微调的背景：vit发展迅速，在大规模的数据集上进行pre-training，然后再针对下游任务进行参数微调成为SOTA的范式，但是随着vit规模增大，参数越来越多，全量参数微调成本变高，而且对于每个一个下游任务都需要储存一套模型参数，内存成本也增大。并且在小规模下游训练数据上微调大模型的权重时，容易发生过拟合。



在prompt，LoRA，adapter中，adapter参数量虽然是最大的，但是内存效率却是最高的，这里指的是显存占用，prompt额外增加的序列长度，LoRa则在反向传播时需要计算AB矩阵的梯度，需要存储矩阵乘法激活梯度图



这类文章，注重优化参数量，计算成本，内存效率等，因此大部分都会有**列公式计算参数量**等级的部分，并与其他baseline拉表做对比



### AdaptFormer

**AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition**

提出adapterformer轻量级模块，在transformer中即插即用，能扩展到多种视觉任务

问题：先前提出的linear probing方法(固定预训练模型参数，为每个任务微调特定的分类头)，性能太差，无法获取非线性特征

论文中的adapterformer主要与transformer中的MLP(多层感知机)层结合，组成adaptMLP瓶颈模块

<img src="./assets/image-20251013125928444.png" alt="image-20251013125928444" style="zoom:67%;" />



### convpass

**Convolutional Bypasses Are Better Vision Transformer Adapters** 

问题：首先提出的adapter是为了语言模型提出的，对于视觉任务考虑不足，因此提出借助卷积层的硬编码归纳偏置，设计adapter

使用卷积块作为旁路，使预训练vit适配下游视觉任务

说明了adapter模块放在mhsa模块前后的区别，放在mhsa模块之前的方式相当于对带有复杂qvk变换的mhsa进行微调

convpass的设计得益于卷积层的设计，具有针对视觉任务的归纳偏置，其模块与mhsa和mlp模块进行的都是残差连接，因此具有类似resnet的特性

做了fewshot的实验和domain generalization实验，这两类实验中convpass表现也很出色

domain generalization做了对clip的泛化实验，其中convpass插入在图像编码器中



### COMPACTER

**COMPACTER: Efficient Low-Rank Hypercomplex Adapter Layers**

**本文针对llm的参数高效微调提出，实验都基于nlp的模型和benchmark，针对的也是nlp任务。**

问题: 在此文中提到的prompt方法的短板在于依赖大模型，这是由于在输入层前面加上可学习向量只能激活和利用已有知识，而小模型往往缺乏大量泛化知识的能力

提出COMPACTER方法，结合adapter原理，将task-specific权重矩阵插入到预训练模型权重中，这个权重矩阵通过一种共享的权重和提出的快秩矩阵的kronecker积高效计算获得(PHM方法)。

在这个**Compact and Efficient Adapter Layers**中，投影层的权重矩阵W，将利用上述方法拆解为$A_i$ : 慢权重，跨所有adapter共享，可以捕捉到通用适配信息；$B_i$ : 快权重，每层adapter专属，学习该层的特异性适配信息，而且这个矩阵B将会采用低秩分解的方式表示，进一步减小参数量 (LPHM)

总结：就是把adapter的上下投影层更换成了这种LPHM层(低秩参数化超复乘层)

benchmark: GLUE and SUPERGLUE 基于T5-BASE模型，T5解码器

baseline: full-finetuning, adapter, PFEIFFER-ADAPTER，ADAPTER-LOWRANK，PROMPT TUNING，INTRINSIC-SAID ，ADAPTERDROP，BITFIT



# LoRand

### LoRA

**1% VS 100%: Parameter-Efficient Low Rank Adapter for Dense Predictions**

低资源下微调模型反而会降低特征理解能力，数据太少->反而训练不佳 (文中实验部分做了验证)

传统的adapter瓶颈结构，面对如语义分割等更具挑战性的密集预测任务重，性能无法达到full finetune的效果，文章提出了LoRand，通过**低秩合成**对adapter中的矩阵进行参数稀疏化，fc层中的投影矩阵由多个低秩矩阵乘积而得，使得fc层参数大大减少

相比adapter form，LoRand的参数量有所减少，而且性能有所增加

在adapter form中，其adapter结构会对传入x进行上投影，gelu层，下投影，而参数的主要开销即来自投影的过程$y=Wx+b$ 其中绝大多数参数存储在矩阵$W$中，因此LoRand对W进行了低秩合成，$W=\sum_{i=1}^{a}P^{T}_{i}K_iQ_i$  此处$PQ$为两个低秩矩阵，$K$为核矩阵用于控制LoRand的参数规模，并且$K$在上投影层和下投影层共享，由此，产生了两个超参数：核矩阵$K$的维度$\beta$ , 矩阵$W$的分支数$\alpha$ (矩阵$W$被分解为多个分支$W_i$)。

benchmark: COCO [28], ADE20K [62], and PASCAL VOC [14] 数据集

baseline : full finetuning, fixed(仅训练部分结构), adapter form



### SCT

**SCT: A Simple Baseline for Parameter-Efficient Fine-Tuning via Salient Channels**

显然这是一种通过Adapter改变了中间特征流的方法

问题：认为现有的vpt方法中针对任务(task-specific)的prompt通过监督学习获取，这种prompt是一组关于视觉图像的数值向量，不同于nlp的文本，人类不好理解，泛化能力较弱，换其他任务则需要探索新的最优prompt长度，重新训练prompt；而adapter form未考虑task-specific information，可训练的参数量还是太多

在vit或cnn中，特征的表示一般是[Batch, Channel, Height, Width] 或 [Batch, Seq_len, Channel]

论文的**SCT(显著通道调优)**基于channel bias这一现象而提出，针对adapter模块，筛选出**task-specific的通道**，实现高效微调，而且省去了下采样和非线性操作，降低了训练参数。**SCT的关键在于融入task-specific information**，不同任务就会选择不同的通道

<img src="./assets/image-20250913165330155.png" alt="image-20250913165330155" style="zoom: 50%;" />

**SCTM**即是文章所提出的模块，插入在mhsa层之后，mlp层之前，首先训练集输入backbone网络，提取到不同层的中间特征，利用文章中的一种算法CAIS来确定**显著通道**，保存这些channel的索引，微调阶段时，在每个transformer层插入SCTM，通过一个线性层对选中channel进行微调，其他channel则冻结

**方法步骤**：把下游任务的数据输入到预训练模型中做一次向前传播，提取出特征表示，从而能够知道哪些通道在目标任务中作用大(该过程不更新权重)，利用算法计算出贡献度最高的前K%的通道作为显著通道，然后在选中的通道维度上，加上SCTM模块，非显著通道则冻结。

benchmark: VTAB-1K

baseline: full finetuning, linear, bias, adapter form, LoRA, VPT, NOAH, SSF(对两个可学习因子进行缩放和偏移), adapter(传统的在transformer层注入MLP模块方式)

疑问：这里说的在通道上插入adapter是什么过程？



### PESF-KD

**Parameter-Efficient and Student-Friendly Knowledge Distillation**

针对knowledge distillation过程提出的加入adapter方法的高效参数方法。固定teacher model的参数，仅更新adapter参数，实现参数高效的对student的微调

问题：现有的knowledge distillation技术中，若冻结teacher model的参数，迁移效果差，若同时对teacher model进行参数微调，则太低效。因此提出**PESF-KD**框架，为teacher model设计一个adapter，实现高效知识迁移

![image-20250914183725361](./assets/image-20250914183725361.png)

在师生网络进行软标签知识迁移时，由于两个网络capacity不同，会导致迁移困难，而解决方法是调整温度参数来平滑teacher model的输出，但是手动调整难度大，而且teacher的标签平滑度由对student model影响巨大。因此文章实验了同时使用真实标签对teacher进行微调，能够使生成标签更平滑，即b图

文章同时还探索了不同参数和方法对多个指标的影响，以及可能的原因

dataset : CIFAR-100[23]、ImageNet[8]；GULE

网络组合 :ResNet56→ResNet20、ResNet110→ResNet32、VGG13→VGG8；ResNet56→VGG8

baseline : vanilla KD [17],probabilistic knowledge transfer(PKT) [31].  (在线蒸馏)

 knowledge distillation via collaborative learning (KDCL) [13], deep mutual learning (DML) [50], (离线蒸馏)

contrastive representation distillation (CRD) [41], relational knowledge distillation (RKD) [30] (表征蒸馏)



# VPT

### VPT

**Visual Prompt Tuning**

**对梯度流产生影响，以这些learnable Prompt为入口，训练过程梯度不会流入冻结的transformer层，而是流入这些Prompt中**

针对**ViT**提出的参数微调方法，**在输入空间引入可训练参数**。在transformer层输入中加上可学习的prompt，仅对这些prompt和线性head的参数进行微调。

在多种peft方法中，在数据量少的情况下，往往peft表现比full好，full在数据量少的情况下容易过拟合，但是在大数据量的情况下，full往往表现会好

VPT：图像数据->patches->embedding->input->backbone->head

**文章设计了deep和shallow两种vpt，前一种只在开始的input阶段加上prompt，第二种在每一层的transformer encoder层都加上了可学习的prompt**

实验框架：Vit，swin

benchmark：Fine-Grained Visual Classification tasks including CUB-200-2011 [75], NABirds [72], Oxford Flowers [59], Stanford Dogs [41] and Stanford Cars [23]

basesline: full, linear, Partial-k， Mlp-k，Sidetune，Bias，Adapter

针对不同下游任务需要调整prompt长度，这是需要调整的超参数

对于prompt插入位置和插入深度都进行了实验和探讨

插入位置：latent space: CLS之后-最佳(default) ，prompt与每一个patch的embedding叠加

pixel space(embed之前): 设置在patch前面，prompt与每一个patch叠加(同时对embed层也进行微调)

<img src="./assets/image-20250918153609035.png" alt="image-20250918153609035" style="zoom: 80%;" />

插入深度：插入层越多越深，效果越好，但是仅在深层进行插入，效果会变差，对vpt来说浅层的提示更重要

还在自监督预训练模型上进行了实验。监督和自监督预训练的Vit模型会存在差异，vpt表现不再最优

还将其思路应用在convnet上，即在输入图像填充可学习像素，但是表现并未达到最优



### DePT

**VISUAL PROMPT TUNING FOR TEST-TIME DOMAIN ADAPTATION** 

针对未见数据，源域(source domain)与目标域分布存在偏移，即real-world应用场景，文章称为test-time adaptation (TTA) problem，在TTA中全程无法访问源域的数据，只用目标域数据进行适配(“一次训练、适配任意未知测试分布” 的流程)。

问题：含**噪音的无监督学习**目标下有效调整**在源域训练的模型**；现有方法在无标注目标域数据下，无监督目标与主任务对齐度低或者提供的信号存在噪音。

提出了**Data-efficient Prompt Tuning (DePT)** : 通过visual prompt解决问题1；针对无标注目标域数据，采用预测伪标签，再利用记忆库中近邻数据点的软投票对伪标签进行优化，并设计**层级细粒度自监督正则化项**解决问题2。

离线：先利用无标注目标域数据更新参数再推理

在线：在更新参数的同时完成推理

在学习目标中，即需要最小化的函数，要优化两个损失：在线记忆库的伪标签损失、分层自监督损失(针对prompt的多层自监督约束)

模型在特定数据上进行训练即源域 	

方法：对ViT进行了**分阶段设计(m个transformer层组成一个阶段)**，在每个阶段输入插入prompt，因此有两个超参数，阶段数M，prompt的数量p，可对参数量进行调整控制，训练阶段prompt与模型一同在源域训练，适配阶段仅微调prompt。采用**师生模型与在线记忆库优化机制**生成伪标签，进行学习：教师模型不更新梯度，而是通过学生模型权重EMA更新，并且维护一个记忆库，学生模型在训练时更新速度快，生成伪标签含噪音，通过这个记忆库的软投票操作，使得生成伪标签更精准。

dataset: **VisDA-C**,**ImageNet-C**,**DomainNet-126**

model: ViT-B

baseline: 现有的多种UDA(基于convnet backbone，实验采用res101) 和TTA方法



### LPT

**LPT: LONG-TAILED PROMPT TUNING FOR IMAGE CLASSIFICATION**

针对long tail task，即类别分布不均匀导致的训练困难容易过拟合。全量微调时，会削弱泛化能力，容易过拟合。

提出**LPT**，prompt分为两组：shared prompt(针对整个数据集)、group-specific prompt(针对特征相似的样本组, 捕捉专属特征,增强细粒度判别能力)，设计两阶段的训练范式。

通过实验，可视化和计算证明，VPT微调后的ViT具备更强的特征判别能力，但原始VPT性能仍落后于SOTA

两阶段范式：**一阶段**：即VPT-deep，每层transformer都插入shared prompt。

​	**二阶段**：在ViT的前K个blocks秩只使用shared prompt，并且冻结。

​			引入group-specific prompt set，这个set里的元素为$\{ k_i, r^i \}$, k为key(用来匹配样本)，r为可训练			token(prompt本身)。

​			挑选group-specific prompt，通过phase1产生的class token输出作为query，利用余弦相似度来匹			配group-specific prompt set里所有keys中最接近的top-k。如果有多个则进行拼接

​			在后L-K个blocks中注入这些group-specific prompts。在注意力机制模块每个qvk都包含这些			prompt

​			训练分类器，group prompt，key

<img src="./assets/image-20250920170401405.png" alt="image-20250920170401405" style="zoom: 67%;" />

在二阶段训练采样过程还使用了Dual sampling strategy（双重采样策略）来平衡类别；非对称 Gaussian Clouded Logit Loss用来强化匹配group prompt的质量和样本分类区别能力

benchmark: Places-LT,CIFAR100-LT,iNaturalist 2018.



### IDPT

**Instance-aware Dynamic Prompt Tuning for Pre-trained Point Cloud Models**

将prompt tuning应用于预训练点云模型中

提出**IDPT**，不同于VPT的静态prompt，由于真实点云场景存在点缺失、点噪声、distribution，设计了一种动态prompt策略，捕捉每个点云的semantic prior features，根据不同输入生成自适应提示，插入在transformer的最后一层之前

方法：将最后第二层输出patch tokens经过动态prompt生成模块(三层Edgeconv->linear->maxpooling)得到动态prompt，然后与cls, patch tokens一起输入最后一层transformer



### LION

**LION: Implicit Vision Prompt Tuning**

受deep implicit model提出了**LION**，在参数冻结的pretrain backbone的两端插入两个**equilibrium implicit layers**，可以适配多种视觉模型。这两个层能够为下游任务输入数据生成专属的判别prompt。并且使用lottery hypothesis对这两个层进行剪枝，保留关键参数，防止过拟合。**即对视觉输入和输出表征提供prompt**。

输入图像经过平衡层，获取prompt与图像融合 -> backbone -> 输入图像经过平衡层，prompt再与经过backbone的特征融合 -> 全连接层

对可训练参数的robust train：对于参数剪枝方法，设置一个阈值t，对关键参数按照常规方法更新，非关键参数采用distinct optimization strategy，通过严格正则化被约束向0收敛

实验：

dataset: CIFAR10, CIFAR100, ImageNet100, Flower, Stanford Dogs, Stanford Cars, Clothing

baseline: retrain,  fine-tuning, head fine-tuning, adapter, Bias, VPT

model: res50. res101, Vit, swin transformer

还在长尾数据集上进行了测试，LION全部优于VPT，并且参数量更少。进行了few shot实验；消融实验证明两个平衡层设计和robust train设计的重要性和优越性



视觉语言大模型依赖于视觉概念是如何建模的   closed-set visual concepts， open-set visual concepts

对**梯度流**产生影响，以这些learnable Prompt为入口，训练过程梯度不会流入冻结的transformer层，而是流入这些Prompt中。总的，梯度只更新Prompt

对**信息流**产生影响，由于自注意力机制是动态输入依赖的，Prompt会改变注意力分布，使得注意力产生偏移，使得部分特征能够产生交流或者部分特征被抑制。总的，改变tokens的交互和语义聚焦

CLIP的ViT学习到了通用视觉概念等注意力模式，如物体的边缘，Prompt会导致注意力转移，如从狗转移到车的边缘特征。总的，Prompt会竞争注意力权重



# CLIP

**Source**：[Awesome-Prompt-Adapter-Learning-for-VLMs](https://github.com/zhengli97/Awesome-Prompt-Adapter-Learning-for-VLMs?tab=readme-ov-file)

2022年的**1 2**  

2023年的**1** **2** **6** **8** **10** 13  

2024年的3 5 6 **7** **8** **9** **10** 11 12 18   

2025年的**1** 2 4 6 7 15 16 18 19



注意力机制公式：$output = softmax(\frac{QK^T}{\sqrt{d_k}})V$ ,  $Q=XW_q,K=XW_k,V=XW_v$   , X为输入的tokens

在冻结的模型中，冻结的参数为权重矩阵$W_q,W_k,W_v,W_o$等

输入X(token embedding)由于加入了learning Prompt改变了，随之QKV也改变，

每一层的$QK^T$结构也改变，即patch与patch之间的内积变成了patch+Prompt和patch+Prompt的内积，softmax分布改变，从而影响整个注意力图，attention权重矩阵A改变了，即注意力被Prompt所引导了，

最终改变ViT的输出表征。

所以Prompt tuning技术，重点在于**Prompt该如何设计、该如何被训练**才能避免破坏内部的注意力，影响泛化性能。



在图像编码器中，特征最终会被聚合在[cls]这一token中；而文本编码器中，特征会被聚合在[EOS]中

训练过程，会有多个标签被嵌入[CLS]中，得到K个值，用其与真实标签的损失进行优化



**在相关实验中衡量的三个指标： base acc, novel acc, harmonic mean**



### CoOp

**Learning to Prompt for Vision-Language Models** 

问题：目前预训练视觉-语言模型在实际部署的时候所需要的prompt engineering需要耗时进行用词调优。经过预训练的视觉语言模型，仅需要调整prompt即可适配下游任务，但是prompt的context如何设置对准确率影响显著。

提出**CoOp**，将context建模为连续的可学习参数, 更新context token, 使其学会最优的类别描述，使得实现在zero shot或者few shot表现优秀。即不再使用自然语言作为上下文而是引入可学习向量

统一上下文（unified context）：所有类别共享一组context

类别特定上下文（class-specific context）：为每个类别学习一组特定的context token  -> 适用于部分细粒度类别



prompt的形式为**M**个token以及一个**class token(位置在结尾或中间)**，每个token都是与word embeddings维度相同的向量。然后将这个prompt输入文本编码器，与图像特征进行余弦相似度计算。对于class-specific context(CSC)来说每个类别的token相互独立



<img src="./assets/image-20250924171259979.png" alt="image-20250924171259979" style="zoom:50%;" />

训练过程与CLIP一致，采用交叉熵函数来最小化分类损失函数

**实验**

benchmark : 选用了11个公开的图像分类数据集作为基准

同CLIP进行了{1,2,4,8,16} shot训练；

以res50作为图像编码器；

context vector的初始化采用高斯分布$N(0,0.02^2)$; SGD; lr=0.002 ; scheduler.CosineAnnealingLR

epoch: 16/8 shot -- 200; 4/2 shot -- 100; 1 shot -- 50.   第一个epoch: lr=1e-5

baseline : 0 shot CLIP; linear probe 

在两个细粒度的数据集上，CoOp性能较差，对于噪声标签敏感。CSC方法在低于8 shot场景大多低于unified, 由于所需的参数量更多，所以需要更多数据进行训练

在CoOp中，context Prompt的初始化方式对结果几乎没有影响

**泛化实验** : 选取了一系列与ImageNet兼容的不同图像数据，证明CoOp对分布偏移仍然具有robust性能，并没有因为在特定数据分布上训练而导致产生虚假相关

context token长度探讨 : 实验证明token数量增多能提升性能。但实际在存在**分布偏移**场景需要进行权衡

与CLIP提出的prompt ensembling相比，也保持优势





### CoCoOp

**Conditional Prompt Learning for Vision-Language Models** 

问题：CoOp方法 模型学到的context无法泛化到unseen classes, 而对base classes存在过拟合。静态设计: 训练后context就固定了。 

将关注从特定类别集合转移到每个输入实例上

<img src="./assets/image-20250923141307384.png" alt="image-20250923141307384" style="zoom:67%;" />

提出**CoCoOp**, 在CoOp基础上额外学习一个轻量级神经网络(meta net)，**为每张图像生成一个输入条件token**，与可学习的context向量结合, 让prompt将输入的图像实例作为条件，而不是固定不变。这种通过实例进行优化的prompt对于类别偏移更加robust

meta net : linear-relu-linear, 输入为图像编码器的特征

![image-20251024163833680](./assets/image-20251024163833680.png)

与图像描述任务存在相似性 : 使用图像token来作为条件学习prompt，类似学习自然语言描述图像

benchmark：与CoOp相同      并且做16 shot实验

baseline: CoOp, 人工提示0 shot

model: CLIP(vit-b/16)

实验结果上，原本CoOp表现差的两个数据集，CoCoOp都表现优秀。在unseen classes上全部都有提升，在base classes性能却下降。相比人工提示的0 shot CLIP，展现更好的泛化性能

对于泛化实验，在单个数据集内的泛化实验，还进行了跨数据集迁移，领域泛化(分布外数据)。 

类别增量测试: 在训练中无法获取任何与新类别有关的数据

消融实验：初始化方式(word embedding优于随机)，context长度(8 tokens)



CoOp和CoCoOp的训练策略



build_dataloader



build_model : 加载原始的CLIP以便做修改。基于原本的CLIP构建包含可学习Prompt的CLIP



### MaPLe

**MaPLe: Multi-modal Prompt Learning**

现有prompt方式，仅关注单一分支(只关注文本或者只关注视觉)

提出MaPLe，为两个分支都设计prompt learning，强化视觉与文本表征的对齐。采用类似deep VPT的方式

同样针对泛化性，对新类别，新目标数据集，未见领域偏移泛化进行实验

<img src="./assets/image-20250926154133915.png" alt="image-20250926154133915" style="zoom:50%;" />

在文本和图像编码器上都加入可学习的prompt，并且两条分支的prompt通过一个耦合函数进行关联依赖，微调过程只调整context prompt和耦合函数

耦合函数的设计也是因为：图文输入是完全不同的，如果仅凭损失优化计算去拟合，计算代价相当大

language prompt上，采用类似deep VPT方式，在每个transformer层之前都引入新的可学习prompt，如果只引入一层则与CoOp一致；

vision prompt，也是采用类似deep VPT方式，但是vision prompt由language promp经过耦合函数投影获得；

耦合函数coupling function，为一个线性层，实现将维度$d_l$映射到另一个维度$d_v$

![image-20251028145838082](./assets/image-20251028145838082.png)

实验以CoCoOp为baseline，在相同的数据集上进行训练和各种泛化实验进行对比

消融实验：shallow设计，单视觉prompt，单语言prompt，视觉和语言prompt同时采用但不进行耦合

prompt深度(9) , prompt长度(过长会过拟合)，在分布偏移更大的数据集上表现会更优于CoCoOp

计算复杂度增加很少，参数量达到原来的2%



### KgCoOp

**Visual-Language Prompt Tuning with Knowledge-guided Context Optimization**

KgCoOp核心是**减小可学习Prompt与人工设计的Prompt的text embedding的差异**

将人工设计的'a photo of {class}'输入，同时优化一组可学习的Prompt，最小化两者的欧氏距离，在下游任务使用任务专属文本嵌入与视觉嵌入之间的对比损失，对可学习提示进行优化。

这种方式对训练时间影响小，与CoOp相当。

<img src="./assets/image-20251010143504955.png" alt="image-20251010143504955" style="zoom:67%;" />

即将人工设计的固定Prompt和可学习Prompt分别输入文本编码器，对两者特征进行欧式距离计算，作为目标函数的一个损失项，来引导Prompt的优化

实验：与CoCoOp类似，新类别和领域泛化测试，以及在附加材料中展示跨数据集结果



### PRO

**Read-only Prompt Optimization for Vision-Language Few-shot Learning**

注意到现有方法对**自注意力模块和内部表征**的负面影响，在few shot条件下影响更严重。

Prompt learning会通过注意力机制改变模型的隐藏表征。

提出PRO，利用masked attention防止预训练模型内部表征的偏移，即仅能从VLM的注意力交互中'只读'信息机制。使用read-only prompt，基于预训练模型的特殊tokens进行初始化。

1.同时对文本和图像输入都加上Prompt，文中把[CLS]和[EOS]称作为图像和文本编码器中特殊的token，还进行了一个**不同于CoOp的操作**，不是用Prompt去替换'a photo of'而是将'a photo of{class}'编码，然后再将learning Prompt拼接上去，这就是read-only机制

2.利用[CLS]和[EOS]，设置初始化方差，对learning Prompt进行初始化，而不是随机或按照'a photo of'的embedding进行初始化

3.通过注意力掩码机制限制注意力流动，防止原始特征被可学习提示嵌入破坏

4.成对评分函数：对编码器文本和视觉的输出进行相似度计算，基于其概率分布进行优化。文本和图像设置的Prompt长度相同，可以使模态对齐

<img src="./assets/image-20251012153852022.png" alt="image-20251012153852022" style="zoom:67%;" />

**简单说，read-only就是原本文本输入'a photo of {class}'和图像patch都正常进行embedding，然后再拼接上可学习Prompt，而不是替换掉原来的模版**

新类别、领域泛化

实验：CoCoOp。结果显示在PRO**平均**下在9个数据集都优于baseline，但是其实在基础类别上相比CoOp并没有优势(一定程度缓解了过拟合，但是基础类的性能没提升)，领域泛化实验表现则较优。

计算效率上确实相比CoCoOp要优





### PromptSRC

**Self-regulating Prompts: Foundational Model Adaptation without Forgetting** 

与MaPle同一个作者所发，由于maple是在few shot上训练，其deep结构的prompt以及图像和文本之间prompt的映射实际上产生了过拟合，因此本文通过正则化来解决过拟合的问题

针对模态对齐，从正则化角度进行研究

在prompt方法中，prompt的参数，即context embedding是在下游任务的loss上直接进行优化的(task-specific)，随着训练推进，embedding越来越偏向与训练集的分布和规律，忽略原本CLIP的更大分布，产生过拟合。因此针对prompt如何才能够**同时对task-specific和task无关进行学习**很关键

提出**PromptSRC**，对prompt的自正则化框架，来引导prompt的学习，提高性能和泛化能力。

<img src="./assets/image-20251001142341252.png" alt="image-20251001142341252" style="zoom: 67%;" />

promptSRC通过三个方式同时对prompt进行调节：

1.最大化prompt和CLIP特征之间的agreement，施加明确的一致性约束(保留冻结的预训练CLIP本身的分布表达能力)。 具体来说将可学习的特征与预训练CLIP的特征施加约束即**损失函数**，以及原本的优化prompt对下游任务的损失函数相加获得的作为最终损失函数  **(简单说：图像经过图像编码器获得一个表征，拼接上可学习的prompt也得到一个表征，然后对这两个表征进行一致性约束)** 

**通过三个正则化去优化了MaPLe中文本编码器和图像编码器可学习prompt之间耦合函数带来的过拟合**

2.自集成调节，对prompt采用weighted prompt aggregation technique，权重从高斯分布中采样，即对于每一轮的visual prompt和text prompt都**按照高斯分布进行加权平均集成**(减少针对任务训练过程中越来越偏向于训练集的分布和规律, 因此初期和后期的权重都相对较低) (产生两个超参数即高斯分布的均值和标准差)

代码层面实现: 当运行**本个epoch的最后一个batch**时候，将**当前模型参数进行高斯加权**，然后与上一个epoch的模型**权重进行叠加**，更新模型权重。

<img src="./assets/image-20251106191526277.png" alt="image-20251106191526277" style="zoom:67%;" />

<img src="./assets/image-20251106204253166.png" alt="image-20251106204253166" style="zoom:67%;" />

3.为text部分设计多样化label，**为特定类别定义多个文本label**，即对文本进行augmentation，构建多个prompt模版，减少与图像模态的差异(视觉一个类别对应多个图像，文本则一个类别只对应一个label)。在训练阶段为prompt ensembling feature和prompt feature进行正则化，推理时仍为prompt feature  

在文本输入端进行增强的部分: 直接将类别套上了Prompt emsembling的模版, 然后取平均获得一个融合版的文本向量

<img src="./assets/image-20251104152049693-1762432995806-33.png" alt="image-20251104152049693" style="zoom: 67%;" />

实验：

从基础类别到新类别的泛化、few shot实验{1,2,4,8,16}、分布外数据集的领域泛化、跨数据集评估

baseline: CoOp,CoCoOp     benchmark: baseline中使用的11类data sets

model: ViT-B/16 CLIP

消融实验：对于第一个组件，测试了对learnable prompt feature和CLIP feature之间用不同损失函数施加一致性约束的性能，结果标明L1效果优于余弦相似度和MSE；对第二个组件，测试了施加权重分配的性能；

对训练和推理计算成本也进行了分析，参数量相比maple减少77倍，训练时间低于CoCoOp，推理阶段相比单独的VL prompt结构没有额外开销

学习率相比于maple有所调整



### PromptKD

**PromptKD: Unsupervised Prompt Distillation for Vision Language Models**

将Prompt作为蒸馏器，**利用教师模型来指导Prompt更新**。将KD与PEFT相结合

通过教师的文本编码器预计算和存储文本特征，作为类别向量，即**复用预存储的文本特征**，减少额外计算开销。学生只需要学习如何使**生成的图像特征与这些固定的文本特征对齐**即可，固定文本的方式一定程度上保证了蒸馏的稳定性

训练完成的教师文本以类别向量的形式来供学生模型蒸馏使用，因此学生模型**不需要再进行文本分支相关计算。**

<img src="./assets/image-20251009144723670.png" alt="image-20251009144723670" style="zoom:50%;" />

在训练学生模型的时候，训练数据包含了已见类别和未见类别，但是不使用任何真实标签，因此学生模型见到了比教师模型更多的数据内容，性能得到了提升

<img src="./assets/image-20251009150304760.png" alt="image-20251009150304760" style="zoom:67%;" />

教师模型预训练 -> 存储文本编码器特征 -> 将未标签的图像输入师生的图像编码器生成特征 -> 学生的图像特征会通过一个投影层(两层MLP)来匹配存储的教师的文本特征维度 -> 将图像特征与存储的文本特征相乘 -> 进行KL散度计算蒸馏

推理阶段：存储的教师文本特征仍让复用，将图像经过学生图像编码器，结合文本特征进行预测

实验：教师模型采用PromptSRC的方法训练，学生模型采用transductive的few shot方式进行训练

消融实验：探究了不同蒸馏形式的影响，结果证明基于对数概率的蒸馏优于基于特征的蒸馏

蒸馏方法上，与全量微调学生模型、projector only、文本不共享进行对比

教师模型的预训练方法和教师模型的选择，教师模型的性能与学生模型性能成正相关

计算成本：由于是蒸馏模型，计算量少推理效率高。但是在训练时**蒸馏过程也存在开销**





### ArGue

**ArGue: Attribute-Guided Prompt Tuning for Vision-Language Models**

针对泛化能力:新类别和分布外上的表现。利用视觉属性进行引导。

1.直接在类别{class}前面加上可学习的prompt，利用LLM生成原始**视觉属性**(class包含的属性)进行对齐。

作者认为直接将class与可学习的tokens做拼接是导致过拟合的原因，由**LLM生成的类别的属性**，让模型通过属性来优化prompt，能够避免虚假关联(捷径: 将图像中其他内容与类别做关联)，并且由于多个类别可能共享这些属性，因此能提高泛化能力

2.提出属性采样，对LLM生成的这些属性进行筛选，筛除无用属性(视觉无关,语义无关)，保留语义有效的属性。

属性采样能降低计算成本(筛去无用属性)，还能提高模型精度

3.提出negative prompt，即包含无关属性，虚假关联等，模型针对这些prompt来引导微调

使用**属性去引导**prompt微调，negative prompt即例如 {class}的背景 ，其与class无关，模型对这些prompt输出表现应该**不做任何倾向**

避免模型太过依赖class的名称，使得属性的作用被削弱

正则化：最小化可学习prompt与文本提示(用a photo of的格式包含class和attribute的编码后特征)的交叉熵，最大化负属性与可学习prompt的交叉熵

<img src="./assets/image-20251003194037194.png" alt="image-20251003194037194" style="zoom: 67%;" />

实验：两种方法ArGue 和 ArGue-N(加入negative prompt进行正则化)

以CoCoOp的实验配置作为比对

baseline主要目标是**LASP**

属性采样为GPT生成量的20%，在消融实验中验证了少量高效的特点

加入N的方法不仅平均比ArGue好，而且对于虚假关联的数据集性能表现更好

通过可视化来解释属性和negative prompt的作用

实验并没有提及参数量，计算成本(额外计算属性正则化和负属性正则化)，但是要通过LLM生成属性，提高了推理成本，**依赖于外部知识**



### TextRefiner

**TextRefiner: Internal Visual Feature as Efficient Refiner for Vision-Language Models Prompt Tuning**

额外用于提取信息的模块会增大参数量和计算成本，影响推理速度等。文中的模块只是缓存了模型内部信息，推理速度仍然高效

目前在VLM的prompt learning主要以粗粒度方式学习，即prompt在类别之间是共享的

想要实现模态之间，更加细粒度、局部视觉特征的对齐。**借助图像编码器**来生成类别的细粒度局部特征，用于**加强文本prompt的性能**

提出**TextRefiner**，这是一个**即插即用的模块**(可以直接插入现有模型提高性能)，能够将细粒度语义的信息持续写入这个模块，这些信息与class embedding进行拼接，然后与文本的原始输出进行聚合和对齐，无需依赖外部知识

视觉网络中间层的输出生成精准的类别描述进行对齐

<img src="./assets/image-20251013133857503.png" alt="image-20251013133857503" style="zoom:67%;" />

三大部分：

1.局部缓存: 设置一个存储矩阵A，存储M个条目，将具有相似特征的tokens聚类到一个条目中，将经过ViT的token与A的条目做余弦相似度和softmax计算概率，概率最高的token分配到对应条目中

2.特征对齐：将经过ViT的局部特征通过两层MLP转换到文本嵌入空间，减少模态差距

3.特征聚合：将文本embedding与存储矩阵A的条目进行余弦相似度和softmax，获得的详细信息再与embedding结合，最后再通过残差连接将这个详细embedding和原始embedding聚合

损失函数增加语义损失和正则化损失

实验：同CoCoOp，泛化实验为新类别和跨领域

将TextRefiner插入CoOp和PromptKD进行实验，与PromptKD的组合实现了SOAT

推理阶段计算开销几乎不增多

消融实验：探讨了新增的超参数的作用和对模型准确率影响规律和原因



### DePT

**DePT: Decoupled Prompt Tuning**

和textrefiner类似，直接应用于现有基线方法上升点的

目前的方法存在BNT问题：基任务上的泛化能力 与 新任务的泛化能力 呈现负相关

主要原因是通道偏移，**绝大多数的特征通道被基础任务知识给占据**

DEPT核心是保留原始特征空间中的任务共享知识，将通道中的基任务专属知识与任务共享知识进行解耦，即把基础类别的特征 和 共享特征 进行隔离。

设计了一个CAT通道调整迁移头，即通过逐通道变换，将文本和图像特征通过可训练的缩放和偏移向量进行变换，将其合并后作为一个输入S，one-hot标签作为另一个输入Y，一起输入一个线性分类器，然后最小化两者(s,y)的交叉熵损失Lcat。即把基础类别特征隔离到CAT头，使用CAT头来专门进行基础类别任务

因此学习目标成为最小化Lcat和原本的图文对齐的损失Litm

<img src="./assets/image-20251016150101926.png" alt="image-20251016150101926" style="zoom:67%;" />

消融实验主要针对DEPT所设计的CAT头。损失项的权重参数，训练轮次，样本数量

计算成本上，在训练和推理时间都可忽略不计，增加的参数量主要来自类别数量





### CuPL

 **What does a platypus look like? Generating customized prompts for zero-shot image classification.**

开放词汇模型与LLM的结合，**借助LLM来生成Prompt**

一种完全的**zero shot方法**，不需要额外训练任何模型

方法：给指定数据集的每个类别生成Prompt，使用这些Prompt进行zero shot分类

简单说，直接问LLM '描述一下class或者class的识别特征是什么'(这部分文本需要人工进行) , 让LLM生成对应的描述，将这些描述输入文本编码器进行embedding之后取均值，然后就与图像的特征进行比对，获得分类结果

LLM与CLIP的文本编码器的语言空间存在不匹配问题





### CoPrompt

**CONSISTENCY-GUIDED PROMPT LEARNING FOR VISION-LANGUAGE MODELS**

结合CLIP-Adapter，maple，PromptSRC，强化了正则效果

对**可训练模型**和**预训练模型**的预测结果做一致性约束。两个组件：对扰动输入(即做数据增强)进行一致性约束；Adapter和Prompt两种方法的融合)

与KgCoOp的思路有相似之处，为了保证原有模型的泛化能力，对经过微调的输出和原本的输出做一致性约束

涉及知识蒸馏概念，让预训练模型通过一致性约束，实现冻结编码器到可学习编码器的知识蒸馏

文本分支，也利用LLM作为外部知识，让LLM对原有文本生成更加详细的描述，将其输出与可训练端进行一致性约束；图像分支，则通过图形增强，将其输出与可训练端进行一致性约束

<img src="./assets/image-20251024141414647.png" alt="image-20251024141414647" style="zoom:67%;" />

设计与**PromptSRC的多模态方法**十分相似，文章强调不同点在于：加入了Adapter模块，需要同时对Adapter进行调参；在文本端所做的数据增强由LLM生成，比起手工设计要更加细致；使用的损失计算方法也不同

创新点在于同时在文本和图像的可训练端的编码器输出位置都加上了Adapter(文中说先前研究中在双侧加上Adapter效果较差，但事实效果好)，Adapter的设计与常用的相同(两层线性层+非线性激活)

实验：在跨数据集上的表现比PromptSRC优秀，领域泛化上数据稍低

消融实验

![image-20251031161852023](./assets/image-20251031161852023.png)



### PLOT

**PLOT: PROMPT LEARNING WITH OPTIMAL TRANSPORT FOR VISION-LANGUAGE MODELS**

同样是针对文本端，利用LLM的知识来增强文本端的输入，提出使用多个Prompt进行学习。但问题在于Prompt与视觉特征之间的匹配，提出对应算法进行解决。

问题：文中提到直接使用多个Prompt分别与图像特征对齐，会导致Prompt特征收敛到同一个点：每个Prompt与图像特征对齐过程中，只是接近，而不是关注差异，导致最终都靠拢到一起

PLOT：用于实现局部视觉特征与多个文本提示词的对齐，实现细粒度的模态对齐。(传统欧氏距离是计算全局特征与Prompt的距离)

将OT(最优运输算法)应用于多个Prompt与图像特征之间，使用OT算法替换原来图文特征之间的余弦相似度计算

分为两阶段：1.用sinkhorn算法来优化计算过程 2.固定算法参数进行反向传播更新Prompt

消融实验：

PLOT计算方法相比余弦相似度使得推理速度有所下降，训练时间也有所延长

仅针对few shot的方法起作用



| method (shot=16) | lr     | epoch | n_ctx | prompt_depth |      | others                                                       |
| ---------------- | ------ | ----- | ----- | ------------ | ---- | ------------------------------------------------------------ |
|                  |        |       |       |              |      |                                                              |
| maple            | 0.0035 | 2     | 2     | 9            |      |                                                              |
| promptSRC        | 0.0025 | 20    | 4     | 9            |      | 损失项权重： $L_{SCL-image}=10$ $L_{SCL-text}=25$ 高斯加权集成: |



### PAmethod

**Prompt-based Adaptation in Large-scale Vision Models: A Survey**

给VPT和VP做区分, 将现有方法按照可学习提示、生成式提示与非可学习提示三类，以及从语义空间还是像素空间注入进行区分，将其统一为PA(Prompt-based Adapter)方法内的两大模块

VP针对像素级做Prompt。而VPT作用与内部的token和特征序列

探讨了这两种范式的差异，应用场景，基础理论的分析和问题挑战

| 方法                                                         | 分类                                             | 主要应用领域/场景        |
| ------------------------------------------------------------ | ------------------------------------------------ | ------------------------ |
| **AttrVP** (Chen & Wang, 2025)                               | **VP – Learnable / Pixel**                       | 通用视觉（参数高效适配） |
| **LøR-VP** (Jin et al., 2025)                                | **VP – Learnable / Pixel**                       | 通用视觉（参数高效适配） |
| **AdaPrompt** (Le et al., 2025)                              | **VPT – Learnable / Token**                      | 通用视觉（跨任务）       |
| **SG-VPT** (Ren et al., 2025)                                | **VPT – Learnable / Token**                      | 通用视觉；注重跨任务泛化 |
| **DVPT** (He et al., 2025a)                                  | **VPT – Generated / Token**                      | 医学分析                 |
| **DDFP** (Yin et al., 2025)                                  | **VP – Learnable / Pixel**                       | 医学图像分割             |
| **BiomedDPT** (Peng et al., 2025)                            | **VPT**                                          | 生物医学图像分类         |
| **OT-VP** (Zhang et al., 2025d)                              | **VP – Learnable / Pixel**                       | TTA                      |
| **ZoRI** (Huang et al., 2025a）                              | **VP – Fixed（输入线索）/ Pixel**                | 遥感实例分割             |
| **DynaPrompt** (Xiao et al., 2025f)                          | **VPT – Generated / Token**                      | 测试时自适应             |
| **DPCore** (Zhang et al., 2025e)                             | **VPT – Learnable / Token**（“动态提示核心集”）  | 持续测试时适应           |
| **Image-aware Dynamic Prompts for Anomaly Segmentation** (Zhang et al., 2025c) | **VPT – Learnable / Token**（“dynamic prompts”） | 异常分割                 |
| **IA Instance** (Li et al., 2025f)                           | VP-Learned                                       | 遥感实例分割             |
| **RLita**（Zhang et al., 2025b).                             | VPT                                              | 遥感图文对齐             |
| **Layerlink** （Zhu et al., 2025).                           | VPT                                              | 遥感; VLM                |
| **SPT** (Yang et al., 2025)                                  | VP-Learned                                       | 异常分割                 |
| **SAID** (Huang et al., 2025b)                               | VP-Learned                                       | 异常分割                 |
| **ClipSAM** (Li et al., 2025e)                               | VP-Generated                                     | 异常分割                 |
| **IAPAS** (Zhang et al., 2025c)                              | VP-Generated                                     | Industrial               |
| **UWSAM** (Li et al., 2025c)                                 | VP-Generated                                     | Underwater               |
| **GAPrompt** Ai et al. (2025)                                | VPT                                              | 点云                     |
| **PointLoRA** Wang et al. (2025a)                            | VP                                               | 点云                     |
| **RoadBench** Xiao et al. (2026; 2025a)                      | VP                                               | 自动驾驶                 |
| **PF3Det **(Li et al., 2025d).                               | VPT                                              | 三维检测 工业            |
| **MagicID**(Li et al., 2023a; 2025b).                        |                                                  | 压缩视频微调             |
| **TP-CLIP** (Gowda et al., 2025).                            | VP-generated CLIP                                | 视频动作识别             |
| **STOP** (Liu et al., 2025d).                                | VPT                                              | 开放领域视频理解         |
| **TEST-V** (Yan et al., 2025).                               |                                                  | 零样本视频分类           |
| **SEA-Net** (He et al., 2025b).                              | VPT                                              | 水下语义分割             |
| **OSLOPROMPT** Gupta et al., 2025)                           |                                                  | 领域偏移情景下健壮性     |



| 标题                                                         | 分类              | 应用                             |
| ------------------------------------------------------------ | ----------------- | -------------------------------- |
| Draw-and-understand: Leveraging visual prompts to enable mllms to comprehend what you want. | VP                | visual prompt引导MLLM            |
| Exploring the Transferability of Visual Prompting for Multimodal Large Language Models | VP                | visual prompt引导MLLM            |
| A remote sensing change detection network using visualprompt enhanced CLIP | VPT-enhanced CLIP | 时序推理任务                     |
| **ClipSAM**: CLIP and SAM Collaboration for Zero-Shot Anomaly Segmentation | VP-Generated      | 异常分割                         |
| @ **BIOMED-DPT**: DUAL MODALITY PROMPT TUNING FOR BIOMEDICAL VISION-LANGUAGE MODELS | VPT               | 生物医学图像分类                 |
| @ **Cibr**: Cross-modal information bottleneck regularization for robust clip generalization. | VPT-enhanced      | CLIP通用的跨模态正则化框架       |
| RoadBench: A Vision-Language Foundation Model and Benchmark for Road Damage Understanding **(RoadCLIP)** | VP                | 自动驾驶; CLIP编码器融入先验知识 |
| **BEV-CLIP**: Multi-modal BEV Retrieval Methodology for Complex Scene in Autonomous Driving | VPT-generated     | 自动驾驶; LoRA微调编码器适配领域 |



# 部分想法

CLIP-Adapter与传统Adapter不同：传统为在每个transformer层都加上一个Adapter瓶颈层；而CLIP-Adapter在编码器的输出端加上这个轻量级的Adapter然后与原始输出做残差连接

linear probe和CLIP-Adapter的区别：前者为在输出端训练一个全连接层，相当于只做一个线性映射去适配下游任务；而Adapter为一个MLP，包含非线性激活部分，并且包含残差连接，适配下游任务的效果更好



​	CLIP本身训练时候，主要就是对文本进行了prompt ensembling, 来提高语义泛化性，而对图像却没有做过多增强

​	在文本分支，文本的信息丰富度远不如图像，因此大部分引入额外数据信息的工作都是在文本分支上进行的

​	设计关于背景的Prompt

​	**裁剪(多尺度裁剪)，对局部信息进行操作** 再固定大小  多分辨率assemble 进行一定数据增强，再设计方法降低成本

​	**针对图像编码器，来做一个即插即用的模块，能够插入现有方法的CLIP模型，再度提升其性能**，动态Prompt

​	泛化能力: 要能够保留CLIP本身已经学习到的广泛的分布、通用表征

​	类似CoCoOp，再把文本信息编码进行融合

​	**增强模态对齐**   正则化:限制过拟合  损失函数

​	多组件

​	原理性上的问题为什么模型参数都被冻结了，输入仍然能够通过注意力机制改变模型的表征分布

​	通过Prompt改变输入，通过Adapter改变中间特征流

​	梯度流，信息流，attention重分配机制，模型表征如何被重构

# 草稿

参数流: 上下文向量随机初始化 ：按照维度创建空容器

![image-20251018195131471](./assets/image-20251018195131471.png)

随机高斯初始化

![image-20251018195328257](./assets/image-20251018195328257.png)



将上下文与classname拼接为Prompt

![image-20251018201234362](./assets/image-20251018201234362.png)

Prompt做tokenize并且embedding之后

![image-20251018201207615](./assets/image-20251018201207615.png)



epoch=3的图像编码器某个卷积层的权重参数：

![image-20251019202926093](./assets/image-20251019202926093.png)

可学习Prompt的参数：

![image-20251019203001174](./assets/image-20251019203001174.png)







# 早期的部分学习记录

基于扩散模型的风格化方法：

Mark Hamazaspyan and Shant Navasardyan. **Diffusionenhanced patchmatch: A framework for arbitrary style transfer with diffusion models.** In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 797–805, 2023

Tianhao Qi, Shancheng Fang, Yanze Wu, Hongtao Xie, Jiawei Liu, Lang Chen, Qian He, and Yongdong Zhang. **Deadiff: An efficient stylization diffusion model with disentangled representations**. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8693–8702, 2024.

 使用适配器：Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. **Adding conditional control to text-to-image diffusion models.** In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.

采用反演和共享注意力机制：Yuxin Zhang, Nisha Huang, Fan Tang, Haibin Huang, Chongyang Ma, Weiming Dong, and Changsheng Xu. **Inversion-based style transfer with diffusion models.** In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10146–10156, 2023.

测试时微调方法：Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. **Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation.** In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023.



**Attention is all you need**

提出transformer架构, self-attention机制

在transformer之前: 主流的**序列转导模型** 1.基于**CNN/RNN**模型 2. 通过**注意力机制**来连接**编码器-解码器**结构

transformer:  1.**仅基于注意力机制**,  2.摒弃了CNN/RNN



在网络结构中，还采用了多头注意力机制，通过将高维向量映射为多组低维向量，在不同的语义子空间中使用注意力机制，使得模型能够学习到更加详细丰富的表示



**Rein**

**针对VFM**，将其应用于**DGSS(domain generalized sematic segmentation)任务** 的 **微调**方法

现有微调方法, 主要是针对大语言模型的微调， 对于VFM，尤其在DGSS任务上受限

核心 : 一组随机初始化 与不同实例相联系的 token(可学习的)  点积  VFM的features  -> attention-like similarity map

一开始 该层的tokens序列$T_i$是随机初始化的，然后与该层的输出$f_i$直接进行点积，然后生成一个map，捕捉$f_i$ 与tokens的关联，然后再通过softmax将图像与实例进行对齐

在层与层之间插入了rein机制进行特征优化

相关工作中 : 域泛化语义分割 过去探索的方法有数据增强等 大部分应用于过时的骨干网络，对与VFM的不充分



# 初期探索

**AMPCliff: Quantitative definition and benchmarking of activity cliffs in antimicrobial peptides**

是做benchmark的，由于对cv方向感兴趣，之前看了一些cvpr的文章，其中也有做benchmark的，虽然这篇论文是生物信息方向，但是与我之前看的一篇视觉图像的benchmark文章有很多相似之处，同样是目前阶段的数据集的样本比较少，然后对其进行处理和扩充，同样也进行了划分，最终构建出一个新的基准数据集。然后根据现有的在这一方向上表现良好的模型作为baseline，进行一定调整，然后进行训练实验数据分析，得到表现优秀的模型，分析表现差的模型的问题，并为将来这一方向的研究指出优秀的方法或者解决某个不足的思路。

对现有数据集进行处理划分   (结合现有数据集，进行补充，构建一个更大的数据集，然后进行划分，做成基准数据集)。

对于某个问题现有的研究有一定局限，需要进行进一步探讨和研究分析，(定义一个评估指标？)。

对不同模型进行基准测试，得到不同模型在训练时候存在的问题，是什么影响了模型训练，可能可以如何解决。



**benchmark**

提供统一，客观的评估标准 : 例如构建一个benchmark的基准数据集，对某个方向的研究提供标准

为某个方向发展提供思路，明确发展方向 : 指出可能 可以进行研究的方面

揭示不同模型的缺陷 : 设计benchmark来测试不同的模型的脆弱性，更加强调泛化能力或稳健性，例如在ood数据上的实验

如何做 : **在某个benchmark上，提出某个方法来达到sota** (一定也要保证泛化能力，防止过度拟合(消融实验)、控制成本)

​		**提出一个新的benchmark (更加符合上述意义)**



**A Simple Graph Contrastive Learning Framework for Short Text Classification**

**Dual-level Mixup for Graph Few-shot Learning with Fewer Tasks**

这两篇论文都是做few shot方法的文章，一篇是文本分析，一篇是图像

第一篇做的是短文本的few shot方法，我认为是比较难的，文章提出一个新的框架，不依赖数据增强，从而保证了语义和信息上的准确性，表现相当优秀。第二篇是针对传统GNN依赖大量labeled samples进行训练，提出few shot的方法。

**A framework for global role-based author name disambiguation**

提出一个针对于作者名消歧的框架，对于当前的数据进行分析，采用聚类算法和映射算法。



**Efficient feature selection for pre-trained vision transformers**

研究ViT, 旨在优化目前阶段ViT存在的计算成本高问题，首先验证ViT在训练过程中存在不少的冗余组件(注意力头)，设计剪枝方法，针对自注意力头和mlp层隐藏神经元进行剪枝，并且实现效率与准确度优异的权衡，其优越性还体现在对数据吞吐量比现有方法要大

提出的**EffiSelecViT**方法，基于梯度下降算法的end to end选择框架，相比目前的剪枝方法，计算成本更低，而且不会损失性能。方法: 通过算法计算mhsa层的注意力头和mlp层的神经元重要性得分，通过mask的方法剪枝

通过消融实验，证明了部分注意力头存在无关冗余，并且去除这一部分注意力头不仅能够节约计算成本，甚至不影响性能或提升性能



**WST: Wavelet-Based Multi-scale Tuning for Visual Transfer Learning**

ViT模型微调，主要想法是将patch size缩小以获取更多的信息，然而会增加成本，因此提出WST，平衡性能和成本。

预训练任务和下游任务存在scale和granularity(粒度)上的差异

对于视觉模型patch size是指在embedding阶段会对图像进行分割，再进行position embedding

提出**WST**方法，在patch embedding阶段就通过不同尺度分割，直接生成多尺度特征，而且计算量少。通过wavelet transform替代下采样也避免了信息损失。