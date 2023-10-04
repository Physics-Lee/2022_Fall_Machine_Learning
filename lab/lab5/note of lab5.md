# Observe

观察feature，可以发现他们大都都属于以下2种情况
* 连续：均匀、正态、指数、Γ、β
* 离散：以上的离散化

只有两个例外

* feature 69：双峰
* feature 73：0-1分布

但这样的离散特征很棘手，因为它们可能有2种情况：

* 第一种离散特征：把NBA球员的场均得分离散化成几个类别（比如0-10分、10-20分、20-30分、30分以上分别用0、1、2、3表示）
* 第二种离散特征：把黑色、红色、黄色、绿色的汽车编码成0、1、2、3（==这时候最好用one-hot/dummy编码，也可以用我在文章最后写的方法编码，像助教这样编码是不对的==）

这两者的区别在于，前者是有距离含义的，后者是没有距离含义的。前者中1和2比1和3更近，后者1和2、1和3一样近。

再举个例子，手写数据集中的X是第一种离散特征（灰度），y是第二种离散特征（数字）。

==我在算RELIEF和互信息时不知道该把feature 0、feature 3这样的特征算成离散特征还是连续特征，如果是前者我应该算成连续特征，如果是后者我应该算成离散特征，这可能是RELIEF和互信息的效果不算特别好的原因之一。==

但考虑到恐怕助教已经编错了，权衡之下我选择了23当unique value的threshold。



观察label，0、1、2、3大约各占四分之一，没有类别不平衡问题。



我观察到，无论选取多少特征，Logistic Regression和LASSO在训练集上的正确率都接近25%。这暗示我们特征和标签的关系是非线性关系。因此：

* **Least Square、Logistic Regression、LASSO、SVM不合适**
* **Pearson's r、PCA、LDA不合适**

# Pre-process

* 我选择用中位数代替nan

  * nan不能删去，否则没法预测test_feature中的nan，你总不能把test_feature中的某一行删掉

* 我选择用中位数代替outlier

  * 为了找出outlier，我用了Tukey test, this method is a combination of percentile and standard error.

  * IQR_index可以视为一个超参数，==经验是取3即可，本来我无需调参，但这次助教给的数据的异常值太过离谱，如果取3会漏掉一些正常值，而且这些正常值恰好是正态、指数等的尾巴，即，恰好是一些重要的正常值。因此我不得不调参==。
    * IQR can't be too large, because it is a kind of overfit and it will regard some outliers as normal value.
    * IQR can't be too small, because it will regard some normal value as outliers.
    * 调参中，我观察到
      * 5：feature 8、100少了很多正常值
      * 7：feature 37多了1个异常值
    * 权衡了一下，我选择了7。（其实我也可以选择15，因为从7到15，囊括的正常值比异常值多）

# Split

我遭遇的问题：

* 先划分还是先归一化？答：先划分再归一化，因为先归一化再划分会泄露验证集/测试集的信息。
* 先划分还是先特征选择/降维？答：先划分再特征选择/降维，因为先特征选择/降维再划分会泄露验证集/测试集的信息。
* 划分成训练集、测试集还是训练集、验证集、测试集？答：如果你需要调参，必须有验证集；如果你不需要调参（比如懒惰学习），有没有验证集无所谓。
  * ==也就是说，验证集的信息可以泄露在调参过程里，但是测试集的信息不可以泄露在调参过程中。==
  * 在我的做法中，RELIEF、互信息中也有一到两个超参数，用它俩进行特征选择的过程可以看成调参的一部分。决策树当然需要调参。在这三个调参过程中，我泄露了验证集的信息，但是没有泄露测试集的信息。


# Normalization

和lab1一样，我采用了经典的min-max-normalization。

# Feature Selection

特征选择和降维是解决维度诅咒的两个方法。这一小节介绍我采用的特征选择，下一小节介绍我采用的降维。

First, I use the number of unique value of each feature to split all features into the continuous and the categorical。unique value threshold也是一个超参数，权衡之下我选择了23。

接下来，我尝试了如下特征选择：

* 过滤法
  * variance
    * 不合适
    * 我进行min-max-normalization之后，发现120个特征中并没有方差显著比其它小的。事实上，这120个特征的分布都是均匀、正态、指数、Γ、β，没有那种特别集中的。
  * 卡方独立性检验
    * 不合适
    * 我在上面花了很长时间，但是我后来终于想清楚
    * 主要原因是，$\chi^2$ test对于大样本量会失效。原因是这时微小的变化也会被检测成$p-value < \alpha$
    * 次要原因是，大样本量是，小自由度所受的这种影响比大自由度的大。
    * 另一个次要原因是，也有很多人说$\chi^2$ test对于大自由度也会失效。但我暂时没有想明白为什么。我知道这时卡方分布会约为正态分布。
  * ==I mutual==
    * I mutual 不可以描述变量间有相互作用的情况。
      * 但是，我几乎可以确定，I mutual给出的排名前几的特征是有用特征的子集。
      * 而另一边，RELIEF不是万能的，有些有用特征RELIEF也挑不出来。
      * 于是，我把I mutual的前几名和RELIEF的前几名结合。后者多给一些名额，前者少给一些名额。
    * I mutual 可以描述非线性关系（当然可以！）
      * <img src="C:\Users\11097\AppData\Roaming\Typora\typora-user-images\image-20230118214146262.png" alt="image-20230118214146262" style="zoom:50%;" />
      * r_Pearson是线性的，上图显然有相关性，但是没有线性相关性(r_Pearson也等于0)
    * I mutual可以用于连续特征也可以用于类别特征
      * 类别特征时，算法很简单，用定义即可。
      * 连续特征时，划分bin不是一种好方法，[sklearn里用的是2篇2014年的论文提出的方法](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)。
    * I mutual算的是分布，所以是否归一化无影响（min-max-normalization不改变分布）。与之对比，RELIEF必须归一化之后再做。
  * ==RELIEF==
    * RELIEF可以描述变量间有相互作用的情况。
      * 据1992年的论文，这是这个算法的最大优点。
      * 相互作用的一个例子：
        * label: NBA球员的场均得分
        * feature 0: 5个位置分别为1、2、3、4、5
        * feature 1: 身高
      * 相互作用的另一个例子：1992年的论文里提出的异或
    * RELIEF可以描述非线性关系
    * RELIEF可以用于连续特征也可以用于类别特征
    * 我阅读了3篇论文，随后手动实现了RELIEF，因为RELIEF库写得不好。
    * n_neighbor是超参数
      * 不能太大，因为高维空间只有在局部可以用欧氏距离。
      * 不能太小，因为太小时噪声的影响会变大。
      * 10是一个经验数值，我试了一下，在lab5的数据集上，5、7、10的结果相近，于是我就选择10了。
  * [这篇CSDN博客](https://blog.csdn.net/qq_33431368/article/details/123080737)介绍了许多其他指标，但是都不合适，在此不详述。
    * **我最开始还傻傻地用Kendall tau，经过某位热心同学的指出后才改正错误。**
* 过滤法
  * RFE：不行
* 嵌入法
  * LASSO
    * 120个特征全扔进去，LASSO把所有系数都置为0。用过滤法保留24、84再LASSO，还是全0。这是因为
      * ==All the features are correlated.==
      * ==The data is not linear.==
        * 可能性更大，因为如果是上者，我试了一下，LASSO至少会保留一个不为0的coef
  * 决策树：见下方

# Dimension Reduction

我试了PCA、LDA、t-SNE、Kernelized PCE，都不行

# Train

我先用RELIEF保留了20个特征、互信息保留了10个特征，然后用这30个特征去决策树调。

调参顺序

* criterion
* max_depth
* min_samples_split
* min_samples_leaf
* grid，联合调2、3、4



第一轮调参过后，4折交叉验证的准确率达到了0.265125，==测试集准确率达到了0.268==。

严格来说，我对测试集进行验证之后就不可以进行任何操作，当然也不可以进行下一轮特征选择。

# hypothesis test

略


# Others

* 类别变量的编码方法
  * 直接把黄色、红色、绿色编码成1、2、3。（我在lab1里的做法，有时也很好用）
  * 用one-hot/dummy来编码，但缺点是会引入很多的feature，可能有1000个。
  * 用频率来编码，缺点是如果有的频率一样，模型会认为这两个类别一样。
  * 用平均数来编码，比如说，黄色用黄色车的价格的平均数编码。
* 过滤法，参考[这篇CSDN博客](https://blog.csdn.net/qq_33431368/article/details/123080737)
  * 单变量：略
  * 多变量
    * 自变量与自变量之间的相关性: 相关性越高，会引发多重共线性问题，进而导致模型稳定性变差，样本微小扰动都会带来大的参数变化，建议在具有共线性的特征中选择一个即可，其余剔除。
    * 自变量和因变量之间的相关性: 相关性越高，说明特征对模型预测目标更重要，建议保留。
* One vs Rest需要基学习器给出概率，但是只有Logistic Regression能给出概率
  * sklearn的解决方法：
    * Platt calibration, a way of transforming the score of a classification model into a probability distribution over classes
    * use the scores instead of probability

# 一些思考

* RELIEF+KNN失败的可能原因？答：欧氏距离在高维空间会失效，曼哈顿距离和余弦距离也不一定有效。我采用了曼哈顿距离，这个lab5是120维空间，它不一定有效。即，两个曼哈顿距离近的点不一定“真的”近。
* 互信息失败的可能原因？答：特征之间存在相互作用。
* 我尝试进行第三次特征选择，从30个里面再选出16个，但是我发现交叉验证准确率反而下降，这是为什么？答：可能是因为在这里中决策树的importance没有真正反映一个特征的importance。
