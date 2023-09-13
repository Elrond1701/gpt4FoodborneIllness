# gpt4FoodborneIllness

## OPENAI Account

gptfoodborneillness@gmail.com

## EMD

使用上下文学习的技术来提升效果，包括以下部分：

1. 对GPT给出了以下prompt。

```
I'm an excellent linguist. The task is to label XXXX entities in the given sentences. Below are some examples in the form of @@location1, location2, ...##.
```

2. 上下文学习使用的例子，具体可以分为三种

a. 不提供例子。

b. 从一个数据集中随机选取k个例子。

c. 从一个数据集中选取k个例子，但是选择的k个例子采用k近邻方式进行查询。具体方式为将待标注句子进行词向量映射，将一段英文/中文语句通过GPT提供的text-embedding-ada-002映射为一个高维词向量$x$，然后在数据集中寻找距离待标注句子最近的k个例子。使用`cosine_similarity`函数作为距离的度量。

3. 输入需要进行标注的句子进行达标。

## TRC

该部分的工作原理是进行词向量映射，将一段英文/中文语句通过GPT提供的text-embedding-ada-002映射为一个高维词向量$x$，然后比较$x$与“有相关性”（使用`This sentence indicates a possible foodborne illness incidient`作为比较对象）的距离和$x$与“无相关性”（使用`This sentence doesn't indicate a possible foodborne illness incidient`作为比较对象）的距离的差值。

若该差值大于0，则表明其距离“有相关性”的距离更小，确定为有相关性；若该差值小于0，则表明其距离“无相关性”的距离更小，确定为无相关性。

使用`cosine_similarity`函数作为距离的度量。

<!-- ## SF

not fully implement.

不过和EMD差不多。 -->

sample size与accuracy的图

in-context learning与few-shot
