# Lagent & AgentLego 智能体应用搭建笔记

## 为什么要有智能体

### 大语言模型的局限性

- **生成虚假信息**：大语言模型可能生成与现实不符的信息。例如，错误地回答“鲁迅和周树人不是同一个人”。
- **训练数据过时**：模型的训练数据可能无法反映最新的趋势和信息。
- **复杂任务中的错误**：在处理复杂任务时，模型可能频繁输出错误，影响信任度。

**经典文献**：
1. Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?". Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency. [DOI](https://dl.acm.org/doi/10.1145/3442188.3445922 )

> 大语言模型在处理信息时的局限性强调了我们需要智能体系统，这些系统能够利用最新的数据并在复杂任务中更加可靠。

## 什么是智能体

智能体具备三大核心功能：
1. **感知环境**：能感知环境中的动态条件。
2. **采取动作**：能采取动作影响环境。
3. **推理能力**：能运用推理能力理解信息、解决问题、产生推断和决定动作。

**经典文献**：
Hayes-Roth, Barbara (1995): An architecture for adaptive intelligent systems. In Artificial Intelligence 72 (1-2), pp. 329–365. DOI: 10.1016/0004-3702(94)00004-K. https://www.sciencedirect.com/science/article/pii/000437029400004K

> 智能体的多功能性使其在复杂动态环境中具有重要应用价值，例如自动驾驶和智能家居系统。

## 智能体组成

- **大脑**：作为控制器，负责记忆、思考和决策。
- **感知**：对外部环境的多模态信息进行处理。
- **动作**：利用并执行工具以影响环境。

**经典文献**：
Russell, S., & Norvig, P. (2021). Artificial Intelligence, Global Edition. (4th ed.). Pearson Education. https://elibrary.pearson.de/book/99.150005/9781292401171 https://elibrary.pearson.de/book/99.150005/9781292401171

> 智能体的组成模块化设计使其能够灵活适应不同应用场景，从而实现更高效的任务处理和决策能力。

## 智能体范式

### ReAct

- **输入** -> **选择工具** -> **执行工具** -> **结束条件** -> **结束**

### AutoGPT

- **输入** -> **选择工具** -> **人工干预** -> **执行工具** -> **结束条件** -> **结束**

### ReWoo

- **输入** -> **计划拆分** -> **DAG** -> **计划执行** -> **结束**

**经典文献**：
Yan Duan; Xi Chen; Rein Houthooft; John Schulman; Pieter Abbeel (2016): Benchmarking Deep Reinforcement Learning for Continuous Control. In International Conference on Machine Learning, pp. 1329–1338. Available online at https://proceedings.mlr.press/v48/duan16.html.

> 不同范式的智能体设计体现了在复杂任务中不同的决策路径和优化策略，提供了多种解决方案以应对不同的应用需求。

## Lagent & AgentLego

### Lagent

Lagent 是一个轻量级开源智能体框架，支持多种智能体范式（如 AutoGPT、ReWoo、ReAct）和多种工具（如谷歌搜索、Python 解释器等）。

### AgentLego

AgentLego 是一个多模态工具包，旨在像乐高积木一样，可以快速简便地拓展自定义工具，支持多个智能体框架（如 Lagent、LangChain、Transformers Agents）。

**经典文献**：
Brown, Tom; Mann, Benjamin; Ryder, Nick; Subbiah, Melanie; Kaplan, Jared D.; Dhariwal, Prafulla et al. (2020): Language Models are Few-Shot Learners. In Advances in Neural Information Processing Systems 33, pp. 1877–1901. https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html

**个人想法**：
Lagent 和 AgentLego 提供了强大的工具和框架，帮助开发者快速构建和部署智能体应用，具有广泛的应用前景和实用性。

## 实战一：Lagent 轻量级智能体框架

通过 Lagent，可以高效构建基于大语言模型的智能体应用，适用于多种场景和任务。

## 实战二：AgentLego 组装智能体“乐高”

AgentLego 提供了灵活的工具组装能力，用户可以根据具体需求定制智能体的功能和行为。
