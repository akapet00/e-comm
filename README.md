# Recruitment assignment

Original recruitment assignment for the senior AI research scientist position is available in the `material` subdirectory.

## Prerequisites

First install `uv` by Astral: https://docs.astral.sh/uv/getting-started/installation/.

Then, set up the project by first cloning the remote repository on your local machine:

```bash
git clone https://github.com/akapet00/e-comm-assignment.git
```

and inside the local directory run:

```bash
uv sync .
```

This will set up the virtual environment (`.venv`) and install all the dependencies listed in `pyproject.toml`.

## Connecting to the Claude API

In order to use the Claude large language model (LLM) from within the LangChain as provided in the code examples:

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("anthropic:claude-3-7-sonnet-latest")
```

the Claude API key should be loaded during the execution.

The easiest way is to expose the API key in the `.env` file:

```
ANTHROPIC_API_KEY = "sk-ant-..."
```

Then the `.env` file should be pursed to load all the variables found as environment variables implicitly during the run time:

```python
from dotenv import load_dotenv

load_dotenv()
```

Otherwise, the instantiation of the Claude LLM can be rewritten as:

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-7-sonnet-latest",
    api_key="sk-ant-..."
)
```

but the API key can be provided manually.

The API key will be provided on demand: mailto:akapet00@gmail.com

## Research scenario introduction (+ notes)
- e-commerce clients in South East Asia (possibly dealing with different regional languages)
- clients use who use telecommunication APIs to send mass-scale newsletters via e-mail
- each newsletter campaign reaches hundreds of thousands + end-users, which may or may not interact with the products advertised in those campaigns
- some clients send newsletters with links to the products where you can track clicks, whereas other clients don’t (static catalogues)
- some of those clients (without trackable click system) need help to boost their revenue using the simple data

---

![alt text](media/e-commerce.png)

## Task (+ notes)
- build a conversational recommender system (e.g., as a WhatsApp chatbot) for up-selling and cross-selling of products that a specific client advertise 
- leverage the available historical newsletter data
- assume here is already a system available to deploy a conversational agent (e.g., a WhatsApp chatbot) and develop only the necessary APIs that take a textual input in and can return any kind of output (e.g., text, image, etc.)

---

![alt text](media/automation.png)

## Methods

### Data extraction

**Extracting structured product metadata.** One simple approach to extract structured product metadata (e.g., product names, categories, prices, availability) and embedded URLs from historical e-mail newsletter messages is by using HTML parsing libraries to traverse the DOM and locate the targeted entries. Typically, production sections use `<div>` or `<article>` HTML blocks with identifiable classes or data attributes. Within each block, fields with CSS selectors or XPath: the product name (often an `<h1>`/`<h2>` tag or a link text), price (by currency symbols like “¥” in Chinese locales), image URLs (`<img src="…">`), and product URLs (`<a href="…">`) can be extracted. Additionally, schema.org or JSON-LD product markup embedded in the HTML could also be used. Run:
```bash
uv run extract_data.py
```
to see the data extraction in action by using the 4th edition of the [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) library.

Additional parsing can also be done with the help of the regular expression or - as it's common thing to do nowadays - by copy-pasting the raw input to a LLM model and lets it do its magic.

**Extracting additional insights.** Historical campaign data on open rates and click-throughs can be used to infer popular product categories. For instance, product categories or keywords from past newsletters can be aggregated and clustered using topic modeling techniques (e.g., neural topic models [[Wu2024](#Wu2024)]) to uncover recurring themes. In the case of Chinese text, segmentation tools like [Jieba](https://github.com/messense/jieba-rs) should be applied prior to topic modeling. Additionally, clickstream data or user feedback logs can be analyzed to construct user profiles by tallying clicks or purchases per category. Old-school techniques such as tf-idf [[Spa72](#Spa72)] or transformer embeddings [[Vas2017](#Vas2017)] can then be applied to the newsletter content to align articles with user interests. To detect trending topics, time-series analysis such as applying moving averages to product or category click counts can be used to identify "hot" items. All this derived data can in turn serve us as KPIs.

## Research foundation

**Product metadata extraction (semi-structured data).** Kumar et al. introduced context-aware visual attention (CoVA) for extracting product fields from e-commerce webpages [[Kum2022](#Kum2022)]. They treat each HTML element as an object and combine DOM structure with visual features via attention [[Vas2017](#Vas2017)]. By using the collected e-commerce dataset (with manually labeled product title, price, image elements), authors claim that CoVA outperforms classical DOM-based extractors significantly (by measuring gini impurity-based importance of features).

Similarly, Potta et al. extracted attribute values (e.g., color, size) from product descriptions by constructing a graph of token co-occurrences and applying graph neural networks (GNNs) [[Pot2024](#Pot2024)]. Their GNN-based model showed significantly higher F1 than sequence tagging baselines.

Extending this direction, Zou et al. introduced a large open-source benchmark and dataset specifically focused on the more challenging task of implicit attribute value extraction, i.e., cases where attributes like material, fit, or style are not explicitly mentioned in product titles or descriptions but inferred from context (e.g., "perfect for winter" implying "thick" or "warm") [[Zou2024](#Zou2024)]. The benchmark includes 91k samples from diverse e-commerce categories and annotates both implicit and explicit values for 13 attributes. It also evaluates several LLMs (GPT-4, Claude, Gemini) and vision-language models (Gemini Pro Vision, GPT-4V, Claude 3 Sonnet) using prompt-based zero-shot and few-shot settings. Results show that even the strongest LLMs struggle with implicit cues, highlighting a significant gap in current systems’ ability to perform nuanced, multimodal reasoning. This work sets a new standard for evaluating product understanding in semi-structured settings and complements prior approaches by addressing a more subtle and semantically complex aspect of product metadata extraction.

**Topic modeling and user intent.** Rodrigues et al. used LLMs for topic/intent discovery by using the synthetic data where they expand a small set of user intents via hierarchical topic modeling and then prompt an LLM to generate synthetic utterances for each intent [[Rod2025](#Rod2025)]. Their results show that LLM-driven topic expansion created 278 intents (vs. 36 original) with coherent descriptions. Synthetic queries generated by GPT-style models improved few-shot intent classification. This highlights how LLM-based topic modeling can capture nuanced user intents even in sparse domains.

For multilingual contexts, especially involving Asian languages, some recent work has shown the limitations of general-purpose multilingual transformers and highlighted the need for region-specific adaptations. For example, Moghe et al. introduced MULTI3NLU++, a multi-domain, multi-intent dataset created by translating an English natural-language understanding (NLU) corpus into multiple languages (excluding Chinese), revealing that models like mBERT and XLM-R struggle with low-resource language generalization [[Mog2023](#Mog2023)].Conversely, for Chinese topic modeling and intent detection, recent state-of-the-art approaches increasingly rely on Chinese-specific pretrained models like [ERNIE 4.0](https://aimode.co/model/ernie-4/), a fourth iteration of the ERNIE generative AI model developed by the Chinese search giant, Baidu, and is designed to be a more powerful and capable natural language generation model.

In applied settings, models like [BERTopic](https://maartengr.github.io/BERTopic/index.html) have been successfully adapted to Chinese by leveraging sentence-transformer variants such as [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). So for our use case we would either leverage Chinese-specific models (ERNIE, [ChatGLM](https://chatglm.cn/main/alltoolsdetail?lang=en)) or multilingual ones ([mBERT](https://github.com/google-research/bert/blob/master/multilingual.md)) trained on Chinese corpora. Topic modeling in Chinese often requires segmentation where methods such as [BERTopic](https://maartengr.github.io/BERTopic/index.html) with Chinese embeddings or, in more traditional settings, the latent Dirichlet analysis on jieba-tokenized text. Simple demonstration of embedding-based recommender using `scikit-learn` is available by running

```python
uv run recommender.py
```

Again, contrary to the above simple example, in production, one would use a pre-trained Chinese embedding model (e.g., mBERT or ChatGLM) and a proper retrieval index (e.g., [FAISS](https://ai.meta.com/tools/faiss/)) to find nearest neighbors to the user’s query vector.

**Multilingual NLP and conversational recommender systems.** Liu et al. combined conversational recommender systems (CRS) with LLMs in e-commerce dialogues [[Liu2023](#Liu2023)]. While pure CRSs learn user preferences to recommend, they have limited language generation. On the other hand, LLMs generate fluent dialogue but lack domain knowledge. Authors propose two hybrid modes: "LLM assists CRS" (LLM generates candidate utterances) and "CRS assists LLM" (recommender feeds item IDs into LLM prompts). They show both collaborations can improve recommendation accuracy in pre-sales chats.

More broadly, recent surveys note a trend of using large pre-trained LMs (GPT, ChatGPT, Claude) to augment or endow recommenders with language understanding and generation [[Jan2023](#Jan2023)].

For multilingual scenarios, models like ChatGLM (Chinese-centric) or Google’s multilingual PaLM help handle Chinese queries and product descriptions. Frameworks such as LangChain or LangGraph can orchestrate LLM prompts, retrieval (via vector stores or KBs), and other pipelines in a chatbot. 

## Architectural proposal

- What specific components (e.g., Data pipelines, NLP extraction methods, embedding models, generative AI models, recommendation algorithms) are necessary?
- Which NLP tools or models are suitable for multilingual processing, particularly for Chinese? 
- How will the data extraction system integrate with a generative conversational recommender system (recommendation logic, personalization capabilities, 
conversational flow)?

TBA.

## Evaluation and metrics

- What robust success metrics can be used to evaluate the system in production (e.g., engagement rates, conversion rates, accuracy of metadata extraction, recommendation relevance, user satisfaction)?

TBA.

## Showcase

TBA.

## References

[<a id="Wu2024">Wu2024</a>] &emsp; Wu, X., Nguyen, T., and Luu, A.T. A survey on neural topic models: Methods, applications, and challenges. Artificial Intelligence Review 57, 18. (2024). https://doi.org/10.1007/s10462-023-10661-7

[<a id="Spa72">Spa72</a>] &emsp; Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval". Journal of Documentation. 28, 1 (1972). https://doi.org/10.1108/eb026526

[<a id="Vas2017">Vas2017</a>] &emsp; Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. Proceedings of the 31st International Conference on Neural Information Processing Systems. (2017). https://dl.acm.org/doi/10.5555/3295222.3295349

[<a id="Kum2022">Kum2022</a>] &emsp; Kumar, V., Ghosh, S., Jain, A., and Talukdar, P. CoVA: Context-aware visual attention for webpage information extraction. Proceedings of the Fifth Workshop on e-Commerce and NLP (ECNLP 5). (2022). https://doi.org/10.18653/v1/2022.ecnlp-1.11

[<a id="Pot2024">Pot2024</a>] &emsp; Potta, V., Krishna, S., and Saini, R. AttriSage: Product attribute value extraction using graph neural networks. Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: Student Research Workshop (2024). https://aclanthology.org/2024.eacl-srw.8/

[<a id="Zou2024">Zou2024</a>] &emsp; Zou, H., Samuel, V., Zhou, Y., Zhang, W., Fang, L., Song, Z., Yu, P., and Caragea, C. ImplicitAVE: An open-source dataset and multimodal LLMs benchmark for implicit attribute value extraction. Findings of the Association for Computational Linguistics: ACL 2024. (2024). https://doi.org/10.18653/v1/2024.findings-acl.20

[<a id="Rod2025">Rod2025</a>] &emsp; Rodrigues, K., Hegazy, M., J., and Naeem, A. Topic and intent discovery from sparse data with LLM-augmented synthetic utterances. (2025). https://doi.org/10.48550/arXiv.2505.11176

[<a id="Mog2023">Mog2023</a>] &emsp; Moghe, A., Agarwal, D., and Singh, S. Multi3NLU++: A multilingual, multi-intent, multi-domain dataset for natural language understanding in task-oriented dialogue. Findings of the Association for Computational Linguistics: ACL 2023. (2023). https://doi.org/10.18653/v1/2023.findings-acl.230


[<a id="Liu2023">Liu2023</a>] &emsp; Liu, Y., Zhang, W. Chen, Y., Zhang, Y., Bai, H., Feng, F., Cui, H., Li, Y. and Che. W. Conversational recommender system and large language model are made for each other in e-commerce pre-sales dialogue. Findings of the Association for Computational Linguistics: EMNLP 2023. (2023.) https://doi.org/10.18653/v1/2023.findings-emnlp.643

[<a id="Jan2023">Jan2023</a>] &emsp; Jannach, D. Evaluating conversational recommender systems. Artificial Intelligence Review 56. (2023). https://doi.org/10.1007/s10462-022-10229-x
