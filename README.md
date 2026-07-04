\# Hybrid RAG Q\&A Chatbot



A Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in a custom document corpus, built to investigate whether hybrid retrieval (dense + sparse) and reranking reduce hallucination compared to naive single-method retrieval.



🔗 \*\*\[Live Demo](https://hybrid-rag-app-chatbot-bfkhgjywtkycv5ypwvjvty.streamlit.app)\*\* — try the chatbot yourself, no setup required.

\---



\## Table of Contents



\- \[Motivation](#motivation)

\- \[Architecture](#architecture)

\- \[Dataset \& Corpus](#dataset--corpus)

\- \[Results](#results)

\- \[Key Findings](#key-findings)

\- \[Project Structure](#project-structure)

\- \[Setup](#setup)

\- \[Limitations \& Future Work](#limitations--future-work)

\- \[Tech Stack](#tech-stack)



\---



\## Motivation



Most RAG tutorials use plain vector similarity search (dense retrieval) and stop there. This project investigates a known gap in that approach: dense retrieval can miss exact keyword matches, while sparse (keyword) retrieval can miss semantic paraphrases.



This project builds and evaluates three progressively more sophisticated retrieval pipelines to quantify whether combining both methods — and adding a reranking layer — actually improves answer accuracy and reduces hallucination, rather than assuming it does.



\---



\## Architecture



User Question

|

v

Dense Retrieval (FAISS + MiniLM)   +   Sparse Retrieval (BM25)

|

v

Reciprocal Rank Fusion (weighted: dense x 2)

|

v

Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)

|

v

Top-k Context -> Groq LLM (Llama 3.1 8B) -> Answer



\*\*Components\*\*



| Component | Implementation |

|---|---|

| Dense retrieval | `all-MiniLM-L6-v2` sentence embeddings + FAISS similarity search |

| Sparse retrieval | BM25 keyword-based ranking |

| Fusion | Reciprocal Rank Fusion (RRF), weighted 2:1 toward dense retrieval |

| Reranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores fused candidates |

| Generation | Groq-hosted Llama 3.1 8B Instant, prompted to abstain on insufficient context |

| Interface | Streamlit chat UI with retrieved-context transparency panel |



\---



\## Dataset \& Corpus



\- \*\*Source:\*\* SQuAD 2.0 (Wikipedia passages + questions, including unanswerable questions)

\- \*\*Corpus size:\*\* 20 Wikipedia articles (Beyoncé, Chopin, Solar Energy, New York City, Buddhism, etc.), chunked into 21 passages

\- \*\*Evaluation set:\*\* 15 SQuAD answerable questions + 10 unanswerable questions + 5 hand-crafted paraphrase questions designed to stress-test semantic vs. keyword retrieval



\---



\## Results



| Metric | Naive RAG (dense only) | Hybrid RAG | Hybrid + Reranking |

|---|---|---|---|

| SQuAD accuracy | 93.3% (14/15) | 93.3% (14/15) | 93.3% (14/15) |

| Paraphrase accuracy | — | 60.0% (3/5) | 60.0% (3/5) |

| Overall answerable accuracy | — | 85.0% (17/20) | 85.0% (17/20) |

| Hallucination rate (unanswerable) | 0.0% | 0.0% | 0.0% |



\---



\## Key Findings



1\. \*\*Hybrid retrieval recovers cases where keyword search fails.\*\* For the query \*"When did Beyonce start becoming popular?"\*, BM25 alone ranked the correct passage last; RRF fusion (weighted toward dense) restored it to rank #1.

2\. \*\*Fusion weighting matters.\*\* Equal-weighted RRF initially hurt accuracy relative to naive dense retrieval (86.7% vs. 93.3%) by letting noisy BM25 matches dilute correct results. Weighting dense retrieval 2x recovered — and slightly exceeded — baseline performance.

3\. \*\*Reranking showed no measurable gain on this corpus.\*\* On a small (21-chunk), low-ambiguity corpus, the fused top-5 already contained the correct passage before reranking, so the cross-encoder had nothing left to improve. This suggests reranking's value is corpus-size- and noise-dependent rather than automatic — a candidate for future work on a larger corpus.

4\. \*\*Hallucination rate remained at 0% throughout\*\*, including on out-of-corpus questions (e.g., "What is the capital of France?"), across all three pipeline variants.



\---



\## Project Structure



├── app.py                          # Streamlit chat interface

├── prepare\_corpus.py               # Builds the SQuAD-derived document corpus

├── build\_index.py                  # Builds the FAISS dense index

├── build\_bm25.py                   # Builds the BM25 sparse index

├── hybrid\_retrieval.py             # Dense + sparse fusion (RRF)

├── naive\_rag.py                    # Baseline: dense-only RAG pipeline

├── hybrid\_rag.py                   # Hybrid (dense + sparse) RAG pipeline

├── hybrid\_rerank\_rag.py            # Hybrid + cross-encoder reranking pipeline

├── evaluate\_naive\_rag.py           # Quantitative evaluation: naive baseline

├── evaluate\_hybrid\_rag.py          # Quantitative evaluation: hybrid pipeline

├── evaluate\_hybrid\_rerank\_rag.py   # Quantitative evaluation: hybrid + rerank

├── requirements.txt

└── .env                            # GROQ\_API\_KEY (not committed)





\---



\## Setup



\*\*1. Clone this repository\*\*

```bash

git clone <repo-url>

cd hybrid\_rag\_qa

```



\*\*2. Create and activate a virtual environment\*\*

```bash

python -m venv venv

venv\\Scripts\\activate

```



\*\*3. Install dependencies\*\*

```bash

pip install -r requirements.txt

```



\*\*4. Create a `.env` file with a free Groq API key\*\*



Get a key at https://console.groq.com

GROQ\_API\_KEY=your\_key\_here



\*\*5. Build the indexes (first-time setup)\*\*

```bash

python prepare\_corpus.py

python build\_index.py

python build\_bm25.py

```



\*\*6. Launch the chatbot\*\*

```bash

streamlit run app.py

```



\---



\## Limitations \& Future Work



\- \*\*Small corpus (21 chunks):\*\* results, particularly the reranking null-result, may not generalize to larger, noisier document sets. A natural next step is scaling to 100+ documents.

\- \*\*Corpus completeness gaps:\*\* SQuAD contexts capture only single paragraphs per topic; some legitimate questions (e.g., "when did Beyoncé leave the group") aren't answerable from the stored passage even though the fact is publicly known. The system correctly treats these as "not in context."

\- \*\*Generation-level imprecision:\*\* in one case, the correct passage was retrieved but the model's answer paraphrased rather than named specific terms, indicating a prompt-engineering opportunity rather than a retrieval issue.



\---



\## Tech Stack



Python · FAISS · rank\_bm25 · sentence-transformers · Groq API (Llama 3.1) · Streamlit





\---



\## Author



\*\*Swarnali Ghosh\*\*



Built as part of an independent project exploring hybrid retrieval strategies for RAG-based Q\&A systems.

