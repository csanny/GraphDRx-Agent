# GraphDRx-Agent

GraphDRx-Agent is a knowledge graph-based framework for interpretable drug repurposing candidate retrieval and evidence-grounded scientific review.

The framework consists of two main components: **GraphDRx**, which retrieves and prioritizes drug repurposing candidates from a heterogeneous biomedical knowledge graph, and **DRxAgent**, which performs structured multi-agent review of GraphDRx candidates using retrieved mechanistic and pharmacological evidence.


## Requirements
- Python 3.11
- Neo4j 5.26
- Ollama for local embedding and language-model inference
- A CUDA-capable GPU is strongly recommended for local LLM inference; experiments were conducted on an NVIDIA RTX A4500 GPU

## Citation
A manuscript describing GraphDRX-Agent has been submitted for publication.
