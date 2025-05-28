# SAP AI Agent Factory

![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

A modular, extensible framework for rapidly building AI-driven agents around SAP solutions using LangGraph. The SAP AI Agent Factory lets you define graph-based pipelines that connect SAP data sources, AI models, and business logic—so you can spin up intelligent assistants for invoice processing, payment recovery, classification, and more, in just a few lines of configuration.

## 🚀 Key Features

- **Graph-Based Pipelines**  
  Leverage LangGraph to declaratively wire together SAP connectors, AI/ML models, and orchestrator logic.

- **Pre-Built Agent Templates**  
  - **1-TTYD**: “Talk to Your Data” conversational interface  
  - **2-FailedPayments**: Automated identification and resolution of payment failures  
  - **3-Classification**: Document classification agent for invoices, orders, claims

- **SAP Integration**  
  Ready-to-use connectors for SAP S/4HANA, SAP ECC, and SAP Business One, enabling secure data retrieval and transaction execution.

- **AI/ML Model Support**  
  Plug in any OpenAI/LLM, Hugging Face, or custom model for tasks like summarization, classification, generation, and more.

- **Configurable & Extensible**  
  Turn on/off steps in the graph, swap models, or inject custom business rules via simple JSON/YAML definitions.

- **Observability & Logging**  
  Built-in tracing of pipeline execution, step-level logging, and error handling for seamless debugging and auditability.

---

## 📂 Repository Structure

```text
sap-ai-agent-factory/
├── 1-TTYD/                  # “Talk to Your Data” conversational agent
│   ├── config.yaml
│   └── app.py
├── 2-FailedPayments/        # Failed payment recovery agent
│   ├── config.yaml
│   └── agent.py
├── 3-Classification/        # Document classification agent
│   ├── config.yaml
│   └── classify.py
├── langgraph/               # Core LangGraph integration modules
│   ├── builder.py
│   └── connectors.py
├── examples/                # Additional sample pipelines
├── requirements.txt
└── README.md
```

---

## 🛠️ Getting Started

### Prerequisites

- Python 3.8 or higher  
- Access to an SAP system (S/4HANA, ECC, or Business One)  
- API credentials for your SAP instance  
- OpenAI API key (or credentials for any other LLM provider)

### Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/innovation-lab-belux/sap-ai-agent-factory.git
   cd sap-ai-agent-factory
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your environment variables**  
   ```bash
   export SAP_BASE_URL="https://my-sap-instance.example.com"
   export SAP_USER="USERNAME"
   export SAP_PASSWORD="PASSWORD"
   export OPENAI_API_KEY="sk-..."
   ```

---

## ⚙️ Configuration

Each agent lives in its own folder with a `config.yaml` defining:

```yaml
sap:
  base_url: "${SAP_BASE_URL}"
  user: "${SAP_USER}"
  password: "${SAP_PASSWORD}"

llm:
  provider: openai
  model: gpt-4
  temperature: 0.2

pipeline:
  - name: fetch_data
    type: sap_query
    params:
      endpoint: /sap/opu/odata/sap/API_INVOICE_PROCESS_SRV/...
  - name: ai_process
    type: llm_chain
    params:
      prompt_template: prompts/process_invoice.j2
  - name: postback
    type: sap_update
    params:
      endpoint: /sap/opu/odata/sap/API_INVOICE_PROCESS_SRV/...
```

Customize or extend steps to fit your business logic.

---

## 🚩 Usage Examples

### 1. Talk to Your Data (TTYD)

```bash
cd 1-TTYD
python app.py --config config.yaml
```

Start a web UI where business users can chat in natural language to query SAP data.

### 2. Failed Payments Recovery

```bash
cd 2-FailedPayments
python agent.py --config config.yaml
```

Automatically identify failed payments, generate recovery actions via LLM, and post updates back to SAP.

### 3. Document Classification

```bash
cd 3-Classification
python classify.py --config config.yaml
```

Classify incoming documents (invoices, purchase orders, claims) into categories and update records in SAP.

---

## 📈 Architecture

```text
[ SAP System ] ──> [ SAP Connector ] ──> [ LangGraph Pipeline ] ──> [ LLM / AI Model ]
                                     │
                                     └──> [ Custom Business Logic ]
                                         [ Observability & Logging ]
```

- **SAP Connector**: Handles authentication, batching, and OData/REST calls.  
- **LangGraph Pipeline**: Defines the directed acyclic graph of steps.  
- **AI Model**: Any LLM or ML model for NLP tasks.  
- **Business Logic**: Python modules that enforce rules, transform data, and handle errors.

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/awesome-agent`)  
3. Commit your changes (`git commit -m 'Add awesome agent'`)  
4. Push to the branch (`git push origin feature/awesome-agent`)  
5. Open a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 📬 Contact

Innovation Lab BELUX – [innovation-lab-belux@company.com](mailto:innovation-lab-belux@company.com)  
Feel free to reach out for questions, feature requests, or collaboration!
