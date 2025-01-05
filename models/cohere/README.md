# **Cohere Rerank Project**

This project demonstrates how to use the **Cohere Rerank API** to rank documents based on relevance and evaluate the model's performance using metrics like **MRR** and **Precision@k**.

---

### **Project Structure**
```
cohere-rerank/
├── data/
│   ├── documents.json          # Documents for reranking
│   ├── labeled_dataset.json    # Labeled dataset for evaluation
├── src/
│   ├── rerank.py               # Script for reranking documents
│   ├── evaluate_model.py       # Script for evaluating model performance
├── .env                        # Environment variables (API key)
├── README.md                   # Project instructions
├── requirements.txt            # Python dependencies
```

---

### **Setup Instructions**

1. **Clone the Repository**  
   ```bash
   git clone <repo-url>
   cd cohere-rerank
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up API Key**  
   Add your **Cohere API Key** in a `.env` file:
   ```
   COHERE_API_KEY=your_actual_api_key
   ```

4. **Add Your Data**  
   - Update `data/documents.json` with documents for reranking.
   - Add labeled queries and documents in `data/labeled_dataset.json` for evaluation.

---

### **Usage**

#### **1. Rerank Documents**
Run the `rerank.py` script to rerank documents based on a query:
```bash
python src/rerank.py
```

#### **2. Evaluate the Model**
Run the `evaluate_model.py` script to calculate MRR and Precision@k:
```bash
python src/evaluate_model.py
```

---

### **Expected Output**
- **`rerank.py`**:
  Ranks documents based on relevance to the query.
  ```
  --- Reranked Documents ---
  Rank: 1
  Score: 0.92
  Document: We offer gym memberships and yoga classes.
  ```

- **`evaluate_model.py`**:
  Displays model performance metrics:
  ```
  Mean Reciprocal Rank (MRR): 0.8333
  Precision@3: 0.6667
  ```