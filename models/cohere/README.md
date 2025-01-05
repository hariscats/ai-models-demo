# Get Started with Cohere Rerank

This is a simple Python script that uses the **Cohere Rerank API** to rank a set of documents based on a user query. Follow these steps to get up and running quickly:

---

### **How It Works**
- You provide a query (e.g., *"Are there fitness-related perks?"*).
- The script takes a set of documents (stored in a `documents.json` file).
- It uses the Cohere Rerank model to find and rank the most relevant results.

---

### **Setup Instructions**

#### **1. Clone the Repo**
If this is part of a repo, clone it. Otherwise, copy the script and `documents.json` to your project folder.

#### **2. Install Dependencies**
You’ll need Python 3.7+ and a couple of libraries:
```bash
pip install cohere python-dotenv
```

#### **3. Get a Cohere API Key**
- Sign up at [Cohere](https://cohere.com/).
- Grab your free API key from the [API Key Dashboard](https://dashboard.cohere.com/api-keys).

#### **4. Set Up Your Environment**
Set your API key as an environment variable for security:
- On macOS/Linux:
  ```bash
  export COHERE_API_KEY="your_actual_api_key"
  ```
- On Windows:
  ```cmd
  set COHERE_API_KEY=your_actual_api_key
  ```

Alternatively, create a `.env` file in the project folder:
```
COHERE_API_KEY=your_actual_api_key
```

#### **5. Add Your Documents**
Update `documents.json` with the content you want to rank. Here’s an example format:
```json
[
    {"text": "Reimbursing Travel Expenses: Easily manage your travel expenses by submitting them through our finance tool."},
    {"text": "Health and Wellness Benefits: We offer gym memberships, yoga classes, and health insurance."}
]
```

---

### **Run the Script**
Let’s rank some documents! 🚀
```bash
python rerank.py
```

---

### **Example Output**
For the query **"Are there fitness-related perks?"**, you might see:
```
--- Reranked Documents ---

Rank: 1
Score: 0.92
Document: Health and Wellness Benefits: We offer gym memberships, yoga classes, and health insurance.

Rank: 2
Score: 0.30
Document: Reimbursing Travel Expenses: Easily manage your travel expenses by submitting them through our finance tool.
```