from fastapi import FastAPI, HTTPException
from google.cloud import storage
import os
import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from google.oauth2 import service_account


app = FastAPI()

# 從 Render 環境變數取得憑證資訊
credentials_json = os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
credentials_info = json.loads(credentials_json)

# 建立憑證物件
credentials = service_account.Credentials.from_service_account_info(credentials_info)

# 使用憑證建立 Google Cloud Storage Client
storage_client = storage.Client(credentials=credentials)

# 設定 Google Cloud Storage bucket 名稱
bucket_name = "tao-language-qna"
bucket = storage_client.bucket(bucket_name)

def download_if_not_exists(blob_name, file_path):
    if not os.path.exists(file_path):
        blob = bucket.blob(blob_name)
        try:
            blob.download_to_filename(file_path)
            print(f"下載成功: {file_path}")
        except Exception as e:
            print(f"下載失敗: {blob_name}，原因: {e}")
            raise

# 啟動時檢查並下載檔案
download_if_not_exists('tao_qna.csv', 'tao_qna.csv')
download_if_not_exists('tao_index.faiss', 'tao_index.faiss')

# 資料與索引載入
df = pd.read_csv('tao_qna.csv')
index = faiss.read_index('tao_index.faiss')
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

@app.get("/ask")
def ask(question: str, top_k: int = 3, threshold: float = 0.3):
    # 關鍵字搜尋
    keyword_results = df[df["question"].str.contains(question, case=False, na=False)]

    if not keyword_results.empty:
        return {
            "query": question,
            "results": keyword_results[["question", "answer", "reference"]].to_dict(orient="records")
        }

    # 語意搜尋
    query_vector = model.encode([question])
    distances, indices = index.search(query_vector, top_k)

    results = []
    similarity_scores = 1 - distances[0]

    for idx, sim_score in zip(indices[0], similarity_scores):
        if sim_score >= threshold:
            result = df.iloc[idx]
            results.append({
                "question": result["question"],
                "answer": result["answer"],
                "reference": result["reference"],
                "similarity": float(sim_score)
            })

    if not results:
        raise HTTPException(status_code=404, detail="未找到相關答案")

    return {
        "query": question,
        "results": results
    }
