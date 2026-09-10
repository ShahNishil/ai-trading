import sys, os, json
sys.path.insert(0, r'D:\ai-trading')
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv(r'D:\ai-trading\.env')
key = os.getenv("GROQ_API_KEY")
client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1", timeout=20)

for model in ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "groq/compound", "groq/compound-mini", "qwen/qwen3.6-27b"]:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You are a senior equity analyst. Respond with JSON only: {\"symbol\":\"TEST\",\"action\":\"BUY\",\"confidence\":0.75,\"entry_price\":100,\"stop_loss\":97,\"target\":104,\"timeframe_hours\":48,\"reasoning\":\"test\"}"},
                {"role":"user","content":'{"symbol":"RELIANCE","close":1268.7,"indicators":{"rsi":39.7,"macd_hist":-3.28,"ema9":1293,"ema21":1299}}'},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        c = r.choices[0].message.content or ""
        print(f"OK {model}: {c[:400]!r}")
    except Exception as e:
        print(f"FAIL {model}: {e}")
