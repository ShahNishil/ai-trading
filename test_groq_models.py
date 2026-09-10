import sys
sys.path.insert(0, r'D:\ai-trading')
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(r'D:\ai-trading\.env')
key = os.getenv("GROQ_API_KEY")
print("Key prefix:", key[:12] if key else "NONE")

client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1", timeout=15)
# List models
try:
    models = client.models.list()
    print("Available models:")
    for m in models.data:
        print(" -", m.id)
except Exception as e:
    print("List error:", e)

# Try each candidate
candidates = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "openai/gpt-oss-20b",
]

for model in candidates:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":"Say hi in one word"}],
            max_tokens=10,
            temperature=0.1,
        )
        print(f"OK {model}: {r.choices[0].message.content!r}")
    except Exception as e:
        print(f"FAIL {model}: {e}")
