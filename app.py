import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import gradio as gr

print("กำลังดาวน์โหลดโมเดล GGUF...")
model_path = hf_hub_download(
    repo_id="BlossomsAI/Qwen2.5-Coder-7B-Instruct-Uncensored-GGUF",
    filename="q4_k_s.gguf"
)

print("กำลังโหลดโมเดลเข้าสู่ระบบ...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

def predict(message, history):
    prompt = f"System: You are Qwen2.5-Coder, an AI programming assistant.\n"
    for human, assistant in history:
        prompt += f"User: {human}\nAI: {assistant}\n"
    prompt += f"User: {message}\nAI:"

    output = llm(
        prompt,
        max_tokens=512,
        stop=["User:", "System:"],
        echo=False
    )
    
    response = output['choices'][0]['text'].strip()
    return response

demo = gr.ChatInterface(
    fn=predict,
    title="Qwen2.5-Coder-7B Uncensored",
    description="ระบบแชท AI รันโมเดล GGUF ควบคุมผ่าน GitHub และ Hugging Face",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
