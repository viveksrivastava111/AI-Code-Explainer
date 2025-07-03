import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@st.cache_resource
def load_model():
    model_name = "EleutherAI/gpt-neo-125M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)  # No torch_dtype or device_map
    return tokenizer, model


def generate_explanation(code_snippet: str, tokenizer, model) -> str:
    prompt = (
        "You are a helpful programming assistant. "
        "Explain what the following Python code does in plain English:\n\n"
        f"{code_snippet}\n\nExplanation:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return text.strip()

def main():
    st.title("🔍 Local AI Code Explainer")
    tokenizer, model = load_model()

    code_input = st.text_area("Enter your Python code here:", height=200)
    if st.button("Explain Code") and code_input.strip():
        with st.spinner("Generating explanation…"):
            explanation = generate_explanation(code_input, tokenizer, model)
        st.markdown("### 💡 Explanation")
        st.write(explanation)

if __name__ == "__main__":
    main()
