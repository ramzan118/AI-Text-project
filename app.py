import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load GODEL model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")

# Define the chatbot function
def generate_response(message, history):
    # Format dialog history
    dialog = [turn["content"] for turn in history if turn["role"] == "user"]
    dialog.append(message)
    dialog_text = " EOS ".join(dialog)

    # GODEL expects an instruction and context
    instruction = "Instruction: given a dialog context, respond appropriately."
    query = f"{instruction} [CONTEXT] {dialog_text}"

    # Tokenize and generate
    input_ids = tokenizer(query, return_tensors="pt").input_ids
    output_ids = model.generate(
        input_ids,
        max_length=128,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"role": "assistant", "content": response}

# Launch the chatbot
gr.ChatInterface(
    fn=generate_response,
    title="Muhammad’s GODEL Chatbot",
    description="A grounded chatbot powered by Microsoft's GODEL model.",
    type="messages"
).launch()
