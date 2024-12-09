import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

model_id = "./model"  # Point to the local folder
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)


def generate_response(message, temperature=0.7, max_length=200):
   prompt = f"Human: {message}\nAssistant:"
   inputs = tokenizer(prompt, return_tensors="pt")
   
   with torch.no_grad():
       outputs = model.generate(
           inputs.input_ids,
           max_length=max_length,
           temperature=temperature,
           do_sample=True,
           pad_token_id=tokenizer.eos_token_id
       )
   
   response = tokenizer.decode(outputs[0], skip_special_tokens=True)
   return response.split("Assistant:")[-1].strip()

with gr.Blocks() as demo:
   gr.Markdown("# SmolLM2 Bootleg Instruct Chat")
   
   with gr.Row():
       with gr.Column():
           message = gr.Textbox(label="Message")
           temp = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, label="Temperature")
           max_len = gr.Slider(minimum=50, maximum=500, value=200, label="Max Length") 
           submit = gr.Button("Send")
           
       with gr.Column():
           output = gr.Textbox(label="Response")
           
   submit.click(
       generate_response,
       inputs=[message, temp, max_len],
       outputs=output
   )

if __name__ == "__main__":
   demo.launch()