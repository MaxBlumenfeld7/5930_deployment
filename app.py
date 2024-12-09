import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# Load base model from HuggingFace and instruction model from local directory
base_model_id = "HuggingFaceTB/SmolLM2-135M"
# instruct_model_path = "5930Final/Fine-tuning/smollm2_finetuned/05"  # Updated path
instruct_model_path = "MaxBlumenfeld/smollm2-135m-bootleg-instruct"


base_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
# instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_path, local_files_only=True)
instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_path)

base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
# instruct_model = AutoModelForCausalLM.from_pretrained(instruct_model_path, local_files_only=True)
instruct_model = AutoModelForCausalLM.from_pretrained(instruct_model_path)


def generate_response(model, tokenizer, message, temperature=0.5, max_length=200, system_prompt="", is_instruct=False):
    # Prepare input based on model type
    if is_instruct:
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nHuman: {message}\nAssistant:"
        else:
            full_prompt = f"Human: {message}\nAssistant:"
    else:
        # For base model, use simpler prompt format
        full_prompt = message
    
    inputs = tokenizer(full_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id  # Add padding token
        )
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if is_instruct:
        try:
            response = response.split("Assistant:")[-1].strip()
        except:
            pass
    else:
        response = response[len(full_prompt):].strip()
        
    return response

def chat(message, temperature, max_length, system_prompt):
    # Generate responses from both models
    base_response = generate_response(
        base_model, 
        base_tokenizer, 
        message, 
        temperature, 
        max_length, 
        system_prompt,
        is_instruct=False
    )
    
    instruct_response = generate_response(
        instruct_model, 
        instruct_tokenizer, 
        message, 
        temperature, 
        max_length, 
        system_prompt,
        is_instruct=True
    )
    
    return base_response, instruct_response

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# SmolLM2-135M Comparison Demo")
    gr.Markdown("Compare responses between base and fine-tuned versions of SmolLM2-135M")
    
    with gr.Row():
        with gr.Column():
            message_input = gr.Textbox(label="Input Message")
            system_prompt = gr.Textbox(
                label="System Prompt (Optional)",
                placeholder="Set context or personality for the model",
                lines=3
            )
            
        with gr.Column():
            temperature = gr.Slider(
                minimum=0.1, 
                maximum=2.0, 
                value=0.5, 
                label="Temperature"
            )
            max_length = gr.Slider(
                minimum=50, 
                maximum=500, 
                value=200, 
                step=10, 
                label="Max Length"
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Base Model Response")
            base_output = gr.Textbox(label="Base Model (SmolLM2-135M)", lines=5)
            
        with gr.Column():
            gr.Markdown("### Bootleg Instruct Model Response") 
            instruct_output = gr.Textbox(label="Fine-tuned Model", lines=5)
    
    submit_btn = gr.Button("Generate Responses")
    submit_btn.click(
        fn=chat,
        inputs=[message_input, temperature, max_length, system_prompt],
        outputs=[base_output, instruct_output]
    )

if __name__ == "__main__":
    demo.launch()