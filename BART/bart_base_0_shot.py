import torch
from transformers import BartForCausalLM, BartTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load GPT-NEO 125M model and tokenizer
    model = BartForCausalLM.from_pretrained('facebook/bart-base').to(device)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    # Set the padding token
    tokenizer.pad_token = tokenizer.eos_token

    # Load CoLA dataset
    cola_dataset = load_dataset("glue", "cola", split="validation")
    sst2_dataset = load_dataset("glue", "sst2", split="validation")

    # Perform zero-shot evaluation
    evaluate_zero_shot(model, tokenizer, device, cola_dataset, "CoLA")
    evaluate_zero_shot(model, tokenizer, device, sst2_dataset, "SST-2")

def evaluate_zero_shot(model, tokenizer, device, dataset, task_name):
    model.eval()

    true_labels = []
    predictions = []

    for batch in dataset:
        sentence = batch["sentence"]
        true_label = batch["label"]

        # Create a prompt for classification
        if task_name == "CoLA":
            prompt = f"Is the following sentence grammatically correct? '{sentence}'"
        elif task_name == "SST-2":
            prompt = f"The sentiment of the sentence '{sentence}' is:"
        else:
            raise ValueError("Invalid task name")

        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=len(inputs["input_ids"][0]) + 5)

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the result from the generated text
        result = generated.split()[-1].strip().lower()

        # Convert the result to the corresponding label
        if task_name == "CoLA":
            pred_label = 1 if result == "yes" else 0
        elif task_name == "SST-2":
            pred_label = 1 if result == "positive" else 0

        true_labels.append(true_label)
        predictions.append(pred_label)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="weighted")
    recall = recall_score(true_labels, predictions, average="weighted")
    f1 = f1_score(true_labels, predictions, average="weighted")

    # Print results
    print(f"{task_name} Zero-Shot Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")

if __name__ == "__main__":
    main()

