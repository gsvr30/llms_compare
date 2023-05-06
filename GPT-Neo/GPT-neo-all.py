import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Available GPT-NEO models
    gpt_neo_models = ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]

    # Load CoLA and SST-2 datasets
    cola_dataset = load_dataset("glue", "cola", split="validation")
    sst2_dataset = load_dataset("glue", "sst2", split="validation")

    for model_name in gpt_neo_models:
        print(f"Processing {model_name} model...")

        # Load GPT-NEO model and tokenizer
        model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Perform evaluations
        with open(f"{model_name.replace('/', '-')}_results.txt", "w") as result_file:
            for task_name in ["CoLA", "SST-2"]:
                result_file.write(f"{task_name} Zero-Shot Results:\n")
                evaluate(model, tokenizer, device,  cola_dataset if task_name == "CoLA" else sst2_dataset, result_file, task_name, zero_shot=True,few_shot=False)

                result_file.write(f"{task_name} One-Shot Results:\n")
                evaluate(model, tokenizer, device,  cola_dataset if task_name == "CoLA" else sst2_dataset, result_file, task_name, zero_shot=False, few_shot=False)

                result_file.write(f"{task_name} Few-Shot Results:\n")
                evaluate(model, tokenizer, device,  cola_dataset if task_name == "CoLA" else sst2_dataset, result_file, task_name, zero_shot=False, few_shot=True)

            result_file.write("\n")

def evaluate(model, tokenizer, device, dataset, result_file, task_name, zero_shot=True, few_shot=False):
    model.eval()

    true_labels = []
    predictions = []

    # Few-shot examples
    cola_examples = [
        ('He went to the store.', 'yes'),
        ('The children was playing.', 'no'),
        ('She is writing an essay.', 'yes')
    ]

    sst2_examples = [
        ('I love this movie!', 'positive'),
        ('The food was terrible.', 'negative'),
        ('This book is really boring.', 'negative')
    ]

    for batch in dataset:
        sentence = batch["sentence"]
        true_label = batch["label"]

        # Create a prompt for classification
        if task_name == "CoLA":
            if zero_shot:
                prompt = f"Is the following sentence grammatically correct? '{sentence}'"
            else:
                if few_shot:
                    example_text = "\n".join([f'- Sentence: "{ex[0]}"\n- Grammatically correct: {ex[1]}' for ex in cola_examples])
                else:
                    example_text = f'- Sentence: "He went to the store."\n- Grammatically correct: yes'
                prompt = f'Given the following example(s):\n{example_text}\n\nDetermine if the following sentence is grammatically correct:\n- Sentence: "{sentence}"\n- Grammatically correct: '
        elif task_name == "SST-2":
            if zero_shot:
                prompt = f"The sentiment of the sentence '{sentence}' is:"
            else:
                if few_shot:
                    example_text = "\n".join([f'- Sentence: "{ex[0]}"\n- Sentiment: {ex[1]}' for ex in sst2_examples])
                else:
                    example_text = f'- Sentence: "I love this movie!"\n- Sentiment: positive'
                prompt = f'Given the following example(s):\n{example_text}\n\nDetermine the sentiment of the following sentence:\n- Sentence: "{sentence}"\n- Sentiment: '
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

    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="weighted")
    recall = recall_score(true_labels, predictions, average="weighted")
    f1 = f1_score(true_labels, predictions, average="weighted")


    result_file.write(f"Accuracy: {accuracy}\n")
    result_file.write(f"Precision: {precision}\n")
    result_file.write(f"Recall: {recall}\n")
    result_file.write(f"F1 Score: {f1}\n")
    result_file.write("\n")


if __name__ == "__main__":
    main()