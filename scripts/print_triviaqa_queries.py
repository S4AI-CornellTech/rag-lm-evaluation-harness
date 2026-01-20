import argparse
import sys


def format_triviaqa(doc):
    """Format: Question: {question}?\nAnswer:"""
    question = doc.get("question", "")
    return f"Question: {question}?\nAnswer:"


def format_nq_open(doc):
    """Format: Q: {question}?\nA:"""
    question = doc.get("question", "")
    return f"Q: {question}?\nA:"


def format_mmlu(doc):
    """Format: {question}\nA. {choice0}\nB. {choice1}\nC. {choice2}\nD. {choice3}\nAnswer:"""
    question = doc.get("question", "").strip()
    choices = doc.get("choices", [])
    if len(choices) < 4:
        # Fallback for missing choices
        return f"{question}\nAnswer:"
    formatted = f"{question}\n"
    formatted += f"A. {choices[0]}\n"
    formatted += f"B. {choices[1]}\n"
    formatted += f"C. {choices[2]}\n"
    formatted += f"D. {choices[3]}\n"
    formatted += "Answer:"
    return formatted


# Task configurations: (hf_dataset_path, hf_config_name, default_split, formatter_fn)
TASK_CONFIGS = {
    "triviaqa": ("trivia_qa", "rc.web.nocontext", "train", format_triviaqa),
    "nq_open": ("nq_open", None, "train", format_nq_open),
    "mmlu": ("cais/mmlu", "all", "test", format_mmlu),
}


def main(argv=None) -> int:
    try:
        import datasets
    except Exception:
        print("Error: This script requires 'datasets'. Install: pip install datasets")
        return 1

    parser = argparse.ArgumentParser(
        description="Standalone: print TriviaQA, NQ-Open, or MMLU queries"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=list(TASK_CONFIGS.keys()),
        default="triviaqa",
        help="Task name: triviaqa | nq_open | mmlu (default: triviaqa)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split (default: uses task default). Common: train|validation|test|dev",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of documents to print",
    )
    parser.add_argument(
        "--mmlu_subject",
        type=str,
        default=None,
        help="For MMLU: specific subject (e.g., 'abstract_algebra'). Default: 'all'",
    )

    args = parser.parse_args(argv)

    # Retrieve task configuration
    hf_dataset, hf_config, default_split, formatter = TASK_CONFIGS[args.task]

    # Override config for MMLU if subject specified
    if args.task == "mmlu" and args.mmlu_subject:
        hf_config = args.mmlu_subject

    # Load dataset
    if hf_config:
        ds = datasets.load_dataset(hf_dataset, hf_config)
    else:
        ds = datasets.load_dataset(hf_dataset)

    # Determine split
    split = args.split if args.split else default_split
    if split not in ds:
        print(
            f"Split '{split}' not available. Available: {list(ds.keys())}. Using first available."
        )
        split = list(ds.keys())[0]

    docs = ds[split]
    print(f"# Task: {args.task} | Dataset: {hf_dataset} | Config: {hf_config} | Split: {split}")

    count = 0
    for idx, doc in enumerate(docs):
        query = formatter(doc)
        print(f"{idx}: {query}")
        count += 1
        if args.limit is not None and count >= args.limit:
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
