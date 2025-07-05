import os
import sys
import json
from tqdm import tqdm
import requests
import time
from pathlib import Path
from typing import Dict, List

from .config import EVAL_MODEL, TEMPERATURE

def get_anscheck_prompt(task, question, answer, response, abstention=False):
    """Generate evaluation prompt based on task type"""
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError(f"Task type {task} not implemented")
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt

def query_openai_with_retry(prompt: str, max_retries: int = 3) -> str:
    """Query LLM with retry logic using the configured client factory"""
    import asyncio
    import sys
    from pathlib import Path
    
    # Add project root to Python path to import persona modules
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Load .env file when running outside Docker
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
    
    from persona.llm.client_factory import get_chat_client
    from persona.llm.providers.base import ChatMessage
    
    async def async_query():
        client = get_chat_client()
        messages = [ChatMessage(role="user", content=prompt)]
        
        for attempt in range(max_retries):
            try:
                response = await client.chat(
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=10
                )
                return response.content.strip()
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
                if attempt == max_retries - 1:
                    # Instead of raising, we will return an empty string to be handled by the caller
                    return ""
        
        return ""
    
    # Run the async function
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new event loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, async_query())
                return future.result()
        else:
            return asyncio.run(async_query())
    except Exception as e:
        print(f"Error in LLM query: {e}")
        return ""

def evaluate_qa(hypotheses_file: str, reference_file: str, 
               metric_model: str = EVAL_MODEL, verbose: bool = False) -> Dict:
    """
    Evaluate QA results using LongMemEval methodology
    
    Args:
        hypotheses_file: Path to JSONL file with model hypotheses
        reference_file: Path to JSON file with reference data
        metric_model: Model to use for evaluation
        verbose: Whether to print detailed results
    
    Returns:
        Dictionary with evaluation results
    """
    
    # Load data
    try:
        with open(hypotheses_file, 'r') as f:
            hypotheses = [json.loads(line) for line in f]
    except:
        with open(hypotheses_file, 'r') as f:
            hypotheses = json.load(f)
    
    try:
        with open(reference_file, 'r') as f:
            references = json.load(f)
    except:
        with open(reference_file, 'r') as f:
            references = [json.loads(line) for line in f]
    
    # Create lookups
    qid2qdata = {entry['question_id']: entry for entry in references}
    qid2qtype = {entry['question_id']: entry['question_type'] for entry in references}
    
    # Initialize results tracking
    qtypes = set(list(qid2qtype.values()))
    qtype2acc = {t: [] for t in qtypes}
    
    # Process each hypothesis
    logs = []
    for entry in tqdm(hypotheses, desc="Evaluating"):
        if entry['question_id'] not in qid2qtype:
            print(f'Warning: skipping {entry["question_id"]} as it is not in reference data.')
            continue
        
        qtype = qid2qtype[entry['question_id']]
        q = qid2qdata[entry['question_id']]['question']
        ans = qid2qdata[entry['question_id']]['answer']
        hyp = entry['hypothesis']
        
        # Create evaluation prompt
        prompt = get_anscheck_prompt(qtype, q, ans, hyp, abstention='_abs' in entry['question_id'])
        
        # Get LLM evaluation
        eval_response = query_openai_with_retry(prompt)
        try:
            label = 'yes' in eval_response.lower()
        except Exception as e:
            print(f"Error evaluating {entry['question_id']}: {e}")
            label = False
        
        # Store results
        entry['autoeval_label'] = {
            'model': metric_model,
            'label': label,
            'raw_response': eval_response
        }
        logs.append(entry)
        
        if verbose:
            print(f"Question: {q}")
            print(f"Gold: {ans}")
            print(f"Hypothesis: {hyp}")
            print(f"Label: {label}")
            print("-" * 50)
        
        qtype2acc[qtype].append(1 if label else 0)
    
    # Calculate metrics
    all_scores = []
    task_scores = []
    
    results = {
        'overall_accuracy': 0,
        'task_accuracies': {},
        'total_questions': len(logs),
        'model_used': metric_model
    }
    
    for k, v in qtype2acc.items():
        if v:  # Only calculate if there are instances
            acc = sum(v) / len(v)
            results['task_accuracies'][k] = {
                'accuracy': acc,
                'count': len(v)
            }
            task_scores.append(acc)
            all_scores.extend(v)
    
    if all_scores:
        results['overall_accuracy'] = sum(all_scores) / len(all_scores)
    
    if task_scores:
        results['task_averaged_accuracy'] = sum(task_scores) / len(task_scores)
    
    return results, logs

def print_results(results: Dict):
    """Print evaluation results in a formatted way"""
    print("\n" + "="*60)
    print("📊 LONGMEMEVAL EVALUATION RESULTS")
    print("="*60)
    
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Task-Averaged Accuracy: {results.get('task_averaged_accuracy', 0):.4f}")
    print(f"Total Questions: {results['total_questions']}")
    print(f"Evaluation Model: {results['model_used']}")
    
    print(f"\nTask-Specific Results:")
    for task, metrics in results['task_accuracies'].items():
        print(f"  {task}: {metrics['accuracy']:.4f} ({metrics['count']} questions)")
    
    print("="*60)

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate LongMemEval QA results")
    parser.add_argument("hypotheses_file", help="Path to hypotheses JSONL file")
    parser.add_argument("reference_file", help="Path to reference JSON file")
    parser.add_argument("--model", default=EVAL_MODEL, help="Evaluation model to use")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    parser.add_argument("--output", help="Output file for detailed results")
    
    args = parser.parse_args()
    
    # Run evaluation
    results, logs = evaluate_qa(args.hypotheses_file, args.reference_file, 
                               args.model, args.verbose)
    
    # Print results
    print_results(results)
    
    # Save detailed results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'evaluation_results': results,
                'detailed_logs': logs
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    main() 