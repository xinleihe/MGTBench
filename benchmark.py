import argparse
import datetime
import os
import json
import dataset_loader
from methods.utils import load_base_model, load_base_model_and_tokenizer, filter_test_data, sample_dataset
from methods.supervised import run_supervised_experiment
from methods.detectgpt import run_perturbation_experiments
from methods.gptzero import run_gptzero_experiment
from methods.metric_based import get_ll, get_rank, get_entropy, get_rank_GLTR, run_threshold_experiment, run_GLTR_experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Essay")
    parser.add_argument('--detectLLM', type=str, default="ChatGPT")
    parser.add_argument('--method', type=str, default="Log-Likelihood")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--mask_filling_model_name',
                        type=str, default="t5-base")
    parser.add_argument('--cache_dir', type=str, default=".cache")
    parser.add_argument('--DEVICE', type=str, default="cuda")

    # params for DetectGPT
    parser.add_argument('--pct_words_masked', type=float, default=0.3)
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_perturbation_list', type=str, default="10")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--random_fills_tokens', action='store_true')

    # params for GPTZero
    parser.add_argument('--gptzero_key', type=str, default="")

    args = parser.parse_args()

    DEVICE = args.DEVICE

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

    print(f'Loading dataset {args.dataset}...')
    data = dataset_loader.load(args.dataset, detectLLM=args.detectLLM)
    # data = sample_dataset(data, 50, 10)
    # data = filter_test_data(data, max_length=25)

    base_model_name = args.base_model_name.replace('/', '_')
    SAVE_PATH = f"update_results/{base_model_name}-{args.mask_filling_model_name}/{args.dataset}-{args.detectLLM}"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_PATH)}")

    # write args to file
    with open(os.path.join(SAVE_PATH, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    mask_filling_model_name = args.mask_filling_model_name
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    n_perturbation_rounds = args.n_perturbation_rounds
    n_similarity_samples = args.n_similarity_samples

    cache_dir = args.cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    # get generative model
    base_model, base_tokenizer = load_base_model_and_tokenizer(
        args.base_model_name, cache_dir)
    load_base_model(base_model, DEVICE)

    # def phd_criterion(text): return get_phd(
    #     text, base_model, base_tokenizer, DEVICE)

    def ll_criterion(text): return get_ll(
        text, base_model, base_tokenizer, DEVICE)

    def rank_criterion(text): return -get_rank(text,
                                               base_model, base_tokenizer, DEVICE, log=False)

    def logrank_criterion(text): return -get_rank(text,
                                                  base_model, base_tokenizer, DEVICE, log=True)

    def entropy_criterion(text): return get_entropy(
        text, base_model, base_tokenizer, DEVICE)

    def GLTR_criterion(text): return get_rank_GLTR(
        text, base_model, base_tokenizer, DEVICE)

    outputs = []
    # outputs.append(run_threshold_experiment(data, phd_criterion, "phd"))

    # method_list = ["Log-Likelihood", "Rank", "Log-Rank", "Entropy", "GLTR", "OpenAI-D", "ConDA", "ChatGPT-D", "LM-D", "DetectGPT", "LRR", "NPR", "GPTZero"]

    if args.method == "Log-Likelihood":
        outputs.append(run_threshold_experiment(
            data, ll_criterion, "likelihood"))
    elif args.method == "Rank":
        outputs.append(run_threshold_experiment(data, rank_criterion, "rank"))
    elif args.method == "Log-Rank":
        outputs.append(run_threshold_experiment(
            data, logrank_criterion, "log_rank"))
    elif args.method == "Entropy":
        outputs.append(run_threshold_experiment(
            data, entropy_criterion, "entropy"))
    elif args.method == "GLTR":
        outputs.append(run_GLTR_experiment(data, GLTR_criterion, "rank_GLTR"))

    # model-based method. Note that we can also fine-tune model and save it below.
    # elif args.method == "OpenAI-D":
    #     outputs.append(run_supervised_experiment(data, model='roberta-base-openai-detector',
    #                cache_dir=cache_dir, batch_size=batch_size, DEVICE=DEVICE, finetune=True, save_path=SAVE_PATH + f"/OpenAI-D"))
    # elif args.method == "ConDA":
    #     outputs.append(run_supervised_experiment(data, model='update_results/ConDA',
    #                cache_dir=cache_dir, batch_size=batch_size, DEVICE=DEVICE, finetune=True, save_path=SAVE_PATH + f"/ConDA"))
    # elif args.method == "ChatGPT-D":
    #     outputs.append(run_supervised_experiment(data, model='Hello-SimpleAI/chatgpt-detector-roberta',
    # cache_dir=cache_dir, batch_size=batch_size, DEVICE=DEVICE, pos_bit=1,
    # finetune=True, save_path=SAVE_PATH + f"/ChatGPT-D"))
    elif args.method == "OpenAI-D":
        outputs.append(
            run_supervised_experiment(
                data,
                model='roberta-base-openai-detector',
                cache_dir=cache_dir,
                batch_size=batch_size,
                DEVICE=DEVICE))
    elif args.method == "ConDA":
        outputs.append(
            run_supervised_experiment(
                data,
                model='update_results/ConDA',
                cache_dir=cache_dir,
                batch_size=batch_size,
                DEVICE=DEVICE))
    elif args.method == "ChatGPT-D":
        outputs.append(
            run_supervised_experiment(
                data,
                model='Hello-SimpleAI/chatgpt-detector-roberta',
                cache_dir=cache_dir,
                batch_size=batch_size,
                DEVICE=DEVICE,
                pos_bit=1))
    elif args.method == "LM-D":
        outputs.append(
            run_supervised_experiment(
                data,
                model='distilbert-base-uncased',
                cache_dir=cache_dir,
                batch_size=batch_size,
                DEVICE=DEVICE,
                pos_bit=1,
                finetune=True,
                save_path=SAVE_PATH +
                f"/LM-D"))

    # run LRR
    elif args.method == "LRR":
        outputs.append(run_perturbation_experiments(
            args, data, base_model, base_tokenizer, method="LRR"))

    # # run GPTZero: pleaze specify your gptzero_key in the args
    elif args.method == "GPTZero":
        outputs.append(run_gptzero_experiment(data, api_key=args.gptzero_key))

    # run DetectGPT
    elif args.method == "DetectGPT":
        outputs.append(run_perturbation_experiments(
            args, data, base_model, base_tokenizer, method="DetectGPT"))

    # run NPR
    elif args.method == "NPR":
        outputs.append(run_perturbation_experiments(
            args, data, base_model, base_tokenizer, method="NPR"))

    # save results
    import pickle as pkl
    with open(os.path.join(SAVE_PATH, f"{args.method}_benchmark_results.pkl"), "wb") as f:
        pkl.dump(outputs, f)

    if not os.path.exists("logs/"):
        os.makedirs("logs/")
    with open("logs/performance.csv", "a") as wf:
        for row in outputs:
            wf.write(
                f"{args.dataset},{args.detectLLM},{args.base_model_name},{args.method},{json.dumps(row['general'])}\n")

    print("Finish")
