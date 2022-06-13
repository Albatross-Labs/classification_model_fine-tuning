import argparse
import json
import logging
import os
import glob
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from fastprogress.fastprogress import master_bar, progress_bar
from attrdict import AttrDict

from transformers import AutoTokenizer

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from src import (
    CONFIG_CLASSES,
    TOKENIZER_CLASSES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION,
    init_logger,
    set_seed,
    compute_metrics
)
from processor import seq_cls_load_and_cache_examples as load_and_cache_examples
from processor import seq_cls_tasks_num_labels as tasks_num_labels
from processor import seq_cls_processors as processors
from processor import seq_cls_output_modes as output_modes

logger = logging.getLogger(__name__)

def evaluate(args, model, eval_dataset, mode, epoch=None):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    if epoch != None:
        logger.info("***** Running evaluation on {} dataset ({}) *****".format(mode, epoch))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            if args.model_type not in ["distilkobert", "xlm-roberta"]:
                inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if output_modes[args.task] == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_modes[args.task] == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(args.task, out_label_ids, preds)
    results.update(result)

    output_dir = os.path.join('./test_on_third_data', f'{args.output_dir}') #output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir, f"test_on_third_data_{args.task}_{epoch}.txt") #output txt
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in results.keys():
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))
            f_w.write('real_label')
            for i in out_label_ids:
                f_w.write(str(i))
            f_w.write('predict_label')
            for i in preds:
                f_w.write(str(i))

    return results


def main(cli_args):
    # Read from config file and make args
    with open(os.path.join(cli_args.config_dir, cli_args.task, cli_args.config_file)) as f:
        args = AttrDict(json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))

    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    init_logger()
    set_seed(args)

    processor = processors[args.task](args)
    labels = processor.get_labels()
    if output_modes[args.task] == "regression":
        config = CONFIG_CLASSES[args.model_type].from_pretrained(
            args.model_name_or_path,
            num_labels=tasks_num_labels[args.task]
        )
    else:
        config = CONFIG_CLASSES[args.model_type].from_pretrained(
            args.model_name_or_path,
            num_labels=tasks_num_labels[args.task],
            id2label={str(i): label for i, label in enumerate(labels)},
            label2id={label: i for i, label in enumerate(labels)},
        )

    tokenizer_output_dir = os.path.join(args.output_dir, "tokenizer")

    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        tokenizer_output_dir,
        do_lower_case=args.do_lower_case
    )

    # Load dataset
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test") if args.test_file else None


    dirs=os.listdir(args.output_dir)
    ckpt=[]
    for dir in dirs:
        if re.match('epoch', dir):
            ckpt.append(dir)


    results={}

    # 모델 최고 성능이었던 epoch 설정
    if args.task=="sentiment":
        epoch=15
    elif args.task=='theme':
        epoch=33
    elif args.task=='da':
        epoch=13
    model_output_dir=os.path.join(args.output_dir, f'epoch-{epoch}') #최고 성능이었던 epoch의 모델델 불러오기
    model = MODEL_FOR_SEQUENCE_CLASSIFICATION[args.model_type].from_pretrained(model_output_dir)

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    result=evaluate(args, model, test_dataset, mode="test", epoch=epoch)
    result = dict((k + "_{}".format(epoch), v) for k, v in result.items())
    results.update(result)

    output_dir = os.path.join('./test_on_third_data', f'{args.output_dir}') #output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir, f"{args.task}_test_eval_results.txt")
    with open(output_eval_file, "w") as f_w:
        for key in sorted(results.keys()):
            f_w.write("{} = {}\n".format(key, str(results[key])))


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--task", type=str, required=True)
    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, required=True)

    cli_args = cli_parser.parse_args()

    main(cli_args)


