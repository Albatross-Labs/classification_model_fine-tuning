import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset
import json

logger = logging.getLogger(__name__)

'''
task별로 보는 것이 아니라, 모델 3개 별로 다르게 처리할 수 있게 
args에서 받아오는 부분만 달리하면 모델 3개를 하나의 코드로 돌릴 수 있게 수정하기 
--task 로 받아올 수 있게 수정하면 될듯...?
sentiment, theme, da
'''


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def seq_cls_convert_examples_to_features(args, examples, tokenizer, max_length, task):
    processor = seq_cls_processors[task](args)
    label_list = processor.get_labels()
    logger.info("Using label list {} for task {}".format(label_list, task))
    output_mode = seq_cls_output_modes[task]
    logger.info("Using output mode {} for task {}".format(output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example):
        if output_mode == "classification":
            return label_map[example.label] #label_map은 각 label을 인덱스로 표시하고 있음
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples] #이 labels들을 불러오면 전체 데이터의 label들을 확인할 수 있음

    batch_encoding = tokenizer.batch_encode_plus(
        [(str(example.text_a), str(example.text_b)) for example in examples],
        max_length=max_length,
        padding="max_length",
        add_special_tokens=True,
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = [0] * len(inputs["input_ids"])  # For xlm-roberta

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: {}".format(example.guid))
        logger.info("input_ids: {}".format(" ".join([str(x) for x in features[i].input_ids])))
        logger.info("attention_mask: {}".format(" ".join([str(x) for x in features[i].attention_mask])))
        logger.info("token_type_ids: {}".format(" ".join([str(x) for x in features[i].token_type_ids])))
        logger.info("label: {}".format(features[i].label))

    return features



class SentimentProcessor(object):
    """Processor for the AUC data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["-1", "0", "1"]

    @classmethod
    def _read_file(cls, input_file): #input 파일을 json 파일로 넘겨주기

        '''json 파일 읽어오기'''

        with open(input_file, "r", encoding="utf-8") as f:
            json_file=json.load(f)
            lines = []

            for d in json_file:
                line=str(d["header"])+str(d['content'])+'\t'+str(d['sentiment'])
                lines.append(line.strip())

            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[0:]):
            try:
                line = line.split("\t")
                guid = "%s-%s" % (set_type, i)
                text_a = str(line[0])
                label = str(line[1])
            except:
                print(i, text_a) ### label이 비어있는 경우
                continue
            if label not in self.get_labels(): ### label이 다른 것으로 정의 된 경우
                print(i, text_a, label)
                continue
            if i % 10000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        print(examples[0], type(examples[0]))
        return examples #여기에서 label을 가져올 수 있음

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, self.args.task, file_to_read)))
        return self._create_examples(
            self._read_file(os.path.join(self.args.data_dir, self.args.task, file_to_read)), mode
        )


class ThemeProcessor(object):
    """Processor for the AUC data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["캐릭터", "아이템", "레이드", "업데이트", "이벤트", "버그", "해킹", "점검", "굿즈", "유저", "회사", "기타"]

    @classmethod
    def _read_file(cls, input_file):  # input 파일을 json 파일로 넘겨주기

        '''json 파일 읽어오기'''

        with open(input_file, "r", encoding="utf-8") as f:
            json_file = json.load(f)
            lines = []

            for d in json_file:
                line = str(d["header"]) + str(d['content']) + '\t' + str(d['theme1'])
                lines.append(line.strip())

            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[0:]):
            try:
                line = line.split("\t")
                guid = "%s-%s" % (set_type, i)
                text_a = str(line[0])
                label = str(line[1])
            except:
                print(i, text_a)
                continue
            if label not in self.get_labels(): ### label이 다른 것으로 정의 된 경우
                print(i, text_a, label)
                continue
            if i % 10000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        print(examples[0], type(examples[0]))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, self.args.task, file_to_read)))
        return self._create_examples(
            self._read_file(os.path.join(self.args.data_dir, self.args.task, file_to_read)), mode
        )


class DAProcessor(object):
    """Processor for the AUC data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["질문", "의견", "건의", "인증", "친목", "정보"]

    @classmethod
    def _read_file(cls, input_file):  # input 파일을 json 파일로 넘겨주기

        '''json 파일 읽어오기'''

        with open(input_file, "r", encoding="utf-8") as f:
            json_file = json.load(f)
            lines = []

            for d in json_file:
                line = str(d["header"]) + str(d['content']) + '\t' + str(d['da'])
                lines.append(line.strip())

            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[0:]):
            try:
                line = line.split("\t")
                guid = "%s-%s" % (set_type, i)
                text_a = str(line[0])
                label = str(line[1])
            except:
                print(i, text_a)
                continue
            if label not in self.get_labels(): ### label이 다른 것으로 정의 된 경우
                print(i, text_a, label)
                continue
            if i % 10000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        print(examples[0], type(examples[0]))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, self.args.task, file_to_read)))
        return self._create_examples(
            self._read_file(os.path.join(self.args.data_dir, self.args.task, file_to_read)), mode
        )

### 벤치마크 데이터
class NsmcProcessor(object):
    """Processor for the NSMC data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["0", "1"]

    @classmethod
    def _read_file(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[1:]):
            line = line.split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[2]
            if i % 10000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, self.args.task, file_to_read)))
        return self._create_examples(
            self._read_file(os.path.join(self.args.data_dir, self.args.task, file_to_read)), mode
        )

class HateSpeechProcessor(object):
    """Processor for the Korean Hate Speech data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["none", "offensive", "hate"]

    @classmethod
    def _read_file(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[1:]):
            line = line.split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[3]
            if i % 1000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, self.args.task, file_to_read)))
        return self._create_examples(
            self._read_file(os.path.join(self.args.data_dir, self.args.task, file_to_read)), mode
        )


seq_cls_processors = {
    "sentiment": SentimentProcessor,
    "theme": ThemeProcessor,
    "da": DAProcessor,
    "nsmc": NsmcProcessor,
    "hate-speech": HateSpeechProcessor
}

seq_cls_tasks_num_labels = {"sentiment":3, "theme":12 ,"da":6, "nsmc": 2, "hate-speech": 3}

seq_cls_output_modes = {
    "sentiment":"classification",
    "theme":"classification",
    "da": "classification",
    "nsmc": "classification",
    "hate-speech": "classification",
}


def seq_cls_load_and_cache_examples(args, tokenizer, mode):
    processor = seq_cls_processors[args.task](args)
    output_mode = seq_cls_output_modes[args.task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            str(args.task), list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_len), mode
        ),
    )

    logger.info("Creating features from dataset file at %s", args.data_dir)
    if mode == "train":
        examples = processor.get_examples("train")
    elif mode == "dev":
        examples = processor.get_examples("dev")
    elif mode == "test":
        examples = processor.get_examples("test")
    else:
        raise ValueError("For mode, only train, dev, test is avaiable")
    features = seq_cls_convert_examples_to_features(
        args, examples, tokenizer, max_length=args.max_seq_len, task=args.task
    )
    logger.info("Saving features into cached file %s", cached_features_file)
    torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset
