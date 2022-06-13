import json
import os
import torch
import numpy as np
import re

from transformers import AutoTokenizer, ElectraConfig, ElectraForSequenceClassification, TextClassificationPipeline


def get_preds_label (model, inputs):
    output=model(**inputs)
    logits=output[0]
    preds = logits.detach().cpu().numpy()
    label=np.argmax(preds)

    return int(label)


'''tokenizer 가져오기'''
tokenizer = AutoTokenizer.from_pretrained( "beomi/KcELECTRA-base", do_lower_case=False)

'''sentiment model 불러오기 '''
# local ver
epoch=15 #모델 최고 성능이었던 epoch 설정
model_output_dir=os.path.join("ckpt", "sentiment-stove_third-kcelectra_weightedsampler", f'epoch-{epoch}') #최고 성능이었던 epoch의 모델 불러오기
sentiment_model = ElectraForSequenceClassification.from_pretrained(model_output_dir)

# huggingface ver
#sentiment_model = ElectraForSequenceClassification.from_pretrained("huggingface library 주소")

''' theme model 불러오기'''
# local ver
epoch=33 #모델 최고 성능이었던 epoch 설정
model_output_dir=os.path.join("ckpt", "theme-stove_third_kcelectra_weightedsampler", f'epoch-{epoch}') #최고 성능이었던 epoch의 모델 불러오기
theme_model = ElectraForSequenceClassification.from_pretrained(model_output_dir)

# huggingface ver
#theme_model = ElectraForSequenceClassification.from_pretrained("huggingface library 주소")

''' da model 불러오기'''
# local ver
epoch=13 #모델 최고 성능이었던 epoch 설정
model_output_dir=os.path.join("ckpt", "da-stove_third-kcelectra_weightedsampler", f'epoch-{epoch}') #최고 성능이었던 epoch의 모델 불러오기
da_model = ElectraForSequenceClassification.from_pretrained(model_output_dir)

# huggingface ver
#da_model = ElectraForSequenceClassification.from_pretrained("huggingface library 주소")





''' dataset 가져오기'''
input_file='data_final.json' ### 설정하기
with open(input_file, "r", encoding="utf-8") as f:
    json_file = json.load(f)
    lines = []

    for d in json_file:
        line = str(d["header"]) + str(d['content'])
        line=re.sub("&nbsp;", "", line)
        line=re.sub("<br>", "", line)
        line=re.sub("<.*?>", "", line)
        lines.append(line.strip())



''' labels dict 만들기'''
sentiment_labels=["-1", "0", "1"]
theme_labels=["캐릭터", "컨텐츠", "이벤트", "버그", "점검",  "유저", "회사", "기타"]
da_labels=["질문", "의견", "건의", "정보", "일상"]

sentiment_label_dict={i:x for i,x in enumerate(sentiment_labels)}
theme_label_dict={i:x for i,x in enumerate(theme_labels)}
da_label_dict={i:x for i,x in enumerate(da_labels)}




''' lines의 모든 line들을 encoding하기'''
batch_encoding=tokenizer.batch_encode_plus(
    [str(line) for line in lines],
    max_length=512,
    padding="max_length",
    truncation=True
)

''' label 가져와서 label predict하고 json으로 저장하기 '''

results=[]
for (i, data) in enumerate(json_file):
    inputs = {
        "input_ids": torch.tensor([batch_encoding['input_ids'][i]]),
        "attention_mask": torch.tensor([batch_encoding['attention_mask'][i]])
    }
    data["sentiment"]=sentiment_label_dict[get_preds_label(sentiment_model, inputs)]
    data["theme"]=theme_label_dict[get_preds_label(theme_model, inputs)]
    data["da"]=da_label_dict[get_preds_label(da_model, inputs)]

    results.append(data)

with open('data_results_final2.json', 'w', encoding='utf-8') as fw:
    json.dump(results, fw, indent=4, ensure_ascii=False)
