import json
import os
import torch
import numpy as np

from transformers import AutoTokenizer, ElectraConfig, ElectraForSequenceClassification, TextClassificationPipeline


'''tokenizer 가져오기'''
tokenizer = AutoTokenizer.from_pretrained( "beomi/KcELECTRA-base", do_lower_case=False)

'''sentiment model 불러오기 '''
# local ver
epoch=49 #모델 최고 성능이었던 epoch 설정
model_output_dir=os.path.join("ckpt", "sentiment-stove_second-kcelectra_weightedsampler", f'epoch-{epoch}') #최고 성능이었던 epoch의 모델 불러오기
sentiment_model = ElectraForSequenceClassification.from_pretrained(model_output_dir)

# huggingface ver
#sentiment_model = ElectraForSequenceClassification.from_pretrained("huggingface library 주소")

''' theme model 불러오기'''
# local ver
epoch=31 #모델 최고 성능이었던 epoch 설정
model_output_dir=os.path.join("ckpt", "theme-stove_second_kcelectra_weightedsampler2", f'epoch-{epoch}') #최고 성능이었던 epoch의 모델 불러오기
theme_model = ElectraForSequenceClassification.from_pretrained(model_output_dir)

# huggingface ver
#theme_model = ElectraForSequenceClassification.from_pretrained("huggingface library 주소")

''' da model 불러오기'''
# local ver
epoch=47 #모델 최고 성능이었던 epoch 설정
model_output_dir=os.path.join("ckpt", "da-stove_second-kcelectra_weightedsampler", f'epoch-{epoch}') #최고 성능이었던 epoch의 모델 불러오기
da_model = ElectraForSequenceClassification.from_pretrained(model_output_dir)

# huggingface ver
#da_model = ElectraForSequenceClassification.from_pretrained("huggingface library 주소")


''' dataset 가져오기'''
input_file='data.json' ### 설정하기
with open(input_file, "r", encoding="utf-8") as f:
    json_file = json.load(f)
    lines = []

    for d in json_file:
        line = str(d["header"]) + str(d['content'])
        lines.append(line.strip())


''' pipeline으로 가져오기
sentiment_pipe=TextClassificationPipeline(
    model=sentiment_model,
    tokenizer=tokenizer,
    device=0,
    function_to_apply="softmax"
)

theme_pipe=TextClassificationPipeline(
    model=theme_model,
    tokenizer=tokenizer,
    device=0,
    function_to_apply="softmax"
)

da_pipe=TextClassificationPipeline(
    model=da_model,
    tokenizer=tokenizer,
    device=0,
    function_to_apply="softmax"
)

results=[]
for (i, data) in enumerate(json_file): 
    tokenize_results=tokenizer.tokenize(lines[i])
    print(lines[i])
    if len(tokenize_results)>512:
        #lines[i]=' '.join(re.sub('#', '', i) for i in tokenize_results)
        #print(lines[i])
        continue
    data["sentiment"] = sentiment_pipe(lines[i])[0]['label']
    data["theme"] = theme_pipe(lines[i])[0]['label']
    data["da"] = da_pipe(lines[i])[0]['label']
    results.append(data)
'''