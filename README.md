# Turkish-question-paraphrase-generator
mT5 based pre-trained model to generate question paraphrases in Turkish language.


## Acknowledgement
In this project, which we undertook as an BLM3010 Computer Project of Yildiz Technical University, our goal was to conduct research on Turkish in area that has not been studied much. In this process, we compared the models trained with different algorithms. Developed a dataset and shared it by writing article for our model. We would like to thank our mentor teacher <a href="https://github.com/mfatihamasyali">Mehmet Fatih Amasyali</a> who has always supported us on this path.

## Usage

```python
import transformers, sentencepiece

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("secometo/mt5-base-turkish-question-paraphrase-generator")
model = AutoModelForSeq2SeqLM.from_pretrained("secometo/mt5-base-turkish-question-paraphrase-generator")

original_sentence = tokenizer.encode_plus("Ülke genelinde bisiklet kullanımının artması hakkında ne düşünüyorsun?", return_tensors='pt')
paraphrased_sentences = model.generate(original_sentence['input_ids'], max_length=150, num_return_sequences=5, num_beams=5)
tokenizer.batch_decode(paraphrased_sentences, skip_special_tokens=True)
```
## Input
```
Ülke genelinde bisiklet kullanımının artması hakkında ne düşünüyorsun?
```
## Outputs

```
['ülke genelinde bisiklet kullanımının artması ile ilgili düşünceniz nedir?',
 'ülke genelinde bisiklet kullanımının artması hakkında düşünceniz nedir?',
 'ülke genelinde bisiklet kullanımının artması için ne düşünüyorsunuz?',
 'ülke genelinde bisiklet kullanımının artması hakkında ne düşünüyorsunuz?',
 'ülke genelinde bisiklet kullanımının artması hakkında fikirleriniz nedir?']
 ```
 
## Dataset
 
We used 50994 question sentence pairs, which are created manually, to train our model. The dataset is provided our mentor. Sentences were extracted from the titles of topics in popular Turkish forum website donanimhaber.com. We augmented the dataset by writing ten thousand sentences per person.
 

 ## Authors & Citation
 
 <a href="https://github.com/metinbinbir">Metin Binbir</a></br>
 <a href="https://github.com/sercaksoy">Sercan Aksoy</a>
```
Metin Binbir, Sercan Aksoy, Paraphrase Generation for Turkish Language, Computer Project, Advisor: Mehmet Fatih Amasyali, Yildiz Technical University, Computer Engineering Dept. , Istanbul, Turkey , 2021.
```
