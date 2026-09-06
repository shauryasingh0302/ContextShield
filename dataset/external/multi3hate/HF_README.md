---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: Meme ID
    dtype: int64
  - name: Language
    dtype: string
  - name: Caption
    dtype: string
  - name: US
    dtype: float64
  - name: DE
    dtype: float64
  - name: MX
    dtype: float64
  - name: CN
    dtype: float64
  - name: IN
    dtype: float64
  - name: Template Name
    dtype: string
  - name: Category
    dtype: string
  - name: Subcategory
    dtype: string
  splits:
  - name: train
    num_bytes: 62034606
    num_examples: 1500
  download_size: 62780586
  dataset_size: 62034606
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
language:
- en
- de
- hi
- zh
- es
pretty_name: Multi^3Hate
size_categories:
- 1K<n<10K
---
# Multi<sup>3</sup>Hate Dataset: Multimodal, Multilingual, and Multicultural Hate Speech Dataset

<!-- Provide a quick summary of the dataset. -->

*Warning: this dataset contains content that may be offensive or upsetting*


**Multi<sup>3</sup>Hate** dataset, introduced in our paper: **[Multi<sup>3</sup>Hate: Advancing Multimodal, Multilingual, and Multicultural Hate Speech Detection with Vision–Language Models](https://arxiv.org/pdf/2411.03888)**

**Abstract:** Hate speech moderation on global platforms poses unique challenges due to the multimodal and multilingual nature of content, along with the varying cultural perceptions. How well do current vision-language models (VLMs) navigate these nuances?
To investigate this, we create the first multimodal and multilingual parallel hate speech dataset, annotated by a multicultural set of annotators, called Multi3Hate. It contains 300 parallel meme samples across 5 languages: English, German, Spanish, Hindi, and Mandarin.
We demonstrate that cultural background significantly affects multimodal hate speech annotation in our dataset. The average pairwise agreement among countries is just 74%, significantly lower than that of randomly selected annotator groups.
Our qualitative analysis indicates that the lowest pairwise label agreement—only 67% between the USA and India—can be attributed to cultural factors. We then conduct experiments with 5 large VLMs in a zero-shot setting, 
finding that these models align more closely with annotations from the US than with those from other cultures, even when the memes and prompts are presented in the dominant language of the other culture.

<img src="https://cdn-uploads.huggingface.co/production/uploads/65e0e6fa4394fc3d1b59627a/cVJJ-pE-41DJIElsf-cN_.png" width="500"/>


## Dataset Details

The dataset comprises a curated collection of 300 memes—images paired with embedded captions—a prevalent form of multimodal content,
presented in five languages: English (en), German (de), Spanish (es), Hindi (hi), and Mandarin (zh).
Each of the 1,500 memes (300×5 languages) is annotated for hate speech in the respective target language by at least five native speakers
from the same country. These countries were chosen based on the largest number of native speakers of each target language: USA (US), Germany (DE),
Mexico (MX), India (IN), and China (CN).

<img src="https://cdn-uploads.huggingface.co/production/uploads/65e0e6fa4394fc3d1b59627a/gIc6rLoHaA0ji5XWHYqFD.png" width="900"/>


We curate a list of culturally relevant templates by filtering them according to sociopolitical categories.
These categories were discussed and decided among the authors. Each category is further divided into specific topics
based on established criteria.

<img src="https://cdn-uploads.huggingface.co/production/uploads/65e0e6fa4394fc3d1b59627a/mBi8QLRazQ2glLaC12qMm.png" width="500"/>



### Dataset Sources

<!-- Provide the basic links for the dataset. -->

- **Repository:** [Link](https://github.com/MinhDucBui/Multi3Hate/tree/main)
- **Paper:** [Link](https://arxiv.org/pdf/2411.03888)



#### Annotation process & Demographic

<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->

We refer to our paper for more details.

<img src="https://cdn-uploads.huggingface.co/production/uploads/65e0e6fa4394fc3d1b59627a/3egu-bS9rF4CM5zJcCJoi.png" width="500"/>



#### Who are the source data producers?

<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->

The original meme images were crawled from https://memegenerator.net

## License

CC BY-NC-ND 4.0


## Citation

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```
@misc{bui2024multi3hatemultimodalmultilingualmulticultural,
      title={Multi3Hate: Multimodal, Multilingual, and Multicultural Hate Speech Detection with Vision-Language Models}, 
      author={Minh Duc Bui and Katharina von der Wense and Anne Lauscher},
      year={2024},
      eprint={2411.03888},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.03888}, 
}
```

