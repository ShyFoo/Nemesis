# Nemesis: Normalizing the Soft-prompt Vectors of Vision-Language Models (ICLR 2024 Spotlight)

Paper Link: [Nemesis: Normalizing the Soft-prompt Vectors of Vision-Language Models](https://openreview.net/pdf?id=zmJDzPh1Dm)

## Highlights

To answer an unexplored research question: "Do we need to normalize the soft prompts in VLMs?", 
we first uncover a phenomenon, called the **Low-Norm Effect** by performing extensive corruption experiments,
suggesting that reducing the norms of certain learned prompts occasionally enhances the performance of VLMs,
while increasing them often degrades it.
To harness this effect, we propose a novel method named 
**N**ormalizing th**e** soft-pro**m**pt v**e**ctors of vi**si**on-language model**s** (**Nemesis**) to normalize soft-prompt vectors in VLMs. 
To the best of our knowledge, our work is the first to systematically investigate the role of norms of soft-prompt vector in VLMs,
offering valuable insights for future research in soft-prompt tuning.

Besides, we also conduct preliminary to verify the generalizability and effectiveness of Nemesis on other **P**arameter-**EF**ficient **T**uning (**PEFT**) methods,
including [**visual prompt tuning**](https://github.com/KMnP/vpt) and [**prefix-tuning**](https://github.com/XiangLi1999/PrefixTuning). 
Detailed results can be found in the following tables.

## The Low-Norm Effect
<figure>
    <img src="./figures/low_norm_effect.jpg" style="width: 60%; text-align: center" alt="The Low-Norm Effect">
    <figcaption><strong><em>The schematic diagram of the Low-Norm Effect</em></strong></figcaption>
</figure>

A schematic diagram of the Low-Norm Effect: the reduction of norms at specific positions within these prompts enhances performance,
whereas an increase in norms typically results in performance deterioration. 
_**Top**_: corrupted soft prompts with increased norms leading to decreased performance;
_**Middle**_: soft prompts learned by CoOp; 
_**Bottom**_: corrupted soft prompts with reduced norms resulting in enhanced performance.

---
<figure>
    <img src="./figures/low_norm_effect_frequency.jpg" style="width: 49%; text-align: center" alt="The frequency across 11 datasets">
    <figcaption><strong><em>The frequency of the Low-Norm Effect across 11 datasets</em></strong></figcaption>
</figure>

The 11 datasets have exhibited varying frequencies of the Low-Norm Effect. 
This observation indicates that tackling the Low-Norm Effect is a challenging task,
given its inconsistent manifestation across the 11 datasets.

---

<div style="display:flex; flex-direction: row">
    <img src="./figures/low_norm_effect_explanation1.jpg" alt="Explanation 1" style="width: 49%">
    <img src="./figures/low_norm_effect_explanation2.jpg" alt="Explanation 2" style="width: 49%">
</div>

From **the left figure**, 
it is apparent that the norms of soft prompts in CoOp first increase and then level off,
while test accuracy falls into degradation as norms slowly flatten out.
By performing corruption operations that decrease the norms of prompt vectors,
the last green circle may be pushed away from the degradation area and get closer to those small green circles that demonstrate superior performance.

From **the right figure**, 
we observe a distinct norm variation pattern in CoOp+Nemesis (ours) that differs from CoOp.
This pattern demonstrates an initial increase in norms, followed by a subsequent decrease,
and eventually reaching a stable state.
Furthermore, the test accuracy exhibits a consistent upward trend before reaching a plateau,
whereas a declining trend is observed in CoOp.

This implies that our method can delay the time point where soft prompts tend to plateau during the learning process,
thereby reducing the probability of learning degradation.

---

## Main Results
<div style="display:flex; flex-direction: row">
    <img src="./figures/result_few_shot_classification.jpg" alt="Result 1" style="width: 29%">
    <img src="./figures/result_domain_generalization.jpg" alt="Result 2" style="width: 29%">
    <img src="./figures/reulst_base_to_new.jpg" alt="Result 3" style="width: 29%">
</div>

## How to Run

## Citation
If you use our work, please consider citing:
```bibtex
@inproceedings{nemesis,
    title={Nemesis: Normalizing the Soft-prompt Vectors of Vision-Language Models},
    author={Shuai Fu, Xiequn Wang, Qiushi Huang and Yu Zhang},
    booktitle={The International Conference on Learning Representations (ICLR)},
    year={2024}
}
```


## Acknowledgements
Our code is based on [Co-CoOp and CoOp](https://github.com/KaiyangZhou/CoOp). 
We thank the authors for releasing their code. 
If you use our model and code, please consider citing these works as well.
