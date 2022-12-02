# Paraphrasing via Ranking Many Candidates (INLG 2022)
- [paper](https://arxiv.org/pdf/2107.09274.pdf)

## Paraphrasing
```bash
python3 ranking.py
```

## Downstream task
- excute downstream task (classification task)
- dataset: hate_speech (eng), hate_speech (kor), financial
```bash
cd downstream
python3 ranking.py {--arguments}
# For data augmentation of data of downstream tasks

cd model
python3 train.py {--arguments}
# training classification task
```

## Citation

```bibtex
@inproceedings{lee-2022-paraphrasing,
    title = "Paraphrasing via Ranking Many Candidates",
    author = "Lee, Joosung",
    booktitle = "Proceedings of the 15th International Conference on Natural Language Generation",
    month = jul,
    year = "2022",
    address = "Waterville, Maine, USA and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.inlg-main.6",
    pages = "68--72",
    abstract = "",
}
```
