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
@article{lee2021paraphrasing,
  title={Paraphrasing via Ranking Many Candidates},
  author={Lee, Joosung},
  journal={arXiv preprint arXiv:2107.09274},
  year={2021}
}
```