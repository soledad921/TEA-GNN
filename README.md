#  Time-aware Graph Neural Network for Entity Alignment betweenTemporal Knowledge Graphs


## Enviroment
Anaconda>=4.3.30
Python>=3.5
Keras>=2.2.4
Tensorflow>=1.13.1
Scipy
Numpy


## Datasets
Datasets include ICEWS05-15, YAGO-WIKI20K, YAGO-WIKI50K.

```
ent_ids_1: ids for entities in source KG;
ent_ids_2: ids for entities in target KG;
ref_ent_ids: entity links encoded by ids;
triples_1: relation triples encoded by ids in source KG;
triples_2: relation triples encoded by ids in target KG;
rel_ids_1: ids for entities in source KG;
rel_ids_2: ids for entities in target KG;
sup_pairs + ref_pairs: entity alignments
```

## Usage:
To reproduce the reported results of our models, use the following command:

```
python main.py --data ICEWS05-15 --seed 200 --gamma 3 --dim 100
```

'seed' denotes the number of pre-aligned entity pairs
'gamma' denotes the margin used in the loss function
'dim' denotes the embedding dimension

Please refer to the hyperparamters listed in the paper for reproducing the reported results. Since TEA-GNN significantly outperforms other baseline models, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly due to the instability of embedding-based methods.


## License
TEA-GNN is CC-BY-NC licensed, as found in the LICENSE file.

## Acknowledgement
We refer to the code of RREA. Thanks for their great contributions!

## Citation
If you use the codes, please cite the following paper:

        @inproceedings{TEAGNN,
            title = "Time-aware Graph Neural Network for Entity Alignment between Temporal Knowledge Graphs",
            author = "Xu, Chengjin  and
              Su, Fenglong  and
              Lehmann, Jens",
            booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
            month = nov,
            year = "2021",
            address = "Online and Punta Cana, Dominican Republic",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2021.emnlp-main.709",
            doi = "10.18653/v1/2021.emnlp-main.709",
            pages = "8999--9010",
        }
