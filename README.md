# SpatialRGPT-Bench Tutorial

A beginner-friendly quickstart tutorial for [SpatialRGPT](https://github.com/AnjieCheng/SpatialRGPT) - test different AI models from various providers (OpenAI, Google, Nebius) on the SpatialRGPT benchmark through a unified API wrapper.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KJ-rc/spatialrgpt-tutorial/blob/main/spatial_reasoning_tutorial.ipynb)

> **Note**: This tutorial is designed for quick understanding and hands-on experience, not for fully reproducing the results in SpatialRGPT paper Table 1.

## Quick Start

```bash
pip install -r requirements.txt
cd datasets && bash download_datasets.sh && cd ..
jupyter notebook spatial_reasoning_tutorial.ipynb
```

## Learning Path

1. **Start with the notebook** - `spatial_reasoning_tutorial.ipynb`
2. **Try single examples** - Understand the evaluation process
3. **Run batch evaluations** - Test different models and categories
4. **Experiment with prompts** - Modify system and question prompts
5. **Analyze results** - Compare model performance across settings

##  Command Line Usage

### Test Single Example
```bash
python single_evaluator.py --annotation-file datasets/spatial_category_subsets/below_above_subset.jsonl --example-index 0 --model openai:gpt-4
```

### Batch Evaluation
```bash  
python batch_evaluator.py --annotation-file datasets/spatial_category_subsets/below_above_subset.jsonl --model google:gemini-2.5-flash --output-path ./eval_output/results
```

### Score Results
```bash
python llm_judge_scorer.py ./eval_output/results/responses_with_predictions.jsonl openai:gpt-5-nano
```


## 📖 References

If you use this tutorial or the SpatialRGPT methodology in your research, please cite the original paper:

```bibtex
@inproceedings{cheng2024spatialrgpt,
    title={SpatialRGPT: Grounded Spatial Reasoning in Vision-Language Models},
    author={Cheng, An-Chieh and Yin, Hongxu and Fu, Yang and Guo, Qiushan and Yang, Ruihan and Kautz, Jan and Wang, Xiaolong and Liu, Sifei},
    booktitle={NeurIPS},
    year={2024}
}
```

**Note on AISwuite**: The aisuite used in this repository is forked and revised from: https://github.com/andrewyng/aisuite/tree/main, where some minor updates are applied. Please check the commit log in https://github.com/KJ-rc/aisuite for details.

## Acknowledgments

Special thanks to [An-Chieh Cheng](https://www.anjiecheng.me/) for kindly helping with this tutorial.