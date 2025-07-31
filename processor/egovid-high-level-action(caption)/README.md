
## 1. Clone Repo of LLaVA-NeXT
```
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
```

## 2. Follow the installation steps in LLaVA-NeXT
```
Please refer to https://github.com/LLaVA-VL/LLaVA-NeXT?tab=readme-ov-file#installation
Note that we use huggingface/models--lmms-lab--LLaVA-NeXT-Video-32B-Qwen for video caption
```

## 3. Run LLaVA for video caption
```
bash run_llava.sh 0 5000000
```


## 4. Prepare model and env for Qwen2
```
Please refer to https://huggingface.co/Qwen/Qwen2.5-32B-Instruct.
```

## 5. Run Qwen for caption reformat
```
bash run_qwen.sh 0 5000000
```