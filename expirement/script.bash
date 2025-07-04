python fake_news.py --provider ollama --model llama3:8b  --task ANS --no_shots 0
python fake_news.py --provider ollama --model llama3:8b  --prompt prompt_shots.txt --task ANS --no_shots 2
python fake_news.py --provider ollama --model llama3:8b  --prompt prompt_shots.txt --task ANS --no_shots 4
python fake_news.py --provider ollama --model llama3:8b  --prompt prompt_shots.txt --task ANS --no_shots 8
python fake_news.py --provider ollama --model llama3:8b  --prompt prompt_shots.txt --task ANS --no_shots 16
python fake_news.py --provider ollama --model llama3:8b  --prompt prompt_shots.txt --task ANS --no_shots 32
python fake_news.py --provider ollama --model llama3:8b  --prompt prompt_cot.txt --task ANS --no_shots 0
