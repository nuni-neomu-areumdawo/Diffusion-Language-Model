# Diffusion-Language-Model
Implementation of a LLaDA-inspired Masked Diffusion Model for Text using PURE BYTE-LEVEL TOKENIZATION (it's simple, sweet and nice. You may try other tokenizers, I am interested in your findings!) and Mixed Precision Training for speed.

# Why?
Traditional language models like ChatGPT generate text auto-regressively, meaning left to right. This prevents them from changing any of the decisions they have previously made. This issue presents itself by making it harder for them to plan, do iterative tasks, prevent “true” reasoning and contribute to dangerous behavior arising such as “jailbreaks”. Diffusion has shown promise in Novel Image Synthesis models like Hourglass Diffusion Transformers and audio generation models like Tortoise-TTS. Diffusion allows the model to generate content iteratively, allowing them to refine their output over multiple inferences. Another successful advance is that of models which operate at least partially on compressed latent spaces such as Stable-Diffusion and Deepseek (using multi-head latent attention). 

# Disclaimer and Warranty
I am a high school student and much of my knowledge is self-researched. This software is provided solely for demonstration and/or educational purposes and was made for personal learning. It is provided "AS IS" without warranty of any kind, express or implied. The author(s) and copyright holder(s) disclaim all warranties, including, but not limited to, the implied warranties of merchantability, fitness for a particular purpose, title, and non-infringement. This project contains code from other repositories such as for RoPE or MLA which are cited and compared in the program. I was too lazy to clean-up documentation/conventions and *might* be slightly misleading. Please report any issues.

# Design Choices
I am a high school student and much of my knowledge is self-researched, therefore some might seem arbitrary (and to varying degrees, this is true.) I chose Multi-Head Latent Attention because of a Medium article I read which claimed it was better and it seemed to work fine. Otherwise many of the choices are similar to the original paper. I changed the SwiGLU Feed-Forward Network to Mish which is slower by quite a bit but makes the network train in much fewer steps. I chose byte-level tokenization becasue of simplicity though it may harm performance.

# Inference Choices
The prompt length is larger than generation length. I figured it would help i tlearn though never ran validation on that. 64 inference steps works well. At least in my implementation it is interesting with too many more.
