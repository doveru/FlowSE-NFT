from setuptools import setup, find_namespace_packages


setup(
    name="diffusion-nft",
    version="0.0.1",
    packages=find_namespace_packages(
        include=[
            "flow_grpo", "flow_grpo.diffusers_patch", "flow_grpo.speech_flowse",
            "flow_grpo.speech_flowse.model*", "config", "scripts",
        ],
        exclude=["*.__pycache__"],
    ),
    package_data={"flow_grpo.speech_flowse": ["Emilia_ZH_EN_pinyin/vocab.txt", "vocos-mel-24khz/config.yaml"]},
    python_requires=">=3.10",
    install_requires=[
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "transformers==4.40.0",
        "accelerate==1.4.0",
        "diffusers==0.33.1",
        "numpy==1.26.4",
        "tqdm==4.67.1",
        "wandb==0.18.7",
        "peft==0.10.0",
        "huggingface-hub==0.29.1",
        "tokenizers==0.19.1",
        "einops==0.8.1",
        "absl-py",
        "ml_collections",
        "soundfile",
        "vocos",
        "soxr",
        # Direct speech imports; pin these to the validated training environment
        # when producing the paper's environment lock file.
        "librosa",
        "x-transformers",
        "onnxruntime",
    ],
    extras_require={
        "dev": [
            "ipython==8.34.0",
            "black==24.2.0",
            "pytest==8.2.0"
        ]
    }
)
