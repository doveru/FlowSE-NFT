from setuptools import setup, find_namespace_packages


setup(
    name="flowse-nft",
    version="0.0.1",
    packages=find_namespace_packages(
        include=[
            "flow_nft", "flow_nft.diffusers_patch", "flow_nft.speech_flowse",
            "flow_nft.speech_flowse.model*", "config", "scripts",
        ],
        exclude=["*.__pycache__"],
    ),
    package_data={"flow_nft.speech_flowse": ["Emilia_ZH_EN_pinyin/vocab.txt", "vocos-mel-24khz/config.yaml"]},
    python_requires=">=3.10",
    install_requires=[
        "torch==2.2.0",
        "torchaudio==2.2.0",
        "transformers==4.44.1",
        "accelerate==1.2.1",
        "numpy==1.26.4",
        "tqdm==4.66.4",
        "wandb==0.26.0",
        "peft==0.19.1",
        "huggingface-hub==0.32.0",
        "tokenizers==0.19.1",
        "einops==0.8.0",
        "absl-py==2.1.0",
        "ml_collections==1.1.0",
        "soundfile==0.12.1",
        "vocos==0.1.0",
        "soxr==0.3.7",
        "librosa==0.10.1",
        "x-transformers==1.43.2",
        "onnxruntime-gpu==1.12.0",
        "PyYAML==6.0.1",
        "scipy==1.14.1",
    ],
    extras_require={
        "dev": [
            "ipython==8.25.0",
            "ruff==0.11.11"
        ]
    }
)
