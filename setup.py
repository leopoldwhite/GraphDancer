# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 GraphDancer Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fallback installation script when pyproject.toml is not used.
# Installs the `verl` fork and the `graphdancer` agent package.
import os
from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).parent
with open(os.path.join(here, "verl/version/version")) as f:
    __version__ = f.read().strip()

install_requires = [
    "accelerate",
    "codetiming",
    "datasets",
    "dill",
    "hydra-core",
    "numpy<2.0.0",
    "pandas",
    "peft",
    "pyarrow>=19.0.0",
    "pybind11",
    "pylatexenc",
    "ray[default]>=2.41.0",
    "torchdata",
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
    "transformers",
    "packaging>=20.0",
    "sentence-transformers",
    "faiss-cpu",
    "rouge-score",
]

extras_require = {
    "vllm": ["vllm==0.8.5.post1"],
    "gpu": ["flash-attn"],
    "eval": ["openai"],
    "logging": ["wandb", "tensorboard"],
}

setup(
    name="graphdancer",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where=".", include=["verl", "verl.*", "graphdancer", "graphdancer.*"]),
    license="Apache 2.0",
    description="GraphDancer: curriculum PPO for LLM agents that explore and reason over graphs (built on verl)",
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.10",
    package_data={
        "": ["version/*"],
        "verl": ["trainer/config/*.yaml", "trainer/config/*/*.yaml"],
    },
    include_package_data=True,
    long_description=(here / "README.md").read_text(),
    long_description_content_type="text/markdown",
)
