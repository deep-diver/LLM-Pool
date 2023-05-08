from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='llmpool',
    version='0.0.1',
    description="Large Language Models' pool management library",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='chansung park',
    author_email='deep.diver.csp@gmail.com',
    url='https://github.com/deep-diver/LLM-Pool',
    packages=['llmpool'],
    package_dir={'':'src'},
    keywords=['LLM', 'instance pool', 'management'],
    python_reuqires='>=3.8',
    package_data={},
    zip_safe=False,
    install_requires=[
        'transformers',
        'optimum',
        'peft',
        'text_generation',
        'accelerate',
        'bitsandbytes',
    ],
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ]
)