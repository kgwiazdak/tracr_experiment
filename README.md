### Automatic task recognition of tracr model
#### Overview
A project to develop an automated interpreter for neural networks, specifically focusing on Tracrcompiled transformers where we have ground-truth knowledge of their internal mechanisms.
I transformed tracr model to transformer lens model, performed simple task like reversing and
then provide combination of parameters from cache. Lower results are obtained from attention
patterns and positional embedding. The task of LLM was to describe inner mechanism basing on
data and then predict a task.
I didn’t add input/output pairs in purpose to check whether LLM can predict task without this
information.
#### Important classes
restapi.py - api to communicate with ChatGPT O1<br>
tracr_models.py - class that produces files
