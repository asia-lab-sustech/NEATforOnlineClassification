
# Online NEAT for classification
-------------

> Email: [liur3@mail.sustech.edu.cn](liur3@mail.sustech.edu.cn) 

This project uses NEAT (Neuroevolution) to complete the task of 2-classification prediction.  

<!---->
## *Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites
```sh
python3 tensorflow numpy sklearn pandas 
```

### 4 types of fitness calculation 
Fitness is calculated as follows:

Type	|	Method
----	|	----
ACC	|	accuracy
PAN	|	combines Recall and Specificity  fitness=Recall+ Specificity
PRO	|	utilizes profits of each loan
PAP	|	fitness(t)=α∙profit(t)+ β∙fitness(t-1)


<!-- USAGE EXAMPLES -->
## *Usage
Use this space to show useful examples of how a project can be used. How to run online NEAT

```sh
python3 newevolve.py a1 a2 a3 a4  
```

- a1: The type of Fitness calculation (0-ACC, 1-PAN, 2-PRO, 3-PAP) 
- a2: The size of Time window 
- a3: α in PAP 
- a4: β in PAP 

for example: python3 newevolve.py 0 500 0 0 

If you want to set other parameters in evolution (like the population size), you can change them in this file: /Neat/config, more explanations can be found in this [link](https://github.com/CodeReclaimers/neat-python)

How to run LSTM

```sh
python3 LSTM.py 
```

<!-- About Results -->
## *About Results
The results will show accuracy, Recall and Specificity in each generation. 
for example: <br>
0.716 0.9466292134831461 0.14583333333333334 <br>
0.722 0.9358288770053476 0.0873015873015873 <br>
0.732 0.9523809523809523 0.04918032786885246 <br>
....<br>

<!-- About Data -->
## *About Data
The data is provided by Lendclub

<!-- -->
## Acknowledgements
* [NEAT-python](https://github.com/CodeReclaimers/neat-python)
* [Lending Club](https://www.lendingclub.com/info/download-data.action)
