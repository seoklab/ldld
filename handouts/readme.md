# Deep Learning Handouts
---
## Deep Learning 공부에 앞서 ...
<img width="60%" src=https://user-images.githubusercontent.com/80088917/148338597-397b9910-26df-4bdd-8821-5c069576182c.png>

- CNN(Convolutional Neural Network)
- RNN(Recurrent Neural Network)
- ResNet(Residual Neural Network)
- GAN(Generative Adversarial neural Network)
- Reinforcement Learning
- Transformer(with Attention)
 - BERT(Bidirectional Encoder Representations from Transformers)
 - GPT(Generative Pre-trained Transformer) ex. DALL-E
 - CLIP(Contrastive Language–Image Pre-training) --- vision transformer
 - MLP mixer, etc   \
이 중에서 CNN과 RNN, 그리고 alphafold의 co-evolution information에 유용하게 사용되는 transformer에 대해서 알아봅시다

### Deep Learning이 주목받게 된 이유    

Deep Learning은 **복잡한** 문제를 풀기 위하여 사용한다.    \
즉, 적당히 풀 수 있는 문제들은 **SVM**(support vector machine,) **MLP**(multi-layer perceptron)으로 충분히 풀 수 있다.    \
SVM, MLP로 풀 수 있는 문제들을 Deep Learning으로 풀게 된다면 **Overfitting** 문제가 발생    \
컴퓨팅 파워가 향상되고, 학습 데이터가 많아져, 아주 많은 데이터를 학습시킴으로써 딥러닝과 같은 복잡한 모델구조가 과다학습되지 않게 되었고    \
복잡한 모델구조를 학습시키는 Deep Learning이 복잡한 문제를 푸는데 매우 유용하게 사용되기 시작(ex. AlexNet (2012))    

### 모델 복잡도 이론과 정규화
주어진 데이터나 문제에 대해서 가장 최적의 모델 복잡도는 무엇일까(**model selection problem**)   \
모델 복잡도(model complexity):   데이터나 문제의 복잡도에 비해 모델의 복잡도가 크면, 훈련데이터에 대한 정확도가 우수하나,   \
과다학습(overfitting)하면 일반화 성능이 저하되는 문제가 발생
<img width = "60%" src = https://user-images.githubusercontent.com/80088917/148343166-6ec822c4-dacd-4a8d-80f5-59e2fece9cd3.png>
<img width = "45%" src = https://user-images.githubusercontent.com/80088917/148343196-ae2dfefe-5062-4db2-8427-9d60055866d7.png>
<img width = "60%" src =https://user-images.githubusercontent.com/80088917/148343828-3f5d98db-281d-438d-ad86-223d187d207f.png>
<img width = "60%" src =https://user-images.githubusercontent.com/80088917/148343726-813d2289-6f76-4d4e-bad9-86596339086a.png>

### Discriminative and Generative model(변별모델과 생성모델)
<img width = "60%" src =https://user-images.githubusercontent.com/80088917/148343749-311bfb42-4207-4676-9311-6c582df156d5.png>
