# Wsr-NAS

This is the official github repository for "Neural Architecture Search for Wide Spectrum Adversarial Robustness"

# Introduction
![alt text](https://github.com/zhicheng2T0/Wsr-NAS/blob/master/demo.PNG)
One major limitation of CNNs is that they are vulnerable to adversarial attacks. Currently, adversarial robustness in neural networks is commonly optimized with respect to a small pre-selected adversarial noise strength, causing them to have potentially limited performance when under attack by larger adversarial noises in real-world scenarios. In this research, we aim to find Neural Architectures that have improved robustness on a wide range of adversarial noise strengths through Neural Architecture Search. In detail, we propose a lightweight Adversarial Noise Estimator to reduce the high cost of generating adversarial noise with respect to different strengths. Besides, we construct an Efficient Wide Spectrum Searcher to reduce the cost of adjusting network architecture with the large adversarial validation set during the search. With the two components proposed, the number of adversarial noise strengths searched can be increased significantly while having a limited increase in search time. Extensive experiments on benchmark datasets such as CIFAR and ImageNet demonstrate that with a significantly richer search signal in robustness, our method can find architectures with improved overall robustness while having a limited impact on natural accuracy and around 40% reduction in search time compared with the naive approach of searching.

# Method
In this research, we aim to address the limitations in the existing research by finding Neural Architectures with wide spectrum adversarial robustness (WsrNets). When trained with the commonly used adversarial training techniques (i.e. TRADES or PGD), on a single model without having a significant increase in model parameters, the WsrNets found have improved average robust accuracy (average model accuracy on adversarial examples with different adversarial noise strengths) on a wide range of adversarial noise strengths while maintaining high clean accuracy (accuracy on clean data without adversarial noises). To find such WsrNets, we propose a search algorithm named Neural Architecture Search for Wide Spectrum Adversarial Robustness(Wsr-NAS) leveraging the One-Shot-NAS framework. 

To prevent a significant increase in computational costs when simultaneously generating adversarial noise at different strengths during the search, we propose a lightweight Adversarial Noise Estimator (AN-Estimator). The AN-Estimator can be trained to generate adversarial noises for each input based on a few existing adversarial noises corresponding to the input, allowing adversarial noises at different adversarial noise strengths to be generated at a much lower cost. The architecture of the AN-Estimator is demonstrated below.


# To search and retrain

To search for WsrNets on CIFAR-10 using the Wsr-NAS algorithm, please refer to ./search/script/search_pgd.sh. The architecture for AN-Estimator used when searching for WsrNet-Plus is included in ./search/robust_train_search_official.py, the name of this AN-Estimator is AN_estimator_plus.

To retrain WsrNet-Basic, WsrNet-Robust, WsrNet-M, WsrNet-M-1, WsrNet-N, WsrNet-Plus on CIFAR-10, please refer to "train_wsrnet_basic.sh", "train_wsrnet_robust.sh", "train_wsrnet_m.sh", "train_wsrnet_m_1.sh", "train_wsrnet_n.sh", "train_wsrnet_plus.sh" in ./retrain/script/.

To retrain WsrNet-Basic, RACL, AdvRush on ImageNet using Fast-AT, please refer to "train_ImgNet_WsrNet_basic.sh", "train_ImgNet_RACL.sh" and "train_ImgNet_AdvRush.sh" ./retrain/script/.

Codes for testing different versions of AN-Estimator is provided in ./test_AN_estimator.
