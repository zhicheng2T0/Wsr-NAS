# Wsr-NAS

This is the official github repository for "Neural Architecture Search for Wide Spectrum Adversarial Robustness"

# Introduction
One major limitation of CNNs is that they are vulnerable to adversarial attacks. Currently, adversarial robustness in neural networks is commonly optimized with respect to a small pre-selected adversarial noise strength, causing them to have potentially limited performance when under attack by larger adversarial noises in real-world scenarios. In this research, we aim to find Neural Architectures that have improved robustness on a wide range of adversarial noise strengths through Neural Architecture Search. In detail, we propose a lightweight Adversarial Noise Estimator to reduce the high cost of generating adversarial noise with respect to different strengths. Besides, we construct an Efficient Wide Spectrum Searcher to reduce the cost of adjusting network architecture with the large adversarial validation set during the search. With the two components proposed, the number of adversarial noise strengths searched can be increased significantly while having a limited increase in search time. Extensive experiments on benchmark datasets such as CIFAR and ImageNet demonstrate that with a significantly richer search signal in robustness, our method can find architectures with improved overall robustness while having a limited impact on natural accuracy and around 40% reduction in search time compared with the naive approach of searching.

# Method
xxx

# To search and retrain

To search for WsrNets on CIFAR-10 using the Wsr-NAS algorithm, please refer to ./search/script/search_pgd.sh. The architecture for AN-Estimator used when searching for WsrNet-Plus is included in ./search/robust_train_search_official.py, the name of this AN-Estimator is AN_estimator_plus.

To retrain WsrNet-Basic, WsrNet-Robust, WsrNet-M, WsrNet-M-1, WsrNet-N, WsrNet-Plus on CIFAR-10, please refer to "train_wsrnet_basic.sh", "train_wsrnet_robust.sh", "train_wsrnet_m.sh", "train_wsrnet_m_1.sh", "train_wsrnet_n.sh", "train_wsrnet_plus.sh" in ./retrain/script/.

To retrain WsrNet-Basic, RACL, AdvRush on ImageNet using Fast-AT, please refer to "train_ImgNet_WsrNet_basic.sh", "train_ImgNet_RACL.sh" and "train_ImgNet_AdvRush.sh" ./retrain/script/.

Codes for testing different versions of AN-Estimator is provided in ./test_AN_estimator.
