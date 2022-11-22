# Wsr-NAS

To search for WsrNets on CIFAR-10 using the Wsr-NAS algorithm, please refer to ./search/script/search_pgd.sh. The architecture for AN-Estimator used when searching for WsrNet-Plus is included in ./search/robust_train_search_official.py, the name of this AN-Estimator is AN_estimator_plus.

To retrain WsrNet-Basic, WsrNet-Robust, WsrNet-M, WsrNet-M-1, WsrNet-N, WsrNet-Plus on CIFAR-10, please refer to "train_wsrnet_basic.sh", "train_wsrnet_robust.sh", "train_wsrnet_m.sh", "train_wsrnet_m_1.sh", "train_wsrnet_n.sh", "train_wsrnet_plus.sh" in ./retrain/script/.

To retrain WsrNet-Basic, RACL, AdvRush on ImageNet using Fast-AT, please refer to "train_ImgNet_WsrNet_basic.sh", "train_ImgNet_RACL.sh" and "train_ImgNet_AdvRush.sh" ./retrain/script/.

Codes for testing different versions of AN-Estimator is provided in ./test_AN_estimator.
