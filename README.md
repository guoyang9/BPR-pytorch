# Pytorch-BPR

Note that I use the two sub datasets provided by Xiangnan's [repo](https://github.com/hexiangnan/neural_collaborative_filtering/tree/master/Data). Another pytorch NCF implementaion can be found at this [repo](https://github.com/guoyang9/NCF).

I utilized a factor number **32**, and posted the results in the NCF paper and this implementation here. Since there is no specific numbers in their paper, I found this implementation achieved a better performance than the original curve. Moreover, the batch_size is not very sensitive with the final model performance.

Models 			| MovieLens HR@10 | MovieLens NDCG@10 | Pinterest HR@10 | Pinterest NDCG@10
------ 			| --------------- | ----------------- | --------------- | -----------------
pytorch-BPR    	| 0.700 		  | 0.418             | 0.877 			| 0.551


## The requirements are as follows:
	* python==3.6
	* pandas==0.24.2
	* numpy==1.16.2
	* pytorch==1.0.1
	* tensorboardX==1.6 (mainly useful when you want to visulize the loss, see https://github.com/lanpa/tensorboard-pytorch)

## Example to run:
```
python main.py --factor_num=16 --lamda=0.001
```
