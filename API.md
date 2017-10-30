# SIFTandFlannn
#简单的api
#成员变量：
#sift:         通过cv2建立的SIFT扫描器
#image_or      存原始的彩色图片(ndarray)
#image         存灰度图片(ndarray)
#pixel         存像素信息(两个数)
#key           存关键点信息的list
#description   存关键点的属性(一个由128维向量组成的List)
#flann         通过cv2建立的flann匹配器
#成员函数：
#My:           构造函数
#match:        对两个My对象进行匹配，返回较好的匹配点所构造的list，较好的匹配点的个数可以反应两幅图片的匹配程度
#homography    对两个My对象进行单应性估计，返回三个参数(目前由于对于那个算法缺少了解，暂时不知道其作用以及是否需要进行单应性估计)
