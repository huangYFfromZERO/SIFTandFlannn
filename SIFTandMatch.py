import cv2
import numpy as np
#定义封装类
class My:
    def __init__(self, image_add, keypoints=3000):                      #定义构造函数，参数为照片的目录
        self.sift = cv2.xfeatures2d.SIFT_create(keypoints)              #第1个成员变量为SIFT扫描器，
        self.image_or = cv2.imread(image_add,1)                         #第2个成员变量原始图片
        self.image = cv2.cvtColor(self.image_or, cv2.COLOR_BGR2GRAY)    #第3个成员变量为灰度化的图像
        self.pixel = self.image.shape[:2]                               #第4个成员变量为像素信息
        self.key, self.description = self.sift.detectAndCompute(self.image,None)
        #得到第5，6个成员变量，关键点以及描述符
        #其中关键点key为一个list，存的应为关键点序号
        #描述符为一个二维数组，第一个维度为关键点数，第二维度为属性数,默认属性为128个特征
        self.key_nums = keypoints                     #第7个成员变量为关键点的数目
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        #设置Flann匹配算法的参数
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        #第8个成员变量为flann匹配器

    def match(self, aimage,rates=0.65, point_numbres=2):
        #此处的rates用于评估匹配的好坏，简单来说距离最近的点的距离越小于其他k个点越好，而rates用来限制
        #最好点的距离与其他点的比值小于多少时可以称之为一个好的匹配
        #此处的point_numbres为寻找的与目标点匹配的点的个数。
        matches = self.flann.knnMatch(self.description, aimage.description, k=point_numbres)
        #其中参数k的意义为找出对于第一幅图片中的特征点，第二张图离它最近的k个点
        #matches 为由一组点list组成的list,每个点为cv2.DMatch类
        best_point = []
        for i in range(len(matches)):
            #对于第一幅图的每个点，挑选出最好的点
            flag = 1
            for j in range(len(matches[i])):
                if j==0:
                    continue
                if matches[i][0].distance>rates*matches[i][j].distance:
                    flag=0
                    break
            if flag==1:
                best_point.append(matches[i][0])
        return best_point
        #最后返回最佳匹配，为一个list，其中的元素为满足约束的最近点
    def homography(self, aimage, best_point):
        src_points = np.float32([self.key[m.queryIdx].pt for m in best_point]).reshape(-1, 1, 2)
        #queryIdx存的是第一幅图中的点的下标
        dst_points = np.float32([aimage.key[m.trainIdx].pt for m in best_point]).reshape(-1, 1, 2)
        #trainIdx存的是第一幅图中的点的下标
        #通过.pt得到效果好的keypoint的坐标，然后再重构造变成数组类型，第一个维度为关键点，第二个维度表述是y还是x
        #故为m*2的二维数组。
        #中间那个1实际上没啥用，但是要好些库函数的计算需要为三个维度的
        matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        #得到的matrix是单应性矩阵，但是由于并不清楚单应性估计的方法，并不值得它有什么用,也并不清楚第三个参数有何用处
        #mask应该是用来对于匹配进行单应性估计的一个结果
        matchesMask = mask.ravel().tolist()
        #将mask从数组化为list
        y, x = m1.image.shape[:2]
        #y,x分别是图片的像素值
        src = np.float32([[0, 0], [0, y - 1], [x - 1, y - 1], [x - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(src, matrix)
        #最后这个dst_corners只是得到的一个参数？
        return dst, matrix, matchesMask
        #返回三个单应性估计的参数

m1 = My("C:\cat2.jpg")
m2 = My("C:\cat1.jpg")
tt = m1.match(m2)
#当好的匹配数目少于4时，认为匹配失败。
if len(tt)<=4:
    print("Failed")
else:
    print("The amount of matches is",len(tt))
    a,b,c = m1.homography(m2,tt)
    img_match = cv2.drawMatchesKnn(m1.image_or, m1.key, m2.image_or, m2.key, [tt], None, flags=2)
    cv2.imshow("ssss",img_match)
    cv2.waitKey(10000)
