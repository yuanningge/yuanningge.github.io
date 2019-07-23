---
layout: post
title: Machine Learning note: The first part
tag: machine_learning
---

矩阵运算：<br>
1.维度相同的矩阵才可以相加<br>
2.行x列<br>
3.矩阵不存在交换律，存在结合律<br>
4.只有维度是m*m的矩阵才存在逆矩阵，矩阵与其逆矩阵相乘等于单位矩阵；零矩阵没有逆矩阵。<br>
  不存在逆矩阵的矩阵叫做奇异矩阵或退化矩阵<br>


**1.定义**<br>
Arthur Samuel(1959)未直接编程而赋予计算机学习的能力。<br>
Tom Mitchell(1998)定义机器学习，计算机程序从经验E中学习解决某一任务T进行某一性能度量P，通过P测定在T上的表现因经验E而提高。<br>
                  A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.
                  例如计算机判断邮箱中邮件是否为垃圾邮件的过程中，T为进行分类，E为人工分类结果，P为正确分类的邮件数量。<br>
**2.分类**<br>
**监督学习 superivised learning**<br>
    对于数据集中的每个数据，都有相应的正确答案，训练集算法就是基于此整理二者关系，以此来做出预测。<br>
	监督分类问题可以分为以下两类：<br>
	1）回归问题“regression”即为通过回归来预测一个连续值得输出(map input variables to some continuous function)<br>
	2）分类问题“classification”即为预测离散值得输出(map input variables into discrete categories)<br>
**非监督学习 unsupervised learning**<br>
    是一种学习机制，仅提供大量的数据，要求算法自动找出数据中蕴含的类型结构。<br>

**3.线性回归算法 Learning regression**<br>
常用notation：<br>
    n=特征量的数目<br>
    m=训练样本的数目<br>
	x=输入的变量/特征<br>
	y=输出的变量/特征<br>
	(x,y)一个训练样本<br>
	(x^(i),y^(i))第i个训练样本（从1开始计数哦），x^(i)为第i个训练样本，若有下脚标j，则代表第i个训练样本的第j个特征量<br>
	h(hypothesis)是一个从x到y的函数映射，<br><br>
**单变量线性回归** h_θ(x)=θ_0+θ_1*x<br>
平方误差函数square error function（代价函数）cost function<br>
  j(θ_0,θ_1 )=1/2m ∑_(i=0)^m▒〖(h_θ (x^((i) ) )-y^((i)))〗^2 
    梯度下降算法 gradient descent:用于最小化函数，起始点位置略有不同会导致得到一个非常不同的局部最优解，接近或在全局最优解附近。<br>
    θ_j:=θ_j-α ϑ/(ϑθ_j ) J(θ_0,θ_1)
	α为学习速率,控制以多大的幅度更新参数θ。 θ_0与θ_1应该同时更新（计算）<br>
	Batch gradient descent:批量梯度下降，在梯度下降的每一步，都使用所有训练数据。无局部最优解，只有一个全局最优解。<br><br>
**多元线性回归**h_θ(x)=θ^T*x
