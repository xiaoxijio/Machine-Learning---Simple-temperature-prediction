# 机器学习-简单的气温预测

 1、导入数据 temp_2为前天最高温度，temp_1为昨天最高温度，actual为今天最高温度（也是我们要预测的值），friend为朋友瞎预测的值

![image](https://github.com/user-attachments/assets/1ada6c71-0647-47e2-a3c8-3813fc3de233)

2、类别变量需要进行独热编码

![image](https://github.com/user-attachments/assets/65e28484-8fbb-4608-97e8-c3f15d76ef2f)

3、 将标签actual去掉只保留特征（因为我们需要训练特征来预测标签） --> 标准化去量纲（这里只打印了第一天标准化后的数据）

![image](https://github.com/user-attachments/assets/fa90c508-602f-4b68-81cb-17fe758189a9)

4、建立模型训练 --> 画图（对比训练出的数据与原数据）

![image](https://github.com/user-attachments/assets/8ef394ab-cc61-4ca2-abbf-063d98240af5)


非常简单的一个基于机器学习的数据预测小demo，代码中包括了分解内容，就是我注释掉的代码，帮助初学者理解
