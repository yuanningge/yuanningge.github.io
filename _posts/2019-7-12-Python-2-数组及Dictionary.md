---
layout: post
title: Python 2 note: The first part
tags: Python_2

---
布尔符号并不是从左到右依次实现的，他们的优先级为not>and>or，即非>与>或 <br>
3**16 means 3 to the power of 16,即3的16次方<br>
range(51)生成的是0至50<br>

>pyg = 'ay'<br>
>original =raw_input("Enter a word: ")#存为string<br>
>if len(original)>0 and original.isalpha()==True:#用以判断字符串中是否仅含字母<br>
>  print original<br>
>  word=original.lower()<br>
>  first=word[0]<br>
>else:<br>
>  print "empty"<br>
>new_word=word + first + pyg<br>
>new_word=new_word[1:len(new_word)]<br>

调用函数有两种方式<br>
其一：<br>
>from math import sqrt #from math import *<br>
>print sqrt(25)<br>

  其二：<br>
>print math.sqrt(25)<br>


math库里的函数包括：'__doc__', '__name__', '__package__', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'ceil', 'copysign', 'cos', 'cosh', 'degrees', 'e', 'erf', 'erfc', 'exp', 'expm1', 'fabs', 'factorial', 'floor', 'fmod', 'frexp', 'fsum', 'gamma', 'hypot', 'isinf', 'isnan', 'ldexp', 'lgamma', 'log', 'log10', 'log1p', 'modf', 'pi', 'pow', 'radians', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'trunc'<br>
>import math # Imports the math module<br>
>everything = dir(math) # Sets everything to a list of things from math<br>
>print everything # Prints 'em all!<br>



**数组及循环**<br>
命令：<br>①x.pop(index) 按位置删除<br> ②x.remove(item) 按值删除<br> ③del(x[index]) <br>
   ④range(stop) range(start,stop) range (start,stop,step) 生成数组list of numbers，默认从0开始步长为1 eg.range(0,3,1)输出[0,1,2],start(inclusive),end(exclusive) <br>
list支持使用[start:end:stride]格式，default值分别为0:最后:1，当stride为-1时你懂的<br>
⑤两种遍历方式<br>
>	  for item in list:                  <br>
>       print item       <br><br>     
>       for i in range(len(list)):<br>
>                    print list[i]<br>

⑥x=[] y=[] x+y可将两个数组连起来输出<br>
⑦print " ".join(x) 输出时在数组的item之间添加某种字符串<br>
⑧print xx,  逗号代表输出时在同一行输出<br>
⑨while与for 也可以与else搭配。但是需要for正常结束（break）之后，else才能执行<br>
⑩enumerate 自动将index+1 <br>
①①zip 将两个甚至多个数组打包操作 for a, b in zip(list_a, list_b):<br>

>start_list = [5, 3, 1, 2, 4]<br>
>square_list = []<br>

>for number in start_list:<br>
>  square_list.append(number**2)#添加<br>
>square_list.sort()#排序<br>
>square_list.remove("5")#移除<br>
>print square_list<br>

**Dictionary**冒号前为KEY名称，后为值<br>
命令：<br>①.item() 输出[('A', 1), ('B', 2)]<br>
      ②.keys() 输出['A', 'B']           .values()输出[1, 2]<br>
	  ③循环（注意逗号输出为空格）：
	  for key in my_dict:                        A 1
        print key,my_dict[key]                   B 2
	  ④evens_to_50 = [i for i in range(51) if i % 2 == 0]<br><br>
>zoo_animals = { 'Unicorn' : 'Cotton Candy House','Sloth' : 'Rainforest Exhibit','Bengal Tiger1' : 'Jungle House'}<br>
>del zoo_animals['Bengal Tiger']<br>
>zoo_animals["Rockhopper Penguin"]="not"<br>

    数组及Dictionary的小练习
>	inventory = {'gold' : 500,<br>
>                'pouch' : ['flint', 'twine', 'gemstone'], # Assigned a new list to 'pouch' key<br>
>                'backpack' : ['xylophone','dagger', 'bedroll','bread loaf']<br>
>               }<br>


>   inventory['burlap bag'] = ['apple', 'small ruby', 'three-toed sloth']<br>

>   inventory['pouch'].sort() <br>

>   inventory['pocket'] =[]<br>
>   inventory["pocket"].append("seashell")<br>
>   inventory["pocket"].append("strange berry")<br>
>   inventory["pocket"].append("lint")<br>
>   inventory["backpack"].sort()<br>
>   inventory["backpack"].remove("dagger")<br>
>   inventory["gold"]+=50<br>
>   for key in inventory:<br>
>     print key            #输出的是key的名称<br>
>     print inventory[key] #输出的是key值<br>

    enumerate的用法<br>
>choices = ['pizza', 'pasta', 'salad', 'nachos']<br>

>print 'Your choices are:'<br>
>for index, item in enumerate(choices):<br>
>  print index+1, item<br>



常用库：<br>
from random import randit<br>
from math import sqrt<br>
