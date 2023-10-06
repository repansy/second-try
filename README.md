# depth map python深度图生成实验
## 陈书洋 21312687
1.使用代码的环境opencv-python 4.5.2
2.原理实现步骤
> 读取左右图片
> 
>> 读取相机内参和外参， 使用之前先将标定得到的内外参数填写到stereoconfig.py中的StereoCamera类中
>
>> 立体校正，使用getRectifyTransform，rectifyImage
>
>> 绘制等间距平行线，检查立体校正的效果，使用draw_line
>
>> 立体匹配， 并绘制视差图，使用stereoMatchSGBM
>
>> 最后计算深度图
