# depth map python深度图生成实验
## 陈书洋 21312687
1.使用代码的环境opencv-python 4.5.2
2.原理过程
> 读取左右图片
> 
>> 读取相机内参和外参， 使用之前先将标定得到的内外参数填写到stereoconfig.py中的StereoCamera类中
>
>> 立体校正
>
>> 绘制等间距平行线，检查立体校正的效果
>
>> 立体匹配， 并绘制视差图
>
>>  计算深度图
>
>>> SGBM匹配参数设置
>>> \
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    paraml = {'minDisparity': 0,       # 一般置零，最小视差，也影响噪声
              'numDisparities': 64,    # 16倍数，改变深度的详细程度，如果增大的视差范围超过了场景中实际的深度范围，就会出现黑色区域。黑色区域表示无法进行有效的匹配或估计深度
              'blockSize': blockSize,  # 奇数>3，所选的窗口大小越大，所包含的像素就越多，从而产生更稳定，但粗略的视差图
              'P1': 8 * img_channels * blockSize,  # 如果参数值太高，将导致平滑的结果，丢失更多的细节和锐度
              'P2': 32 * img_channels * blockSize,
              'disp12MaxDiff': 50,      # 用于限制左右视图之间的最大视差数量差异。增加这个值可能会导致插值和未对齐的像素点在图像中显示，太小的值，则视差较光滑，缺少细节特征
              'preFilterCap': 63,
              'uniquenessRatio': 1,   # 像素值的唯一性，如果唯一性比例越高，则得到的视差图的噪声和未对齐的像素会越小。但如果唯一性比例太高，则有可能会失去细节
              'speckleWindowSize': 40,
              'speckleRange': 1,       # 规定一个视差变化的阈值，如果发现视差变化超出了这个阈值，则这个像素应该是一些无用的孤立像素
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }
>>> \

