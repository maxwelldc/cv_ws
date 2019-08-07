/*

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <stdio.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_publisher");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Publisher pub = it.advertise("camera/image", 1);

  cv::Mat image = cv::imread("/home/wenhou/Desktop/1.jpg", CV_LOAD_IMAGE_COLOR);
  if(image.empty()){
   printf("open error\n");
   }
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();

  ros::Rate loop_rate(5);
  while (nh.ok()) {
    pub.publish(msg);
    ros::spinOnce();
    loop_rate.sleep();
  }
}

*/

/*


#include "opencv2/opencv.hpp"
#include<iostream>
using namespace std;
using namespace cv;

int main()
{
    Mat img = imread("/home/wenhou/Desktop/1.png");
    imshow("src", img);
    Mat result = img.clone();
    Mat gray, dst , corner_img;//corner_img存放检测后的角点图像
    cvtColor(img, gray, CV_BGR2GRAY);

    cornerHarris(gray, corner_img, 2, 3, 0.04);//cornerHarris角点检测
    //imshow("corner", corner_img);
    threshold(corner_img, dst, 0.015, 255, CV_THRESH_BINARY);
    imshow("dst", dst);

    int rowNumber = gray.rows;  //获取行数
    int colNumber = gray.cols;  //获取每一行的元素
    cout << rowNumber << endl;
    cout << colNumber << endl;
    cout << dst.type() << endl;

    for (int i = 0; i<rowNumber; i++)
    {
        for (int j = 0; j<colNumber; j++)
        {
            if (dst.at<float>(i, j) == 255)//二值化后，灰度值为255为角点
            {
                circle(result, Point(j, i),3, Scalar(0, 255, 0), 2, 8);
            }
        }
    }

    imshow("result", result);
    waitKey(0);

return 0;
}

*/

/*
 #include <ros/ros.h>  
    //Use image_transport for publishing and subscribing to images in ROS  
    #include <image_transport/image_transport.h>  
    //Use cv_bridge to convert between ROS and OpenCV Image formats  
    #include <cv_bridge/cv_bridge.h>  
      
    #include <sensor_msgs/image_encodings.h>  
    //Include headers for OpenCV Image processing  
    #include <opencv2/imgproc/imgproc.hpp>  
    //Include headers for OpenCV GUI handling  
    #include <opencv2/highgui/highgui.hpp>  
    #include<string>      
    #include <sstream>   
     
    using namespace cv;    
using namespace std;  
      
    Mat image;  
    Mat imageGray;  
    int thresh=80;   //角点个数控制  
    int MaxThresh=255;  
      
int a[90]={0};
int b[90]={0};
int x,y=0;
    void Trackbar(int,void*);    


  void Trackbar(int,void*)  
    {  
        Mat dst,imageSource;  
    vector<Point2f> corners;
        dst=Mat::zeros(image.size(),CV_32FC1);    
        imageSource=image.clone();  

        goodFeaturesToTrack(imageGray,corners,thresh,0.01,10,Mat());  
        for(int i=0;i<corners.size();i++)  
        {  
            circle(imageSource,corners[i],2,Scalar(0,0,255),2); 
           a[i]=corners[i].x;   b[i]=corners[i].y;
  //      cout<<"原始坐标"<<i<<"  ("<<corners[i].x<<","<<corners[i].y<<")"<<endl;  
 
        }  

for(int i=0;i<80;i++)
{
for(int j=0;j<80;j++)
{
if(a[i]<a[j])
{x=a[i]; a[i]=a[j]; a[j]=x;}
}
}

for(int i=0;i<80;i++)
{
for(int j=0;j<80;j++)
{
if(b[i]<b[j])
{y=b[i]; a[i]=b[j]; b[j]=y;}
}
}


for(int i=0;i<80;i++)
cout<<a[i]<<endl;
        imshow("Corner Detected",imageSource);   
    }  

      
    int main(int argc,char*argv[])    
    {    
        image=imread("/home/wenhou/Desktop/1.png");  
 imshow("Corner",image);   
        cvtColor(image,imageGray,CV_RGB2GRAY);  
        GaussianBlur(imageGray,imageGray,Size(5,5),1); // 滤波  
        namedWindow("Corner Detected");  
        createTrackbar("threshold：","Corner Detected",&thresh,MaxThresh,Trackbar);  
     //   imshow("Corner Detected",image);  
        Trackbar(0,0);  
        waitKey();  
        return 0;  
    }    
      
  */


 #include <ros/ros.h>  
    //Use image_transport for publishing and subscribing to images in ROS  
    #include <image_transport/image_transport.h>  
    //Use cv_bridge to convert between ROS and OpenCV Image formats  
    #include <cv_bridge/cv_bridge.h>  
      #include <cv.h>
    #include <sensor_msgs/image_encodings.h>  
    //Include headers for OpenCV Image processing  
 #include <opencv2/imgproc/imgproc.hpp>  
    //Include headers for OpenCV GUI handling  
    #include <opencv2/highgui/highgui.hpp>  
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <highgui.h>
    #include<string>      
    #include <sstream>  


  using namespace cv;    
using namespace std;  

int a[10]={0};
int b[40]={0};
/*
int main()
{
 Mat imageSource; 
    Mat img = imread("/home/wenhou/Desktop/4.png");
 //namedWindow("Corner Detected");  
imageSource=img.clone();
//    imshow("aaa", img);
    Mat DstPic, edge, grayImage,hh,imageGray,open,close,two;

    //创建与src同类型和同大小的矩阵
    DstPic.create(img.size(), img.type());

    //将原始图转化为灰度图
    cvtColor(img, grayImage, COLOR_BGR2GRAY);

*/
/*
    //先使用3*3内核来降噪
    blur(grayImage, edge, Size(3, 3));

    //运行canny算子
    Canny(edge, edge, 30, 90, 3);
 cvtColor(edge,hh,CV_GRAY2RGB); 
cvtColor(hh,imageGray,CV_RGB2GRAY);  
imshow("Corner Detected",imageGray); 
*/

/*
 //开操作 (去除一些噪点)  如果二值化后图片干扰部分依然很多，增大下面的size  

    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));    
    morphologyEx(grayImage, open, MORPH_OPEN, element);    
   // morphologyEx(g_src,g_dst,MORPH_TOPHAT,element);  
    //闭操作 (连接一些连通域)   
    morphologyEx(open, close, MORPH_CLOSE, element);    
    
 //   namedWindow("Thresholded Image",CV_WINDOW_NORMAL);  
  //  imshow("Thresholded Image", close);   

threshold(close, two, 96, 255, CV_THRESH_BINARY);  
 imshow("Two", two);   
        
       // namedWindow("Corner Detected");  

 cvtColor(two,hh,CV_GRAY2RGB); 
cvtColor(hh,imageGray,CV_RGB2GRAY);  
GaussianBlur(imageGray,imageGray,Size(5,5),1); // 滤波  

vector<Point2f> corners;
goodFeaturesToTrack(imageGray,corners,80,0.01,10,Mat()); 
 for(int i=0;i<corners.size();i++)  
        {  
            circle(imageSource,corners[i],2,Scalar(0,0,255),2); 
       //    a[i]=corners[i].x;   b[i]=corners[i].y;
  //      cout<<"原始坐标"<<i<<"  ("<<corners[i].x<<","<<corners[i].y<<")"<<endl;  
 
        }  



  //  imshow("bbb", edge);

       imshow("Corner ",imageSource); 
    waitKey(0);
return 0;

}

*/


int main()
{
    cv::Mat image_color =imread("/home/wenhou/cv_ws/src0.jpg");  
      
    cv::Mat image_gray;  
    cv::cvtColor(image_color, image_gray, cv::COLOR_BGR2GRAY);  
      
    std::vector<cv::Point2f> corners;  
      
    int ret = cv::findChessboardCorners(image_gray,  
                                         cv::Size(5, 7),  
                                         corners,  
                                         cv::CALIB_CB_ADAPTIVE_THRESH |  
                                         cv::CALIB_CB_NORMALIZE_IMAGE);  
      
      cout<<ret<<endl;
    //指定亚像素计算迭代标注  
    cv::TermCriteria criteria = cv::TermCriteria(  
        cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,  
        40,  
        0.1);  
      
    //亚像素检测  
   cv::cornerSubPix(image_gray, corners, cv::Size(5, 7), cv::Size(-1, -1), criteria);  
      
    //角点绘制  
    cv::drawChessboardCorners(image_color, cv::Size(5, 7), corners, ret);  
     for(int i=0;i<corners.size();i++)  
        {  
  //          circle(imageSource,corners[i],2,Scalar(0,0,255),2); 
           a[i]=corners[i].x;   b[i]=corners[i].y;
        cout<<"原始坐标"<<i<<"  ("<<corners[i].x<<","<<corners[i].y<<")"<<endl;  
 
        }  
    cv::imshow("chessboard corners", image_color);  
    cv::waitKey(0);  
    
//cv::imshow("chessboard corners", image_color);  
 //   cv::waitKey(10);  
     

   
}






