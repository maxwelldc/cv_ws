 
 //Includes all the headers necessary to use the most common public pieces of the ROS system.  
    #include <ros/ros.h>  
    //Use image_transport for publishing and subscribing to images in ROS  
    #include <image_transport/image_transport.h>  
    //Use cv_bridge to convert between ROS and OpenCV Image formats  
    #include <cv_bridge/cv_bridge.h>  
      
    #include <sensor_msgs/image_encodings.h>  
    //Include headers for OpenCV Image processing  
    #include <opencv2/imgproc/imgproc.hpp>  
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
 #include "opencv2/video/tracking.hpp"  
    //Include headers for OpenCV GUI handling  
    #include <opencv2/highgui/highgui.hpp>  
    #include<string>      
#include<iostream>  
    #include <sstream>  
#include <serial/serial.h>
# include <stdio.h>
# include <stdlib.h>
    using namespace cv;  
    using namespace std;  
      
   //蓝色笔筒颜色的HSV范围  
    int iLowH = 100 ;    
    int iHighH = 124;    
  
    int iLowS = 43;     
    int iHighS = 255;    
  
    int iLowV =46;  //46  
    int iHighV = 100;      //V越小越适合暗的环境


int flag=0;

int a[90]={0};
int b[90]={0};

int c[10]={0};
int x,y=0;

int bit_x,bit_y=0;
char str[10]={0};
char rec[10]={0};

int bit=0;

int seqx,seqy=0;
  cv_bridge::CvImagePtr cv_ptr;   
  
         RNG rng;  
        //1.kalman filter setup  
        const int stateNum=4;                                      //状态值4×1向量(x,y,△x,△y)  
        const int measureNum=2;                                    //测量值2×1向量(x,y)    
        KalmanFilter KF(stateNum, measureNum, 0);     

    
    Mat measurement = Mat::zeros(measureNum, 1, CV_32F);                           //初始测量值x'(0)，因为后面要更新这个值，所以必须先定义
       
 //   void Trackbar(int,void*); 
// std_msgs::String msg;
 //       std::stringstream ss;

//int count = 0;
    //Store all constants for image encodings in the enc namespace to be used later.    
serial::Serial ser;
    namespace enc = sensor_msgs::image_encodings;    
    void image_socket(Mat inImg);  
void store(Mat inImg);
void Kalman_initial(); 
void Trackbar(Mat inImg);
    Mat image1;  
    static int imgWidth, imgHeight;  
      stringstream sss;      
    string strs;  
    //char *output_file = "/home/hsn/catkin_ws/src/rosopencv";  
      
    //This function is called everytime a new image_info message is published  
    void camInfoCallback(const sensor_msgs::CameraInfo & camInfoMsg)  
    {  
      //Store the image width for calculation of angle  
      imgWidth = camInfoMsg.width;  
      imgHeight = camInfoMsg.height;  
    }  
      
    //This function is called everytime a new image is published  
    void imageCallback(const sensor_msgs::ImageConstPtr& original_image)  
    {  
        //Convert from the ROS image message to a CvImage suitable for working with OpenCV for processing  
      
        try    
        {    
            //Always copy, returning a mutable CvImage    
            //OpenCV expects color images to use BGR channel order.    
            cv_ptr = cv_bridge::toCvCopy(original_image, enc::BGR8);    
        }    
        catch (cv_bridge::Exception& e)    
        {    
            //if there is an error during conversion, display it    
            ROS_ERROR("tutorialROSOpenCV::main.cpp::cv_bridge exception: %s", e.what());    
            return;    
         }  

if(flag==0)
{
while(1)     
{ser.write("S012E");
     cout<<"send\r\n"<<endl;
if(ser.read(2)=="ok")
break;
}

flag=1;
}



if(flag==1)
{

Trackbar(cv_ptr->image) ;
flag=2;   
}
if(flag==2)
{
image_socket(cv_ptr->image);
}

if(seqx>=1&&seqx<=6&&seqy>=1&&seqy<=8)
bit=1;

else if(seqx>=1&&seqx<=6&&seqy>8&&seqy<=22)
bit=2;

else if(seqx>=1&&seqx<=6&&seqy>22&&seqy<=35)
bit=3;

else if(seqx>6&&seqx<=18&&seqy>=1&&seqy<=8)
bit=4;

else if(seqx>6&&seqx<=18&&seqy>8&&seqy<=22)
bit=5;

else if(seqx>6&&seqx<=18&&seqy>22&&seqy<=35)
bit=6;

else if(seqx>18&&seqx<=30&&seqy>=1&&seqy<=8)
bit=7;

else if(seqx>18&&seqx<=30&&seqy>8&&seqy<=22)
bit=8;

else if(seqx>18&&seqx<=30&&seqy>22&&seqy<=35)
bit=9;

else if(seqx>30&&seqx<=35&&seqy>=1&&seqy<=8)
bit=10;

else if(seqx>30&&seqx<=35&&seqy>8&&seqy<=22)
bit=11;

else if(seqx>30&&seqx<=35&&seqy>22&&seqy<=35)
bit=12;

else bit=0;

//cout<<bit<<endl;

cout<<"x="<<seqx<<"y="<<seqy<<"  "<<bit<<endl;
// store(cv_ptr->image);


if(bit>9)bit_x=2;else bit_x=1;
//if(seqy>9)bit_y=2;else bit_y=1; 
sprintf(str, "S0%d%dE",bit_x,bit);

  ser.write(str);

//  function one
/*
if(ser.read(3)=="run")
{
 ser.write(str);
cout<<"send\r\n"<<endl;
 //str[10]={0};
}
*/
//cout<<rec[0]<<endl;
//store(cv_ptr->image);

    }  
 /*     
    void image_socket(Mat inImg)  
    {  
       imshow("image_socket", inImg);//显示图片  
        if( inImg.empty() )  
        {  
          ROS_INFO("Camera image empty");  
          return;//break;  
        }  
        stringstream sss;      
        string strs;  
        static int image_num = 1;  
        char c = (char)waitKey(1);  
      
        if( c == 27 )  
          ROS_INFO("Exit boss");//break;  
        switch(c)  
        {  
          case 'p':  
          resize(inImg,image1,Size(imgWidth/6,imgHeight/6),0,0,CV_INTER_LINEAR);    
          image1=image1(Rect(image1.cols/2-32,image1.rows/2-32, 64, 64));  
      
          strs="/home/hsn/catkin_ws/src/rosopencv";  
          sss.clear();      
          sss<<strs;      
          sss<<image_num;      
          sss<<".jpg";      
          sss>>strs;      
          imwrite(strs,image1);//保存图片  
          image_num++;  
          break;  
      default:  
          break;  
      }  
      
    }  
 */


void store(Mat img)
{
int ok=0;
    static  int image_num=0;
 
    Mat image;  

image=img;

   cv::Mat image_color =image;  
 


 cv::imshow("chessboard corners", image_color); 
char c = (char)waitKey(1);  


    if( c == 'q' )  
      ROS_INFO("Exit boss");//break;  
    switch(c)  
    {  
cout<<"qq"<<endl;
      case 'p':  

      resize(image,image1,Size(imgWidth,imgHeight),0,0,CV_INTER_LINEAR);    
   //   image1=image1(Rect(image1.cols/2-32,image1.rows/2-32, 64, 64));  
  
      strs="/home/wenhou/cv_ws/src";  
      sss.clear();      
      sss<<strs;      
      sss<<image_num;      
      sss<<".jpg";      
      sss>>strs;      
      imwrite(strs,image1);//保存图片  
ok=1;
cout<<"save ok!"<<endl;
      image_num++;  
      break;  
 // default:  
  //    break;  
}

}

void Trackbar(Mat inImg)  
    {
    static  int image_num=0;
 //   int key,count=0; 
    Mat image;  
 image=inImg; 
 //  char filename[200]; 
 //   Mat imageGray; 
 //cv::resize(img, image, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);

   //   convertIr(image, image);



   cv::Mat image_color;  
      
  
  cv::Mat image_gray;  
  
//image_color =imread("/home/wenhou/cv_ws/src0.jpg");  
image_color=image;
    cv::cvtColor(image_color, image_gray, cv::COLOR_BGR2GRAY);  
      
    std::vector<cv::Point2f> corners;  
      
    int ret = cv::findChessboardCorners(image_gray,  
                                         cv::Size(5, 7),  
                                         corners,  
   //                                      cv::CALIB_CB_ADAPTIVE_THRESH |  
   //                                      cv::CALIB_CB_NORMALIZE_IMAGE
                                         cv::CALIB_CB_FAST_CHECK);  
      
      cout<<ret<<endl;

if (ret==0)
		{			
			cout<<"can not find chessboard corners!\n"; //找不到角点
			exit(1);
		} 

    //指定亚像素计算迭代标注  

    cv::TermCriteria criteria = cv::TermCriteria(  
        cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,  
        40,  
        0.1);  
 
    //亚像素检测  
  // cv::cornerSubPix(image_gray, corners, cv::Size(5, 7), cv::Size(-1, -1), criteria);  
      
    //角点绘制  
    cv::drawChessboardCorners(image_color, cv::Size(5, 7), corners, ret);  
     for(int i=0;i<corners.size();i++)  
        {  
  //          circle(imageSource,corners[i],2,Scalar(0,0,255),2); 
           a[i]=corners[i].x;   b[i]=corners[i].y;
        cout<<"原始坐标"<<i<<"  ("<<corners[i].x<<","<<corners[i].y<<")"<<endl;  
 
        }  
   

for(int i=0;i<=35;i++)
{
for(int j=0;j<=35;j++)
{
if(a[i]<=a[j])
{x=a[i]; a[i]=a[j]; a[j]=x;}
}
}

for(int i=0;i<=35;i++)
{
for(int j=0;j<=35;j++)
{
if(b[i]<=b[j])
{y=b[i]; b[i]=b[j]; b[j]=y;}
}
}
a[35]+=200;
/*
for(int i=0;i<=35;i++)
cout<<"a_i:   "<<"x="<<a[i]<<"   "<<"y="<<b[i]<<endl;
*/
// cv::imshow("chessboard corners", image_color);  
//    cv::waitKey(0);  

     waitKey(5);      

//waitKey(500);  
    }  

void Kalman_initial()   
{
   KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1,0,1,0,0,1,0,1,0,0,1,0,0,0,0,1);  //转移矩阵A  

        setIdentity(KF.measurementMatrix);                                             //测量矩阵H  
        setIdentity(KF.processNoiseCov, Scalar::all(1e-5));                            //系统噪声方差矩阵Q  
        setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));                        //测量噪声方差矩阵R  
        setIdentity(KF.errorCovPost, Scalar::all(1));                                  //后验错误估计协方差矩阵P  
        rng.fill(KF.statePost,RNG::UNIFORM,0,0);   //初始状态值x(0)  
}    
    
  
  /* 
   void Kalman_initial()   

    {    RNG rng;  
        //1.kalman filter setup  
        const int stateNum=4;                                      //状态值4×1向量(x,y,△x,△y)  
        const int measureNum=2;                                    //测量值2×1向量(x,y)    
        KalmanFilter KF(stateNum, measureNum, 0);     

       KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1,0,1,0,0,1,0,1,0,0,1,0,0,0,0,1);  //转移矩阵A  

        setIdentity(KF.measurementMatrix);                                             //测量矩阵H  
        setIdentity(KF.processNoiseCov, Scalar::all(1e-5));                            //系统噪声方差矩阵Q  
        setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));                        //测量噪声方差矩阵R  
        setIdentity(KF.errorCovPost, Scalar::all(1));                                  //后验错误估计协方差矩阵P  
        rng.fill(KF.statePost,RNG::UNIFORM,0,winHeight>winWidth?winWidth:winHeight);   //初始状态值x(0)  
    //    Mat measurement = Mat::zeros(measureNum, 1, CV_32F);                           //初始测量值x'(0)，因为后面要更新这个值，所以必须先定义
      }
*/

  /*     void Kalman_filter() 
        {  
            Point mousePosition;  
            mousePosition.x=c[0];   
            mousePosition.y=c[1]; 

            //2.kalman prediction  
            Mat prediction = KF.predict();  
            Point predict_pt = Point(prediction.at<float>(0),prediction.at<float>(1) );   //预测值(x',y')  
      
            //3.update measurement  
            measurement.at<float>(0) = (float)mousePosition.x;  
            measurement.at<float>(1) = (float)mousePosition.y;          
      
            //4.update  
            KF.correct(measurement);  
      
 //draw   
        //    image.setTo(Scalar(255,255,255,0));  
            circle(image,predict_pt,5,Scalar(0,255,0),3);    //predicted point with green  
        //    circle(image,mousePosition,5,Scalar(255,0,0),3); //current position with red    
   //  cout<<predict_pt.x<<"    "<<predict_pt.y<<endl;
            }         
 

*/


void image_socket(Mat img)  
    {
c[0]=0;c[1]=0;
//Mat img = imread("蓝色笔筒.jpg",1);  
  
    Mat imgHSV;    
    cvtColor(img, imgHSV, COLOR_BGR2HSV);//转为HSV  
  
  //  imwrite("hsv.jpg",imgHSV);  
   // imshow("gray", imgHSV);  

    Mat imgThresholded;  
  //二值化
    inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image    
//  imshow("gray", imgThresholded);  
    //开操作 (去除一些噪点)  如果二值化后图片干扰部分依然很多，增大下面的size  

    Mat element = getStructuringElement(MORPH_RECT, Size(6, 6));    
    morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);    
   // morphologyEx(g_src,g_dst,MORPH_TOPHAT,element);  
    //闭操作 (连接一些连通域)   
    morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);    
    
  //  namedWindow("Thresholded Image",CV_WINDOW_NORMAL);  
//    imshow("Thresholded Image", imgThresholded);   
 

vector<vector<Point> > contours;
findContours(imgThresholded,contours,RETR_EXTERNAL,CHAIN_APPROX_NONE);
vector<Rect>rect(contours.size());

vector<Moments> mu(contours.size() );   

 vector<Point2f> mc( contours.size() );

for(int i=0;i<contours.size();i++)
{
rect[i]=boundingRect(contours[i]);
mu[i] = moments( contours[i], false );  
//cout<<mu[i]<<endl; 
  Point center(mu[i].m10/mu[i].m00,mu[i].m01/mu[i].m00);
circle(imgThresholded, center, 2, Scalar(0,0,255), 5, 8, 0);//绘制目标位置  

 mc[i] = Point2d( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );   

c[0]=mc[i].x;  c[1]=mc[i].y;


 // ss << mc[i] << count;
///cout<<mc[i]<<endl;
 //ROS_INFO("%f\r\n", mc[i]); 


int x=rect[i].x;
int y=rect[i].y;
int width=rect[i].width;
int height=rect[i].height;
rectangle(imgThresholded,Point(x,y),Point(x+width,y+height),Scalar(255,255,255),1);
}

//cout<<"x="<<c[0]<<"y="<<c[1]<<endl;



for(int i=0;i<=35;i++)
{
if(a[i]>=c[0])
{seqx=i;break;}
}


for(int i=0;i<=35;i++)
{
if(b[i]>=c[1])
{seqy=i;break;}
}

/*
if(c[1]<=b[1]-76)
{seqx=0;seqy=0;}
if(c[1]>=b[35]+150)
{seqx=0;seqy=0;}
*/


//for(int i=0;i<=36;i++)
//cout<<"seq_x_y:   "<<"x="<<seqx<<"   "<<"y="<<seqy<<endl;
  Point mousePosition;  
            mousePosition.x=c[0];   
            mousePosition.y=c[1]; 
        
            //2.kalman prediction  
            Mat prediction = KF.predict();  
            Point predict_pt = Point(prediction.at<float>(0),prediction.at<float>(1) );   //预测值(x',y')  
      
            //3.update measurement  
            measurement.at<float>(0) = (float)mousePosition.x;  
            measurement.at<float>(1) = (float)mousePosition.y;          
      
            //4.update  
            KF.correct(measurement);  
      
 //draw   
        //    image.setTo(Scalar(255,255,255,0));  
            circle(imgThresholded,predict_pt,5,Scalar(0,255,0),3);    //predicted point with green  
        //    circle(image,mousePosition,5,Scalar(255,0,0),3); //current position with red    
   //  cout<<predict_pt.x<<"    "<<predict_pt.y<<endl;

imshow("result",imgThresholded);



/*
 findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );  
   vector<Moments> mu(contours.size() );       
    for( int i = 0; i < contours.size(); i++ )     
    {   
        mu[i] = moments( contours[i], false );   
    }     
    //计算轮廓的质心     
    vector<Point2f> mc( contours.size() );      
    for( int i = 0; i < contours.size(); i++ )     
    {   
        mc[i] = Point2d( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );   
    }          
    
*/
    //这里是自定义的求取形心函数，当然用连通域计算更好  
   // Point center;  
   // center = GetCenterPoint(imgThresholded);//获取二值化白色区域的形心  
  
    //circle(img, center, 100, Scalar(0,0,255), 5, 8, 0);//绘制目标位置  
 //   imwrite("end.jpg", img);  
     waitKey(2);             //要加一些延时，不然不显示
      
    }  
     
    /** 
    * This is ROS node to track the destination image 
    */  








    int main(int argc, char **argv)  
    {  
        ros::init(argc, argv, "image_socket");  
        ROS_INFO("-----------------");  
 ros::NodeHandle nh;  

   Kalman_initial();

   

try 
    { 
    //设置串口属性，并打开串口 
        ser.setPort("/dev/ttyUSB0"); 
        ser.setBaudrate(57600); 
        serial::Timeout to = serial::Timeout::simpleTimeout(1000); 
        ser.setTimeout(to); 
        ser.open(); 
    } 
    catch (serial::IOException& e) 
    { 
        ROS_ERROR_STREAM("Unable to open port "); 
        return -1; 
    } 

    //检测串口是否已经打开，并给出提示信息 
    if(ser.isOpen()) 
    { 
        ROS_INFO_STREAM("Serial Port initialized"); 
    } 
    else 
    { 
        return -1; 
    } 
         

  //    while(1){
// ser.write("aaa");}


  //     ros::Publisher chatter_pub = nh.advertise<std_msgs::String>("chatter", 1000);
     
 // std_msgs::String msg;
  //      std::stringstream ss;
  //      ss << mc[i] << count;
/*
        msg.data = ss.str();
        ROS_INFO("%s", msg.data.c_str());
        chatter_pub.publish(msg);
*/
      //  ros::spinOnce();
        
      



 image_transport::ImageTransport it(nh);  
      
      //  image_transport::Subscriber sub = it.subscribe("camera/rgb/image_raw", 1, imageCallback);  
       image_transport::Subscriber sub = it.subscribe("/kinect2/qhd/image_color", 1, imageCallback);  
 // image_sub_ = it_.subscribe("camera/rgb/image_raw", 1, &RGB_GRAY::convert_callback, this); 
    // image_transport::Subscriber sub = it.subscribe("/kinect2/bond", 1, imageCallback);  
     //   ros::Subscriber camInfo         = nh.subscribe("camera/rgb/camera_info", 1, camInfoCallback);  
      ros::Subscriber camInfo         = nh.subscribe("/kinect2/qhd/camera_info", 1, camInfoCallback);

    
//ros::Rate loop_rate(10); 
//ros::spinOnce(); 
  



        ros::spin();  
      


        //ROS_INFO is the replacement for printf/cout.  
        ROS_INFO("tutorialROSOpenCV::main.cpp::No error.");  



    } 

