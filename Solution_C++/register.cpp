/**
 * @file register.cpp
 * @author Nikola Andric (namdd@umsystem.edu)
 * @brief This program is used to warm an image to another perspective such that objects in the image align with each other. 
 * @version 0.1
 * @date 2022-04-09
 * 
 * @copyright Copyright (c) 2022
 * 
 * 
 * NOTE: This was implemented in Google Colab. Use the link below to acces the file.
 *          
 *         Google Colab Link: https://colab.research.google.com/drive/1dmNuubuKA98iEyOOKmm_2xsQGpOkww8A?usp=sharing
 * 
 */

#include <opencv2/opencv.hpp>
//#include <opencv2/xfeatures2d.hpp>  //does not work in google colab
#include "opencv2/features2d.hpp"


#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace std;
using namespace cv;
//using namespace cv::xfeatures2d;

const int MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.15f;

void alignImages(Mat &incorrect_img, Mat &correct_img, Mat &incorrect_img_Reg, Mat &h)
{
  // Convert images to grayscale
  Mat incorrect_img_gray, correct_img_gray;
  cvtColor(incorrect_img, incorrect_img_gray, cv::COLOR_BGR2GRAY);
  cvtColor(correct_img, correct_img_gray, cv::COLOR_BGR2GRAY);

  // Variables to store keypoints and descriptors
  std::vector<KeyPoint> keypoints1, keypoints2;
  Mat descriptors1, descriptors2;

  // Detect ORB features and compute descriptors.
  Ptr<Feature2D> orb = ORB::create();
  orb->detectAndCompute(incorrect_img_gray, Mat(), keypoints1, descriptors1);
  orb->detectAndCompute(correct_img_gray, Mat(), keypoints2, descriptors2);

  // Match features.
  std::vector<DMatch> matches;
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  matcher->match(descriptors1, descriptors2, matches, Mat());

  // Sort matches by score
  std::sort(matches.begin(), matches.end());

  // Remove not so good matches
  const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
  matches.erase(matches.begin()+numGoodMatches, matches.end());

  // Draw top matches
  Mat imMatches;
  drawMatches(incorrect_img, keypoints1, correct_img, keypoints2, matches, imMatches);
  imwrite("matches.jpg", imMatches);
  
 
  // Extract location of good matches
  std::vector<Point2f> points1, points2;

  for( size_t i = 0; i < matches.size(); i++ )
  {
    points1.push_back( keypoints1[ matches[i].queryIdx ].pt );
    points2.push_back( keypoints2[ matches[i].trainIdx ].pt );
  }

  // Find homography
  h = findHomography( points1, points2, RANSAC );

  // Use homography to warp image
  warpPerspective(incorrect_img, incorrect_img_Reg, h, correct_img.size());
}

int main(int argc, char **argv)
{
  // Read reference image
  string refFilename("shelf_good.png"); 
  cout << "Reading reference image : " << refFilename << endl; 
  
  Mat imReference = imread(refFilename);

  // Check for invalid input
  if(! imReference.data ){
      cout <<  "Could not open or find the reference image" << std::endl ;
      return -1;
  }
 
  // Read image to be aligned
  string imFilename("shelf_not_good.png");
  cout << "Reading image to align : " << imFilename << endl; 
  Mat im = imread(imFilename);

  // Check for invalid input
  if(! im.data ){
      cout <<  "Could not open or find the image that is to be aligned" << std::endl ;
      return -1;
  }
 
  // Registered image will be resotred in imReg. 
  // The estimated homography will be stored in h. 
  Mat imReg, h;

  // Align images
  cout << "Aligning images ..." << endl; 
  alignImages(im, imReference, imReg, h);

  // Write aligned image to disk. 
  string outFilename("aligned_shelf.jpg");
  cout << "Saving aligned image : " << outFilename << endl; 
  imwrite(outFilename, imReg);

  // Print estimated homography
  cout << "Estimated homography : \n" << h << endl; 
}