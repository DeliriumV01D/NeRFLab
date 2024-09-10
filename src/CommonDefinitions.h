#ifndef COMMON_DEFINITIONS_H
#define COMMON_DEFINITIONS_H

//#include <dlib/opencv/cv_image.h>
//#include <dlib/data_io.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types_c.h>
#ifdef QT_VERSION
#include <string>
#include <QtCore/QDebug>
#include <QtGui/QImage>
#include <QtGui/QPixmap>

#include <opencv2/imgproc/types_c.h>

///Преобразование cv::Mat в QImage
inline QImage cvMatToQImage(const cv::Mat &mat)
{
  switch (mat.type())
  {
    // 8-bit, 4 channel
    case CV_8UC4:
    {
      QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_ARGB32);
      return image;
    }

    // 8-bit, 3 channel
    case CV_8UC3:
    {
      QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_RGB888);
      return image.rgbSwapped();
    }

    // 8-bit, 1 channel
    case CV_8UC1:
    {
      #if QT_VERSION >= QT_VERSION_CHECK(5, 5, 0)
      QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Grayscale8);
      #else
      static QVector<QRgb>  sColorTable;
      // only create our color table the first time
      if (sColorTable.isEmpty())
      {
        sColorTable.resize(256);
        for (int i = 0; i < 256; ++i) sColorTable[i] = qRgb(i, i, i);
      }
      QImage image(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_Indexed8);
      image.setColorTable(sColorTable);
      #endif
      return image;
    }

    //default:
    //  throw TException("cvMatToQImage() error: cv::Mat image type not supported: " /*+ std::to_string(mat.type())*/);
    //break;
  }
  return QImage();
}

///Преобразование cv::Mat в QPixmap
inline QPixmap cvMatToQPixmap(const cv::Mat &mat)
{
  return QPixmap::fromImage(cvMatToQImage(mat));
}

/// If inImage exists for the lifetime of the resulting cv::Mat, pass false to inCloneImageData to share inImage's data with the cv::Mat directly
///    NOTE: Format_RGB888 is an exception since we need to use a local QImage and thus must clone the data regardless
inline cv::Mat QImageToCvMat(const QImage& inImage, bool inCloneImageData = true)
{
	switch (inImage.format())
	{
		// 8-bit, 4 channel
	case QImage::Format_ARGB32:
	case QImage::Format_ARGB32_Premultiplied:
	{
		cv::Mat  mat(inImage.height(), inImage.width(),
			CV_8UC4,
			const_cast<uchar*>(inImage.bits()),
			static_cast<size_t>(inImage.bytesPerLine())
		);

		return (inCloneImageData ? mat.clone() : mat);
	}

	// 8-bit, 3 channel
	case QImage::Format_RGB32:
	{
		if (!inCloneImageData)
		{
			qWarning() << "QImageToCvMat() - Conversion requires cloning so we don't modify the original QImage data";
		}

		cv::Mat  mat(inImage.height(), inImage.width(),
			CV_8UC4,
			const_cast<uchar*>(inImage.bits()),
			static_cast<size_t>(inImage.bytesPerLine())
		);

		cv::Mat  matNoAlpha;

		cv::cvtColor(mat, matNoAlpha, cv::COLOR_BGRA2BGR);   // drop the all-white alpha channel

		return matNoAlpha;
	}

	// 8-bit, 3 channel
	case QImage::Format_RGB888:
	{
		if (!inCloneImageData)
		{
			qWarning() << "QImageToCvMat() - Conversion requires cloning so we don't modify the original QImage data";
		}

		QImage   swapped = inImage.rgbSwapped();

		return cv::Mat(swapped.height(), swapped.width(),
			CV_8UC3,
			const_cast<uchar*>(swapped.bits()),
			static_cast<size_t>(swapped.bytesPerLine())
		).clone();
	}

	// 8-bit, 1 channel
	case QImage::Format_Indexed8:
	{
		cv::Mat  mat(inImage.height(), inImage.width(),
			CV_8UC1,
			const_cast<uchar*>(inImage.bits()),
			static_cast<size_t>(inImage.bytesPerLine())
		);

		return (inCloneImageData ? mat.clone() : mat);
	}

	default:
		qWarning() << "QImageToCvMat() - QImage format not handled in switch:" << inImage.format();
		break;
	}

	return cv::Mat();
}

// If inPixmap exists for the lifetime of the resulting cv::Mat, pass false to inCloneImageData to share inPixmap's data with the cv::Mat directly
//    NOTE: Format_RGB888 is an exception since we need to use a local QImage and thus must clone the data regardless
inline cv::Mat QPixmapToCvMat(const QPixmap& inPixmap, bool inCloneImageData = true)
{
	return QImageToCvMat(inPixmap.toImage(), inCloneImageData);
}

#endif  //#ifdef QT_VERSION

inline void rotateImage(const cv::Mat &in, cv::Mat &out, const double &angle)
{
  // get rotation matrix for rotating the image around its center
  cv::Point2f center(in.cols/2.0f, in.rows/2.0f);
  cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
  // determine bounding rectangle
  cv::Rect bbox = cv::RotatedRect(center, in.size(), (float)angle).boundingRect();
  // adjust transformation matrix
  rot.at<double>(0,2) += bbox.width/2.0 - center.x;
  rot.at<double>(1,2) += bbox.height/2.0 - center.y;

  cv::warpAffine(in, out, rot, bbox.size());
}

///Преобразование в градации серого
static void ToGrayscale(const cv::Mat &image, cv::Mat &gray)
{
	if (image.type() != CV_8UC1)
	{
		cv::cvtColor(image, gray, CV_BGR2GRAY);
		gray.convertTo(gray, CV_8UC1);
	} else {
		image.copyTo(gray);
	}
}

///
static void GetCorners(
	cv::Mat H,
	int width,
	int height,
	std::vector<cv::Point2f>& sceneCorners
)
{
	assert(!H.empty());
	std::vector<cv::Point2f> objCorners = {
		cv::Point(0, 0),
		cv::Point(width, 0),
		cv::Point(width, height),
		cv::Point(0, height)
	};

	cv::perspectiveTransform(objCorners, sceneCorners, H);
}

///Преобразование в градации серого
static void toGrayscale8U(const cv::Mat& image, cv::Mat& gray)
{
	if (image.type() != CV_8UC1)
	{
		cv::cvtColor(image, gray, CV_BGR2GRAY);
		gray.convertTo(gray, CV_8UC1);
	}
	else {
		image.copyTo(gray);
	}
}

/////Преобразование в градации серого
//static void toGrayscale32F(const cv::Mat &image, cv::Mat &gray)
//{
//	if (image.type() == CV_32FC1)
//	{
//		image.copyTo(gray);
//	} else {
//		cv::cvtColor(image, gray, CV_BGR2GRAY);
//	}
//	if (image.type() == CV_8UC3)
//	{
//		gray.convertTo(gray, CV_32FC1, 1.0f / 255.0f);
//	}
//}

/////Преобразование матрицы чисел с плавающей запятой из формата opencv в формат dlib
//static void CVMatToDlibMatrixFC1(const cv::Mat &mat, dlib::matrix<float> &dlib_matrix)
//{
//	cv::Mat temp(mat.cols, mat.rows, CV_32FC1);
//	cv::normalize(mat, temp, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1);
//	dlib::assign_image(dlib_matrix, dlib::cv_image<float>(temp));
//}
//
/////Преобразование матрицы целых чисел из формата opencv в формат dlib
//static void CVMatToDlibMatrix8U(const cv::Mat &mat, dlib::matrix<unsigned char> &dlib_matrix)
//{
//	cv::Mat temp(mat.cols, mat.rows, CV_8U);
//	cv::normalize(mat, temp, 0, 255, cv::NORM_MINMAX, CV_8U);
//	dlib::assign_image(dlib_matrix, dlib::cv_image<unsigned char>(temp));
//}

inline void rotateCvMat(cv::Mat & mat, int rotation_angle) {
	if (rotation_angle < 0) {
		rotation_angle = 360 - std::abs(rotation_angle) % 360;
	}
	else {
		rotation_angle %= 360;
	}
	switch (rotation_angle)
	{
	case 90:
		cv::rotate(mat, mat, cv::ROTATE_90_CLOCKWISE);
		break;
	case 180:
		cv::rotate(mat, mat, cv::ROTATE_180);
		break;
	case 270:
		cv::rotate(mat, mat, cv::ROTATE_90_COUNTERCLOCKWISE);
		break;
	default:
		break;
	}
}

#endif // COMMON_DEFINITIONS_H